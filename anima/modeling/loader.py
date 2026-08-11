"""Load Anima's four components from a diffusers-format repo.

Produce the repo once with diffusers' own script:

    python scripts/convert_anima_to_diffusers.py \
        --transformer_ckpt_path anima-base-v1.0.safetensors \
        --text_encoder_ckpt_path qwen_3_06b_base.safetensors \
        --vae_ckpt_path qwen_image_vae.safetensors \
        --qwen_tokenizer_path <qwen3 tokenizer dir> \
        --t5_tokenizer_path <t5 tokenizer dir> \
        --output_path anima-diffusers --save_pipeline

Anima is Cosmos-Predict2's DiT plus one extra conditioning module, so there is no custom model
class here: the trunk is a stock CosmosTransformer3DModel and the LLMAdapter is upstream's
AnimaTextConditioner. Keeping them as separate components (rather than one fused module) is what
lets us freeze, set LRs on, and quantize the text path independently of the trunk.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

# The LLMAdapter always emits at least this many tokens; the DiT cross-attends over the
# padded sequence with no mask (padded rows are zeroed). Matches the reference implementation.
CROSSATTN_SEQ_LEN = 512


@dataclass
class AnimaComponents:
    transformer: CosmosTransformer3DModel
    text_conditioner: AnimaTextConditioner
    text_encoder: Qwen3Model | None
    vae: AutoencoderKLQwenImage | None
    tokenizer: AutoTokenizer | None
    t5_tokenizer: T5TokenizerFast | None
    scheduler: FlowMatchEulerDiscreteScheduler

    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """The two modules training ever touches. The Qwen3 encoder and VAE stay frozen."""
        return {"transformer": self.transformer, "text_conditioner": self.text_conditioner}


def load_components(
    path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    load_text_encoder: bool = True,
    load_vae: bool = True,
    load_tokenizers: bool = True,
    shift: float = 3.0,
) -> AnimaComponents:
    """Load from a converted diffusers-format Anima repo.

    With cached latents and cached Qwen3 embeddings, the VAE and text encoder are dead weight at
    train time — skip them to keep ~1.4GB off the GPU.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no Anima repo at {path}; run convert_anima_to_diffusers.py first")

    transformer = CosmosTransformer3DModel.from_pretrained(
        path, subfolder="transformer", torch_dtype=dtype
    )
    text_conditioner = AnimaTextConditioner.from_pretrained(
        path, subfolder="text_conditioner", torch_dtype=dtype
    )

    text_encoder = None
    if load_text_encoder:
        text_encoder = Qwen3Model.from_pretrained(path / "text_encoder", torch_dtype=dtype)
        text_encoder.requires_grad_(False).eval()

    vae = None
    if load_vae:
        vae = AutoencoderKLQwenImage.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        vae.requires_grad_(False).eval()

    tokenizer = t5_tokenizer = None
    if load_tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        t5_tokenizer = T5TokenizerFast.from_pretrained(path / "t5_tokenizer")

    return AnimaComponents(
        transformer=transformer,
        text_conditioner=text_conditioner,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=shift),
    )


@torch.no_grad()
def encode_prompts(
    components: AnimaComponents,
    prompts: list[str],
    device: torch.device | str = "cuda",
    max_length: int = CROSSATTN_SEQ_LEN,
) -> dict[str, torch.Tensor]:
    """Prompts -> the tensors the DiT needs. Cache these; the Qwen3 pass is pure overhead per step.

    Returns qwen_embeds/qwen_attention_mask (Qwen3 side) and t5_input_ids/t5_attention_mask
    (the LLMAdapter's query tokens — the T5 *tokenizer* is used, never a T5 encoder).
    """
    qwen = components.tokenizer(
        prompts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length
    )
    t5 = components.t5_tokenizer(
        prompts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length
    )

    enc = components.text_encoder.to(device)
    out = enc(
        input_ids=qwen.input_ids.to(device),
        attention_mask=qwen.attention_mask.to(device),
    ).last_hidden_state
    # Padded positions must not carry signal into the adapter's cross-attention.
    out = out.masked_fill(~qwen.attention_mask.to(device).bool().unsqueeze(-1), 0.0)

    return {
        "qwen_embeds": out,
        "qwen_attention_mask": qwen.attention_mask.to(device),
        "t5_input_ids": t5.input_ids.to(device),
        "t5_attention_mask": t5.attention_mask.to(device),
    }
