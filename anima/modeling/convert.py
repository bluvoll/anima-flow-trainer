"""diffusers -> native Anima checkpoint, for ComfyUI export.

diffusers ships the *forward* conversion (scripts/convert_anima_to_diffusers.py), so we do not
duplicate it — run that script once to produce a diffusers-format Anima repo. What upstream does
NOT provide is the reverse: turning a trained diffusers model back into the single-file
`anima-base-v1.0.safetensors` layout that ComfyUI loads. That is this module's job.

The rename table below is the exact inverse of upstream's TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0
(scripts/convert_cosmos_to_diffusers.py). Verified: an independently-derived mapping matched
upstream's entry for entry, and 567 transformer + 118 conditioner keys account for all 685
tensors in the released checkpoint.
"""

import re

import torch
from safetensors.torch import save_file

NATIVE_PREFIX = "net."
ADAPTER_PREFIX = "llm_adapter."

# diffusers name -> native name. Inverse of upstream's rename dict.
_TOPLEVEL = {
    "patch_embed.proj": "x_embedder.proj.1",
    "time_embed.t_embedder": "t_embedder.1",
    "time_embed.norm": "t_embedding_norm",
    "norm_out.linear_1": "final_layer.adaln_modulation.1",
    "norm_out.linear_2": "final_layer.adaln_modulation.2",
    "proj_out": "final_layer.linear",
}

_BLOCK = {
    "norm1.linear_1": "adaln_modulation_self_attn.1",
    "norm1.linear_2": "adaln_modulation_self_attn.2",
    "norm2.linear_1": "adaln_modulation_cross_attn.1",
    "norm2.linear_2": "adaln_modulation_cross_attn.2",
    "norm3.linear_1": "adaln_modulation_mlp.1",
    "norm3.linear_2": "adaln_modulation_mlp.2",
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn1.norm_q": "self_attn.q_norm",
    "attn1.norm_k": "self_attn.k_norm",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "attn2.norm_q": "cross_attn.q_norm",
    "attn2.norm_k": "cross_attn.k_norm",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
}

_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.+?)\.(weight|bias)$")

# Longest prefix first so `time_embed.t_embedder` is tried before any shorter overlap.
_TOPLEVEL_BY_LEN = sorted(_TOPLEVEL.items(), key=lambda kv: -len(kv[0]))


class ConversionError(KeyError):
    pass


def _strip_wrapper_key(key: str) -> str:
    """Drop leading DDP `module.` components and every `_orig_mod` component anywhere.

    Done component-wise rather than with a regex so nesting cannot defeat it: `module._orig_mod.`,
    `module.module.`, and `_orig_mod` sitting mid-path all reduce correctly. Neither name can be a
    real Anima submodule, so this never removes anything meaningful.
    """
    parts = key.split(".")
    while parts and parts[0] == "module":
        parts.pop(0)
    return ".".join(p for p in parts if p != "_orig_mod")


def strip_wrappers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Undo DDP's `module.` and torch.compile's `_orig_mod` in state-dict keys.

    Belt and braces for `unwrap_model(..., keep_torch_compile=False)`, and it has already earned
    its keep once. Accelerate's `compile_regions` stamps `_orig_mod` onto the *root* object it
    returns; because regional compilation runs after the DDP wrap, that root is the DDP module. On
    the way out, `extract_model_from_parallel` sees that stamp and -- with the default
    `keep_torch_compile=True` -- re-attaches the compiled wrapper it had just unwrapped, handing
    back keys like

        module.transformer_blocks.0._orig_mod.attn1.to_q.lora_A.default.weight

    with `module.` at the root and `_orig_mod` inside every block. That reached the rename tables
    as 560 unmapped keys and killed the run at its first checkpoint -- after the epoch, not at
    startup, which is the expensive place to find out.

    `_orig_mod` is matched mid-path as well as at the root precisely because regional compilation
    puts it there: one wrapper per repeated block rather than one at the top.
    """
    return {_strip_wrapper_key(key): value for key, value in state_dict.items()}


# Cross-checked against diffusers' `_convert_non_diffusers_anima_lora_to_diffusers`: all 22 of its
# entries agree with the tables above entry-for-entry. Our extra 4 (attn{1,2}.norm_{q,k}) are the
# QK RMSNorms, which upstream's LoRA path omits because a norm is never a LoRA target -- they are
# needed for full checkpoints only, where the 685/685 round trip covers them.


def transformer_to_native(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """CosmosTransformer3DModel state dict -> `net.*` native keys."""
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in strip_wrappers(state_dict).items():
        if m := _BLOCK_RE.match(key):
            idx, inner, suffix = m.groups()
            if inner in _BLOCK:
                out[f"{NATIVE_PREFIX}blocks.{idx}.{_BLOCK[inner]}.{suffix}"] = value
                continue
            unmapped.append(key)
            continue

        # Prefix replacement, longest first: `time_embed.t_embedder` must win over `time_embed.norm`
        # and still handle the trailing `.linear_1.weight`.
        for diffusers_prefix, native_prefix in _TOPLEVEL_BY_LEN:
            if key == diffusers_prefix or key.startswith(diffusers_prefix + "."):
                out[NATIVE_PREFIX + native_prefix + key[len(diffusers_prefix) :]] = value
                break
        else:
            unmapped.append(key)

    if unmapped:
        raise ConversionError(f"{len(unmapped)} transformer keys unmapped: {unmapped[:8]}")
    return out


def text_conditioner_to_native(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """AnimaTextConditioner state dict -> `net.llm_adapter.*`. Names already agree."""
    return {f"{NATIVE_PREFIX}{ADAPTER_PREFIX}{k}": v
            for k, v in strip_wrappers(state_dict).items()}


LORA_PREFIX = "diffusion_model."

# peft/diffusers decorate state-dict keys; strip them so the rename tables see bare module paths.
_PEFT_STRIP = ("base_model.model.", "transformer.", "text_conditioner.")


# Adapter markers peft emits. LoKr factorises as w1 (or w1_a/w1_b) x w2 (or w2_a/w2_b), with an
# optional Tucker core t2 -- exactly the names ComfyUI's LoKrAdapter looks up
# (comfy/weight_adapter/lokr.py:211).
_ADAPTER_MARKER = r"(lora_[AB]|lokr_(?:w\d+(?:_[ab])?|t\d+)|hada_w\d+_[ab])"


def _strip_peft_key(key: str) -> str:
    """Remove peft's decoration so the rename tables see bare module paths.

    Two shapes, because peft stores the two adapter families differently:

        LoRA   `...lora_A.default.weight`  -> `...lora_A.weight`   (nn.Linear, has .weight)
        LoKr   `...lokr_w1.default`        -> `...lokr_w1`         (nn.Parameter, no suffix)

    The second case used to fall through untouched, so a LoKr export emitted
    `diffusion_model.<path>.lokr_w1.default` where every consumer looks for
    `diffusion_model.<path>.lokr_w1` -- the file loaded nowhere. The `.weight`/`.bias` lookahead
    keeps the second rule from stripping an already-clean LoRA key back to `lora_A`.
    """
    for prefix in _PEFT_STRIP:
        if key.startswith(prefix):
            key = key[len(prefix) :]
    key = re.sub(rf"{_ADAPTER_MARKER}\.[^.]+\.(weight|bias)$", r"\1.\2", key)
    return re.sub(rf"{_ADAPTER_MARKER}\.(?!weight$|bias$)[^.]+$", r"\1", key)


def _split_lora_key(key: str) -> tuple[str, str]:
    """`transformer_blocks.0.attn1.to_q.lora_A.weight` -> ("transformer_blocks.0.attn1.to_q",
    "lora_A.weight"). The tail is everything from the adapter marker onward."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part.startswith(("lora_", "lokr_", "hada_")) or part in ("alpha", "dora_scale"):
            return ".".join(parts[:i]), ".".join(parts[i:])
    raise ConversionError(f"no adapter marker in LoRA key: {key!r}")


def lora_to_native(
    transformer_lora: dict[str, torch.Tensor] | None = None,
    text_conditioner_lora: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """PEFT adapter state dicts -> the `diffusion_model.*` layout ComfyUI and diffusers both read.

    This is the exact inverse of diffusers' `_convert_non_diffusers_anima_lora_to_diffusers`, so
    what we write here loads back through `AnimaLoraLoaderMixin.load_lora_weights` unchanged.
    Upstream provides that loader but no matching saver -- every other model family has
    `save_lora_weights`, Anima does not -- which is why this exists.
    """
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in strip_wrappers(transformer_lora or {}).items():
        module, tail = _split_lora_key(_strip_peft_key(key))

        if m := re.match(r"^transformer_blocks\.(\d+)\.(.+)$", module):
            idx, inner = m.groups()
            if inner not in _BLOCK:
                unmapped.append(key)
                continue
            out[f"{LORA_PREFIX}blocks.{idx}.{_BLOCK[inner]}.{tail}"] = value
            continue

        for diffusers_prefix, native_prefix in _TOPLEVEL_BY_LEN:
            if module == diffusers_prefix or module.startswith(diffusers_prefix + "."):
                native = native_prefix + module[len(diffusers_prefix) :]
                out[f"{LORA_PREFIX}{native}.{tail}"] = value
                break
        else:
            unmapped.append(key)

    for key, value in strip_wrappers(text_conditioner_lora or {}).items():
        module, tail = _split_lora_key(_strip_peft_key(key))
        out[f"{LORA_PREFIX}{ADAPTER_PREFIX}{module}.{tail}"] = value

    if unmapped:
        raise ConversionError(f"{len(unmapped)} LoRA keys unmapped: {unmapped[:8]}")
    if not out:
        raise ConversionError("no LoRA tensors to save")
    return out


def _attach_alpha(native: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    """Add `<module>.alpha` beside every lora_A/lora_B pair, so the trained scale survives export.

    Keyed off lora_A because its first dim IS the rank, which is what the consumer divides by.
    """
    out = dict(native)
    for key in native:
        if key.endswith(".lora_A.weight"):
            out[key[: -len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(alpha))
    return out


def native_lora_to_kohya(native: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    """`diffusion_model.*` (lora_A/lora_B) -> sd-scripts/kohya `lora_unet_*` (lora_down/lora_up).

    Both formats load in ComfyUI: its generic loop over the model state dict registers
    `lora_unet_<underscored>` and `diffusion_model.<dotted>` for the same weight (comfy/lora.py:192).
    Only the diffusers form round-trips through `AnimaLoraLoaderMixin`, so it stays the default;
    this exists because the surrounding kohya tooling expects the other one.

    `alpha` transfers unchanged -- peft scales by alpha/r and kohya by alpha/dim, the same thing.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in native.items():
        module, tail = _split_lora_key(key[len(LORA_PREFIX) :])
        name = "lora_unet_" + module.replace(".", "_")
        if tail.startswith("lora_A"):
            out[f"{name}.lora_down.weight"] = value
        elif tail.startswith("lora_B"):
            out[f"{name}.lora_up.weight"] = value
        else:
            raise ConversionError(f"kohya export supports plain LoRA only, got {tail!r} in {key!r}")
        out.setdefault(f"{name}.alpha", torch.tensor(float(alpha)))
    return out


def save_lora_checkpoint(
    path: str,
    transformer_lora: dict[str, torch.Tensor] | None = None,
    text_conditioner_lora: dict[str, torch.Tensor] | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    metadata: dict[str, str] | None = None,
    fmt: str = "diffusers",
    alpha: float | None = None,
    rank: int | None = None,
) -> int:
    """Write a LoRA. `fmt` is "diffusers" (default, round-trips through diffusers) or "kohya"."""
    native = lora_to_native(transformer_lora, text_conditioner_lora)

    if fmt == "kohya":
        if alpha is None or rank is None:
            raise ValueError("kohya format needs `alpha` and `rank`; they are not in the tensors")
        native = native_lora_to_kohya(native, alpha, rank)
    elif fmt != "diffusers":
        raise ValueError(f"unknown LoRA format: {fmt!r} (expected 'diffusers' or 'kohya')")
    else:
        # peft applies `alpha / r` at runtime, and lora_A/lora_B carry no trace of it. Without an
        # `.alpha` beside them the scale is simply lost: ComfyUI's loader falls back to
        # `scale = 1.0` when the key is absent (comfy/weight_adapter/lora.py:314), so a LoRA
        # trained at alpha 16 rank 8 was being applied at HALF its trained strength, and alpha 8
        # vs alpha 64 produced byte-identical files. Emitted per module, matching the key ComfyUI
        # looks up (`<module>.alpha`) and what the kohya path already writes.
        if alpha is not None:
            native = _attach_alpha(native, alpha)

    out = {}
    for k, v in native.items():
        v = v.detach()
        # `.alpha` is a scalar hyperparameter, not a weight -- keep it fp32 so a bf16 cast cannot
        # perturb the scale the LoRA was trained at.
        out[k] = (v if dtype is None or k.endswith(".alpha") else v.to(dtype)).contiguous()
    save_file(out, path, metadata=metadata)
    return len(out)


def save_native_checkpoint(
    path: str,
    transformer_state_dict: dict[str, torch.Tensor],
    text_conditioner_state_dict: dict[str, torch.Tensor] | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a single-file checkpoint ComfyUI can load. Returns the tensor count."""
    native = transformer_to_native(transformer_state_dict)
    if text_conditioner_state_dict is not None:
        native.update(text_conditioner_to_native(text_conditioner_state_dict))

    out = {}
    for k, v in native.items():
        v = v.detach()
        if dtype is not None:
            v = v.to(dtype)
        out[k] = v.contiguous()

    save_file(out, path, metadata=metadata)
    return len(out)
