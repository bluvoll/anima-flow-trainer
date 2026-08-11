"""Reference-side oracle. Runs under diffusion-pipe's OWN interpreter, not ours.

diffusion-pipe lives in a different venv (torch 2.12-nightly, diffusers 0.39). Rather than
reinstalling its dependency tree into ours -- or worse, editing it until it agrees with us -- we
run it untouched as a subprocess and exchange tensors through .npz.

Invoked by test_transformer_parity.py; not meant to be run by hand.

Usage: <diffusion-pipe-venv>/bin/python ref_dump.py <inputs.npz> <outputs.npz>
       cwd MUST be the diffusion-pipe root (its imports are cwd-relative).
"""

import os
import sys

# Python puts *this script's* directory on sys.path, not the cwd. diffusion-pipe's imports are
# cwd-relative (`models.*`, `utils.*`), so the working directory has to go on the path explicitly.
sys.path.insert(0, os.getcwd())

# ...and importing `models.*` prepends submodules/ComfyUI to sys.path[0]. ComfyUI ships a *regular*
# `utils` package (with __init__.py), which shadows diffusion-pipe's namespace `utils` and has no
# common.py -- so `models.base` then dies on `from utils.common import ...`. Binding the right
# `utils` in sys.modules first makes the later resolution hit the cache instead.
import utils.common  # noqa: F401,E402  (import for side effect: pin before ComfyUI shadows it)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main(inputs_path: str, outputs_path: str) -> None:
    from models.anima_modeling import Anima
    from models.cosmos_predict2 import get_dit_config
    from safetensors.torch import load_file

    data = np.load(inputs_path)
    ckpt = str(data["ckpt_path"])
    dev = "cuda"

    # Native keys carry a `net.` prefix; both get_dit_config and the module want it stripped.
    state_dict = {k.removeprefix("net."): v for k, v in load_file(ckpt).items()}

    cfg = get_dit_config(state_dict)
    print("ref dit_config:", {k: cfg[k] for k in sorted(cfg)}, file=sys.stderr)

    model = Anima(**cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # These are derived RoPE buffers, recomputed in __init__ and never stored in the checkpoint.
    # Upstream's converter drops them too (TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0).
    derived = {"pos_embedder.seq", "pos_embedder.dim_spatial_range", "pos_embedder.dim_temporal_range"}
    missing = [k for k in missing if k not in derived]
    assert not missing and not unexpected, f"ref load mismatch {missing[:4]} {unexpected[:4]}"

    # fp32 throughout: comparing in bf16 cannot distinguish a real bug from kernel noise.
    model = model.to(dev, torch.float32).eval()

    t = lambda k, d=torch.float32: torch.from_numpy(data[k]).to(dev, d)
    latents = t("latents")
    timesteps = t("timesteps")
    qwen_embeds = t("qwen_embeds")
    t5_input_ids = t("t5_input_ids", torch.long)
    qwen_mask = t("qwen_attention_mask", torch.long)
    t5_mask = t("t5_attention_mask", torch.long)

    # Both sides end up in F.scaled_dot_product_attention, but the two venvs run different torch
    # versions (2.10 vs 2.12-nightly) which select different fused kernels. Forcing the exact MATH
    # backend removes kernel choice from the comparison, so any residual difference is real.
    from contextlib import nullcontext

    ctx = nullcontext()
    if os.environ.get("PARITY_MATH_SDPA") == "1":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        ctx = sdpa_kernel(SDPBackend.MATH)

    out = {}
    with torch.no_grad(), ctx:
        # --- LLMAdapter ---
        crossattn = model.llm_adapter(
            qwen_embeds,
            t5_input_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        # anima.py:1252-1256 -- zero padded rows, then right-pad to 512.
        crossattn[~t5_mask.bool()] = 0
        if crossattn.shape[1] < 512:
            crossattn = torch.nn.functional.pad(
                crossattn, (0, 0, 0, 512 - crossattn.shape[1])
            )
        out["crossattn_emb"] = crossattn.float().cpu().numpy()

        # --- RoPE, isolated: the highest-risk piece ---
        b, _, tt, h, w = latents.shape
        padding_mask = latents.new_zeros((latents.shape[0], 1, h, w))
        _, rope_emb, extra_pos = model.prepare_embedded_sequence(
            latents, fps=None, padding_mask=padding_mask
        )
        assert extra_pos is None, "Anima should have no extra positional embedding"
        out["rope_emb"] = rope_emb.float().cpu().numpy()

        # --- pre-block inputs: isolate whether divergence starts before the blocks at all ---
        x_emb, _, _ = model.prepare_embedded_sequence(latents, fps=None, padding_mask=padding_mask)
        out["patch_embed"] = (
            x_emb.detach().float().reshape(x_emb.shape[0], -1, x_emb.shape[-1]).cpu().numpy()
        )
        ts = timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps
        t_emb, adaln_lora = model.t_embedder(ts)
        out["t_emb_raw"] = t_emb.detach().float().cpu().numpy()
        out["t_emb"] = model.t_embedding_norm(t_emb).detach().float().cpu().numpy()
        if adaln_lora is not None:
            out["adaln_lora"] = adaln_lora.detach().float().cpu().numpy()

        # --- per-block activations, to tell accumulation drift from a real divergence ---
        taps: dict[int, torch.Tensor] = {}

        def make_hook(i):
            def hook(_mod, _inp, output):
                x = output[0] if isinstance(output, (tuple, list)) else output
                taps[i] = x.detach().float().reshape(x.shape[0], -1, x.shape[-1]).cpu()
            return hook

        handles = [
            model.blocks[i].register_forward_hook(make_hook(i))
            for i in (0, 1, 6, 13, 27)
        ]

        # Sub-block taps in block 0: attn1 is where RoPE is actually applied to q/k.
        sub: dict[str, torch.Tensor] = {}

        def sub_hook(name):
            def hook(_m, _i, o):
                x = o[0] if isinstance(o, (tuple, list)) else o
                sub[name] = x.detach().float().reshape(x.shape[0], -1, x.shape[-1]).cpu()
            return hook

        b0 = model.blocks[0]
        handles += [
            b0.self_attn.register_forward_hook(sub_hook("b0_self_attn")),
            b0.cross_attn.register_forward_hook(sub_hook("b0_cross_attn")),
            b0.mlp.register_forward_hook(sub_hook("b0_mlp")),
        ]

        # --- full trunk ---
        sample = model(
            x_B_C_T_H_W=latents,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn,
            fps=None,
            padding_mask=padding_mask,
        )
        out["sample"] = sample.float().cpu().numpy()
        for h in handles:
            h.remove()
        for i, v in taps.items():
            out[f"block{i}"] = v.numpy()
        for k, v in sub.items():
            out[k] = v.numpy()

    np.savez(outputs_path, **out)
    print(f"ref dumped: { {k: v.shape for k, v in out.items()} }", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
