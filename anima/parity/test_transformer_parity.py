"""Phase 2 gate: our diffusers stack must match diffusion-pipe's Anima numerically.

Everything runs in fp32. This is not fussiness -- comparing two *mathematically identical*
implementations in bf16 showed ~5e-3 relative difference from kernel accumulation order alone,
which is large enough to hide a real bug and large enough to flag a non-bug. In fp32 the same
comparison lands at exactly 0.

Tested, cheapest-to-diagnose first:
  1. RoPE       -- diffusers unbinds on dim -2, Cosmos uses an `interleaved` flag. Both turn out
                   to be the same half-split, but that is worth proving, not assuming.
  2. LLMAdapter -- upstream AnimaTextConditioner vs diffusion-pipe's LLMAdapter.
  3. Full trunk -- the end-to-end number that actually matters.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# PARITY_MATH_SDPA=1 pins both sides to the exact SDPA math backend, removing fused-kernel
# selection (which differs between the two venvs' torch versions) from the comparison.
MATH_SDPA = os.environ.get("PARITY_MATH_SDPA") == "1"

# This gate compares against **diffusion-pipe running in its own venv**, as a subprocess exchanging
# tensors via .npz -- so it needs a second trainer checked out and installed, plus the original
# (non-diffusers) Anima checkpoint. None of that exists on a fresh clone, so all three are env
# vars and the gate skips with instructions rather than failing.
#
#   ANIMA_NATIVE_CKPT=/path/to/anima-base-v1.0.safetensors \
#   ANIMA_MODEL=/path/to/anima-diffusers \
#   ANIMA_REF_ROOT=/path/to/diffusion-pipe \
#       .venv/bin/python anima/parity/test_transformer_parity.py
CKPT = os.environ.get("ANIMA_NATIVE_CKPT", "")
ANIMA_REPO = os.environ.get("ANIMA_MODEL", "../anima-diffusers")
REF_ROOT = Path(os.environ.get("ANIMA_REF_ROOT", ""))
REF_PYTHON = REF_ROOT / "venv" / "bin" / "python"

# Per-check tolerances, calibrated to what each check can actually prove.
#
# Wiring checks must be BIT-EXACT: identical weights through identical ops must give identical
# results, so any nonzero difference in the patch/timestep path is a real defect.
#
# The end-to-end trunk cannot be bit-exact -- two independently written 28-block networks
# accumulate fp32 rounding differently. The bound that matters is the one training actually
# operates at: measured bf16 kernel noise between *mathematically identical* implementations is
# ~5e-3 relative. A 1e-3 gate sits an order of magnitude below that, so anything it admits is
# provably invisible to bf16 training, while a genuine structural error (wrong RoPE convention,
# swapped modulation order, mismatched mask) shifts outputs by O(0.1-1) and still fails loudly.
TOL = {
    "patch_embed": 0.0,
    "t_emb": 0.0,
    "adaln_lora": 0.0,
    "rope_cos": 1e-6,
    "rope_sin": 1e-6,
    "llm_adapter": 1e-4,
    # Measured across the CASES sweep below: worst observed 8.9e-4, and it is non-monotonic in
    # resolution (8.9e-4 at 384px but 1.3e-4 at 768px and 3.5e-4 at 1920px) -- rounding luck, not
    # a resolution-dependent defect. 1e-3 sat too close to the observed max to be stable across
    # seeds; 2e-3 still sits 2.5x below the ~5e-3 bf16 kernel-noise floor that training runs at.
    "full_trunk": 2e-3,
}
DEFAULT_TOL = 2e-3

# The gate runs over resolutions and batch sizes, not one config. RoPE is resolution-dependent and
# 34% of its dims are temporal (inert when T=1), so a single spatial size proves very little.
#
# Capped at 1536px: that is the practical ceiling on 24GB cards. The architecture allows 2048px
# (diffusers errors above 128 patch units; diffusion-pipe asserts at 1920px), so if this ever runs
# on H100-class hardware, extend the sweep to (1, 256, x) = 2048px before trusting that range.
CASES = [
    (1, 16, 6),    # 128px  - low end
    (1, 32, 0),
    (1, 48, 1),    # 384px  - worst observed
    (1, 64, 2),
    (1, 96, 3),
    (1, 128, 4),   # 1024px
    (2, 64, 5),    # batch > 1
    (1, 192, 7),   # 1536px - practical ceiling on a 4090
]


def make_inputs(batch=1, latent_hw=32, seq=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    t5_mask = torch.ones(batch, seq, dtype=torch.int64)
    t5_mask[:, 300:] = 0  # exercise the padding path
    return {
        "latents": torch.randn(batch, 16, 1, latent_hw, latent_hw, generator=g).numpy(),
        "timesteps": torch.rand(batch, generator=g).numpy(),
        "qwen_embeds": torch.randn(batch, seq, 1024, generator=g).numpy(),
        "t5_input_ids": torch.randint(0, 32128, (batch, seq), generator=g).numpy(),
        "qwen_attention_mask": torch.ones(batch, seq, dtype=torch.int64).numpy(),
        "t5_attention_mask": t5_mask.numpy(),
        "ckpt_path": np.array(CKPT),
    }


def run_reference(inputs_path: Path, outputs_path: Path) -> dict:
    if not REF_PYTHON.exists():
        raise FileNotFoundError(f"diffusion-pipe venv not found at {REF_PYTHON}")
    script = Path(__file__).parent / "ref_dump.py"
    proc = subprocess.run(
        [str(REF_PYTHON), str(script), str(inputs_path), str(outputs_path)],
        cwd=REF_ROOT,  # diffusion-pipe's imports are cwd-relative
        capture_output=True,
        text=True,
        env={**os.environ, "PARITY_MATH_SDPA": "1" if MATH_SDPA else "0"},
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("reference dump failed")
    for line in proc.stderr.splitlines():
        if line.startswith("ref "):
            print(f"  [ref] {line}")
    return dict(np.load(outputs_path))


def run_ours(inputs: dict) -> dict:
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel

    dev, dt = "cuda", torch.float32
    tr = CosmosTransformer3DModel.from_pretrained(
        ANIMA_REPO, subfolder="transformer", torch_dtype=dt
    ).to(dev).eval()
    tc = AnimaTextConditioner.from_pretrained(
        ANIMA_REPO, subfolder="text_conditioner", torch_dtype=dt
    ).to(dev).eval()

    g = lambda k, d=dt: torch.from_numpy(inputs[k]).to(dev, d)
    latents, timesteps = g("latents"), g("timesteps")
    qwen_embeds = g("qwen_embeds")
    t5_input_ids = g("t5_input_ids", torch.long)
    qwen_mask, t5_mask = g("qwen_attention_mask", torch.long), g("t5_attention_mask", torch.long)

    from contextlib import nullcontext

    ctx = nullcontext()
    if MATH_SDPA:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        ctx = sdpa_kernel(SDPBackend.MATH)

    out = {}
    with torch.no_grad(), ctx:
        crossattn = tc(
            qwen_embeds,
            t5_input_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        out["crossattn_emb"] = crossattn.float().cpu().numpy()

        _, _, _, h, w = latents.shape
        padding_mask = latents.new_zeros((1, 1, h, w))

        # RoPE: diffusers returns (cos, sin); the reference returns raw frequencies.
        # Compare in the reference's space by recovering angles via atan2.
        cos, sin = tr.rope(
            torch.cat(
                [
                    latents,
                    padding_mask.unsqueeze(2).repeat(
                        latents.shape[0], 1, latents.shape[2], 1, 1
                    ),
                ],
                dim=1,
            ),
            fps=None,
        )
        out["rope_cos"] = cos.float().cpu().numpy()
        out["rope_sin"] = sin.float().cpu().numpy()

        # --- pre-block inputs, mirroring the reference taps ---
        hs = torch.cat(
            [
                latents,
                padding_mask.unsqueeze(2).repeat(latents.shape[0], 1, latents.shape[2], 1, 1),
            ],
            dim=1,
        )
        patched = tr.patch_embed(hs).flatten(1, 3)
        out["patch_embed"] = patched.float().cpu().numpy()
        temb, embedded_timestep = tr.time_embed(patched, timesteps)
        # diffusers: temb is the adaln-lora term, embedded_timestep the normalised time embedding.
        out["adaln_lora"] = temb.float().cpu().numpy()
        out["t_emb"] = embedded_timestep.float().cpu().numpy()

        taps = {}

        def make_hook(i):
            def hook(_mod, _inp, output):
                x = output[0] if isinstance(output, (tuple, list)) else output
                taps[i] = x.detach().float().reshape(x.shape[0], -1, x.shape[-1]).cpu()
            return hook

        handles = [
            tr.transformer_blocks[i].register_forward_hook(make_hook(i)) for i in (0, 1, 6, 13, 27)
        ]

        sub = {}

        def sub_hook(name):
            def hook(_m, _i, o):
                x = o[0] if isinstance(o, (tuple, list)) else o
                sub[name] = x.detach().float().reshape(x.shape[0], -1, x.shape[-1]).cpu()
            return hook

        b0 = tr.transformer_blocks[0]
        handles += [
            b0.attn1.register_forward_hook(sub_hook("b0_self_attn")),
            b0.attn2.register_forward_hook(sub_hook("b0_cross_attn")),
            b0.ff.register_forward_hook(sub_hook("b0_mlp")),
        ]

        out["sample"] = tr(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=crossattn,
            attention_mask=None,
            padding_mask=padding_mask,
        ).sample.float().cpu().numpy()

        for h in handles:
            h.remove()
        for i, v in taps.items():
            out[f"block{i}"] = v.numpy()
        for k, v in sub.items():
            out[k] = v.numpy()
    return out


def compare(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} vs {b.shape}")
        return False
    tol = TOL.get(name, DEFAULT_TOL)
    d = np.abs(a - b)
    scale = max(np.abs(a).max(), 1e-12)
    rel = d.max() / scale
    # Bit-exactness is judged absolutely; everything else on relative error, since activation
    # magnitudes in this model span four orders of magnitude across depth.
    ok = (d.max() == 0.0) if tol == 0.0 else (rel <= tol)
    print(
        f"  {'PASS' if ok else 'FAIL'} {name:14s} max|d|={d.max():.3e} "
        f"mean|d|={d.mean():.3e} rel={rel:.3e}  (tol {tol:.0e}{' exact' if tol == 0 else ''})"
    )
    return ok


def run_case(batch: int, latent_hw: int, seed: int, verbose: bool) -> bool:
    inputs = make_inputs(batch=batch, latent_hw=latent_hw, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        ip, op = Path(d) / "in.npz", Path(d) / "out.npz"
        np.savez(ip, **inputs)
        ref = run_reference(ip, op)
    ours = run_ours(inputs)

    if not verbose:
        # Compact row: the three things that would expose a structural error.
        rope = max(
            np.abs(np.cos(ref["rope_emb"].reshape(ref["rope_emb"].shape[0], -1)) - ours["rope_cos"]).max(),
            np.abs(np.sin(ref["rope_emb"].reshape(ref["rope_emb"].shape[0], -1)) - ours["rope_sin"]).max(),
        )
        wiring = max(
            np.abs(ref[k] - ours[k].reshape(ref[k].shape)).max()
            for k in ("patch_embed", "t_emb", "adaln_lora")
        )
        rel = np.abs(ref["sample"] - ours["sample"]).max() / np.abs(ref["sample"]).max()
        ok = rope <= TOL["rope_cos"] and wiring == 0.0 and rel <= TOL["full_trunk"]
        print(
            f"  {'PASS' if ok else 'FAIL'}  b{batch} {latent_hw:>3}lat ({latent_hw * 8:>4}px)  "
            f"rope={rope:.2e}  wiring={wiring:.1e}  trunk={rel:.2e}"
        )
        return ok

    return _detailed(ref, ours)


def _detailed(ref: dict, ours: dict) -> bool:
    print("\ncomparisons (fp32):")
    results = []

    # RoPE: reference freqs are (L,1,1,D); diffusers gives cos/sin of those same angles.
    ref_freqs = ref["rope_emb"].reshape(ref["rope_emb"].shape[0], -1)
    results.append(compare("rope_cos", np.cos(ref_freqs), ours["rope_cos"]))
    results.append(compare("rope_sin", np.sin(ref_freqs), ours["rope_sin"]))

    results.append(compare("llm_adapter", ref["crossattn_emb"], ours["crossattn_emb"]))

    # Everything feeding block 0. If divergence is already here, the blocks are innocent.
    for name in ("patch_embed", "t_emb", "adaln_lora"):
        if name in ref and name in ours:
            a, b = ref[name], ours[name]
            if a.shape != b.shape and a.size == b.size:
                b = b.reshape(a.shape)
            results.append(compare(name, a, b))

    for name in ("b0_self_attn", "b0_cross_attn", "b0_mlp"):
        if name in ref and name in ours:
            a, b = ref[name], ours[name]
            d = np.abs(a - b); sc = max(np.abs(a).max(), 1e-12)
            print(f"     {name:14s} max|d|={d.max():.3e}  rel={d.max()/sc:.3e}")

    # Per-block drift: steady growth across depth = accumulation; a jump = a real divergence.
    print("  -- per-block drift (diagnostic, not a gate) --")
    for i in (0, 1, 6, 13, 27):
        k = f"block{i}"
        if k not in ref or k not in ours:
            continue
        a, b = ref[k], ours[k]
        d = np.abs(a - b)
        scale = np.abs(a).max()
        # Activations grow by orders of magnitude with depth, so absolute drift alone is
        # uninterpretable; relative is what says whether the computation agrees.
        print(
            f"     block{i:<3d} max|d|={d.max():.3e}  |x|max={scale:.3e}  rel={d.max()/scale:.3e}"
        )

    results.append(compare("full_trunk", ref["sample"], ours["sample"]))
    return all(results)


def main() -> int:
    missing = [name for name, value in (("ANIMA_NATIVE_CKPT", CKPT),
                                        ("ANIMA_REF_ROOT", str(REF_ROOT)))
               if not value]
    if missing or not REF_PYTHON.exists():
        print("SKIP  transformer parity compares against diffusion-pipe in its own venv, so it "
              "needs\n      a second trainer installed and the native Anima checkpoint. Set:\n"
              "        ANIMA_NATIVE_CKPT=/path/to/anima-base-v1.0.safetensors\n"
              "        ANIMA_MODEL=/path/to/anima-diffusers\n"
              "        ANIMA_REF_ROOT=/path/to/diffusion-pipe")
        if missing:
            print(f"      (unset: {', '.join(missing)})")
        elif not REF_PYTHON.exists():
            print(f"      (no interpreter at {REF_PYTHON})")
        return 0

    detail = "--detail" in sys.argv
    if detail:
        # One config, every intermediate tap -- for diagnosing a failure the sweep reported.
        print("detailed single-case parity (256px)\n")
        ok = run_case(1, 32, 0, verbose=True)
    else:
        print("parity sweep: resolutions x batch, fp32, vs diffusion-pipe\n")
        results = [run_case(b, hw, seed, verbose=False) for b, hw, seed in CASES]
        ok = all(results)
        print(f"\n{sum(results)}/{len(results)} configs passed")

    print(f"{'PARITY PASS' if ok else 'PARITY FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
