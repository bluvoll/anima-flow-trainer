"""Phase 5 gate: what does quantizing the trunk actually cost, and where?

SDNQ has no `CosmosTransformer3DModel` entry in its per-model skip table, so Anima falls back to
the generic list -- which does not cover the per-block AdaLN. Every DiT family that *does* have an
entry skips at least the first block's modulation. This measures whether that matters here rather
than inheriting the assumption.

Reference is the bf16 model's own output on identical inputs, so the number reported is the error
quantization *adds*. Calibration from Phase 2: two mathematically identical implementations differ
by ~5e-3 relative in bf16, so a policy landing near that floor is indistinguishable from noise,
and one an order of magnitude above it is a real quality loss.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.modeling.loader import load_components  # noqa: E402
from anima.training.quant import QuantConfig, quantize_module, quantized_layer_report  # noqa: E402

MODEL = os.environ.get("ANIMA_MODEL", "../anima-diffusers")
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
BF16_NOISE_FLOOR = 5e-3  # measured in Phase 2 between identical implementations


def make_inputs(h_lat: int, w_lat: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, 16, 1, h_lat, w_lat, generator=g).to(DEV, DTYPE)
    context = torch.randn(1, 512, 1024, generator=g).to(DEV, DTYPE)
    timestep = torch.full((1,), 0.5, device=DEV, dtype=DTYPE)
    padding_mask = torch.zeros(1, 1, h_lat * 8, w_lat * 8, device=DEV, dtype=DTYPE)
    return latents, context, timestep, padding_mask


@torch.no_grad()
def run(model, inputs):
    latents, context, timestep, padding_mask = inputs
    return model(
        hidden_states=latents, timestep=timestep, encoder_hidden_states=context,
        padding_mask=padding_mask, return_dict=False,
    )[0].float()


def bench(fn, warmup=5, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def load_transformer():
    return load_components(
        MODEL, dtype=DTYPE, load_text_encoder=False, load_vae=False, load_tokenizers=False
    ).transformer


def sweep(sizes: tuple[int, ...] = (48, 64, 96, 128, 160, 192)) -> int:
    """Map the quantized-matmul crossover.

    Quantizing activations costs a fixed amount per call and saves time proportional to the matmul.
    Below some sequence length the overhead wins and int8 is *slower than bf16* -- which is the
    opposite of the headline claim, and silently so. This finds where that happens.

    Two different crossovers live in this table and they are not the same number:
      * `vs bf16` answers "is quantizing worth it at all" -- a memory-vs-speed question;
      * `qmm vs no-qmm` (last column) is what `use_quantized_matmul="auto"` actually decides, since
        by then the model is quantized either way and the flag only picks the matmul path. That is
        the crossover `QMM_TOKEN_CROSSOVER` encodes, so pass `--sizes` to pin it down.
    """
    print("\nquantized-matmul crossover vs sequence length\n")
    print(f"  {'px':>6} {'tokens':>8} {'bf16':>9} {'int8 qmm':>10} {'int8 no-qmm':>12} "
          f"{'vs bf16':>9} {'qmm/no-qmm':>11}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 9} {'-' * 10} {'-' * 12} {'-' * 9} {'-' * 11}")

    for lat in sizes:
        inputs = make_inputs(lat, lat)
        tokens = (lat // 2) ** 2

        torch.cuda.empty_cache()
        ref_model = load_transformer().to(DEV).eval()
        bf16_ms = bench(lambda: run(ref_model, inputs))
        del ref_model
        torch.cuda.empty_cache()

        times = {}
        for qmm in (True, False):
            model = load_transformer()
            model = quantize_module(
                model,
                QuantConfig(mode="frozen", skip_policy="default", use_quantized_matmul=qmm),
                DEV, DTYPE,
            ).to(DEV).eval()
            times[qmm] = bench(lambda: run(model, inputs))
            del model
            torch.cuda.empty_cache()

        ratio = bf16_ms / times[True]
        # The decision `auto` makes: >1 means turning quantized matmul ON is the faster of the two
        # quantized paths at this sequence length.
        qmm_gain = times[False] / times[True]
        flag = "  <- qmm WINS" if qmm_gain > 1.0 else ""
        print(f"  {lat * 8:>6} {tokens:>8} {bf16_ms:8.1f}ms {times[True]:9.1f}ms "
              f"{times[False]:11.1f}ms {ratio:8.2f}x {qmm_gain:10.2f}x{flag}", flush=True)
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96, help="latent side (96 = 768px)")
    ap.add_argument("--sweep", action="store_true", help="map the quantized-matmul crossover")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="latent sides for --sweep; tokens = (side/2)^2. e.g. 96 104 112 120 128 "
                         "brackets the 2304-4096 gap where QMM_TOKEN_CROSSOVER sits")
    args = ap.parse_args()

    if args.sweep:
        return sweep(tuple(args.sizes)) if args.sizes else sweep()

    inputs = make_inputs(args.size, args.size)

    print(f"\nreference: bf16, {args.size * 8}px\n")
    ref_model = load_transformer().to(DEV).eval()
    ref = run(ref_model, inputs)
    ref_ms = bench(lambda: run(ref_model, inputs))
    ref_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  bf16            {ref_mem:5.2f} GB  {ref_ms:7.1f} ms")
    del ref_model
    torch.cuda.empty_cache()

    cases = [
        ("int8 default skips", QuantConfig(mode="frozen", skip_policy="default")),
        ("int8 +block0 adaln", QuantConfig(mode="frozen", skip_policy="first_block_adaln")),
        ("int8 +all adaln", QuantConfig(mode="frozen", skip_policy="all_adaln")),
        ("int8 no qmatmul", QuantConfig(mode="frozen", skip_policy="first_block_adaln",
                                        use_quantized_matmul=False)),
        ("fp8 +block0 adaln", QuantConfig(mode="frozen", weights_dtype="float8_e4m3fn",
                                          skip_policy="first_block_adaln")),
    ]

    print(f"\n  {'config':<20} {'quant/total':>12} {'mem':>9} {'speed':>10} {'rel err':>10}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 9} {'-' * 10} {'-' * 10}")

    results = []
    for label, qcfg in cases:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = load_transformer()
        model = quantize_module(model, qcfg, DEV, DTYPE).to(DEV).eval()
        nq, ntotal, _ = quantized_layer_report(model)

        out = run(model, inputs)
        rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
        ms = bench(lambda: run(model, inputs))
        mem = torch.cuda.memory_allocated() / 1e9

        print(f"  {label:<20} {f'{nq}/{ntotal}':>12} {mem:6.2f} GB "
              f"{ms:7.1f} ms {rel:10.5f}")
        results.append((label, rel, ms, mem))

        del model
        torch.cuda.empty_cache()

    print(f"\n  bf16 baseline: {ref_mem:.2f} GB, {ref_ms:.1f} ms")
    print(f"  Phase 2 bf16 noise floor for reference: {BF16_NOISE_FLOOR:.1e} relative\n")

    best = min(results, key=lambda r: r[1])
    print(f"  lowest error: {best[0]} at {best[1]:.5f} "
          f"({best[1] / BF16_NOISE_FLOOR:.1f}x the bf16 noise floor)")

    # The gate is speed, not error: error is a tuning choice the table informs, but int8 without
    # a working quantized matmul is a silent regression to bf16 speed and always a bug.
    qmm = next(r for r in results if r[0] == "int8 +block0 adaln")
    no_qmm = next(r for r in results if r[0] == "int8 no qmatmul")
    speedup = no_qmm[2] / qmm[2]
    print(f"  quantized matmul: {speedup:.2f}x vs the same config without it")

    ok = speedup > 1.2
    print(f"\nSDNQ QUALITY {'PASS' if ok else 'FAIL (quantized matmul is not engaging)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
