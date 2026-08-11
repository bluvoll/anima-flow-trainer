"""Phase 0.4/0.5 gate: prove SDNQ's stock Triton kernels are correct AND faster on Ada,
and that the quantized-master-weight training path passes gradients.

We do not write kernels. This only measures SDNQ's own.

Shapes are Anima's real ones: the MLP (2048 -> 8192) dominates the 28-block trunk.
"""

import time

import torch
from sdnq import SDNQConfig, sdnq_quantize_layer
from sdnq.training import convert_sdnq_layer_to_training

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
IN_F, OUT_F = 2048, 8192  # Anima block mlp.layer1
TOKENS = 4096  # ~1024x1024 latent seq len (128/2 * 128/2)


def _bench(fn, warmup=5, iters=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def run_case(weights_dtype, use_quantized_matmul, ref_lin, x, ref_out):
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=DTYPE)
    lin.load_state_dict(ref_lin.state_dict())

    cfg = SDNQConfig(
        weights_dtype=weights_dtype,
        use_quantized_matmul=use_quantized_matmul,
        quantization_device=DEV,
        return_device=DEV,
    )
    lin, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=DTYPE, param_name="mlp.layer1")
    lin = lin.to(DEV)

    with torch.no_grad():
        out = lin(x)
        err = (out.float() - ref_out.float()).abs()
        rel = (err.mean() / ref_out.float().abs().mean()).item()
        ms = _bench(lambda: lin(x))

    tag = f"{weights_dtype:<8} qmm={str(use_quantized_matmul):<5}"
    print(f"  {tag}  rel_err={rel:.5f}  {ms:7.3f} ms")
    return rel, ms


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  sm={torch.cuda.get_device_capability(0)}")
    from sdnq.common import use_torch_compile

    print(f"sdnq use_torch_compile={use_torch_compile}\n")

    torch.manual_seed(0)
    ref_lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=DTYPE).to(DEV)
    x = torch.randn(1, TOKENS, IN_F, device=DEV, dtype=DTYPE)

    with torch.no_grad():
        ref_out = ref_lin(x)
        base_ms = _bench(lambda: ref_lin(x))
    print(f"  {'bf16':<8} {'(reference)':<10}  rel_err=0.00000  {base_ms:7.3f} ms")

    results = {}
    for wd in ("int8", "float8_e4m3fn"):
        for qmm in (False, True):
            results[(wd, qmm)] = run_case(wd, qmm, ref_lin, x, ref_out)

    print("\n  speedup vs bf16 (quantized matmul on):")
    for wd in ("int8", "float8_e4m3fn"):
        rel, ms = results[(wd, True)]
        print(f"    {wd:<14} {base_ms / ms:5.2f}x   rel_err={rel:.5f}")

    # --- Phase 0.5: gradients through quantized master weights ---
    print("\n  training path (convert_sdnq_model_to_training):")
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=DTYPE).to(DEV)
    cfg = SDNQConfig(
        weights_dtype="int8",
        use_stochastic_rounding=True,
        quantization_device=DEV,
        return_device=DEV,
        is_training=True,
    )
    lin, cfg = sdnq_quantize_layer(lin, cfg, torch_dtype=DTYPE, param_name="mlp.layer1")
    # convert_sdnq_model_to_training() wants a model carrying .quantization_config;
    # at layer granularity the layer-level entry point is the right one.
    model = convert_sdnq_layer_to_training(lin, use_stochastic_rounding=True, inplace=True)
    model = model.to(DEV)

    xg = torch.randn(1, 512, IN_F, device=DEV, dtype=DTYPE, requires_grad=True)
    out = model(xg)
    out.float().pow(2).mean().backward()

    params = [p for p in model.parameters() if p.requires_grad]
    with_grad = [p for p in params if p.grad is not None]
    print(f"    trainable params: {len(params)}, with grad: {len(with_grad)}")
    print(f"    input grad finite: {torch.isfinite(xg.grad).all().item()}")
    for p in with_grad:
        print(f"    weight grad: shape={tuple(p.grad.shape)} finite={torch.isfinite(p.grad).all().item()}")
    assert with_grad, "no gradients reached the quantized weights"
    print("\n  PASS")


if __name__ == "__main__":
    main()
