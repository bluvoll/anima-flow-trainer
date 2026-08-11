"""SDNQ quantization for the trunk.

Two distinct modes, and the distinction matters:

**frozen** -- for LoRA. Base weights become int8 and never change; only the adapter is trained in
bf16. This is the big win: the base is 84% of memory and 100% of it is dead weight during a LoRA
run.

**training** -- for full finetuning. Weights become `SDNQTensor` master weights that are quantized
*and* updated, using stochastic rounding so that updates smaller than half an ulp are not silently
discarded. Round-to-nearest at a 1e-5 LR throws away most of the signal.

`use_quantized_matmul` is where the speed lives -- but only above a sequence-length crossover. On
an isolated 2048->8192 matmul at 4096 tokens it is 2.91x bf16; across the *whole trunk* it is
1.39x at 1024px and **0.44x at 512px**, because it carries a large fixed per-call cost. See
`QMM_TOKEN_CROSSOVER` for the measured table. `"auto"` is the default for that reason.

## Skip lists

SDNQ's `common_skip_keys` covers `time_embed`, `patch_embed`, `proj_out`, `norm_out` -- the
in/out projections. It has no `CosmosTransformer3DModel` entry in its per-model table, so Anima
falls back to that generic list.

Every DiT family that *does* have an entry -- Flux, Flux2, QwenImage, LongCatVideo -- explicitly
skips the first block's modulation weight (`transformer_blocks.0.img_mod.1.weight`,
`blocks.0.adaLN_modulation.1.weight`). AdaLN emits shift/scale/gate applied to the whole residual
stream, so error there is multiplicative rather than additive.

Measured for Anima: the default path already leaves block 0's modulation unquantized (12 layers
skipped, 442/454 quantized), so the `first_block_adaln` policy is a no-op here -- it is kept as an
explicit statement of intent rather than removed. Skipping *every* block's modulation
(`all_adaln`, 280/454) lowers output error ~13% and costs ~0.13GB. `test_sdnq_quality.py` measures
all three.

Note also that quantized matmul quantizes activations, not just weights: at 1024px it roughly
*doubles* relative output error (0.014 -> 0.024) versus int8 weights alone. That is a real
speed/quality trade, not a free win.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass, field

import torch
from torch import nn

# Anima's per-block AdaLN modulation. Not in SDNQ's generic list; see the module docstring.
#
# Key syntax is SDNQ's, and it is not substring matching (`sdnq/utils.py:29`): a key matches if it
# equals the full parameter name, or is one whole dot-separated component of it, or is a glob.
# `norm1.linear_` matches nothing at all -- it is neither. `norm1` matches as a component, and in
# Cosmos blocks norm{1,2,3} *are* the modulation modules, so it is exactly the right granularity.
_ADALN_ALL = ["norm1", "norm2", "norm3"]
_ADALN_FIRST_BLOCK = [
    f"transformer_blocks.0.norm{i}.linear_{j}.weight" for i in (1, 2, 3) for j in (1, 2)
]

# Supplied by SDNQ's author for Anima specifically, as the layers to leave in bf16 when int8
# matmul is on. Full parameter names, which is the right syntax: `check_param_name_in`
# (`sdnq/utils.py:29`) matches on `param_name == param`, on a whole dot-separated component, or on
# a glob -- so a name containing dots is an exact name, never a substring pattern.
#
# The shape of the list is the interesting part, and it is NOT "skip the MLP":
#
#   ff.net.2 (down-proj, 8192->2048)   16 blocks -- 2,3,5,7,9,10,11,12,13,21..27
#   ff.net.0.proj (up-proj)             3 blocks -- 0,1,2 only
#   attn1/attn2 projections            14 layers -- blocks 0,1,2,3 and 27 only
#
# i.e. the down-projection through the deep half, plus the first three and last blocks. That
# matches the standard activation-outlier story: `ff.net.2` reads the post-activation 8192-wide
# tensor, where a few channels are orders of magnitude larger than the rest, so a single per-row
# int8 scale spends its range on the outliers. Skipping the whole `ff` stack instead would be 56
# layers and 48% of the trunk's Linear parameters; this is 33 layers and 19.3% (377M params,
# ~+377MB over int8), which is why it is worth having as its own policy.
#
# Caveat worth stating: it was given for *int8 matmul*, which quantizes activations. In
# `mode="training"` we run with quantized matmul off, where only the weights are int8. The
# sensitive layers overlap heavily but are not guaranteed identical -- treat this as a strong prior
# rather than a measurement of our configuration.
_ANIMA_INT8_MM = [
    "transformer_blocks.0.attn1.to_k.weight",
    "transformer_blocks.0.attn1.to_out.0.weight",
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn2.to_q.weight",
    "transformer_blocks.0.ff.net.0.proj.weight",
    "transformer_blocks.1.attn1.to_k.weight",
    "transformer_blocks.1.attn1.to_v.weight",
    "transformer_blocks.1.attn2.to_q.weight",
    "transformer_blocks.1.ff.net.0.proj.weight",
    "transformer_blocks.2.attn1.to_k.weight",
    "transformer_blocks.2.attn1.to_q.weight",
    "transformer_blocks.2.attn1.to_v.weight",
    "transformer_blocks.2.attn2.to_q.weight",
    "transformer_blocks.2.ff.net.0.proj.weight",
    "transformer_blocks.2.ff.net.2.weight",
    "transformer_blocks.3.attn1.to_v.weight",
    "transformer_blocks.3.ff.net.2.weight",
    "transformer_blocks.5.ff.net.2.weight",
    "transformer_blocks.7.ff.net.2.weight",
    "transformer_blocks.9.ff.net.2.weight",
    "transformer_blocks.10.ff.net.2.weight",
    "transformer_blocks.11.ff.net.2.weight",
    "transformer_blocks.12.ff.net.2.weight",
    "transformer_blocks.13.ff.net.2.weight",
    "transformer_blocks.21.ff.net.2.weight",
    "transformer_blocks.22.ff.net.2.weight",
    "transformer_blocks.23.ff.net.2.weight",
    "transformer_blocks.24.ff.net.2.weight",
    "transformer_blocks.25.ff.net.2.weight",
    "transformer_blocks.26.ff.net.2.weight",
    "transformer_blocks.27.attn1.to_k.weight",
    "transformer_blocks.27.attn1.to_q.weight",
    "transformer_blocks.27.ff.net.2.weight",
]

# Every `ff.net.2` in the trunk -- the friend's "don't quantize the MLP", narrowed to the half of
# the MLP the outlier argument actually implicates. 28 layers, 470M params (~+470MB over int8).
# Between `anima_int8_mm` (16 of 28) and this lies the question of whether the author's sweep found
# the shallow blocks genuinely insensitive or simply did not flag them; unmeasured here.
_MLP_DOWN_ALL = [f"transformer_blocks.{i}.ff.net.2.weight" for i in range(28)]

SKIP_POLICIES = ("default", "first_block_adaln", "all_adaln", "anima_int8_mm", "mlp_down")


@dataclass
class QuantConfig:
    """Quantization settings. `mode='none'` leaves the model in bf16."""

    mode: str = "none"                  # "none" | "frozen" | "training"
    weights_dtype: str = "int8"         # int8 default: faster than fp8 here, and 2x lower error
    # True | False | "auto". "auto" enables it only above QMM_TOKEN_CROSSOVER -- see the table on
    # `resolve_quantized_matmul`, where below 1024px it is a 2x *slowdown*.
    use_quantized_matmul: bool | str = "auto"
    quantize_text_conditioner: bool = False
    quantize_text_encoder: bool = False

    # "default"           -- SDNQ's generic list, which for this model already leaves block 0's
    #                        modulation in bf16 (verified: 12 layers skipped, 442/454 quantized)
    # "first_block_adaln" -- explicit form of the same thing; measured identical to "default", kept
    #                        because it documents the intent and survives an upstream change
    # "all_adaln"         -- every block's modulation stays bf16 (280/454 quantized)
    skip_policy: str = "default"
    extra_skip: list[str] = field(default_factory=list)

    # Per-layer bit-width search: SDNQ raises precision for any layer whose quantization error
    # exceeds this. SD.Next's guidance is 1e-4 / 1e-3 / 1e-2 for 8 / 6 / 4-bit targets.
    dynamic_loss_threshold: float | None = None

    group_size: int = 0
    use_stochastic_rounding: bool = True

    def __post_init__(self):
        if self.mode not in ("none", "frozen", "training"):
            raise ValueError(f"unknown quant mode: {self.mode!r}")
        if self.skip_policy not in SKIP_POLICIES:
            raise ValueError(f"unknown skip_policy: {self.skip_policy!r} (expected {SKIP_POLICIES})")
        if self.use_quantized_matmul not in (True, False, "auto"):
            raise ValueError(
                f"use_quantized_matmul must be true, false, or \"auto\", "
                f"got {self.use_quantized_matmul!r}"
            )

    def skip_keys(self) -> list[str]:
        keys = list(self.extra_skip)
        if self.skip_policy == "all_adaln":
            keys += _ADALN_ALL
        elif self.skip_policy == "first_block_adaln":
            keys += _ADALN_FIRST_BLOCK
        elif self.skip_policy == "anima_int8_mm":
            keys += _ANIMA_INT8_MM
        elif self.skip_policy == "mlp_down":
            keys += _MLP_DOWN_ALL
        return keys


# Measured on Ada (GPU:1), full trunk, bf16 vs int8 with and without quantized matmul:
#
#     px    tokens    bf16    int8 qmm   int8 no-qmm   qmm vs bf16   qmm vs no-qmm
#    384       576   57.8ms    122.5ms       109.4ms         0.47x
#    512      1024   56.6ms    129.0ms       101.8ms         0.44x
#    704      1936   76.5ms    123.9ms       107.0ms         0.62x           0.86x
#    768      2304   93.6ms    121.7ms       107.1ms         0.77x           0.88x
#    832      2704  110.5ms    129.0ms       115.2ms         0.86x           0.89x
#    896      3136  132.8ms    123.6ms       137.3ms         1.07x           1.11x  <- flips here
#    960      3600  153.6ms    120.3ms       159.4ms         1.28x           1.33x
#   1024      4096  169.4ms    130.1ms       174.9ms         1.30x           1.34x
#   1280      6400  308.5ms    229.2ms       315.5ms         1.35x
#   1536      9216  493.6ms    380.0ms       500.4ms         1.30x
#
# int8-with-matmul is nearly *flat* -- 123.9ms at 1936 tokens, 130.1ms at 4096 -- so it is
# dominated by a fixed per-call cost (442 quantized layers, each quantizing its activations)
# rather than by the matmul itself. The no-qmm path scales with tokens (107 -> 175ms over the same
# span), so the two cross.
#
# Note the LAST column is the one this constant encodes. `auto` runs on an already-quantized model,
# so the choice is qmm vs no-qmm, not vs bf16 -- two different crossovers, and they do not
# coincide. The sign flip is bracketed by measurement: qmm loses at 2704 (0.89x) and wins at 3136
# (1.11x), so the crossover lies between them and this constant is their midpoint. It was 3200
# until those two points were measured, which sat *above* the real flip and left the 2900-3200
# band incorrectly on the no-qmm side.
#
# Re-measure with:  test_sdnq_quality.py --sweep --sizes 88 96 104 112 120 128
QMM_TOKEN_CROSSOVER = 2900


def resolve_quantized_matmul(
    cfg: QuantConfig, tokens: "Sequence[int] | int | None"
) -> bool:
    """Turn `use_quantized_matmul="auto"` into a decision.

    `tokens` is one entry per training sample (an int is accepted as a single-bucket shorthand).

    Sized on a **time-weighted mean**, not the maximum. The maximum was correct while a run had one
    resolution -- every step looked like the largest bucket. Under multi-resolution it is actively
    wrong: a 768/1280 ladder has a 6240-token top bucket, so a max-based rule would switch quantized
    matmul on globally and pay the measured ~2x slowdown on every 768-token batch. Weighting each
    sample by its token count approximates its share of wall-clock, which is the thing the decision
    should actually turn on.

    This setting is one flag for the whole model, so it cannot be right for every bucket in a
    straddling ladder -- see `straddles_crossover`. Per-batch switching is possible but unmeasured.

    Compressing a timing curve into one threshold is lossy, so the loss was measured rather than
    assumed: across 162 ladders, comparing this rule against interpolating the measured table
    directly, worst-case regret is **3.8%** of step time, mean **0.15%**, and no ladder exceeds 5%.
    That is well inside the noise a real run sees, so the simple rule stays.
    """
    if cfg.use_quantized_matmul != "auto":
        return bool(cfg.use_quantized_matmul)

    # `mode = "training"` is a different regime from the one the crossover was measured in, and the
    # constant does not transfer. Every measurement behind QMM_TOKEN_CROSSOVER was taken on FROZEN
    # weights, where the weight-side quantization is done once and amortised over the run. In
    # training mode the master weights change every step, so that work is repaid every step -- and
    # the Triton kernels are compiled per shape, so a bucketed dataset pays a JIT spike each time a
    # new bucket first appears, at any point in the run.
    #
    # Measured on a 1.76B full finetune, 91 images, 21 buckets (3312-4096 tokens, i.e. every bucket
    # ABOVE the crossover, so the token rule would have said yes), 16 steps, warm Triton cache:
    #
    #                 mean s/it   median s/it
    #     qmm on         19.02        9.54
    #     qmm off         8.76        8.88
    #
    # The medians are a wash -- steady-state qmm is ~7% SLOWER here, not faster. The 2.2x mean gap
    # is two 79 s/it compile spikes at steps 6 and 7, where new bucket shapes appeared mid-run. On
    # a cold cache the same config measured 113-178 s/it against 5-13 with qmm off.
    #
    # So: no measured upside, a recurring downside. `auto` says no. Setting it True explicitly
    # still works and is how the A/B is run -- see `_report`, which says what that costs.
    if cfg.mode == "training":
        return False

    if tokens is None:
        return False
    if isinstance(tokens, int):
        return tokens >= QMM_TOKEN_CROSSOVER
    tokens = list(tokens)
    if not tokens:
        return False
    total = sum(tokens)
    if total <= 0:
        return False
    return sum(t * t for t in tokens) / total >= QMM_TOKEN_CROSSOVER


def straddles_crossover(tokens: "Sequence[int]") -> tuple[int, int]:
    """(samples below the crossover, samples at or above it). Both non-zero means no single
    `use_quantized_matmul` setting is right for the whole run."""
    below = sum(1 for t in tokens if t < QMM_TOKEN_CROSSOVER)
    return below, len(tokens) - below


def tokens_for_bucket(bucket: tuple[int, int]) -> int:
    """Pixels -> DiT sequence length: /8 for the VAE, then /2 for the 2x2 patch embed."""
    w, h = bucket
    return (w // 16) * (h // 16)


def build_sdnq_config(
    cfg: QuantConfig, device: torch.device, is_training: bool, use_qmm: bool | None = None
):
    from sdnq import SDNQConfig

    return SDNQConfig(
        weights_dtype=cfg.weights_dtype,
        use_quantized_matmul=resolve_quantized_matmul(cfg, None) if use_qmm is None else use_qmm,
        group_size=cfg.group_size,
        dynamic_loss_threshold=cfg.dynamic_loss_threshold,
        use_stochastic_rounding=cfg.use_stochastic_rounding,
        modules_to_not_convert=cfg.skip_keys(),
        add_skip_keys=True,      # also apply SDNQ's generic list
        quantization_device=device,
        return_device=device,
        is_training=is_training,
    )


def quantize_module(
    module: nn.Module,
    cfg: QuantConfig,
    device: torch.device,
    dtype: torch.dtype,
    use_qmm: bool | None = None,
) -> nn.Module:
    """Quantize one module in place-ish (returns the converted module)."""
    from sdnq import apply_sdnq_to_module
    from sdnq.training import add_module_skip_keys, apply_sdnq_training_to_module

    is_training = cfg.mode == "training"
    sdnq_cfg = build_sdnq_config(cfg, device, is_training, use_qmm)

    # Mirrors what SDNQ's own model-level entry point does: fold the generic and per-model skip
    # keys into the config before walking the module tree.
    module, sdnq_cfg = add_module_skip_keys(module, sdnq_cfg)

    apply = apply_sdnq_training_to_module if is_training else apply_sdnq_to_module
    module, _ = apply(module, sdnq_cfg, torch_dtype=dtype)
    return module


def dequantize_state_dict(
    state_dict: dict, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Turn `SDNQTensor` master weights back into plain tensors.

    Required before export: safetensors cannot serialize an SDNQTensor (it raises
    "Attempted to access the data pointer on an invalid python storage"), so a quantized full
    finetune would train fine and then fail at the first checkpoint without this.

    The exported weights are the dequantized values, so the checkpoint is an ordinary bf16 model --
    it carries the quantization error the training accumulated, but needs no special loader.
    """
    from sdnq.training import SDNQTensor

    out = {}
    for key, value in state_dict.items():
        if isinstance(value, SDNQTensor):
            value = value.dequantize(dtype)
        out[key] = value.detach().to(dtype)
    return out


def quantized_layer_report(module: nn.Module) -> tuple[int, int, list[str]]:
    """-> (quantized Linear count, total Linear count, names left in bf16).

    Worth printing: a skip list that matches nothing is indistinguishable from a correct one until
    quality drops, and a typo'd key fails silently.
    """
    from sdnq.layers import SDNQLayer, SDNQLinear

    quantized, skipped = 0, []
    for name, m in module.named_modules():
        if isinstance(m, (SDNQLinear, SDNQLayer)):
            quantized += 1
        elif isinstance(m, nn.Linear):
            skipped.append(name)
    return quantized, quantized + len(skipped), skipped
