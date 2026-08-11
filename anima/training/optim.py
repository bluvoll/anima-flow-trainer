"""Optimizer and LR-schedule construction.

The memory arithmetic is the whole reason this file has options. Anima's trunk is 2.09B:

    bf16 weights                    4.2 GB
    bf16 gradients                  4.2 GB
    fp32 AdamW moments (2x)        16.7 GB
                                   -------
                                   25.1 GB   -- over a 4090 before a single activation

So for full finetuning a plain fp32 AdamW is not a choice, it is a non-starter. `sdnq.optim`
quantizes the moments (uint8 by default, ~4x off the 16.7 GB) and can additionally keep them in
pinned host memory, which is what makes the run fit. LoRA has no such problem -- a rank-32 adapter
has a few tens of millions of parameters -- so it defaults to stock AdamW.
"""

from __future__ import annotations

import math

import torch

from .config import OptimizerConfig, ScheduleConfig

# SDNQ optimizers assert that a param group's keys match their schema exactly, so our bookkeeping
# key has to come off first. Stock torch optimizers tolerate it, but stripping unconditionally
# keeps the two paths from diverging.
_BOOKKEEPING_KEYS = ("component",)

_SDNQ_KINDS = {"adamw8bit", "adafactor", "came", "lion", "muon"}


def _strip(groups: list[dict]) -> list[dict]:
    return [{k: v for k, v in g.items() if k not in _BOOKKEEPING_KEYS} for g in groups]


def build_optimizer(groups: list[dict], cfg: OptimizerConfig) -> torch.optim.Optimizer:
    kind = cfg.kind.lower()
    clean = _strip(groups)

    if kind == "adamw":
        if cfg.quantize_state or cfg.offload_state:
            raise ValueError(
                "quantize_state/offload_state need an sdnq optimizer; use kind='adamw8bit'"
            )
        return torch.optim.AdamW(clean, betas=tuple(cfg.betas), eps=cfg.eps)

    if kind not in _SDNQ_KINDS:
        raise ValueError(f"unknown optimizer: {cfg.kind!r} (expected 'adamw' or one of {sorted(_SDNQ_KINDS)})")

    from sdnq.optim import CAME, Adafactor, AdamW, Lion, Muon

    factory = {"adamw8bit": AdamW, "adafactor": Adafactor, "came": CAME,
               "lion": Lion, "muon": Muon}[kind]

    return factory(
        clean,
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        weight_decay=cfg.weight_decay,
        use_quantized_buffers=cfg.quantize_state,
        offload_buffers=cfg.offload_state,
        # Stochastic rounding is what makes a bf16 master weight trainable at all: round-to-nearest
        # silently discards updates smaller than half an ulp, which at 1e-5 LRs is most of them.
        use_stochastic_rounding=True,
        # Kahan composes with SR rather than replacing it -- sdnq's kahan branch still routes the
        # cast through `copy_stochastic_`. SR makes the *expected* discarded update zero; Kahan
        # makes the *realised* one zero by carrying the remainder forward. Off by default because
        # it costs a buffer per parameter and SR alone is what every measurement here was taken on.
        use_kahan=cfg.use_kahan,
    )


def _rex_shape(progress: float, d: float) -> float:
    """REX's reflected-exponential decay, normalised to 1.0 at progress 0 and 0.0 at progress 1.

    https://arxiv.org/abs/2107.04197. `d` controls how much of the run is spent near the peak:
    d -> 0 degenerates to linear, and larger d holds the LR high for longer and then drops it
    sharply at the end. sd-scripts hardcodes d = 0.9; that is the default here, not a constant.
    """
    remaining = 1.0 - progress
    return remaining / ((1.0 - d) + d * remaining)


def _rerex_lambda(decay_steps: int, cfg: ScheduleConfig):
    """Recursive REX: a global REX curve sampled at segment boundaries, with a second REX decay
    run *within* each segment between those two endpoints.

    This is NOT a warm-restart schedule -- segment i ends exactly where segment i+1 begins, so the
    curve is continuous and monotone, same as REX. What the recursion buys is control over the
    *pacing* of the decay that a single `d` cannot express: `weight_power` skews the step budget
    toward early segments (0 = equal lengths), which stretches the high-LR region and then
    compresses the tail, and `local_d` reshapes the descent inside each segment. At the defaults
    the result tracks REX early and pulls below it after the midpoint (mean multiplier over a
    1000-step run: REX 0.83, ReREX 0.76, cosine 0.50).

    Ported from the sd-scripts fork's `lr_lambda_rerex` (numpy dropped; identical arithmetic --
    verified to 0.0 by anima/parity/test_schedule_parity.py).
    """
    floor = cfg.min_lr_ratio
    n = cfg.num_segments

    def global_rex(step: int) -> float:
        return floor + (1.0 - floor) * _rex_shape(step / decay_steps, cfg.global_d)

    weights = [(n - i) ** cfg.weight_power for i in range(n)]
    total_w = sum(weights)
    steps = [int(decay_steps * w / total_w) for w in weights]
    # The floors lose a few steps; the reference gives the remainder to the final segment so the
    # boundaries land exactly on decay_steps.
    steps[-1] += decay_steps - sum(steps)

    bounds = [0]
    for s in steps:
        bounds.append(bounds[-1] + s)
    # Endpoints are read off the *global* curve, so the segment tops descend along REX.
    ends = [(global_rex(bounds[i]), global_rex(bounds[i + 1])) for i in range(n)]

    def lr_lambda(step: int) -> float:
        for i in range(n):
            if step < bounds[i + 1]:
                span = bounds[i + 1] - bounds[i]
                if span <= 0:                       # a segment rounded down to nothing
                    continue
                start_lr, end_lr = ends[i]
                p = (step - bounds[i]) / span
                return end_lr + (start_lr - end_lr) * _rex_shape(p, cfg.local_d)
        return floor

    return lr_lambda


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ScheduleConfig, total_steps: int
):
    """A LambdaLR whose multiplier is applied to each group's own peak LR, so the per-component
    ratios set in `[component_lr]` are preserved across warmup and decay.

    `total_steps` is expected to already be scaled by world size -- Accelerate's wrapper advances
    the inner scheduler once per process per `step()`. See `Trainer._build_optimizer`.
    """
    warmup = max(0, cfg.warmup_steps)
    decay_steps = max(1, total_steps - warmup)
    floor = cfg.min_lr_ratio

    if cfg.kind == "rerex":
        decay = _rerex_lambda(decay_steps, cfg)
    else:
        def decay(step: int) -> float:
            progress = min(1.0, step / decay_steps)
            if cfg.kind == "constant":
                return 1.0
            if cfg.kind == "linear":
                return floor + (1 - floor) * (1 - progress)
            if cfg.kind == "cosine":
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
            if cfg.kind == "rex":
                return floor + (1 - floor) * _rex_shape(progress, cfg.d)
            raise ValueError(f"unknown schedule: {cfg.kind!r}")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # (step+1)/warmup rather than step/warmup: the sd-scripts form makes the first
            # optimizer step a no-op at lr exactly 0.
            return (step + 1) / max(1, warmup)
        return decay(step - warmup)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def estimate_optimizer_bytes(num_trainable: int, cfg: OptimizerConfig) -> int:
    """Rough optimizer-state footprint, for the pre-flight report. Advisory, not a guarantee."""
    kind = cfg.kind.lower()
    if kind in ("adafactor",):
        per_param = 0.5          # factored second moment; first moment optional
    elif kind in ("lion",):
        per_param = 4.0          # one momentum buffer
    elif kind in ("came",):
        per_param = 6.0
    else:
        per_param = 8.0          # two fp32 moments

    if cfg.quantize_state and kind in _SDNQ_KINDS:
        per_param /= 4.0         # uint8 buffers + per-group scales
    if cfg.use_kahan and kind in _SDNQ_KINDS:
        # `zeros_like(param)` -- the residual buffer follows the parameter dtype (bf16 = 2 bytes),
        # not fp32, unless quantized buffers are on (`optim/optimizer.py:102-104`).
        per_param += 0.5 if cfg.quantize_state else 2.0
    if cfg.offload_state and kind in _SDNQ_KINDS:
        return 0                 # lives in host memory
    return int(num_trainable * per_param)
