"""Per-component parameter groups, freezing, and PEFT adapter setup.

Anima's trunk is not homogeneous, and training all of it at one LR is rarely what you want. The
component split below is the one diffusion-pipe exposes (anima.py:1060) and it is behavioural, not
cosmetic:

    self_attn    attn1.*        image-internal structure -- composition, anatomy
    cross_attn   attn2.*        text->image binding; where prompt adherence lives
    mlp          ff.*           the bulk of capacity; where style is stored
    adaln        norm{1,2,3}.linear_*, norm_out.*, time_embed.*   timestep modulation
    base         patch_embed, proj_out                            in/out projections
    llm_adapter  the whole AnimaTextConditioner

An LR of exactly 0 means freeze -- `requires_grad_(False)`, not a zero-LR group. The distinction
matters: a frozen parameter allocates no gradient and no optimizer state, which on 24GB is the
difference between fitting and not. adaln and llm_adapter default to frozen; adaln because
modulation is a global property that a finetune destabilises easily, llm_adapter because moving
the text path invalidates every prompt the base model already understands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

# Order matters: first match wins, so the specific adaln patterns are tested before the
# `transformer_blocks.N.` catch-alls they live inside.
_COMPONENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("adaln", re.compile(r"^transformer_blocks\.\d+\.norm[123]\.linear_[12]\.")),
    ("self_attn", re.compile(r"^transformer_blocks\.\d+\.attn1\.")),
    ("cross_attn", re.compile(r"^transformer_blocks\.\d+\.attn2\.")),
    ("mlp", re.compile(r"^transformer_blocks\.\d+\.ff\.")),
    ("adaln", re.compile(r"^(norm_out|time_embed)\.")),
    ("base", re.compile(r"^(patch_embed|proj_out)\.")),
]

COMPONENTS = ("self_attn", "cross_attn", "mlp", "adaln", "base", "llm_adapter", "text_encoder")

# LoRA-injectable sites, per component. Only Linear layers -- norms and embeddings are excluded
# because PEFT would either refuse them or adapt something with no meaningful low-rank structure.
#
# Measured against sd-scripts' Anima defaults (`Block`, `PatchEmbed`, `TimestepEmbedding`,
# `FinalLayer`), by share of the DiT's 1956M Linear parameters:
#     attn1 24.0% | attn2 18.0% | ff 48.0% | adaln 9.0% | patch/time/out 1.0%
# So kohya's default reaches 11.1% more than self_attn+cross_attn+mlp, and 9 of those 11 points
# are adaln. Selecting `adaln` here closes essentially the whole gap; `base` is the last 1%.
_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "self_attn": ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"),
    "cross_attn": ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"),
    "mlp": ("ff.net.0.proj", "ff.net.2"),
    "adaln": ("norm1.linear_1", "norm1.linear_2", "norm2.linear_1", "norm2.linear_2",
              "norm3.linear_1", "norm3.linear_2", "norm_out.linear_1", "norm_out.linear_2",
              "time_embed.t_embedder.linear_1", "time_embed.t_embedder.linear_2"),
    "base": ("patch_embed.proj", "proj_out"),
}

# Qwen3's own attention/MLP projections, for the optional text-encoder LoRA (sd-scripts' `lora_te`).
_TEXT_ENCODER_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def classify(name: str) -> str:
    """Map a transformer parameter name to its component. Unmatched names are an error, not a
    silent drop into a default bucket -- a diffusers version bump that renames a submodule would
    otherwise quietly stop training it."""
    for component, pattern in _COMPONENT_PATTERNS:
        if pattern.match(name):
            return component
    raise KeyError(f"unclassified transformer parameter: {name!r}")


@dataclass
class ComponentLRs:
    """Per-component learning rates. None means "use the global lr"; 0.0 means freeze.

    Applies to **both** full finetuning and LoRA. Under LoRA the split is over the adapter tensors,
    grouped by the component of the base module each one wraps -- `classify()` matches LoRA
    parameter names unchanged, because the peft suffix (`.lora_A.default.weight`) sits after the
    part the patterns key on.

    `text_encoder` covers a Qwen3 adapter (`adapter.train_text_encoder`) and exists because that is
    the one component where a shared LR is actively dangerous: sd-scripts exposes `text_encoder_lr`
    separately for a reason, and the usual guidance is well below the trunk's.
    """

    self_attn: float | None = None
    cross_attn: float | None = None
    mlp: float | None = None
    adaln: float | None = 0.0
    base: float | None = 0.0
    llm_adapter: float | None = 0.0
    text_encoder: float | None = None

    def resolve(self, component: str, default_lr: float) -> float:
        lr = getattr(self, component)
        return default_lr if lr is None else float(lr)

    def explicit(self) -> dict[str, float]:
        """Components the user actually set, for reporting and for validation against
        `adapter.components` -- a LR on a component with no adapter injected is a no-op."""
        default = ComponentLRs()
        return {
            c: getattr(self, c)
            for c in COMPONENTS
            if getattr(self, c) != getattr(default, c)
        }


@dataclass
class ParamGroupReport:
    groups: list[dict]
    counts: dict[str, int]      # component -> trainable parameter count
    frozen: dict[str, int]      # component -> frozen parameter count

    def summary(self) -> str:
        lines = []
        for c in COMPONENTS:
            n_train, n_frozen = self.counts.get(c, 0), self.frozen.get(c, 0)
            if not (n_train or n_frozen):
                continue
            lr = next((g["lr"] for g in self.groups if g["component"] == c), 0.0)
            state = f"lr={lr:.2e}" if n_train else "FROZEN"
            lines.append(f"  {c:<12} {n_train / 1e6:8.2f}M trainable  {n_frozen / 1e6:8.2f}M frozen  {state}")
        total = sum(self.counts.values())
        lines.append(f"  {'TOTAL':<12} {total / 1e6:8.2f}M trainable "
                     f"({total / max(total + sum(self.frozen.values()), 1):.1%})")
        return "\n".join(lines)


def build_param_groups(
    transformer: nn.Module,
    text_conditioner: nn.Module | None,
    lrs: ComponentLRs,
    default_lr: float,
    weight_decay: float = 0.0,
) -> ParamGroupReport:
    """Split parameters by component, freeze the zero-LR ones, and return optimizer groups.

    Freezing happens here rather than in the caller so that `requires_grad` and the optimizer
    groups can never disagree -- a parameter is in a group if and only if it is trainable.
    """
    buckets: dict[str, list[torch.nn.Parameter]] = {c: [] for c in COMPONENTS}
    counts: dict[str, int] = {}
    frozen: dict[str, int] = {}

    named = [(classify(n), p) for n, p in transformer.named_parameters()]
    if text_conditioner is not None:
        named += [("llm_adapter", p) for _, p in text_conditioner.named_parameters()]

    for component, param in named:
        if lrs.resolve(component, default_lr) == 0.0:
            param.requires_grad_(False)
            frozen[component] = frozen.get(component, 0) + param.numel()
            continue
        param.requires_grad_(True)
        buckets[component].append(param)
        counts[component] = counts.get(component, 0) + param.numel()

    groups = [
        {
            "params": params,
            "lr": lrs.resolve(component, default_lr),
            "weight_decay": weight_decay,
            "component": component,
        }
        for component, params in buckets.items()
        if params
    ]
    if not groups:
        raise ValueError("every component is frozen; nothing to train")

    return ParamGroupReport(groups=groups, counts=counts, frozen=frozen)


def build_adapter_param_groups(
    transformer: nn.Module,
    text_conditioner: nn.Module | None,
    text_encoder: nn.Module | None,
    lrs: ComponentLRs,
    default_lr: float,
    weight_decay: float = 0.0,
) -> ParamGroupReport:
    """The LoRA counterpart of `build_param_groups`: split the *adapter* tensors by the component
    of the base module they wrap.

    Only parameters that are already trainable are considered -- `apply_adapter` has frozen the
    bases by then, so this sees adapter tensors and nothing else. A component set to 0.0 is frozen,
    though dropping it from `adapter.components` is better: that skips injecting the adapter at all
    rather than carrying dead zero tensors into the export.
    """
    buckets: dict[str, list[torch.nn.Parameter]] = {c: [] for c in COMPONENTS}
    counts: dict[str, int] = {}
    frozen: dict[str, int] = {}

    named: list[tuple[str, torch.nn.Parameter]] = [
        (classify(n), p) for n, p in transformer.named_parameters() if p.requires_grad
    ]
    for module, component in ((text_conditioner, "llm_adapter"), (text_encoder, "text_encoder")):
        if module is not None:
            named += [(component, p) for _, p in module.named_parameters() if p.requires_grad]

    for component, param in named:
        if lrs.resolve(component, default_lr) == 0.0:
            param.requires_grad_(False)
            frozen[component] = frozen.get(component, 0) + param.numel()
            continue
        buckets[component].append(param)
        counts[component] = counts.get(component, 0) + param.numel()

    groups = [
        {
            "params": params,
            "lr": lrs.resolve(component, default_lr),
            "weight_decay": weight_decay,
            "component": component,
        }
        for component, params in buckets.items()
        if params
    ]
    if not groups:
        raise ValueError(
            "every adapter component is frozen by component_lr; nothing to train"
        )
    return ParamGroupReport(groups=groups, counts=counts, frozen=frozen)


def lora_target_modules(components: list[str]) -> list[str]:
    """Suffix patterns PEFT matches against module names, for the listed components."""
    unknown = [c for c in components if c not in _LORA_TARGETS]
    if unknown:
        raise ValueError(f"no LoRA targets defined for {unknown}; valid: {sorted(_LORA_TARGETS)}")
    targets = [t for c in components for t in _LORA_TARGETS[c]]
    if not targets:
        raise ValueError(f"components {components} contain no LoRA-injectable Linear layers")
    return targets


@dataclass
class AdapterConfig:
    """LoRA / LoKr settings. LoKr factorises the update as a Kronecker product, so it reaches a
    given expressivity with far fewer parameters than LoRA at the same rank -- worth it when
    optimizer state is the binding constraint, which on 24GB it usually is."""

    kind: str = "lora"                    # "lora" | "lokr" | "none"
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    components: list[str] = field(default_factory=lambda: ["self_attn", "cross_attn", "mlp"])
    train_llm_adapter: bool = False       # inject into the text conditioner too
    # LoRA on Qwen3 itself -- sd-scripts' `lora_te`. Off by default and deliberately so: diffusers'
    # AnimaLoraLoaderMixin only knows `transformer` and `text_conditioner`, so a text-encoder LoRA
    # cannot round-trip through diffusers and must be exported in kohya format.
    train_text_encoder: bool = False
    text_encoder_rank: int | None = None   # defaults to `rank`
    # LoKr only.
    lokr_factor: int = -1                 # -1 = pick the most balanced factorisation
    lokr_decompose_both: bool = False

    def __post_init__(self):
        if self.kind not in ("lora", "lokr", "none"):
            raise ValueError(f"unknown adapter kind: {self.kind}")
        if self.train_text_encoder and self.kind == "lokr":
            raise ValueError("text-encoder adapter supports kind='lora' only")

    def build(self):
        """-> a peft config. Imported lazily so full-finetune runs need no peft import."""
        from peft import LoKrConfig, LoraConfig

        targets = lora_target_modules(self.components)
        if self.kind == "lora":
            return LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=targets,
                init_lora_weights="gaussian",
            )
        return LoKrConfig(
            r=self.rank,
            alpha=self.alpha,
            rank_dropout=self.dropout,
            target_modules=targets,
            decompose_factor=self.lokr_factor,
            decompose_both=self.lokr_decompose_both,
        )


def apply_adapter(
    transformer: nn.Module,
    text_conditioner: nn.Module | None,
    cfg: AdapterConfig,
    text_encoder: nn.Module | None = None,
) -> tuple[nn.Module, nn.Module | None]:
    """Wrap the modules with PEFT adapters and freeze everything else.

    The base weights are frozen *before* injection so that the only trainable tensors are the
    adapter's own -- which is the whole point, and also what lets the base be quantized later
    without touching the optimizer.
    """
    if cfg.kind == "none":
        return transformer, text_conditioner

    peft_config = cfg.build()
    transformer.requires_grad_(False)
    transformer.add_adapter(peft_config)

    if text_conditioner is not None:
        text_conditioner.requires_grad_(False)
        if cfg.train_llm_adapter:
            # The conditioner's Linears are named differently from the trunk's; target them by
            # their own names rather than reusing the trunk's target list.
            from peft import LoraConfig

            text_conditioner.add_adapter(
                LoraConfig(
                    r=cfg.rank,
                    lora_alpha=cfg.alpha,
                    lora_dropout=cfg.dropout,
                    target_modules=["to_q", "to_k", "to_v", "to_out.0", "linear_1", "linear_2"],
                    init_lora_weights="gaussian",
                )
            )

    if cfg.train_text_encoder:
        if text_encoder is None:
            raise ValueError("adapter.train_text_encoder=true but the text encoder was not loaded")
        from peft import LoraConfig

        text_encoder.requires_grad_(False)
        text_encoder.add_adapter(
            LoraConfig(
                r=cfg.text_encoder_rank or cfg.rank,
                lora_alpha=cfg.alpha,
                lora_dropout=cfg.dropout,
                target_modules=list(_TEXT_ENCODER_TARGETS),
                init_lora_weights="gaussian",
            )
        )
        text_encoder.train()

    return transformer, text_conditioner


def trainable_parameters(*modules: nn.Module | None) -> list[torch.nn.Parameter]:
    return [p for m in modules if m is not None for p in m.parameters() if p.requires_grad]


def count_parameters(*modules: nn.Module | None) -> tuple[int, int]:
    """-> (trainable, total)."""
    trainable = total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
    return trainable, total
