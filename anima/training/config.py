"""TOML training config.

Unknown keys are a hard error. A typo'd `learing_rate` that silently keeps the default would cost
a full training run to notice, and this trainer exists specifically for runs that are expensive.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from ..data.caption import CaptionConfig, load_protected_tags
from ..data.dataset import DatasetConfig, SubsetConfig
from ..data.texture import TextureConfig
from .curriculum import Curriculum, Phase
from .flow import FlowConfig
from .params import AdapterConfig, ComponentLRs
from .quant import QuantConfig


@dataclass
class OptimizerConfig:
    kind: str = "adamw"                 # "adamw" | "adamw8bit" | "adafactor" | "came" | "lion"
    lr: float = 1e-5
    # Two for adamw/lion; CAME takes THREE (it keeps a third moment for its instability factor).
    betas: tuple[float, ...] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # sdnq.optim: quantize and/or offload optimizer state. For a 2B full finetune this is not
    # optional -- fp32 AdamW state alone is 16GB.
    quantize_state: bool = False
    offload_state: bool = False
    # Kahan summation over the bf16 master weight. sdnq keeps a per-parameter residual buffer and
    # folds the part of each update that the bf16 cast threw away back into the next one, so
    # updates below an ulp accumulate instead of vanishing. This is *not* an alternative to
    # stochastic rounding -- sdnq applies both (`optim/utils.py:57`), SR on the cast and Kahan on
    # what the cast lost. Costs one extra buffer per trainable parameter.
    use_kahan: bool = False

    def __post_init__(self):
        self.betas = tuple(self.betas)
        # CAME keeps a third moment (its instability factor), so it unpacks `betas` into three.
        # Left unchecked, the default two-tuple sails through config loading and dies inside
        # sdnq's `came_update` at the first optimizer step -- after model load, dataset scan and
        # the whole report, with a bare "not enough values to unpack (expected 3, got 2)" that
        # names neither the config key nor the optimizer.
        want = 3 if self.kind == "came" else 2
        if len(self.betas) != want:
            raise ValueError(
                f"optimizer.kind = {self.kind!r} takes {want} betas, got {len(self.betas)}: "
                f"{list(self.betas)}."
                + ("\nCAME's defaults are betas = [0.9, 0.999, 0.9999]." if want == 3 else "")
            )
        # torch's AdamW keeps fp32 master weights and has no residual buffer to offer, so the flag
        # would be accepted and dropped -- the dead-key failure mode this codebase keeps finding.
        if self.use_kahan and self.kind.lower() == "adamw":
            raise ValueError(
                "optimizer.use_kahan needs an sdnq optimizer (it is sdnq's bf16 master-weight "
                "residual buffer). Use kind = 'adamw8bit' for the sdnq equivalent, or drop the key "
                "-- torch's 'adamw' already keeps fp32 masters and has nothing to correct."
            )


@dataclass
class ScheduleConfig:
    kind: str = "constant"              # "constant" | "cosine" | "linear" | "rex" | "rerex"
    warmup_steps: int = 0
    # Floor for every decaying schedule, as a fraction of the group's peak LR. sd-scripts hardcodes
    # 0.001 for REX/ReREX; leaving this at 0.0 makes them decay all the way to zero instead.
    min_lr_ratio: float = 0.0

    # --- rex ---
    d: float = 0.9                      # decay sharpness; 0 == linear, higher holds the peak longer

    # --- rerex ---
    global_d: float = 0.78              # `d` of the outer curve the segment endpoints ride
    local_d: float = 0.85               # `d` of the decay inside each segment
    weight_power: float = 1.5           # step-budget skew toward early segments; 0 == equal lengths
    num_segments: int = 8

    def __post_init__(self):
        if self.kind not in ("constant", "cosine", "linear", "rex", "rerex"):
            raise ValueError(f"unknown schedule.kind: {self.kind!r}")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError(f"schedule.min_lr_ratio must be in [0, 1), got {self.min_lr_ratio}")
        # d == 1 makes the REX denominator collapse to the numerator: a flat curve that falls off a
        # cliff at the very last step, and a 0/0 there. Rejected rather than clamped.
        for name in ("d", "global_d", "local_d"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"schedule.{name} must be in [0, 1), got {v}")
        if self.num_segments < 1:
            raise ValueError(f"schedule.num_segments must be >= 1, got {self.num_segments}")
        if self.weight_power < 0.0:
            raise ValueError(f"schedule.weight_power must be >= 0, got {self.weight_power}")


# The converted diffusers repo (see README section 0). `ANIMA_MODEL` overrides it, so a machine
# that keeps the model somewhere fixed does not have to set `model_path` in every config; the
# relative fallback resolves for the common layout where the model sits beside this checkout.
DEFAULT_MODEL_PATH = os.environ.get("ANIMA_MODEL", "../anima-diffusers")


@dataclass
class TrainConfig:
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = "output"
    run_name: str = "anima"

    epochs: int = 1
    max_steps: int | None = None
    # int, or {longest_side_threshold: micro_batch_size} e.g. {512 = 8, 1024 = 4, 1536 = 1}.
    batch_size: int | dict[int, int] = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    seed: int = 42
    num_workers: int = 2
    # Images per VAE forward under `dataset.source = "encode"`. The encode runs under no_grad and
    # is freed before the transformer allocates, so the step peak is max(train, encode) -- this
    # knob decides which of the two wins. Raising it buys nothing: measured at 1024px on 413
    # images, mean step time is 6.52 / 6.52 / 6.53 s/it at chunk 1 / 2 / 6 while peak goes
    # 18.2 / 19.3 / 23.1 GB, and chunk 12 OOMs outright. The VAE encode is not launch-bound, so
    # there is no throughput to recover by batching it. Leave this at 1.
    vae_encode_chunk: int = 1

    save_every_steps: int | None = None
    save_every_epochs: int | None = 1
    save_native: bool = True            # also write a ComfyUI-loadable single file
    keep_last_n: int | None = None
    resume_from: str | None = None
    # Optimizer + scheduler state alongside each checkpoint. Required to resume; costs roughly the
    # optimizer-state size on disk per checkpoint, which for a quantized full FT is ~3.5GB.
    save_optimizer_state: bool = False
    skip_final_save: bool = False       # benchmarking only
    # Texture curricula are refused on more than one process. Two runs degraded anatomy
    # progressively against single-GPU at MATCHED steps -- so it is not the halved step count, and
    # more steps do not fix it. The cause is not identified: ranks seeding identically was found
    # and fixed (`set_seed(device_specific=True)`), but was never shown to be the mechanism.
    #
    # This exists so re-testing that fix is a deliberate act rather than the default. It is not a
    # "I know better" switch -- if a run under it comes out clean against a single-GPU control at
    # matched steps, the gate should be removed, not the override left on.
    allow_multi_gpu_texture: bool = False

    log_every: int = 1
    # Progress display. "auto" draws a tqdm bar when stdout is a terminal and falls back to the
    # plain per-step line when it is not -- a bar redirected to a file is thousands of \r-separated
    # fragments. "bar"/"plain" force it either way; "off" suppresses the per-step line entirely.
    progress: str = "auto"              # "auto" | "bar" | "plain" | "off"

    # torch.compile, via Accelerate's TorchDynamoPlugin so the compile/DDP ordering is handled.
    # None = off. Measure before enabling: bucketing means many input shapes, and every distinct
    # shape is a potential recompile.
    compile: str | None = None          # "default" | "reduce-overhead" | "max-autotune"
    # Dynamic shapes. Effectively mandatory here -- with static shapes each bucket compiles its
    # own graph and dynamo's recompile limit (8) is exhausted almost immediately, after which it
    # silently falls back to eager for the rest of the run.
    compile_dynamic: bool = True
    # Compile the repeated transformer block once instead of the whole trunk. 28 identical blocks,
    # so this cuts compile time by roughly that factor for nearly all of the benefit.
    compile_regional: bool = True

    def __post_init__(self):
        if isinstance(self.batch_size, dict):
            self.batch_size = {int(k): int(v) for k, v in self.batch_size.items()}
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unknown dtype: {self.dtype}")
        if self.vae_encode_chunk < 1:
            raise ValueError(
                f"train.vae_encode_chunk must be >= 1, got {self.vae_encode_chunk}")
        if self.progress not in ("auto", "bar", "plain", "off"):
            raise ValueError(
                f"train.progress must be auto|bar|plain|off, got {self.progress!r}")
        if self.compile is not None and self.compile not in (
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ):
            raise ValueError(
                f"unknown train.compile mode: {self.compile!r}. Use one of 'default', "
                f"'reduce-overhead', 'max-autotune', or omit it to disable."
            )


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(path=""))
    flow: FlowConfig = field(default_factory=FlowConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    component_lr: ComponentLRs = field(default_factory=ComponentLRs)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    # `[[curriculum]]` -- an array of tables, so it is built by hand in `load_config` rather than
    # through `_SECTIONS`. Empty by default; an empty curriculum is exactly today's behaviour.
    curriculum: Curriculum = field(default_factory=Curriculum)

    @property
    def is_lora(self) -> bool:
        return self.adapter.kind != "none"


# `from __future__ import annotations` makes dataclass field types plain strings, so the section
# classes are resolved from this table rather than from field introspection.
_SECTIONS = {
    "train": TrainConfig,
    "dataset": DatasetConfig,
    "flow": FlowConfig,
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "component_lr": ComponentLRs,
    "adapter": AdapterConfig,
    "quant": QuantConfig,
}


def _build(cls, data: dict, path: str):
    """Instantiate a flat dataclass from a dict, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"[{path}] unknown key(s): {unknown}. Valid: {sorted(known)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))

    unknown = sorted(set(raw) - {f.name for f in fields(Config)})
    if unknown:
        raise ValueError(f"unknown top-level section(s): {unknown}")

    sections = {}
    for f in fields(Config):
        if f.name not in raw:
            continue
        if f.name == "curriculum":
            phases = raw["curriculum"]
            if not isinstance(phases, list):
                raise ValueError(
                    "[curriculum] must be an array of tables -- write `[[curriculum]]` (double "
                    "brackets), one block per phase."
                )
            sections[f.name] = Curriculum([_build(Phase, dict(p), "curriculum") for p in phases])
            continue
        data = dict(raw[f.name])
        if f.name == "dataset":
            # Both keys set is always a mistake: `resolutions` wins, so a leftover
            # `resolution = 1024` above `resolutions = [768, 1280]` would be silently ignored.
            # Checked here rather than in the dataclass because only the raw TOML distinguishes
            # "explicitly set" from "left at the default".
            if "resolution" in data and "resolutions" in data:
                raise ValueError(
                    "[dataset] sets both `resolution` and `resolutions`. Use `resolution` for a "
                    "single tier or `resolutions` for several, not both -- `resolution` would be "
                    "ignored."
                )
            # `caption` and `texture` are nested inside `[dataset]` in TOML but are their own
            # dataclasses. Without this they arrive as plain dicts, which fails nowhere at load
            # time and then raises deep inside the first texture batch.
            caption = data.pop("caption", {})
            texture = data.pop("texture", {})
            # `[[dataset.subsets]]` arrives as a list of dicts and needs the same treatment: built
            # per element so an unknown key is named against `dataset.subsets` rather than being
            # swallowed. Passed through the constructor, not assigned after, so `DatasetConfig`'s
            # path/subsets exclusivity check sees both at once.
            if "subsets" in data:
                data["subsets"] = [_build(SubsetConfig, dict(s), "dataset.subsets")
                                   for s in data["subsets"]]
            protected_file = caption.pop("protected_tags_file", None)
            cfg = _build(DatasetConfig, data, "dataset")
            cfg.caption = _build(CaptionConfig, caption, "dataset.caption")
            cfg.texture = _build(TextureConfig, texture, "dataset.texture")
            if protected_file:
                cfg.caption.protected_tags = load_protected_tags(protected_file)
            sections[f.name] = cfg
        else:
            sections[f.name] = _build(_SECTIONS[f.name], data, f.name)

    cfg = Config(**sections)
    if not cfg.dataset.path and not cfg.dataset.subsets:
        raise ValueError("dataset.path (or a [[dataset.subsets]] list) is required")

    # A per-tier `batch_size` map must name exactly the tiers that exist. Checked here because it
    # is the one rule that spans two sections, and because getting it wrong is expensive: the old
    # threshold semantics let `{512 = 32, 1024 = 12}` hand a 960x960 bucket 32 images and OOM
    # several minutes into a run. One second at load beats that.
    if isinstance(cfg.train.batch_size, dict):
        tiers = set(cfg.dataset.tiers)
        keys = set(cfg.train.batch_size)
        declared = "resolutions" if cfg.dataset.resolutions else "resolution"
        if unknown := sorted(keys - tiers):
            raise ValueError(
                f"train.batch_size has key(s) {unknown} that are not declared resolutions. "
                f"dataset.{declared} declares {sorted(tiers)}; the map is keyed by tier, matched "
                f"exactly -- it is not a threshold ladder. Use `batch_size = <int>` for one size "
                f"everywhere."
            )
        if missing := sorted(tiers - keys):
            raise ValueError(
                f"train.batch_size gives no size for tier(s) {missing}. dataset.{declared} "
                f"declares {sorted(tiers)}, so every one of them needs an entry (or use a plain "
                f"int for the same size everywhere)."
            )
    # Texture crops are chosen per step from image content, so no cache can hold them. Checked at
    # load rather than at the first texture batch -- which could be several hundred steps in, and
    # which the GUI's status bar would never reach, since it validates the config without running.
    if any(p.mode == "texture" for p in cfg.curriculum.phases) and cfg.dataset.source != "encode":
        raise ValueError(
            f"a curriculum phase uses mode = 'texture', which crops per step from the source "
            f"image, but dataset.source is {cfg.dataset.source!r}. Set "
            f'dataset.source = "encode" -- cached latents are a frozen centre crop and cannot '
            f"express a per-step crop."
        )

    if cfg.is_lora:
        # component_lr now applies to adapters too, so the check is no longer "is it allowed" but
        # "would it do anything". The previous guard only looked at adaln/base, which meant a
        # `component_lr.mlp` under LoRA was accepted and then silently ignored.
        adapter_components = set(cfg.adapter.components)
        if cfg.adapter.train_llm_adapter:
            adapter_components.add("llm_adapter")
        if cfg.adapter.train_text_encoder:
            adapter_components.add("text_encoder")

        noop = {
            c: lr for c, lr in cfg.component_lr.explicit().items()
            if lr not in (None, 0.0) and c not in adapter_components
        }
        if noop:
            raise ValueError(
                f"[component_lr] sets {noop} but no adapter is injected there, so it would have "
                f"no effect. Add the component to adapter.components "
                f"(currently {sorted(adapter_components)}), or remove the LR."
            )
    if cfg.quant.mode == "frozen" and not cfg.is_lora:
        # A frozen-quantized base has no trainable weights at all, so this would run and produce
        # nothing. Caught here rather than after the model loads.
        raise ValueError(
            "quant.mode='frozen' quantizes the base and trains nothing; it requires an adapter. "
            "For a quantized full finetune use quant.mode='training'."
        )
    if cfg.quant.mode == "training" and cfg.is_lora:
        raise ValueError(
            "quant.mode='training' is for full finetuning; with an adapter use 'frozen'"
        )
    return cfg
