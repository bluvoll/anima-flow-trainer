"""TOML <-> GUI bridge.

Ours, not Aozora's. Two decisions carry the whole module.

**Widgets are keyed by dotted TOML path** (`flow.hf_scale`, `dataset.caption.tag_dropout_percent`)
rather than by an invented flat name space. There is then no rename table to drift: a key either
exists in the dataclasses or it does not, and `missing_ui_keys()` turns that into an assertion the
gate can run.

**Only non-default values are written.** Defaults come from the dataclasses themselves, so a
GUI-written TOML reads like a hand-written one -- and, more importantly, a field whose default
changes later is not silently pinned to the old value by every config the GUI ever saved.

Everything is validated by `anima.training.config.load_config`, the same function the trainer
calls. The GUI does not reimplement a single rule; it shows you the error the trainer would raise.
"""

from __future__ import annotations

import dataclasses as dc
import tempfile
from pathlib import Path

import toml

from ..data.caption import CaptionConfig
from ..data.texture import TextureConfig
from ..data.dataset import DatasetConfig
from ..training.config import (
    AdapterConfig,
    ComponentLRs,
    FlowConfig,
    OptimizerConfig,
    QuantConfig,
    ScheduleConfig,
    TrainConfig,
    load_config,
)

# TOML section -> dataclass. `dataset.caption` is nested one level deeper in the file than it is in
# the dataclass tree, which is the one place the two shapes disagree.
SECTIONS: dict[str, type] = {
    "train": TrainConfig,
    "dataset": DatasetConfig,
    "dataset.caption": CaptionConfig,
    "dataset.texture": TextureConfig,
    "flow": FlowConfig,
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "adapter": AdapterConfig,
    "quant": QuantConfig,
    "component_lr": ComponentLRs,
}

# Fields that exist on a dataclass but are not config keys in their own section -- they are
# nested tables with their own entry in SECTIONS, so listing them here too would demand a widget
# for the container itself.
_SKIP = {("dataset", "caption"), ("dataset", "texture")}

# `[[curriculum]]` is an array of tables, not a section of scalars, so it has no dataclass field to
# key a widget off. It is still a real config key with a real widget, so it is carried alongside
# the schema rather than special-cased at each call site. The value is a list of phase dicts; the
# empty list is the default and means "no curriculum", which is exactly today's behaviour.
ARRAY_KEYS: dict[str, list] = {"curriculum": []}


def schema() -> dict[str, dc.Field]:
    """Every settable dotted key -> its dataclass field."""
    out: dict[str, dc.Field] = {}
    for section, klass in SECTIONS.items():
        for f in dc.fields(klass):
            if (section, f.name) in _SKIP:
                continue
            out[f"{section}.{f.name}"] = f
    return out


def default_of(f: dc.Field):
    if f.default is not dc.MISSING:
        return f.default
    if f.default_factory is not dc.MISSING:       # type: ignore[misc]
        return f.default_factory()                # type: ignore[misc]
    return None


def defaults() -> dict:
    return {**{k: default_of(f) for k, f in schema().items()},
            **{k: list(v) for k, v in ARRAY_KEYS.items()}}


# ---------------------------------------------------------------- nested <-> flat


def flatten(nested: dict) -> dict:
    """Nested TOML dict -> {dotted key: value}, for the sections we know about."""
    flat = {}
    for section in SECTIONS:
        node = nested
        for part in section.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if (section, key) in _SKIP:
                continue
            dotted = f"{section}.{key}"
            # TOML bare keys are strings, so `batch_size = { 512 = 4 }` reads back with string
            # keys while the widget and `TrainConfig.__post_init__` both use ints. Canonicalise
            # here so the flat dict has one key type and comparisons mean something.
            if dotted == "train.batch_size" and isinstance(value, dict):
                value = {int(k): int(v) for k, v in value.items()}
            flat[dotted] = value
    for key in ARRAY_KEYS:
        value = nested.get(key)
        if isinstance(value, list):
            flat[key] = [dict(entry) for entry in value if isinstance(entry, dict)]
    return flat


def nest(flat: dict) -> dict:
    """{dotted key: value} -> nested TOML dict. Keys mapped to None are dropped, which is how the
    GUI spells "leave this unset" for the `int | None` and `float | None` fields."""
    out: dict = {}
    for key, value in flat.items():
        if value is None:
            continue
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def prune_defaults(flat: dict) -> dict:
    """Drop every value equal to its dataclass default. `dataset.path` and `train.run_name` are
    kept regardless -- a config without them is not obviously a config."""
    # `adapter.kind` is always written even at its default: it is the mode selector -- full
    # finetune vs LoRA vs LoKr -- and a config whose most consequential key is implicit is a config
    # someone will misread.
    keep = {"dataset.path", "train.run_name", "train.model_path", "train.output_dir",
            "adapter.kind"}
    base = defaults()
    out = {}
    for key, value in flat.items():
        if value is None:
            continue
        if key in keep or key not in base:
            out[key] = value
            continue
        d = base[key]
        # tuple/list mismatch: `betas` defaults to a tuple and round-trips through TOML as a list.
        if isinstance(d, tuple) and isinstance(value, list):
            d = list(d)
        if value != d:
            out[key] = value
    # `load_config` rejects a file that sets both. The GUI writes the file, so the right behaviour
    # is to emit the one that is in effect rather than to hand the user an error about a key they
    # cannot see. A multi-tier ladder wins; `resolution` is what "one tier" means.
    if out.get("dataset.resolutions"):
        out.pop("dataset.resolution", None)
    else:
        out.pop("dataset.resolutions", None)
    # Same rule, same reason: `path` and `subsets` are mutually exclusive in the loader, and the
    # GUI keeps a value in the path box even after subsets are added. Emit the one in effect.
    if out.get("dataset.subsets"):
        out.pop("dataset.path", None)
    else:
        out.pop("dataset.subsets", None)
    return out


# ---------------------------------------------------------------- file I/O


def read_toml(path: str | Path) -> dict:
    return toml.loads(Path(path).read_text(encoding="utf-8"))


class _Inline(dict, toml.decoder.InlineTableDict):
    """Emit `batch_size = { 512 = 4, 1024 = 12 }` instead of a `[train.batch_size]` sub-table.

    Both parse identically, but the sub-table form has to be written *after* every scalar in the
    file and reads as a separate section, which is exactly wrong for what is one value.
    """



def _mark_inline(nested: dict) -> dict:
    """Leaf dicts (batch_size, mixed_weights) become inline tables; section dicts do not."""
    out = {}
    for key, value in nested.items():
        if isinstance(value, dict) and value and not any(
            isinstance(v, dict) for v in value.values()
        ) and key not in SECTIONS and key != "caption":
            out[key] = _Inline({str(k): v for k, v in value.items()})
        elif isinstance(value, dict):
            out[key] = _mark_inline(value)
        else:
            out[key] = value
    return out


def dump_toml(flat: dict) -> str:
    nested = _mark_inline(nest(prune_defaults(flat)))
    return toml.dumps(nested, encoder=toml.encoder.TomlPreserveInlineDictEncoder())


def write_toml(path: str | Path, flat: dict) -> None:
    Path(path).write_text(dump_toml(flat), encoding="utf-8")


def validate(flat: dict) -> tuple[bool, str]:
    """Run the trainer's own loader over a candidate config.

    Written to a temp file rather than constructing `Config` directly on purpose: several rules --
    `resolution` vs `resolutions` above all -- are enforced against the *raw TOML*, because only
    the file distinguishes "explicitly set" from "left at the default". Validating the object would
    silently skip them.
    """
    text = dump_toml(flat)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        load_config(tmp)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        Path(tmp).unlink(missing_ok=True)


def summarize(flat: dict) -> str:
    """One-line description of what this config will do, for the status bar."""
    cfg_ok, _ = validate(flat)
    if not cfg_ok:
        return "config invalid"
    kind = flat.get("adapter.kind", "none")
    mode = "full finetune" if kind in (None, "none", "") else f"LoRA/{kind}"
    tiers = flat.get("dataset.resolutions") or [flat.get("dataset.resolution", 1024)]
    quant = flat.get("quant.mode", "none")
    bits = [mode, f"{'/'.join(str(t) for t in tiers)}px"]

    # Whether a run crops is decided by the curriculum, which lives in a table on another tab --
    # so without this the status bar reads identically for a plain LoRA and a texture run.
    phases = flat.get("curriculum") or []
    if phases:
        n_tex = sum(1 for p in phases if p.get("mode") == "texture")
        bits.append(f"curriculum {len(phases)} phase(s)"
                    + (f", {n_tex} texture" if n_tex else ", no texture"))
        if not n_tex and flat.get("dataset.source") == "encode":
            bits.append("(source=encode with no texture phase: +16% step time for nothing)")
    elif flat.get("dataset.source") == "encode":
        bits.append("(source=encode with no curriculum: +16% step time for nothing)")

    if quant not in (None, "none"):
        bits.append(f"sdnq {quant}/{flat.get('quant.weights_dtype', 'int8')}")
    if flat.get("train.compile"):
        bits.append(f"compile {flat['train.compile']}")
    return "  ".join(bits)


# ---------------------------------------------------------------- advisories


def _texture_phases(flat: dict) -> list[dict]:
    return [p for p in (flat.get("curriculum") or []) if p.get("mode") == "texture"]


def _restricted_share(flat: dict) -> tuple[float, float]:
    """(share of the run in a restricted t_range, run-average t^2 factor for the hf term).

    Under `rescale` -- the default -- a phase maps the full distribution onto [lo, hi], so
    t' = lo + (hi-lo)*t and with lo = 0 the hf term's t^2 factor is exactly (hi-lo)^2. That closed
    form is used rather than sampling, because this runs on every keystroke. Checked against a
    200k-sample draw: predicted 0.36x for [0, 0.6], measured 0.36x.
    """
    phases = flat.get("curriculum") or []
    if not phases:
        return 0.0, 1.0
    share, factor = 0.0, 0.0
    for i, p in enumerate(phases):
        end = phases[i + 1].get("at", 1.0) if i + 1 < len(phases) else 1.0
        span = max(0.0, end - p.get("at", 0.0))
        lo, hi = p.get("t_range", (0.0, 1.0))
        width = hi - lo
        # lo^2 + 2*lo*(hi-lo)*E[t] + (hi-lo)^2*E[t^2], with the run's own E[t]/E[t^2].
        # E[t]=0.676 and E[t^2]=0.528 for uniform+shift 3, this trainer's measured default.
        f = lo * lo + 2 * lo * width * 0.676 + width * width * 0.528
        factor += span * (f / 0.528)
        if width < 1.0 or lo > 0.0:
            share += span
    return share, factor


def advisories(flat: dict, num_processes: int = 1) -> list[tuple[str, str]]:
    """(level, message) for combinations that load fine and then train badly.

    Distinct from `validate`, which surfaces what the trainer would REJECT. Everything here is
    legal and runnable; it is just known -- from a measurement, not a hunch -- to be a bad idea, or
    to mean something different from what it looks like. Each message carries the number that
    justifies it, because an unexplained warning is one people learn to click past.

    `num_processes` is a GUI-only input: the config cannot express how many GPUs are ticked, and
    the two worst traps here are both about that.
    """
    out: list[tuple[str, str]] = []
    tex = _texture_phases(flat)
    n = max(1, int(num_processes))

    if n > 1:
        if tex and not flat.get("train.allow_multi_gpu_texture"):
            out.append(("error",
                        (f"{n} GPUs with a texture curriculum is REFUSED. Two runs here degraded "
                        f"anatomy progressively against single-GPU at matched steps -- so it is "
                        f"not undertraining, and more steps do not fix it. The cause is not "
                        f"identified; ranks seeding identically was found and fixed, but is not "
                        f"proven to be it. Use one GPU, or set train.allow_multi_gpu_texture to "
                        f"re-test the fix deliberately.")))
        elif tex:
            out.append(("warn",
                        (f"{n} GPUs with a texture curriculum, override ON. This configuration "
                        f"produced progressively worsening anatomy twice. You are re-testing an "
                        f"unproven fix -- compare against a single-GPU run at matched steps.")))
        out.append(("note",
                    (f"{n} GPUs means {n}x FEWER optimizer steps than the same config on one, at "
                    f"{n}x the effective batch. Multiply epochs by {n} to match a single-GPU run.")))

    if tex:
        bs = flat.get("train.batch_size", 1)
        if isinstance(bs, dict):
            bs = max(bs.values()) if bs else 1
        if bs and bs > 1:
            out.append(("note",
                        (f"batch_size {bs} in texture mode buys no throughput: measured 0.89-0.93 "
                        f"images/s flat from batch 1 to 8, while peak VRAM went 8.6 -> 14.6GB. A "
                        f"1024px canvas already saturates the GPU. It also shrinks the set of "
                        f"canvases every image in the batch can hold, so padding returns.")))
        tiers = flat.get("dataset.resolutions") or []
        if len(tiers) > 1:
            out.append(("note",
                        (f"{len(tiers)} resolution tiers with a texture curriculum: tiers set the "
                        f"FULLRES bucket, but a texture crop overrides shape with the canvas. So "
                        f"this costs {len(tiers)}x the epoch length and only benefits the fullres "
                        f"phases. Vary `texture.canvases` instead -- fit-aware selection makes a "
                        f"mixed-scale list adaptive per image.")))
        if flat.get("dataset.texture.oversize") == "cover":
            out.append(("warn",
                        ("oversize='cover' upscales without a ceiling. It lost its A/B here: it "
                        "fired on 57% of the texture set draws at a mean 1.25x (max 1.85x) and softened "
                        "faces. `cover_max_scale` is the capped version and is already on.")))

    hf = flat.get("flow.hf_scale") or 0.0
    if hf > 0.0:
        share, factor = _restricted_share(flat)
        if flat.get("flow.phase_mapping", "rescale") == "rescale" and factor < 0.9:
            out.append(("note",
                        (f"hf loss carries a t^2 factor and {share:.0%} of this run sits in a "
                        f"restricted t_range, so the term lands at {factor:.2f}x the strength the "
                        f"same hf_scale gives on a full-range run. hf_scale "
                        f"{hf / max(factor, 0.01):.2f} would match.")))
        if tex and flat.get("dataset.texture.pad_mode") == "black" \
                and not flat.get("dataset.texture.fit_aware", True):
            out.append(("warn",
                        ("hf loss with black padding and fit-aware selection off: the image/black "
                        "boundary is a maximal Laplacian edge, so detail weighting concentrates on "
                        "exactly the artifact you do not want learned. Padding fires on 56% of "
                        "draws in this configuration.")))

    if flat.get("flow.use_ot"):
        bs = flat.get("train.batch_size", 1)
        if (max(bs.values()) if isinstance(bs, dict) and bs else bs) == 1:
            out.append(("note",
                        "use_ot is a no-op at batch size 1 -- there is nothing to permute."))

    return out


def missing_ui_keys(ui_keys) -> tuple[list[str], list[str]]:
    """(config keys with no widget, widget keys that are not config keys).

    The gate asserts both are empty, which is what stops a new dataclass field from being
    unreachable from the GUI -- the exact failure mode that made `keep_last_n` and `torch.compile`
    dead keys in the TOML path.
    """
    known = set(schema()) | set(ARRAY_KEYS)
    ui = set(ui_keys)
    return sorted(known - ui), sorted(ui - known)
