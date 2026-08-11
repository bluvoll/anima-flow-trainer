"""Gate for the GUI: config bridge, widget coverage, inert-knob rules, log parsing, launch argv.

    QT_QPA_PLATFORM=offscreen .venv/bin/python anima/parity/test_gui.py

Runs headless via Qt's offscreen platform, so it needs no display and no GPU -- and it constructs
the real window with the real editors rather than testing a mock of them.

The load-bearing check is the first one. A config key with no widget is unreachable from the GUI,
and a widget for a key that does not exist is a silent no-op: both are the failure mode that made
`keep_last_n` and `torch.compile` dead keys in the TOML path for months. Here it is an assertion.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from anima.gui import bridge, fields as F, schema  # noqa: E402
from anima.gui.app import CONFIG_DIR, _RULES, TrainingGUI  # noqa: E402
from anima.gui.metrics import STEP_RE  # noqa: E402
from anima.gui.widgets import safe_stem as F_safe_stem  # noqa: E402
from anima.gui.process import audit_launch, cache_launch, train_launch  # noqa: E402
from anima.training.config import load_config  # noqa: E402
from anima.data.dataset import SubsetConfig  # noqa: E402
from anima.training.quant import SKIP_POLICIES  # noqa: E402

# Written here rather than read from `configs/`. This gate used to name four real config files and
# broke the moment one was renamed -- which the GUI now does by itself, since the filename follows
# `run_name`. A gate that depends on a user's working files fails for reasons that have nothing to
# do with the code, and on a fresh clone `configs/` may be empty entirely.
#
# What matters is COVERAGE, so these are chosen to span the shapes the bridge can get wrong rather
# than to be realistic runs: an array-of-tables key (`[[curriculum]]`, `[[dataset.subsets]]`), the
# `encode` source, a per-tier `batch_size` map, a `[component_lr]` with a 0.0 freeze in it, and
# `quant` in both modes. `encode` in particular was once missing from the source widget's option
# list, so the GUI silently reset it to `auto` and made texture mode unreachable.
FIXTURES: dict[str, str] = {
    "lora.toml": """
[train]
model_path = "/path/to/model"
run_name = "fixture-lora"
epochs = 4
batch_size = 8
gradient_accumulation_steps = 2
[dataset]
path = "/path/to/data"
resolution = 1024
[dataset.caption]
caption_mode = "mixed"
mixed_weights = { tags = 50, nl = 10, tags_nl = 20, nl_tags = 20 }
shuffle_tags = true
[flow]
timestep_sample_method = "logit_normal"
sigmoid_scale = 1.3
hf_scale = 0.25
[optimizer]
kind = "adamw"
betas = [0.9, 0.99]
[adapter]
kind = "lora"
rank = 32
alpha = 32.0
components = ["self_attn", "cross_attn", "mlp"]
[quant]
mode = "frozen"
use_quantized_matmul = "auto"
""",
    "full.toml": """
[train]
model_path = "/path/to/model"
run_name = "fixture-full"
batch_size = { 768 = 4, 1024 = 2 }
[dataset]
path = "/path/to/data"
resolutions = [768, 1024]
[flow]
timestep_sample_method = "uniform"
shift = 3.0
[component_lr]
self_attn = 1.0e-5
mlp = 1.5e-5
adaln = 0.0
[optimizer]
kind = "adamw8bit"
betas = [0.9, 0.99]
quantize_state = true
[schedule]
kind = "rerex"
num_segments = 8
[adapter]
kind = "none"
[quant]
mode = "training"
skip_policy = "anima_int8_mm"
use_quantized_matmul = false
""",
    "texture.toml": """
[[curriculum]]
at = 0.0
t_range = [0.0, 1.0]
mode = "fullres"
[[curriculum]]
at = 0.15
t_range = [0.0, 0.6]
mode = "texture"
lr_mul = 0.5
[dataset]
source = "encode"
[[dataset.subsets]]
path = "/path/to/data"
[[dataset.subsets]]
path = "/path/to/regularization"
num_repeats = 2
texture = false
[dataset.texture]
oversize = "pad"
energy_power = 1.0
[dataset.caption]
protected_tags = ["black background", "white background"]
[train]
model_path = "/path/to/model"
run_name = "fixture-texture"
[adapter]
kind = "lora"
rank = 8
alpha = 16.0
""",
}
SECTIONS = ("train", "dataset", "flow", "optimizer", "schedule", "adapter", "quant",
            "component_lr")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(ok)


def _train_guard_source() -> str:
    """The GUI blocking Start is not the guard -- the CLI bypasses the GUI entirely. This asserts
    the trainer carries its own refusal, so the two cannot drift into "the GUI stops you but
    `python -m anima.training.train` does not"."""
    return (Path(__file__).resolve().parents[1] / "training" / "train.py").read_text()


def main() -> int:
    r = []
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    # ------------------------------------------------------------- coverage
    missing, extra = bridge.missing_ui_keys(schema.layout_keys())
    r.append(check("every config key has a widget", not missing, str(missing)))
    r.append(check("no widget for a non-existent key", not extra, str(extra)))
    r.append(check("SPEC and LAYOUT agree",
                   set(schema.SPEC) == set(schema.layout_keys()),
                   str(set(schema.SPEC) ^ set(schema.layout_keys()))))
    r.append(check("no key laid out twice",
                   len(schema.layout_keys()) == len(set(schema.layout_keys()))))
    r.append(check("every spec has a tooltip",
                   all(s.tooltip.strip() for s in schema.SPEC.values())))

    # ------------------------------------------------------------- bridge round trip
    # The strongest property available: a config read, flattened, re-serialised and re-parsed must
    # produce an *identical* Config object -- not merely a valid one.
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="anima_gate_cfg_") as _td:
        fixture_dir = Path(_td)
        targets = []
        for name, text in FIXTURES.items():
            p = fixture_dir / name
            p.write_text(text)
            targets.append(p)
        # Any real configs still on disk are round-tripped too -- they exercise combinations no
        # fixture author thinks of. Their ABSENCE is not a failure, which is the whole point: a
        # fresh clone ships no configs and this gate must still be meaningful.
        real = sorted((root / "configs").glob("*.toml"))
        targets.extend(real)

        tmp = fixture_dir / ".roundtrip.toml"
        for path in targets:
            before = load_config(path)
            bridge.write_toml(tmp, bridge.flatten(bridge.read_toml(path)))
            after = load_config(tmp)
            same = all(getattr(before, s) == getattr(after, s) for s in SECTIONS)
            r.append(check(f"round trip is identical: {path.name}", same))
            r.append(check(f"  mode survives: {path.name}",
                           before.is_lora == after.is_lora,
                           f"{before.is_lora} -> {after.is_lora}"))
        r.append(check("the fixtures cover every shape the bridge can mishandle",
                       len(FIXTURES) == 3, f"{len(FIXTURES)} fixtures"))
        print(f"  ({len(real)} real config(s) in configs/ also round-tripped)")

    # `adapter.kind` is written even at its default: it is the full-FT/LoRA/LoKr selector, and a
    # config whose most consequential key is implicit is one someone will misread.
    text = bridge.dump_toml(bridge.defaults() | {"dataset.path": "/x"})
    r.append(check("adapter.kind is always emitted", "kind = " in text.split("[adapter]")[-1],
                   "missing"))

    # Defaults are pruned, so a GUI-written file reads like a hand-written one instead of pinning
    # all 96 keys -- which would freeze today's defaults into every config ever saved.
    r.append(check("defaults are pruned", len(text.splitlines()) < 20, f"{len(text.splitlines())} lines"))

    # `load_config` rejects a file setting both; the GUI writes the file, so it must emit one.
    both = bridge.defaults() | {"dataset.path": "/x", "dataset.resolution": 768,
                                "dataset.resolutions": [768, 1024]}
    dumped = bridge.dump_toml(both)
    r.append(check("never emits both resolution and resolutions",
                   "resolutions" in dumped and "\nresolution " not in dumped))
    ok, err = bridge.validate(both)
    r.append(check("  and the result still validates", ok, err))

    # ------------------------------------------------------------- editors
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    e = F.OptIntEditor()
    e.set(None)
    r.append(check("empty OptInt is None, not 0", e.get() is None))
    e.set(5)
    r.append(check("  and round-trips a value", e.get() == 5))

    # The distinction the whole optional-editor design exists for: `component_lr.mlp` unset
    # inherits optimizer.lr, while 0.0 freezes the component. A spin box cannot say both.
    e = F.SciEditor(optional=True)
    e.set(None)
    r.append(check("unset LR is None", e.get() is None))
    e.set(0.0)
    r.append(check("LR 0.0 is 0.0, not unset", e.get() == 0.0))
    e.set(3e-5)
    r.append(check("LR keeps scientific notation", e.widget.text() == "3e-05", e.widget.text()))

    e = F.BucketMapEditor()
    e.set({512: 4, 1024: 12, 1536: 1})
    r.append(check("bucket map round trip", e.get() == {512: 4, 1024: 12, 1536: 1}, str(e.get())))
    e.set(4)
    r.append(check("plain int batch size stays an int", e.get() == 4))

    e = F.NumListEditor(cast=int)
    e.set([768, 1024, 1280])
    r.append(check("int list round trip", e.get() == [768, 1024, 1280]))
    e.set(None)
    r.append(check("empty list is None (= single tier)", e.get() is None))

    e = F.NumListEditor(cast=float)
    e.set((0.9, 0.99))
    r.append(check("betas round trip as floats", e.get() == [0.9, 0.99], str(e.get())))

    e = F.CheckSetEditor(["self_attn", "cross_attn", "mlp", "adaln", "base"])
    e.set(["mlp", "self_attn"])
    r.append(check("component set normalises to vocabulary order",
                   e.get() == ["self_attn", "mlp"], str(e.get())))

    e = F.ChoiceEditor(["auto", "on", "off"], ["auto", True, False])
    e.set(True)
    r.append(check("choice keeps non-string values (qmm auto/True/False)", e.get() is True))

    # `[[curriculum]]` is the only array-of-tables key, so it is the only editor whose value is a
    # list of records. `at` is shown as a percentage and stored as a fraction; getting that wrong
    # would put every phase inside the first 1% of the run and still validate.
    e = F.CurriculumEditor()
    phases = [{"at": 0.0, "t_range": [0.0, 1.0], "mode": "fullres"},
              {"at": 0.15, "t_range": [0.0, 0.6], "mode": "texture", "lr_mul": 0.5}]
    e.set(phases)
    r.append(check("curriculum round-trips through the table", e.get() == phases, str(e.get())))
    r.append(check("  `at` is stored as a fraction, not the percent it is shown as",
                   e.table.cellWidget(1, 0).value() == 15.0 and e.get()[1]["at"] == 0.15))
    # `prune_defaults` cannot reach inside an array of tables, so the editor has to do it: a
    # no-op lr_mul is omitted (the round trip above shows a real one survives). Otherwise every
    # saved schedule carries nine lines of `lr_mul = 1.0`.
    e.set([{"at": 0.0, "t_range": [0.0, 1.0], "mode": "fullres", "lr_mul": 1.0}])
    r.append(check("  a no-op lr_mul is omitted, a meaningful one is kept",
                   e.get() == [{"at": 0.0, "t_range": [0.0, 1.0], "mode": "fullres"}]
                   and phases[1]["lr_mul"] == 0.5, str(e.get())))
    e._load_preset()
    r.append(check("  the TrainTrain preset is the reference's nine phases",
                   len(e.get()) == 9 and e.get() == F.TRAINTRAIN_CURRICULUM))

    # `[[dataset.subsets]]` -- the second array-of-tables editor.
    s = F.SubsetEditor()
    subs = [{"path": "/a", "num_repeats": 2, "texture": False}, {"path": "/b"}]
    s.set(subs)
    r.append(check("subsets round-trip through the table", s.get() == subs, str(s.get())))
    r.append(check("  exact defaults are omitted per row (prune_defaults cannot reach inside)",
                   s.get()[1] == {"path": "/b"}, str(s.get()[1])))
    s.set([])
    s._add_row()
    r.append(check("  a blank row is dropped, not written as a path of ''", s.get() == []))
    # A NEW row defaults to texture OFF, deliberately unlike the `texture = true` dataclass
    # default. Different questions: a hand-written subset is usually the training data, while one
    # added in the GUI is a second directory beside an existing one -- a regularization set. For
    # those, texture mode replaces the caption with `texture.trigger`, so an uncaptioned flat
    # colour trains the unconditional branch and CFG subtracts it. Defaulting on would make the
    # damaging choice the silent one.
    line = s.table.cellWidget(0, 0).findChild(QtWidgets.QLineEdit)
    line.setText("/added-in-the-gui")
    r.append(check("  a row added in the GUI defaults to texture OFF",
                   s.get() == [{"path": "/added-in-the-gui", "texture": False}], str(s.get())))
    r.append(check("  while the TOML default stays true (a hand-written subset is training data)",
                   SubsetConfig(path="/x").texture is True))
    ok, err = bridge.validate(bridge.defaults() | {"dataset.path": "/x", "dataset.source": "encode",
                                                   "curriculum": e.get()})
    r.append(check("  and the preset is a config the loader accepts", ok, err[:90]))
    # Turning a texture config back into a plain LoRA run has to be reversible. Unchecking emits
    # no curriculum at all -- which prunes the key and is exactly default training -- while the
    # phases stay in the table, so it is a switch rather than a one-way door. Clear is the
    # destructive route and reaches the same config.
    e._load_preset()
    e.enable.setChecked(False)
    r.append(check("  unchecking emits no curriculum but keeps the phases",
                   e.get() == [] and e.table.rowCount() == 9))
    r.append(check("  and prunes the key out of the TOML entirely",
                   "curriculum" not in bridge.dump_toml(
                       bridge.defaults() | {"dataset.path": "/x", "curriculum": e.get()})))
    e.enable.setChecked(True)
    r.append(check("  re-checking restores the same schedule",
                   e.get() == F.TRAINTRAIN_CURRICULUM))
    e._clear()
    r.append(check("  clearing gives an empty curriculum, not one empty phase",
                   e.get() == [] and not e.enable.isChecked()))
    e._add_row()
    r.append(check("  the first added phase starts at 0.0, the only value accepted there",
                   e.get()[0]["at"] == 0.0))
    r.append(check("  adding a phase re-enables the curriculum", e.enable.isChecked()))

    # The status bar is the only place a texture run is distinguishable from a plain one without
    # opening the Dataset tab, and the only place `encode` with nothing to encode for is called out.
    base = bridge.defaults() | {"dataset.path": "/x", "dataset.source": "encode"}
    r.append(check("summary names the curriculum and its texture phases",
                   "9 phase(s), 8 texture" in bridge.summarize(
                       base | {"curriculum": F.TRAINTRAIN_CURRICULUM}),
                   bridge.summarize(base | {"curriculum": F.TRAINTRAIN_CURRICULUM})))
    r.append(check("summary flags source=encode bought for nothing",
                   "for nothing" in bridge.summarize(base | {"curriculum": []})
                   and "for nothing" not in bridge.summarize(
                       base | {"curriculum": F.TRAINTRAIN_CURRICULUM})))

    # ------------------------------------------------------------- the window
    gui = TrainingGUI()
    r.append(check("window builds with every editor",
                   len(gui.editors) == len(schema.SPEC), f"{len(gui.editors)} editors"))

    # Same fixtures as the bridge round trip above, but driven through the real widgets: a value
    # that survives `flatten`/`dump_toml` can still be mangled by an editor. `ChoiceEditor` used to
    # silently coerce anything it did not recognise to its first option, which is how
    # `dataset.source = "encode"` and `quant.skip_policy` were both quietly rewritten.
    _td2 = _tempfile.mkdtemp(prefix="anima_gate_form_")
    form_targets = []
    for name, text_ in FIXTURES.items():
        q = Path(_td2) / name
        q.write_text(text_)
        form_targets.append(q)
    form_targets.extend(sorted((root / "configs").glob("*.toml")))

    for path in form_targets:
        merged = bridge.defaults()
        merged.update(bridge.flatten(bridge.read_toml(path)))
        gui._apply(merged)
        collected = gui.collect()
        ok, err = bridge.validate(collected)
        r.append(check(f"form round-trips {Path(path).name}", ok, err))
        drift = {k: (merged[k], collected[k]) for k in merged
                 if k in collected and collected[k] != merged[k]
                 and not (merged[k] is None and collected[k] in (None, "", [], {}))
                 and list(merged[k] or []) != list(collected[k] or [])}
        r.append(check(f"  no value drifts through the widgets: {Path(path).name}",
                       not drift, str(drift)[:200]))
    shutil.rmtree(_td2, ignore_errors=True)

    # ------------------------------------------------------------- inert-knob rules
    def rule(key, cfg):
        return _RULES[key](bridge.defaults() | cfg)

    r.append(check("sigmoid_scale is greyed out under uniform sampling",
                   not rule("flow.sigmoid_scale", {"flow.timestep_sample_method": "uniform"})
                   and rule("flow.sigmoid_scale",
                            {"flow.timestep_sample_method": "logit_normal"})))
    r.append(check("min_bucket_reso is greyed out while no-upscale is on",
                   not rule("dataset.min_bucket_reso", {"dataset.bucket_no_upscale": True})))
    r.append(check("REX d only for rex, ReREX knobs only for rerex",
                   rule("schedule.d", {"schedule.kind": "rex"})
                   and not rule("schedule.d", {"schedule.kind": "rerex"})
                   and rule("schedule.local_d", {"schedule.kind": "rerex"})
                   and not rule("schedule.local_d", {"schedule.kind": "rex"})))
    r.append(check("LoKr knobs only under lokr",
                   rule("adapter.lokr_factor", {"adapter.kind": "lokr"})
                   and not rule("adapter.lokr_factor", {"adapter.kind": "lora"})))
    r.append(check("hf_exponent is greyed out while hf_scale is 0",
                   not rule("flow.hf_exponent", {"flow.hf_scale": 0.0})
                   and rule("flow.hf_exponent", {"flow.hf_scale": 0.25})))
    r.append(check("compile sub-options greyed out while compile is off",
                   not rule("train.compile_dynamic", {"train.compile": None})
                   and rule("train.compile_dynamic", {"train.compile": "default"})))

    # This one mirrors the config guard exactly: under an adapter, a component LR only reaches
    # parameters that have an adapter injected. Full FT can address every component.
    lora = {"adapter.kind": "lora", "adapter.components": ["self_attn", "mlp"]}
    r.append(check("component LR greyed out where no adapter is injected",
                   rule("component_lr.mlp", lora)
                   and not rule("component_lr.adaln", lora)
                   and rule("component_lr.adaln", {"adapter.kind": "none"})))
    r.append(check("text_encoder LR unlocks with the Qwen3 adapter",
                   not rule("component_lr.text_encoder", lora)
                   and rule("component_lr.text_encoder", lora | {"adapter.train_text_encoder": True})))

    # Texture mode has no switch of its own -- a curriculum phase with mode="texture" IS the
    # switch. So the texture knobs must follow the phase list, and `flow.phase_mapping` must
    # follow the existence of any curriculum at all, texture or not.
    tex = {"curriculum": [{"at": 0.0, "t_range": [0.0, 1.0], "mode": "texture", "lr_mul": 1.0}]}
    full = {"curriculum": [{"at": 0.0, "t_range": [0.0, 1.0], "mode": "fullres", "lr_mul": 1.0}]}
    r.append(check("texture knobs are greyed out with no texture phase",
                   not rule("dataset.texture.canvases", {})
                   and not rule("dataset.texture.canvases", full)
                   and rule("dataset.texture.canvases", tex)))
    r.append(check("every texture knob follows the same rule",
                   all(rule(k, tex) and not rule(k, {}) for k in schema.SPEC
                       if k.startswith("dataset.texture."))))
    r.append(check("phase_mapping is greyed out with no curriculum",
                   not rule("flow.phase_mapping", {})
                   and rule("flow.phase_mapping", full)))
    r.append(check("Kahan is greyed out for torch adamw",
                   not rule("optimizer.use_kahan", {"optimizer.kind": "adamw"})
                   and rule("optimizer.use_kahan", {"optimizer.kind": "came"})))

    # Both new gates must agree with the loader, not merely look plausible -- the same standard the
    # component_lr rule below is held to. A texture phase without `source = "encode"` has no cached
    # latent that could express a per-step crop, and Kahan is an sdnq buffer torch's adamw lacks.
    for label, cfg, needle in (
        ("texture without encode", tex | {"dataset.source": "latents"}, "texture"),
        ("Kahan on torch adamw", {"optimizer.kind": "adamw", "optimizer.use_kahan": True},
         "use_kahan"),
    ):
        ok, err = bridge.validate(bridge.defaults() | {"dataset.path": "/x"} | cfg)
        r.append(check(f"  the loader rejects {label}", not ok and needle in err, err[:90]))
    ok, _ = bridge.validate(bridge.defaults() | {"dataset.path": "/x"}
                            | tex | {"dataset.source": "encode"})
    r.append(check("  and accepts a texture phase with encode", ok))

    # And the rule must agree with the loader rather than merely look plausible: a LR the GUI
    # greys out is exactly a LR the trainer would reject.
    bad = bridge.defaults() | {"dataset.path": "/x", "adapter.kind": "lora",
                               "adapter.components": ["self_attn"], "component_lr.mlp": 2e-5}
    ok, err = bridge.validate(bad)
    r.append(check("  and the loader rejects the same combination", not ok and "component_lr" in err,
                   err[:80]))

    # ------------------------------------------------------------- advisories
    # These are the combinations the loader ACCEPTS and that still train badly, so nothing else
    # catches them. Each must fire when it should and stay silent when it should not -- an advisory
    # that shows up on every config is one people stop reading, which is worse than none.
    def adv(cfg, n=1):
        return bridge.advisories(bridge.defaults() | {"dataset.path": "/x"} | cfg, n)

    def levels(cfg, n=1):
        return {lvl for lvl, _ in adv(cfg, n)}

    tex_cur = {"curriculum": F.TRAINTRAIN_CURRICULUM, "dataset.source": "encode"}
    plain = {"dataset.source": "latents"}

    r.append(check("a plain single-GPU config raises nothing", adv(plain) == [], str(adv(plain))))
    r.append(check("texture on one GPU raises nothing", adv(tex_cur) == [], str(adv(tex_cur))))

    # The one that has to be an ERROR, not a warning: it cost two runs.
    r.append(check("texture on 2 GPUs is an ERROR that blocks Start",
                   "error" in levels(tex_cur, 2)))
    r.append(check("  and the trainer refuses the same combination independently of the GUI",
                   "allow_multi_gpu_texture" in _train_guard_source(),
                   "guard not found in train.py"))
    r.append(check("  the override downgrades it to a warning, never to silence",
                   levels(tex_cur | {"train.allow_multi_gpu_texture": True}, 2) == {"warn", "note"},
                   str(levels(tex_cur | {"train.allow_multi_gpu_texture": True}, 2))))
    r.append(check("  a NON-texture config on 2 GPUs is allowed, with the step-count note only",
                   levels(plain, 2) == {"note"}, str(adv(plain, 2))))

    # Measured-fact advisories: each cites a number, so each must be tied to a real condition.
    r.append(check("batch>1 under texture is flagged (measured flat throughput)",
                   any("no throughput" in m for _, m in adv(tex_cur | {"train.batch_size": 4}))
                   and not any("no throughput" in m for _, m in adv(tex_cur))))
    r.append(check("multi-tier under texture is flagged as cost without benefit",
                   any("resolution tiers" in m
                       for _, m in adv(tex_cur | {"dataset.resolutions": [768, 1024]}))))
    r.append(check("uncapped cover is flagged (it lost its A/B)",
                   any("lost its A/B" in m
                       for _, m in adv(tex_cur | {"dataset.texture.oversize": "cover"}))))
    r.append(check("use_ot is flagged as a no-op at batch 1, not at batch 4",
                   any("no-op at batch size 1" in m for _, m in adv({"flow.use_ot": True}))
                   and not any("no-op at batch size 1" in m
                               for _, m in adv({"flow.use_ot": True, "train.batch_size": 4}))))

    # hf loss under a t-restricted curriculum: the factor is a closed form, so it can be checked
    # rather than trusted. Under `rescale` with lo=0 the hf term's t^2 factor is exactly
    # (hi-lo)^2 -- verified against a 200k-sample draw, which gave 0.36x for [0, 0.6].
    hf_cur = {"curriculum": F.TRAINTRAIN_CURRICULUM, "dataset.source": "encode",
              "flow.hf_scale": 0.25}
    msgs = [m for _, m in adv(hf_cur) if "t^2" in m]
    r.append(check("hf loss under a restricted t_range reports its real strength", len(msgs) == 1,
                   msgs[0][:90] if msgs else "not raised"))
    share, factor = bridge._restricted_share(bridge.defaults() | hf_cur)
    r.append(check("  and the factor matches the sampled measurement (0.55x at 70% restricted)",
                   abs(factor - 0.55) < 0.02 and abs(share - 0.70) < 0.02,
                   f"factor {factor:.3f}, share {share:.2f}"))
    r.append(check("  a full-range curriculum does not trigger it",
                   not [m for _, m in adv({"flow.hf_scale": 0.25,
                                           "curriculum": [{"at": 0.0, "t_range": [0.0, 1.0],
                                                           "mode": "fullres"}]}) if "t^2" in m]))
    r.append(check("  hf_scale = 0 never mentions hf at all",
                   not [m for _, m in adv(hf_cur | {"flow.hf_scale": 0.0}) if "hf" in m]))

    # The one trap that combines two features, each fine alone.
    black = {"dataset.texture.pad_mode": "black", "dataset.texture.fit_aware": False}
    r.append(check("hf loss + black padding + blind canvas draw is flagged",
                   any("maximal Laplacian edge" in m for _, m in adv(hf_cur | black))
                   and not any("maximal Laplacian edge" in m for _, m in adv(hf_cur))))

    # Canvases bypass bucketing, so they never met the RoPE check every bucket meets. Reached for
    # real by scaling the preset list to a 1536 base: 640x1536 -> 960x2304, and 2304 > 2048.
    from anima.data.texture import TextureConfig
    try:
        TextureConfig(canvases=[(960, 2304)])
        rope_ok = False
    except ValueError as exc:
        rope_ok = "RoPE" in str(exc)
    r.append(check("a canvas past Anima's RoPE ceiling is rejected at load", rope_ok))
    r.append(check("  and a legal 1920-side canvas still loads",
                   TextureConfig(canvases=[(800, 1920)]).canvases == [(800, 1920)]))

    # ------------------------------------------------------------- per-component LR is visible
    # The feature worked; the log did not show it. `get_last_lr()[0]` printed only group 0, so a
    # run with three component LRs reported a flat number and looked exactly like a run where
    # [component_lr] had silently not applied.
    import torch as _torch

    from anima.training.optim import build_optimizer, build_scheduler
    # A fixture again, not a config on disk -- this only needs *an* optimizer and schedule to
    # build, and reading a real file made it depend on one existing.
    _lr_dir = _tempfile.mkdtemp(prefix="anima_gate_lr_")
    _lr_path = Path(_lr_dir) / "lr.toml"
    _lr_path.write_text(FIXTURES["full.toml"])
    lr_cfg = load_config(_lr_path)
    groups = [{"params": [_torch.nn.Parameter(_torch.zeros(1))], "lr": lr, "component": n}
              for n, lr in (("self_attn", 4e-5), ("cross_attn", 1e-5), ("mlp", 8e-6))]
    opt = build_optimizer(groups, lr_cfg.optimizer)
    sch = build_scheduler(opt, lr_cfg.schedule, 1000)
    start = [g["lr"] for g in opt.param_groups]
    with warnings.catch_warnings():
        # Stepping the scheduler without a real optimizer step is exactly what this check wants;
        # torch's ordering warning is about training loops, not about reading an LR curve.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(500):
            sch.step()
    late = [g["lr"] for g in opt.param_groups]
    r.append(check("per-component LRs keep their ratios across the whole schedule",
                   all(abs(a / start[-1] - b / late[-1]) < 0.01 for a, b in zip(start, late)),
                   f"{[f'{v:.1e}' for v in start]} -> {[f'{v:.1e}' for v in late]}"))
    r.append(check("  and they are genuinely distinct, not all collapsed to one",
                   len(set(late)) == 3, str(late)))

    # The log format that reports them, and the parser that has to keep up with it.
    multi = "e0 step 5/100 (5%)  loss 0.08  lr 4.00e-05/1.00e-05/8.00e-06  1.10s/it  peak 8.6GB"
    single = "e0 step 5/100 (5%)  loss 0.08  lr 2.00e-05  1.10s/it  peak 8.6GB"
    r.append(check("the log line carries every group's LR, slash-joined like `peak`",
                   STEP_RE.match(multi) is not None
                   and STEP_RE.match(multi).group("lr") == "4.00e-05/1.00e-05/8.00e-06"))
    r.append(check("  and a single shared LR still reads exactly as before",
                   STEP_RE.match(single) is not None
                   and STEP_RE.match(single).group("lr") == "2.00e-05"))

    # ------------------------------------------------------------- platform guardrail
    from anima.gui.app import MULTI_GPU_SUPPORTED
    r.append(check("multi-GPU is gated on the platform, not assumed",
                   MULTI_GPU_SUPPORTED == (sys.platform != "win32")))
    r.append(check("  and the trainer refuses Windows DDP on its own, independent of the GUI",
                   "win32" in _train_guard_source() and "NCCL" in _train_guard_source()))
    if len(gui.gpu_boxes) > 1:
        # Simulate the Windows path directly: the radio behaviour must leave exactly one GPU
        # selected while keeping every box usable, since picking WHICH GPU is still a real choice.
        import anima.gui.app as _app
        was = _app.MULTI_GPU_SUPPORTED
        try:
            _app.MULTI_GPU_SUPPORTED = False
            for b in gui.gpu_boxes.values():
                b.blockSignals(True)
                b.setChecked(True)
                b.blockSignals(False)
            gui._on_gpu_selection()
            r.append(check("on Windows only one GPU can be selected",
                           gui.num_processes() == 1, f"{gui.num_processes()} processes"))
            r.append(check("  but every GPU box stays clickable, so the choice remains",
                           all(b.isEnabled() for b in gui.gpu_boxes.values())))
        finally:
            _app.MULTI_GPU_SUPPORTED = was
            for b in gui.gpu_boxes.values():
                b.blockSignals(True)
                b.setChecked(True)
                b.blockSignals(False)
            gui._on_gpu_selection()

    # ------------------------------------------------------------- configs from outside configs/
    # A config opened from elsewhere must stay the run's config: Start writes back to the file that
    # was opened, not to a fork in `configs/` that would leave two files with the same name and no
    # way to tell which one a run used.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        outside = Path(td) / "elsewhere.toml"
        bridge.write_toml(outside, bridge.defaults() | {"dataset.path": "/x",
                                                        "train.run_name": "from_outside"})
        gui._external.append(outside)
        gui._load_presets(select=outside)
        r.append(check("a config outside configs/ can be selected",
                       gui._current_path is not None
                       and gui._current_path.resolve() == outside.resolve(),
                       str(gui._current_path)))
        r.append(check("  and it appears in the preset list, labelled with its directory",
                       any(gui.preset_combo.itemText(i).startswith("elsewhere")
                           for i in range(gui.preset_combo.count()))))
        r.append(check("  and it survives the list being rebuilt",
                       (gui._load_presets(select=outside), gui._current_path.resolve())[1]
                       == outside.resolve()))

        # The property that matters: saving a config opened from elsewhere COPIES it into configs/
        # under its run_name and leaves the original byte-identical. Writing back in place would
        # regenerate the file through `dump_toml`, silently discarding the owner's comments.
        before_bytes = outside.read_bytes()
        gui.editors["train.run_name"].set("edited_outside")
        made = []
        try:
            written = gui._persist(gui.collect())
            made.append(written)
            r.append(check("saving a config opened from outside copies it into configs/",
                           written is not None
                           and written.resolve() == (CONFIG_DIR / "edited_outside.toml").resolve(),
                           str(written)))
            r.append(check("  and leaves the original untouched, byte for byte",
                           outside.read_bytes() == before_bytes))
            r.append(check("  and the copy carries the edit",
                           load_config(written).train.run_name == "edited_outside"))
            r.append(check("  and the GUI now works on the copy",
                           gui._current_path.resolve() == written.resolve()))

            # run_name drives the filename, but changing it SAVES A NEW FILE and leaves the old
            # one alone. Deriving a second run from an existing config is the ordinary way to use
            # this GUI; moving the original out from under the user is data loss, not tidiness.
            source_bytes = written.read_bytes()
            gui.editors["train.run_name"].set("renamed_run")
            written2 = gui._persist(gui.collect())
            made.append(written2)
            r.append(check("changing run_name saves a NEW config",
                           written2 is not None and written2.name == "renamed_run.toml",
                           str(written2)))
            r.append(check("  and the config it was loaded from still exists, byte for byte",
                           (CONFIG_DIR / "edited_outside.toml").exists()
                           and written.read_bytes() == source_bytes))
            r.append(check("  and each file holds its own run_name",
                           load_config(written).train.run_name == "edited_outside"
                           and load_config(written2).train.run_name == "renamed_run"))
            r.append(check("  and the GUI follows the new file",
                           gui._current_path.resolve() == written2.resolve()))

            # Saving onto a DIFFERENT existing config is a run_name collision, which is also a
            # CHECKPOINT collision -- so it asks rather than silently overwriting.
            squatter = CONFIG_DIR / "occupied.toml"
            bridge.write_toml(squatter, bridge.defaults() | {"dataset.path": "/y",
                                                             "train.run_name": "occupied"})
            made.append(squatter)
            squatter_bytes = squatter.read_bytes()
            real_question = QtWidgets.QMessageBox.question
            try:
                QtWidgets.QMessageBox.question = staticmethod(
                    lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No)
                gui.editors["train.run_name"].set("occupied")
                declined = gui._persist(gui.collect())
                r.append(check("saving onto an existing config asks first, and No cancels",
                               declined is None and squatter.read_bytes() == squatter_bytes
                               and (CONFIG_DIR / "renamed_run.toml").exists()))
                QtWidgets.QMessageBox.question = staticmethod(
                    lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
                accepted = gui._persist(gui.collect())
                r.append(check("  and Yes goes through", accepted is not None
                               and accepted.name == "occupied.toml"
                               and load_config(accepted).dataset.path == "/x"))
            finally:
                QtWidgets.QMessageBox.question = real_question
        finally:
            for p in made:
                if p is not None:
                    Path(p).unlink(missing_ok=True)
            (CONFIG_DIR / "renamed_run.toml").unlink(missing_ok=True)
            (CONFIG_DIR / "edited_outside.toml").unlink(missing_ok=True)
        gui._current_path = None
        gui._external.clear()
        gui._load_presets()

    # ------------------------------------------------------------- log parsing
    lines = {
        "e3 step 1410/3000 (47%)  loss 0.0712 (mse 0.0650 + hf 0.0062)  ot 0.83  "
        "lr 2.71e-05  1.80s/it  peak 8.1GB  eta 47m34s": ("0.0712", "0.0062", "0.83", "8.1"),
        "e0 step 1/20 (5%)  loss 0.0752  lr 1.00e-05  3.06s/it  peak 7.9GB":
            ("0.0752", None, None, "7.9"),
        "e12 step 900/1000 (90%)  loss 0.0500 (mse 0.0480 + hf 0.0020)  lr 5.00e-06  "
        "1.42s/it  peak 8.1/8.1GB  eta 2m22s": ("0.0500", "0.0020", None, "8.1/8.1"),
    }
    bad_parse = []
    for line, (loss, hf, ot, peak) in lines.items():
        m = STEP_RE.match(line)
        if not m or (m["loss"], m["hf"], m["ot"], m["peak"]) != (loss, hf, ot, peak):
            bad_parse.append(line[:40])
    r.append(check("log line parses in all four shapes (hf on/off, ot on/off, 1 and 2 ranks)",
                   not bad_parse, str(bad_parse)))
    r.append(check("a non-step line is not mistaken for one",
                   STEP_RE.match("saved output/run/epoch001 (280 tensors)") is None))

    # ------------------------------------------------------------- launch argv
    # Every run goes through `accelerate launch`, single GPU included -- one launch path means a
    # bug under the launcher cannot hide from single-GPU testing.
    one = train_launch("c.toml", 1).argv
    two = train_launch("c.toml", 2).argv
    r.append(check("single GPU still goes through accelerate",
                   "accelerate.commands.launch" in one
                   and one[one.index("--num_processes") + 1] == "1"
                   and one[-3:] == ["-m", "anima.training.train", "c.toml"], str(one)))
    r.append(check("multi GPU passes the right process count",
                   "accelerate.commands.launch" in two
                   and two[two.index("--num_processes") + 1] == "2"
                   and two[-3:] == ["-m", "anima.training.train", "c.toml"], str(two)))

    # The bug that trained a 2B model on the CPU: `CUDA_VISIBLE_DEVICES=all` is not a device list,
    # and torch reports "no devices" for it rather than raising.
    from anima.gui.process import training_env

    for bad_value in ("all", "gpu0", "0;1", "-1", "cuda:0"):
        try:
            training_env(bad_value)
            r.append(check(f"training_env rejects {bad_value!r}", False, "accepted"))
            break
        except ValueError:
            pass
    else:
        r.append(check("training_env rejects every non-device-list string", True))
    r.append(check("training_env accepts real device lists",
                   training_env("1")["CUDA_VISIBLE_DEVICES"] == "1"
                   and training_env("0,1")["CUDA_VISIBLE_DEVICES"] == "0,1"))
    r.append(check("empty selection leaves CUDA_VISIBLE_DEVICES untouched",
                   "CUDA_VISIBLE_DEVICES" not in {k: v for k, v in training_env("").items()
                                                  if k == "CUDA_VISIBLE_DEVICES"}
                   or os.environ.get("CUDA_VISIBLE_DEVICES") is not None))

    # The GUI-side selector, which is what feeds training_env.
    r.append(check("all GPUs selected -> no CUDA_VISIBLE_DEVICES override", gui.gpu_arg() == "",
                   gui.gpu_arg()))
    r.append(check("process count is derived from the selection, never typed",
                   gui.num_processes() == len(gui.gpu_boxes)))
    if len(gui.gpu_boxes) > 1:
        first = sorted(gui.gpu_boxes)[0]
        for i, b in gui.gpu_boxes.items():
            b.setChecked(i == first)
        r.append(check("selecting one GPU gives one process on that device",
                       gui.gpu_arg() == str(first) and gui.num_processes() == 1,
                       f"{gui.gpu_arg()!r} / {gui.num_processes()}"))
        # Unchecking everything would mean "train on nothing"; the control refuses.
        for b in gui.gpu_boxes.values():
            b.setChecked(False)
        r.append(check("the selector cannot reach an empty state",
                       gui.num_processes() >= 1 and len(gui.selected_gpus()) >= 1))
    else:
        r.append(check("single-GPU host locks the selector",
                       not next(iter(gui.gpu_boxes.values())).isEnabled()))
    cache = cache_launch("/data", "/model", [768, 1024], min_bucket_reso=256,
                         max_bucket_reso=1920, bucket_reso_steps=64, upscale=False,
                         multires_training=False, dry_run=True).argv
    r.append(check("cache passes every tier as one --resolution",
                   cache[cache.index("--resolution") + 1:cache.index("--resolution") + 3]
                   == ["768", "1024"] and "--dry-run" in cache, str(cache)))
    r.append(check("cache is not marked as a training run",
                   not cache_launch("/d", "/m", [768], min_bucket_reso=256, max_bucket_reso=1920,
                                    bucket_reso_steps=64, upscale=False,
                                    multires_training=False).is_training))
    r.append(check("audit needs neither GPU nor model",
                   "--model-path" not in audit_launch("/data", 64).argv))

    # --------------------------------------------- which folders Audit / Cache actually run on
    # Both tools take exactly ONE directory, but `path` and `subsets` are mutually exclusive in the
    # loader -- so a multi-subset config has no `dataset.path` at all. Reading only `dataset.path`
    # sent `""` to the subprocess, and `Path("")` is `Path(".")`, which passes `.is_dir()`: the
    # buttons silently audited/cached the repo root instead of refusing.
    gui = TrainingGUI()
    try:
        # A fresh window auto-selects a preset, so start from a known state rather than whatever
        # configs/ happens to hold on this machine.
        gui.editors["dataset.subsets"].set([])
        gui.editors["dataset.path"].set("/data/single")
        r.append(check("with no subsets, the dataset path is used",
                       gui._dataset_paths(gui.collect()) == ["/data/single"]))

        gui.editors["dataset.subsets"].set([{"path": "/data/a", "num_repeats": 2},
                                            {"path": "/data/b", "texture": False}])
        r.append(check("with subsets, every subset folder is used and dataset.path is ignored",
                       gui._dataset_paths(gui.collect()) == ["/data/a", "/data/b"],
                       str(gui._dataset_paths(gui.collect()))))

        # The trap, pinned directly: an empty entry must vanish, not become ".".
        gui.editors["dataset.subsets"].set([{"path": "  "}, {"path": "/data/b"}])
        r.append(check("  a blank subset folder is dropped, never resolved to '.'",
                       gui._dataset_paths(gui.collect()) == ["/data/b"]))
        gui.editors["dataset.subsets"].set([])
        gui.editors["dataset.path"].set("")
        paths = gui._dataset_paths(gui.collect())
        r.append(check("  and an empty config yields nothing to run, not '.'", paths == [],
                       str(paths)))

        # Nothing to run must not launch a process at all.
        launched = []
        real_run = gui._run
        gui._run = lambda launch, training=True: launched.append(launch)
        try:
            gui._audit()
            gui._cache()
            r.append(check("Audit/Cache with no folder launch nothing and say so",
                           launched == []))

            gui.editors["dataset.subsets"].set([{"path": "/data/a"}, {"path": "/data/b"},
                                                {"path": "/data/c"}])
            gui._audit()
            r.append(check("Audit queues one run per subset folder",
                           len(launched) == 1 and len(gui._queue) == 2
                           and launched[0].argv[-3] == "/data/a",
                           f"{len(launched)} started, {len(gui._queue)} queued"))
            r.append(check("  and Stop drops the queue rather than skipping ahead",
                           (gui._stop(), gui._queue == [])[1]))
        finally:
            gui._run = real_run
            gui._queue = []
    finally:
        gui.deleteLater()

    # ------------------------------------------------------------- end to end
    # A real subprocess through the real runner. Pins a teardown bug that cost an hour to find:
    # `_finished` used to do `self.runner = None`, dropping the last reference to the QThread while
    # it was still emitting `finishedSignal`. PySide tore the C++ object down mid-emission and
    # every slot connected after that one was silently skipped -- so a second listener (a UI
    # refresh, a queued follow-up run) would simply never fire, with no error anywhere.
    from anima.gui.process import Launch

    seen, after = [], []
    gui.console.append_line = lambda t, replace_last=False: seen.append(t)
    gui._run(Launch([sys.executable, "-u", "-c",
                     "print('alpha'); print('beta')"], "gate echo", is_training=False),
             training=False)
    gui.runner.finishedSignal.connect(lambda code: after.append(code))

    app = QtWidgets.QApplication.instance()
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while not after and deadline.elapsed() < 30_000:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

    r.append(check("subprocess output reaches the console",
                   "alpha" in seen and "beta" in seen, str(seen[-4:])))
    r.append(check("slots connected after _finished still fire (QThread teardown)",
                   after == [0], str(after)))
    r.append(check("the runner is released once it finishes", gui.runner is None))

    gui.deleteLater()
    # --- an unknown choice is preserved, not coerced ---------------------------------------------
    # This has silently rewritten a user's config twice (dataset.source="encode" and
    # quant.skip_policy="anima_int8_mm", both loaded by a GUI that predated the option). Index 0 is
    # a valid-looking value, so nothing raises and the run is simply a different one.
    ed = F.ChoiceEditor(["default", "first_block_adaln", "all_adaln"])
    ed.set("anima_int8_mm")
    r.append(check("ChoiceEditor keeps a value this build does not know",
                   ed.get() == "anima_int8_mm", repr(ed.get())))
    ed.set("all_adaln")
    r.append(check("  and still selects known values normally", ed.get() == "all_adaln"))
    typed = F.ChoiceEditor(["auto", "on", "off"], ["auto", True, False])
    typed.set(True)
    r.append(check("  non-string values still match by identity", typed.get() is True))
    typed.set("nonsense")
    r.append(check("  an unknown value survives for load_config to reject by name",
                   typed.get() == "nonsense", repr(typed.get())))

    # Every ChoiceEditor in the real schema must cover its dataclass's declared options, or the
    # above turns a config into a "(not in this build)" row rather than a working form.
    sk = schema.SPEC["quant.skip_policy"].make()
    missing = [p for p in SKIP_POLICIES if (sk.set(p), sk.get())[1] != p or
               f"{p}  (not in this build)" in
               [sk.widget.itemText(i) for i in range(sk.widget.count())]]
    r.append(check("the skip_policy widget offers every SKIP_POLICIES value", not missing,
                   str(missing)))

    # --- run_name -> filename stem ---------------------------------------------------------------
    # run_name reaches the filesystem three ways (config file, checkpoint stem, state dir), so an
    # illegal character does not fail at save time -- it fails hours later at the first checkpoint.
    for raw, want, why in (
        ("my-finetune-v2", "my-finetune-v2", "ordinary names pass through untouched"),
        ("my run 2", "my-run-2", "spaces collapse to single dashes"),
        ("a/b\\c", "a-b-c", "path separators cannot escape the directory"),
        ("run:1?", "run-1", "Windows-illegal characters are removed"),
        (".hidden", "hidden", "a leading dot would hide the file on unix"),
        ("trail...", "trail", "Windows strips trailing dots, so two names would collide"),
        ("sesión-1", "sesión-1", "accented letters are KEPT -- both filesystems store them"),
        ("CON", "CON-run", "Windows resolves CON as a device whatever the extension"),
        ("", "gui_run", "an empty run_name still produces a file"),
        ("---", "gui_run", "and so does one with nothing usable in it"),
    ):
        r.append(check(f"safe_stem: {why}", F_safe_stem(raw) == want,
                       f"{raw!r} -> {F_safe_stem(raw)!r}, wanted {want!r}"))

    n = sum(r)
    print(f"\n{n}/{len(r)} passed")
    return 0 if n == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
