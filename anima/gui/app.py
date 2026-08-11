"""The window.

Shell structure (tab bar / stacked pages / footer) follows Aozora's; every page is ours.

The one idea worth stating: **the GUI reimplements no validation.** Every keystroke re-serialises
the form to TOML and hands it to `anima.training.config.load_config` -- the same function the
trainer calls -- and shows whatever it raises. Start stays disabled until that passes. So the
config-level guards this trainer accumulated (both `resolution` and `resolutions` set;
`sigmoid_scale` under `uniform`; a `component_lr` on a component with no adapter injected; `frozen`
quantization with nothing to train) are enforced here for free and cannot drift.

On top of that, `_RULES` greys out controls that the config would *accept* but that would do
nothing in the current combination. That is one step ahead of the validators: a disabled box says
"this knob is inert right now" before you spend a run finding out.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from . import bridge
from .metrics import LiveMetricsWidget
from .process import ProcessRunner, audit_launch, cache_launch, train_launch, training_env
from .schema import LAYOUT, SPEC
from .widgets import (
    DANGER,
    PROJECT_ROOT,
    STYLESHEET,
    SUCCESS,
    THEME,
    WARN,
    VirtualConsoleWidget,
    group_box,
    make_btn,
    make_label,
    prevent_sleep,
    safe_stem,
    usable_dialog_start,
    set_role,
)

CONFIG_DIR = PROJECT_ROOT / "configs"

# DDP here goes over NCCL, which has no Windows build -- torch falls back to gloo, which does not
# support the GPU collectives this needs. So on Windows the GPU picker is a one-of-N choice rather
# than a multi-select. Every VRAM and throughput measurement in this repo is single-GPU anyway;
# what is lost is the 1.75x, not the ability to train.
MULTI_GPU_SUPPORTED = sys.platform != "win32"


def detect_gpus() -> list[tuple[int, str]]:
    """[(index, name)], via nvidia-smi.

    Deliberately not via torch: importing torch here would create a CUDA context in the GUI
    process and hold a few hundred MB for the whole session, so the trainer would start with less
    VRAM than every measurement in this repo assumes.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        gpus = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            idx, _, name = line.partition(",")
            gpus.append((int(idx.strip()), name.strip()))
        return gpus or [(0, "GPU 0")]
    except Exception:
        return [(0, "GPU 0")]


# Controls that exist but would do nothing in the current combination. Each entry is
# key -> predicate(flat_config) -> enabled.
_RULES = {
    # Measured: under `uniform`, scale 0.5/1.0/2.0 give byte-identical distributions.
    "flow.sigmoid_scale": lambda c: c.get("flow.timestep_sample_method") == "logit_normal",
    "flow.hf_exponent": lambda c: float(c.get("flow.hf_scale") or 0) > 0,

    "dataset.resolution": lambda c: not c.get("dataset.resolutions"),
    "dataset.tier_collapse": lambda c: bool(c.get("dataset.resolutions")),
    "dataset.min_source_area": lambda c: bool(c.get("dataset.resolutions")),
    # Inert on the no_upscale path -- verified, 256/128/64 give identical buckets.
    "dataset.min_bucket_reso": lambda c: not c.get("dataset.bucket_no_upscale", True),
    "dataset.caption.mixed_weights": lambda c: c.get("dataset.caption.caption_mode") == "mixed",
    "dataset.caption.shuffle_keep_first_n": lambda c: bool(c.get("dataset.caption.shuffle_tags")),
    "dataset.caption.nl_keep_first_sentence":
        lambda c: bool(c.get("dataset.caption.nl_shuffle_sentences")),

    # Texture settings only mean anything if some phase asks for texture. There is no separate
    # "texture mode" switch to gate on -- the curriculum IS the switch -- so the rule reads the
    # phase list. `flow.phase_mapping` is inert without any curriculum at all, texture or not.
    "flow.phase_mapping": lambda c: bool(c.get("curriculum")),

    "optimizer.use_kahan": lambda c: c.get("optimizer.kind") != "adamw",

    "schedule.min_lr_ratio": lambda c: c.get("schedule.kind") != "constant",
    "schedule.d": lambda c: c.get("schedule.kind") == "rex",
    "schedule.global_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.local_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.weight_power": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.num_segments": lambda c: c.get("schedule.kind") == "rerex",

    "adapter.rank": lambda c: c.get("adapter.kind") != "none",
    "adapter.alpha": lambda c: c.get("adapter.kind") != "none",
    "adapter.dropout": lambda c: c.get("adapter.kind") != "none",
    "adapter.components": lambda c: c.get("adapter.kind") != "none",
    "adapter.train_llm_adapter": lambda c: c.get("adapter.kind") != "none",
    "adapter.train_text_encoder": lambda c: c.get("adapter.kind") != "none",
    "adapter.text_encoder_rank": lambda c: bool(c.get("adapter.train_text_encoder")),
    "adapter.lokr_factor": lambda c: c.get("adapter.kind") == "lokr",
    "adapter.lokr_decompose_both": lambda c: c.get("adapter.kind") == "lokr",

    "train.compile_dynamic": lambda c: bool(c.get("train.compile")),
    "train.compile_regional": lambda c: bool(c.get("train.compile")),

    "quant.weights_dtype": lambda c: c.get("quant.mode") != "none",
    "quant.use_quantized_matmul": lambda c: c.get("quant.mode") != "none",
    "quant.skip_policy": lambda c: c.get("quant.mode") != "none",
    "quant.extra_skip": lambda c: c.get("quant.mode") != "none",
    "quant.quantize_text_conditioner": lambda c: c.get("quant.mode") != "none",
    "quant.quantize_text_encoder": lambda c: c.get("quant.mode") != "none",
    "quant.group_size": lambda c: c.get("quant.mode") != "none",
    "quant.dynamic_loss_threshold": lambda c: c.get("quant.mode") != "none",
    "quant.use_stochastic_rounding": lambda c: c.get("quant.mode") == "training",
}


def _has_texture_phase(c: dict) -> bool:
    return any(p.get("mode") == "texture" for p in (c.get("curriculum") or []))


# Derived from the schema rather than listed, so adding a texture key cannot leave it ungated --
# which is exactly what happened when the oversize cascade added four of them at once.
for _tex in (k for k in SPEC if k.startswith("dataset.texture.")):
    _RULES[_tex] = lambda c: _has_texture_phase(c)


def _adapter_targets(c: dict) -> set[str]:
    t = set(c.get("adapter.components") or [])
    if c.get("adapter.train_llm_adapter"):
        t.add("llm_adapter")
    if c.get("adapter.train_text_encoder"):
        t.add("text_encoder")
    return t


for _comp in ("self_attn", "cross_attn", "mlp", "adaln", "base", "llm_adapter", "text_encoder"):
    # Under an adapter, a per-component LR only reaches parameters that actually have one injected
    # -- which is exactly what `load_config` rejects. Full FT can address every component.
    _RULES[f"component_lr.{_comp}"] = (
        lambda c, comp=_comp: c.get("adapter.kind") == "none" or comp in _adapter_targets(c)
    )


class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anima Trainer")
        self.setMinimumSize(1100, 820)
        self.resize(1600, 1000)

        self.editors: dict[str, object] = {}
        self.rows: dict[str, tuple] = {}
        self.runner = None
        self._retired = None
        # Launches still to run after the current one -- one per dataset folder (see `_run_each`).
        self._queue: list = []
        self._applying = False
        self._current_path: Path | None = None
        # Configs opened from outside `configs/`, kept for the session so the preset
        # dropdown can list them alongside the built-ins.
        self._external: list[Path] = []

        self._setup_ui()
        self._on_gpu_selection()
        self._load_presets()

    # ---------------------------------------------------------------- build

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(0)

        nav = QtWidgets.QWidget()
        set_role(nav, "navigation")
        nav_lay = QtWidgets.QHBoxLayout(nav)
        nav_lay.setContentsMargins(0, 0, 8, 0)
        nav_lay.setSpacing(0)
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        nav_lay.addWidget(self.tab_bar, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        nav_lay.addStretch(1)
        nav_lay.addWidget(make_label("config", color=THEME.text_muted))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setMinimumWidth(240)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        nav_lay.addWidget(self.preset_combo)
        nav_lay.addWidget(make_btn("Open", self._open))
        nav_lay.addWidget(make_btn("Save", self._save))
        nav_lay.addWidget(make_btn("Save As", self._save_as))
        nav_lay.addWidget(make_btn("Reload", self._reload))
        main.addWidget(nav, 0)

        frame = QtWidgets.QFrame()
        set_role(frame, "mainFrame")
        frame_lay = QtWidgets.QVBoxLayout(frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QtWidgets.QStackedWidget()
        frame_lay.addWidget(self.stack, 1)

        for title, groups in LAYOUT:
            self.tab_bar.addTab(title)
            self.stack.addWidget(self._build_page(groups))

        self.metrics = LiveMetricsWidget()
        self.tab_bar.addTab("Live Metrics")
        self.stack.addWidget(self.metrics)

        console_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(console_page)
        cl.setContentsMargins(12, 12, 12, 12)
        self.console = VirtualConsoleWidget(visible_lines=900)
        cl.addWidget(self.console, 1)
        self.tab_bar.addTab("Console")
        self.stack.addWidget(console_page)

        self.tab_bar.currentChanged.connect(self.stack.setCurrentIndex)
        main.addWidget(frame, 1)
        main.addWidget(self._build_footer(), 0)

    def _build_page(self, groups):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        # Two columns: config forms are tall and narrow, and a single column wastes half a 1600px
        # window while forcing a scroll.
        cols = QtWidgets.QHBoxLayout(inner)
        cols.setContentsMargins(12, 12, 12, 12)
        cols.setSpacing(12)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        right.setSpacing(10)

        # Balance by control count rather than by group count, so one 10-key group does not sit
        # beside three 2-key ones.
        total = sum(len(keys) for _, keys in groups)
        running, target = 0, total / 2
        for title, keys in groups:
            gb, lay = group_box(title, QtWidgets.QVBoxLayout)
            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                   | QtCore.Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key in keys:
                self._add_field(form, key)
            lay.addLayout(form)
            (left if running < target else right).addWidget(gb)
            running += len(keys)

        left.addStretch(1)
        right.addStretch(1)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return page

    def _add_field(self, form, key):
        spec = SPEC[key]
        editor = spec.make()
        editor.changed.connect(self._on_edit)
        editor.widget.setToolTip(spec.tooltip)
        self.editors[key] = editor
        if spec.inline_label:
            form.addRow(editor.widget)
            self.rows[key] = (editor.widget,)
        else:
            label = QtWidgets.QLabel(spec.label)
            label.setToolTip(spec.tooltip)
            form.addRow(label, editor.widget)
            self.rows[key] = (label, editor.widget)

    def _build_footer(self):
        footer = QtWidgets.QWidget()
        set_role(footer, "footer")
        lay = QtWidgets.QVBoxLayout(footer)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Advisories sit below the status line and above the controls, because several of them are
        # about the GPU checkboxes right underneath -- the two worst traps are only visible once
        # the process count is known, which the config file cannot express.
        self.advice = QtWidgets.QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setVisible(False)
        lay.addWidget(self.advice)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        # Checkboxes, not a text field. This used to be a QLineEdit whose placeholder read "all",
        # which is exactly what someone types into it -- and `CUDA_VISIBLE_DEVICES=all` is not a
        # device list, so torch reported zero GPUs and trained a 2B model on the CPU without one
        # word of complaint. A control that cannot express an invalid value is the fix; the trainer
        # now also refuses to start on CPU, and `training_env` rejects a malformed list.
        gpus = detect_gpus()
        row.addWidget(make_label("GPUs", color=THEME.text_muted))
        self.gpu_boxes: dict[int, QtWidgets.QCheckBox] = {}
        for index, name in gpus:
            box = QtWidgets.QCheckBox(str(index))
            # On Windows only the first is preselected, because only one can be selected at all.
            box.setChecked(index == gpus[0][0] if not MULTI_GPU_SUPPORTED else True)
            box.setToolTip(f"GPU {index}: {name}" + ("" if MULTI_GPU_SUPPORTED else
                           "\n\nWindows: pick which GPU to train on. Multi-GPU is unavailable "
                           "here because DDP needs NCCL, which is Linux-only."))
            box.stateChanged.connect(self._on_gpu_selection)
            # One GPU means there is nothing to choose: locked on, so the control cannot be put
            # into a state that differs from what will actually run. On Windows the boxes stay
            # ENABLED even though only one may be picked -- choosing *which* GPU is still a real
            # choice, and disabling them would take that away to prevent a combination that the
            # radio behaviour below already prevents.
            if len(gpus) == 1:
                box.setEnabled(False)
            row.addWidget(box)
            self.gpu_boxes[index] = box

        # Process count is derived, never entered. Under DDP one process per selected GPU is the
        # only sensible pairing, and a spin box that could disagree with the selection is just a
        # way to launch 2 ranks onto 1 visible device.
        self.proc_label = make_label("", color=THEME.text_muted)
        self.proc_label.setToolTip(
            "One process per selected GPU. Every run goes through `accelerate launch`, single GPU "
            "included, so there is only one code path to reason about.\n\nDDP here is bounded by "
            "PCIe 3.0 x8: a full finetune all-reduces ~3.5GB per optimizer step. Raise gradient "
            "accumulation before adding GPUs.")
        row.addWidget(self.proc_label)

        row.addSpacing(16)
        self.audit_btn = make_btn("Audit dataset", self._audit)
        self.audit_btn.setToolTip(
            "Prints the source-size percentile table and proposes a resolution ladder. Reads image "
            "headers only -- no GPU, no model.")
        row.addWidget(self.audit_btn)
        self.cache_btn = make_btn("Cache latents", self._cache)
        self.cache_btn.setToolTip("Encode the dataset at the configured tier(s).")
        row.addWidget(self.cache_btn)
        self.cache_dry_btn = make_btn("Cache (dry run)", lambda: self._cache(dry=True))
        self.cache_dry_btn.setToolTip("Report the bucket plan and cache size; write nothing.")
        row.addWidget(self.cache_dry_btn)

        row.addStretch(1)
        self.start_btn = make_btn("Start Training", self._start, style="accent")
        self.start_btn.setFixedWidth(160)
        row.addWidget(self.start_btn)
        self.stop_btn = make_btn("Stop", self._stop, style="danger")
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setVisible(False)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        return footer

    # ---------------------------------------------------------------- gpu selection

    def selected_gpus(self) -> list[int]:
        return sorted(i for i, b in self.gpu_boxes.items() if b.isChecked())

    def gpu_arg(self) -> str:
        """CUDA_VISIBLE_DEVICES for the child. Empty when every GPU is selected -- setting it
        redundantly only creates another place for the index mapping to be wrong."""
        chosen = self.selected_gpus()
        if not chosen or len(chosen) == len(self.gpu_boxes):
            return ""
        return ",".join(str(i) for i in chosen)

    def num_processes(self) -> int:
        return max(1, len(self.selected_gpus()))

    def _on_gpu_selection(self):
        # Deselecting the last GPU would mean "train on nothing"; keep at least one checked.
        if not self.selected_gpus():
            for box in self.gpu_boxes.values():
                box.blockSignals(True)
                box.setChecked(True)
                box.blockSignals(False)

        # Windows: the boxes behave as radio buttons. DDP needs a NCCL backend and NCCL is
        # Linux-only, so more than one process cannot work here at all -- but *which* GPU to use is
        # still a real choice, so the controls stay live and the newest tick simply wins. Enforcing
        # it here rather than by disabling the boxes keeps that choice available.
        if not MULTI_GPU_SUPPORTED and len(self.selected_gpus()) > 1:
            keep = self.sender()
            chosen = keep if isinstance(keep, QtWidgets.QCheckBox) and keep.isChecked() else None
            for index, box in self.gpu_boxes.items():
                box.blockSignals(True)
                box.setChecked(box is chosen if chosen is not None
                               else index == self.selected_gpus()[0])
                box.blockSignals(False)

        n = self.num_processes()
        self.proc_label.setText(
            f"→ {n} process{'es' if n > 1 else ''}"
            + ("" if MULTI_GPU_SUPPORTED else "  (Windows: single GPU only)"))

    # ---------------------------------------------------------------- config I/O

    def _load_presets(self, select: Path | None = None):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        known = set()
        for path in sorted(CONFIG_DIR.glob("*.toml")):
            self.preset_combo.addItem(path.stem, str(path))
            known.add(path.resolve())
        # Configs opened from outside `configs/` stay in the list for the session, marked, so the
        # dropdown does not silently drop the file you are actually working on the moment the list
        # is rebuilt (which `Save As` does).
        for path in self._external:
            if path.resolve() not in known and path.exists():
                self.preset_combo.addItem(f"{path.stem}  —  {path.parent}", str(path))
        self.preset_combo.blockSignals(False)

        if select is not None:
            for i in range(self.preset_combo.count()):
                if Path(self.preset_combo.itemData(i)).resolve() == select.resolve():
                    self.preset_combo.setCurrentIndex(i)
                    self._on_preset_changed(i)
                    return
        if self.preset_combo.count():
            self._on_preset_changed(0)
        else:
            self._apply(bridge.defaults())

    def _open(self):
        """Load a TOML from anywhere on disk, not just `configs/`.

        The path is remembered, so pressing Start writes back to the file you opened rather than
        quietly forking a copy into `configs/` -- which would leave two files with the same name
        and no way to tell which one a run actually used.
        """
        start = usable_dialog_start(
            str(self._current_path) if self._current_path else None, str(CONFIG_DIR))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open config", start, "TOML configs (*.toml);;All files (*)")
        if not path:
            return
        p = Path(path)
        if p.resolve() not in {q.resolve() for q in self._external}:
            self._external.append(p)
        self._load_presets(select=p)

    def _on_preset_changed(self, index):
        path = self.preset_combo.itemData(index)
        if not path:
            return
        self._current_path = Path(path)
        try:
            flat = bridge.flatten(bridge.read_toml(path))
        except Exception as exc:
            self.log(f"ERROR reading {path}: {exc}")
            return
        merged = bridge.defaults()
        merged.update(flat)
        self._apply(merged)
        self.log(f"Loaded {path}")

    def _reload(self):
        self._on_preset_changed(self.preset_combo.currentIndex())

    def _apply(self, flat):
        self._applying = True
        try:
            for key, editor in self.editors.items():
                editor.set(flat.get(key))
        finally:
            self._applying = False
        self._refresh()

    def collect(self) -> dict:
        return {key: editor.get() for key, editor in self.editors.items()}

    def _managed_target(self, flat: dict) -> Path:
        """Where a Save/Start should land: `configs/<run_name>.toml`.

        The config file is *named by* `run_name`. Keeping the two in sync by hand is the thing this
        removes -- and they were never independent anyway, since `run_name` already decides the
        checkpoint stem (`out_dir/<run_name>.safetensors`) and the copy of the TOML the trainer
        writes beside it. One name, three places.
        """
        return CONFIG_DIR / f"{safe_stem(flat.get('train.run_name'))}.toml"

    def _persist(self, flat: dict) -> Path | None:
        """Write `flat` to the managed config and return where it went, or None if cancelled.

        Two rules, both about not destroying a file the user did not mean to touch:

        **A config opened from outside `configs/` is never written back to.** Saving copies it in
        instead, and the original is left exactly as it was. Opening someone's config to read its
        settings, tweaking a knob and pressing Start used to rewrite their file in place -- and
        `write_toml` regenerates the TOML, so their comments went with it.

        **Changing `run_name` writes a NEW config and leaves the old one in place.** The filename
        still follows `run_name`, so the two names never drift -- but following a name is not a
        reason to delete a file. Deriving a second run from an existing config is the ordinary way
        to use this GUI, and the previous behaviour moved the original out from under the user:
        load `modan-ft.toml`, rename to `modan-ft-v2`, Save, and `modan-ft.toml` was gone.

        **A save never lands on a *different* existing config without asking.** `run_name` collides
        with more than this file: two runs sharing it also share `out_dir/<run_name>.safetensors`,
        so the checkpoints overwrite each other too. That is worth a dialog.
        """
        target = self._managed_target(flat)
        src = self._current_path
        src_resolved = src.resolve() if src is not None and src.exists() else None
        external = src is not None and src.parent.resolve() != CONFIG_DIR.resolve()

        if target.exists() and (src_resolved is None or target.resolve() != src_resolved):
            answer = QtWidgets.QMessageBox.question(
                self, "Config already exists",
                f"{target.name} already exists in configs/.\n\n"
                f"run_name is {flat.get('train.run_name')!r}, which also names the checkpoints "
                f"(output/{flat.get('train.run_name')}.safetensors). Two runs sharing it overwrite "
                f"each other's output as well as this file.\n\nOverwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.log(f"Save cancelled -- {target.name} already exists")
                return None

        if external:
            self.log(f"{src.name} was opened from {src.parent} -- saving a copy to "
                     f"{target.name}, original untouched")
        elif src_resolved is not None and target.resolve() != src_resolved:
            self.log(f"run_name changed -- saved as {target.name}, {src.name} left as it was")

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        bridge.write_toml(target, flat)
        self._current_path = target
        return target

    def _save(self):
        path = self._persist(self.collect())
        if path is None:
            return
        self.log(f"Saved {path}")
        self._load_presets(select=path)

    def _save_as(self):
        # A real file dialog rather than a name prompt, so a config can be saved beside the dataset
        # or the run it belongs to. Defaults to `configs/`, which is where it used to be forced.
        start = usable_dialog_start(
            str(self._current_path) if self._current_path else None, str(CONFIG_DIR))
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config as", start, "TOML configs (*.toml)")
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix != ".toml":
            path = path.with_suffix(".toml")
        if path.parent.resolve() != CONFIG_DIR.resolve():
            self._external.append(path)
        bridge.write_toml(path, self.collect())
        self._current_path = path
        self._on_gpu_selection()
        self._load_presets(select=path)
        idx = self.preset_combo.findData(str(path))
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.log(f"Saved {path}")

    # ---------------------------------------------------------------- reactivity

    def _on_edit(self):
        if not self._applying:
            self._refresh()

    def _refresh(self):
        flat = self.collect()
        for key, rule in _RULES.items():
            editor = self.editors.get(key)
            if editor is None:
                continue
            try:
                on = bool(rule(flat))
            except Exception:
                on = True
            editor.set_enabled(on)
            for widget in self.rows.get(key, ()):
                widget.setEnabled(on)

        ok, err = bridge.validate(flat)
        running = self.runner is not None and self.runner.isRunning()

        # Advisories are evaluated against the GPU SELECTION, not just the config -- the two worst
        # ones cannot be seen from the file alone. An "error" advisory blocks Start, because the
        # trainer would refuse the same combination anyway; blocking here turns a 30-second wait
        # and a traceback into an immediate, readable no.
        notes = bridge.advisories(flat, self.num_processes())
        blocking = [m for lvl, m in notes if lvl == "error"]
        self.start_btn.setEnabled(ok and not running and not blocking)
        for btn in (self.cache_btn, self.cache_dry_btn, self.audit_btn):
            btn.setEnabled(not running)
        if ok:
            self.status.setText(
                f"<span style='color:{SUCCESS}'>&#10003;</span> {bridge.summarize(flat)}")
        else:
            self.status.setText(f"<span style='color:{DANGER}'>{err}</span>")

        colour = {"error": DANGER, "warn": WARN, "note": THEME.text_muted}
        mark = {"error": "&#10007;", "warn": "&#9888;", "note": "&#8226;"}
        self.advice.setText("<br>".join(
            f"<span style='color:{colour[lvl]}'>{mark[lvl]} {msg}</span>" for lvl, msg in notes))
        self.advice.setVisible(bool(notes))

    # ---------------------------------------------------------------- running

    def log(self, text, replace_last=False):
        self.console.append_line(text, replace_last=replace_last)

    def _handle_output(self, line, is_progress):
        self.log(line, replace_last=is_progress and self._last_was_progress)
        self._last_was_progress = is_progress

    _last_was_progress = False

    def _run(self, launch, training=True):
        if self.runner is not None and self.runner.isRunning():
            self.log("A process is already running.")
            return
        try:
            env = training_env(self.gpu_arg())
        except ValueError as exc:            # unreachable from the checkboxes; a guard, not a path
            self.log(f"CONFIG ERROR: {exc}")
            return
        self.runner = ProcessRunner(launch, str(PROJECT_ROOT), env)
        self.runner.logSignal.connect(self.log)
        self.runner.errorSignal.connect(self.log)
        self.runner.progressSignal.connect(self._handle_output)
        self.runner.metricsSignal.connect(self.metrics.parse_and_update)
        self.runner.finishedSignal.connect(self._finished)
        self.log("\n" + "=" * 60 + f"\n{launch.label}\n" + " ".join(launch.argv) + "\n" + "=" * 60)
        self.tab_bar.setCurrentIndex(self.tab_bar.count() - (2 if training else 1))
        if training:
            self.metrics.clear_data()
            prevent_sleep(True)
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self._refresh()
        self.runner.start()

    def _finished(self, code):
        prevent_sleep(False)
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.log(f"Process finished with exit code {code}")
        # `self.runner = None` here would drop the last reference to the QThread *while it is still
        # emitting* `finishedSignal`, and PySide then tears the C++ object down mid-emission: every
        # slot connected after this one is silently skipped. Observed, not theoretical -- it is how
        # this was found. Hand the object to the next event-loop turn instead, by which time the
        # emit has returned.
        self._retired, self.runner = self.runner, None
        QtCore.QTimer.singleShot(0, lambda: setattr(self, "_retired", None))
        self._refresh()

        if self._queue:
            # A folder that failed does not cancel the rest: auditing three directories and having
            # the first one fail should still report on the other two. `_run` refuses to start
            # while a runner is live, so this waits for the next event-loop turn -- `self.runner`
            # is already None above, but the QThread it referred to is still finishing.
            nxt = self._queue.pop(0)
            QtCore.QTimer.singleShot(0, lambda: self._run(nxt, training=False))

    def _start(self):
        flat = self.collect()
        ok, err = bridge.validate(flat)
        if not ok:
            self.log(f"CONFIG ERROR: {err}")
            return
        # The trainer reads a file, so what runs is exactly what is on disk -- no hidden in-memory
        # variant that a later "why did it use that value" cannot account for. It goes through
        # `_persist` for the same reason Save does: starting a run must not rewrite a config that
        # was opened from somewhere else on disk.
        path = self._persist(flat)
        if path is None:
            return
        self._load_presets(select=path)
        self._run(train_launch(path, self.num_processes()))

    def _stop(self):
        # Drop the rest of the queue first: Stop means stop, not "skip to the next folder".
        if self._queue:
            self.log(f"Stopped -- {len(self._queue)} queued folder(s) skipped.")
            self._queue = []
        if self.runner is not None and self.runner.isRunning():
            self.runner.stop()

    def _dataset_paths(self, flat: dict) -> list[str]:
        """Every directory this config actually reads images from, in config order.

        `path` and `subsets` are mutually exclusive in the loader, so a config with subset rows has
        no single dataset path -- and both tools below take exactly one directory per invocation.

        Empty entries are dropped rather than passed through, because `Path("")` is `Path(".")` and
        passes `.is_dir()`: an empty box would silently audit or cache the repo root instead of
        failing. That was the live bug here -- with subsets configured, `dataset.path` is unset, so
        both buttons ran against `.`.
        """
        subsets = flat.get("dataset.subsets") or []
        raw = [s.get("path") for s in subsets] if subsets else [flat.get("dataset.path")]
        return [p for p in (str(x or "").strip() for x in raw) if p]

    def _run_each(self, launches: list, what: str) -> None:
        """Run one launch per dataset directory, in turn.

        Sequential rather than concurrent: caching is GPU-bound, and interleaving the output of
        several would make the per-folder reports unreadable.
        """
        if not launches:
            return
        self._queue = list(launches[1:])
        if self._queue:
            self.log(f"{what} {len(launches)} dataset folders in turn.")
        self._run(launches[0], training=False)

    def _cache(self, dry=False):
        c = self.collect()
        paths = self._dataset_paths(c)
        if not paths:
            self.log("Nothing to cache -- set a Dataset path, or give the subset rows a folder.")
            return
        tiers = c.get("dataset.resolutions") or [c.get("dataset.resolution") or 1024]
        self._run_each([cache_launch(
            p, c.get("train.model_path") or "", tiers,
            min_bucket_reso=c.get("dataset.min_bucket_reso") or 256,
            max_bucket_reso=c.get("dataset.max_bucket_reso") or 1920,
            bucket_reso_steps=c.get("dataset.bucket_reso_steps") or 64,
            upscale=not c.get("dataset.bucket_no_upscale", True),
            multires_training=bool(c.get("dataset.multires_training")),
            dry_run=dry, gpus=self.gpu_arg(),
        ) for p in paths], "Caching")

    def _audit(self):
        c = self.collect()
        paths = self._dataset_paths(c)
        if not paths:
            self.log("Nothing to audit -- set a Dataset path, or give the subset rows a folder.")
            return
        steps = c.get("dataset.bucket_reso_steps") or 64
        self._run_each([audit_launch(p, steps) for p in paths], "Auditing")

    def closeEvent(self, event):
        if self.runner is not None and self.runner.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "Training is running",
                "Stop the running process and quit?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        prevent_sleep(False)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName("Anima Trainer")
    window = TrainingGUI()
    window.show()
    sys.exit(app.exec())
