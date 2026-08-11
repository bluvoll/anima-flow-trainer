"""Turn the single-file Anima release into a diffusers-format repo.

    # headless
    venv/bin/python -m anima.tools.convert_model \
        --anima anima-base-v1.0.safetensors \
        --qwen qwen_3_06b_base.safetensors \
        --vae qwen_image_vae.safetensors \
        --out /path/to/anima-diffusers

    # or pick the files in a window
    venv/bin/python -m anima.tools.convert_model --gui

The conversion itself lives in `anima/modeling/to_diffusers.py`; this is only a front end, so the
CLI and the GUI cannot diverge in what they produce.

**Why PySide6 and not tkinter.** tkinter would be the lighter choice -- it is stdlib, so a
convert-only user would not need the `gui` extra. It is not available in practice: CPython built
by pyenv (and most distro-packaged builds) omits `_tkinter` unless Tk headers were present at
build time, and this trainer's venv is exactly that case. The converter has to run inside the venv
to reach torch and safetensors, so a tkinter front end would fail to start on the machine it was
written for. PySide6 is already installed by both installers.
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from ..modeling.to_diffusers import DEFAULT_TOKENIZER_REPO, TOKENIZER_NONE, convert_to_diffusers


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="anima.tools.convert_model",
        description="Convert single-file Anima checkpoints into a diffusers-format repo.",
    )
    ap.add_argument("--anima", help="anima-base-v1.0.safetensors (transformer + text conditioner)")
    ap.add_argument("--qwen", help="Qwen3 0.6B safetensors")
    ap.add_argument("--vae", help="Qwen-Image VAE safetensors")
    ap.add_argument("--out", help="output directory (must be empty or absent)")
    ap.add_argument("--tokenizers", default=None, metavar="PATH|REPO_ID|none",
                    help=f"where tokenizer/ and t5_tokenizer/ come from: a local directory, a hub "
                         f"repo id, or 'none' to skip. Default: download from "
                         f"{DEFAULT_TOKENIZER_REPO} (~13MB, cached, so later conversions are "
                         f"offline). ~13MB of vocabulary that cannot be derived from the weights, "
                         f"and REQUIRED to train: both are used on every step, because text "
                         f"embeddings are not cached (tag shuffling and caption dropout change "
                         f"the caption each epoch).")
    ap.add_argument("--no-modular-index", action="store_true",
                    help="skip modular_model_index.json. The trainer never reads it -- it loads "
                         "each component by subfolder -- and it embeds absolute paths.")
    ap.add_argument("--gui", action="store_true", help="pick the files in a window instead")
    args = ap.parse_args(argv)

    if args.gui or not any((args.anima, args.qwen, args.vae, args.out)):
        return _gui()

    missing = [f"--{n}" for n in ("anima", "qwen", "vae", "out") if not getattr(args, n)]
    if missing:
        ap.error(f"missing {', '.join(missing)} (or pass --gui)")

    try:
        convert_to_diffusers(args.anima, args.qwen, args.vae, args.out,
                             tokenizer_src=args.tokenizers,
                             write_modular_index=not args.no_modular_index,
                             progress=lambda m: print(f"  {m}", flush=True))
    except Exception as exc:                       # noqa: BLE001 - a CLI reports, never traces
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _gui() -> int:
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError:
        print("The GUI needs PySide6, which is installed by install.sh / install.bat.\n"
              "Either re-run the installer, or convert headlessly:\n"
              "  venv/bin/python -m anima.tools.convert_model --anima ... --qwen ... "
              "--vae ... --out ...", file=sys.stderr)
        return 1

    from ..gui.widgets import STYLESHEET, make_btn, usable_dialog_start

    class Worker(QtCore.QThread):
        line = QtCore.Signal(str)
        done = QtCore.Signal(bool, str)

        def __init__(self, kwargs):
            super().__init__()
            self.kwargs = kwargs

        def run(self):
            try:
                out = convert_to_diffusers(progress=self.line.emit, **self.kwargs)
                self.done.emit(True, str(out))
            except Exception as exc:               # noqa: BLE001 - surfaced in the log pane
                self.done.emit(False, f"{type(exc).__name__}: {exc}")

    class Window(QtWidgets.QWidget):
        # (attribute, label, dialog caption, filter). "" filter means "pick a directory".
        FIELDS: ClassVar[list[tuple[str, str, str, str]]] = [
            ("anima", "Anima checkpoint",
             "transformer + text conditioner, one file (685 tensors)", "Safetensors (*.safetensors)"),
            ("qwen", "Qwen3 0.6B", "the text encoder", "Safetensors (*.safetensors)"),
            ("vae", "Qwen-Image VAE", "the VAE", "Safetensors (*.safetensors)"),
            ("tokenizers", "Tokenizers (optional)",
             (f"leave blank to download from {DEFAULT_TOKENIZER_REPO} (~13MB, cached). Or pick a "
              f"local repo, or type '{TOKENIZER_NONE}' to skip. REQUIRED to train: both run on "
              f"every step, since text embeddings are not cached (shuffling/dropout change the "
              f"caption)."),
             ""),
            ("out", "Output folder", "must be empty or not yet exist", ""),
        ]

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Anima -> diffusers converter")
            self.setStyleSheet(STYLESHEET)
            self.resize(820, 560)
            self.edits: dict[str, QtWidgets.QLineEdit] = {}
            self.worker = None

            lay = QtWidgets.QVBoxLayout(self)
            lay.setSpacing(10)
            head = QtWidgets.QLabel(
                "Converts the files Anima ships as into the diffusers layout this trainer reads.\n"
                "Nothing is modified in place -- the three inputs are only read.")
            head.setWordWrap(True)
            lay.addWidget(head)

            form = QtWidgets.QGridLayout()
            form.setColumnStretch(1, 1)
            for row, (attr, label, hint, filt) in enumerate(self.FIELDS):
                lbl = QtWidgets.QLabel(label)
                lbl.setToolTip(hint)
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText(hint)
                edit.setToolTip(hint)
                btn = make_btn("Browse", lambda _=False, a=attr, c=label, f=filt: self._pick(a, c, f))
                btn.setFixedWidth(90)
                form.addWidget(lbl, row, 0)
                form.addWidget(edit, row, 1)
                form.addWidget(btn, row, 2)
                self.edits[attr] = edit
            lay.addLayout(form)

            self.index_box = QtWidgets.QCheckBox(
                "Also write modular_model_index.json (only needed for the diffusers inference "
                "pipeline; the trainer ignores it)")
            self.index_box.setChecked(True)
            lay.addWidget(self.index_box)

            self.log = QtWidgets.QPlainTextEdit(readOnly=True)
            self.log.setPlaceholderText("Progress appears here.")
            lay.addWidget(self.log, 1)

            row = QtWidgets.QHBoxLayout()
            self.status = QtWidgets.QLabel("")
            self.status.setWordWrap(True)
            self.go = make_btn("Convert", self._convert)
            self.go.setMinimumWidth(140)
            row.addWidget(self.status, 1)
            row.addWidget(self.go)
            lay.addLayout(row)

        def _pick(self, attr, caption, filt):
            start = usable_dialog_start(self.edits[attr].text())
            if filt:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start, filt)
            else:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, caption, start)
            if path:
                self.edits[attr].setText(path)

        def _convert(self):
            if self.worker is not None and self.worker.isRunning():
                return
            vals = {k: e.text().strip() for k, e in self.edits.items()}
            missing = [lbl for attr, lbl, _, _ in self.FIELDS
                       if attr != "tokenizers" and not vals[attr]]
            # Blank now means "download them", so only an explicit `none` is worth confirming --
            # that is the one path that produces a repo which cannot train.
            if not missing and vals["tokenizers"].strip().lower() == TOKENIZER_NONE:
                self.status.setText(
                    "Skipping tokenizers -- the result will hold every weight but cannot train. "
                    "Press Convert again to do it anyway.")
                if not getattr(self, "_warned_tokenizers", False):
                    self._warned_tokenizers = True
                    return
            self._warned_tokenizers = False
            if missing:
                self.status.setText(f"Still needed: {', '.join(missing)}")
                return
            self.log.clear()
            self.status.setText("")
            self.go.setEnabled(False)
            self.worker = Worker({
                "anima_path": vals["anima"], "qwen_path": vals["qwen"],
                "vae_path": vals["vae"], "out_dir": vals["out"],
                "tokenizer_src": vals["tokenizers"] or None,
                "write_modular_index": self.index_box.isChecked(),
            })
            self.worker.line.connect(lambda m: self.log.appendPlainText(m))
            self.worker.done.connect(self._finished)
            self.worker.start()

        def _finished(self, ok, detail):
            self.go.setEnabled(True)
            if ok:
                self.log.appendPlainText("")
                self.status.setText(f"Done. Point train.model_path at {detail}")
            else:
                self.log.appendPlainText(f"\nFAILED: {detail}")
                self.status.setText("Failed -- see the log above.")

        def closeEvent(self, event):
            if self.worker is not None and self.worker.isRunning():
                self.worker.wait(30_000)
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    return _cli(argv)


if __name__ == "__main__":
    sys.exit(main())
