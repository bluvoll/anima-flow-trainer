"""Live graphs, driven by parsing the trainer's stdout.

The plumbing (batched pending queue, single-shot repaint timer, `GraphPanel`) follows Aozora's
`LiveMetricsWidget`; the parsing is ours because the log line is ours.

One line carries everything:

    e3 step 1410/3000 (47%)  loss 0.0712 (mse 0.0650 + hf 0.0062)  ot 0.83  lr 2.71e-05  \
        1.80s/it  peak 8.1GB  eta 47m34s

The bracketed decomposition, `ot`, and `eta` are all optional -- `hf_scale = 0` removes the first,
batch-1 buckets and `use_ot = false` remove the second, and the first logged step has no ETA yet.
Under DDP `peak` becomes `8.1/8.1GB`, one figure per rank, because the ranks are not symmetric and
reporting only rank 0 would hide which rank is the batch-size ceiling.
"""

from __future__ import annotations

import re
from collections import deque

from PySide6 import QtCore, QtWidgets

from .widgets import (
    ACCENT,
    ACCENT2,
    GraphPanel,
    SUCCESS,
    THEME,
    WARN,
    group_box,
    make_btn,
    set_role,
)

STEP_RE = re.compile(
    r"^e(?P<epoch>\d+)\s+step\s+(?P<step>\d+)/(?P<total>\d+)"
    r"(?:\s+\((?P<pct>\d+)%\))?"
    r"\s+loss\s+(?P<loss>[\d.eE+-]+)"
    r"(?:\s+\(mse\s+(?P<mse>[\d.eE+-]+)\s+\+\s+hf\s+(?P<hf>[\d.eE+-]+)\))?"
    r"(?:\s+ot\s+(?P<ot>[\d.]+))?"
    # Slash-joined when [component_lr] gives groups different rates, exactly like
    # `peak A/B GB` does for per-rank memory: "lr 4.00e-05/1.00e-05/8.00e-06".
    r"\s+lr\s+(?P<lr>[\d.eE+-]+(?:/[\d.eE+-]+)*)"
    r"\s+(?P<sit>[\d.]+)s/it"
    r"\s+peak\s+(?P<peak>[\d./]+)GB"
    r"(?:\s+eta\s+(?P<eta>\S+))?"
)

# Startup summary lines worth surfacing on the metrics tab rather than leaving in the console.
REPORT_RE = re.compile(r"^(dataset|steps|mode|params|quant|flow|sched|compile|optim|hf loss)\s{2,}")


class LiveMetricsWidget(QtWidgets.QWidget):
    """Four graphs plus a header of latest values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_points = 60000
        self.loss_ema_beta = 0.95
        self.pending = deque()
        self.max_pending_per_tick = 2000
        self.graphs: dict[str, dict] = {}
        self._reset_state()

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._flush)
        self._setup_ui()

    # ------------------------------------------------------------------ state

    def _reset_state(self):
        self.loss_ema = None
        self.latest = {}
        self.report_lines = []

    def clear_data(self):
        self._reset_state()
        self.pending.clear()
        for g in self.graphs.values():
            g["widget"].clear_all_data()
        self._set_header()
        self.report_box.setPlainText("")

    # ------------------------------------------------------------------ ui

    def _add_graph(self, name, title, y_label, lines):
        gb, lay = group_box("")
        set_role(gb, "flat")
        graph = GraphPanel(title, y_label)
        graph.set_fill(True)
        self.graphs[name] = {"widget": graph, "lines": {}}
        for label, color, style in lines:
            idx = graph.add_line(color, label, self.max_points, 2, style)
            self.graphs[name]["lines"][label] = idx
        lay.addWidget(graph, 1)
        return gb

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.header = QtWidgets.QLabel("")
        self.header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        top.addWidget(self.header, 1)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedWidth(90)
        top.addWidget(self.pause_btn)
        top.addWidget(make_btn("Clear", self.clear_data))
        main.addLayout(top)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        # The loss decomposition shares an axis on purpose: total = mse + hf exactly, so seeing the
        # three together is the only way to read what the hf term is actually contributing.
        grid.addWidget(self._add_graph("loss", "Loss", "loss", [
            ("Loss", ACCENT, "solid"),
            ("Loss EMA", ACCENT2, "solid"),
            ("MSE", THEME.text_muted, "dash"),
            ("HF", SUCCESS, "dash"),
        ]), 0, 0)
        grid.addWidget(self._add_graph("lr", "Learning rate", "lr", [
            ("LR", ACCENT, "solid"),
        ]), 0, 1)
        grid.addWidget(self._add_graph("speed", "Step time", "s/it", [
            ("s/it", ACCENT, "solid"),
        ]), 1, 0)
        grid.addWidget(self._add_graph("mem", "Peak VRAM", "GB", [
            ("rank 0", ACCENT, "solid"),
            ("rank 1", WARN, "solid"),
        ]), 1, 1)
        main.addLayout(grid, 1)

        rb, rlay = group_box("Run summary")
        self.report_box = QtWidgets.QPlainTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setFixedHeight(120)
        rlay.addWidget(self.report_box)
        main.addWidget(rb)

        self._set_header()

    def _set_header(self):
        def cell(label, value, color=None):
            c = color or THEME.text
            return (f"<span style='color:{THEME.text_muted}'>{label}</span> "
                    f"<span style='color:{c};font-weight:600'>{value}</span>")

        L = self.latest
        peak = L.get("peak")
        parts = [
            cell("step", f"{L.get('step', '-')}/{L.get('total', '-')}"),
            cell("epoch", L.get("epoch", "-")),
            cell("loss", f"{L['loss']:.4f}" if "loss" in L else "-", ACCENT),
            cell("lr", f"{L['lr']:.2e}" if "lr" in L else "-"),
            cell("s/it", f"{L['sit']:.2f}" if "sit" in L else "-"),
            cell("peak", "/".join(f"{p:.1f}" for p in peak) + " GB" if peak else "-"),
            cell("eta", L.get("eta", "-"), SUCCESS),
        ]
        if "hf" in L:
            parts.insert(3, cell("hf", f"{L['hf']:.4f}"))
        if "ot" in L:
            parts.append(cell("ot", f"{L['ot']:.2f}"))
        self.header.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))

    # ------------------------------------------------------------------ ingest

    def parse_and_update(self, text):
        if self.pause_btn.isChecked():
            return
        if REPORT_RE.match(text):
            self.report_lines.append(text)
            self.report_box.setPlainText("\n".join(self.report_lines[-12:]))
            return
        m = STEP_RE.match(text.strip())
        if not m:
            return
        d = m.groupdict()
        step = int(d["step"])
        rec = {
            "epoch": int(d["epoch"]),
            "step": step,
            "total": int(d["total"]),
            "loss": float(d["loss"]),
            # The graph plots the first group's LR; the schedule shape is identical across
            # groups (they differ by a constant ratio), so one curve describes all of them.
            "lr": float(d["lr"].split("/")[0]),
            "lrs": [float(v) for v in d["lr"].split("/") if v],
            "sit": float(d["sit"]),
            "peak": [float(p) for p in d["peak"].split("/") if p],
        }
        if d["mse"] is not None:
            rec["mse"] = float(d["mse"])
            rec["hf"] = float(d["hf"])
        if d["ot"] is not None:
            rec["ot"] = float(d["ot"])
        if d["eta"] is not None:
            rec["eta"] = d["eta"]
        self.pending.append(rec)
        self.latest = rec
        if not self.timer.isActive():
            self.timer.start(0)

    def _flush(self):
        if not self.pending:
            self.timer.stop()
            return
        processed = 0
        while self.pending and processed < self.max_pending_per_tick:
            rec = self.pending.popleft()
            processed += 1
            step = rec["step"]
            g = self.graphs
            self._append("loss", "Loss", step, rec["loss"])
            self.loss_ema = (rec["loss"] if self.loss_ema is None
                             else self.loss_ema_beta * self.loss_ema
                             + (1 - self.loss_ema_beta) * rec["loss"])
            self._append("loss", "Loss EMA", step, self.loss_ema)
            if "mse" in rec:
                self._append("loss", "MSE", step, rec["mse"])
                self._append("loss", "HF", step, rec["hf"])
            self._append("lr", "LR", step, rec["lr"])
            self._append("speed", "s/it", step, rec["sit"])
            for i, p in enumerate(rec["peak"][:2]):
                self._append("mem", f"rank {i}", step, p)
            # A single-rank run must not draw a flat rank-1 line at zero.
            if len(rec["peak"]) < 2:
                g["mem"]["widget"].set_line_visible(g["mem"]["lines"]["rank 1"], False)
        self._set_header()
        if self.pending:
            self.timer.start(0)
        else:
            self.timer.stop()

    def _append(self, graph, line, x, y):
        g = self.graphs[graph]
        g["widget"].append_data(g["lines"][line], x, y)
