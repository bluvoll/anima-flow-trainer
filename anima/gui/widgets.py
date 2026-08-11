"""Reusable Qt chrome, vendored from Aozora Trainer (Apache-2.0) with only the
Aozora-specific feature code removed.

Upstream: https://github.com/Hysocs/Aozora_Trainer -- `gui/gui.py`, commit b9efef2.
See LICENSE.Aozora in this directory and NOTICE at the repo root.

What lives here is deliberately generic: the widget factories, the themed
splitter/console/folder-dialog, and `GraphPanel`. Nothing in this module knows
what a training config looks like -- that is `schema.py` and `app.py`.
"""

import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
import zlib
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QFileInfo
from PySide6.QtWidgets import QFileIconProvider

from .theme import THEME, make_stylesheet, set_role

# Compatibility aliases keep feature code readable while every value comes from
# the single semantic palette in gui_theme.py.
DARK_BG = THEME.window
PANEL_BG = THEME.surface_raised
NESTED_GROUP_BG = THEME.chart
GRAPH_BG = THEME.canvas
BORDER = THEME.border
ACCENT = THEME.accent
ACCENT2 = THEME.accent_alt
TEXT_PRI = THEME.text
TEXT_SEC = THEME.text_muted
DANGER = THEME.danger
SUCCESS = THEME.success
WARN = THEME.warning
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")


def platform_path(value):
    """Normalize a user/config path for the host OS without resolving symlinks."""
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        return ""
    if not IS_WINDOWS:
        text = text.replace("\\", "/")
    return os.path.normpath(text)


def usable_dialog_start(value, fallback=None):
    candidate = platform_path(value)
    if candidate and os.path.exists(candidate):
        return candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
    fallback = platform_path(fallback)
    if fallback and os.path.isdir(fallback):
        return fallback
    return QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.HomeLocation) or str(Path.home())


_STEM_EXTRA = set("-_.")
# Windows resolves these as DOS devices whatever the extension, so `CON.toml` is not a file --
# opening it succeeds and writes to the console. Case-insensitive, and still reserved with a
# suffix. ext4 does not care, but a config is meant to be portable between the two.
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL",
                 *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_stem(name, fallback="gui_run"):
    """`train.run_name` -> a filename stem that is legal on both platforms.

    `run_name` is free text and reaches the filesystem three ways -- the config file, the
    checkpoint stem and the state directory -- so a name with a slash in it does not fail at save
    time, it fails several hours later at the first checkpoint. Windows additionally rejects
    `<>:"/\\|?*`, trailing dots and trailing spaces.

    Accented and non-Latin letters are KEPT: `str.isalnum()` is unicode-aware and both NTFS and
    ext4 store them fine, so folding them to ASCII would mangle perfectly good names (`sesión` ->
    `sesi-n`) for no safety gain. Only genuinely illegal characters collapse, to a single `-`, so
    the common case stays readable: `my run 2` -> `my-run-2`.
    """
    text = str(name or "").strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in _STEM_EXTRA:
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    # A leading dot hides the file on unix; a trailing dot or dash is legal but reads as a typo,
    # and Windows silently strips trailing dots, which would make two names collide.
    stem = "".join(out).strip("-.")
    if stem.upper() in _WIN_RESERVED:
        stem += "-run"
    return stem or fallback


def report_gui_exception(context, exc):
    print(f"GUI warning: {context}: {type(exc).__name__}: {exc}", file=sys.stderr)

TEXT_MUTED = THEME.text_disabled
BORDER_MUTED = THEME.border_muted

STYLESHEET = make_stylesheet()

DEFAULT_THEME_COLORS = {
    "primary": "#7c6af7",
    "secondary": "#5839b0",
}
CLAY_CHART_COLORS = {
    "loss_ema": "#c2ad55",
    "sigma_samples": "#c1845b",
    "mean_loss_by_sigma": "#c2ad55",
    "learning_rate": "#c1845b",
    "gradient_norm": "#c1845b",
    "clipped_gradient": "#c2ad55",
    "lr_scheduler": "#c1845b",
    "ticket_allocator": "#c2ad55",
    "loss_weight": "#c2ad55",
}
PURPLE_CHART_COLORS = {
    "loss_ema": "#3d8943",
    "sigma_samples": "#7c6af7",
    "mean_loss_by_sigma": "#7c6af7",
    "learning_rate": "#45aeb4",
    "gradient_norm": "#c1845b",
    "clipped_gradient": "#c2ad55",
    "lr_scheduler": "#45aeb4",
    "ticket_allocator": "#7c6af7",
    "loss_weight": "#3d8943",
}
DEFAULT_CHART_COLORS = PURPLE_CHART_COLORS
THEME_COLOR_PRESETS = [
    ("Clay & Ochre", "#c1845b", "#c2ad55", CLAY_CHART_COLORS),
    ("Violet & Amethyst", "#7c6af7", "#5839b0", PURPLE_CHART_COLORS),
]

_sleep_inhibitor_process = None


def prevent_sleep(enable=True):
    """Toggle OS sleep inhibition while a training process is active."""
    global _sleep_inhibitor_process
    try:
        if IS_WINDOWS:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            kernel32 = ctypes.windll.kernel32
            state = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if enable else ES_CONTINUOUS
            return bool(kernel32.SetThreadExecutionState(state))

        if IS_LINUX:
            if not enable:
                if _sleep_inhibitor_process and _sleep_inhibitor_process.poll() is None:
                    _sleep_inhibitor_process.terminate()
                    try:
                        _sleep_inhibitor_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        _sleep_inhibitor_process.kill()
                        _sleep_inhibitor_process.wait()
                _sleep_inhibitor_process = None
                return True

            if _sleep_inhibitor_process and _sleep_inhibitor_process.poll() is None:
                return True
            inhibitor = shutil.which("systemd-inhibit")
            sleeper = shutil.which("sleep")
            if not inhibitor or not sleeper:
                return False
            _sleep_inhibitor_process = subprocess.Popen(
                [inhibitor, "--what=sleep:idle", "--mode=block",
                 "--why=Aozora model training is active", sleeper, "infinity"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return _sleep_inhibitor_process.poll() is None
        return False
    except Exception as exc:
        report_gui_exception("could not change OS sleep inhibition", exc)
        _sleep_inhibitor_process = None
        return False


def fixed_width_font(point_size=9, bold=False):
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(point_size)
    if bold:
        font.setWeight(QtGui.QFont.Weight.Bold)
    return font


class NoScrollSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, e): e.ignore()

class CommitOnPressComboBox(QtWidgets.QComboBox):
    """Combo box that reliably commits popup rows on the first mouse press."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if (
            watched is self.view().viewport()
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                row = index.row()
                self.setCurrentIndex(row)
                self.hidePopup()
                self.activated.emit(row)
                return True
        return super().eventFilter(watched, event)

class NoScrollComboBox(CommitOnPressComboBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollSlider(QtWidgets.QSlider):
    def wheelEvent(self, e): e.ignore()


class ResponsivePixmapLabel(QtWidgets.QLabel):
    """A preview label that fits its source pixmap to the available layout space."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = QtGui.QPixmap()
        self._fit_timer = QtCore.QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(60)
        self._fit_timer.timeout.connect(self._fit_pixmap)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = QtGui.QPixmap(pixmap)
        self._schedule_fit()

    def clear(self):
        self._fit_timer.stop()
        self._source_pixmap = QtGui.QPixmap()
        super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_fit()

    def sizeHint(self):
        # A displayed pixmap must not feed its dimensions back into the layout.
        return QtCore.QSize(640, 480)

    def minimumSizeHint(self):
        return QtCore.QSize(160, 160)

    def _schedule_fit(self):
        if not self._source_pixmap.isNull():
            self._fit_timer.start()

    def _fit_pixmap(self):
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            super().clear()
            return
        fitted = self._source_pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(fitted)


class ResizeSplitterHandle(QtWidgets.QSplitterHandle):
    """Distinct resize gutter with an accent grip, unlike a scrollbar thumb."""
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._hovered = False

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), THEME.color("surface_hover" if self._hovered else "nested_group"))
        painter.setPen(QtGui.QPen(THEME.color("border"), 1))
        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            painter.drawLine(0, 0, 0, self.height())
            painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
            center = self.rect().center()
            painter.setBrush(THEME.color("accent_hover" if self._hovered else "accent"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for offset in (-9, 0, 9):
                painter.drawEllipse(QtCore.QPointF(center.x(), center.y() + offset), 2.0, 2.0)
        else:
            painter.drawLine(0, 0, self.width(), 0)
            painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
            center = self.rect().center()
            painter.setBrush(THEME.color("accent_hover" if self._hovered else "accent"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for offset in (-9, 0, 9):
                painter.drawEllipse(QtCore.QPointF(center.x() + offset, center.y()), 2.0, 2.0)


class ThemedSplitter(QtWidgets.QSplitter):
    def __init__(self, orientation=QtCore.Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(10)

    def createHandle(self):
        return ResizeSplitterHandle(self.orientation(), self)


class EmptyStateListWidget(QtWidgets.QListWidget):
    def __init__(self, empty_text="", parent=None):
        super().__init__(parent)
        self.empty_text = empty_text

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() or not self.empty_text:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setPen(THEME.color("text_muted"))
        text_rect = self.viewport().rect().adjusted(16, 18, -16, -16)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignTop |
            QtCore.Qt.AlignmentFlag.AlignHCenter |
            QtCore.Qt.TextFlag.TextWordWrap,
            self.empty_text,
        )


class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other):
        try: return float(self.text()) < float(other.text())
        except ValueError: return super().__lt__(other)

class DateTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __init__(self, display_text, timestamp):
        super().__init__(display_text)
        self.timestamp = timestamp
    def __lt__(self, other):
        return self.timestamp < other.timestamp


def make_spin(lo, hi, val=None, *, scroll=False):
    w = QtWidgets.QSpinBox() if scroll else NoScrollSpinBox()
    w.setRange(lo, hi)
    if val is not None: w.setValue(val)
    return w

def make_dspin(lo, hi, val=None, step=0.1, decimals=2, *, scroll=False):
    w = QtWidgets.QDoubleSpinBox() if scroll else NoScrollDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    if val is not None: w.setValue(val)
    return w

def make_combo(items, *, scroll=False):
    w = CommitOnPressComboBox() if scroll else NoScrollComboBox()
    w.addItems(items)
    return w

def make_btn(text, callback=None, style=None):
    b = QtWidgets.QPushButton(text)
    if callback: b.clicked.connect(callback)
    if style: b.setStyleSheet(style)
    return b

def style_role(widget, role):
    """Use a global theme variant; avoids per-widget QSS parsing."""
    return set_role(widget, role)

def make_label(text, color=None, bold=False, size=None):
    lbl = QtWidgets.QLabel(text)
    role = None
    for candidate, value in (
        ("accent", ACCENT),
        ("accent_alt", ACCENT2),
        ("warning", WARN),
        ("danger", DANGER),
        ("success", SUCCESS),
    ):
        if color and QtGui.QColor(color) == QtGui.QColor(value):
            role = candidate
            break
    if role:
        lbl.setProperty("themeColorRole", role)
        lbl.setProperty("themeBold", bool(bold))
        lbl.setProperty("themePointSize", int(size or 0))
    parts = []
    if color: parts.append(f"color: {color};")
    if bold: parts.append("font-weight: bold;")
    if size: parts.append(f"font-size: {size}pt;")
    if parts: lbl.setStyleSheet(" ".join(parts))
    return lbl


class ThemeSwatchButton(QtWidgets.QAbstractButton):
    """Compact stacked theme swatches attached to the tab strip."""

    def __init__(self, primary, secondary, parent=None):
        super().__init__(parent)
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.setFixedSize(20, 20)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Choose GUI colors")

    def set_colors(self, primary, secondary):
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(THEME.color("accent") if self.underMouse() else THEME.color("border"), 1))
        painter.setBrush(THEME.color("window"))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 4, 4)
        swatch = QtCore.QRectF(2.5, 2.5, 15, 15)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(swatch, 2, 2)
        painter.save()
        painter.setClipPath(clip)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.primary)
        painter.drawPolygon(QtGui.QPolygonF([
            swatch.topLeft(), swatch.topRight(), swatch.bottomLeft(),
        ]))
        painter.setBrush(self.secondary)
        painter.drawPolygon(QtGui.QPolygonF([
            swatch.topRight(), swatch.bottomRight(), swatch.bottomLeft(),
        ]))
        painter.restore()


class ThemePresetButton(QtWidgets.QAbstractButton):
    """Popup row showing a preset name and its two colors."""

    def __init__(self, text, primary, secondary, parent=None):
        super().__init__(parent)
        self.text = text
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.setFixedHeight(34)
        self.setMinimumWidth(235)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.underMouse():
            painter.fillRect(self.rect(), THEME.color("surface_hover"))
        painter.setPen(THEME.color("text"))
        painter.drawText(
            self.rect().adjusted(10, 0, -58, 0),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text,
        )
        painter.setPen(QtGui.QPen(THEME.color("border"), 1))
        painter.setBrush(self.primary)
        painter.drawRoundedRect(QtCore.QRectF(self.width() - 50, 8, 18, 18), 3, 3)
        painter.setBrush(self.secondary)
        painter.drawRoundedRect(QtCore.QRectF(self.width() - 27, 8, 18, 18), 3, 3)

def make_separator(horizontal=True):
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.HLine if horizontal else QtWidgets.QFrame.Shape.VLine)
    f.setStyleSheet(f"border: 1px solid {BORDER}; margin: 4px 0;")
    return f

# Semantic group colouring. `group_box(title)` tints a group by which of these its title is in;
# the sets themselves are config knowledge, so schema.py fills them at import time.
RAW_GROUP_TITLES: set[str] = set()
TRANSFORMED_GROUP_TITLES: set[str] = set()

def set_semantic_color(widget, semantic):
    if widget.property("semanticColor") == semantic:
        return widget
    widget.setProperty("semanticColor", semantic)
    if widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_Polished):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
    return widget

def inherit_semantic_colors(root):
    """Give controls the semantic color of their nearest enclosing group."""
    control_types = (
        QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
        QtWidgets.QTextEdit, QtWidgets.QComboBox, QtWidgets.QSpinBox,
        QtWidgets.QDoubleSpinBox, QtWidgets.QCheckBox, QtWidgets.QSlider,
        QtWidgets.QListWidget, QtWidgets.QTableWidget,
    )
    for widget in root.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, control_types) or widget.property("semanticColor"):
            continue
        parent = widget.parentWidget()
        while parent is not None and parent is not root:
            if isinstance(parent, QtWidgets.QGroupBox):
                semantic = parent.property("semanticColor")
                if semantic:
                    set_semantic_color(widget, semantic)
                    break
            parent = parent.parentWidget()

def group_box(title, layout_type=QtWidgets.QVBoxLayout, role="nested"):
    gb = QtWidgets.QGroupBox(title)
    set_role(gb, role)
    if title in RAW_GROUP_TITLES:
        set_semantic_color(gb, "raw")
    elif title in TRANSFORMED_GROUP_TITLES:
        set_semantic_color(gb, "transformed")
    else:
        set_semantic_color(gb, "raw")
    lay = layout_type(gb)
    return gb, lay

def form_row(layout, label_text, widget, tooltip=None):
    lbl = QtWidgets.QLabel(label_text)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    layout.addRow(lbl, widget)

class CompressedLogBuffer:
    def __init__(self, block_size=128, compression_level=6, max_active_bytes=64 * 1024):
        self.block_size = max(16, int(block_size))
        self.compression_level = max(1, min(9, int(compression_level)))
        self.max_active_bytes = max(4096, int(max_active_bytes))
        self.blocks = []
        self.current_lines = []
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def clear(self):
        self.blocks.clear()
        self.current_lines.clear()
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def append(self, text, replace_last=False):
        if replace_last:
            self.remove_last_line()
        lines = str(text).rstrip('\n').splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            line_bytes = len(line.encode('utf-8', errors='replace')) + 1
            self.current_lines.append(line)
            self.current_bytes += line_bytes
            self.line_count += 1
            self.uncompressed_bytes += line_bytes
            if len(self.current_lines) >= self.block_size or self.current_bytes >= self.max_active_bytes:
                self._seal_current_block()

    def remove_last_line(self):
        if self.line_count <= 0:
            return
        if self.current_lines:
            removed = self.current_lines.pop()
            self.current_bytes = max(0, self.current_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
            self.line_count -= 1
            self.uncompressed_bytes = max(0, self.uncompressed_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
            return
        if not self.blocks:
            return
        lines = self._decode_block(len(self.blocks) - 1)
        old_payload_size = len(self.blocks[-1][1])
        if lines:
            removed = lines.pop()
            self.line_count -= 1
            self.uncompressed_bytes = max(0, self.uncompressed_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
        self.blocks.pop()
        self.compressed_bytes = max(0, self.compressed_bytes - old_payload_size)
        if lines:
            self.current_lines = lines
            self.current_bytes = sum(len(line.encode('utf-8', errors='replace')) + 1 for line in self.current_lines)
            if len(self.current_lines) >= self.block_size or self.current_bytes >= self.max_active_bytes:
                self._seal_current_block()

    def get_lines(self, start_line, count):
        if count <= 0 or self.line_count <= 0:
            return []
        start_line = max(0, min(int(start_line), self.line_count - 1))
        end_line = min(self.line_count, start_line + int(count))
        output = []
        cursor = 0
        for line_count, payload in self.blocks:
            block_start = cursor
            block_end = cursor + line_count
            if block_end > start_line and block_start < end_line:
                lines = zlib.decompress(payload).decode('utf-8', errors='replace').split('\n')
                local_start = max(0, start_line - block_start)
                local_end = min(line_count, end_line - block_start)
                output.extend(lines[local_start:local_end])
            cursor = block_end
            if cursor >= end_line:
                break
        if cursor < end_line and self.current_lines:
            block_start = cursor
            block_end = cursor + len(self.current_lines)
            if block_end > start_line and block_start < end_line:
                local_start = max(0, start_line - block_start)
                local_end = min(len(self.current_lines), end_line - block_start)
                output.extend(self.current_lines[local_start:local_end])
        return output

    def memory_summary(self):
        stored = self.compressed_bytes + self.current_bytes
        ratio = 1.0 if self.uncompressed_bytes <= 0 else stored / self.uncompressed_bytes
        return stored, self.uncompressed_bytes, ratio

    def get_all_text(self):
        if self.line_count <= 0:
            return ""
        return "\n".join(self.get_lines(0, self.line_count))

    def _seal_current_block(self):
        if not self.current_lines:
            return
        raw = '\n'.join(self.current_lines).encode('utf-8', errors='replace')
        payload = zlib.compress(raw, self.compression_level)
        self.blocks.append((len(self.current_lines), payload))
        self.compressed_bytes += len(payload)
        self.current_lines = []
        self.current_bytes = 0

    def _decode_block(self, index):
        if not (0 <= index < len(self.blocks)):
            return []
        line_count, payload = self.blocks[index]
        lines = zlib.decompress(payload).decode('utf-8', errors='replace').split('\n')
        return lines[:line_count]


class VirtualConsoleWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, visible_lines=900, clear_callback=None):
        super().__init__(parent)
        self.visible_lines = max(100, int(visible_lines))
        self.buffer = CompressedLogBuffer(block_size=128, compression_level=6)
        self.pending_render = False
        self.follow_output = True
        self.clear_callback = clear_callback
        self._internal_scroll_update = False
        self._internal_text_update = False
        self._build_ui()
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render_now)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.textbox = QtWidgets.QPlainTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setMinimumHeight(200)
        self.textbox.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.textbox.setUndoRedoEnabled(False)
        set_role(self.textbox, "consoleText")
        self.textbox.viewport().installEventFilter(self)
        self.textbox.verticalScrollBar().valueChanged.connect(self._on_inner_scrollbar_changed)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)

        row.addWidget(self.textbox, 1)
        row.addWidget(self.scrollbar)
        root.addLayout(row, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self.status_label = make_label("Lines: 0 | Buffer: empty", color=TEXT_SEC)
        self.follow_button = QtWidgets.QPushButton("Following Output")
        self.follow_button.setObjectName("FollowOutputButton")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.setToolTip("Toggle auto-scroll to the latest console output.")
        self.follow_button.toggled.connect(self._on_follow_toggled)
        self.clear_button = QtWidgets.QPushButton("Clear Console")
        self.clear_button.setToolTip("Clear the console output.")
        if self.clear_callback:
            self.clear_button.clicked.connect(self.clear_callback)
        self.copy_button = QtWidgets.QPushButton("Copy Full Logs")
        self.copy_button.setToolTip("Copy the complete log buffer, including lines not currently visible.")
        self.copy_button.clicked.connect(self.copy_full_logs)
        footer.addWidget(self.status_label)
        footer.addStretch()
        footer.addWidget(self.follow_button)
        footer.addWidget(self.copy_button)
        footer.addWidget(self.clear_button)
        root.addLayout(footer)

    def copy_full_logs(self):
        QtWidgets.QApplication.clipboard().setText(self.buffer.get_all_text())

    def eventFilter(self, obj, event):
        if obj is self.textbox.viewport() and event.type() == QtCore.QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta:
                inner_sb = self.textbox.verticalScrollBar()
                inner_can_scroll = (
                    (delta > 0 and inner_sb.value() > inner_sb.minimum()) or
                    (delta < 0 and inner_sb.value() < inner_sb.maximum())
                )
                if inner_can_scroll or self.scrollbar.maximum() <= 0:
                    return False
                step = max(1, self.scrollbar.singleStep())
                self.scrollbar.setValue(self.scrollbar.value() - int(delta / 120) * step)
                self._set_follow_output(self._at_console_bottom())
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def append_line(self, text, replace_last=False):
        was_at_bottom = self._at_console_bottom()
        self.buffer.append(text, replace_last=replace_last)
        if self.follow_output and was_at_bottom:
            self.follow_output = True
        self._schedule_render()

    def clear(self):
        self.buffer.clear()
        self.textbox.clear()
        self.scrollbar.setRange(0, 0)
        self.status_label.setText("Lines: 0 | Buffer: empty")
        self.follow_output = True
        self._sync_follow_button()

    def _schedule_render(self):
        if not self.pending_render:
            self.pending_render = True
            self.render_timer.start(33)

    def _on_follow_toggled(self, checked):
        self.follow_output = bool(checked)
        self._sync_follow_button()
        if self.follow_output:
            self.scrollbar.setValue(self.scrollbar.maximum())
            self._schedule_render()

    def _on_scrollbar_changed(self, value):
        if self._internal_scroll_update:
            return
        self._set_follow_output(self._at_console_bottom())
        self._schedule_render()

    def _on_inner_scrollbar_changed(self, value):
        if self._internal_text_update:
            return
        self._set_follow_output(self._at_console_bottom())

    def _at_console_bottom(self):
        inner_sb = self.textbox.verticalScrollBar()
        outer_bottom = self.scrollbar.value() >= self.scrollbar.maximum() - 1
        inner_bottom = inner_sb.value() >= inner_sb.maximum() - 1
        return outer_bottom and inner_bottom

    def _set_follow_output(self, follow):
        self.follow_output = bool(follow)
        self._sync_follow_button()

    def _sync_follow_button(self):
        self.follow_button.blockSignals(True)
        self.follow_button.setChecked(self.follow_output)
        self.follow_button.setText("Following Output" if self.follow_output else "Jump to Bottom")
        self.follow_button.blockSignals(False)

    def _render_now(self):
        self.pending_render = False
        total = self.buffer.line_count
        max_start = max(0, total - self.visible_lines)
        self._internal_scroll_update = True
        self.scrollbar.setRange(0, max_start)
        self.scrollbar.setPageStep(self.visible_lines)
        self.scrollbar.setSingleStep(max(1, self.visible_lines // 20))
        if self.follow_output:
            self.scrollbar.setValue(max_start)
        elif self.scrollbar.value() > max_start:
            self.scrollbar.setValue(max_start)
        start = self.scrollbar.value()
        self._internal_scroll_update = False

        lines = self.buffer.get_lines(start, self.visible_lines)
        inner_sb = self.textbox.verticalScrollBar()
        inner_value = inner_sb.value()
        self._internal_text_update = True
        self.textbox.setPlainText('\n'.join(lines))
        if self.follow_output:
            inner_sb.setValue(inner_sb.maximum())
        else:
            inner_sb.setValue(min(inner_value, inner_sb.maximum()))
        self._internal_text_update = False

        stored, uncompressed, ratio = self.buffer.memory_summary()
        shown_start = 0 if total == 0 else start + 1
        shown_end = min(total, start + len(lines))
        self.status_label.setText(
            f"Lines: {total:,} | Showing: {shown_start:,}-{shown_end:,} | "
            f"Memory: {self._fmt_bytes(stored)} compressed from {self._fmt_bytes(uncompressed)} ({ratio:.2%})"
        )

    def _fmt_bytes(self, value):
        value = float(value)
        for unit in ["B", "KB", "MB", "GB"]:
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0




class CustomFolderDialog(QtWidgets.QDialog):
    def __init__(self, start_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset Folder")
        self.resize(1100, 700)
        self.selected_path = None
        self.history = []
        self.history_idx = -1
        self.is_navigating_history = False
        self.icon_provider = QFileIconProvider()
        self.current_path = usable_dialog_start(start_path)
        self._build_ui()
        self.load_directory(self.current_path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(5)
        def nav_btn(icon, tip, cb):
            b = QtWidgets.QPushButton()
            b.setIcon(self.style().standardIcon(icon))
            b.setFixedWidth(35)
            b.setToolTip(tip)
            b.clicked.connect(cb)
            return b

        self.btn_back    = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack, "Back", self.go_back)
        self.btn_fwd     = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward, "Forward", self.go_forward)
        self.btn_up      = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp, "Up", self.go_up)
        self.btn_refresh = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload, "Refresh",
                                   lambda: self.load_directory(self.current_path))
        self.btn_back.setEnabled(False)
        self.btn_fwd.setEnabled(False)

        self.path_edit = QtWidgets.QLineEdit(self.current_path)
        self.path_edit.returnPressed.connect(self._on_path_entered)

        for w in [self.btn_back, self.btn_fwd, self.btn_up, self.path_edit, self.btn_refresh]:
            nav.addWidget(w) if w is not self.path_edit else nav.addWidget(w, 1)
        layout.addLayout(nav)

        splitter = ThemedSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setIconSize(QtCore.QSize(24, 24))
        self.sidebar.setStyleSheet(f"""
            QListWidget {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; outline: none; }}
            QListWidget::item {{ padding: 6px; color: {TEXT_PRI}; margin: 2px; }}
            QListWidget::item:hover {{ background: {BORDER}; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {ACCENT}; color: white; border-radius: 4px; }}
        """)
        self.sidebar.itemClicked.connect(lambda item: item.data(QtCore.Qt.ItemDataRole.UserRole) and self.load_directory(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        self._populate_sidebar()
        splitter.addWidget(self.sidebar)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Images", "Date Modified", "HiddenPath"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnHidden(3, True)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QtCore.QSize(20, 20))
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda r, _: self.load_directory(self.table.item(r, 3).text()))
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: {PANEL_BG}; alternate-background-color: {DARK_BG}; border: 1px solid {BORDER}; border-radius: 4px; }}
            QTableWidget::item {{ padding: 4px; }}
            QHeaderView::section {{ background: {BORDER}; padding: 6px; border: none; border-right: 1px solid {BORDER}; font-weight: bold; }}
        """)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        bottom = QtWidgets.QHBoxLayout()
        self.status_label = make_label("", color=TEXT_SEC)
        cancel_btn = make_btn("Cancel", self.reject)
        select_btn = make_btn("Select Current Folder", self.select_current)
        set_role(select_btn, "accent")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(cancel_btn)
        bottom.addWidget(select_btn)
        layout.addLayout(bottom)

    def _populate_sidebar(self):
        paths = QtCore.QStandardPaths
        for text, loc in [("Desktop", paths.StandardLocation.DesktopLocation),
                          ("Documents", paths.StandardLocation.DocumentsLocation),
                          ("Pictures", paths.StandardLocation.PicturesLocation),
                          ("Downloads", paths.StandardLocation.DownloadLocation)]:
            p = paths.writableLocation(loc)
            if os.path.exists(p):
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(p)), text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
                self.sidebar.addItem(item)
        sep = QtWidgets.QListWidgetItem("───── Drives ─────")
        sep.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        sep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sidebar.addItem(sep)
        for vol in QtCore.QStorageInfo.mountedVolumes():
            if vol.isValid() and vol.isReady():
                name = vol.name() or vol.rootPath()
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(vol.rootPath())), name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, vol.rootPath())
                self.sidebar.addItem(item)

    def _on_path_entered(self):
        p = platform_path(self.path_edit.text())
        if os.path.isdir(p): self.load_directory(p)
        else: QtWidgets.QMessageBox.warning(self, "Invalid Path", "Path does not exist or is not a directory.")

    def load_directory(self, path):
        if not self.is_navigating_history:
            if self.history_idx == -1 or self.history[self.history_idx] != path:
                self.history = self.history[:self.history_idx + 1]
                self.history.append(path)
                self.history_idx += 1
        self.btn_back.setEnabled(self.history_idx > 0)
        self.btn_fwd.setEnabled(self.history_idx < len(self.history) - 1)
        self.is_navigating_history = False
        self.current_path = path
        self.path_edit.setText(path)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.status_label.setText("Scanning...")
        QtWidgets.QApplication.processEvents()

        img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        try:
            entries = [e for e in os.scandir(path) if e.is_dir()]
            self.table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                info = QFileInfo(entry.path)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(self.icon_provider.icon(info), entry.name))
                try:
                    count = sum(1 for f in os.scandir(entry.path)
                                if f.is_file() and os.path.splitext(f.name)[1].lower() in img_exts)
                    ci = NumericTableWidgetItem(str(count))
                    ci.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    ci.setForeground(QtGui.QColor(ACCENT2 if count > 0 else TEXT_SEC))
                    if count > 0: ci.setFont(fixed_width_font(9, bold=True))
                    self.table.setItem(row, 1, ci)
                except PermissionError:
                    self.table.setItem(row, 1, NumericTableWidgetItem("-1"))
                try:
                    ts = entry.stat().st_mtime
                    di = DateTableWidgetItem(datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"), ts)
                    di.setForeground(QtGui.QColor(TEXT_SEC))
                    self.table.setItem(row, 2, di)
                except Exception:
                    self.table.setItem(row, 2, DateTableWidgetItem("N/A", 0))
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(entry.path))
            self.status_label.setText(f"{len(entries)} folders found.")
        except PermissionError:
            QtWidgets.QMessageBox.warning(self, "Access Denied", f"Cannot access {path}")
            self.go_back()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
        self.table.setSortingEnabled(True)

    def go_up(self):
        parent = os.path.dirname(self.current_path)
        if parent and parent != self.current_path: self.load_directory(parent)

    def go_back(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def go_forward(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def select_current(self):
        rows = self.table.selectionModel().selectedRows()
        self.selected_path = (self.table.item(rows[0].row(), 3).text() if rows else self.current_path)
        self.accept()


class GraphPanel(QtWidgets.QWidget):
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.lines = []
        # The top band is a compact, Chart.js-style series legend.
        # It replaces the centered title and stays outside the plot area.
        self.padding = {'top': 42, 'bottom': 40, 'left': 70, 'right': 20}
        self.bg_color = THEME.color("canvas")
        self.graph_bg_color = THEME.color("canvas")
        self.grid_color = THEME.color("border")
        self.text_color = THEME.color("text")
        self.title_color = THEME.color("accent")
        self.x_min, self.x_max = 0, 100
        self.y_min, self.y_max = 0, 1
        self.data_x_min, self.data_x_max = 0, 100
        self.fill_enabled = False
        self.view_x_min = None
        self.view_x_max = None
        self.render_x_min = None
        self.render_x_max = None
        self.target_y_min = 0
        self.target_y_max = 1
        self.render_y_min = None
        self.render_y_max = None
        self.drag_start_pos = None
        self.drag_start_range = None
        self.hover_point = None
        self._dirty_bounds = True
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._animate_view)
        self.repaint_timer = QtCore.QTimer(self)
        self.repaint_timer.setSingleShot(True)
        self.repaint_timer.timeout.connect(self.update)
        self.setMouseTracking(True)
        self.setMinimumHeight(220)

    def _graph_rect(self):
        return QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])

    def add_line(self, color, label, max_points=2000, linewidth=2, line_style="solid"):
        self.lines.append({
            'data': [],
            'x_values': [],
            'max_points': max_points,
            'version': 0,
            'color': QtGui.QColor(color),
            'label': label,
            'linewidth': linewidth,
            'line_style': line_style,
            'visible': True,
        })
        return len(self.lines) - 1

    def set_line_visible(self, line_index, visible):
        if 0 <= line_index < len(self.lines):
            self.lines[line_index]['visible'] = bool(visible)
            self._dirty_bounds = True
            self.update()

    def append_data(self, line_index, x, y):
        if 0 <= line_index < len(self.lines):
            line = self.lines[line_index]
            if line['x_values'] and x <= line['x_values'][-1]:
                pos = bisect_left(line['x_values'], x)
                if pos < len(line['x_values']) and line['x_values'][pos] == x:
                    line['data'][pos] = (x, y)
                else:
                    line['data'].insert(pos, (x, y))
                    line['x_values'].insert(pos, x)
            else:
                line['data'].append((x, y))
                line['x_values'].append(x)
            line['version'] += 1
            if len(line['data']) > line['max_points']:
                self._compact_line(line)
            self._refresh_data_range()
            self._fit_full_range(False)
            self._dirty_bounds = True
            self._schedule_repaint()

    def clear_all_data(self):
        for line in self.lines:
            line['data'].clear()
            line['x_values'].clear()
            line['version'] += 1
        self.view_x_min = None
        self.view_x_max = None
        self.render_x_min = None
        self.render_x_max = None
        self.render_y_min = None
        self.render_y_max = None
        self.hover_point = None
        self._dirty_bounds = True
        self._update_bounds()
        self.update()

    def _refresh_data_range(self):
        firsts = [line['data'][0][0] for line in self.lines if line['data']]
        lasts = [line['data'][-1][0] for line in self.lines if line['data']]
        if not firsts or not lasts:
            self.data_x_min, self.data_x_max = 0, 100
            self.view_x_min, self.view_x_max = None, None
            return
        self.data_x_min = min(firsts)
        self.data_x_max = max(lasts)
        if self.data_x_min == self.data_x_max:
            self.data_x_max = self.data_x_min + 1

    def _fit_full_range(self, animate=True):
        self.view_x_min = self.data_x_min
        self.view_x_max = self.data_x_max
        if not animate:
            self.render_x_min = self.view_x_min
            self.render_x_max = self.view_x_max
            self.render_y_min = None
            self.render_y_max = None
        self._dirty_bounds = True
        if animate:
            self._start_view_animation()

    def _compact_line(self, line):
        data = line['data']
        target = max(256, line['max_points'] // 2)
        if len(data) <= target:
            return
        bucket_count = max(2, (target - 2) // 2)
        middle = data[1:-1]
        bucket_size = len(middle) / bucket_count
        compacted = [data[0]]
        for bucket in range(bucket_count):
            start = int(bucket * bucket_size)
            end = int((bucket + 1) * bucket_size)
            if bucket == bucket_count - 1:
                end = len(middle)
            segment = middle[start:end]
            if not segment:
                continue
            min_i = min(range(len(segment)), key=lambda i: segment[i][1])
            max_i = max(range(len(segment)), key=lambda i: segment[i][1])
            for local_i in sorted({min_i, max_i}):
                compacted.append(segment[local_i])
        compacted.append(data[-1])
        line['data'] = compacted
        line['x_values'] = [x for x, _ in compacted]
        line['version'] += 1

    def _get_visible_slice(self, line):
        data = line['data']
        if not data: return []
        view_min = self.x_min
        view_max = self.x_max
        if len(data) <= 2:
            return data[:]
        x_values = line['x_values']
        start = bisect_left(x_values, view_min)
        end = bisect_right(x_values, view_max)
        start = max(0, start - 1)
        end = min(len(data), end + 1)
        if start >= end:
            if start >= len(data):
                return data[-1:]
            return data[start:start + 1]
        return data[start:end]

    def _sample_visible_points(self, raw, max_points):
        count = len(raw)
        if count <= max_points:
            return raw[:count]

        bucket_count = max(2, max_points // 2)
        bucket_size = count / bucket_count
        sampled = []

        for bucket in range(bucket_count):
            start = int(bucket * bucket_size)
            end = int((bucket + 1) * bucket_size)
            if bucket == bucket_count - 1:
                end = count
            if end <= start:
                continue

            segment = raw[start:end]
            if not segment:
                continue
            min_i = min(range(len(segment)), key=lambda i: segment[i][1])
            max_i = max(range(len(segment)), key=lambda i: segment[i][1])
            for local_i in sorted({min_i, max_i}):
                sampled.append(raw[start + local_i])

        return sampled

    def _update_bounds(self):
        all_y, all_x = [], []
        target_x_min = self.view_x_min if self.view_x_min is not None else self.data_x_min
        target_x_max = self.view_x_max if self.view_x_max is not None else self.data_x_max
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min, self.render_x_max = target_x_min, target_x_max
        self.x_min, self.x_max = self.render_x_min, self.render_x_max
        for line in self.lines:
            if not line.get('visible', True): continue
            raw = self._get_visible_slice(line)
            if raw:
                all_x.extend(x for x, _ in raw)
                all_y.extend(y for _, y in raw)
        if all_x:
            self.x_min = self.render_x_min
            self.x_max = self.render_x_max
            if self.x_min == self.x_max:
                self.x_max = self.x_min + 1
            if all_y:
                yr = max(all_y) - min(all_y) or 1
                self.target_y_min = min(all_y) - yr * 0.08
                self.target_y_max = max(all_y) + yr * 0.08
            else:
                self.target_y_min, self.target_y_max = 0, 1
        else:
            self.x_min, self.x_max = 0, 100
            self.target_y_min, self.target_y_max = 0, 1
        if self.render_y_min is None or self.render_y_max is None:
            self.render_y_min, self.render_y_max = self.target_y_min, self.target_y_max
        self.y_min, self.y_max = self.render_y_min, self.render_y_max
        self._dirty_bounds = False

    def _to_screen(self, x, y):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        xr = self.x_max - self.x_min or 1
        yr = self.y_max - self.y_min or 1
        sx = self.padding['left'] + ((x - self.x_min) / xr) * gw
        sy = self.padding['top'] + gh - ((y - self.y_min) / yr) * gh
        return QtCore.QPointF(sx, sy)

    def _from_screen_x(self, px):
        gr = self._graph_rect()
        xr = self.x_max - self.x_min or 1
        return self.x_min + ((px - gr.left()) / max(1, gr.width())) * xr

    def paintEvent(self, event):
        if self._dirty_bounds:
            self._update_bounds()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        gr = self._graph_rect()
        painter.fillRect(gr, self.graph_bg_color)
        self._draw_grid(painter, gr)
        self._draw_lines(painter, gr)
        self._draw_legend(painter)
        self._draw_hover(painter, gr)

    def _draw_grid(self, painter, rect):
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            y_val = self.y_max - (i / 4) * (self.y_max - self.y_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self._fmt(y_val))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            x_val = self.x_min + (i / 5) * (self.x_max - self.x_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(x_val)))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setPen(self.text_color)
        f = painter.font(); f.setPixelSize(12); painter.setFont(f)
        painter.save()
        painter.translate(15, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-50, -10, 100, 20), QtCore.Qt.AlignmentFlag.AlignCenter, self.y_label)
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Step")

    def _draw_lines(self, painter, rect):
        painter.save()
        painter.setClipRect(rect)
        max_points = max(128, rect.width() * 2)
        for line in self.lines:
            if not line.get('visible', True): continue
            raw = self._get_visible_slice(line)
            if len(raw) < 2: continue
            sampled = self._sample_visible_points(raw, max_points)
            pts = [self._to_screen(x, y) for x, y in sampled]
            if len(pts) < 2:
                continue
            if self.fill_enabled:
                poly = QtGui.QPolygonF(pts)
                poly.append(QtCore.QPointF(pts[-1].x(), rect.bottom()))
                poly.append(QtCore.QPointF(pts[0].x(), rect.bottom()))
                fc = QtGui.QColor(line['color']); fc.setAlpha(self._fill_alpha(len(raw)))
                painter.setBrush(fc); painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(poly)
            width = self._line_width(line)
            painter.setPen(self._line_pen(line, width))
            painter.drawPolyline(QtGui.QPolygonF(pts))
        painter.restore()

    def _line_pen(self, line, width=None):
        pen = QtGui.QPen(line['color'], width if width is not None else self._line_width(line))
        style = line.get('line_style', 'solid')
        if style == 'dotted':
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([1.0, 3.0])
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        elif style == 'dashed':
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([6.0, 3.0])
        return pen

    def _line_width(self, line):
        count = len(line['data'])
        extra = min(1.4, math.log10(max(1, count)) * 0.25)
        return line['linewidth'] + extra

    def _fill_alpha(self, visible_count):
        if visible_count <= 16:
            return 64
        if visible_count <= 128:
            return 52
        return 38


    def _draw_legend(self, painter):
        lx = self.padding['left']
        ly = 13
        f = painter.font(); f.setPixelSize(12); f.setBold(False); painter.setFont(f)
        for line in self.lines:
            if not line.get('visible', True): continue
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(line['color'])
            painter.drawRoundedRect(QtCore.QRectF(lx, ly, 10, 10), 2, 2)
            painter.setPen(self.text_color)
            label_width = painter.fontMetrics().horizontalAdvance(line['label'])
            painter.drawText(QtCore.QRect(lx + 16, ly - 3, label_width + 2, 16),
                             QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, line['label'])
            lx += 16 + label_width + 22

    def _draw_hover(self, painter, rect):
        if not self.hover_point:
            return
        line, x, y, pt = self.hover_point
        hc = QtGui.QColor(line['color'])
        painter.setPen(QtGui.QPen(hc, 1))
        painter.drawLine(QtCore.QPointF(pt.x(), rect.top()), QtCore.QPointF(pt.x(), rect.bottom()))
        painter.drawLine(QtCore.QPointF(rect.left(), pt.y()), QtCore.QPointF(rect.right(), pt.y()))
        painter.setBrush(hc)
        painter.drawEllipse(pt, 4, 4)
        text = f"{line['label']}  Step {int(x)}  {self._fmt(y)}"
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 16
        h = 24
        tx = min(max(rect.left() + 6, int(pt.x()) + 10), rect.right() - w - 6)
        ty = max(rect.top() + 6, int(pt.y()) - h - 10)
        box = QtCore.QRect(tx, ty, w, h)
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setBrush(self.bg_color)
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(self.text_color)
        painter.drawText(box.adjusted(8, 0, -8, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, text)

    def _fmt(self, v):
        if abs(v) < 0.01 or abs(v) > 10000: return f"{v:.1e}"
        if abs(v) < 1: return f"{v:.4f}"
        return f"{v:.2f}"

    def set_fill(self, e): self.fill_enabled = e; self.update()

    def wheelEvent(self, event):
        gr = self._graph_rect()
        if not gr.contains(event.position().toPoint()):
            event.ignore()
            return
        if self.view_x_min is None or self.view_x_max is None:
            self._fit_full_range()
        steps = max(-4, min(4, event.angleDelta().y() / 120))
        factor = 0.94 ** steps
        center = self._from_screen_x(event.position().x())
        span = max(1e-9, (self.view_x_max - self.view_x_min) * factor)
        full_span = max(1e-9, self.data_x_max - self.data_x_min)
        min_span = max(1, full_span / 1000000)
        span = max(min_span, min(span, full_span))
        rel = (center - self.view_x_min) / max(1e-9, self.view_x_max - self.view_x_min)
        self.view_x_min = center - span * rel
        self.view_x_max = self.view_x_min + span
        self._clamp_view()
        self._dirty_bounds = True
        self._start_view_animation()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._graph_rect().contains(event.position().toPoint()):
            self.drag_start_pos = event.position()
            self.drag_start_range = (self.view_x_min, self.view_x_max)
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_start_pos is not None and self.drag_start_range[0] is not None:
            dx = event.position().x() - self.drag_start_pos.x()
            span = self.drag_start_range[1] - self.drag_start_range[0]
            shift = -dx / max(1, self._graph_rect().width()) * span
            self.view_x_min = self.drag_start_range[0] + shift
            self.view_x_max = self.drag_start_range[1] + shift
            self._clamp_view()
            self._dirty_bounds = True
            self._start_view_animation()
            event.accept()
            return
        self.hover_point = self._nearest_point(event.position())
        self._schedule_repaint()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drag_start_pos = None
            self.drag_start_range = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()

    def leaveEvent(self, event):
        self.hover_point = None
        self._schedule_repaint()

    def _nearest_point(self, pos):
        gr = self._graph_rect()
        if not gr.contains(pos.toPoint()):
            return None
        nearest = None
        nearest_dist = 12 * 12
        for line in self.lines:
            for x, y in self._get_visible_slice(line):
                pt = self._to_screen(x, y)
                dist = (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2
                if dist < nearest_dist:
                    nearest = (line, x, y, pt)
                    nearest_dist = dist
        return nearest

    def _clamp_view(self):
        if self.view_x_min is None or self.view_x_max is None:
            return
        span = self.view_x_max - self.view_x_min
        full_span = self.data_x_max - self.data_x_min
        if span >= full_span:
            self.view_x_min = self.data_x_min
            self.view_x_max = self.data_x_max
        elif self.view_x_min < self.data_x_min:
            self.view_x_min = self.data_x_min
            self.view_x_max = self.data_x_min + span
        elif self.view_x_max > self.data_x_max:
            self.view_x_max = self.data_x_max
            self.view_x_min = self.data_x_max - span

    def _start_view_animation(self):
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min = self.view_x_min
            self.render_x_max = self.view_x_max
        if not self.anim_timer.isActive():
            self.anim_timer.start(16)
        self._schedule_repaint()

    def _animate_view(self):
        if self.view_x_min is None or self.view_x_max is None:
            self.anim_timer.stop()
            return
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min, self.render_x_max = self.view_x_min, self.view_x_max
        self._dirty_bounds = True
        self._update_bounds()
        targets = [
            (self.render_x_min, self.view_x_min),
            (self.render_x_max, self.view_x_max),
            (self.render_y_min, self.target_y_min),
            (self.render_y_max, self.target_y_max),
        ]
        done = True
        next_values = []
        for current, target in targets:
            if current is None:
                current = target
            delta = target - current
            scale = max(1.0, abs(target))
            if abs(delta) > scale * 0.001:
                done = False
            next_values.append(current + delta * 0.28)
        self.render_x_min, self.render_x_max, self.render_y_min, self.render_y_max = next_values
        if done:
            self.render_x_min, self.render_x_max = self.view_x_min, self.view_x_max
            self.render_y_min, self.render_y_max = self.target_y_min, self.target_y_max
            self.anim_timer.stop()
        self._dirty_bounds = True
        self._schedule_repaint()

    def _schedule_repaint(self):
        if not self.repaint_timer.isActive():
            self.repaint_timer.start(0)
