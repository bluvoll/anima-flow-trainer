"""Subprocess launching for the GUI.

`ProcessRunner` is vendored from Aozora Trainer (Apache-2.0) with the Windows-only branches kept
intact; the launch specs below are ours.

The GUI never imports the trainer -- it spawns it and reads stdout. That is what keeps a crashing
run from taking the window down with it, and it is why a torch/CUDA import never happens in the Qt
process.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .widgets import ANSI_ESCAPE_RE, IS_WINDOWS, PROJECT_ROOT


@dataclass
class Launch:
    """One thing the GUI can run. `argv` is passed to Popen verbatim -- no shell."""

    argv: list[str]
    label: str
    # Marks a run that must NOT be treated as training (no graph reset, no "training finished").
    is_training: bool = True
    env: dict[str, str] = field(default_factory=dict)


def _python() -> str:
    return sys.executable


def train_launch(config_path: str | Path, num_processes: int = 1, gpus: str = "") -> Launch:
    """Always through Accelerate, single GPU included -- the same choice sd-scripts makes.

    One launch path means one set of behaviours to reason about: the same `BatchSamplerShard`,
    the same `AcceleratedScheduler` horizon scaling, the same trailing-partial-group sync. A bug
    that only appears under the launcher cannot then hide from single-GPU testing, and a config
    that runs on one card runs on two without a second code path having to agree with the first.

    `python -m accelerate.commands.launch` rather than the `accelerate` console script: the script
    lives in the venv's bin/ and may not be on PATH when the GUI is started from a desktop entry,
    while the module is importable by construction if accelerate is installed at all. Verified to
    give identical results to the console script (MULTI_GPU, correct ranks and devices).
    """
    argv = [_python(), "-u", "-m", "accelerate.commands.launch",
            "--num_processes", str(num_processes),
            "-m", "anima.training.train", str(config_path)]
    label = f"train ({num_processes} GPU{'s, DDP' if num_processes > 1 else ''})"
    return Launch(argv, label)


def cache_launch(
    dataset_path: str,
    model_path: str,
    resolutions: list[int],
    *,
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_reso_steps: int,
    upscale: bool,
    multires_training: bool,
    overwrite: bool = False,
    dry_run: bool = False,
    gpus: str = "",
) -> Launch:
    argv = [_python(), "-u", "-m", "anima.tools.cache_latents", "cache", dataset_path,
            "--model-path", model_path,
            "--resolution", *[str(r) for r in resolutions],
            "--min-bucket-reso", str(min_bucket_reso),
            "--max-bucket-reso", str(max_bucket_reso),
            "--bucket-reso-steps", str(bucket_reso_steps)]
    if upscale:
        argv.append("--upscale")
    if multires_training:
        argv.append("--multires")
    if overwrite:
        argv.append("--overwrite")
    if dry_run:
        argv.append("--dry-run")
    tiers = "/".join(str(r) for r in resolutions)
    return Launch(argv, f"cache latents ({tiers})", is_training=False,
                  env={"CUDA_VISIBLE_DEVICES": gpus} if gpus else {})


def audit_launch(dataset_path: str, bucket_reso_steps: int) -> Launch:
    """Reports the source-size distribution and proposes a resolution ladder. Reads nothing but
    image headers, so it needs no GPU and no model."""
    return Launch(
        [_python(), "-u", "-m", "anima.tools.cache_latents", "audit", dataset_path,
         "--bucket-reso-steps", str(bucket_reso_steps)],
        "audit dataset", is_training=False,
    )


class ProcessRunner(QThread):
    """Vendored from Aozora (`gui/gui.py::ProcessRunner`), retargeted at our log lines.

    stdout and stderr are merged into one pipe so ordering is preserved; the trainer runs under
    `-u` so a crash traceback is not lost in a half-flushed buffer.
    """

    logSignal = Signal(str)
    progressSignal = Signal(str, bool)
    finishedSignal = Signal(int)
    errorSignal = Signal(str)
    metricsSignal = Signal(str)

    # Substrings that mean the run is already doomed -- surfaced in the log with emphasis rather
    # than scrolling past at the same weight as a step line.
    FATAL_HINTS = (
        "out of memory", "cuda error", "nccl error", "device-side assert",
        "nan/inf", "memory inaccessible",
    )

    def __init__(self, launch: Launch, working_dir: str, env=None):
        super().__init__()
        self.launch = launch
        self.working_dir = working_dir
        self.env = env
        self.process = None
        self.stop_requested = False

    @staticmethod
    def _clean(line: str) -> str:
        return ANSI_ESCAPE_RE.sub("", line)

    def run(self):
        try:
            popen_options = {}
            if IS_WINDOWS:
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # One process group for the trainer, its dataloader workers, and -- under DDP --
                # every rank Accelerate spawns. Without this, stopping kills the launcher and
                # leaves N ranks holding the GPUs.
                popen_options["start_new_session"] = True
            self.process = subprocess.Popen(
                self.launch.argv, cwd=self.working_dir, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, **popen_options)
            self.logSignal.emit(f"INFO: {self.launch.label} started (PID {self.process.pid})")
            for line in iter(self.process.stdout.readline, ''):
                line = self._clean(line.rstrip("\n"))
                if not line.strip():
                    continue
                low = line.lower()
                if any(hint in low for hint in self.FATAL_HINTS):
                    self.logSignal.emit(f"*** {line} ***")
                else:
                    # tqdm redraws with \r; keep only the newest frame.
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|', line))
                    self.progressSignal.emit(line.split('\r')[-1], is_progress)
                self.metricsSignal.emit(line)
            self.finishedSignal.emit(self.process.wait())
        except Exception as e:
            self.errorSignal.emit(f"Subprocess error: {e}")
            self.finishedSignal.emit(-1)

    def _kill_tree_windows(self) -> bool:
        """Kill the launcher *and* everything under it. Returns False if taskkill was unusable.

        `Popen.terminate()` is TerminateProcess on one PID, and Windows has no killpg, so it takes
        down `accelerate.commands.launch` and leaves the process it spawned alive -- simple_launcher
        is an unconditional `subprocess.Popen` (accelerate/commands/launch.py:989), so there is
        always at least one grandchild, single GPU included. Two things then go wrong at once: the
        real trainer keeps running on the GPU, and because it inherited the stdout handle the pipe
        never reaches EOF, so `iter(readline, '')` blocks forever and `finishedSignal` is never
        emitted. The GUI keeps its Stop button up and keeps printing step lines -- which is exactly
        what "stop does nothing on Windows" looks like from the outside. `/T` walks the tree, `/F`
        is required because the ranks have no window to accept a polite close.
        """
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                capture_output=True,
                # No console flash: the GUI has no terminal attached to borrow.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=15,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stop_requested = True
            killed_tree = False
            if IS_WINDOWS:
                killed_tree = self._kill_tree_windows()
                if not killed_tree:
                    # Last resort. Only reaches the launcher, so a grandchild may survive and hold
                    # the GPU -- said out loud rather than left for the user to find in nvidia-smi.
                    self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if IS_WINDOWS:
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            if IS_WINDOWS and not killed_tree:
                self.logSignal.emit(
                    "WARNING: taskkill unavailable -- only the launcher was stopped. Check Task "
                    "Manager for a surviving python.exe still holding the GPU."
                )
            self.logSignal.emit("Process stopped.")


_DEVICE_LIST_RE = re.compile(r"^\d+(,\d+)*$")


def training_env(gpus: str = "") -> dict[str, str]:
    """Environment for a spawned run: the repo on PYTHONPATH so `-m anima.…` resolves however the
    GUI itself was started.

    `gpus` must be a bare device list. Torch treats an unparseable `CUDA_VISIBLE_DEVICES` as "no
    devices" rather than as an error, so passing something like `all` through here silently moves
    a 2B model onto the CPU -- which is how this check came to exist. Anything that is not
    digits-and-commas is rejected loudly instead.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT))
    gpus = (gpus or "").strip()
    if gpus:
        if not _DEVICE_LIST_RE.match(gpus):
            raise ValueError(
                f"GPU selection must be device indices like '0' or '0,1', got {gpus!r}. "
                f"Leave it empty to use every GPU."
            )
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env
