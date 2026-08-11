"""Post-install smoke check, shared by install.sh and install.bat.

Deliberately imports rather than reading version metadata: a package can be pip-installed and still
fail to import, and torch with the wrong CUDA build imports fine but reports no devices. Both are
things you want to hear about now rather than several minutes into the first run.

Run directly with the installed interpreter -- it must not import anything from `anima` itself, so
that a broken trainer still produces a readable dependency report:

    venv/bin/python -m anima.tools.check_install
"""

from __future__ import annotations

import importlib
import sys

# (module, label). Order is roughly the order a run needs them in, so the first failure is usually
# the informative one.
REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("sdnq", "sdnq"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
]
OPTIONAL = [
    ("PySide6", "PySide6 (GUI)"),
    ("scipy", "scipy (optimal transport)"),
    ("triton", "triton"),
    ("openvino", "openvino"),
]
# `--require-gui` promotes these. The installers always install the `gui` extra and always write a
# start-gui launcher, so for THEM a missing PySide6 is a failed install, not a choice -- without
# this the installer would print MISSING, exit 0, and still say "done, start the GUI with...".
# Run bare (no flag) the same check stays advisory, which is right for a CLI-only environment.
GUI_REQUIRED = [("PySide6", "PySide6 (GUI)")]


def _version(mod) -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "ok"


def _probe(entries):
    failures = []
    for name, label in entries:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:                       # noqa: BLE001 - report, never propagate
            failures.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"    {label:<26} MISSING")
        else:
            print(f"    {label:<26} {_version(mod)}")
    return failures


def main() -> int:
    require_gui = "--require-gui" in sys.argv
    major, minor, micro = sys.version_info[:3]
    print(f"    {'python':<26} {major}.{minor}.{micro}")
    required = REQUIRED + GUI_REQUIRED if require_gui else REQUIRED
    optional = [e for e in OPTIONAL if e not in GUI_REQUIRED] if require_gui else OPTIONAL
    failures = _probe(required)
    print()
    _probe(optional)          # optional: reported, never fatal

    # diffusers is the one version that is not merely "installed or not". Anima support exists only
    # on main, so a PyPI release imports perfectly and then fails at model load with a bare KeyError.
    try:
        import diffusers
        if not hasattr(diffusers, "AnimaModularPipeline"):
            failures.append((
                "diffusers",
                (f"{diffusers.__version__} has no AnimaModularPipeline -- this is a release build, "
                 "not main. Reinstall from git: pip install "
                 "'diffusers @ git+https://github.com/huggingface/diffusers'"),
            ))
    except Exception as exc:                           # noqa: BLE001
        print(f"    (could not inspect diffusers: {type(exc).__name__})")

    try:
        import torch
        if torch.cuda.is_available():
            names = ", ".join(torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count()))
            print(f"\n    CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s): {names}")
        else:
            # Not fatal: caching latents and every parity gate but two run on CPU. Training does not.
            print("\n    CUDA NOT AVAILABLE -- the GUI and the CPU gates work, training will not.")
    except Exception as exc:                           # noqa: BLE001
        print(f"\n    (could not query CUDA: {type(exc).__name__}: {exc})")

    if failures:
        print("\n  PROBLEMS:")
        for label, detail in failures:
            print(f"    {label}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
