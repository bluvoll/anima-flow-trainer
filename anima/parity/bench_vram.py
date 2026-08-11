"""Measure the full-finetune VRAM ladder, one rung per subprocess.

Each rung runs in its own process because peak-memory accounting does not reset cleanly in-process:
a freed 4GB optimizer state leaves the caching allocator holding the block, and the next rung's
"peak" would inherit it. Separate processes make each number stand on its own.

The claim being tested is that a 2B full finetune does not fit on a 4090 without help, and that
each rung buys a specific amount. An OOM here is a result, not a failure -- it is recorded.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (label, overrides). Each dict is {section: {key: value}} applied over the base config.
RUNGS = [
    ("baseline: fp32 AdamW, no grad ckpt", {
        "train": {"gradient_checkpointing": False},
        "optimizer": {"kind": "adamw", "quantize_state": False, "offload_state": False},
    }),
    ("+ gradient checkpointing", {
        "train": {"gradient_checkpointing": True},
        "optimizer": {"kind": "adamw", "quantize_state": False, "offload_state": False},
    }),
    ("+ sdnq adamw (bf16 states)", {
        "train": {"gradient_checkpointing": True},
        "optimizer": {"kind": "adamw8bit", "quantize_state": False, "offload_state": False},
    }),
    ("+ quantized optimizer state", {
        "train": {"gradient_checkpointing": True},
        "optimizer": {"kind": "adamw8bit", "quantize_state": True, "offload_state": False},
    }),
    ("+ offloaded optimizer state", {
        "train": {"gradient_checkpointing": True},
        "optimizer": {"kind": "adamw8bit", "quantize_state": True, "offload_state": True},
    }),
]

_PEAK = re.compile(r"peak ([\d.]+)GB")
_RATE = re.compile(r"([\d.]+)s/it")


def _dump_toml(data: dict) -> str:
    """Minimal TOML writer -- enough for these configs, and avoids a dependency for one call."""
    lines = []
    for section, body in data.items():
        lines.append(f"[{section}]")
        for k, v in body.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, dict):
                inner = ", ".join(f"{ik} = {iv}" for ik, iv in v.items())
                lines.append(f"{k} = {{ {inner} }}")
            elif isinstance(v, list):
                lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)


def run_rung(base: dict, overrides: dict, steps: int, gpu: str) -> tuple[str, str]:
    cfg = {k: dict(v) for k, v in base.items()}
    for section, body in overrides.items():
        cfg.setdefault(section, {}).update(body)

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, dir=ROOT) as f:
        f.write(_dump_toml(cfg))
        path = f.name

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    try:
        out = subprocess.run(
            [sys.executable, "-m", "anima.training.train", path,
             "--max-steps", str(steps), "--no-save"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
        )
    finally:
        os.unlink(path)

    text = out.stdout + out.stderr
    if "out of memory" in text.lower():
        return "OOM", "-"
    peaks = _PEAK.findall(text)
    rates = _RATE.findall(text)
    if not peaks:
        tail = "\n".join(text.strip().splitlines()[-3:])
        return f"FAILED ({tail[:120]})", "-"
    # Skip the first step: it carries allocator warmup and one-off quantization.
    steady = rates[1:] or rates
    return f"{max(float(p) for p in peaks):.1f} GB", f"{sum(map(float, steady)) / len(steady):.2f} s/it"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="a full-finetune TOML to use as the base")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    base = tomllib.loads(Path(args.config).read_text())
    # `[dataset.caption]` is nested; the flat writer would mangle it, and it does not affect memory.
    base.get("dataset", {}).pop("caption", None)

    print(f"\nfull-finetune VRAM ladder  ({args.steps} steps, GPU {args.gpu})\n")
    print(f"  {'rung':<38} {'peak':>10} {'speed':>10}")
    print(f"  {'-' * 38} {'-' * 10} {'-' * 10}")
    for label, overrides in RUNGS:
        peak, rate = run_rung(base, overrides, args.steps, args.gpu)
        print(f"  {label:<38} {peak:>10} {rate:>10}", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
