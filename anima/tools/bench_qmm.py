"""A/B quantized matmul on the config you actually train with.

    .venv/bin/python -m anima.tools.bench_qmm configs/full_finetune.toml --steps 16

Runs the config twice -- `use_quantized_matmul` false then true, everything else identical -- and
reports MEDIAN as well as mean step time.

The median is the point. Quantized-matmul kernels are Triton JIT and compile per shape, so a
bucketed dataset pays a compilation spike the first time each bucket appears, at any point in the
run. Those spikes move the mean by 2x while the median barely shifts, so a mean-only comparison
cannot tell "this is slower" from "this is still compiling". Both are printed, and their gap is
what tells you how much of the cost is one-time.

Step 1 is always dropped (model load and first allocations land in it), and both arms use the same
seed, so they see the same bucket sequence and the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

STEP_RE = re.compile(r"step (\d+)/\d+.*?([0-9.]+)s/it")


def run(config: Path, qmm: bool, steps: int, gpu: str) -> list[float]:
    text = re.sub(r"\[quant\]\n(?:[^\[]*\n)?",
                  f'[quant]\nmode = "training"\nuse_quantized_matmul = {str(qmm).lower()}\n',
                  config.read_text(), count=1)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        env = {"CUDA_VISIBLE_DEVICES": gpu}
        proc = subprocess.run(
            [sys.executable, "-m", "anima.training.train", tmp,
             "--max-steps", str(steps), "--no-save"],
            capture_output=True, text=True, check=False,   # a failed arm is reported, not raised
            env={**os.environ, **env},
        )
        out = proc.stdout + proc.stderr
        times = [(int(m[0]), float(m[1])) for m in STEP_RE.findall(out)]
        if not times:
            print(out[-2000:])
            raise SystemExit(f"no step lines parsed (qmm={qmm})")
        return [t for step, t in times if step > 1]
    finally:
        Path(tmp).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES for both arms")
    args = ap.parse_args()

    print(f"{'qmm':>5} {'n':>3} {'median':>8} {'mean':>8} {'max':>8}   interpretation")
    results = {}
    for qmm in (False, True):
        ts = run(Path(args.config), qmm, args.steps, args.gpu)
        results[qmm] = ts
        spikes = sum(1 for t in ts if t > 3 * statistics.median(ts))
        note = f"{spikes} compile spike(s)" if spikes else "no spikes"
        print(f"{qmm!s:>5} {len(ts):>3} {statistics.median(ts):>8.2f} "
              f"{statistics.mean(ts):>8.2f} {max(ts):>8.2f}   {note}")

    off, on = results[False], results[True]
    md = statistics.median(on) / statistics.median(off)
    mn = statistics.mean(on) / statistics.mean(off)
    print(f"\nqmm ON vs OFF:  median {md:.2f}x   mean {mn:.2f}x")
    print("median ~1.0 and mean >> 1.0 means the cost is JIT compilation, not the kernels;"
          "\nit is paid again whenever a bucket shape appears for the first time.")
    print("VERDICT: " + ("qmm is not paying for itself here -- leave it off (the `auto` default)."
                         if md >= 0.95 else f"qmm is {1/md:.2f}x faster in steady state."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
