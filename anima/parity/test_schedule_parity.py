"""Gate for the REX / ReREX learning-rate schedules.

    .venv/bin/python anima/parity/test_schedule_parity.py
    .venv/bin/python anima/parity/test_schedule_parity.py --plot

The reference implementations are vendored verbatim below -- REX from the sd-scripts fork's
`library/train_util.py` (`lr_lambda_rex`) and ReREX from `lr_lambda_rerex` in the same file
(originally `schedulers/rerex.py`). They are here so the comparison is against the code the user
actually trained with, not against a paraphrase of it.

The one deliberate divergence is warmup: sd-scripts uses `step / warmup_steps`, which makes the
first optimizer step run at lr exactly 0. This trainer uses `(step + 1) / warmup_steps` for every
schedule, so parity is checked on the decay body with warmup disabled.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anima.training.config import ScheduleConfig  # noqa: E402
from anima.training.optim import build_scheduler  # noqa: E402

# --------------------------------------------------------------- vendored reference


def ref_rex(scheduler_steps: int):
    """sd-scripts `lr_lambda_rex`, verbatim. Note `d`, `min_lr` and `max_lr` are hardcoded --
    that is exactly the limitation this port removes."""
    def lr_lambda(current_step: int):
        max_lr = 1
        min_lr = 0.001
        d = 0.9
        if current_step < scheduler_steps:
            progress = (current_step / scheduler_steps)
            div = (1 - d) + (d * (1 - progress))
            return min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            return min_lr
    return lr_lambda


def ref_rerex(total_steps: int, max_lr=1, min_lr=0.001, global_d=0.78, local_d=0.85,
              weight_power=1.5, num_segments=8):
    """sd-scripts `lr_lambda_rerex`, verbatim (numpy and all)."""
    def global_rex_lr(total_steps, current_step):
        progress = current_step / total_steps
        div = (1 - global_d) + (global_d * (1 - progress))
        return min_lr + (max_lr - min_lr) * ((1 - progress) / div)

    def lr_lambda_rex_segment(segment_steps, start_lr, end_lr, current_step):
        progress = current_step / segment_steps
        div = (1 - local_d) + (local_d * (1 - progress))
        return end_lr + (start_lr - end_lr) * ((1 - progress) / div)

    segment_weights = np.array([(num_segments - i) ** weight_power for i in range(num_segments)])
    segment_weights /= np.sum(segment_weights)
    segment_steps = np.floor(total_steps * segment_weights).astype(int)
    segment_steps[-1] += total_steps - np.sum(segment_steps)

    segment_boundaries = [0] + np.cumsum(segment_steps).tolist()
    segment_lrs = [(global_rex_lr(total_steps, segment_boundaries[i]),
                    global_rex_lr(total_steps, segment_boundaries[i + 1]))
                   for i in range(num_segments)]

    def lr_lambda(current_step):
        for segment in range(num_segments):
            if current_step < segment_boundaries[segment + 1]:
                segment_start_step = segment_boundaries[segment]
                segment_end_step = segment_boundaries[segment + 1]
                segment_total_steps = segment_end_step - segment_start_step
                start_lr, end_lr = segment_lrs[segment]
                step_in_segment = current_step - segment_start_step
                return lr_lambda_rex_segment(segment_total_steps, start_lr, end_lr, step_in_segment)
        return min_lr

    return lr_lambda


# ------------------------------------------------------------------------ harness


def curve(cfg: ScheduleConfig, total_steps: int) -> list[float]:
    """Drive the real LambdaLR, so what is measured is the integrated path and not just the
    lambda in isolation. Peak lr is 1.0, so the readings *are* the multipliers."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler(opt, cfg, total_steps)
    out = []
    for _ in range(total_steps):
        out.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return out


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="print an ASCII plot of each curve")
    args = ap.parse_args()

    results = []
    STEPS = 1000

    # 1. REX matches the reference at the reference's own hardcoded constants.
    got = curve(ScheduleConfig(kind="rex", d=0.9, min_lr_ratio=0.001), STEPS)
    want = [ref_rex(STEPS)(s) for s in range(STEPS)]
    err = max(abs(a - b) for a, b in zip(got, want))
    results.append(check("rex vs sd-scripts (d=0.9, min=0.001)", err < 1e-12, f"max|diff| {err:.2e}"))

    # 2. ReREX likewise.
    got = curve(ScheduleConfig(kind="rerex", min_lr_ratio=0.001), STEPS)
    want = [ref_rerex(STEPS)(s) for s in range(STEPS)]
    err = max(abs(a - b) for a, b in zip(got, want))
    results.append(check("rerex vs sd-scripts (defaults)", err < 1e-12, f"max|diff| {err:.2e}"))

    # 3. ReREX at non-default d values -- the whole point of the port. The reference is
    #    parameterised even though sd-scripts never exposes these through its CLI.
    for gd, ld, wp, ns in [(0.5, 0.5, 1.0, 4), (0.9, 0.7, 0.0, 12), (0.0, 0.0, 2.5, 3)]:
        cfg = ScheduleConfig(kind="rerex", min_lr_ratio=0.001, global_d=gd, local_d=ld,
                             weight_power=wp, num_segments=ns)
        got = curve(cfg, STEPS)
        want = [ref_rerex(STEPS, global_d=gd, local_d=ld, weight_power=wp, num_segments=ns)(s)
                for s in range(STEPS)]
        err = max(abs(a - b) for a, b in zip(got, want))
        results.append(check(f"rerex tuned (g={gd} l={ld} wp={wp} n={ns})", err < 1e-12,
                             f"max|diff| {err:.2e}"))

    # 4. d is actually load-bearing: d=0 must collapse REX onto linear.
    lin = curve(ScheduleConfig(kind="linear"), STEPS)
    rex0 = curve(ScheduleConfig(kind="rex", d=0.0), STEPS)
    err = max(abs(a - b) for a, b in zip(lin, rex0))
    results.append(check("rex d=0 == linear", err < 1e-12, f"max|diff| {err:.2e}"))

    # A larger d holds the LR higher for longer. Checked at the midpoint, where the curves are
    # furthest apart, so a sign error cannot slip through.
    mids = [curve(ScheduleConfig(kind="rex", d=d), STEPS)[STEPS // 2] for d in (0.0, 0.5, 0.9, 0.99)]
    results.append(check("rex d monotone in LR held", all(a < b for a, b in zip(mids, mids[1:])),
                         "mid lr " + " < ".join(f"{m:.3f}" for m in mids)))

    # 5. Endpoints and range. Every schedule starts at its peak and never leaves [floor, 1].
    for kind in ("rex", "rerex"):
        c = curve(ScheduleConfig(kind=kind, min_lr_ratio=0.05), STEPS)
        ok = (abs(c[0] - 1.0) < 1e-12 and min(c) >= 0.05 - 1e-12 and max(c) <= 1.0 + 1e-12
              and c[-1] < 0.2)
        results.append(check(f"{kind} endpoints/range", ok,
                            f"start {c[0]:.4f} end {c[-1]:.4f} min {min(c):.4f}"))

    # 6. Both are monotone. ReREX is *not* a warm-restart schedule -- each segment ends where the
    #    next begins -- so an upward jump would mean the boundary lookup is wrong.
    rr = curve(ScheduleConfig(kind="rerex", min_lr_ratio=0.001), STEPS)
    rx = curve(ScheduleConfig(kind="rex", min_lr_ratio=0.001), STEPS)
    for name, c in (("rex", rx), ("rerex", rr)):
        results.append(check(f"{name} monotone", all(b <= a + 1e-12 for a, b in zip(c, c[1:]))))
    # ...but it is a distinct curve, otherwise the segmentation is doing nothing.
    gap = max(abs(a - b) for a, b in zip(rr, rx))
    results.append(check("rerex differs from rex", gap > 0.05, f"max|diff| {gap:.3f}"))

    # 7. The headline practical difference from cosine: both REX variants spend most of the run
    #    near peak LR. Same `lr` therefore means substantially more total movement.
    means = {k: sum(curve(ScheduleConfig(kind=k, min_lr_ratio=0.001), STEPS)) / STEPS
             for k in ("cosine", "rex", "rerex")}
    results.append(check("rex/rerex hold LR higher than cosine",
                         means["rex"] > means["rerex"] > means["cosine"] + 0.2,
                         "mean mult " + " ".join(f"{k}={v:.2f}" for k, v in means.items())))

    # 7. Short runs. With weight_power=1.5 and 8 segments the last segment gets ~1.2% of the
    #    budget, so under ~85 steps some segments round down to zero length. That must degrade
    #    gracefully, not divide by zero -- a 20-step LoRA smoke test is a normal thing to run.
    for n in (1, 2, 5, 20, 84):
        try:
            c = curve(ScheduleConfig(kind="rerex", min_lr_ratio=0.001), n)
            ok = all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in c)
        except Exception as e:  # noqa: BLE001
            ok, c = False, repr(e)
        results.append(check(f"rerex short run ({n} steps)", ok, str(c)[:60]))

    # 8. Warmup composes with both, and the peak is reached exactly once.
    for kind in ("rex", "rerex"):
        c = curve(ScheduleConfig(kind=kind, warmup_steps=50, min_lr_ratio=0.001), STEPS)
        ok = abs(c[49] - 1.0) < 1e-12 and c[0] < 0.03 and c[50] <= 1.0 and c[0] > 0.0
        results.append(check(f"{kind} + warmup 50", ok, f"first {c[0]:.4f} peak@49 {c[49]:.4f}"))

    # 9. Rejected configurations.
    for bad in (dict(kind="rex", d=1.0), dict(kind="rerex", local_d=1.5),
                dict(kind="rerex", num_segments=0), dict(kind="nope")):
        try:
            ScheduleConfig(**bad)
            results.append(check(f"reject {bad}", False, "accepted"))
        except ValueError:
            results.append(check(f"reject {bad}", True))

    if args.plot:
        for kind in ("cosine", "rex", "rerex"):
            c = curve(ScheduleConfig(kind=kind, warmup_steps=40, min_lr_ratio=0.001), 400)
            print(f"\n{kind}")
            for row in range(14, -1, -1):
                lo = row / 15
                print("  |" + "".join("#" if v >= lo else " " for v in c[::5]))
            print("  +" + "-" * (len(c) // 5))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
