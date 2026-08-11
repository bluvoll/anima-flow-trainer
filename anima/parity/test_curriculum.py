"""Gate for the timestep curriculum and the phase->t_range mappings.

Two classes of failure this is built to catch.

**Silent no-ops.** A curriculum that parses, prints and resolves but never reaches the sampler
trains exactly like no curriculum at all, and the loss curve looks fine. So the checks assert on
sampled *distributions*, not on whether the plumbing ran.

**Silent regression of the default path.** `curriculum` empty must be bit-identical to before the
feature existed, which is asserted against a fixed seed rather than argued.

    .venv/bin/python anima/parity/test_curriculum.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima.training.config import load_config  # noqa: E402
from anima.training.curriculum import Curriculum, Phase  # noqa: E402
from anima.training.flow import FlowConfig, sample_timesteps  # noqa: E402

PASS, FAIL = [], []
DEV = torch.device("cpu")


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def draw(cfg: FlowConfig, n: int, t_range=None, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return sample_timesteps(cfg, n, 64, 64, DEV, t_range=t_range)


def q(x: torch.Tensor, p: float) -> float:
    x, _ = x.sort()
    return float(x[min(len(x) - 1, int(p * len(x)))])


def main() -> int:
    N = 200_000
    ln = FlowConfig(timestep_sample_method="logit_normal", sigmoid_scale=1.3)

    # --- default path is untouched -------------------------------------------------------------
    a = draw(ln, N, None, seed=7)
    b = draw(ln, N, (0.0, 1.0), seed=7)
    check("t_range=None and [0,1] rescale are bit-identical", torch.equal(a, b))
    check("unrestricted draw spans [0,1]", float(a.min()) < 0.02 and float(a.max()) > 0.98,
          f"[{float(a.min()):.3f}, {float(a.max()):.3f}]")

    # --- rescale: shape preserved, quantiles scale exactly --------------------------------------
    # This is the property that distinguishes rescale from truncate, so assert it numerically
    # rather than trusting that the affine map was written correctly.
    lo, hi = 0.0, 0.6
    r = draw(ln, N, (lo, hi), seed=7)
    ok = all(abs(q(r, p) - (q(a, p) * (hi - lo) + lo)) < 1e-5 for p in (0.1, 0.25, 0.5, 0.75, 0.9))
    check("rescale maps every quantile by exactly the span", ok,
          f"p50 {q(r, 0.5):.4f} = {q(a, 0.5):.4f} x {hi - lo}")
    check("rescale stays inside the range",
          float(r.min()) >= lo - 1e-6 and float(r.max()) <= hi + 1e-6)

    # --- truncate: different distribution, same support -----------------------------------------
    tr = FlowConfig(timestep_sample_method="logit_normal", sigmoid_scale=1.3,
                    phase_mapping="truncate")
    t = draw(tr, N, (lo, hi), seed=7)
    check("truncate stays inside the range",
          float(t.min()) >= lo - 1e-6 and float(t.max()) <= hi + 1e-6,
          f"[{float(t.min()):.3f}, {float(t.max()):.3f}]")
    check("truncate is skewed higher than rescale (they are not the same mapping)",
          q(t, 0.5) > q(r, 0.5) + 0.02, f"median {q(t, 0.5):.3f} vs {q(r, 0.5):.3f}")
    # Truncation must reproduce the *conditional* of the base distribution: the fraction of the
    # full draw below any threshold inside the range must match the truncated draw's fraction.
    frac_full = float((a[a <= hi] <= 0.3).float().mean())
    frac_trunc = float((t <= 0.3).float().mean())
    check("truncate reproduces the base distribution conditioned on the range",
          abs(frac_full - frac_trunc) < 0.01, f"{frac_full:.3f} vs {frac_trunc:.3f}")
    check("truncate on [0,1] equals the unrestricted draw in distribution",
          abs(q(draw(tr, N, (0.0, 1.0), seed=7), 0.5) - q(a, 0.5)) < 0.005)

    # --- truncate fails loudly on an impossible range -------------------------------------------
    # Silently falling back to rescale (or hanging) would be worse than an error: the phase would
    # train a distribution the config did not ask for.
    narrow = FlowConfig(timestep_sample_method="logit_normal", sigmoid_scale=0.05,
                        phase_mapping="truncate")
    try:
        draw(narrow, 64, (0.999, 1.0), seed=1)
        check("truncate raises on a range with negligible mass", False, "no raise")
    except RuntimeError as e:
        check("truncate raises on a range with negligible mass", "phase_mapping" in str(e))

    # --- shift is applied before the range map --------------------------------------------------
    # The two orders are not equivalent, and getting it backwards would silently change every
    # phase relative to the reference implementation this was ported from.
    sh = FlowConfig(timestep_sample_method="logit_normal", sigmoid_scale=1.3, shift=3.0)
    full_shift = draw(sh, N, None, seed=7)
    ranged = draw(sh, N, (0.0, 0.6), seed=7)
    check("shift is applied before the range map, not after",
          abs(q(ranged, 0.5) - q(full_shift, 0.5) * 0.6) < 1e-5,
          f"{q(ranged, 0.5):.4f} vs {q(full_shift, 0.5) * 0.6:.4f}")

    # --- phase resolution is a step function ----------------------------------------------------
    c = Curriculum([Phase(at=0.0, t_range=(0.0, 1.0), mode="fullres"),
                    Phase(at=0.1, t_range=(0.0, 1.0), mode="texture"),
                    Phase(at=0.15, t_range=(0.0, 0.6), mode="texture")])
    check("progress 0 picks the first phase", c.resolve(0.0).mode == "fullres")
    check("boundary is inclusive (at <= progress)", c.resolve(0.10).mode == "texture"
          and c.resolve(0.0999).mode == "fullres")
    check("last matching phase wins", c.resolve(0.9).t_range == (0.0, 0.6))
    check("progress past 1.0 stays in the final phase", c.resolve(1.7).t_range == (0.0, 0.6),
          "a trailing partial accumulation group does step")
    check("phases are sorted regardless of config order",
          [p.at for p in Curriculum([Phase(at=0.5), Phase(at=0.0)]).phases] == [0.0, 0.5])
    check("empty curriculum is falsy", not Curriculum([]))

    # --- validation ------------------------------------------------------------------------------
    check("discrete timesteps are rejected with a usable message",
          _msg(lambda: Phase(t_range=(0, 600)), "divide by 1000"))
    check("a curriculum not starting at 0.0 is rejected",
          _msg(lambda: Curriculum([Phase(at=0.3)]), "must start at 0.0"))
    check("duplicate `at` is rejected",
          _msg(lambda: Curriculum([Phase(at=0.0), Phase(at=0.0)]), "same `at`"))
    check("lo >= hi is rejected", _raises(lambda: Phase(t_range=(0.6, 0.6))))
    check("at outside [0,1) is rejected", _raises(lambda: Phase(at=1.0)))
    check("unknown mode is rejected", _raises(lambda: Phase(mode="packed")))
    check("lr_mul must be > 0", _raises(lambda: Phase(lr_mul=0.0)))
    check("unknown phase_mapping is rejected",
          _raises(lambda: FlowConfig(phase_mapping="clamp")))

    # --- TOML round trip --------------------------------------------------------------------------
    check("the ported hybrid config loads", _load_hybrid())
    check("[curriculum] as a single table (not an array) is rejected",
          _msg(_load_bad_curriculum, "double"))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


# The reference nine-phase schedule, written here rather than read from `configs/`. Reading a real
# config made this gate depend on a file the user owns and can rename -- and `configs/` is empty on
# a fresh clone. What is being pinned is that the TOML *shape* survives `load_config`: an array of
# tables, `at` as a fraction, `t_range` as a pair, and `phase_mapping` reaching FlowConfig.
_HYBRID_TOML = """
[dataset]
path = "/path/to/data"
source = "encode"

[train]
model_path = "/path/to/model"

[flow]
phase_mapping = "rescale"
""" + "".join(
    f"""
[[curriculum]]
at = {at}
t_range = [{lo}, {hi}]
mode = "{mode}"
"""
    for at, lo, hi, mode in [
        (0.00, 0.0, 1.0, "fullres"),
        (0.10, 0.0, 1.0, "texture"),
        (0.15, 0.0, 0.6, "texture"),
        (0.40, 0.0, 1.0, "texture"),
        (0.45, 0.0, 0.6, "texture"),
        (0.60, 0.0, 1.0, "texture"),
        (0.65, 0.0, 0.6, "texture"),
        (0.80, 0.0, 1.0, "texture"),
        (0.85, 0.0, 0.6, "texture"),
    ]
)


def _load_hybrid() -> bool:
    with tempfile.TemporaryDirectory(prefix="anima_curric_") as td:
        p = Path(td) / "hybrid.toml"
        p.write_text(_HYBRID_TOML)
        return _check_hybrid(load_config(p))


def _check_hybrid(cfg) -> bool:
    phases = cfg.curriculum.phases
    return (len(phases) == 9
            and phases[0].mode == "fullres"
            and all(x.mode == "texture" for x in phases[1:])
            and [x.at for x in phases] == [0.0, 0.10, 0.15, 0.40, 0.45, 0.60, 0.65, 0.80, 0.85]
            and sum(1 for x in phases if x.t_range == (0.0, 0.6)) == 4
            and cfg.flow.phase_mapping == "rescale"
            and cfg.dataset.source == "encode")


def _load_bad_curriculum():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.toml"
        p.write_text('[dataset]\npath = "/tmp"\n[train]\nmodel_path = "x"\n'
                     '[curriculum]\nat = 0.0\n')
        load_config(p)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _msg(fn, needle: str) -> bool:
    try:
        fn()
    except Exception as e:
        return needle in str(e)
    return False


if __name__ == "__main__":
    sys.exit(main())
