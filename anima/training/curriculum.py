"""Training curriculum: what to train, at which noise levels, as the run progresses.

Ported from TrainTrain's `train_ts_schedule` (`trainer/train.py::_parse_ts_schedule_text`), which
is a step function over training progress:

    step_pct   t_min   t_max   [mode]   [lr]

Here it is an array of tables instead of a text block, so the loader type-checks it and the GUI can
render it:

    [[curriculum]]
    at = 0.00
    t_range = [0.0, 1.0]
    mode = "fullres"

    [[curriculum]]
    at = 0.15
    t_range = [0.0, 0.6]
    mode = "texture"
    lr_mul = 0.5

Resolution is a step function: the **last** phase whose `at <= progress` is active. Because
progress is `global_step / total_steps`, every DDP rank resolves the same phase from the same step
count with no communication, and resume lands in the right phase automatically.

Two deliberate differences from the reference:

**`lr_mul`, not an absolute `lr`.** TrainTrain's fifth column pins a learning rate for the phase,
and its own docstring admits the consequence -- "a decaying scheduler and per-phase LR pins fight
each other". This trainer's REX schedule exists precisely for its shape, so a phase multiplies the
scheduler's current LR instead of replacing it. `lr_mul = 1.0` is an exact no-op.

**`t_range` is continuous.** The reference samples integer timesteps in 0..1000; this model takes
t in [0,1] (`t / num_train_timesteps`), so a `0 600` phase is written `[0.0, 0.6]`. Integer ranges
above 1 are rejected rather than silently reinterpreted -- `[0, 600]` means "clamp everything to
pure noise" under the continuous convention, which would train nothing useful and look like a
tuning problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("fullres", "texture")


@dataclass
class Phase:
    at: float = 0.0                                  # progress in [0,1) at which this row activates
    t_range: tuple[float, float] = (0.0, 1.0)
    mode: str = "fullres"
    lr_mul: float = 1.0

    def __post_init__(self):
        self.t_range = tuple(self.t_range)  # TOML gives a list
        if not 0.0 <= self.at < 1.0:
            raise ValueError(f"curriculum.at must be in [0, 1), got {self.at}")
        if len(self.t_range) != 2:
            raise ValueError(f"curriculum.t_range must be [lo, hi], got {self.t_range}")
        lo, hi = self.t_range
        if any(v > 1 for v in self.t_range):
            raise ValueError(
                f"curriculum.t_range {list(self.t_range)} looks like discrete timesteps. This "
                f"model takes t in [0,1] -- divide by 1000, so '0 600' becomes [0.0, 0.6]."
            )
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"curriculum.t_range must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]")
        if self.mode not in MODES:
            raise ValueError(f"curriculum.mode must be one of {MODES}, got {self.mode!r}")
        if self.lr_mul <= 0.0:
            raise ValueError(f"curriculum.lr_mul must be > 0, got {self.lr_mul}")

    def label(self) -> str:
        lo, hi = self.t_range
        s = f"t{lo:g}-{hi:g}/{self.mode}"
        return s + (f"/lr x{self.lr_mul:g}" if self.lr_mul != 1.0 else "")


@dataclass
class Curriculum:
    phases: list[Phase] = field(default_factory=list)

    def __post_init__(self):
        if not self.phases:
            return
        self.phases = sorted(self.phases, key=lambda p: p.at)
        if self.phases[0].at != 0.0:
            raise ValueError(
                f"the first curriculum phase must start at 0.0, but the earliest is "
                f"{self.phases[0].at}. Steps before it would have no phase."
            )
        ats = [p.at for p in self.phases]
        if len(set(ats)) != len(ats):
            dupes = sorted({a for a in ats if ats.count(a) > 1})
            raise ValueError(
                f"two curriculum phases share the same `at` ({dupes}); which one wins would "
                f"depend on config ordering. Give them distinct start points."
            )

    def __bool__(self) -> bool:
        return bool(self.phases)

    def resolve(self, progress: float) -> Phase:
        """The last phase whose `at` <= progress. `progress` is clamped, not validated: a run that
        overshoots its estimated total (a trailing partial accumulation group does step) must stay
        in the final phase rather than fall off the end."""
        active = self.phases[0]
        for p in self.phases:
            if p.at <= progress:
                active = p
            else:
                break
        return active

    def report(self, total_steps: int) -> str:
        # `--dry-run` pins total_steps to 1, which would make every phase "round to zero steps" and
        # fire the warning nine times on a run that was never going to execute them. A warning that
        # always fires in a routine mode is one people learn to skip past, so it is suppressed
        # there -- the run is still described, just without the false alarm.
        meaningful = total_steps > len(self.phases)
        lines = [f"curric  {len(self.phases)} phase(s)"
                 + ("" if meaningful else "   (step counts omitted: too few steps to divide)")]
        for i, p in enumerate(self.phases):
            end = self.phases[i + 1].at if i + 1 < len(self.phases) else 1.0
            span = f"{p.at:>5.2f}-{end:<5.2f}"
            if not meaningful:
                lines.append(f"        {span} {p.label()}")
                continue
            steps = round((end - p.at) * total_steps)
            warn = "   <- rounds to zero steps" if steps == 0 else ""
            lines.append(f"        {span} {p.label():<28s} ~{steps:>5d} steps{warn}")
        return "\n".join(lines)
