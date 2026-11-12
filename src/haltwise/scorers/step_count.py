"""A calibrated ``max_steps``: the simplest monitor, and the baseline every scorer must beat.

Most agent loops already have a step limit, usually picked by gut feeling. Set it too
low and good runs get cut off; set it too high and doomed runs burn money.
`calibrate_step_cap` picks it from your past runs with the same guarantee as the rest of
haltwise: a successful run on a new task exceeds the cap with probability at most α.

It is also an honest baseline. "Continue or Restart?" (arXiv 2610.35971) shows that much
of what early-failure predictors detect is simply that failing runs run long. On our
SWE-agent benchmarks a calibrated step cap captures most of the savings of the learned
scorers (see `benchmarks/audit.py`). Always compare against it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from haltwise.calibration import conformal_threshold, one_per_group
from haltwise.scorers.base import Scorer
from haltwise.splits import task_key
from haltwise.trace import Run, Step

_COMFORTABLE_TASKS = 100


def calibrate_step_cap(
    runs: Sequence[Run],
    false_kill_rate: float = 0.05,
    *,
    confidence: float | None = None,
    seed: int = 0,
) -> int:
    """Return the largest number of steps to allow, with a false-kill guarantee.

    Uses one random successful run per task (see `haltwise.calibration.one_per_group`).
    A successful run on a new task, from the same agent setup, needs more than the
    returned number of steps with probability at most ``false_kill_rate``.

    Args:
        runs: Past runs with outcomes and, ideally, task ids.
        false_kill_rate: Target share of successful runs you accept cutting off.
        confidence: Optional stricter guarantee, as in `haltwise.Monitor`: with this
            probability the target holds for the returned cap, not just on average.
        seed: Seed for choosing one run per task.

    Raises:
        CalibrationError: If too few tasks have a successful run.
    """
    successes = [(task_key(run, i), run) for i, run in enumerate(runs) if run.succeeded]
    chosen = one_per_group([run for _, run in successes], [t for t, _ in successes], seed=seed)
    lengths = [float(len(run)) for run in chosen]
    cap = conformal_threshold(
        lengths, false_kill_rate, confidence=confidence, unit="tasks with a successful run"
    )
    if len(lengths) < _COMFORTABLE_TASKS:
        warnings.warn(
            f"Step cap based on only {len(lengths)} tasks with a successful run; the "
            f"realised false-kill rate may vary noticeably. {_COMFORTABLE_TASKS}+ tasks "
            "give a steadier cap.",
            UserWarning,
            stacklevel=2,
        )
    return int(cap)


@dataclass(frozen=True)
class StepCountScorer(Scorer):
    """Risk that grows with every step and ignores everything else: ``n / (n + scale)``.

    Paired with `FirstCrossing` and calibrated, it becomes a calibrated step cap, usable
    anywhere a scorer is, including `AdaptiveMonitor`, which then keeps the cap
    calibrated as your agent changes. ``scale`` only sets where risk passes 0.5 and does
    not affect decisions after calibration.
    """

    scale: float = 50.0

    def new_tracker(self) -> _StepCounter:
        """Start following a new run."""
        return _StepCounter(self.scale)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON."""
        return {"type": type(self).__name__, **asdict(self)}

    @classmethod
    def _from_params(cls, params: Mapping[str, Any]) -> StepCountScorer:
        return cls(**params)


class _StepCounter:
    def __init__(self, scale: float) -> None:
        self._scale = scale
        self._steps = 0

    def update(self, step: Step) -> float:
        self._steps += 1
        return self._steps / (self._steps + self._scale)
