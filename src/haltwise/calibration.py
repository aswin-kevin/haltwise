"""Turn "kill at most α of good runs" into a concrete stopping threshold.

The method is split conformal prediction applied to whole trajectories. For each
held-out *successful* run we compute the policy's path statistic: the largest threshold
at which the policy would have stopped that run. The threshold is the
⌈(n+1)(1-α)⌉-th smallest of those n values.

Guarantee: if future successful runs are exchangeable with the calibration runs (same
agent, same kind of tasks), the chance that a future successful run gets stopped is at
most α. This is a *marginal* guarantee (on average over runs and calibration sets), not
a promise about any single run. When the agent, model, prompt or tools change,
recalibrate.

Prior art: StopLoss (arXiv 2610.25525) calibrates its abort threshold on successful runs
in the same spirit, taking the maximum score over checkpoints.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from typing import TypeVar

from haltwise.errors import CalibrationError

T = TypeVar("T")

# Guards against float noise such as 20 * 0.9 == 18.000000000000004 rounding up.
_EPSILON = 1e-9


def validate_rate(false_kill_rate: float) -> None:
    """Raise ValueError unless ``false_kill_rate`` is strictly between 0 and 1."""
    if not 0.0 < false_kill_rate < 1.0:
        raise ValueError(
            f"false_kill_rate must be strictly between 0 and 1 (got {false_kill_rate})"
        )


def validate_confidence(confidence: float | None) -> None:
    """Raise ValueError unless ``confidence`` is None or strictly between 0 and 1."""
    if confidence is not None and not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1 (got {confidence})")


def min_calibration_runs(false_kill_rate: float, confidence: float | None = None) -> int:
    """Return the fewest successful calibration runs that make ``false_kill_rate`` reachable.

    Average guarantee: 9 runs for 10%, 19 for 5%, 99 for 1%. With ``confidence``, the
    largest calibration score must already be safe with that probability, which needs
    (1 - α)^n <= 1 - confidence: e.g. 59 runs for 5% at 95% confidence.
    """
    validate_rate(false_kill_rate)
    validate_confidence(confidence)
    if confidence is None:
        return math.ceil((1.0 - false_kill_rate) / false_kill_rate - _EPSILON)
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - false_kill_rate) - _EPSILON)


def _binomial_cdf_upto(n: int, p: float) -> list[float]:
    """Return [P(Binomial(n, p) <= j) for j = 0..n], computed stably."""
    log_p, log_q = math.log(p), math.log1p(-p)
    cdf, total = [], 0.0
    for j in range(n + 1):
        log_pmf = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        total += math.exp(log_pmf + j * log_p + (n - j) * log_q)
        cdf.append(min(total, 1.0))
    return cdf


def conformal_threshold(
    statistics: Sequence[float],
    false_kill_rate: float,
    *,
    confidence: float | None = None,
    unit: str = "successful runs",
) -> float:
    """Return the stopping threshold for the given successful-run path statistics.

    Two kinds of guarantee:

    * ``confidence=None`` (default): a new successful run is stopped with probability at
      most α, **on average** over calibration sets. Uses the ⌈(n+1)(1-α)⌉-th smallest
      statistic.
    * ``confidence=c``: with probability at least c over the calibration set, the true
      share of successful runs stopped is at most α. This is a stronger promise about
      *your* threshold, not just the average one. With the threshold at the k-th smallest
      of n statistics, the true kill rate is Beta(n+1-k, k) distributed, so we pick the
      smallest k with P(Binomial(n, α) <= n-k) <= 1 - c (Vovk 2012; Angelopoulos & Bates,
      "A Gentle Introduction to Conformal Prediction", 2021).

    Args:
        statistics: One path statistic per held-out *successful* run
            (see `haltwise.policies.Policy.path_statistic`).
        false_kill_rate: Target fraction α of successful runs that may be stopped.
        confidence: Optional probability that the target holds for this threshold.
        unit: What one statistic stands for, used in error messages.

    Raises:
        CalibrationError: If there are too few runs to reach the target.
    """
    validate_rate(false_kill_rate)
    validate_confidence(confidence)
    needed = min_calibration_runs(false_kill_rate, confidence)
    n = len(statistics)
    if n < needed:
        level = f" at {confidence:.0%} confidence" if confidence is not None else ""
        raise CalibrationError(
            f"A {false_kill_rate:.0%} false-kill rate{level} needs at least {needed} {unit} "
            f"for calibration, but only {n} were given. Add more data, allow a higher "
            f"false_kill_rate, or lower the confidence."
        )
    ordered = sorted(statistics)
    if confidence is None:
        rank = math.ceil((n + 1) * (1.0 - false_kill_rate) - _EPSILON)
        return ordered[rank - 1]

    cdf = _binomial_cdf_upto(n, false_kill_rate)
    # default=0: at the exact minimum size, rounding could otherwise leave no candidate.
    allowed_above = max((j for j in range(n + 1) if cdf[j] <= 1.0 - confidence), default=0)
    return ordered[n - allowed_above - 1]


def one_per_group(items: Sequence[T], groups: Sequence[Hashable], *, seed: int = 0) -> list[T]:
    """Keep one randomly chosen item from each group (e.g. one run per task).

    Runs on the same task tend to behave alike, so they are not independent. Calibrating
    on all of them makes the threshold look far more certain than it is: a task with 40
    near-identical successful runs would count 40 times. Keeping one random run per
    task restores exchangeability between tasks. The guarantee then covers a successful
    run on a *new* task, and the effective sample size is honestly the number of tasks.
    (This is the "single subsample" recipe for hierarchical data; see Dunn, Wasserman &
    Ramdas, "Distribution-free prediction sets for two-layer hierarchical models", 2022.)
    """
    if len(items) != len(groups):
        raise ValueError("items and groups must have the same length")
    members: dict[Hashable, list[T]] = {}
    for item, group in zip(items, groups, strict=True):
        members.setdefault(group, []).append(item)
    rng = random.Random(seed)
    return [rng.choice(members[group]) for group in sorted(members, key=repr)]
