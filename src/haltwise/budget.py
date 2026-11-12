"""What should you *do* when a run looks doomed? Replay your logs and find out.

Stopping a doomed run only pays off if the saved budget is used well. `simulate_budget`
replays your logged runs, grouped by task, under a per-task step budget and compares:

* **single attempt**: one attempt, never stopped (the baseline for *give up*);
* **give up**: one attempt, stopped early when the monitor says so;
* **continue**: always finish; retry only after an attempt fails on its own;
* **restart**: stop doomed attempts early and retry with a fresh attempt;
* **random restart**: stop attempts just as often and as early as the monitor does,
  but at random. If *restart* can't beat this, the monitor isn't choosing *which*
  attempts to stop any better than chance.

Each task's logged runs are treated as independent attempts, and a restart draws the next
unused one (a cold restart). "Continue or Restart?" (arXiv 2610.35971) found continuing
beat restarting in 216 of 224 configurations on coding agents; this lets you check that
for *your* agent before wiring restarts into production.

Limits: steps stand in for cost; warm restarts (keeping progress) and switching to a
stronger model can't be replayed from logs; attempts are assumed independent.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from haltwise.errors import NotCalibratedError
from haltwise.monitor import Monitor
from haltwise.splits import task_key
from haltwise.trace import Run


class Strategy(str, Enum):
    """What to do with a task's attempts under a budget."""

    SINGLE = "single attempt"
    GIVE_UP = "give up"
    CONTINUE = "continue"
    RESTART = "restart"
    RANDOM_RESTART = "random restart"


@dataclass(frozen=True)
class Attempt:
    """One logged attempt, reduced to what the replay needs.

    Attributes:
        length: Steps the attempt took when run to the end.
        succeeded: Whether it succeeded when run to the end.
        stop_index: Step at which the monitor would stop it (``None``: never).
    """

    length: int
    succeeded: bool
    stop_index: int | None


StopProfile = tuple[float, list[float]]
"""How often the monitor stops attempts, and at which fractions of their length."""


def _planned_stop(
    attempt: Attempt, strategy: Strategy, rng: random.Random, profile: StopProfile
) -> int | None:
    if strategy in (Strategy.RESTART, Strategy.GIVE_UP):
        return attempt.stop_index
    if strategy is Strategy.RANDOM_RESTART:
        stop_rate, fractions = profile
        if fractions and rng.random() < stop_rate:
            steps = math.ceil(rng.choice(fractions) * attempt.length)
            return min(max(steps, 1), attempt.length) - 1
    return None


def play_task(
    attempts: Sequence[Attempt],
    budget: int | None,
    strategy: Strategy,
    rng: random.Random,
    *,
    stop_profile: StopProfile,
) -> tuple[bool, int]:
    """Play one task's attempts, in order, under a step budget.

    Returns ``(solved, steps_spent)``. A stopped attempt never reveals its outcome, so
    stopping an attempt that would have succeeded costs the solve. An attempt that would
    overrun the remaining budget is cut off and fails.
    """
    remaining = math.inf if budget is None else budget
    spent = 0
    for attempt in attempts:
        if remaining <= 0:
            break
        stop = _planned_stop(attempt, strategy, rng, stop_profile)
        planned = attempt.length if stop is None else stop + 1
        if planned > remaining:
            return False, spent + int(remaining)
        spent += planned
        remaining -= planned
        if stop is None and attempt.succeeded:
            return True, spent
        if strategy in (Strategy.SINGLE, Strategy.GIVE_UP):
            return False, spent
    return False, spent


@dataclass(frozen=True)
class BudgetRow:
    """Results of one strategy at one budget, averaged over tasks and random orders."""

    strategy: Strategy
    budget: int | None
    solve_rate: float
    steps_per_task: float

    @property
    def solves_per_1000_steps(self) -> float:
        """Tasks solved per thousand steps spent: the efficiency of the strategy."""
        return 1000 * self.solve_rate / self.steps_per_task if self.steps_per_task else 0.0


@dataclass(frozen=True)
class BudgetReport:
    """Every strategy at every budget. Print it for a table."""

    rows: list[BudgetRow]
    tasks: int
    mean_attempts_per_task: float
    seeds: int

    def row(self, strategy: Strategy, budget: int | None) -> BudgetRow:
        """Return the result for one strategy and budget."""
        return next(r for r in self.rows if r.strategy is strategy and r.budget == budget)

    def __str__(self) -> str:
        lines = [
            f"{self.tasks} tasks, {self.mean_attempts_per_task:.1f} logged attempts per task, "
            f"{self.seeds} random orders",
            f"{'budget':>8}  {'strategy':<16}{'solved':>8}{'steps/task':>12}"
            f"{'solves/1k steps':>17}",
        ]
        for row in self.rows:
            budget = "none" if row.budget is None else str(row.budget)
            lines.append(
                f"{budget:>8}  {row.strategy.value:<16}{row.solve_rate:>8.1%}"
                f"{row.steps_per_task:>12.1f}{row.solves_per_1000_steps:>17.2f}"
            )
        return "\n".join(lines)


def simulate_budget(
    monitor: Monitor,
    runs: Sequence[Run],
    budgets: Sequence[int | None] = (50, 100, 200, None),
    *,
    seeds: int = 20,
    min_attempts: int = 2,
) -> BudgetReport:
    """Replay logged runs to compare stop / restart / continue strategies under budgets.

    Use runs the monitor was *not* trained or calibrated on, with several attempts per
    task. Every strategy sees the same random order of attempts (a paired comparison).

    Args:
        monitor: A calibrated monitor (its threshold decides when attempts are stopped).
        runs: Labelled runs with task ids.
        budgets: Steps allowed per task; ``None`` means unlimited.
        seeds: Number of random attempt orders to average over.
        min_attempts: Tasks with fewer logged attempts are left out.
    """
    threshold = monitor.threshold
    if threshold is None:
        raise NotCalibratedError("simulate_budget needs a calibrated monitor (call fit first)")

    tasks: dict[str, list[Attempt]] = {}
    for index, run in enumerate(runs):
        if run.succeeded is None or not run.steps:
            continue
        risks = monitor.scorer.risk_path(run.steps)
        attempt = Attempt(
            len(run.steps), bool(run.succeeded), monitor.policy.stop_index(risks, threshold)
        )
        tasks.setdefault(task_key(run, index), []).append(attempt)
    tasks = {key: attempts for key, attempts in tasks.items() if len(attempts) >= min_attempts}
    if not tasks:
        raise ValueError(
            f"No task has at least {min_attempts} logged attempts; the simulation needs "
            "repeated attempts per task to replay restarts."
        )

    all_attempts = [attempt for attempts in tasks.values() for attempt in attempts]
    stopped = [a for a in all_attempts if a.stop_index is not None]
    profile: StopProfile = (
        len(stopped) / len(all_attempts),
        [(a.stop_index + 1) / a.length for a in stopped if a.stop_index is not None],
    )

    totals: dict[tuple[Strategy, int | None], list[int]] = {
        (strategy, budget): [0, 0] for budget in budgets for strategy in Strategy
    }
    for seed in range(seeds):
        rng = random.Random(seed)
        for key in sorted(tasks):
            order = list(tasks[key])
            rng.shuffle(order)
            for (strategy, budget), total in totals.items():
                solved, spent = play_task(order, budget, strategy, rng, stop_profile=profile)
                total[0] += solved
                total[1] += spent

    runs_played = len(tasks) * seeds
    rows = [
        BudgetRow(strategy, budget, solved / runs_played, spent / runs_played)
        for (strategy, budget), (solved, spent) in totals.items()
    ]
    return BudgetReport(
        rows=rows,
        tasks=len(tasks),
        mean_attempts_per_task=len(all_attempts) / len(tasks),
        seeds=seeds,
    )
