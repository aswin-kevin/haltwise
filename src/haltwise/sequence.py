"""Which recent *sequences* of steps tend to precede failure?

Each step is reduced to a symbol, ``kind:tool:status`` (e.g.
``tool_result:run_tests:error``), and the last ``order`` symbols form the run's current
**state**. From past runs we learn, for every state, the share of runs passing through it
that ended in failure. While a run is live, the rates of the states it visits become
extra features for `haltwise.FeatureScorer`.

This is a lightweight cousin of the state-machine monitors in *Automata from Agent
Traces* (arXiv 2608.23670) and PrefixGuard (arXiv 2605.06455): no automaton is built,
but the same "where in the behaviour space is this run?" signal is captured. Its gains
on our benchmarks are modest; run `benchmarks/audit.py` to compare scorers.

Evidence is counted in *tasks*, not runs: forty near-copy attempts at one task are one
piece of evidence. Rare states are shrunk towards the overall failure rate (a Beta
prior), and states seen in fewer than ``min_count`` tasks are dropped, so a handful of
tasks cannot be memorised. `cross_fit` builds out-of-fold tables, so a classifier trained
on these features never sees rates learned from its own training task.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from haltwise.splits import task_key
from haltwise.trace import Run, Step

State = tuple[str, ...]

SEQUENCE_FEATURE_NAMES: tuple[str, ...] = (
    "state_failure_rate",
    "max_state_failure_rate",
    "mean_state_failure_rate",
    "recent_state_failure_rate",
)
"""Order of the values returned by `StateTracker.update`."""

_RECENT = 5


def symbol(step: Step) -> str:
    """Reduce a step to its behavioural symbol, e.g. ``tool_result:run_tests:error``."""
    return f"{step.kind.value}:{step.name or ''}:{step.status.value}"


def states(steps: Sequence[Step], order: int) -> Iterator[State]:
    """Yield the state (last ``order`` symbols) after each step."""
    window: deque[str] = deque(maxlen=order)
    for step in steps:
        window.append(symbol(step))
        yield tuple(window)


class StateRates:
    """A learned table: state -> smoothed share of runs through it that failed.

    Args:
        order: How many recent steps make up a state.
        prior_strength: How many runs' worth of weight the overall failure rate gets
            when smoothing a state's rate. Higher means more shrinkage.
        min_count: States seen in fewer tasks than this are ignored (base rate used).
    """

    def __init__(self, order: int = 3, prior_strength: float = 20.0, min_count: int = 5) -> None:
        if order < 1:
            raise ValueError(f"order must be at least 1 (got {order})")
        self.order = order
        self.prior_strength = prior_strength
        self.min_count = min_count
        self.base_rate = 0.5
        self.rates: dict[State, float] = {}

    def fit(self, runs: Sequence[Run]) -> StateRates:
        """Learn state failure rates from labelled runs (unlabelled runs are skipped)."""
        labelled = [run for run in runs if run.succeeded is not None]
        if not labelled:
            raise ValueError("StateRates needs labelled runs to learn from")
        self.base_rate = sum(run.failed for run in labelled) / len(labelled)

        seen: Counter[State] = Counter()
        failed: Counter[State] = Counter()
        tasks: dict[State, set[str]] = {}
        for index, run in enumerate(labelled):
            task = task_key(run, index)
            for state in set(states(run.steps, self.order)):  # each run counts once per state
                seen[state] += 1
                failed[state] += run.failed
                tasks.setdefault(state, set()).add(task)

        self.rates = {}
        for state, runs_through in seen.items():
            evidence = len(tasks[state])  # tasks, not near-copy runs
            if evidence >= self.min_count:
                observed = failed[state] / runs_through
                self.rates[state] = (observed * evidence + self.prior_strength * self.base_rate) / (
                    evidence + self.prior_strength
                )
        return self

    def rate(self, state: State) -> float:
        """Return the learned failure rate of ``state`` (the base rate if unknown)."""
        return self.rates.get(state, self.base_rate)

    def new_tracker(self) -> StateTracker:
        """Start following a new run."""
        return StateTracker(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON."""
        return {
            "order": self.order,
            "prior_strength": self.prior_strength,
            "min_count": self.min_count,
            "base_rate": self.base_rate,
            "rates": [[list(state), rate] for state, rate in self.rates.items()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateRates:
        """Rebuild from `to_dict` output."""
        table = cls(data["order"], data["prior_strength"], data["min_count"])
        table.base_rate = data["base_rate"]
        table.rates = {tuple(state): rate for state, rate in data["rates"]}
        return table


def cross_fit(
    runs: Sequence[Run], *, folds: int = 5, seed: int = 0, **table_options: Any
) -> dict[str, StateRates]:
    """Return, for each task key, a `StateRates` table fitted *without* that task.

    Tasks are dealt into ``folds`` groups; each group gets a table fitted on the other
    groups. Use these tables to build training features, so the downstream classifier
    sees the same kind of (out-of-sample) rates it will see in production.

    Args:
        runs: Labelled runs.
        folds: Number of task groups (fewer if there are fewer tasks).
        seed: Seed for dealing tasks into groups.
        table_options: Passed to `StateRates` (``order``, ``prior_strength``, ``min_count``).
    """
    if folds < 2:
        raise ValueError(f"cross_fit needs folds >= 2 (got {folds})")
    keys = [task_key(run, index) for index, run in enumerate(runs)]
    unique = sorted(set(keys))
    if len(unique) < 2:
        raise ValueError("cross_fit needs runs from at least two tasks")
    random.Random(seed).shuffle(unique)
    fold_of = {
        task: position % max(1, min(folds, len(unique))) for position, task in enumerate(unique)
    }

    tables: dict[str, StateRates] = {}
    for fold in set(fold_of.values()):
        others = [run for run, key in zip(runs, keys, strict=True) if fold_of[key] != fold]
        table = StateRates(**table_options)
        if any(run.succeeded is not None for run in others):
            table.fit(others)
        for task, task_fold in fold_of.items():
            if task_fold == fold:
                tables[task] = table
    return tables


class StateTracker:
    """Follows one run and returns its `SEQUENCE_FEATURE_NAMES` values after each step."""

    def __init__(self, table: StateRates) -> None:
        self._table = table
        self._window: deque[str] = deque(maxlen=table.order)
        self._recent: deque[float] = deque(maxlen=_RECENT)
        self._count = 0
        self._total = 0.0
        self._max = 0.0

    def update(self, step: Step) -> list[float]:
        """Account for ``step`` and return the sequence features."""
        self._window.append(symbol(step))
        rate = self._table.rate(tuple(self._window))
        self._count += 1
        self._total += rate
        self._max = max(self._max, rate)
        self._recent.append(rate)
        return [rate, self._max, self._total / self._count, sum(self._recent) / len(self._recent)]
