"""Splitting runs without leaking tasks across the split."""

from __future__ import annotations

import random
from collections.abc import Sequence

from haltwise.trace import Run


def task_key(run: Run, index: int) -> str:
    """Return the grouping key of a run: its task, else its id, else its position."""
    if run.task is not None:
        return f"task:{run.task}"
    if run.id is not None:
        return f"run:{run.id}"
    return f"index:{index}"


def split_runs(
    runs: Sequence[Run], test_share: float = 0.2, *, seed: int = 0
) -> tuple[list[Run], list[Run]]:
    """Split runs into ``(rest, held_out)``, keeping every task entirely on one side.

    If runs on the same task landed on both sides, a model could memorise
    task-specific patterns and look better than it is, and the calibration guarantee
    would be over-optimistic. Runs without a ``task`` are treated as their own task.

    Tasks are held out in random order, but only while that moves the held-out size
    towards the target. So the held-out side leans slightly towards smaller tasks (tasks
    with fewer runs). Keep that in mind if attempts per task track task difficulty.

    Args:
        runs: Runs to split.
        test_share: Approximate fraction of runs to hold out.
        seed: Random seed; the same seed gives the same split.
    """
    if not 0.0 < test_share < 1.0:
        raise ValueError(f"test_share must be strictly between 0 and 1 (got {test_share})")

    groups: dict[str, list[Run]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(task_key(run, index), []).append(run)

    if len(groups) < 2:
        raise ValueError("split_runs needs runs from at least two tasks")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target = test_share * len(runs)
    held_keys: list[str] = []
    rest_keys: list[str] = []
    held_size = 0
    for key in keys:
        size = len(groups[key])
        # Hold the task out only if that moves the held-out size closer to the target,
        # so one huge task can't swamp either side.
        if abs(held_size + size - target) < abs(held_size - target):
            held_keys.append(key)
            held_size += size
        else:
            rest_keys.append(key)
    if not held_keys:  # tiny inputs: both sides always get at least one task
        held_keys.append(rest_keys.pop())
    elif not rest_keys:
        rest_keys.append(held_keys.pop())

    rest = [run for key in rest_keys for run in groups[key]]
    held_out = [run for key in held_keys for run in groups[key]]
    return rest, held_out
