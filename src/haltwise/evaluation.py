"""Measure a monitor on held-out runs: does it keep its promise, and what does it save?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from haltwise.policies import Policy
from haltwise.scorers.base import Scorer
from haltwise.splits import task_key
from haltwise.trace import Run

STEP_POINTS: tuple[int, ...] = (5, 10, 20, 40)
"""Fixed step counts at which discrimination is measured.

Fixed steps, not fractions of each run: "the first 25% of a run" depends on how long
the run turned out to be, which leaks the outcome into an "early" score.
"""


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half).

    Returns ``None`` when either group is empty.
    """
    if not positives or not negatives:
        return None
    # Mann-Whitney U from average ranks: O(n log n) and tie-aware.
    values = np.concatenate([np.asarray(positives, float), np.asarray(negatives, float)])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    n_pos, n_neg = len(positives), len(negatives)
    u_statistic = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def within_task_auroc(samples: Sequence[tuple[str, bool, float]]) -> float | None:
    """AUROC comparing failed and successful attempts *on the same task*.

    ``samples`` are ``(task, failed, score)`` triples. Each task with both outcomes
    contributes its own AUROC, weighted by its number of (failed, successful) pairs.
    A scorer that only recognises hard tasks scores 0.5 here, however good its pooled
    AUROC looks ("Continue or Restart?", arXiv 2610.35971). Returns ``None`` when no task
    has both outcomes.
    """
    by_task: dict[str, tuple[list[float], list[float]]] = {}
    for task, failed, score in samples:
        failed_scores, successful_scores = by_task.setdefault(task, ([], []))
        (failed_scores if failed else successful_scores).append(score)

    weighted = pairs = 0.0
    for failed_scores, successful_scores in by_task.values():
        task_auroc = auroc(failed_scores, successful_scores)
        if task_auroc is not None:
            weight = len(failed_scores) * len(successful_scores)
            weighted += task_auroc * weight
            pairs += weight
    return weighted / pairs if pairs else None


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@dataclass(frozen=True)
class EvaluationReport:
    """How a monitor did on a set of labelled runs.

    Attributes:
        runs: Labelled runs evaluated (unlabelled runs are skipped).
        successes: How many of them succeeded.
        failures: How many failed.
        target_false_kill_rate: The rate the monitor was calibrated for, if any.
        false_kill_rate: Share of successful runs the monitor would have stopped.
        task_false_kill_rate: The same, averaged per task: for each task, the share of
            its successful runs that would be stopped, then the mean over tasks. This is
            the quantity the calibration guarantee bounds.
        failures_stopped: Share of failed runs the monitor would have stopped.
        steps_saved: Share of all failed-run steps that stopping would have skipped.
        tokens_saved: Same, weighted by tokens; ``None`` when tokens aren't recorded.
        median_stop_progress: For stopped failures, how far in (0–1) the stop came.
        auroc: How well the whole-run risk separates failures from successes. Includes
            the run's full length, so it flatters any scorer that counts steps.
        auroc_at_steps: Pooled AUROC using only each run's first k steps (runs shorter
            than k are left out), for k in `STEP_POINTS`.
        within_task_auroc_at_steps: The same, but comparing attempts on the same task
            only (see `within_task_auroc`). The honest test of an *early* warning.
    """

    runs: int
    successes: int
    failures: int
    target_false_kill_rate: float | None
    false_kill_rate: float
    task_false_kill_rate: float
    failures_stopped: float
    steps_saved: float
    tokens_saved: float | None
    median_stop_progress: float | None
    auroc: float | None
    auroc_at_steps: dict[int, float | None] = field(default_factory=dict)
    within_task_auroc_at_steps: dict[int, float | None] = field(default_factory=dict)

    def __str__(self) -> str:
        target = (
            f" (target {self.target_false_kill_rate:.0%})"
            if self.target_false_kill_rate is not None
            else ""
        )
        rows = [
            ("runs evaluated", f"{self.runs} ({self.successes} succeeded, {self.failures} failed)"),
            ("false-kill rate per task", f"{_percent(self.task_false_kill_rate)}{target}"),
            ("false-kill rate per run", _percent(self.false_kill_rate)),
            ("failed runs stopped", _percent(self.failures_stopped)),
            ("failed-run steps saved", _percent(self.steps_saved)),
            ("failed-run tokens saved", _percent(self.tokens_saved)),
            ("median stop point", _percent(self.median_stop_progress)),
            ("AUROC (whole run)", _score(self.auroc)),
            *[(f"AUROC at step {k}", _score(v)) for k, v in self.auroc_at_steps.items()],
            *[
                (f"within-task AUROC at step {k}", _score(v))
                for k, v in self.within_task_auroc_at_steps.items()
            ],
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def evaluate(
    runs: Sequence[Run],
    *,
    scorer: Scorer,
    policy: Policy,
    threshold: float,
    target_false_kill_rate: float | None = None,
) -> EvaluationReport:
    """Replay labelled runs through a scorer and policy and summarise what would happen.

    Prefer `haltwise.Monitor.evaluate`, which fills in these arguments for you.
    """
    labelled = [(task_key(run, i), run) for i, run in enumerate(runs) if run.succeeded is not None]
    stats: dict[bool, list[float]] = {True: [], False: []}  # keyed by: did the run fail?
    early: dict[int, list[tuple[str, bool, float]]] = {k: [] for k in STEP_POINTS}
    kills_by_task: dict[str, list[bool]] = {}
    stopped_failures = 0
    failed_steps = skipped_steps = 0
    failed_tokens = skipped_tokens = 0
    stop_points: list[float] = []

    for task, run in labelled:
        risks = scorer.risk_path(run.steps)
        failed = run.failed
        stats[failed].append(policy.path_statistic(risks))
        for k in STEP_POINTS:
            if len(risks) >= k:
                early[k].append((task, failed, policy.path_statistic(risks[:k])))

        stop = policy.stop_index(risks, threshold)
        if not failed:
            kills_by_task.setdefault(task, []).append(stop is not None)
            continue

        tokens = [step.tokens or 0 for step in run.steps]
        failed_steps += len(run.steps)
        failed_tokens += sum(tokens)
        if stop is not None:
            stopped_failures += 1
            skipped_steps += len(run.steps) - (stop + 1)
            skipped_tokens += sum(tokens[stop + 1 :])
            stop_points.append((stop + 1) / len(run.steps))

    successes, failures = len(stats[False]), len(stats[True])
    return EvaluationReport(
        runs=len(labelled),
        successes=successes,
        failures=failures,
        target_false_kill_rate=target_false_kill_rate,
        false_kill_rate=_fraction(sum(sum(k) for k in kills_by_task.values()), successes),
        task_false_kill_rate=_fraction(
            sum(sum(kills) / len(kills) for kills in kills_by_task.values()), len(kills_by_task)
        ),
        failures_stopped=_fraction(stopped_failures, failures),
        steps_saved=_fraction(skipped_steps, failed_steps),
        tokens_saved=_fraction(skipped_tokens, failed_tokens) if failed_tokens else None,
        median_stop_progress=sorted(stop_points)[len(stop_points) // 2] if stop_points else None,
        auroc=auroc(stats[True], stats[False]),
        auroc_at_steps={
            k: auroc([v for _, f, v in samples if f], [v for _, f, v in samples if not f])
            for k, samples in early.items()
        },
        within_task_auroc_at_steps={k: within_task_auroc(s) for k, s in early.items()},
    )
