"""The honest scorecard: does each scorer beat a calibrated step cap?

Usage (from the project root, after `uv sync --extra datasets`):

    uv run python benchmarks/audit.py [--splits 5]

For each dataset and each of several task-grouped splits, every monitor is calibrated
to the same 5% false-kill target, on the same calibration tasks (learned scorers train
on a separate part), and evaluated on held-out tasks. Reported per scorer
(mean over splits):

* per-task false-kill rate (the guaranteed quantity), failed runs stopped, and
  failed-run steps saved;
* pooled and within-task AUROC at fixed step counts. Within-task AUROC compares
  attempts on the same task, so it can't be inflated by recognising hard tasks
  (arXiv 2610.35971). It needs repeated attempts, which only the nebius data has.

The step cap is the baseline to beat: much of what failure predictors detect is that
failing runs run long.
"""

from __future__ import annotations

import argparse
import statistics
import warnings
from collections import defaultdict
from collections.abc import Callable

from haltwise import (
    EvaluationReport,
    FirstCrossing,
    Monitor,
    RuleScorer,
    Run,
    StepCountScorer,
    split_runs,
)
from haltwise.datasets.swe_agent import download_swe_agent_shard, load_swe_agent
from haltwise.datasets.swe_smith import download_swe_smith_shard, load_swe_smith

MONITORS: dict[str, Callable[[], Monitor]] = {
    "step cap": lambda: Monitor(StepCountScorer(), policy=FirstCrossing()),
    "RuleScorer": lambda: Monitor(RuleScorer()),
    "FeatureScorer": Monitor,
}


def mean(values: list[float | None]) -> str:
    present = [v for v in values if v is not None]
    return f"{statistics.mean(present):.3f}" if present else "n/a"


def audit(name: str, runs: list[Run], splits: int) -> None:
    runs = [run for run in runs if run.steps]
    print(f"\n##### {name}: {len(runs)} runs, {len({r.task for r in runs})} tasks, {splits} splits")
    reports: dict[str, list[EvaluationReport]] = defaultdict(list)
    for split in range(splits):
        fit, test = split_runs(runs, test_share=0.3, seed=300 + split)
        # Same calibration tasks for every monitor, so thresholds are equally tight.
        train, calibration = split_runs(fit, test_share=0.3, seed=split)
        for label, make in MONITORS.items():
            monitor = make()
            if monitor.scorer.learns_from_data:
                monitor.scorer.fit(train)
            monitor.calibrate(calibration, seed=split)
            reports[label].append(monitor.evaluate(test))

    steps = list(next(iter(reports.values()))[0].auroc_at_steps)
    header = "".join(f"{f'@{k}':>7}" for k in steps)
    columns = f"{'FKR/task':>9}{'stopped':>9}{'saved':>8}"
    print(f"{'':14}{columns}   pooled AUROC{header}   within-task{header}")
    for label, results in reports.items():
        pooled = "".join(f"{mean([r.auroc_at_steps[k] for r in results]):>7}" for k in steps)
        within = "".join(
            f"{mean([r.within_task_auroc_at_steps[k] for r in results]):>7}" for k in steps
        )
        print(
            f"{label:14}{statistics.mean(r.task_false_kill_rate for r in results):>9.1%}"
            f"{statistics.mean(r.failures_stopped for r in results):>9.1%}"
            f"{statistics.mean(r.steps_saved for r in results):>8.1%}"
            f"   {'':12}{pooled}   {'':11}{within}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    nebius = load_swe_agent([download_swe_agent_shard(shard) for shard in range(6)])
    audit("nebius SWE-agent (llama)", nebius, args.splits)
    smith = load_swe_smith([download_swe_smith_shard(shard) for shard in (1, 2)])
    audit("SWE-smith (Claude / GPT-4o)", smith, args.splits)


if __name__ == "__main__":
    main()
