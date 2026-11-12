"""Continue, stop or restart? Replay nebius SWE-agent runs under per-task step budgets.

Usage (from the project root, after `uv sync --extra datasets`):

    uv run python benchmarks/budget.py [--seeds 20]

The monitors are fitted and calibrated on 70% of tasks; the remaining tasks (about 22
logged attempts each) are replayed with `simulate_budget`.
"""

from __future__ import annotations

import argparse
import warnings

from haltwise import FirstCrossing, Monitor, StepCountScorer, simulate_budget, split_runs
from haltwise.datasets.swe_agent import download_swe_agent_shard, load_swe_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    runs = [r for r in load_swe_agent([download_swe_agent_shard(s) for s in range(6)]) if r.steps]
    fit, held_out = split_runs(runs, test_share=0.3, seed=0)
    budgets = [50, 100, 200, None]

    for name, monitor in (
        ("FeatureScorer monitor", Monitor()),
        ("calibrated step cap", Monitor(StepCountScorer(), policy=FirstCrossing())),
    ):
        monitor.fit(fit)
        print(f"\n=== {name} (threshold {monitor.threshold:.3f}) ===")
        print(simulate_budget(monitor, held_out, budgets, seeds=args.seeds))


if __name__ == "__main__":
    main()
