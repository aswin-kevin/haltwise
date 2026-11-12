"""Second dataset: SWE-smith trajectories (OpenAI tool-call format, Claude/GPT-4o).

Usage (from the project root, after `uv sync --extra datasets`):

    uv run python benchmarks/swe_smith.py [--shards 1 2] [--seeds 20]

Three questions:
1. In-distribution: does calibration hold its promise on a different agent format?
2. Model shift: calibrate on Claude 3.7 Sonnet, stream Claude 3.5 Sonnet runs.
3. Dataset shift: calibrate on nebius SWE-agent (llama, fenced commands), stream
   SWE-smith runs (Claude/GPT-4o, tool calls). Nothing about the agent is shared.
For 2 and 3, a fixed threshold is compared with AdaptiveMonitor over random orders.
"""

from __future__ import annotations

import argparse
import random
import statistics
import warnings

from haltwise import AdaptiveMonitor, Monitor, RuleScorer, Run, split_runs
from haltwise.datasets.swe_agent import download_swe_agent_shard, load_swe_agent
from haltwise.datasets.swe_smith import DATASET_URL, download_swe_smith_shard, load_swe_smith

SONNET_37 = "claude-3-7-sonnet-20250219"
SONNET_35 = "claude-3-5-sonnet-20241022"


def shuffled(runs: list[Run], seed: int) -> list[Run]:
    copy = list(runs)
    random.Random(seed).shuffle(copy)
    return copy


def compare_streams(title: str, monitor: Monitor, stream: list[Run], seeds: int) -> None:
    print(f"\n=== {title}: {len(stream)} runs, {seeds} random orders ===")
    print(f"{'':10}{'false-kill rate (mean, worst)':>32}{'failed stopped':>17}{'steps saved':>14}")
    for label, adapt in (("fixed", False), ("adaptive", True)):
        reports = [
            AdaptiveMonitor(monitor, seed=seed).dry_run(shuffled(stream, seed), adapt=adapt)
            for seed in range(seeds)
        ]
        fkr = [r.false_kill_rate for r in reports]
        print(
            f"{label:10}{statistics.mean(fkr):>21.1%} {max(fkr):>9.1%}"
            f"{statistics.mean(r.failures_stopped for r in reports):>17.1%}"
            f"{statistics.mean(r.steps_saved for r in reports):>14.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    shards = [download_swe_smith_shard(shard) for shard in args.shards]
    runs = [run for run in load_swe_smith(shards) if run.steps]
    resolved = sum(bool(r.succeeded) for r in runs)
    print(f"Dataset: {DATASET_URL} (tool split, shards {args.shards})")
    tasks = len({r.task for r in runs})
    print(f"{len(runs)} runs, {tasks} tasks, {resolved / len(runs):.1%} resolved")

    # 1. In-distribution guarantee and savings.
    fit, test = split_runs(runs, test_share=0.3, seed=0)
    for name, monitor in (
        ("RuleScorer", Monitor(RuleScorer(), false_kill_rate=args.alpha)),
        ("FeatureScorer", Monitor(false_kill_rate=args.alpha)),
    ):
        monitor.fit(fit)
        print(
            f"\n=== In-distribution, {name} (threshold {monitor.threshold:.3f} "
            f"from {monitor.calibration_size} tasks) ==="
        )
        print(monitor.evaluate(test))

    # 2. Model shift inside SWE-smith.
    sonnet_37 = [r for r in load_swe_smith(shards, models=[SONNET_37]) if r.steps]
    sonnet_35 = [r for r in load_swe_smith(shards, models=[SONNET_35]) if r.steps]
    shifted = Monitor(false_kill_rate=args.alpha).fit(sonnet_37)
    compare_streams(
        "Model shift: calibrated on Sonnet 3.7, streaming Sonnet 3.5",
        shifted,
        sonnet_35,
        args.seeds,
    )

    # 3. Dataset shift: a monitor from a different agent entirely.
    nebius = load_swe_agent([download_swe_agent_shard(s) for s in range(6)])
    foreign = Monitor(false_kill_rate=args.alpha).fit([r for r in nebius if r.steps])
    compare_streams(
        "Dataset shift: calibrated on nebius SWE-agent, streaming SWE-smith",
        foreign,
        runs,
        args.seeds,
    )


if __name__ == "__main__":
    main()
