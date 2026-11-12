"""Benchmark haltwise on real SWE-agent runs (nebius/SWE-agent-trajectories).

Usage (from the project root, after `uv sync --extra datasets`):

    uv run python benchmarks/swe_agent.py [--shards 0 1] [--alpha 0.05]

Protocol: split runs by task (GitHub issue) into 70% fit / 30% test. `Monitor.fit`
itself holds out 30% of the fit tasks for calibration. The test split is used only
once, for the report. Seeds are fixed.
"""

from __future__ import annotations

import argparse
import time

from haltwise import Monitor, RuleScorer, Run, split_runs
from haltwise.datasets.swe_agent import DATASET_URL, download_swe_agent_shard, load_swe_agent


def describe(name: str, runs: list[Run]) -> None:
    successes = sum(bool(run.succeeded) for run in runs)
    print(f"{name}: {len(runs)} runs, {successes / len(runs):.1%} succeeded")


def report(title: str, monitor: Monitor, test: list[Run]) -> None:
    print(f"\n=== {title} ===")
    print(f"threshold {monitor.threshold:.3f} from {monitor.calibration_size} calibration tasks")
    print(monitor.evaluate(test))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, nargs="+", default=[0])
    parser.add_argument("--alpha", type=float, default=0.05, help="target false-kill rate")
    args = parser.parse_args()

    print(f"Dataset: {DATASET_URL} (shards {args.shards})")
    started = time.perf_counter()
    runs = load_swe_agent([download_swe_agent_shard(shard) for shard in args.shards])
    fit, test = split_runs(runs, test_share=0.3, seed=0)
    describe("fit", fit)
    describe("test", test)

    rules = Monitor(RuleScorer(), false_kill_rate=args.alpha).fit(fit)
    report("RuleScorer (no training, calibrated)", rules, test)

    learned = Monitor(false_kill_rate=args.alpha).fit(fit)
    report("FeatureScorer (learned, calibrated)", learned, test)

    # Transfer: calibrate on the 70B model's runs, test on the 8B model's runs.
    shards = [download_swe_agent_shard(shard) for shard in args.shards]
    big = load_swe_agent(shards, models=["swe-agent-llama-70b"])
    small = load_swe_agent(shards, models=["swe-agent-llama-8b"])
    shifted = Monitor(false_kill_rate=args.alpha).fit(big)
    report("Transfer: fit on llama-70b, test on llama-8b (no recalibration)", shifted, small)

    print(f"\nTotal time {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
