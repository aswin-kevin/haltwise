"""Benchmark AdaptiveMonitor against a fixed threshold when the agent changes.

Usage (from the project root, after `uv sync --extra datasets`):

    uv run python benchmarks/online.py [--shards 0 1 2 3 4 5] [--seeds 20]

Setup: calibrate a monitor on llama-70b runs (split by task), then stream runs in
random order through three scenarios:

* no shift: held-out llama-70b runs (the adaptive monitor should not hurt);
* shift: llama-8b runs (the fixed threshold breaks its promise here);
* switch: held-out llama-70b runs, then llama-8b runs (drift in the middle).

Outcomes are fed back as in production: stopped runs reveal nothing.
"""

from __future__ import annotations

import argparse
import random
import statistics
import warnings
from collections.abc import Callable

from haltwise import AdaptiveMonitor, Monitor, Run, split_runs
from haltwise.datasets.swe_agent import download_swe_agent_shard, load_swe_agent


def shuffled(runs: list[Run], seed: int) -> list[Run]:
    copy = list(runs)
    random.Random(seed).shuffle(copy)
    return copy


def summarise(
    name: str, adaptive: AdaptiveMonitor, make_stream: Callable[[int], list[Run]], seeds: int
) -> None:
    rows: dict[str, list[tuple[float, float, float]]] = {"fixed": [], "adaptive": []}
    for seed in range(seeds):
        stream = make_stream(seed)
        for label, adapt in (("fixed", False), ("adaptive", True)):
            replay = AdaptiveMonitor(
                adaptive.monitor,
                step_size=adaptive.step_size,
                shadow_share=adaptive.shadow_share,
                seed=seed,
            )
            report = replay.dry_run(stream, adapt=adapt)
            rows[label].append(
                (report.false_kill_rate, report.failures_stopped, report.steps_saved)
            )

    print(f"\n=== {name}: {len(make_stream(0))} runs, {seeds} random orders ===")
    print(f"{'':10}{'false-kill rate (mean, worst)':>32}{'failed stopped':>17}{'steps saved':>14}")
    for label, results in rows.items():
        fkr = [r[0] for r in results]
        print(
            f"{label:10}{statistics.mean(fkr):>21.1%} {max(fkr):>9.1%}"
            f"{statistics.mean(r[1] for r in results):>17.1%}"
            f"{statistics.mean(r[2] for r in results):>14.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    shards = [download_swe_agent_shard(shard) for shard in args.shards]
    big = load_swe_agent(shards, models=["swe-agent-llama-70b"])
    small = load_swe_agent(shards, models=["swe-agent-llama-8b"])
    fit, held_out = split_runs(big, test_share=0.3, seed=0)

    monitor = Monitor(false_kill_rate=args.alpha).fit(fit)
    adaptive = AdaptiveMonitor(monitor)
    print(f"Calibrated on llama-70b: threshold {monitor.threshold:.3f}, target {args.alpha:.0%}")
    print(f"AdaptiveMonitor: step_size {adaptive.step_size}, shadow_share {adaptive.shadow_share}")

    summarise("No shift (llama-70b)", adaptive, lambda seed: shuffled(held_out, seed), args.seeds)
    summarise("Shift (llama-8b)", adaptive, lambda seed: shuffled(small, seed), args.seeds)
    summarise(
        "Switch mid-stream (llama-70b, then llama-8b)",
        adaptive,
        lambda seed: shuffled(held_out, seed)[: len(small)] + shuffled(small, seed),
        args.seeds,
    )


if __name__ == "__main__":
    main()
