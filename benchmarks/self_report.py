"""Does asking an LLM add signal beyond behavioural features? (SWE-agent runs)

Usage (from the project root; needs OLLAMA_API_KEY set in the environment):

    uv run python benchmarks/self_report.py [--model gpt-oss:20b] [--runs 200]

Protocol:
* `FeatureScorer` is trained on the fit split (by task) of shards 0-5, as in
  `benchmarks/swe_agent.py`.
* Held-out runs (with text) are sampled from shard 0: half succeeded, half failed, at
  most one run per task where possible.
* At 25/50/75/100% of each run, the LLM sees the most recent steps (with text) and
  estimates the chance of failure. The feature scorer's comparable number is its
  highest risk so far.
* Answers are cached in data/self_report_cache.jsonl, so reruns cost nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from haltwise import FeatureScorer, Run, split_runs
from haltwise.datasets.swe_agent import download_swe_agent_shard, load_swe_agent
from haltwise.evaluation import auroc
from haltwise.llm import ollama
from haltwise.scorers import SelfReportScorer

PROGRESS = (0.25, 0.5, 0.75, 1.0)
CACHE = Path("data/self_report_cache.jsonl")


def sample_runs(runs: list[Run], per_outcome: int, seed: int) -> list[Run]:
    """Pick ``per_outcome`` successes and failures, preferring distinct tasks."""
    rng = random.Random(seed)
    picked: list[Run] = []
    for outcome in (True, False):
        pool = [run for run in runs if run.succeeded is outcome and run.steps]
        rng.shuffle(pool)
        seen: set[str | None] = set()
        first_per_task = [r for r in pool if not (r.task in seen or seen.add(r.task))]
        rest = [r for r in pool if r not in first_per_task]
        picked += (first_per_task + rest)[:per_outcome]
    rng.shuffle(picked)  # never let sample order line up with outcomes
    return picked


def load_cache() -> dict[str, float | None]:
    if not CACHE.exists():
        return {}
    entries = (json.loads(line) for line in CACHE.read_text().splitlines() if line)
    return {entry["key"]: entry["risk"] for entry in entries}


def ask_with_retry(ask: Callable[[str], str], prompt: str, attempts: int = 4) -> str:
    for attempt in range(attempts):
        try:
            return ask(prompt)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def bootstrap_auroc(pos: list[float], neg: list[float], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos), np.asarray(neg)
    draws = [
        auroc(list(rng.choice(p, len(p))), list(rng.choice(n, len(n)))) or 0.5 for _ in range(1000)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def percentile_ranks(values: list[float]) -> list[float]:
    """Rank-transform to [0, 1], giving tied values their average rank.

    Tie-aware ranks matter: LLM answers are coarse (70, 80, 90...), and breaking ties by
    position would leak the order of the sample into the scores.
    """
    _, inverse, counts = np.unique(np.asarray(values), return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    return list(average_rank[inverse] / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    shards = [download_swe_agent_shard(shard) for shard in range(6)]
    fit, test = split_runs(load_swe_agent(shards), test_share=0.3, seed=0)
    features = FeatureScorer().fit(fit)
    test_tasks = {run.task for run in test}

    with_text = [r for r in load_swe_agent(shards[0], keep_text=True) if r.task in test_tasks]
    sample = sample_runs(with_text, args.runs // 2, seed=0)
    print(f"Evaluating {len(sample)} held-out runs from {len({r.task for r in sample})} tasks")

    self_report = SelfReportScorer(ollama(args.model), every=1)
    cache = load_cache()
    jobs = []
    for run in sample:
        for point in PROGRESS:
            visible = max(1, math.ceil(point * len(run.steps)))
            key = f"{args.model}|{run.id}|{visible}"
            if key not in cache:
                recent = list(enumerate(run.steps[:visible]))[-self_report.recent_steps :]
                jobs.append((key, self_report.prompt(recent)))

    print(f"{len(jobs)} model calls needed ({len(cache)} cached)")
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool, CACHE.open("a") as out:
        answers = pool.map(lambda job: (job[0], ask_with_retry(self_report.ask, job[1])), jobs)
        for key, answer in answers:
            cache[key] = self_report.read_risk(answer)
            out.write(json.dumps({"key": key, "answer": answer, "risk": cache[key]}) + "\n")
    print(
        f"done in {time.perf_counter() - started:.0f}s; unreadable answers: "
        f"{self_report.unreadable_answers}"
    )

    print(f"\n{'progress':>9}  {'features':>20}  {'LLM self-report':>20}  {'combined':>20}")
    for point in PROGRESS:
        rows = []  # (failed, feature risk, llm risk)
        for run in sample:
            visible = max(1, math.ceil(point * len(run.steps)))
            llm = cache[f"{args.model}|{run.id}|{visible}"]
            feature = max(features.risk_path(run.steps[:visible]))
            rows.append((run.failed, feature, 0.5 if llm is None else llm))
        combined = [
            (a + b) / 2
            for a, b in zip(
                percentile_ranks([r[1] for r in rows]),
                percentile_ranks([r[2] for r in rows]),
                strict=True,
            )
        ]
        cells = []
        for values in ([r[1] for r in rows], [r[2] for r in rows], combined):
            pos = [v for v, r in zip(values, rows, strict=True) if r[0]]
            neg = [v for v, r in zip(values, rows, strict=True) if not r[0]]
            low, high = bootstrap_auroc(pos, neg)
            cells.append(f"{auroc(pos, neg):.3f} [{low:.2f}-{high:.2f}]")
        print(f"{point:>9.0%}  " + "  ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
