"""Phase B: the live trial. Does haltwise save real tokens on a real agent, and at what cost?

Usage (from the project root, after phase A; needs OLLAMA_API_KEY set in the environment):

    uv run python -m trials.ollama_agent.trial --tasks 1000 1030

Pre-registered protocol (decided before looking at any live results):

1. Choose the monitor with `audit_monitors` on the phase-A logs: step cap, RuleScorer or
   FeatureScorer, whichever saves the most failed-run *tokens* at the target false-kill
   rate across task-grouped splits. Fit the winner on all phase-A runs.
2. On new tasks, under the same per-task token budget and attempt cap, run three arms:
   * continue: never stop; haltwise watches in shadow mode (live false-kill measurement);
   * cold restart: haltwise stops doomed attempts; retry with a fresh attempt;
   * warm restart: as cold, but the retry is told what the stopped attempt was doing.
3. Report solves, tokens per task and solves per 100k tokens, with task-level bootstrap
   intervals for each arm's difference from *continue*.

Runs already in the output file are skipped, so the trial resumes after interruption.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from haltwise import (
    CalibrationError,
    FeatureScorer,
    FirstCrossing,
    Monitor,
    RuleScorer,
    Run,
    StepCountScorer,
    split_runs,
)
from trials.ollama_agent.agent import Attempt, Usage, run_attempt, summarise_for_retry
from trials.ollama_agent.world import make_task

ARMS = ("continue", "cold restart", "warm restart")
CANDIDATES: dict[str, Callable[[float], Monitor]] = {
    "step cap": lambda rate: Monitor(
        StepCountScorer(), policy=FirstCrossing(), false_kill_rate=rate
    ),
    "RuleScorer": lambda rate: Monitor(RuleScorer(), false_kill_rate=rate),
    "FeatureScorer": lambda rate: Monitor(FeatureScorer(), false_kill_rate=rate),
}


def load_phase_a(path: Path) -> tuple[list[Run], list[int]]:
    """Return the phase-A runs (with token counts on steps) and each attempt's tokens."""
    runs, tokens = [], []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        runs.append(Run.from_dict(record["run"]))
        tokens.append(int(record["tokens"]))
    return runs, tokens


def audit_monitors(runs: list[Run], rate: float, splits: int = 5) -> dict[str, float | None]:
    """Mean share of failed-run tokens each candidate would save on held-out tasks.

    A candidate that can't be calibrated in every split (too few tasks with a success
    in its calibration part) is ineligible and reported as ``None``.
    """
    saved: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    eligible = set(CANDIDATES)
    for split in range(splits):
        fit, test = split_runs(runs, test_share=0.3, seed=split)
        for name, make in CANDIDATES.items():
            try:
                report = make(rate).fit(fit, seed=split).evaluate(test)
            except CalibrationError:
                eligible.discard(name)
                continue
            saved[name].append(report.tokens_saved or 0.0)
    return {name: statistics.mean(saved[name]) if name in eligible else None for name in CANDIDATES}


def play_arm(
    arm: str, task_id: int, monitor: Monitor, *, model: str, budget: int, max_attempts: int,
    max_steps: int, usage: Usage,
) -> dict[str, Any]:  # fmt: skip
    """Play one task in one arm until solved, out of budget, or out of attempts."""
    task = make_task(task_id)
    attempts: list[Attempt] = []
    hint: str | None = None
    spent = 0
    while len(attempts) < max_attempts and spent < budget:
        attempt = run_attempt(
            task,
            model=model,
            usage=usage,
            max_steps=max_steps,
            watch=monitor.watch(),
            enforce=arm != "continue",
            hint=hint,
            token_budget=budget - spent,
        )
        attempts.append(attempt)
        spent += attempt.tokens
        if attempt.solved:
            break
        hint = summarise_for_retry(attempt) if arm == "warm restart" and attempt.stopped else None
    return {
        "arm": arm,
        "task_id": task_id,
        "solved": any(a.solved for a in attempts),
        "tokens": spent,
        "attempts": [a.to_dict() for a in attempts],
    }


def bootstrap_difference(
    arm: list[dict[str, Any]], base: list[dict[str, Any]], key: str, seed: int = 0
) -> tuple[float, float, float]:
    """Mean per-task difference (arm - base) in ``key`` with a 95% task-level bootstrap CI."""
    by_task = {r["task_id"]: r for r in base}
    diffs = [
        float(r[key]) - float(by_task[r["task_id"]][key]) for r in arm if r["task_id"] in by_task
    ]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return statistics.mean(diffs), means[50], means[1949]


def report(records: list[dict[str, Any]]) -> None:
    """Print the arm comparison and the live shadow measurements."""
    by_arm = {arm: [r for r in records if r["arm"] == arm] for arm in ARMS}
    print(f"\n{'arm':<14}{'tasks':>6}{'solved':>9}{'tokens/task':>13}{'solves/100k tok':>17}")
    for arm, rows in by_arm.items():
        if not rows:
            continue
        solved = statistics.mean(float(r["solved"]) for r in rows)
        tokens = statistics.mean(r["tokens"] for r in rows)
        print(
            f"{arm:<14}{len(rows):>6}{solved:>9.1%}{tokens:>13,.0f}{1e5 * solved / tokens:>17.2f}"
        )

    base = by_arm["continue"]
    for arm in ARMS[1:]:
        if by_arm[arm] and base:
            s, s_lo, s_hi = bootstrap_difference(by_arm[arm], base, "solved")
            t, t_lo, t_hi = bootstrap_difference(by_arm[arm], base, "tokens")
            print(
                f"{arm} vs continue: solved {s:+.1%} [{s_lo:+.1%}, {s_hi:+.1%}], "
                f"tokens/task {t:+,.0f} [{t_lo:+,.0f}, {t_hi:+,.0f}]"
            )

    shadow = [a for r in base for a in r["attempts"]]
    good = [a for a in shadow if a["solved"]]
    bad = [a for a in shadow if not a["solved"]]
    if good and bad:
        false_kills = sum(a["first_flag_step"] is not None for a in good) / len(good)
        flagged_bad = [a for a in bad if a["first_flag_step"] is not None]
        saveable = sum(a["tokens"] - a["tokens_at_flag"] for a in flagged_bad)
        print(
            f"\nLive shadow check (continue arm, {len(shadow)} attempts): good attempts "
            f"haltwise would have stopped: {false_kills:.1%}; failed attempts flagged: "
            f"{len(flagged_bad) / len(bad):.1%}; failed-attempt tokens it would have saved: "
            f"{saveable / sum(a['tokens'] for a in bad):.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=int, nargs=2, default=[1000, 1030], metavar=("FIRST", "END")
    )
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--rate", type=float, default=0.10, help="target false-kill rate")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=16, help="turn limit per attempt")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phase-a", type=Path, default=Path("data/live/phase_a16.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/live/phase_b16.jsonl"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    runs, attempt_tokens = load_phase_a(args.phase_a)
    print(
        f"Phase A: {len(runs)} runs, {statistics.mean(r.succeeded or 0 for r in runs):.0%} solved"
    )
    audit = audit_monitors(runs, args.rate)
    for name, saved in audit.items():
        result = (
            "ineligible (too few tasks to calibrate)" if saved is None else f"saves {saved:.1%}"
        )
        print(f"  audit: {name:<14} {result} of failed-run tokens (held-out tasks)")
    eligible = {name: saved for name, saved in audit.items() if saved is not None}
    chosen = max(eligible, key=eligible.__getitem__)
    monitor = CANDIDATES[chosen](args.rate).fit(runs)
    budget = 3 * int(statistics.median(attempt_tokens))
    print(
        f"Chosen monitor: {chosen} (threshold {monitor.threshold:.3f}); "
        f"budget {budget:,} tokens/task"
    )

    done = set()
    if args.out.exists():
        done = {
            (r["arm"], r["task_id"]) for r in map(json.loads, args.out.read_text().splitlines())
        }
    jobs = [(arm, t) for t in range(*args.tasks) for arm in args.arms if (arm, t) not in done]
    print(f"{len(jobs)} arm-task pairs to run ({len(done)} already done)")

    usage, lock = Usage(), threading.Lock()

    def work(job: tuple[str, int]) -> None:
        record = play_arm(
            job[0], job[1], monitor, model=args.model, budget=budget,
            max_attempts=args.max_attempts, max_steps=args.max_steps, usage=usage,
        )  # fmt: skip
        with lock, args.out.open("a") as out:
            out.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, jobs))
    print(f"Model calls this session: {usage.calls:,}; tokens: {usage.tokens:,}")

    records = [json.loads(line) for line in args.out.read_text().splitlines()]
    report([r for r in records if args.tasks[0] <= r["task_id"] < args.tasks[1]])


if __name__ == "__main__":
    main()
