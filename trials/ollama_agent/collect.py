"""Phase A: collect labelled runs of the live agent (no stopping), resumably.

Usage (from the project root; needs OLLAMA_API_KEY set in the environment):

    uv run python -m trials.ollama_agent.collect --tasks 0 50 --attempts 4

Each finished attempt is appended to the output file at once, and attempts already in
the file are skipped, so an interrupted collection resumes without paying twice.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trials.ollama_agent.agent import Usage, run_attempt
from trials.ollama_agent.world import make_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs=2, default=[0, 50], metavar=("FIRST", "END"))
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("data/live/phase_a.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, int]] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            record = json.loads(line)
            if record["model"] == args.model:
                done.add((record["task_id"], record["attempt"]))
    jobs = [
        (task_id, attempt)
        for task_id in range(*args.tasks)
        for attempt in range(args.attempts)
        if (task_id, attempt) not in done
    ]
    print(f"{len(jobs)} attempts to run with {args.model} ({len(done)} already done)")

    usage, lock, started = Usage(), threading.Lock(), time.perf_counter()

    def work(job: tuple[int, int]) -> dict[str, object]:
        task_id, attempt_no = job
        attempt = run_attempt(
            make_task(task_id), model=args.model, usage=usage, max_steps=args.max_steps
        )
        record = {**attempt.to_dict(), "attempt": attempt_no, "model": args.model}
        with lock, args.out.open("a") as out:
            out.write(json.dumps(record) + "\n")
        return record

    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(work, jobs))

    if records:
        solved = sum(bool(r["solved"]) for r in records)
        steps = [len(r["run"]["steps"]) for r in records]  # type: ignore[index]
        print(
            f"solved {solved}/{len(records)} ({solved / len(records):.0%}); "
            f"median steps {statistics.median(steps)}; calls {usage.calls}; "
            f"tokens {usage.tokens:,}; {time.perf_counter() - started:.0f}s"
        )


if __name__ == "__main__":
    main()
