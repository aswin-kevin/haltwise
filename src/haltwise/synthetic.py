"""Synthetic agent runs for demos and tests. No downloads, no API keys.

The generator imitates a coding agent with four tools. Every run starts out healthy.
Doomed runs eventually get stuck: they repeat the same failing call, then wander. Some
healthy runs hit a temporary error burst and recover, and some doomed runs look fine
until late. This keeps the problem realistic rather than trivially separable.

Synthetic data shows how haltwise works. It says nothing about how well it works on
your agent. Fit on your own traces for that.
"""

from __future__ import annotations

import random

from haltwise.trace import Run, Step

_TOOLS = ("search", "read_file", "edit_file", "run_tests")


def _healthy_action(rng: random.Random, step_number: int) -> list[Step]:
    tool = rng.choice(_TOOLS)
    call = Step.tool_call(tool, {"target": f"{tool}-{step_number}-{rng.randint(0, 9)}"})
    failed = rng.random() < 0.1
    result = Step.error("tool failed") if failed else Step.tool_result(tool, "ok")
    return [Step.llm("thinking"), call, result]


def _stuck_action(rng: random.Random, stuck_call: Step, step_number: int) -> list[Step]:
    if rng.random() < 0.6:
        return [Step.llm("retrying"), stuck_call, Step.error("same failure again")]
    wander = _healthy_action(rng, step_number)
    if rng.random() < 0.5:
        wander[-1] = Step.error("still failing")
    return wander


def _make_run(rng: random.Random, index: int, tasks: int) -> Run:
    doomed = rng.random() < 0.4
    actions = rng.randint(12, 30) if doomed else rng.randint(4, 14)
    onset = int(actions * rng.uniform(0.2, 0.8)) if doomed else actions
    stuck_call = Step.tool_call("run_tests", {"target": f"tests-{index}"})
    burst_at = rng.randrange(actions) if not doomed and rng.random() < 0.25 else -1

    steps: list[Step] = []
    for number in range(actions):
        if number >= onset:
            steps += _stuck_action(rng, stuck_call, number)
        elif number == burst_at:
            steps += [Step.llm("hmm"), stuck_call, Step.error("transient"), Step.error("again")]
        else:
            steps += _healthy_action(rng, number)

    return Run(steps, succeeded=not doomed, id=f"run-{index}", task=f"task-{index % tasks}")


def make_runs(n: int = 500, *, seed: int = 0, tasks: int | None = None) -> list[Run]:
    """Generate ``n`` labelled synthetic agent runs.

    Args:
        n: Number of runs.
        seed: Random seed; the same seed always gives the same runs.
        tasks: Number of distinct task ids to spread runs across (default: ``n // 4``).
    """
    rng = random.Random(seed)
    task_count = tasks if tasks is not None else max(1, n // 4)
    return [_make_run(rng, index, task_count) for index in range(n)]
