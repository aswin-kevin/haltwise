"""A small tool-using agent on Ollama Cloud, optionally guarded by a haltwise `Watch`.

The agent answers one task by emitting one JSON action per turn, e.g.
``{"tool": "get_employee", "args": {"id": "E123"}}``. Every model call's real token
count (prompt + completion) is recorded on its step, so savings can be measured in
tokens, not just steps.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from haltwise import Run, Step, Watch
from trials.ollama_agent.tools import TOOL_DOCS, Toolbox
from trials.ollama_agent.world import Task, check_answer, make_world

SYSTEM_PROMPT = f"""You are an agent answering a question about a company database.
You can only learn facts by calling tools. Available tools:
{TOOL_DOCS}

Reply with exactly ONE JSON object per turn and nothing else, for example:
{{"tool": "search_employees", "args": {{"query": "Silva"}}}}
When you know the answer, call submit with ONLY the value (a number like 123000, or a
full name like Ana Silva), no explanation: {{"tool": "submit", "args": {{"answer": "..."}}}}"""

_ENDPOINT = "https://ollama.com/api/chat"


@dataclass
class Usage:
    """Running totals of model calls and tokens, shared by an experiment."""

    calls: int = 0
    tokens: int = 0


def chat(model: str, messages: list[dict[str, str]], temperature: float) -> tuple[str, int]:
    """Send a chat request; return ``(reply, tokens used)``. Retries transient failures."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    tokens = 0
    for attempt in range(5):
        request = urllib.request.Request(
            _ENDPOINT, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                reply: dict[str, Any] = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2**attempt)
            continue
        tokens += int(reply.get("prompt_eval_count", 0)) + int(reply.get("eval_count", 0))
        content = str(reply.get("message", {}).get("content", ""))
        if content.strip():
            return content, tokens
    return "", tokens


def parse_action(reply: str) -> tuple[str, dict[str, Any]] | None:
    """Find the first JSON object with a "tool" key; ``None`` if there is none."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(reply):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(reply[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("tool"), str):
            args = value.get("args")
            return value["tool"], args if isinstance(args, dict) else {}
    return None


@dataclass
class Attempt:
    """One attempt at a task, as it actually happened."""

    task_id: int
    run: Run
    solved: bool
    tokens: int
    stopped: bool = False
    submitted: str | None = None
    first_flag_step: int | None = None  # step index at which haltwise first said stop
    tokens_at_flag: int | None = None
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run": self.run.to_dict(),
            "solved": self.solved,
            "tokens": self.tokens,
            "stopped": self.stopped,
            "submitted": self.submitted,
            "first_flag_step": self.first_flag_step,
            "tokens_at_flag": self.tokens_at_flag,
            "calls": self.calls,
        }


def summarise_for_retry(attempt: Attempt, recent: int = 6) -> str:
    """A short note for a warm restart: what the stopped attempt was doing."""
    actions = [
        f"{s.name}({'error' if s.failed else 'ok'})"
        for s in attempt.run.steps
        if s.kind.value in ("tool_call", "tool_result") and s.name
    ][-recent * 2 :]
    return (
        "Note: a previous attempt at this question was stopped because it was going "
        f"nowhere. Its last actions were: {', '.join(actions) or 'none'}. "
        "Try a different, more direct approach."
    )


def run_attempt(
    task: Task,
    *,
    model: str,
    usage: Usage,
    max_steps: int = 30,
    watch: Watch | None = None,
    enforce: bool = False,
    hint: str | None = None,
    token_budget: int | None = None,
    temperature: float = 0.7,
) -> Attempt:
    """Run one attempt at ``task``.

    Args:
        watch: A haltwise watch. With ``enforce=False`` it only records where it would
            have stopped (shadow mode); with ``enforce=True`` it really stops the run.
        hint: Extra text for the first user message (used for warm restarts).
        token_budget: Stop (as a failure) once this many tokens have been spent.
    """
    toolbox = Toolbox(make_world(task.id))
    question = task.question if hint is None else f"{hint}\n\n{task.question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[Step] = []
    attempt = Attempt(task.id, Run([]), solved=False, tokens=0)

    for _ in range(max_steps):
        if token_budget is not None and attempt.tokens >= token_budget:
            attempt.notes.append("token budget exhausted")
            break
        reply, tokens = chat(model, messages, temperature)
        attempt.tokens += tokens
        attempt.calls += 1
        usage.calls += 1
        usage.tokens += tokens
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        new_steps: list[Step]
        if action is None:
            new_steps = [Step.error("reply was not a JSON action", tokens=tokens)]
            observation = (
                'Error: reply with exactly one JSON object like {"tool": ..., "args": {...}}'
            )
        elif action[0] == "submit":
            attempt.submitted = str(action[1].get("answer", ""))
            new_steps = [Step.tool_call("submit", action[1], tokens=tokens)]
            observation = ""
        else:
            name, args = action
            observation, ok = toolbox.call(name, args)
            new_steps = [
                Step.tool_call(name, args, tokens=tokens),
                Step.tool_result(name, status="ok" if ok else "error"),
            ]
        steps.extend(new_steps)

        if watch is not None:
            for step in new_steps:
                decision = watch.observe(step)
                if decision.should_stop and attempt.first_flag_step is None:
                    attempt.first_flag_step = len(steps) - 1
                    attempt.tokens_at_flag = attempt.tokens
            if enforce and watch.stopped:
                attempt.stopped = True
                break
        if attempt.submitted is not None:
            break
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    attempt.solved = attempt.submitted is not None and check_answer(task, attempt.submitted)
    # A run haltwise stopped has an unknown outcome (it never finished), not a failure.
    outcome = None if attempt.stopped else attempt.solved
    attempt.run = Run(steps, succeeded=outcome, task=f"task-{task.id}")
    return attempt
