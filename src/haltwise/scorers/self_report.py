"""Risk from asking an LLM how the run is going (verbal self-assessment)."""

from __future__ import annotations

import re
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

from haltwise.errors import HaltwiseError
from haltwise.llm import Ask
from haltwise.scorers.base import Scorer
from haltwise.trace import Step

_NUMBER = r"\d+(?:\.\d+)?"
_PERCENT = re.compile(rf"({_NUMBER})\s*%")
_OUT_OF = re.compile(rf"({_NUMBER})\s*(?:/|out of)\s*({_NUMBER})", re.IGNORECASE)
_SCALE_RANGE = re.compile(rf"{_NUMBER}\s*(?:-|–|to)\s*{_NUMBER}")
_ANY_NUMBER = re.compile(_NUMBER)


def parse_risk(answer: str) -> float | None:
    """Read a failure probability in [0, 1] from a model's answer; ``None`` if absent.

    Rules, in order:

    1. A percentage wins: "85%", "0.5%".
    2. If the whole answer is "X/N" or "X out of N", it is X/N: "7/10", "85 out of 100".
    3. Otherwise scale descriptions ("0-100", "3/10") are ignored and the last remaining
       number is used: "On a 0-100 scale: 85", "Step 3/10 failed; risk 90".
       Integers are percentages; a decimal up to 1 ("0.85") is a probability.
    """
    text = answer.strip()
    percent = _PERCENT.search(text)
    if percent is not None:
        return min(float(percent.group(1)), 100.0) / 100.0

    whole = _OUT_OF.fullmatch(text)
    if whole is not None and float(whole.group(2)) > 0:
        return min(float(whole.group(1)) / float(whole.group(2)), 1.0)

    remaining = _SCALE_RANGE.sub(" ", text)
    remaining = _OUT_OF.sub(lambda m: m.group(1) if float(m.group(2)) == 100 else " ", remaining)
    numbers = _ANY_NUMBER.findall(remaining)
    if not numbers:
        return None
    last = numbers[-1]
    if "." in last and float(last) <= 1.0:
        return float(last)
    return min(float(last), 100.0) / 100.0


PROMPT = """You are monitoring an AI agent while it works on a task.
Here are its most recent steps (oldest first):

{steps}

Based only on this behaviour, how likely is it that this run will FAIL to complete its task?
Answer with a single integer from 0 (certainly succeeds) to 100 (certainly fails)."""


def _describe(index: int, step: Step, text_chars: int) -> str:
    line = f"[{index}] {step.kind.value}"
    if step.name:
        line += f" {step.name}"
    line += f" ({step.status.value})"
    if step.text:
        snippet = " ".join(step.text.split())[:text_chars]
        line += f": {snippet}"
    return line


class SelfReportScorer(Scorer):
    """Ask a language model, every few steps, how likely the run is to fail.

    Needs no training data, and it reads message text when steps carry it. That makes it
    complementary to the behavioural scorers. Research suggests such verbal confidence is
    weak early in long runs and strong near the end ("Last Step Matters", arXiv
    2608.29685); BAGEN (arXiv 2606.00198) stops runs on the model's own infeasibility
    judgement. Calibrate it like any scorer. Note that calibration then costs one model call
    per ``every`` steps of each calibration run.

    Args:
        ask: Function that sends a prompt and returns the reply, e.g.
            ``haltwise.llm.ollama("gpt-oss:20b")``.
        every: Ask after every ``every`` steps; risk is held constant in between.
            Before the first answer the risk is 0.
        recent_steps: How many recent steps to show the model.
        text_chars: Characters of each step's text to include.
    """

    def __init__(
        self, ask: Ask, *, every: int = 5, recent_steps: int = 20, text_chars: int = 300
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1 (got {every})")
        self.ask = ask
        self.every = every
        self.recent_steps = recent_steps
        self.text_chars = text_chars
        self.unreadable_answers = 0
        self._lock = threading.Lock()

    def new_tracker(self) -> _SelfReportTracker:
        """Start following a new run."""
        return _SelfReportTracker(self)

    def prompt(self, recent: list[tuple[int, Step]]) -> str:
        """Build the question for the model from recent ``(index, step)`` pairs."""
        lines = "\n".join(_describe(i, step, self.text_chars) for i, step in recent)
        return PROMPT.format(steps=lines)

    def read_risk(self, answer: str) -> float | None:
        """Turn the model's answer into a risk in [0, 1] (see `parse_risk`)."""
        risk = parse_risk(answer)
        if risk is None:
            with self._lock:
                self.unreadable_answers += 1
        return risk

    def to_dict(self) -> dict[str, Any]:
        """Refuse: the model client is code, not data."""
        raise HaltwiseError(
            "SelfReportScorer holds a live model client and cannot be saved as JSON; "
            "rebuild it in code and set the threshold from the saved monitor."
        )

    @classmethod
    def _from_params(cls, params: Mapping[str, Any]) -> SelfReportScorer:
        raise HaltwiseError("SelfReportScorer cannot be loaded from JSON; rebuild it in code.")


class _SelfReportTracker:
    def __init__(self, scorer: SelfReportScorer) -> None:
        self._scorer = scorer
        self._recent: deque[tuple[int, Step]] = deque(maxlen=scorer.recent_steps)
        self._risk = 0.0
        self._steps_seen = 0

    def update(self, step: Step) -> float:
        self._recent.append((self._steps_seen, step))
        self._steps_seen += 1
        if self._steps_seen % self._scorer.every == 0:
            answer = self._scorer.ask(self._scorer.prompt(list(self._recent)))
            risk = self._scorer.read_risk(answer)
            if risk is not None:
                self._risk = risk
        return self._risk
