"""Behavioural features of a run so far, updated one step at a time.

The features capture the warning signs reported across the agent-monitoring literature:
repeating the same call, error streaks, no new actions for a long time, and runs that
grow long. They need no message text, so they work even when text is redacted.
"""

from __future__ import annotations

import math
from collections import Counter, deque

from haltwise.trace import Step, StepKind

FEATURE_NAMES: tuple[str, ...] = (
    "log_steps",
    "error_rate",
    "recent_error_rate",
    "error_streak",
    "repeat_in_window",
    "distinct_call_ratio",
    "log_steps_since_new_call",
    "tool_call_fraction",
)
"""Order of the values returned by `TraceFeatures.vector`."""


class TraceFeatures:
    """Running summary of a trace. Feed it steps with `update`; read it with `vector`.

    Args:
        window: How many recent steps (and recent tool calls) count as "recent".
    """

    def __init__(self, window: int = 10) -> None:
        self.steps = 0
        self.errors = 0
        self.error_streak = 0
        self.tool_calls = 0
        self.steps_since_new_call = 0
        self._seen_calls: set[tuple[str | None, str | None]] = set()
        self._recent_failures: deque[bool] = deque(maxlen=window)
        self._recent_calls: deque[tuple[str | None, str | None]] = deque(maxlen=window)

    def update(self, step: Step) -> None:
        """Account for one more step."""
        self.steps += 1
        self.steps_since_new_call += 1
        self._recent_failures.append(step.failed)

        if step.failed:
            self.errors += 1
            self.error_streak += 1
        elif step.kind is StepKind.TOOL_RESULT:
            self.error_streak = 0

        signature = step.call_signature
        if signature is not None:
            self.tool_calls += 1
            self._recent_calls.append(signature)
            if signature not in self._seen_calls:
                self._seen_calls.add(signature)
                self.steps_since_new_call = 0

    @property
    def repeat_in_window(self) -> int:
        """Most times any single call (same tool, same arguments) appears among recent calls."""
        counts = Counter(self._recent_calls)
        return max(counts.values(), default=0)

    def as_dict(self) -> dict[str, float]:
        """Return the features keyed by name."""
        if self.steps == 0:
            return dict.fromkeys(FEATURE_NAMES, 0.0)
        return {
            "log_steps": math.log1p(self.steps),
            "error_rate": self.errors / self.steps,
            "recent_error_rate": sum(self._recent_failures) / len(self._recent_failures),
            "error_streak": float(self.error_streak),
            "repeat_in_window": float(self.repeat_in_window),
            "distinct_call_ratio": (
                len(self._seen_calls) / self.tool_calls if self.tool_calls else 1.0
            ),
            "log_steps_since_new_call": math.log1p(self.steps_since_new_call),
            "tool_call_fraction": self.tool_calls / self.steps,
        }

    def vector(self) -> list[float]:
        """Return the features in `FEATURE_NAMES` order."""
        values = self.as_dict()
        return [values[name] for name in FEATURE_NAMES]
