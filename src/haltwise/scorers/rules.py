"""Rule-based risk: no training data needed. Good for a cold start."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from haltwise.features import TraceFeatures
from haltwise.scorers.base import Scorer
from haltwise.trace import Step


def _progress_to_limit(value: float, limit: float) -> float:
    """Return progress towards ``limit``: 0 at zero, 1 at the limit, above 1 past it."""
    return max(0.0, value / limit)


@dataclass(frozen=True)
class RuleScorer(Scorer):
    """Risk from three classic warning signs: looping, error streaks and stalling.

    * **Looping:** the same tool call (same arguments) repeated within the recent window.
      The default ``repeat_limit=4`` matches OpenHands' stuck detector.
    * **Error streak:** consecutive failures with no successful tool result in between.
    * **Stalling:** many steps since the agent last tried something new.

    Each sign is measured as progress towards its limit (1.0 = at the limit). The
    strongest sign ``s`` becomes the risk ``s / (1 + s)``, which is 0.5 exactly at a
    limit and keeps rising past it without ever saturating. Not saturating lets
    calibration tell "slightly past the limit" from "far past it".

    Works with or without fitting. For a cold start, ``threshold=0.5`` means "stop once
    any sign goes past its limit". Better still, calibrate on successful runs.
    """

    repeat_limit: int = 4
    error_streak_limit: int = 5
    stall_limit: int = 30
    window: int = 10

    def __post_init__(self) -> None:
        if self.repeat_limit < 2:
            raise ValueError("repeat_limit must be at least 2 (one call is not a repeat)")
        if min(self.error_streak_limit, self.stall_limit, self.window) < 1:
            raise ValueError("error_streak_limit, stall_limit and window must be positive")

    def risk(self, features: TraceFeatures) -> float:
        """Combine the warning signs in ``features`` into one risk."""
        strongest = max(
            _progress_to_limit(features.repeat_in_window - 1, self.repeat_limit - 1),
            _progress_to_limit(features.error_streak, self.error_streak_limit),
            _progress_to_limit(features.steps_since_new_call, self.stall_limit),
        )
        return strongest / (1.0 + strongest)

    def new_tracker(self) -> _RuleTracker:
        """Start following a new run."""
        return _RuleTracker(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON."""
        return {"type": type(self).__name__, **asdict(self)}

    @classmethod
    def _from_params(cls, params: Mapping[str, Any]) -> RuleScorer:
        return cls(**params)


class _RuleTracker:
    def __init__(self, scorer: RuleScorer) -> None:
        self._scorer = scorer
        self._features = TraceFeatures(window=scorer.window)

    def update(self, step: Step) -> float:
        self._features.update(step)
        return self._scorer.risk(self._features)
