"""Stopping policies decide *when* a sequence of risk scores justifies stopping a run.

A policy looks at the risk score produced after every step and decides whether to stop.
Each policy also exposes a **path statistic**: one number per run with the property

    the policy stops the run at threshold θ  ⇔  path_statistic(risks) > θ

That equivalence is what lets `haltwise.calibration` turn a target false-kill rate into a
threshold with a statistical guarantee, even though the policy peeks after every step.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, ClassVar

from haltwise.errors import HaltwiseError


class Policy(ABC):
    """Base class for stopping policies."""

    _registry: ClassVar[dict[str, type[Policy]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Policy._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def lookback(self) -> int:
        """How many of the most recent risk scores the live decision needs."""

    @abstractmethod
    def triggers(self, recent_risks: Sequence[float], threshold: float) -> bool:
        """Decide from the latest risk scores (oldest first) whether to stop now."""

    @abstractmethod
    def path_statistic(self, risks: Sequence[float]) -> float:
        """Return the largest threshold at which this policy would have stopped the run.

        Returns ``-inf`` when no threshold could ever stop it.
        """

    def stop_index(self, risks: Sequence[float], threshold: float) -> int | None:
        """Return the index of the step at which the policy stops the run, if it does."""
        for index in range(len(risks)):
            recent = risks[max(0, index + 1 - self.lookback) : index + 1]
            if self.triggers(recent, threshold):
                return index
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON (policies must be dataclasses to be saved)."""
        if not is_dataclass(self):
            raise HaltwiseError(
                f"{type(self).__name__} can't be saved: make it a @dataclass so its settings "
                "can be written to JSON, or save the Monitor with pickle instead."
            )
        return {"type": type(self).__name__, **asdict(self)}

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Policy:
        """Rebuild a policy saved with `to_dict`."""
        params = dict(data)
        policy_type = Policy._registry[params.pop("type")]
        return policy_type(**params)


@dataclass(frozen=True)
class FirstCrossing(Policy):
    """Stop as soon as one risk score exceeds the threshold.

    Simple and fast to react, but a single noisy score can stop a good run.
    `Density` is usually the safer choice.
    """

    @property
    def lookback(self) -> int:
        """Only the latest score matters."""
        return 1

    def triggers(self, recent_risks: Sequence[float], threshold: float) -> bool:
        """Stop if the latest score is above the threshold."""
        return bool(recent_risks) and recent_risks[-1] > threshold

    def path_statistic(self, risks: Sequence[float]) -> float:
        """Return the highest risk seen during the run."""
        return max(risks, default=-math.inf)


@dataclass(frozen=True)
class Density(Policy):
    """Stop when at least ``hits`` of the last ``window`` risk scores exceed the threshold.

    Requiring repeated warnings filters out one-off spikes. OnTrack (arXiv 2610.12375)
    reports that a density rule like this is far more practical for agent trajectories
    than stopping at the first warning.
    """

    hits: int = 2
    window: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.hits <= self.window:
            raise ValueError(
                f"Density needs 1 <= hits <= window (got hits={self.hits}, window={self.window})"
            )

    @property
    def lookback(self) -> int:
        """The live decision uses the last ``window`` scores."""
        return self.window

    def triggers(self, recent_risks: Sequence[float], threshold: float) -> bool:
        """Stop if enough of the recent scores are above the threshold."""
        recent = recent_risks[-self.window :]
        return sum(risk > threshold for risk in recent) >= self.hits

    def path_statistic(self, risks: Sequence[float]) -> float:
        """Return the best window's ``hits``-th largest score.

        A window fires at θ exactly when its ``hits``-th largest score exceeds θ, so the
        run stops at θ exactly when the best window's value does.
        """
        best = -math.inf
        for end in range(self.hits, len(risks) + 1):
            window = sorted(risks[max(0, end - self.window) : end], reverse=True)
            best = max(best, window[self.hits - 1])
        return best
