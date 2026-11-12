"""The scorer interface: turn the steps of a run so far into a risk in [0, 1]."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypeVar

from haltwise.trace import Run, Step

S = TypeVar("S", bound="Scorer")


class RiskTracker(Protocol):
    """Follows one live run and returns the updated risk after each step."""

    def update(self, step: Step) -> float:
        """Account for ``step`` and return the risk that the run will fail."""
        ...


class Scorer(ABC):
    """Base class for risk scorers.

    A scorer only says *how risky* a run looks. The policy and calibrated threshold in
    `haltwise.Monitor` decide whether to stop. Subclasses implement `new_tracker`, and
    `fit` if they learn from past runs.

    The live path (`new_tracker`) and offline path (`risk_path`) share one code path,
    so risks seen during calibration are exactly the risks seen in production.
    """

    learns_from_data: ClassVar[bool] = False
    """Whether `fit` trains on labelled runs (so it needs its own data split)."""

    _registry: ClassVar[dict[str, type[Scorer]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Scorer._registry[cls.__name__] = cls

    def fit(self: S, runs: Sequence[Run]) -> S:
        """Learn from past runs. Scorers that need no training return themselves."""
        return self

    @abstractmethod
    def new_tracker(self) -> RiskTracker:
        """Start following a new run."""

    def risk_path(self, steps: Sequence[Step]) -> list[float]:
        """Return the risk after each step of a (partial) run."""
        tracker = self.new_tracker()
        return [tracker.update(step) for step in steps]

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON, including a ``"type"`` key."""

    @classmethod
    @abstractmethod
    def _from_params(cls, params: Mapping[str, Any]) -> Scorer:
        """Rebuild from the output of `to_dict` (without ``"type"``)."""

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Scorer:
        """Rebuild any built-in scorer saved with `to_dict`."""
        params = dict(data)
        scorer_type = Scorer._registry[params.pop("type")]
        return scorer_type._from_params(params)
