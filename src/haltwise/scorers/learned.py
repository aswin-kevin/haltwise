"""Learned risk: a classifier over behavioural features, trained on your past runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from haltwise.errors import HaltwiseError
from haltwise.features import FEATURE_NAMES, TraceFeatures
from haltwise.scorers.base import Scorer
from haltwise.scorers.logistic import LogisticRegression
from haltwise.sequence import SEQUENCE_FEATURE_NAMES, StateRates, StateTracker, cross_fit
from haltwise.splits import task_key
from haltwise.trace import Run, Step


class ProbabilisticClassifier(Protocol):
    """Anything with scikit-learn style ``fit`` and ``predict_proba`` (class 1 = failure)."""

    def fit(self, x: Any, y: Any, sample_weight: Any = ...) -> Any:  # noqa: D102
        ...

    def predict_proba(self, x: Any) -> Any:  # noqa: D102
        ...


class FeatureScorer(Scorer):
    """Predict failure from behavioural and sequence features of the run so far.

    Two feature families feed one classifier:

    * behavioural features (`haltwise.features`): looping, error streaks, stalling...
    * sequence features (`haltwise.sequence`): how often the run's recent step patterns
      preceded failure in past runs. Turn them off with ``sequence_order=0``.

    Every prefix of every labelled run becomes one training example, labelled with how
    that run eventually ended. Each run gets equal total weight, so long failing runs do
    not drown out short successful ones. This "label every prefix with the final
    outcome" recipe is the common thread of PrefixGuard (arXiv 2605.06455) and StopLoss
    (arXiv 2610.25525); here it uses cheap features instead of a neural encoder.

    Args:
        model: Classifier to use. Defaults to haltwise's built-in logistic regression.
            Any scikit-learn classifier works, e.g.
            ``FeatureScorer(model=HistGradientBoostingClassifier())``.
        window: Size of the "recent" window used by the behavioural features.
        sequence_order: Steps per sequence state (0 disables sequence features).
    """

    learns_from_data = True

    def __init__(
        self,
        model: ProbabilisticClassifier | None = None,
        window: int = 10,
        sequence_order: int = 3,
    ) -> None:
        self.model: ProbabilisticClassifier = model or LogisticRegression()
        self.window = window
        self.sequence_order = sequence_order
        self.states: StateRates | None = None
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the classifier's input columns, in order."""
        return FEATURE_NAMES + (SEQUENCE_FEATURE_NAMES if self.sequence_order else ())

    def fit(self, runs: Sequence[Run]) -> FeatureScorer:
        """Train on runs with known outcomes; runs without an outcome or steps are skipped."""
        labelled = [run for run in runs if run.succeeded is not None and run.steps]
        if len({run.failed for run in labelled}) < 2:
            raise HaltwiseError(
                "FeatureScorer needs both successful and failed runs to learn from. "
                "Without labelled failures, start with RuleScorer instead."
            )
        training_tables: dict[str, StateRates] = {}
        if self.sequence_order:
            # Live scoring uses rates from all training tasks; training rows use rates
            # learned without their own task, so the classifier doesn't over-trust them.
            self.states = StateRates(order=self.sequence_order).fit(labelled)
            if len({task_key(run, i) for i, run in enumerate(labelled)}) >= 2:
                training_tables = cross_fit(labelled, order=self.sequence_order)
            # With a single task there is nothing to hold out: train on the full table.

        rows: list[list[float]] = []
        labels: list[int] = []
        weights: list[float] = []
        for index, run in enumerate(labelled):
            path = self._feature_path(run.steps, training_tables.get(task_key(run, index)))
            rows.extend(path)
            labels.extend([int(run.failed)] * len(path))
            weights.extend([1.0 / len(path)] * len(path))

        self.model.fit(np.array(rows), np.array(labels), sample_weight=np.array(weights))
        self._fitted = True
        return self

    def new_tracker(self) -> _FeatureTracker:
        """Start following a new run."""
        self._require_fitted()
        return _FeatureTracker(self)

    def risk_path(self, steps: Sequence[Step]) -> list[float]:
        """Return the risk after each step, predicting the whole path in one batch."""
        self._require_fitted()
        if not steps:
            return []
        return self.predict(self._feature_path(steps))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Return the failure probability for each feature row."""
        probabilities = self.model.predict_proba(np.asarray(rows, dtype=np.float64))
        return [float(p) for p in np.asarray(probabilities)[:, 1]]

    def _feature_path(
        self, steps: Sequence[Step], states: StateRates | None = None
    ) -> list[list[float]]:
        rows = _RowBuilder(self.window, states if states is not None else self.states)
        return [rows.update(step) for step in steps]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise HaltwiseError("FeatureScorer is not trained yet; call fit(runs) first.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON (built-in model only)."""
        self._require_fitted()
        if not isinstance(self.model, LogisticRegression):
            raise HaltwiseError(
                "Only the built-in LogisticRegression can be saved as JSON. "
                "For custom models, save the Monitor with pickle or joblib instead."
            )
        return {
            "type": type(self).__name__,
            "window": self.window,
            "sequence_order": self.sequence_order,
            "features": list(self.feature_names),
            "model": self.model.to_dict(),
            "states": self.states.to_dict() if self.states is not None else None,
        }

    @classmethod
    def _from_params(cls, params: Mapping[str, Any]) -> FeatureScorer:
        scorer = cls(
            model=LogisticRegression.from_dict(params["model"]),
            window=params["window"],
            sequence_order=params.get("sequence_order", 0),
        )
        if list(params["features"]) != list(scorer.feature_names):
            raise HaltwiseError(
                "This scorer was saved by a haltwise version with different features; "
                "refit it with the installed version."
            )
        if params.get("states") is not None:
            scorer.states = StateRates.from_dict(params["states"])
        scorer._fitted = True
        return scorer


class _RowBuilder:
    """Builds one feature row per step; shared by training, offline scoring and live use."""

    def __init__(self, window: int, states: StateRates | None) -> None:
        self._behaviour = TraceFeatures(window=window)
        self._sequence: StateTracker | None = states.new_tracker() if states is not None else None

    def update(self, step: Step) -> list[float]:
        self._behaviour.update(step)
        row = self._behaviour.vector()
        if self._sequence is not None:
            row += self._sequence.update(step)
        return row


class _FeatureTracker:
    def __init__(self, scorer: FeatureScorer) -> None:
        self._scorer = scorer
        self._rows = _RowBuilder(scorer.window, scorer.states)

    def update(self, step: Step) -> float:
        return self._scorer.predict([self._rows.update(step)])[0]
