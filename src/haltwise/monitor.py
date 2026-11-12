"""`Monitor` ties everything together: learn from past runs, then guard live ones.

Typical use::

    monitor = Monitor(false_kill_rate=0.05).fit(past_runs)

    watch = monitor.watch()
    for step in my_agent_steps():
        if watch.observe(step).should_stop:
            break
"""

from __future__ import annotations

import json
import math
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from haltwise.calibration import (
    conformal_threshold,
    one_per_group,
    validate_confidence,
    validate_rate,
)
from haltwise.errors import NotCalibratedError
from haltwise.evaluation import EvaluationReport, evaluate
from haltwise.policies import Density, Policy
from haltwise.scorers import FeatureScorer, RiskTracker, Scorer
from haltwise.splits import split_runs, task_key
from haltwise.trace import Run, Step

_FORMAT_VERSION = 1

# Below this many calibration tasks the realised false-kill rate can swing a lot between
# calibration sets (its spread shrinks roughly like 1/sqrt(n)).
_COMFORTABLE_CALIBRATION_SIZE = 100


def _encode_float(value: float | None) -> float | str | None:
    """Store infinities as strings: ``json`` would otherwise write non-standard JSON."""
    if value is not None and not math.isfinite(value):
        return str(value)  # "inf" / "-inf"
    return value


def _decode_float(value: float | str | None) -> float | None:
    return None if value is None else float(value)


class Action(str, Enum):
    """What the monitor recommends after a step."""

    CONTINUE = "continue"
    WARN = "warn"  # risk is above the threshold, but the policy wants more evidence
    STOP = "stop"
    SHADOW = "shadow"  # would stop, but this run keeps going as a calibration sample


@dataclass(frozen=True)
class Decision:
    """The monitor's verdict after one step.

    Attributes:
        action: What to do.
        risk: Current estimated risk that the run fails (0–1).
        threshold: Risk level above which steps count against the run.
        step: Index of the step this decision is about.
        reason: One human-readable sentence explaining the decision.
    """

    action: Action
    risk: float
    threshold: float
    step: int
    reason: str

    @property
    def should_stop(self) -> bool:
        """True when the run should be stopped now."""
        return self.action is Action.STOP


class Monitor:
    """Learns what doomed runs look like and decides when to stop live ones.

    Args:
        scorer: Turns a run-so-far into a risk. Defaults to `FeatureScorer`.
        policy: Decides when risks justify stopping. Defaults to ``Density(2, 3)``:
            stop when 2 of the last 3 steps look too risky.
        false_kill_rate: Target share of *successful* runs you accept stopping by
            mistake. `fit` / `calibrate` pick the threshold that meets it.
        threshold: Set the threshold by hand instead of calibrating (useful for a
            cold start with `RuleScorer`; no guarantee then).
        confidence: Optional stricter guarantee: with this probability (e.g. 0.95) the
            share of good runs stopped is at most ``false_kill_rate`` for *your*
            threshold, not just on average. Needs more calibration tasks (59 for 5% at
            95%) and gives a somewhat higher threshold.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        policy: Policy | None = None,
        *,
        false_kill_rate: float = 0.05,
        threshold: float | None = None,
        confidence: float | None = None,
    ) -> None:
        validate_rate(false_kill_rate)
        validate_confidence(confidence)
        self.confidence = confidence
        self.scorer = scorer if scorer is not None else FeatureScorer()
        self.policy = policy if policy is not None else Density(hits=2, window=3)
        self.false_kill_rate = false_kill_rate
        self.threshold = threshold
        self.calibration_size = 0
        """Independent successful runs behind the threshold (one per task)."""

    def __repr__(self) -> str:
        return (
            f"Monitor(scorer={type(self.scorer).__name__}, policy={self.policy!r}, "
            f"false_kill_rate={self.false_kill_rate}, threshold={self.threshold})"
        )

    # ---- learning -------------------------------------------------------------------

    def fit(self, runs: Sequence[Run], *, calibration_share: float = 0.3, seed: int = 0) -> Monitor:
        """Train the scorer and calibrate the threshold from past labelled runs.

        Runs are split by task: one part trains the scorer, the held-out part sets the
        threshold. Calibrating on runs the scorer never saw is what makes the
        false-kill guarantee honest. Scorers that don't learn (like `RuleScorer`)
        calibrate on all runs.
        """
        if self.scorer.learns_from_data:
            training, calibration = split_runs(runs, test_share=calibration_share, seed=seed)
            self.scorer.fit(training)
        else:
            calibration = list(runs)
        return self.calibrate(calibration, seed=seed)

    def calibrate(self, runs: Sequence[Run], *, seed: int = 0) -> Monitor:
        """Set the threshold from runs the scorer was *not* trained on.

        Only successful runs are used, and only **one per task**. Runs on the same task
        behave alike, so counting each of them would overstate the evidence (see
        `haltwise.calibration.one_per_group`). The guarantee then reads: a successful run
        on a new task is stopped with probability at most ``false_kill_rate``.

        Use runs from the same agent setup you will monitor, and recalibrate after
        changing the model, prompt or tools.
        """
        successes = [(task_key(run, i), run) for i, run in enumerate(runs) if run.succeeded]
        chosen = one_per_group(
            [run for _, run in successes], [task for task, _ in successes], seed=seed
        )
        independent = [
            self.policy.path_statistic(self.scorer.risk_path(run.steps)) for run in chosen
        ]
        threshold = conformal_threshold(
            independent,
            self.false_kill_rate,
            confidence=self.confidence,
            unit="tasks with a successful run",
        )
        if len(independent) < _COMFORTABLE_CALIBRATION_SIZE:
            warnings.warn(
                f"Calibrating on only {len(independent)} tasks with a successful run. The "
                f"guarantee holds on average, but the realised false-kill rate may vary "
                f"noticeably; {_COMFORTABLE_CALIBRATION_SIZE}+ tasks give a steadier threshold.",
                UserWarning,
                stacklevel=2,
            )
        self.threshold = threshold
        self.calibration_size = len(independent)
        return self

    def evaluate(self, runs: Sequence[Run]) -> EvaluationReport:
        """Measure the monitor on held-out labelled runs. Print the result for a summary."""
        return evaluate(
            runs,
            scorer=self.scorer,
            policy=self.policy,
            threshold=self._require_threshold(),
            target_false_kill_rate=self.false_kill_rate if self.calibration_size else None,
        )

    # ---- guarding live runs ---------------------------------------------------------

    def watch(self) -> Watch:
        """Start guarding one new run. Feed it steps with `Watch.observe`."""
        threshold = self._require_threshold()
        return Watch(self.scorer.new_tracker(), self.policy, threshold)

    def check(self, steps: Sequence[Step]) -> Decision:
        """Return the decision for a run so far, replaying all of its steps.

        Handy when you hold the full trace (e.g. in a stateless web handler). For a
        live loop, `watch` is cheaper: it does not replay.
        """
        watch = self.watch()
        decision = watch.idle_decision()
        for step in steps:
            decision = watch.observe(step)
        return decision

    def _require_threshold(self) -> float:
        if self.threshold is None:
            raise NotCalibratedError(
                "This monitor has no stopping threshold yet. Call monitor.fit(runs) with past "
                "labelled runs, or pass threshold=... when creating it."
            )
        return self.threshold

    # ---- persistence ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON."""
        return {
            "format_version": _FORMAT_VERSION,
            "scorer": self.scorer.to_dict(),
            "policy": self.policy.to_dict(),
            "false_kill_rate": self.false_kill_rate,
            "confidence": self.confidence,
            "threshold": _encode_float(self.threshold),
            "calibration_size": self.calibration_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Monitor:
        """Rebuild a monitor saved with `to_dict`."""
        monitor = cls(
            Scorer.from_dict(data["scorer"]),
            Policy.from_dict(data["policy"]),
            false_kill_rate=data["false_kill_rate"],
            threshold=_decode_float(data["threshold"]),
            confidence=data.get("confidence"),
        )
        monitor.calibration_size = data.get("calibration_size", 0)
        return monitor

    def save(self, path: str | Path) -> None:
        """Save to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Monitor:
        """Load a monitor saved with `save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class Watch:
    """Guards one live run. Create it with `Monitor.watch`; call `observe` after each step.

    Args:
        tracker: Follows the run and produces a risk after each step.
        policy: Decides when the risks justify stopping.
        threshold: Risk level above which steps count against the run.
        shadow: Keep going even when the policy says stop, reporting `Action.SHADOW`
            instead. `haltwise.AdaptiveMonitor` uses a few such runs to learn whether
            it is stopping good runs.
    """

    def __init__(
        self, tracker: RiskTracker, policy: Policy, threshold: float, *, shadow: bool = False
    ) -> None:
        self._tracker = tracker
        self._policy = policy
        self._recent: deque[float] = deque(maxlen=policy.lookback)
        self._stop: Decision | None = None
        self.threshold = threshold
        self.shadow = shadow
        self.flagged = False
        """True once the policy has said stop, whether or not the run actually stopped."""
        self.risks: list[float] = []

    @property
    def stopped(self) -> bool:
        """True once the monitor has recommended stopping this run."""
        return self._stop is not None

    def _replace_tracker(self, tracker: RiskTracker) -> None:
        """Swap the risk tracker before the first step (internal: fast offline replay)."""
        if self.risks:
            raise RuntimeError("replace_tracker must be called before the first observe()")
        self._tracker = tracker

    def observe(self, step: Step) -> Decision:
        """Account for one step and return what to do next.

        After a stop, every later call returns that same stop decision.
        """
        if self._stop is not None:
            return self._stop

        risk = self._tracker.update(step)
        self.risks.append(risk)
        self._recent.append(risk)

        if self.flagged or self._policy.triggers(list(self._recent), self.threshold):
            return self._on_flagged(risk)
        if risk > self.threshold:
            return self._decision(
                Action.WARN,
                f"risk {risk:.2f} is above the threshold {self.threshold:.2f}; "
                f"waiting for confirmation before stopping",
            )
        return self._decision(
            Action.CONTINUE, f"risk {risk:.2f} is within the threshold {self.threshold:.2f}"
        )

    def idle_decision(self) -> Decision:
        """Return the decision before any step has been observed."""
        return Decision(Action.CONTINUE, 0.0, self.threshold, -1, "no steps observed yet")

    def _on_flagged(self, risk: float) -> Decision:
        self.flagged = True
        if self.shadow:
            return self._decision(
                Action.SHADOW,
                f"risk {risk:.2f}: this run would be stopped, but it was picked to keep going "
                f"so its outcome can keep the monitor calibrated",
            )
        hits = sum(r > self.threshold for r in self._recent)
        self._stop = self._decision(
            Action.STOP,
            f"risk {risk:.2f}: {hits} of the last {len(self._recent)} steps were above "
            f"the stopping threshold {self.threshold:.2f}",
        )
        return self._stop

    def _decision(self, action: Action, reason: str) -> Decision:
        return Decision(action, self.risks[-1], self.threshold, len(self.risks) - 1, reason)
