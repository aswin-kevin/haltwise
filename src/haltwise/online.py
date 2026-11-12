"""Keep the false-kill rate on target while your agent changes.

A threshold calibrated offline is only valid for the agent setup it was calibrated on.
Swap the model, edit the prompt or add a tool, and the share of good runs being stopped
drifts. On real SWE-agent data, reusing a 70B-calibrated monitor on an 8B model doubled
it from 5% to about 10%.

`AdaptiveMonitor` corrects the threshold continuously from the outcomes of finished runs:

* a good run that the monitor **spared** nudges the threshold down a little
  (by ``step_size * α``);
* a good run that the monitor **would have stopped** nudges it up a lot
  (by ``step_size * (1 - α)``).

The two forces balance exactly when a share α of good runs gets stopped. This is online
quantile tracking: adaptive conformal inference (Gibbs & Candès, NeurIPS 2021) and
conformal PID control (Angelopoulos, Candès & Tibshirani, NeurIPS 2023). Unlike offline
calibration it needs no exchangeability: over a long stream the rate tracks the target
even while the agent drifts. Over short stretches it fluctuates.

**Censoring.** A stopped run never reveals whether it would have succeeded, so the
"would have stopped a good run" signal is invisible exactly when it matters. Therefore a
random ``shadow_share`` of flagged runs is allowed to keep going (`Action.SHADOW`). Their
outcomes are up-weighted by ``1 / shadow_share`` (inverse propensity weighting), which
keeps the update unbiased. The shadow share is the price of staying calibrated: those
runs cost what they would have cost without haltwise.

Updates happen in log-odds space, so ``step_size`` behaves the same whether the
threshold sits at 0.5 or at 0.98.
"""

from __future__ import annotations

import json
import math
import random
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haltwise.calibration import validate_rate
from haltwise.errors import HaltwiseError, NotCalibratedError
from haltwise.monitor import Monitor, Watch
from haltwise.trace import Run

_MAX_LOG_ODDS = 20.0  # keeps the threshold strictly inside (0, 1)


def _log_odds(probability: float) -> float:
    clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _probability(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


class _ReplayTracker:
    """Feeds precomputed risks to a `Watch`, one per step."""

    def __init__(self, risks: Sequence[float]) -> None:
        self._risks = iter(risks)

    def update(self, step: object) -> float:
        return next(self._risks)


@dataclass(frozen=True)
class StreamReport:
    """What happened when a stream of labelled runs was replayed through a monitor.

    Attributes:
        runs: Runs replayed.
        successes: How many succeeded.
        failures: How many failed.
        false_kill_rate: Share of successful runs that were stopped (from the true labels).
        failures_stopped: Share of failed runs that were stopped.
        steps_saved: Share of failed-run steps skipped thanks to stopping.
        shadow_runs: Flagged runs that kept going as calibration samples.
        start_threshold: Threshold before the first run.
        end_threshold: Threshold after the last run.
        thresholds: Threshold in force for each run, in order.
    """

    runs: int
    successes: int
    failures: int
    false_kill_rate: float
    failures_stopped: float
    steps_saved: float
    shadow_runs: int
    start_threshold: float
    end_threshold: float
    thresholds: list[float] = field(repr=False)

    def __str__(self) -> str:
        rows = [
            ("runs replayed", f"{self.runs} ({self.successes} succeeded, {self.failures} failed)"),
            ("false-kill rate", f"{self.false_kill_rate:.1%}"),
            ("failed runs stopped", f"{self.failures_stopped:.1%}"),
            ("failed-run steps saved", f"{self.steps_saved:.1%}"),
            ("shadow runs", str(self.shadow_runs)),
            ("threshold", f"{self.start_threshold:.3f} -> {self.end_threshold:.3f}"),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


class AdaptiveMonitor:
    """A `Monitor` whose threshold keeps adjusting to hold the false-kill rate on target.

    Use it like a monitor, and tell it how each run ended::

        adaptive = AdaptiveMonitor(Monitor().fit(past_runs))

        watch = adaptive.watch()
        for step in agent_steps():
            if watch.observe(step).should_stop:
                break
        adaptive.record(watch, succeeded=run_passed)   # None if unknown

    Args:
        monitor: A calibrated monitor (its threshold is the starting point).
        false_kill_rate: Target share of good runs stopped. Defaults to the monitor's.
        step_size: How fast the threshold moves, in log-odds per run. Larger values
            react faster to change but wobble more.
        shadow_share: Share of flagged runs that keep going so their outcome can be
            observed. Must be above 0: without it, stopped good runs go unnoticed.
        seed: Seed for picking shadow runs (``None`` for fresh randomness).
    """

    def __init__(
        self,
        monitor: Monitor,
        *,
        false_kill_rate: float | None = None,
        step_size: float = 0.3,
        shadow_share: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if monitor.threshold is None:
            raise NotCalibratedError(
                "AdaptiveMonitor needs a starting threshold: pass a monitor that was fitted "
                "with monitor.fit(runs) or created with threshold=..."
            )
        if monitor.threshold in (0.0, 1.0):
            raise ValueError(
                f"The monitor's threshold is exactly {monitor.threshold:g}, which can't be "
                "adapted (it sits at the edge of the risk scale; adaptation works in "
                "log-odds). This usually means calibration found no spread in the scores: "
                "check the scorer, or start from a hand-picked threshold strictly between 0 and 1."
            )
        if not 0.0 < monitor.threshold < 1.0:
            raise ValueError(
                "AdaptiveMonitor needs a scorer whose risks are probabilities: the starting "
                f"threshold must be strictly between 0 and 1 (got {monitor.threshold})"
            )
        if not 0.0 < shadow_share <= 1.0:
            raise ValueError(f"shadow_share must be in (0, 1] (got {shadow_share})")
        if step_size <= 0.0:
            raise ValueError(f"step_size must be positive (got {step_size})")

        self.monitor = monitor
        self.false_kill_rate = (
            false_kill_rate if false_kill_rate is not None else monitor.false_kill_rate
        )
        validate_rate(self.false_kill_rate)
        self.step_size = step_size
        self.shadow_share = shadow_share
        self.recorded_runs = 0
        self.censored_runs = 0
        self.shadow_runs = 0
        self._log_odds = _log_odds(monitor.threshold)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._issued: weakref.WeakSet[Watch] = weakref.WeakSet()
        self._recorded: weakref.WeakSet[Watch] = weakref.WeakSet()

    @property
    def threshold(self) -> float:
        """The stopping threshold in force for the next run."""
        return _probability(self._log_odds)

    def watch(self) -> Watch:
        """Start guarding one new run, using the current threshold."""
        with self._lock:
            shadow = self._rng.random() < self.shadow_share
            threshold = self.threshold
        watch = Watch(
            self.monitor.scorer.new_tracker(), self.monitor.policy, threshold, shadow=shadow
        )
        with self._lock:
            self._issued.add(watch)
        return watch

    def record(self, watch: Watch, succeeded: bool | None) -> None:
        """Report how a watched run ended, and adjust the threshold.

        Args:
            watch: The `Watch` that guarded the run.
            succeeded: The run's outcome. Pass ``None`` if unknown, including for runs
                that were stopped (their outcome is ignored either way).
        """
        with self._lock:
            if watch not in self._issued:
                raise HaltwiseError(
                    "This watch was not created by this AdaptiveMonitor; use adaptive.watch() "
                    "so the run gets the current threshold and a fair chance to shadow."
                )
            if watch in self._recorded:
                raise HaltwiseError("This run was already recorded; record each watch once.")
            self._recorded.add(watch)
            self.recorded_runs += 1
            self.shadow_runs += watch.flagged and watch.shadow

            if watch.stopped:
                self.censored_runs += 1  # outcome unobservable: no update
                return
            if not succeeded:
                return  # failures and unknown outcomes say nothing about false kills
            if watch.flagged:  # a good run we would have stopped, seen thanks to shadowing
                self._move(self.step_size * (1.0 - self.false_kill_rate) / self.shadow_share)
            else:  # a good run we spared
                self._move(-self.step_size * self.false_kill_rate)

    def _move(self, delta: float) -> None:
        self._log_odds = min(max(self._log_odds + delta, -_MAX_LOG_ODDS), _MAX_LOG_ODDS)

    # ---- what-if replay -------------------------------------------------------------

    def dry_run(self, runs: Sequence[Run], *, adapt: bool = True) -> StreamReport:
        """Replay labelled runs in order and report what would have happened.

        Outcomes are fed back exactly as they would be in production: stopped runs
        reveal nothing. The monitor itself is left untouched.

        Args:
            runs: Labelled runs, oldest first.
            adapt: ``False`` keeps the threshold fixed, for a before/after comparison.
        """
        replica = self._replica()
        start = replica.threshold
        thresholds: list[float] = []
        killed_successes = stopped_failures = failed_steps = skipped_steps = 0

        for run in runs:
            thresholds.append(replica.threshold)
            # Score the whole run in one batch, then replay the risks through a real
            # Watch: identical decisions, far fewer model calls than step-by-step scoring.
            replay = _ReplayTracker(self.monitor.scorer.risk_path(run.steps))
            if adapt:
                watch = replica.watch()
                watch._replace_tracker(replay)
            else:  # behave exactly like a plain fixed-threshold monitor: no shadow runs
                watch = Watch(replay, self.monitor.policy, replica.threshold)
            stop_index = next(
                (i for i, step in enumerate(run.steps) if watch.observe(step).should_stop), None
            )
            if run.succeeded:
                killed_successes += stop_index is not None
            elif run.failed:
                failed_steps += len(run.steps)
                if stop_index is not None:
                    stopped_failures += 1
                    skipped_steps += len(run.steps) - (stop_index + 1)
            if adapt:
                replica.record(watch, run.succeeded)

        successes = sum(bool(run.succeeded) for run in runs)
        failures = sum(run.failed for run in runs)
        return StreamReport(
            runs=len(runs),
            successes=successes,
            failures=failures,
            false_kill_rate=killed_successes / successes if successes else 0.0,
            failures_stopped=stopped_failures / failures if failures else 0.0,
            steps_saved=skipped_steps / failed_steps if failed_steps else 0.0,
            shadow_runs=replica.shadow_runs,
            start_threshold=start,
            end_threshold=replica.threshold,
            thresholds=thresholds,
        )

    def _replica(self) -> AdaptiveMonitor:
        """Copy the adaptive state, sharing the (read-only) monitor. No serialisation."""
        replica = AdaptiveMonitor(
            self.monitor,
            false_kill_rate=self.false_kill_rate,
            step_size=self.step_size,
            shadow_share=self.shadow_share,
        )
        replica._log_odds = self._log_odds
        replica.recorded_runs = self.recorded_runs
        replica.censored_runs = self.censored_runs
        replica.shadow_runs = self.shadow_runs
        replica._rng.setstate(self._rng.getstate())
        return replica

    # ---- persistence ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON (the random state for shadow picks is not saved)."""
        return {
            "monitor": self.monitor.to_dict(),
            "false_kill_rate": self.false_kill_rate,
            "step_size": self.step_size,
            "shadow_share": self.shadow_share,
            "threshold": self.threshold,
            "recorded_runs": self.recorded_runs,
            "censored_runs": self.censored_runs,
            "shadow_runs": self.shadow_runs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, monitor: Monitor | None = None) -> AdaptiveMonitor:
        """Rebuild from `to_dict` output. Pass ``monitor`` to reuse an existing one."""
        adaptive = cls(
            monitor if monitor is not None else Monitor.from_dict(data["monitor"]),
            false_kill_rate=data["false_kill_rate"],
            step_size=data["step_size"],
            shadow_share=data["shadow_share"],
        )
        adaptive._log_odds = _log_odds(data["threshold"])
        adaptive.recorded_runs = data["recorded_runs"]
        adaptive.censored_runs = data["censored_runs"]
        adaptive.shadow_runs = data["shadow_runs"]
        return adaptive

    def save(self, path: str | Path) -> None:
        """Save to a JSON file. Do this regularly so the learned threshold survives restarts."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AdaptiveMonitor:
        """Load from a file written by `save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
