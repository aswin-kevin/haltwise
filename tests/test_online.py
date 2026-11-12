import random
from pathlib import Path

import pytest

from haltwise import Action, AdaptiveMonitor, Monitor, RuleScorer, Run, Step
from haltwise.errors import HaltwiseError, NotCalibratedError
from haltwise.policies import FirstCrossing
from haltwise.scorers.base import Scorer
from haltwise.trace import Step as StepType


class CostIsRisk(Scorer):
    """Test scorer: the risk after a step is simply that step's ``cost``."""

    def new_tracker(self) -> "_CostTracker":
        return _CostTracker()

    def to_dict(self) -> dict[str, str]:
        return {"type": "CostIsRisk"}

    @classmethod
    def _from_params(cls, params: object) -> "CostIsRisk":
        return cls()


class _CostTracker:
    def update(self, step: StepType) -> float:
        return float(step.cost or 0.0)


def single_step_run(risk: float, succeeded: bool) -> Run:
    return Run([Step.llm(cost=risk)], succeeded=succeeded)


def adaptive(threshold: float = 0.5, **options: float) -> AdaptiveMonitor:
    monitor = Monitor(CostIsRisk(), policy=FirstCrossing(), threshold=threshold)
    return AdaptiveMonitor(monitor, seed=0, **options)


def observe_all(watch_owner: AdaptiveMonitor, run: Run) -> None:
    watch = watch_owner.watch()
    for step in run.steps:
        watch.observe(step)
    watch_owner.record(watch, run.succeeded)


class TestSetup:
    def test_needs_a_calibrated_monitor(self) -> None:
        with pytest.raises(NotCalibratedError):
            AdaptiveMonitor(Monitor(CostIsRisk()))

    @pytest.mark.parametrize("share", [0.0, 1.5])
    def test_shadow_share_must_be_a_positive_fraction(self, share: float) -> None:
        with pytest.raises(ValueError, match="shadow_share"):
            adaptive(shadow_share=share)

    def test_starts_from_the_monitor_threshold(self) -> None:
        assert adaptive(threshold=0.7).threshold == pytest.approx(0.7)


class TestShadowRuns:
    def test_flagged_shadow_run_keeps_going_and_says_why(self) -> None:
        owner = adaptive(shadow_share=1.0)  # every flagged run becomes a shadow run
        watch = owner.watch()

        decision = watch.observe(Step.llm(cost=0.9))

        assert decision.action is Action.SHADOW
        assert not decision.should_stop
        assert watch.flagged
        assert not watch.stopped
        assert "calibrat" in decision.reason

    def test_shadow_run_stays_in_shadow_after_flagging(self) -> None:
        owner = adaptive(shadow_share=1.0)
        watch = owner.watch()
        watch.observe(Step.llm(cost=0.9))

        assert watch.observe(Step.llm(cost=0.1)).action is Action.SHADOW


class TestThresholdUpdates:
    def test_spared_success_lowers_the_threshold_slightly(self) -> None:
        owner = adaptive()
        before = owner.threshold

        observe_all(owner, single_step_run(0.1, succeeded=True))

        assert owner.threshold < before

    def test_killed_success_seen_in_a_shadow_run_raises_it_more(self) -> None:
        owner = adaptive(shadow_share=1.0)
        before = owner.threshold
        observe_all(owner, single_step_run(0.1, succeeded=True))
        drop = before - owner.threshold
        after_drop = owner.threshold

        observe_all(owner, single_step_run(0.9, succeeded=True))

        assert owner.threshold - after_drop > drop

    @pytest.mark.parametrize(
        "run",
        [
            single_step_run(0.1, succeeded=False),  # failures carry no false-kill signal
            Run([Step.llm(cost=0.1)], succeeded=None),  # outcome unknown
        ],
    )
    def test_runs_without_false_kill_information_change_nothing(self, run: Run) -> None:
        owner = adaptive()
        before = owner.threshold

        observe_all(owner, run)

        assert owner.threshold == before

    def test_stopped_run_is_censored_and_changes_nothing(self) -> None:
        owner = adaptive(shadow_share=1e-9)  # practically never shadow
        before = owner.threshold

        observe_all(owner, single_step_run(0.9, succeeded=True))

        assert owner.threshold == before
        assert owner.censored_runs == 1

    def test_recording_the_same_watch_twice_is_an_error(self) -> None:
        owner = adaptive()
        watch = owner.watch()
        watch.observe(Step.llm(cost=0.1))
        owner.record(watch, succeeded=True)

        with pytest.raises(HaltwiseError, match="already recorded"):
            owner.record(watch, succeeded=True)


def drifting_stream(n: int, seed: int) -> list[Run]:
    """Calibrated for risks ~U(0, 1), but successful runs drift to look riskier over time."""
    rng = random.Random(seed)
    runs = []
    for i in range(n):
        drift = 0.3 * i / n
        succeeded = rng.random() < 0.5
        risk = min(1.0, rng.random() * 0.7 + drift) if succeeded else rng.uniform(0.6, 1.0)
        runs.append(single_step_run(risk, succeeded))
    return runs


def test_keeps_the_false_kill_rate_near_target_under_drift() -> None:
    stream = drifting_stream(6000, seed=1)
    static = Monitor(CostIsRisk(), policy=FirstCrossing(), threshold=0.665)
    owner = AdaptiveMonitor(static, seed=0, false_kill_rate=0.05)

    static_report = owner.dry_run(stream, adapt=False)
    adaptive_report = owner.dry_run(stream)

    assert static_report.false_kill_rate > 0.10
    assert adaptive_report.false_kill_rate == pytest.approx(0.05, abs=0.02)


def test_dry_run_leaves_the_monitor_untouched() -> None:
    owner = adaptive()
    before = owner.to_dict()

    owner.dry_run(drifting_stream(200, seed=0))

    assert owner.to_dict() == before


def test_dry_run_report_reads_well() -> None:
    text = str(adaptive().dry_run(drifting_stream(300, seed=0)))

    assert "false-kill rate" in text
    assert "shadow runs" in text


def test_round_trips_through_a_file(tmp_path: Path) -> None:
    owner = AdaptiveMonitor(Monitor(RuleScorer(), policy=FirstCrossing(), threshold=0.5), seed=0)
    observe_all(owner, Run([Step.llm()], succeeded=True))
    path = tmp_path / "adaptive.json"

    owner.save(path)
    restored = AdaptiveMonitor.load(path)

    assert restored.threshold == pytest.approx(owner.threshold)
    assert restored.shadow_share == owner.shadow_share


def test_fixed_replay_behaves_like_a_plain_monitor_with_no_shadow_runs() -> None:
    owner = adaptive(shadow_share=1.0)
    risky = [single_step_run(0.9, succeeded=True)] * 10

    report = owner.dry_run(risky, adapt=False)

    assert report.shadow_runs == 0
    assert report.false_kill_rate == 1.0


class TestRobustness:
    def test_rejects_a_threshold_outside_zero_one(self) -> None:
        monitor = Monitor(CostIsRisk(), policy=FirstCrossing(), threshold=2.5)

        with pytest.raises(ValueError, match=r"between 0 and 1"):
            AdaptiveMonitor(monitor)

    def test_explains_a_threshold_of_exactly_zero(self) -> None:
        monitor = Monitor(CostIsRisk(), policy=FirstCrossing(), threshold=0.0)

        with pytest.raises(ValueError, match="exactly 0, which can't be adapted"):
            AdaptiveMonitor(monitor)

    @pytest.mark.parametrize("rate", [0.0, 5.0])
    def test_rejects_an_invalid_false_kill_rate(self, rate: float) -> None:
        with pytest.raises(ValueError, match="false_kill_rate"):
            adaptive(false_kill_rate=rate)

    def test_rejects_watches_it_did_not_create(self) -> None:
        owner = adaptive()

        with pytest.raises(HaltwiseError, match=r"watch\(\)"):
            owner.record(owner.monitor.watch(), succeeded=True)

    def test_dry_run_works_with_scorers_that_cannot_be_saved(self) -> None:
        class Unsaveable(CostIsRisk):
            def to_dict(self) -> dict[str, str]:
                raise HaltwiseError("cannot save")

        monitor = Monitor(Unsaveable(), policy=FirstCrossing(), threshold=0.5)

        report = AdaptiveMonitor(monitor, seed=0).dry_run(drifting_stream(50, seed=0))

        assert report.runs == 50
