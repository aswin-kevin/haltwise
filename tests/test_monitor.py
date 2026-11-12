import json
import math
from pathlib import Path

import pytest

from haltwise import Action, Monitor, Run, Step, split_runs
from haltwise.errors import CalibrationError, NotCalibratedError
from haltwise.policies import FirstCrossing
from haltwise.scorers import RuleScorer
from haltwise.synthetic import make_runs


def looping_run() -> list[Step]:
    call = Step.tool_call("run_tests", {"path": "tests/"})
    return [call, Step.error("1 failed")] * 10


def healthy_run() -> list[Step]:
    steps: list[Step] = []
    for i in range(6):
        steps += [Step.tool_call("read_file", {"path": f"f{i}.py"}), Step.tool_result("read_file")]
    return steps


@pytest.fixture(scope="module")
def data() -> tuple[list[Run], list[Run]]:
    return split_runs(make_runs(1500, seed=0, tasks=1500), test_share=0.3, seed=0)


@pytest.fixture(scope="module")
def monitor(data: tuple[list[Run], list[Run]]) -> Monitor:
    train, _ = data
    return Monitor(false_kill_rate=0.05).fit(train)


class TestSplitRuns:
    def test_keeps_each_task_on_one_side(self) -> None:
        train, test = split_runs(make_runs(200, seed=0, tasks=20), test_share=0.25, seed=0)

        assert {r.task for r in train}.isdisjoint({r.task for r in test})
        assert len(train) + len(test) == 200

    def test_is_reproducible(self) -> None:
        runs = make_runs(100, seed=0)

        assert split_runs(runs, seed=3) == split_runs(runs, seed=3)


class TestFitAndEvaluate:
    def test_fit_sets_a_threshold(self, monitor: Monitor) -> None:
        assert monitor.threshold is not None
        assert monitor.calibration_size > 0

    def test_held_out_false_kill_rate_stays_near_target(
        self, monitor: Monitor, data: tuple[list[Run], list[Run]]
    ) -> None:
        _, test = data

        report = monitor.evaluate(test)

        assert report.false_kill_rate <= 0.05 + 0.04
        assert report.failures_stopped > 0.3
        assert report.steps_saved > 0

    def test_report_prints_as_a_readable_summary(
        self, monitor: Monitor, data: tuple[list[Run], list[Run]]
    ) -> None:
        text = str(monitor.evaluate(data[1]))

        assert "false-kill rate per task" in text
        assert "target 5%" in text


class TestWatch:
    def test_stops_a_looping_run(self, monitor: Monitor) -> None:
        watch = monitor.watch()

        decisions = [watch.observe(step) for step in looping_run()]

        assert decisions[-1].should_stop
        assert watch.stopped

    def test_lets_a_healthy_run_continue(self, monitor: Monitor) -> None:
        watch = monitor.watch()

        decisions = [watch.observe(step) for step in healthy_run()]

        assert not any(d.should_stop for d in decisions)

    def test_stays_stopped_once_stopped(self, monitor: Monitor) -> None:
        watch = monitor.watch()
        for step in looping_run():
            watch.observe(step)

        assert watch.observe(Step.tool_result("x")).action is Action.STOP

    def test_stop_decision_explains_itself_and_records_where_it_stopped(
        self, monitor: Monitor
    ) -> None:
        watch = monitor.watch()
        decisions = [watch.observe(step) for step in looping_run()]
        first_stop = next(i for i, d in enumerate(decisions) if d.should_stop)

        assert "threshold" in decisions[first_stop].reason
        assert decisions[first_stop].step == first_stop
        assert all(d == decisions[first_stop] for d in decisions[first_stop:])

    def test_check_matches_a_live_watch(self, monitor: Monitor) -> None:
        steps = looping_run()[:7]
        watch = monitor.watch()
        live = [watch.observe(step) for step in steps][-1]

        assert monitor.check(steps) == live


class TestColdStart:
    def test_rule_scorer_with_a_manual_threshold_needs_no_data(self) -> None:
        monitor = Monitor(RuleScorer(), policy=FirstCrossing(), threshold=0.5)
        watch = monitor.watch()

        assert [watch.observe(s) for s in looping_run()][-1].should_stop

    def test_unfitted_monitor_says_how_to_fix_it(self) -> None:
        with pytest.raises(NotCalibratedError, match="fit"):
            Monitor().watch()

    def test_calibrating_on_few_tasks_warns_about_noise(self) -> None:
        successes = [r for r in make_runs(200, seed=1, tasks=200) if r.succeeded][:30]

        with pytest.warns(UserWarning, match="only 30 tasks"):
            Monitor(RuleScorer()).calibrate(successes)


class TestTaskAwareCalibration:
    def test_many_successes_on_one_task_count_once(self) -> None:
        same_task = [Run([Step.llm()], succeeded=True, task="only-task")] * 200

        with pytest.raises(CalibrationError, match="at least 19 tasks"):
            Monitor(RuleScorer()).calibrate(same_task)

    def test_calibration_size_counts_tasks_not_runs(self) -> None:
        runs = [Run([Step.llm()], succeeded=True, task=f"t{i % 25}", id=str(i)) for i in range(100)]

        with pytest.warns(UserWarning, match="only 25 tasks"):
            monitor = Monitor(RuleScorer()).calibrate(runs)

        assert monitor.calibration_size == 25


class TestPersistence:
    def test_saved_monitor_makes_identical_decisions(
        self, monitor: Monitor, tmp_path: Path
    ) -> None:
        path = tmp_path / "monitor.json"
        monitor.save(path)

        restored = Monitor.load(path)

        assert restored.check(looping_run()) == monitor.check(looping_run())
        assert restored.false_kill_rate == monitor.false_kill_rate


def test_invalid_false_kill_rate_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="false_kill_rate"):
        Monitor(false_kill_rate=5)


class TestSplitBalance:
    def test_one_huge_task_does_not_swamp_the_held_out_side(self) -> None:
        huge = [Run([Step.llm()], succeeded=True, task="huge", id=str(i)) for i in range(300)]
        singles = [Run([Step.llm()], succeeded=True, task=f"t{i}") for i in range(100)]

        rest, held_out = split_runs(huge + singles, test_share=0.3, seed=0)

        assert 80 <= len(held_out) <= 160
        assert rest

    def test_a_single_task_cannot_be_split(self) -> None:
        runs = [Run([Step.llm()], succeeded=True, task="only")] * 10

        with pytest.raises(ValueError, match="at least two tasks"):
            split_runs(runs)


def test_infinite_threshold_is_saved_as_standard_json(tmp_path: Path) -> None:
    monitor = Monitor(RuleScorer(), threshold=-math.inf)
    path = tmp_path / "m.json"
    monitor.save(path)

    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-standard {c}"))
    assert Monitor.load(path).threshold == -math.inf


class TestConfidenceMode:
    def test_monitor_calibrates_with_confidence_and_saves_it(self, tmp_path: Path) -> None:
        runs = make_runs(1200, seed=2, tasks=1200)
        plain = Monitor(RuleScorer()).fit(runs)
        strict = Monitor(RuleScorer(), confidence=0.95).fit(runs)
        path = tmp_path / "strict.json"
        strict.save(path)

        assert strict.threshold is not None
        assert plain.threshold is not None
        assert strict.threshold >= plain.threshold
        assert Monitor.load(path).confidence == 0.95

    def test_invalid_confidence_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Monitor(confidence=95)
