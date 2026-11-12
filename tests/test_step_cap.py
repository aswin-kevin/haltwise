import random

import pytest

from haltwise import FirstCrossing, Monitor, Run, Step, StepCountScorer, calibrate_step_cap
from haltwise.errors import CalibrationError

# Most tests use deliberately tiny calibration sets; the warning itself is tested below.
pytestmark = pytest.mark.filterwarnings("ignore:Step cap based on only")


def run_of_length(length: int, succeeded: bool = True, task: str | None = None) -> Run:
    return Run([Step.llm()] * length, succeeded=succeeded, task=task)


class TestCalibrateStepCap:
    def test_is_the_conformal_order_statistic_of_successful_run_lengths(self) -> None:
        runs = [run_of_length(n, task=f"t{n}") for n in range(1, 20)]  # 19 tasks: 1..19 steps

        # n=19, alpha=0.1 -> rank ceil(20 * 0.9) = 18 -> 18th shortest = 18 steps
        assert calibrate_step_cap(runs, false_kill_rate=0.1) == 18

    def test_failed_runs_do_not_count(self) -> None:
        successes = [run_of_length(5, task=f"t{i}") for i in range(30)]
        failures = [run_of_length(500, succeeded=False, task=f"f{i}") for i in range(30)]

        assert calibrate_step_cap(successes + failures, false_kill_rate=0.1) == 5

    def test_counts_one_run_per_task(self) -> None:
        same_task = [run_of_length(5, task="only")] * 200

        with pytest.raises(CalibrationError, match="tasks"):
            calibrate_step_cap(same_task, false_kill_rate=0.05)

    def test_new_successful_runs_exceed_the_cap_at_most_alpha_of_the_time(self) -> None:
        rng = random.Random(3)
        trials, exceeded, alpha = 2000, 0, 0.1
        for _ in range(trials):
            calibration = [run_of_length(rng.randint(3, 60), task=f"t{i}") for i in range(30)]
            cap = calibrate_step_cap(calibration, false_kill_rate=alpha, seed=rng.randrange(999))
            exceeded += rng.randint(3, 60) > cap

        assert exceeded / trials <= alpha + 0.02


class TestStepCountScorer:
    def test_risk_grows_with_every_step(self) -> None:
        risks = StepCountScorer().risk_path([Step.llm()] * 5)

        assert risks == sorted(risks)
        assert len(set(risks)) == 5

    def test_monitor_stops_right_after_the_calibrated_cap(self) -> None:
        runs = [run_of_length(n, task=f"t{n}") for n in range(1, 20)]
        cap = calibrate_step_cap(runs, false_kill_rate=0.1)
        monitor = Monitor(StepCountScorer(), policy=FirstCrossing(), false_kill_rate=0.1)
        with pytest.warns(UserWarning, match="only 19 tasks"):
            monitor.calibrate(runs)

        watch = monitor.watch()
        decisions = [watch.observe(Step.llm()) for _ in range(cap + 5)]
        first_stop = next(i for i, d in enumerate(decisions) if d.should_stop)

        # The stop comes after step cap+1 is observed (0-based index `cap`): a run of
        # exactly `cap` steps is never stopped, matching `max_steps = cap`.
        assert first_stop == cap

    def test_round_trips_through_dict(self) -> None:
        scorer = StepCountScorer(scale=20)

        assert StepCountScorer.from_dict(scorer.to_dict()) == scorer


def test_warns_when_few_tasks_back_the_cap() -> None:
    runs = [run_of_length(n, task=f"t{n}") for n in range(1, 31)]

    with pytest.warns(UserWarning, match="only 30 tasks"):
        calibrate_step_cap(runs, false_kill_rate=0.1)


def test_step_cap_supports_the_confidence_mode() -> None:
    runs = [run_of_length(n % 40 + 1, task=f"t{n}") for n in range(400)]

    assert calibrate_step_cap(runs, 0.05, confidence=0.95) >= calibrate_step_cap(runs, 0.05)
