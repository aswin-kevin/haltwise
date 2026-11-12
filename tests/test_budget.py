import random

import pytest

from haltwise import FirstCrossing, Monitor, Run, Step, StepCountScorer, simulate_budget
from haltwise.budget import Attempt, Strategy, play_task
from haltwise.synthetic import make_runs

FAIL_LONG = Attempt(length=10, succeeded=False, stop_index=2)  # monitor stops it after 3 steps
SUCCESS = Attempt(length=5, succeeded=True, stop_index=None)


def play(attempts: list[Attempt], budget: int | None, strategy: Strategy) -> tuple[bool, int]:
    return play_task(attempts, budget, strategy, random.Random(0), stop_profile=(0.0, []))


class TestPlayTask:
    def test_continue_finishes_every_attempt_and_retries_after_failure(self) -> None:
        assert play([FAIL_LONG, SUCCESS], budget=20, strategy=Strategy.CONTINUE) == (True, 15)

    def test_budget_cuts_off_the_attempt_that_would_overrun_it(self) -> None:
        # 10 steps fail, then only 2 steps left for a 5-step success: cut off.
        assert play([FAIL_LONG, SUCCESS], budget=12, strategy=Strategy.CONTINUE) == (False, 12)

    def test_restart_stops_doomed_attempts_early_and_retries(self) -> None:
        assert play([FAIL_LONG, SUCCESS], budget=12, strategy=Strategy.RESTART) == (True, 8)

    def test_give_up_never_retries(self) -> None:
        assert play([FAIL_LONG, SUCCESS], budget=None, strategy=Strategy.GIVE_UP) == (False, 3)
        assert play([SUCCESS, FAIL_LONG], budget=None, strategy=Strategy.GIVE_UP) == (True, 5)

    def test_single_attempt_runs_one_attempt_to_the_end(self) -> None:
        assert play([FAIL_LONG, SUCCESS], budget=None, strategy=Strategy.SINGLE) == (False, 10)

    def test_a_wrong_stop_loses_a_run_that_would_have_succeeded(self) -> None:
        wrongly_stopped = Attempt(length=5, succeeded=True, stop_index=1)

        assert play([wrongly_stopped], budget=None, strategy=Strategy.RESTART) == (False, 2)

    def test_unlimited_budget_retries_until_attempts_run_out(self) -> None:
        assert play([FAIL_LONG] * 3, budget=None, strategy=Strategy.CONTINUE) == (False, 30)

    def test_random_restart_stops_as_often_and_as_early_as_the_monitor(self) -> None:
        rng = random.Random(1)
        never_flagged = Attempt(length=100, succeeded=False, stop_index=None)
        stops = 0
        for _ in range(4000):
            _, steps = play_task(
                [never_flagged], None, Strategy.RANDOM_RESTART, rng, stop_profile=(0.25, [0.5])
            )
            stops += steps == 50

        assert stops / 4000 == pytest.approx(0.25, abs=0.03)


@pytest.fixture(scope="module")
def monitor() -> Monitor:
    return Monitor(StepCountScorer(), policy=FirstCrossing(), threshold=0.5)


class TestSimulateBudget:
    def test_reports_every_strategy_at_every_budget(self, monitor: Monitor) -> None:
        runs = make_runs(400, seed=0, tasks=40)

        report = simulate_budget(monitor, runs, budgets=[50, None], seeds=3)

        assert {(row.strategy, row.budget) for row in report.rows} == {
            (strategy, budget) for strategy in Strategy for budget in (50, None)
        }

    def test_single_attempt_solve_rate_matches_first_attempt_success(
        self, monitor: Monitor
    ) -> None:
        runs = [Run([Step.llm()] * 3, succeeded=i % 4 == 0, task=f"t{i % 50}") for i in range(500)]

        report = simulate_budget(monitor, runs, budgets=[None], seeds=50)

        single = report.row(Strategy.SINGLE, None)
        assert single.solve_rate == pytest.approx(0.25, abs=0.04)

    def test_skips_tasks_with_too_few_attempts(self, monitor: Monitor) -> None:
        runs = make_runs(100, seed=0, tasks=100)  # one attempt per task

        with pytest.raises(ValueError, match="attempts"):
            simulate_budget(monitor, runs, budgets=[None], min_attempts=2)

    def test_report_reads_as_a_table(self, monitor: Monitor) -> None:
        text = str(simulate_budget(monitor, make_runs(200, seed=1, tasks=20), budgets=[100]))

        assert "solved" in text
        assert "restart" in text
