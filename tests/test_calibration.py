import math
import random

import pytest

from haltwise.calibration import conformal_threshold, min_calibration_runs, one_per_group
from haltwise.errors import CalibrationError


def test_minimum_runs_needed_for_common_targets() -> None:
    assert min_calibration_runs(0.10) == 9
    assert min_calibration_runs(0.05) == 19
    assert min_calibration_runs(0.01) == 99


def test_threshold_is_the_conformal_order_statistic() -> None:
    statistics = [i / 10 for i in range(1, 20)]  # 19 values: 0.1 .. 1.9

    # n=19, alpha=0.1 -> k = ceil(20 * 0.9) = 18 -> 18th smallest = 1.8
    assert conformal_threshold(statistics, false_kill_rate=0.1) == pytest.approx(1.8)


def test_too_few_runs_raises_an_actionable_error() -> None:
    with pytest.raises(CalibrationError, match="at least 19 successful runs"):
        conformal_threshold([0.1] * 5, false_kill_rate=0.05)


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 1.5])
def test_rejects_rates_outside_zero_one(bad_rate: float) -> None:
    with pytest.raises(ValueError, match="false_kill_rate"):
        conformal_threshold([0.1] * 100, false_kill_rate=bad_rate)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_new_successful_runs_are_killed_at_most_alpha_of_the_time(alpha: float) -> None:
    """The guarantee itself: holds for exchangeable runs, averaged over calibration draws."""
    rng = random.Random(42)
    trials, kills = 4000, 0
    for _ in range(trials):
        calibration = [rng.gauss(0, 1) for _ in range(40)]
        threshold = conformal_threshold(calibration, false_kill_rate=alpha)
        kills += rng.gauss(0, 1) > threshold

    assert kills / trials <= alpha + 0.015


def _clustered_tasks(rng: random.Random, tasks: int) -> tuple[list[float], list[str]]:
    """Tasks differ a lot; runs within a task are near-copies. Cluster sizes vary wildly."""
    statistics, groups = [], []
    for task in range(tasks):
        task_level = rng.gauss(0, 1)
        for _ in range(rng.choice([1, 2, 5, 40])):
            statistics.append(task_level + rng.gauss(0, 0.05))
            groups.append(f"t{task}")
    return statistics, groups


def _new_task_run(rng: random.Random) -> float:
    return rng.gauss(0, 1) + rng.gauss(0, 0.05)


def test_one_run_per_task_keeps_the_guarantee_for_new_tasks() -> None:
    rng = random.Random(7)
    trials, kills, alpha = 3000, 0, 0.1
    for _ in range(trials):
        statistics, groups = _clustered_tasks(rng, tasks=30)
        picked = one_per_group(statistics, groups, seed=rng.randrange(10**6))
        kills += _new_task_run(rng) > conformal_threshold(picked, false_kill_rate=alpha)

    assert kills / trials <= alpha + 0.015


def test_one_per_group_keeps_exactly_one_value_from_each_group() -> None:
    picked = one_per_group([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "c"], seed=0)

    assert len(picked) == 3
    assert {3.0, 4.0} <= set(picked)


def test_one_per_group_is_reproducible() -> None:
    values, groups = [float(i) for i in range(50)], [str(i % 7) for i in range(50)]

    assert one_per_group(values, groups, seed=1) == one_per_group(values, groups, seed=1)


def _binom_cdf(j: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(j + 1))


class TestConfidenceMode:
    def test_minimum_tasks_for_a_high_confidence_bound(self) -> None:
        # (1 - 0.05)^n <= 0.05 first holds at n = 59
        assert min_calibration_runs(0.05, confidence=0.95) == 59
        assert min_calibration_runs(0.10, confidence=0.90) == 22

    def test_picks_the_lowest_order_statistic_that_is_valid_with_confidence(self) -> None:
        statistics = [float(i) for i in range(1, 101)]
        allowed_above = max(j for j in range(101) if _binom_cdf(j, 100, 0.1) <= 0.05)

        threshold = conformal_threshold(statistics, 0.1, confidence=0.95)

        assert threshold == 100 - allowed_above

    def test_is_never_looser_than_the_average_guarantee(self) -> None:
        rng = random.Random(0)
        statistics = [rng.gauss(0, 1) for _ in range(300)]

        assert conformal_threshold(statistics, 0.05, confidence=0.9) >= conformal_threshold(
            statistics, 0.05
        )

    def test_too_few_tasks_explains_the_confidence_requirement(self) -> None:
        with pytest.raises(CalibrationError, match="95% confidence"):
            conformal_threshold([0.1] * 40, 0.05, confidence=0.95)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
    def test_rejects_confidence_outside_zero_one(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            conformal_threshold([0.1] * 100, 0.05, confidence=confidence)

    def test_true_kill_rate_exceeds_target_in_at_most_one_minus_confidence_of_draws(
        self,
    ) -> None:
        rng = random.Random(11)
        alpha, confidence, draws, violations = 0.1, 0.9, 2000, 0
        for _ in range(draws):
            calibration = [rng.gauss(0, 1) for _ in range(80)]
            threshold = conformal_threshold(calibration, alpha, confidence=confidence)
            true_kill_rate = 0.5 * math.erfc(threshold / math.sqrt(2))  # P(Z > threshold)
            violations += true_kill_rate > alpha

        assert violations / draws <= (1 - confidence) + 0.02
