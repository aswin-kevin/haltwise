import math
import random
from collections.abc import Sequence

import pytest

from haltwise.errors import HaltwiseError
from haltwise.policies import Density, FirstCrossing, Policy


class TestFirstCrossing:
    def test_path_statistic_is_the_highest_risk(self) -> None:
        assert FirstCrossing().path_statistic([0.1, 0.7, 0.3]) == 0.7

    def test_empty_path_can_never_stop(self) -> None:
        assert FirstCrossing().path_statistic([]) == -math.inf

    def test_stops_at_first_risk_above_threshold(self) -> None:
        assert FirstCrossing().stop_index([0.1, 0.6, 0.9], threshold=0.5) == 1

    def test_does_not_stop_when_risk_equals_threshold(self) -> None:
        assert FirstCrossing().stop_index([0.5, 0.5], threshold=0.5) is None


class TestDensity:
    def test_needs_the_required_number_of_hits_inside_the_window(self) -> None:
        policy = Density(hits=2, window=3)

        assert policy.stop_index([0.9, 0.1, 0.1, 0.9], threshold=0.5) is None
        assert policy.stop_index([0.9, 0.1, 0.9], threshold=0.5) == 2

    def test_path_statistic_is_best_window_kth_largest(self) -> None:
        policy = Density(hits=2, window=3)

        assert policy.path_statistic([0.9, 0.1, 0.1, 0.8, 0.6]) == 0.6

    def test_short_path_cannot_reach_required_hits(self) -> None:
        assert Density(hits=3, window=5).path_statistic([0.9, 0.9]) == -math.inf

    @pytest.mark.parametrize(("hits", "window"), [(0, 3), (4, 3), (1, 0)])
    def test_rejects_impossible_settings(self, hits: int, window: int) -> None:
        with pytest.raises(ValueError, match="hits"):
            Density(hits=hits, window=window)


@pytest.mark.parametrize("policy", [FirstCrossing(), Density(2, 3), Density(3, 5)])
def test_live_decision_agrees_with_offline_statistic(policy: Policy) -> None:
    """The calibration guarantee holds only if both views agree, so check thoroughly."""
    rng = random.Random(0)
    for _ in range(500):
        risks = [rng.random() for _ in range(rng.randint(0, 12))]
        threshold = rng.random()

        stops_live = policy.stop_index(risks, threshold) is not None
        stops_offline = policy.path_statistic(risks) > threshold

        assert stops_live == stops_offline


def test_policies_round_trip_through_dict() -> None:
    for policy in (FirstCrossing(), Density(hits=2, window=4)):
        assert Policy.from_dict(policy.to_dict()) == policy


def test_custom_policy_that_is_not_a_dataclass_explains_how_to_save() -> None:
    class Custom(Policy):
        lookback = 1

        def triggers(self, recent_risks: Sequence[float], threshold: float) -> bool:
            return False

        def path_statistic(self, risks: Sequence[float]) -> float:
            return -math.inf

    with pytest.raises(HaltwiseError, match="dataclass"):
        Custom().to_dict()
