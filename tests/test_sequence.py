import pytest

from haltwise import FeatureScorer, Run, Step
from haltwise.features import FEATURE_NAMES
from haltwise.sequence import SEQUENCE_FEATURE_NAMES, StateRates, cross_fit, symbol
from haltwise.synthetic import make_runs


def runs_with(steps: list[Step], succeeded: bool, copies: int) -> list[Run]:
    return [Run(steps, succeeded=succeeded) for _ in range(copies)]


LOOP = [Step.tool_call("run_tests"), Step.error("failed")]
HEALTHY = [Step.tool_call("read_file"), Step.tool_result("read_file")]


@pytest.fixture
def rates() -> StateRates:
    training = runs_with(LOOP, False, 30) + runs_with(HEALTHY, True, 30)
    return StateRates(order=2, prior_strength=10, min_count=5).fit(training)


def final_features(rates: StateRates, steps: list[Step]) -> list[float]:
    tracker = rates.new_tracker()
    values: list[float] = []
    for step in steps:
        values = tracker.update(step)
    return values


class TestStateRates:
    def test_states_seen_in_failing_runs_score_above_the_base_rate(self, rates: StateRates) -> None:
        assert final_features(rates, LOOP)[0] > rates.base_rate

    def test_states_seen_in_successful_runs_score_below_it(self, rates: StateRates) -> None:
        assert final_features(rates, HEALTHY)[0] < rates.base_rate

    def test_unknown_states_fall_back_to_the_base_rate(self, rates: StateRates) -> None:
        assert final_features(rates, [Step.tool_call("never_seen")])[0] == rates.base_rate

    def test_rare_states_are_not_memorised(self) -> None:
        rare = runs_with([Step.tool_call("odd")], False, 2) + runs_with(HEALTHY, True, 30)
        rates = StateRates(order=1, min_count=5).fit(rare)

        assert final_features(rates, [Step.tool_call("odd")])[0] == rates.base_rate

    def test_returns_one_value_per_sequence_feature(self, rates: StateRates) -> None:
        assert len(final_features(rates, LOOP)) == len(SEQUENCE_FEATURE_NAMES)

    def test_round_trips_through_dict(self, rates: StateRates) -> None:
        restored = StateRates.from_dict(rates.to_dict())

        assert final_features(restored, LOOP) == final_features(rates, LOOP)

    def test_order_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="order"):
            StateRates(order=0)


@pytest.fixture(scope="module")
def scorer() -> FeatureScorer:
    return FeatureScorer(sequence_order=3).fit(make_runs(300, seed=0))


class TestFeatureScorerWithSequences:
    def test_uses_sequence_features_by_default(self) -> None:
        assert FeatureScorer().feature_names == FEATURE_NAMES + SEQUENCE_FEATURE_NAMES

    def test_can_switch_sequences_off(self) -> None:
        assert FeatureScorer(sequence_order=0).feature_names == FEATURE_NAMES

    def test_live_tracker_matches_offline_path(self, scorer: FeatureScorer) -> None:
        steps = list(make_runs(1, seed=5)[0].steps)
        tracker = scorer.new_tracker()

        live = [tracker.update(step) for step in steps]

        assert live == pytest.approx(scorer.risk_path(steps))

    def test_round_trips_through_dict(self, scorer: FeatureScorer) -> None:
        steps = list(make_runs(1, seed=5)[0].steps)
        restored = FeatureScorer.from_dict(scorer.to_dict())

        assert restored.risk_path(steps) == pytest.approx(scorer.risk_path(steps))


def test_min_count_counts_tasks_not_near_copy_runs() -> None:
    one_task = [Run([Step.tool_call("odd")], succeeded=False, task="t") for _ in range(30)]
    others = [Run(HEALTHY, succeeded=True, task=f"h{i}") for i in range(30)]

    rates = StateRates(order=1, min_count=5).fit(one_task + others)

    assert final_features(rates, [Step.tool_call("odd")])[0] == rates.base_rate


def test_cross_fit_tables_never_see_their_own_task() -> None:
    runs = [
        Run([Step.tool_call(f"unique-{i}")], succeeded=i % 2 == 0, task=f"t{i}")
        for i in range(20)
        for _ in range(3)
    ]
    in_sample = StateRates(order=1, min_count=1).fit(runs)

    tables = cross_fit(runs, order=1, min_count=1, folds=5, seed=0)

    for i in range(20):
        own_state = (symbol(Step.tool_call(f"unique-{i}")),)
        assert in_sample.rate(own_state) != in_sample.base_rate  # memorised in-sample
        assert tables[f"task:t{i}"].rate(own_state) == tables[f"task:t{i}"].base_rate


def test_cross_fit_needs_at_least_two_folds() -> None:
    runs = [Run([Step.llm()], succeeded=i % 2 == 0, task=f"t{i}") for i in range(4)]

    with pytest.raises(ValueError, match="folds"):
        cross_fit(runs, folds=1)


def test_feature_scorer_trains_on_a_single_task_without_crashing() -> None:
    runs = [Run(LOOP, succeeded=False, task="one")] * 5 + [
        Run(HEALTHY, succeeded=True, task="one")
    ] * 5

    scorer = FeatureScorer().fit(runs)

    assert scorer.risk_path(LOOP)[-1] > scorer.risk_path(HEALTHY)[-1]
