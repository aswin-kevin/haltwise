import warnings
from typing import Any

import numpy as np
import pytest

from haltwise import Run, Step
from haltwise.errors import HaltwiseError
from haltwise.scorers import FeatureScorer, RuleScorer, Scorer
from haltwise.scorers.logistic import LogisticRegression
from haltwise.synthetic import make_runs


def looping_steps(repeats: int) -> list[Step]:
    call = Step.tool_call("run_tests", {"path": "tests/"})
    return [call, Step.error("1 failed")] * repeats


class TestRuleScorer:
    def test_healthy_trace_has_zero_risk(self) -> None:
        steps = [Step.tool_call("ls", {"p": str(i)}) for i in range(5)]

        assert RuleScorer().risk_path(steps)[-1] == 0.0

    def test_reaching_a_limit_means_risk_one_half(self) -> None:
        assert RuleScorer(repeat_limit=4).risk_path(looping_steps(4))[-1] == pytest.approx(0.5)

    def test_risk_keeps_rising_past_the_limit_without_saturating(self) -> None:
        risks = RuleScorer(repeat_limit=4).risk_path(looping_steps(10))

        assert risks == sorted(risks)
        assert 0.5 < risks[-1] < 1.0
        assert risks[-1] > risks[-3]

    def test_returns_one_risk_per_step(self) -> None:
        assert len(RuleScorer().risk_path(looping_steps(3))) == 6


class TestLogisticRegression:
    def test_learns_a_simple_boundary(self) -> None:
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        model = LogisticRegression().fit(x, y)
        probabilities = model.predict_proba(x)[:, 1]

        assert probabilities[0] < 0.5 < probabilities[3]

    def test_fit_and_predict_emit_no_floating_point_warnings(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 12))
        y = (x[:, 0] + rng.normal(size=500) > 0).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LogisticRegression().fit(x, y).predict_proba(x)

    def test_non_finite_features_are_rejected_clearly(self) -> None:
        x = np.array([[0.0], [np.inf], [1.0]])

        with pytest.raises(ValueError, match="finite"):
            LogisticRegression().fit(x, np.array([0, 1, 1]))

    def test_round_trips_through_dict(self) -> None:
        x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        model = LogisticRegression().fit(x, np.array([0, 1, 1]))

        restored = LogisticRegression.from_dict(model.to_dict())

        np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))


@pytest.fixture(scope="module")
def scorer() -> FeatureScorer:
    return FeatureScorer().fit(make_runs(300, seed=0))


class TestFeatureScorer:
    def test_looping_trace_scores_higher_than_healthy_one(self, scorer: FeatureScorer) -> None:
        healthy = [Step.tool_call("read", {"f": str(i)}) for i in range(8)]

        assert scorer.risk_path(looping_steps(4))[-1] > scorer.risk_path(healthy)[-1]

    def test_live_tracker_matches_offline_path(self, scorer: FeatureScorer) -> None:
        steps = list(make_runs(1, seed=3)[0].steps)
        tracker = scorer.new_tracker()

        live = [tracker.update(step) for step in steps]

        assert live == pytest.approx(scorer.risk_path(steps))

    def test_needs_both_outcomes_to_learn(self) -> None:
        only_successes = [Run([Step.llm()], succeeded=True)] * 5

        with pytest.raises(HaltwiseError, match="both successful and failed"):
            FeatureScorer().fit(only_successes)

    def test_unfitted_scorer_explains_what_to_do(self) -> None:
        with pytest.raises(HaltwiseError, match="fit"):
            FeatureScorer().risk_path([Step.llm()])

    def test_round_trips_through_dict(self, scorer: FeatureScorer) -> None:
        steps = looping_steps(3)
        restored = Scorer.from_dict(scorer.to_dict())

        assert restored.risk_path(steps) == pytest.approx(scorer.risk_path(steps))

    def test_accepts_any_model_with_fit_and_predict_proba(self) -> None:
        class ConstantModel:
            def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "ConstantModel":
                return self

            def predict_proba(self, x: Any) -> Any:
                return np.tile([0.3, 0.7], (len(x), 1))

        scorer = FeatureScorer(model=ConstantModel()).fit(make_runs(20, seed=0))

        assert scorer.risk_path([Step.llm()]) == [pytest.approx(0.7)]
