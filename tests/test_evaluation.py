import pytest

from haltwise import Monitor, Run, Step
from haltwise.evaluation import auroc
from haltwise.policies import FirstCrossing
from haltwise.scorers import RuleScorer, Scorer


class TestAuroc:
    def test_perfect_separation_is_one(self) -> None:
        assert auroc(positives=[0.9, 0.8], negatives=[0.1, 0.2]) == 1.0

    def test_reversed_separation_is_zero(self) -> None:
        assert auroc(positives=[0.1], negatives=[0.9]) == 0.0

    def test_ties_count_as_half(self) -> None:
        assert auroc(positives=[0.5], negatives=[0.5]) == 0.5

    def test_undefined_without_both_classes(self) -> None:
        assert auroc(positives=[0.5], negatives=[]) is None


def test_report_counts_kills_and_savings_exactly() -> None:
    loop = Step.tool_call("run_tests", {"p": 1})
    doomed = Run([loop] * 6, succeeded=False)  # passes the repeat limit at index 4
    fine = Run([Step.tool_call("ls", {"p": i}) for i in range(4)], succeeded=True)
    unlabelled = Run([loop], succeeded=None)
    monitor = Monitor(RuleScorer(repeat_limit=4), policy=FirstCrossing(), threshold=0.5)

    report = monitor.evaluate([doomed, fine, unlabelled])

    assert report.runs == 2  # unlabelled runs are ignored
    assert report.false_kill_rate == 0.0
    assert report.failures_stopped == 1.0
    assert report.steps_saved == pytest.approx(1 / 6)
    assert report.auroc == 1.0


def test_tokens_saved_is_reported_only_when_tokens_are_known() -> None:
    loop = Step.tool_call("run_tests", {"p": 1}, tokens=10)
    doomed = Run([loop] * 6, succeeded=False)
    monitor = Monitor(RuleScorer(repeat_limit=4), policy=FirstCrossing(), threshold=0.5)

    with_tokens = monitor.evaluate([doomed])
    without = monitor.evaluate([Run([Step.tool_call("x", {})] * 6, succeeded=False)])

    assert with_tokens.tokens_saved == pytest.approx(10 / 60)
    assert without.tokens_saved is None


def test_task_false_kill_rate_averages_over_tasks() -> None:
    loop = Step.tool_call("run_tests", {"p": 1})
    killed = Run([loop] * 6, succeeded=True, task="a")
    spared_same_task = Run([Step.llm()], succeeded=True, task="a")
    spared_other_task = Run([Step.llm()], succeeded=True, task="b")
    monitor = Monitor(RuleScorer(repeat_limit=4), policy=FirstCrossing(), threshold=0.5)

    report = monitor.evaluate([killed, spared_same_task, spared_other_task])

    assert report.false_kill_rate == pytest.approx(1 / 3)
    assert report.task_false_kill_rate == pytest.approx((1 / 2 + 0) / 2)


class _CostIsRisk(Scorer):
    def new_tracker(self) -> "_CostTracker":
        return _CostTracker()

    def to_dict(self) -> dict[str, str]:
        return {"type": "_CostIsRisk"}

    @classmethod
    def _from_params(cls, params: object) -> "_CostIsRisk":
        return cls()


class _CostTracker:
    def update(self, step: Step) -> float:
        return float(step.cost or 0.0)


def runs_for(task: str, risk: float, failures: int, successes: int, length: int = 12) -> list[Run]:
    def make(ok: bool) -> Run:
        return Run([Step.llm(cost=risk)] * length, succeeded=ok, task=task)

    return [make(False) for _ in range(failures)] + [make(True) for _ in range(successes)]


def test_within_task_auroc_exposes_a_scorer_that_only_knows_task_difficulty() -> None:
    # Hard task: risky-looking, mostly fails. Easy task: safe-looking, mostly succeeds.
    # Within each task every run looks the same, so the scorer can't tell attempts apart.
    runs = runs_for("hard", 0.9, failures=3, successes=1) + runs_for("easy", 0.1, 1, 3)
    monitor = Monitor(_CostIsRisk(), policy=FirstCrossing(), threshold=0.95)

    report = monitor.evaluate(runs)

    assert report.auroc_at_steps[10] is not None
    assert report.auroc_at_steps[10] > 0.7
    assert report.within_task_auroc_at_steps[10] == 0.5


def test_fixed_step_auroc_ignores_runs_that_never_reach_that_step() -> None:
    runs = runs_for("a", 0.9, 1, 1, length=6) + runs_for("b", 0.1, 1, 1, length=6)
    monitor = Monitor(_CostIsRisk(), policy=FirstCrossing(), threshold=0.95)

    report = monitor.evaluate(runs)

    assert report.auroc_at_steps[5] is not None
    assert report.auroc_at_steps[10] is None


def test_report_prints_the_honest_metrics() -> None:
    runs = runs_for("hard", 0.9, 3, 1) + runs_for("easy", 0.1, 1, 3)
    text = str(Monitor(_CostIsRisk(), policy=FirstCrossing(), threshold=0.95).evaluate(runs))

    assert "AUROC at step 10" in text
    assert "within-task" in text
