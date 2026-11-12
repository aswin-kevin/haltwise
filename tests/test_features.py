from haltwise import Step
from haltwise.features import FEATURE_NAMES, TraceFeatures


def features_after(steps: list[Step]) -> dict[str, float]:
    tracker = TraceFeatures()
    for step in steps:
        tracker.update(step)
    return tracker.as_dict()


def test_vector_matches_the_published_feature_names() -> None:
    tracker = TraceFeatures()
    tracker.update(Step.llm("plan"))

    assert len(tracker.vector()) == len(FEATURE_NAMES)


def test_repeated_identical_calls_are_counted() -> None:
    call = Step.tool_call("run_tests", {"path": "tests/"})

    features = features_after([call, call, call])

    assert features["repeat_in_window"] == 3


def test_different_arguments_are_not_repeats() -> None:
    features = features_after(
        [Step.tool_call("read", {"f": "a"}), Step.tool_call("read", {"f": "b"})]
    )

    assert features["repeat_in_window"] == 1


def test_error_streak_grows_across_interleaved_llm_steps() -> None:
    steps = [Step.error("x"), Step.llm("retry"), Step.error("x")]

    assert features_after(steps)["error_streak"] == 2


def test_successful_tool_result_resets_the_error_streak() -> None:
    steps = [Step.error("x"), Step.error("x"), Step.tool_result("ls", "ok")]

    assert features_after(steps)["error_streak"] == 0


def test_steps_since_new_call_counts_stalled_progress() -> None:
    call = Step.tool_call("search", {"q": "same"})
    tracker = TraceFeatures()
    for step in [call, Step.llm(), call, Step.llm()]:
        tracker.update(step)

    # The last brand-new call happened 3 steps ago.
    assert tracker.steps_since_new_call == 3


def test_empty_trace_has_neutral_features() -> None:
    assert all(value == 0 for value in TraceFeatures().vector())
