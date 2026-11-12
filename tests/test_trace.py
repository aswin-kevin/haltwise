import numpy as np
import pytest

from haltwise import Run, Step, StepKind, StepStatus


class TestStep:
    def test_accepts_plain_strings_for_kind_and_status(self) -> None:
        step = Step(kind="tool_call", status="error")

        assert step.kind is StepKind.TOOL_CALL
        assert step.status is StepStatus.ERROR

    def test_rejects_unknown_kind_with_helpful_message(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            Step(kind="banana")

    def test_tool_call_digest_ignores_argument_order(self) -> None:
        first = Step.tool_call("search", {"query": "cats", "limit": 5})
        second = Step.tool_call("search", {"limit": 5, "query": "cats"})

        assert first.args_digest == second.args_digest

    def test_tool_call_digest_changes_with_arguments(self) -> None:
        cats = Step.tool_call("search", {"query": "cats"})
        dogs = Step.tool_call("search", {"query": "dogs"})

        assert cats.args_digest != dogs.args_digest

    def test_error_shortcut_marks_status_as_error(self) -> None:
        step = Step.error("Timeout talking to API")

        assert step.kind is StepKind.ERROR
        assert step.status is StepStatus.ERROR
        assert step.failed

    def test_ok_step_is_not_failed(self) -> None:
        assert not Step.llm("thinking...").failed

    def test_round_trips_through_dict(self) -> None:
        step = Step.tool_call("read_file", {"path": "a.py"}, status="ok", tokens=12, cost=0.01)

        assert Step.from_dict(step.to_dict()) == step

    def test_from_dict_computes_digest_from_raw_args(self) -> None:
        step = Step.from_dict({"kind": "tool_call", "name": "ls", "args": {"path": "."}})

        assert step.args_digest == Step.tool_call("ls", {"path": "."}).args_digest


class TestRun:
    def test_stores_steps_as_an_immutable_tuple(self) -> None:
        run = Run(steps=[Step.llm("hi")], succeeded=True)

        assert isinstance(run.steps, tuple)
        assert len(run) == 1

    def test_outcome_is_optional(self) -> None:
        assert Run(steps=[]).succeeded is None

    def test_round_trips_through_dict(self) -> None:
        run = Run(steps=[Step.error("boom")], succeeded=False, id="r1", task="t1")

        assert Run.from_dict(run.to_dict()) == run


class TestOutcomeLabels:
    @pytest.mark.parametrize("label", [False, 0, np.bool_(False), "false", "False", "0"])
    def test_falsy_labels_mean_failed(self, label: object) -> None:
        run = Run([], succeeded=label)  # type: ignore[arg-type]

        assert run.succeeded is False
        assert run.failed

    @pytest.mark.parametrize("label", [True, 1, np.bool_(True), "true", "TRUE", "1"])
    def test_truthy_labels_mean_succeeded(self, label: object) -> None:
        assert Run([], succeeded=label).succeeded is True  # type: ignore[arg-type]

    @pytest.mark.parametrize("label", ["yes", 2, 0.5, np.array([1])])
    def test_ambiguous_labels_are_rejected(self, label: object) -> None:
        with pytest.raises(ValueError, match="succeeded"):
            Run([], succeeded=label)  # type: ignore[arg-type]
