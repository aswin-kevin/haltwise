from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from haltwise import AdaptiveMonitor, Decision, Monitor, StepKind, StepStatus, digest_args
from haltwise.adapters.otel import HaltwiseSpanProcessor, span_to_steps, spans_to_runs
from haltwise.errors import NotCalibratedError
from haltwise.policies import FirstCrossing
from haltwise.scorers import RuleScorer

# ---- fake SDK spans (duck-typed like opentelemetry.sdk.trace.ReadableSpan) -----------------


class FakeStatusCode(Enum):
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class FakeStatus:
    status_code: FakeStatusCode = FakeStatusCode.UNSET
    description: str | None = None


@dataclass
class FakeContext:
    trace_id: int
    span_id: int


@dataclass
class FakeSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: int = 0xABC
    span_id: int = 1
    start_time: int | None = 0
    end_time: int | None = 1
    status: FakeStatus = field(default_factory=FakeStatus)
    parent: FakeContext | None = None

    @property
    def context(self) -> FakeContext:
        return FakeContext(self.trace_id, self.span_id)


TRACE_HEX = format(0xABC, "032x")


def tool_span(
    tool: str = "run_tests",
    args: Any = None,
    *,
    error: bool = False,
    start: int = 0,
    end: int | None = None,
    trace_id: int = 0xABC,
    extra: dict[str, Any] | None = None,
) -> FakeSpan:
    attributes: dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool}
    if args is not None:
        attributes["gen_ai.tool.call.arguments"] = args
    attributes.update(extra or {})
    status = FakeStatus(FakeStatusCode.ERROR, "boom") if error else FakeStatus()
    return FakeSpan(
        f"execute_tool {tool}",
        attributes,
        trace_id=trace_id,
        start_time=start,
        end_time=start + 1 if end is None else end,
        status=status,
    )


def chat_span(
    model: str = "gpt-x", *, start: int = 0, extra: dict[str, Any] | None = None
) -> FakeSpan:
    attributes: dict[str, Any] = {"gen_ai.operation.name": "chat", "gen_ai.request.model": model}
    attributes.update(extra or {})
    return FakeSpan(f"chat {model}", attributes, start_time=start, end_time=start + 1)


# ---- span_to_steps ---------------------------------------------------------------------------


class TestSpanToSteps:
    def test_tool_span_becomes_call_then_result(self) -> None:
        call, result = span_to_steps(tool_span("read_file", '{"path": "a.py"}'))

        assert call.kind is StepKind.TOOL_CALL
        assert call.name == "read_file"
        assert call.args_digest == digest_args({"path": "a.py"})
        assert result.kind is StepKind.TOOL_RESULT
        assert result.name == "read_file"
        assert result.status is StepStatus.OK

    def test_json_arguments_ignore_key_order(self) -> None:
        a = span_to_steps(tool_span("t", '{"a": 1, "b": 2}'))[0]
        b = span_to_steps(tool_span("t", '{"b": 2, "a": 1}'))[0]

        assert a.args_digest == b.args_digest

    def test_non_json_string_arguments_are_fingerprinted_verbatim(self) -> None:
        call = span_to_steps(tool_span("t", "not json {"))[0]

        assert call.args_digest == digest_args("not json {")

    def test_missing_arguments_give_digestless_call(self) -> None:
        call, _ = span_to_steps(tool_span("t"))

        assert call.args_digest is None

    def test_errored_tool_span_gives_failed_result(self) -> None:
        call, result = span_to_steps(tool_span("t", error=True))

        assert call.status is StepStatus.OK
        assert result.status is StepStatus.ERROR

    def test_error_type_attribute_marks_result_failed(self) -> None:
        _, result = span_to_steps(tool_span("t", extra={"error.type": "TimeoutError"}))

        assert result.status is StepStatus.ERROR
        assert result.meta["error_type"] == "TimeoutError"

    @pytest.mark.parametrize("operation", ["chat", "text_completion", "generate_content"])
    def test_inference_span_becomes_llm_step(self, operation: str) -> None:
        span = chat_span(
            "gpt-x",
            extra={
                "gen_ai.operation.name": operation,
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
            },
        )

        (step,) = span_to_steps(span)

        assert step.kind is StepKind.LLM
        assert step.name == "gpt-x"
        assert step.tokens == 120
        assert step.status is StepStatus.OK
        assert step.text is None

    def test_llm_name_falls_back_to_response_model(self) -> None:
        span = FakeSpan("chat", {"gen_ai.operation.name": "chat", "gen_ai.response.model": "m-2"})

        (step,) = span_to_steps(span)

        assert step.name == "m-2"

    def test_llm_tokens_none_when_usage_missing(self) -> None:
        (step,) = span_to_steps(chat_span())

        assert step.tokens is None

    def test_llm_tokens_with_only_one_side(self) -> None:
        (step,) = span_to_steps(chat_span(extra={"gen_ai.usage.output_tokens": 7}))

        assert step.tokens == 7

    def test_errored_inference_span_gives_failed_llm_step(self) -> None:
        span = chat_span()
        span.status = FakeStatus(FakeStatusCode.ERROR)

        (step,) = span_to_steps(span)

        assert step.status is StepStatus.ERROR

    @pytest.mark.parametrize("key", ["gen_ai.provider.name", "gen_ai.system"])
    def test_provider_goes_into_meta(self, key: str) -> None:
        (step,) = span_to_steps(chat_span(extra={key: "openai"}))

        assert step.meta["provider"] == "openai"

    @pytest.mark.parametrize("operation", ["invoke_agent", "create_agent", "invoke_workflow"])
    def test_agent_spans_are_not_steps(self, operation: str) -> None:
        span = FakeSpan(operation, {"gen_ai.operation.name": operation})

        assert span_to_steps(span) == []

    def test_non_genai_span_is_ignored(self) -> None:
        assert span_to_steps(FakeSpan("GET /", {"http.request.method": "GET"})) == []

    def test_unknown_operation_is_ignored(self) -> None:
        assert span_to_steps(FakeSpan("x", {"gen_ai.operation.name": "embeddings"})) == []

    def test_tool_span_without_operation_name_is_recognised(self) -> None:
        steps = span_to_steps(FakeSpan("tool", {"gen_ai.tool.name": "search"}))

        assert [s.kind for s in steps] == [StepKind.TOOL_CALL, StepKind.TOOL_RESULT]

    def test_message_content_is_never_copied(self) -> None:
        span = chat_span(extra={"gen_ai.input.messages": "secret", "gen_ai.output.messages": "x"})

        (step,) = span_to_steps(span)

        assert step.text is None
        assert "secret" not in repr(step.to_dict())


class TestOtlpJsonSpans:
    def test_otlp_key_value_attributes(self) -> None:
        span = {
            "name": "execute_tool search",
            "traceId": "5B8EFFF798038103D269B633813FC60C",
            "spanId": "EEE19B7EC3C1B174",
            "startTimeUnixNano": "1700000000000000000",
            "status": {"code": 2},
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "search"}},
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {
                        "kvlistValue": {"values": [{"key": "q", "value": {"intValue": "3"}}]}
                    },
                },
            ],
        }

        call, result = span_to_steps(span)

        assert call.name == "search"
        assert call.args_digest == digest_args({"q": 3})
        assert result.status is StepStatus.ERROR

    @pytest.mark.parametrize("code", [2, "STATUS_CODE_ERROR", "ERROR", "error"])
    def test_status_code_spellings(self, code: object) -> None:
        span = {
            "name": "chat",
            "attributes": {"gen_ai.operation.name": "chat"},
            "status": {"code": code},
        }

        (step,) = span_to_steps(span)

        assert step.status is StepStatus.ERROR

    def test_sdk_to_json_shape(self) -> None:
        span = {
            "name": "chat",
            "context": {"trace_id": "0xabc", "span_id": "0x1"},
            "parent_id": None,
            "start_time": "2024-01-01T00:00:00.000001Z",
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 3},
            "status": {"status_code": "OK"},
        }

        (run,) = spans_to_runs([span])

        assert run.id == TRACE_HEX
        assert run.steps[0].tokens == 3
        assert run.steps[0].status is StepStatus.OK

    def test_string_numbers_in_usage_are_parsed(self) -> None:
        span = {
            "name": "chat",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "12"}},
            ],
        }

        (step,) = span_to_steps(span)

        assert step.tokens == 12


# ---- spans_to_runs ---------------------------------------------------------------------------


class TestSpansToRuns:
    def test_groups_by_trace_and_orders_by_start_time(self) -> None:
        spans = [
            tool_span("b", start=20),
            chat_span(start=10),
            tool_span("other", start=5, trace_id=0xDEF),
        ]

        runs = spans_to_runs(spans)

        assert [r.id for r in runs] == [format(0xDEF, "032x"), TRACE_HEX]
        main = runs[1]
        assert [s.kind for s in main.steps] == [
            StepKind.LLM,
            StepKind.TOOL_CALL,
            StepKind.TOOL_RESULT,
        ]

    def test_outcomes_label_runs(self) -> None:
        runs = spans_to_runs([chat_span()], outcomes={TRACE_HEX: False})

        assert runs[0].succeeded is False

    def test_unlabelled_runs_have_unknown_outcome(self) -> None:
        assert spans_to_runs([chat_span()])[0].succeeded is None

    def test_task_comes_from_haltwise_task_attribute(self) -> None:
        agent = FakeSpan("invoke_agent", {"gen_ai.operation.name": "invoke_agent"})
        agent.attributes["haltwise.task"] = "fix-bug-42"

        (run,) = spans_to_runs([agent, chat_span()])

        assert run.task == "fix-bug-42"
        assert len(run) == 1

    def test_task_is_none_by_default(self) -> None:
        assert spans_to_runs([chat_span()])[0].task is None

    def test_runs_without_steps_are_dropped(self) -> None:
        agent = FakeSpan("invoke_agent", {"gen_ai.operation.name": "invoke_agent"})

        assert spans_to_runs([agent, FakeSpan("GET /")]) == []

    def test_group_by_conversation_spans_traces(self) -> None:
        first = chat_span(start=1, extra={"gen_ai.conversation.id": "conv-1"})
        second = chat_span(start=2, extra={"gen_ai.conversation.id": "conv-1"})
        second.trace_id = 0xDEF

        (run,) = spans_to_runs([first, second], group_by="conversation")

        assert run.id == "conv-1"
        assert len(run) == 2

    def test_conversation_id_is_inherited_within_a_trace(self) -> None:
        agent = FakeSpan(
            "invoke_agent",
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.conversation.id": "conv-9"},
        )

        (run,) = spans_to_runs([chat_span(start=3), agent], group_by="conversation")

        assert run.id == "conv-9"

    def test_conversation_falls_back_to_trace_id(self) -> None:
        (run,) = spans_to_runs([chat_span()], group_by="conversation")

        assert run.id == TRACE_HEX

    def test_rejects_unknown_grouping(self) -> None:
        with pytest.raises(ValueError, match="group_by"):
            spans_to_runs([], group_by="session")  # type: ignore[arg-type]

    def test_spans_without_timestamps_keep_input_order(self) -> None:
        spans = [tool_span("a", start=5), tool_span("b", start=0)]
        for span in spans:
            span.start_time = None
            span.end_time = None

        (run,) = spans_to_runs(spans)

        assert [s.name for s in run.steps] == ["a", "a", "b", "b"]


# ---- HaltwiseSpanProcessor -------------------------------------------------------------------


def make_monitor() -> Monitor:
    return Monitor(RuleScorer(), policy=FirstCrossing(), threshold=0.5)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Decision]] = []

    def __call__(self, run_id: str, decision: Decision) -> None:
        self.calls.append((run_id, decision))


def feed_loop(processor: HaltwiseSpanProcessor, times: int, trace_id: int = 0xABC) -> None:
    for i in range(times):
        span = tool_span("run_tests", '{"path": "tests/"}', start=i, trace_id=trace_id)
        processor.on_start(span)
        processor.on_end(span)


class TestSpanProcessor:
    def test_works_with_an_adaptive_monitor_and_hands_back_the_watch(self) -> None:
        adaptive = AdaptiveMonitor(make_monitor(), shadow_share=1e-9, seed=0)
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(adaptive, on_stop=recorder)

        feed_loop(processor, 12)
        watch = processor.finish(TRACE_HEX)

        assert len(recorder.calls) == 1
        assert watch is not None
        adaptive.record(watch, succeeded=None)
        assert adaptive.censored_runs == 1

    def test_stops_looping_run_exactly_once(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)

        feed_loop(processor, 12)

        assert len(recorder.calls) == 1
        run_id, decision = recorder.calls[0]
        assert run_id == TRACE_HEX
        assert decision.should_stop

    def test_healthy_run_is_not_stopped(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)

        for i in range(10):
            processor.on_end(tool_span("read_file", f'{{"path": "f{i}.py"}}', start=i))

        assert recorder.calls == []

    def test_runs_are_guarded_independently(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)

        for i in range(3):
            feed_loop(processor, 1, trace_id=0x1)
            processor.on_end(tool_span("read", f'{{"i": {i}}}', trace_id=0x2))

        assert recorder.calls == []

    def test_ignores_non_genai_spans(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)

        processor.on_end(FakeSpan("GET /"))

        assert processor.finish(TRACE_HEX) is None

    def test_finish_returns_and_drops_watch(self) -> None:
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=Recorder())
        processor.on_end(chat_span())

        watch = processor.finish(TRACE_HEX)

        assert watch is not None
        assert len(watch.risks) == 1
        assert processor.finish(TRACE_HEX) is None

    def test_evicts_least_recently_active_runs(self) -> None:
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=Recorder(), max_runs=2)
        for trace_id in (0x1, 0x2):
            processor.on_end(tool_span("t", trace_id=trace_id))
        processor.on_end(tool_span("t", trace_id=0x1))  # 0x1 is now the most recent

        processor.on_end(tool_span("t", trace_id=0x3))

        assert processor.finish(format(0x2, "032x")) is None
        assert processor.finish(format(0x1, "032x")) is not None
        assert processor.finish(format(0x3, "032x")) is not None

    def test_group_by_conversation_uses_id_seen_at_start(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder, group_by="conversation")
        agent = FakeSpan(
            "invoke_agent",
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.conversation.id": "conv-1"},
        )
        processor.on_start(agent)

        feed_loop(processor, 12)

        assert [run_id for run_id, _ in recorder.calls] == ["conv-1"]

    def test_stop_callback_may_call_finish(self) -> None:
        processor: HaltwiseSpanProcessor
        finished: list[object] = []

        def finish_on_stop(run_id: str, decision: Decision) -> None:
            finished.append(processor.finish(run_id))

        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=finish_on_stop)

        feed_loop(processor, 6)  # would deadlock if the callback ran under the lock

        assert len(finished) == 1
        assert finished[0] is not None

    def test_thread_safe_exactly_once(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)
        threads = [threading.Thread(target=feed_loop, args=(processor, 20)) for _ in range(8)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(recorder.calls) == 1

    def test_requires_a_threshold(self) -> None:
        with pytest.raises(NotCalibratedError):
            HaltwiseSpanProcessor(Monitor(RuleScorer()), on_stop=Recorder())

    def test_rejects_bad_options(self) -> None:
        with pytest.raises(ValueError, match="max_runs"):
            HaltwiseSpanProcessor(make_monitor(), on_stop=Recorder(), max_runs=0)
        with pytest.raises(ValueError, match="group_by"):
            HaltwiseSpanProcessor(make_monitor(), on_stop=Recorder(), group_by="x")  # type: ignore[arg-type]

    def test_span_processor_protocol(self) -> None:
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=Recorder())
        processor.on_end(chat_span())

        assert processor.force_flush() is True
        processor.shutdown()
        assert processor.finish(TRACE_HEX) is None


class TestLiveOfflineConsistency:
    def test_offline_orders_steps_by_completion_like_the_live_processor(self) -> None:
        slow = tool_span("slow_tool", start=1, end=10)
        fast = tool_span("fast_tool", start=2, end=3)

        (run,) = spans_to_runs([slow, fast])

        assert [step.name for step in run.steps] == ["fast_tool"] * 2 + ["slow_tool"] * 2

    def test_late_spans_of_a_finished_run_cannot_stop_it_twice(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder)
        feed_loop(processor, 12)

        processor.finish(TRACE_HEX)
        feed_loop(processor, 12)

        assert len(recorder.calls) == 1

    def test_a_slow_scorer_on_one_run_does_not_block_other_runs(self) -> None:
        release = threading.Event()

        class BlockingTracker:
            def update(self, step: Any) -> float:
                if step.name == "block":
                    release.wait(timeout=5)
                return 0.0

        class BlockingScorer(RuleScorer):
            def new_tracker(self) -> Any:
                return BlockingTracker()

        monitor = Monitor(BlockingScorer(), policy=FirstCrossing(), threshold=0.5)
        processor = HaltwiseSpanProcessor(monitor, on_stop=Recorder())
        blocked = threading.Thread(
            target=processor.on_end, args=(tool_span("block", trace_id=1),), daemon=True
        )
        blocked.start()

        other = threading.Thread(
            target=processor.on_end, args=(tool_span("ls", trace_id=2),), daemon=True
        )
        other.start()
        other.join(timeout=1)

        assert not other.is_alive()  # finished while run 1 was still scoring
        release.set()
        blocked.join(timeout=5)


class TestRunLifecycle:
    def test_an_evicted_run_that_keeps_going_is_still_guarded(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder, max_runs=1)
        feed_loop(processor, 2, trace_id=1)  # not stopped yet
        processor.on_end(tool_span("ls", trace_id=2))  # evicts run 1

        feed_loop(processor, 12, trace_id=1)

        assert [run_id for run_id, _ in recorder.calls] == [format(1, "032x")]

    def test_a_stopped_run_is_never_stopped_again_even_after_eviction(self) -> None:
        recorder = Recorder()
        processor = HaltwiseSpanProcessor(make_monitor(), on_stop=recorder, max_runs=1)
        feed_loop(processor, 12, trace_id=1)  # stopped
        for other in range(2, 7):  # many other runs come and go
            processor.on_end(tool_span("ls", trace_id=other))
            processor.finish(format(other, "032x"))

        feed_loop(processor, 12, trace_id=1)

        assert len(recorder.calls) == 1

    def test_finish_waits_for_in_flight_scoring_of_that_run(self) -> None:
        release = threading.Event()

        class BlockingTracker:
            def update(self, step: Any) -> float:
                if step.name == "block":
                    release.wait(timeout=5)
                return 0.0

        class BlockingScorer(RuleScorer):
            def new_tracker(self) -> Any:
                return BlockingTracker()

        monitor = Monitor(BlockingScorer(), policy=FirstCrossing(), threshold=0.5)
        processor = HaltwiseSpanProcessor(monitor, on_stop=Recorder())
        scoring = threading.Thread(target=processor.on_end, args=(tool_span("block"),))
        scoring.start()
        finished: list[Any] = []
        finishing = threading.Thread(target=lambda: finished.append(processor.finish(TRACE_HEX)))
        time.sleep(0.05)
        finishing.start()
        finishing.join(timeout=0.3)

        assert finishing.is_alive()  # waiting for the in-flight span
        release.set()
        finishing.join(timeout=5)
        scoring.join(timeout=5)
        assert finished[0] is not None
        assert len(finished[0].risks) == 2  # the complete call + result
