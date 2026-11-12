"""Guard a live agent through OpenTelemetry spans (no OpenTelemetry install needed here).

With the real SDK you would register the processor once:

    from opentelemetry.sdk.trace import TracerProvider
    provider = TracerProvider()
    provider.add_span_processor(HaltwiseSpanProcessor(monitor, on_stop=cancel_run))

and your GenAI instrumentation would emit the spans. Here we hand-build span-like objects
for an agent stuck re-running the same failing tests.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from haltwise import Decision, Monitor
from haltwise.adapters.otel import HaltwiseSpanProcessor
from haltwise.policies import FirstCrossing
from haltwise.scorers import RuleScorer

TRACE_ID = 0x5B8EFFF798038103D269B633813FC60C


class StatusCode(Enum):
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class Status:
    status_code: StatusCode = StatusCode.UNSET
    description: str | None = None


@dataclass
class SpanContext:
    trace_id: int
    span_id: int


@dataclass
class FakeSpan:
    """The few ReadableSpan fields haltwise reads."""

    name: str
    attributes: dict[str, Any]
    context: SpanContext
    start_time: int
    end_time: int
    status: Status = field(default_factory=Status)
    parent: SpanContext | None = None


def tool_span(i: int, tool: str, arguments: str, *, failed: bool) -> FakeSpan:
    return FakeSpan(
        name=f"execute_tool {tool}",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool,
            "gen_ai.tool.call.arguments": arguments,  # opt-in capture; enable it!
        },
        context=SpanContext(TRACE_ID, span_id=i + 1),
        start_time=i * 1_000_000,
        end_time=i * 1_000_000 + 500_000,
        status=Status(StatusCode.ERROR, "1 test failed") if failed else Status(),
    )


# Cold start: no past runs yet, so use the rule scorer with a hand-set threshold.
# 0.5 means "stop once any warning sign (looping, error streak, stalling) passes its limit".
monitor = Monitor(RuleScorer(), policy=FirstCrossing(), threshold=0.5)
cancelled: set[str] = set()


def cancel_run(run_id: str, decision: Decision) -> None:
    # In a real agent: set a cancellation flag your agent loop checks before its next step.
    cancelled.add(run_id)
    print(f"STOP run {run_id} at step {decision.step + 1}: {decision.reason}")


processor = HaltwiseSpanProcessor(monitor, on_stop=cancel_run)
run_id = format(TRACE_ID, "032x")  # the id on_stop receives (trace id as 32 hex digits)

for i in range(20):
    if run_id in cancelled:
        print(f"Agent loop sees the cancellation and stops after {i} tool calls.")
        break
    span = tool_span(i, "run_tests", '{"path": "tests/"}', failed=True)
    processor.on_start(span)
    print(f"tool call {i + 1}: run_tests -> failed")
    processor.on_end(span)  # steps are observed when the span ends

watch = processor.finish(run_id)  # free the run's state once it is over
assert watch is not None
print("risk per step:", [round(risk, 2) for risk in watch.risks])
