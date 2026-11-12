"""Feed OpenTelemetry GenAI spans into haltwise, offline or live.

Follows the OpenTelemetry GenAI semantic conventions
(https://github.com/open-telemetry/semantic-conventions-genai). Every ``gen_ai.*`` name
is still *Development* status, so the mapping is lenient:

* ``execute_tool`` spans become **two** steps: a `Step.tool_call` (``gen_ai.tool.name``,
  arguments from ``gen_ai.tool.call.arguments``) and a `Step.tool_result` that failed
  when the span status is ERROR or ``error.type`` is set.
* Inference spans (``chat``, ``text_completion``, ``generate_content``) become one
  `Step.llm` named after the model, with input + output usage tokens.
* Agent and workflow spans (``invoke_agent``, ``create_agent``, ``invoke_workflow``...)
  are not steps; the run they belong to is identified by trace (or conversation) id.
* Spans without GenAI attributes, and other GenAI operations (e.g. ``embeddings``), are
  ignored. Spans with no operation name are recognised from ``gen_ai.tool.name`` or a
  model attribute.

Message content is never read, so content capture can stay off. Tool-call arguments are
opt-in in most instrumentations; **turn on ``gen_ai.tool.call.arguments`` capture** for
best results. Without it every call to the same tool looks identical (``args_digest`` is
``None``), so repeated-call detection cannot tell a stuck loop from varied calls.
Arguments are only fingerprinted (`haltwise.digest_args`), never stored.

No OpenTelemetry install is needed: spans are read by duck typing. Both the SDK's
``ReadableSpan`` and plain dicts in OTLP-JSON shape (``attributes`` as a mapping or as a
``[{"key": ..., "value": {"stringValue": ...}}]`` list) are accepted, as is the SDK's
``span.to_json()`` layout.

Typical use::

    runs = spans_to_runs(exported_spans, outcomes={"<trace id>": True, ...})
    monitor = Monitor(false_kill_rate=0.05).fit(runs)

    provider.add_span_processor(HaltwiseSpanProcessor(monitor, on_stop=cancel_run))
"""

from __future__ import annotations

import json
import string
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

from haltwise.monitor import Decision, Monitor, Watch
from haltwise.online import AdaptiveMonitor
from haltwise.trace import Run, Step, StepStatus

__all__ = ["GroupBy", "HaltwiseSpanProcessor", "span_to_steps", "spans_to_runs"]

GroupBy = Literal["trace", "conversation"]
"""How spans are grouped into runs: by trace id, or by ``gen_ai.conversation.id``."""

_GROUPINGS: tuple[GroupBy, ...] = ("trace", "conversation")

# Attribute names (GenAI semantic conventions, plus a few legacy spellings).
_OPERATION = "gen_ai.operation.name"
_TOOL_NAME = "gen_ai.tool.name"
_TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
_REQUEST_MODEL = "gen_ai.request.model"
_RESPONSE_MODEL = "gen_ai.response.model"
_PROVIDER_KEYS = ("gen_ai.provider.name", "gen_ai.system")
_INPUT_TOKEN_KEYS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
_OUTPUT_TOKEN_KEYS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")
_CONVERSATION_ID = "gen_ai.conversation.id"
_ERROR_TYPE = "error.type"
_TASK = "haltwise.task"

_EXECUTE_TOOL = "execute_tool"
_INFERENCE_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})

_TRACE_ID_WIDTH = 32
_SPAN_ID_WIDTH = 16
_STATUS_CODE_ERROR = 2  # numeric StatusCode.ERROR in OTLP


# ---- public API: offline ---------------------------------------------------------------------


def span_to_steps(span: object) -> list[Step]:
    """Convert one span into haltwise steps.

    Args:
        span: An SDK ``ReadableSpan`` or an OTLP-JSON-shaped dict.

    Returns:
        ``[tool_call, tool_result]`` for a tool span, ``[llm]`` for an inference span and
        ``[]`` for anything else (agent spans, non-GenAI spans, other operations).
    """
    return _steps_from(span, _attributes(span))


def spans_to_runs(
    spans: Iterable[object],
    *,
    group_by: GroupBy = "trace",
    outcomes: Mapping[str, bool] | None = None,
) -> list[Run]:
    """Group exported spans into runs, ready for `Monitor.fit` or `Monitor.evaluate`.

    Steps within a run are ordered by span end time, falling back to start time, which
    is the order `HaltwiseSpanProcessor` sees them live (spans without timestamps go
    last, in input order). Runs come out in order of their first step. Runs without any step
    (e.g. a trace holding only an ``invoke_agent`` span) are dropped.

    Args:
        spans: SDK ``ReadableSpan`` objects or OTLP-JSON-shaped dicts, in any order.
        group_by: ``"trace"`` makes one run per trace id (lowercase hex).
            ``"conversation"`` makes one run per ``gen_ai.conversation.id``. Spans without
            that attribute inherit it from another span in their trace (it is often set
            only on the ``invoke_agent`` span), else fall back to their trace id.
        outcomes: Maps run id (trace or conversation id) to whether the run succeeded.
            Runs missing from it get ``succeeded=None``.

    Returns:
        One `Run` per group. ``task`` is taken from a ``haltwise.task`` attribute on any
        span of the run, else ``None``.
    """
    _check_group_by(group_by)
    outcomes = outcomes or {}
    parsed = [(span, _attributes(span)) for span in spans]
    parsed.sort(key=lambda item: _sort_key(item[0]))

    conversations = _conversations_by_trace(parsed) if group_by == "conversation" else {}
    steps: dict[str | None, list[Step]] = {}
    tasks: dict[str | None, str] = {}
    for span, attributes in parsed:
        run_id = _run_id(span, attributes, group_by, conversations)
        steps.setdefault(run_id, []).extend(_steps_from(span, attributes))
        task = attributes.get(_TASK)
        if task is not None:
            tasks.setdefault(run_id, str(task))

    return [
        Run(
            run_steps,
            succeeded=outcomes.get(run_id) if run_id is not None else None,
            id=run_id,
            task=tasks.get(run_id),
        )
        for run_id, run_steps in steps.items()
        if run_steps
    ]


# ---- public API: live ------------------------------------------------------------------------


_CLOSED_RUNS_PER_ACTIVE_RUN = 10


class HaltwiseSpanProcessor:
    """An OpenTelemetry span processor that guards agent runs as their spans end.

    Register it like any processor (``tracer_provider.add_span_processor(...)``). It is
    duck-typed, so haltwise does not depend on OpenTelemetry. Each ended GenAI span is
    converted with `span_to_steps` and fed to its run's `Watch` (created on first sight
    with `Monitor.watch`). When the monitor recommends stopping, ``on_stop(run_id,
    decision)`` is called **once** for that run.

    Steps arrive when spans *end*: a tool call is seen only after the tool has finished,
    and a long LLM call only after it has returned. Stopping therefore happens between
    steps, never in the middle of one.

    Thread-safe: spans may end on any thread. ``on_stop`` runs on the thread that ended
    the triggering span, outside the processor's lock (so it may call `finish`). Keep it
    fast; an exception it raises propagates to the code that ended the span.

    Args:
        monitor: A monitor with a threshold (fitted, calibrated or set by hand), or an
            `AdaptiveMonitor`. With an adaptive one, pass the `Watch` returned by
            `finish` to ``adaptive.record(watch, succeeded)`` once the run's outcome is known.
        on_stop: Called as ``on_stop(run_id, decision)`` when a run should stop. With
            ``group_by="trace"`` the run id is the trace id as 32 lowercase hex digits,
            i.e. ``format(span.get_span_context().trace_id, "032x")``.
        group_by: ``"trace"`` or ``"conversation"`` (see `spans_to_runs`). In conversation
            mode, a span without ``gen_ai.conversation.id`` inherits it from a span of
            the same trace that carried it when it *started* (typically ``invoke_agent``).
        max_runs: Most runs tracked at once. Beyond it the least recently active run is
            forgotten. If it was still going and more of its spans arrive, it is watched
            afresh (its earlier steps are lost); if it had already been stopped, its late
            spans are ignored, so ``on_stop`` never fires twice for it. Finished and
            stopped runs are remembered for ``10 * max_runs`` further runs.

    Raises:
        NotCalibratedError: If ``monitor`` has no threshold yet.
        ValueError: If ``group_by`` or ``max_runs`` is invalid.
    """

    def __init__(
        self,
        monitor: Monitor | AdaptiveMonitor,
        *,
        on_stop: Callable[[str, Decision], None],
        group_by: GroupBy = "trace",
        max_runs: int = 10_000,
    ) -> None:
        _check_group_by(group_by)
        if max_runs < 1:
            raise ValueError(f"max_runs must be at least 1 (got {max_runs})")
        monitor.watch()  # fail now, not on the first span, if there is no threshold
        self._monitor = monitor
        self._on_stop = on_stop
        self._group_by: GroupBy = group_by
        self._max_runs = max_runs
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, _RunState] = OrderedDict()
        # Runs that were finished, or stopped and then evicted: their late spans are
        # ignored, so a run is never stopped twice. Kept much longer than active runs.
        self._closed: OrderedDict[str, None] = OrderedDict()
        self._conversations: OrderedDict[str, str] = OrderedDict()  # trace id -> conversation
        self._shut_down = False

    def __repr__(self) -> str:
        return (
            f"HaltwiseSpanProcessor(monitor={self._monitor!r}, group_by={self._group_by!r}, "
            f"max_runs={self._max_runs})"
        )

    # ---- SpanProcessor protocol -----------------------------------------------------

    def on_start(self, span: object, parent_context: object = None) -> None:
        """Remember the span's conversation id for its trace (conversation mode only)."""
        if self._group_by != "conversation":
            return
        conversation = _conversation_id(_attributes(span))
        trace = _trace_id(span)
        if conversation is None or trace is None:
            return
        with self._lock:
            _remember(self._conversations, trace, conversation, self._max_runs)

    def on_end(self, span: object) -> None:
        """Feed the span's steps to its run and report a stop the first time one is due.

        Only the bookkeeping holds the processor-wide lock; scoring holds a per-run lock,
        so a slow scorer on one run never blocks spans of other runs.
        """
        attributes = _attributes(span)
        steps = _steps_from(span, attributes)
        if not steps:
            return
        with self._lock:
            if self._shut_down:
                return
            run_id = _run_id(span, attributes, self._group_by, self._conversations)
            if run_id is None or run_id in self._closed:
                return
            state = self._state_for(run_id)
        state.observe(steps, lambda decision: self._on_stop(run_id, decision))

    def _on_ending(self, span: object) -> None:
        """Do nothing; recent SDKs call this on every processor just before ``on_end``."""

    def shutdown(self) -> None:
        """Forget every run and ignore spans that end from now on."""
        with self._lock:
            self._shut_down = True
            self._runs.clear()
            self._closed.clear()
            self._conversations.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Return ``True``: nothing is buffered, every span is handled as it ends."""
        return True

    # ---- run bookkeeping ------------------------------------------------------------

    def finish(self, run_id: str) -> Watch | None:
        """Stop tracking a finished run and return its `Watch` (``None`` if unknown).

        Call it when a run is over to free its state at once. Spans of that run that end
        later are ignored.
        """
        with self._lock:
            state = self._runs.pop(run_id, None)
            self._close(run_id)
        if state is None:
            return None
        state.close()  # waits for any span of this run that is still being scored
        return state.watch

    def _state_for(self, run_id: str) -> _RunState:
        """Return the run's state, creating it on first sight. Call with the lock held."""
        state = self._runs.get(run_id)
        if state is None:
            state = _RunState(self._monitor.watch())
            self._runs[run_id] = state
            while len(self._runs) > self._max_runs:
                evicted_id, evicted = self._runs.popitem(last=False)
                if evicted.watch.stopped:
                    self._close(evicted_id)  # never stop it twice
                # A run evicted while still going is watched afresh if it continues.
        self._runs.move_to_end(run_id)
        return state

    def _close(self, run_id: str) -> None:
        """Remember that ``run_id`` is over. Call with the lock held."""
        _remember(self._closed, run_id, None, self._max_runs * _CLOSED_RUNS_PER_ACTIVE_RUN)


class _RunState:
    """One tracked run: its watch, guarded by its own (re-entrant) lock."""

    def __init__(self, watch: Watch) -> None:
        self.watch = watch
        self._lock = threading.RLock()  # re-entrant: on_stop may call finish()
        self._closed = False

    def observe(self, steps: Sequence[Step], on_stop: Callable[[Decision], None]) -> None:
        """Feed ``steps``; call ``on_stop`` if, and only if, this call stops the run.

        ``on_stop`` runs under the run's lock, so it can never fire after `close`.
        """
        with self._lock:
            if self._closed or self.watch.stopped:
                return
            for step in steps:
                decision = self.watch.observe(step)
                if decision.should_stop:
                    on_stop(decision)
                    return

    def close(self) -> None:
        """Wait for in-flight scoring of this run, then ignore any further steps."""
        with self._lock:
            self._closed = True


# ---- span -> steps ---------------------------------------------------------------------------


def _steps_from(span: object, attributes: Mapping[str, Any]) -> list[Step]:
    operation = _operation(attributes)
    if operation == _EXECUTE_TOOL:
        return _tool_steps(span, attributes)
    if operation in _INFERENCE_OPERATIONS:
        return [_llm_step(span, attributes)]
    return []


def _operation(attributes: Mapping[str, Any]) -> str | None:
    """Return the GenAI operation, inferring it when the span does not name one."""
    operation = attributes.get(_OPERATION)
    if isinstance(operation, str) and operation:
        return operation
    if attributes.get(_TOOL_NAME) is not None:
        return _EXECUTE_TOOL
    if _model(attributes) is not None:
        return "chat"
    return None


def _tool_steps(span: object, attributes: Mapping[str, Any]) -> list[Step]:
    name = _tool_name(span, attributes)
    meta = _step_meta(span, attributes)
    status = StepStatus.ERROR if _failed(span, attributes) else StepStatus.OK
    return [
        Step.tool_call(name, _tool_arguments(attributes), meta=meta),
        Step.tool_result(name, status=status, meta=meta),
    ]


def _llm_step(span: object, attributes: Mapping[str, Any]) -> Step:
    return Step.llm(
        name=_model(attributes),
        status=StepStatus.ERROR if _failed(span, attributes) else StepStatus.OK,
        tokens=_total_tokens(attributes),
        meta=_step_meta(span, attributes),
    )


def _tool_name(span: object, attributes: Mapping[str, Any]) -> str:
    name = attributes.get(_TOOL_NAME)
    if name is not None:
        return str(name)
    span_name = str(_field(span, "name") or "")
    return span_name.removeprefix(f"{_EXECUTE_TOOL} ").strip() or "unknown"


def _tool_arguments(attributes: Mapping[str, Any]) -> Any:
    """Return the call arguments, parsing JSON text so key order does not matter."""
    arguments = attributes.get(_TOOL_ARGUMENTS)
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except ValueError:
        return arguments


def _model(attributes: Mapping[str, Any]) -> str | None:
    model = attributes.get(_REQUEST_MODEL) or attributes.get(_RESPONSE_MODEL)
    return None if model is None else str(model)


def _total_tokens(attributes: Mapping[str, Any]) -> int | None:
    counts = [_first_int(attributes, keys) for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)]
    known = [count for count in counts if count is not None]
    return sum(known) if known else None


def _first_int(attributes: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _as_int(attributes.get(key))
        if value is not None:
            return value
    return None


def _failed(span: object, attributes: Mapping[str, Any]) -> bool:
    return _status_is_error(span) or attributes.get(_ERROR_TYPE) is not None


def _step_meta(span: object, attributes: Mapping[str, Any]) -> dict[str, Any]:
    provider = next((attributes[k] for k in _PROVIDER_KEYS if attributes.get(k)), None)
    meta = {
        "provider": provider,
        "span_id": _hex_id(_span_id_raw(span), _SPAN_ID_WIDTH),
        "error_type": attributes.get(_ERROR_TYPE),
    }
    return {key: value for key, value in meta.items() if value is not None}


# ---- grouping --------------------------------------------------------------------------------


def _check_group_by(group_by: str) -> None:
    if group_by not in _GROUPINGS:
        allowed = ", ".join(repr(g) for g in _GROUPINGS)
        raise ValueError(f"group_by must be one of: {allowed} (got {group_by!r})")


def _run_id(
    span: object,
    attributes: Mapping[str, Any],
    group_by: GroupBy,
    conversations: Mapping[str, str],
) -> str | None:
    trace = _trace_id(span)
    if group_by == "conversation":
        conversation = _conversation_id(attributes)
        if conversation is None and trace is not None:
            conversation = conversations.get(trace)
        if conversation is not None:
            return conversation
    return trace


def _conversations_by_trace(parsed: Iterable[tuple[object, Mapping[str, Any]]]) -> dict[str, str]:
    """Map each trace id to the first conversation id seen on one of its spans."""
    found: dict[str, str] = {}
    for span, attributes in parsed:
        conversation = _conversation_id(attributes)
        trace = _trace_id(span)
        if conversation is not None and trace is not None:
            found.setdefault(trace, conversation)
    return found


def _conversation_id(attributes: Mapping[str, Any]) -> str | None:
    conversation = attributes.get(_CONVERSATION_ID)
    return None if conversation in (None, "") else str(conversation)


def _remember(store: OrderedDict[str, Any], key: str, value: Any, limit: int) -> None:
    """Insert or refresh ``key`` as most recent, evicting the oldest entries past ``limit``."""
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def _sort_key(span: object) -> tuple[bool, int]:
    # Order by completion time, as the live processor sees spans (on_end); fall back to
    # the start time. Offline replay must match live order, or calibration stops
    # describing production.
    moment = _time_ns(span, "endTimeUnixNano", "end_time")
    if moment is None:
        moment = _time_ns(span, "startTimeUnixNano", "start_time")
    return (moment is None, moment or 0)


# ---- duck-typed span accessors (SDK ReadableSpan or OTLP-JSON dict) --------------------------


def _field(span: object, name: str) -> Any:
    """Read a top-level field from a span object or dict."""
    if isinstance(span, Mapping):
        return span.get(name)
    return getattr(span, name, None)


def _attributes(span: object) -> Mapping[str, Any]:
    raw = _field(span, "attributes")
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {
            str(item["key"]): _any_value(item.get("value"))
            for item in raw
            if isinstance(item, Mapping) and "key" in item
        }
    return {}


def _any_value(value: Any) -> Any:
    """Decode an OTLP-JSON ``AnyValue`` (``{"stringValue": "x"}`` -> ``"x"``)."""
    if not isinstance(value, Mapping):
        return value
    if "intValue" in value:
        return _as_int(value["intValue"])
    for key in ("stringValue", "boolValue", "doubleValue", "bytesValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        return [_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            str(item["key"]): _any_value(item.get("value"))
            for item in value["kvlistValue"].get("values", [])
        }
    return None


def _status_is_error(span: object) -> bool:
    status = _field(span, "status")
    if status is None:
        return False
    if isinstance(status, Mapping):
        code = status.get("code", status.get("status_code"))
    else:
        code = getattr(status, "status_code", None)
    return _is_error_code(code)


def _is_error_code(code: Any) -> bool:
    """Recognise ERROR as an enum member, a number (2) or a name ("STATUS_CODE_ERROR")."""
    code = getattr(code, "name", code)
    if isinstance(code, bool):
        return False
    if isinstance(code, int):
        return code == _STATUS_CODE_ERROR
    if isinstance(code, str):
        return code.upper().removeprefix("STATUS_CODE_") == "ERROR"
    return False


def _trace_id(span: object) -> str | None:
    if isinstance(span, Mapping):
        raw = span.get("traceId") or span.get("trace_id") or _context_field(span, "trace_id")
    else:
        raw = getattr(getattr(span, "context", None), "trace_id", None)
    return _hex_id(raw, _TRACE_ID_WIDTH)


def _span_id_raw(span: object) -> Any:
    if isinstance(span, Mapping):
        return span.get("spanId") or span.get("span_id") or _context_field(span, "span_id")
    return getattr(getattr(span, "context", None), "span_id", None)


def _context_field(span: Mapping[str, Any], name: str) -> Any:
    context = span.get("context")
    return context.get(name) if isinstance(context, Mapping) else None


def _hex_id(raw: Any, width: int) -> str | None:
    """Normalise an id to lowercase hex of the given width; other strings pass through."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return format(raw, f"0{width}x")
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.lower().removeprefix("0x")
    if text and all(char in string.hexdigits for char in text):
        return text.zfill(width)
    return raw  # e.g. base64 ids from a protobuf-to-JSON export


def _time_ns(span: object, json_key: str, sdk_key: str) -> int | None:
    raw = _field(span, json_key)
    if raw is None:
        raw = _field(span, sdk_key)
    if isinstance(raw, str) and not raw.strip().isdigit():
        return _iso_to_ns(raw)
    return _as_int(raw)


def _iso_to_ns(text: str) -> int | None:
    """Parse an ISO-8601 timestamp (as in the SDK's ``span.to_json()``) to Unix nanoseconds."""
    try:
        moment = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    elapsed = moment - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000_000 + elapsed.microseconds * 1000


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None
