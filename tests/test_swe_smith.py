import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from haltwise import Step, StepKind, StepStatus
from haltwise.datasets.swe_smith import load_swe_smith, parse_swe_smith_record


def tool_calls(*calls: tuple[str, dict[str, str]]) -> str:
    """SWE-smith stores tool calls as a Python repr string."""
    return repr(
        [
            {"index": i, "function": {"name": name, "arguments": json.dumps(args)}, "id": f"c{i}"}
            for i, (name, args) in enumerate(calls)
        ]
    )


def record(messages: Sequence[Mapping[str, object]], resolved: bool = True) -> dict[str, object]:
    return {
        "messages": json.dumps(messages),
        "instance_id": "repo__bug-7",
        "resolved": resolved,
        "model": "claude-3-7-sonnet-20250219",
        "traj_id": "t1",
        "patch": "diff --git ...",
    }


CONVERSATION: list[dict[str, object]] = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Fix the bug in widgets.py"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": tool_calls(("str_replace_editor", {"command": "view", "path": "/a.py"})),
    },
    {"role": "tool", "content": "OBSERVATION:\n     1\timport os"},
    {
        "role": "assistant",
        "content": "Run it.",
        "tool_calls": tool_calls(("bash", {"command": "python reproduce.py"})),
    },
    {"role": "tool", "content": "Traceback (most recent call last):\n  File ..."},
]


def test_tool_calls_and_results_become_steps() -> None:
    run = parse_swe_smith_record(record(CONVERSATION))

    assert [s.kind for s in run.steps] == [
        StepKind.TOOL_CALL,
        StepKind.TOOL_RESULT,
        StepKind.TOOL_CALL,
        StepKind.TOOL_RESULT,
    ]


def test_tool_name_includes_the_sub_command() -> None:
    run = parse_swe_smith_record(record(CONVERSATION))

    calls = [s.name for s in run.steps if s.kind is StepKind.TOOL_CALL]
    assert calls == ["str_replace_editor:view", "bash:python"]


def test_error_observations_are_marked_as_failed() -> None:
    run = parse_swe_smith_record(record(CONVERSATION))

    assert run.steps[1].status is StepStatus.OK
    assert run.steps[3].status is StepStatus.ERROR
    assert run.steps[3].name == "bash:python"


def test_identical_calls_share_a_digest() -> None:
    call: dict[str, object] = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls(("bash", {"command": "pytest"})),
    }
    first, second = parse_swe_smith_record(record([call, call])).steps

    assert first.args_digest == second.args_digest


def test_assistant_message_without_tool_calls_is_an_llm_step() -> None:
    run = parse_swe_smith_record(record([{"role": "assistant", "content": "Thinking..."}]))

    assert run.steps == (Step.llm(),)


def test_outcome_and_task_come_from_the_record() -> None:
    run = parse_swe_smith_record(record(CONVERSATION, resolved=False), run_id="x")

    assert run.succeeded is False
    assert run.task == "repo__bug-7"
    assert run.id == "x"


def test_text_is_kept_only_on_request() -> None:
    without = parse_swe_smith_record(record(CONVERSATION))
    with_text = parse_swe_smith_record(record(CONVERSATION), keep_text=True)

    assert all(s.text is None for s in without.steps)
    assert with_text.steps[3].text is not None
    assert "Traceback" in with_text.steps[3].text


def test_accepts_already_decoded_messages_and_tool_calls() -> None:
    decoded = [dict(m) for m in CONVERSATION]
    for message in decoded:
        calls = message.get("tool_calls")
        if isinstance(calls, str):
            message["tool_calls"] = ast.literal_eval(calls)
    raw = record(CONVERSATION)
    as_objects = {**raw, "messages": decoded}

    assert parse_swe_smith_record(as_objects).steps == parse_swe_smith_record(raw).steps


def test_loader_can_keep_only_some_models(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    rows = [
        {**record(CONVERSATION), "model": "claude-3-7-sonnet-20250219"},
        {**record(CONVERSATION), "model": "gpt-4o-2024-08-06"},
    ]
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    runs = load_swe_smith(path, models=["gpt-4o-2024-08-06"])

    assert len(runs) == 1
    assert runs[0].id == "shard:1"
