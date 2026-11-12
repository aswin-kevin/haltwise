import json

from haltwise import Step, StepKind, StepStatus
from haltwise.datasets.swe_agent import parse_swe_agent_record


def record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "instance_id": "org__repo-1",
        "model_name": "swe-agent-llama-70b",
        "target": True,
        "exit_status": "submitted",
        "trajectory": [
            {"role": "system", "text": "You are an agent"},
            {"role": "user", "text": "ISSUE: something breaks"},
            {"role": "ai", "text": 'Let\'s look.\n\n```\nfind_file "memset.py" src\n```'},
            {"role": "user", "text": "Found 1 matches for memset.py"},
            {"role": "ai", "text": "Edit it.\n```\nedit 10:12\nprint('x'\nend_of_edit\n```"},
            {"role": "user", "text": "Your proposed edit has introduced new syntax error(s)."},
        ],
    }
    base.update(overrides)
    return base


def test_turns_become_tool_calls_and_results() -> None:
    run = parse_swe_agent_record(record())

    kinds = [step.kind for step in run.steps]
    assert kinds == [
        StepKind.TOOL_CALL,
        StepKind.TOOL_RESULT,
        StepKind.TOOL_CALL,
        StepKind.TOOL_RESULT,
    ]


def test_tool_name_is_the_command_word() -> None:
    run = parse_swe_agent_record(record())

    assert [s.name for s in run.steps if s.kind is StepKind.TOOL_CALL] == ["find_file", "edit"]


def test_identical_commands_share_a_digest() -> None:
    command = "```\nopen foo.py\n```"
    trajectory = [{"role": "ai", "text": command}, {"role": "ai", "text": "again\n" + command}]

    first, second = parse_swe_agent_record(record(trajectory=trajectory)).steps

    assert first.args_digest == second.args_digest


def test_error_observations_are_marked_as_failed() -> None:
    run = parse_swe_agent_record(record())

    assert run.steps[-1].status is StepStatus.ERROR
    assert run.steps[1].status is StepStatus.OK


def test_outcome_and_identity_come_from_the_record() -> None:
    run = parse_swe_agent_record(record(target=False))

    assert run.succeeded is False
    assert run.task == "org__repo-1"


def test_outcome_revealing_fields_are_never_used() -> None:
    honest = parse_swe_agent_record(record(exit_status="submitted"))
    leaky = parse_swe_agent_record(record(exit_status="early_exit", generated_patch=None))

    assert honest.steps == leaky.steps


def test_accepts_trajectory_encoded_as_json_string() -> None:
    raw = record()
    as_string = record(trajectory=json.dumps(raw["trajectory"]))

    assert parse_swe_agent_record(as_string).steps == parse_swe_agent_record(raw).steps


def test_message_text_is_not_kept_by_default() -> None:
    assert all(step.text is None for step in parse_swe_agent_record(record()).steps)


def test_ai_turn_without_a_command_is_a_plain_llm_step() -> None:
    run = parse_swe_agent_record(record(trajectory=[{"role": "ai", "text": "just thinking"}]))

    assert run.steps == (Step.llm(),)


def test_missing_label_stays_unknown() -> None:
    assert parse_swe_agent_record(record(target=None)).succeeded is None
