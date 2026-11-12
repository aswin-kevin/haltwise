from haltwise import Run, Step
from trials.ollama_agent.agent import Attempt, parse_action, summarise_for_retry


class TestParseAction:
    def test_reads_a_plain_json_action(self) -> None:
        assert parse_action('{"tool": "get_employee", "args": {"id": "E1"}}') == (
            "get_employee",
            {"id": "E1"},
        )

    def test_finds_the_action_inside_surrounding_text(self) -> None:
        reply = 'Let me look.\n```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```'

        assert parse_action(reply) == ("calculate", {"expression": "1+1"})

    def test_missing_args_become_empty(self) -> None:
        assert parse_action('{"tool": "submit"}') == ("submit", {})

    def test_text_without_an_action_is_none(self) -> None:
        assert parse_action("I think the answer is 42") is None
        assert parse_action('{"answer": 42}') is None


def test_retry_note_lists_the_stopped_attempts_last_actions() -> None:
    steps = [
        Step.tool_call("search_employees"),
        Step.tool_result("search_employees", status="error"),
    ]
    attempt = Attempt(1, Run(steps), solved=False, tokens=10)

    note = summarise_for_retry(attempt)

    assert "search_employees(error)" in note
    assert "different" in note
