import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest

from haltwise import Step
from haltwise.errors import HaltwiseError
from haltwise.llm import ollama
from haltwise.scorers import SelfReportScorer


class ScriptedModel:
    """Stands in for an LLM: returns canned answers and remembers the prompts."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "50"


class TestSelfReportScorer:
    def test_asks_every_n_steps_and_holds_the_answer_in_between(self) -> None:
        model = ScriptedModel("80", "20")
        scorer = SelfReportScorer(model, every=2)

        risks = scorer.risk_path([Step.llm(f"s{i}") for i in range(5)])

        assert len(model.prompts) == 2
        assert risks == [0.0, 0.8, 0.8, 0.2, 0.2]

    def test_prompt_describes_the_recent_steps(self) -> None:
        model = ScriptedModel("10")
        SelfReportScorer(model, every=2).risk_path(
            [Step.tool_call("run_tests"), Step.error("AssertionError in test_x")]
        )

        assert "run_tests" in model.prompts[0]
        assert "AssertionError in test_x" in model.prompts[0]

    @pytest.mark.parametrize(
        ("answer", "risk"),
        [
            ("Probability: 73%", 0.73),
            ("90", 0.9),
            ("On a 0-100 scale: 85", 0.85),
            ("0.85", 0.85),
            ("7/10", 0.7),
            ("250", 1.0),
            ("85 out of 100", 0.85),
            ("Risk: 85 (scale 0-100)", 0.85),
            ("Step 3/10 failed; risk 90", 0.9),
            ("0.5%", 0.005),
        ],
    )
    def test_reads_the_answer_as_a_failure_probability(self, answer: str, risk: float) -> None:
        scorer = SelfReportScorer(ScriptedModel(answer), every=1)

        assert scorer.risk_path([Step.llm()]) == [pytest.approx(risk)]

    def test_keeps_the_previous_risk_when_the_answer_has_no_number(self) -> None:
        scorer = SelfReportScorer(ScriptedModel("60", "no idea"), every=1)

        assert scorer.risk_path([Step.llm(), Step.llm()]) == [0.6, 0.6]
        assert scorer.unreadable_answers == 1

    def test_cannot_be_saved_because_the_model_is_code(self) -> None:
        with pytest.raises(HaltwiseError, match="rebuild"):
            SelfReportScorer(ScriptedModel()).to_dict()

    def test_every_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="every"):
            SelfReportScorer(ScriptedModel(), every=0)


class _FakeOllama(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[dict[str, Any], str | None]]] = []
    replies: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.requests.append((body, self.headers.get("Authorization")))
        content = _FakeOllama.replies.pop(0) if _FakeOllama.replies else "42"
        reply = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture
def ollama_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeOllama.requests.clear()
    _FakeOllama.replies.clear()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestOllamaClient:
    def test_sends_a_deterministic_chat_request_and_returns_the_reply(
        self, ollama_server: str
    ) -> None:
        ask = ollama("gpt-oss:20b", host=ollama_server, api_key="secret")

        assert ask("hello") == "42"
        body, auth = _FakeOllama.requests[0]
        assert body["model"] == "gpt-oss:20b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        assert auth == "Bearer secret"

    def test_local_server_needs_no_key(self, ollama_server: str) -> None:
        ollama("llama3", host=ollama_server, api_key=None)("hi")

        assert _FakeOllama.requests[0][1] is None

    def test_reads_the_key_from_the_environment(
        self, ollama_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_API_KEY", "from-env")

        ollama("llama3", host=ollama_server)("hi")

        assert _FakeOllama.requests[0][1] == "Bearer from-env"

    def test_retries_once_when_a_reasoning_model_returns_an_empty_reply(
        self, ollama_server: str
    ) -> None:
        _FakeOllama.replies.extend(["", "57"])

        assert ollama("gpt-oss:20b", host=ollama_server, api_key=None)("hi") == "57"
        assert len(_FakeOllama.requests) == 2
