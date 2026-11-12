"""Minimal LLM clients for scorers that ask a model questions. Standard library only.

A client is just a function ``ask(prompt) -> reply``. Any provider fits that shape;
`ollama` covers both a local Ollama server and Ollama Cloud.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any, Final

Ask = Callable[[str], str]
"""A function that sends one prompt to a model and returns its reply."""

_LOCAL_HOST: Final = "http://localhost:11434"
_CLOUD_HOST: Final = "https://ollama.com"


class _FromEnvironment:
    def __repr__(self) -> str:
        return "<read OLLAMA_API_KEY>"


FROM_ENVIRONMENT: Final = _FromEnvironment()


def ollama(
    model: str,
    *,
    host: str | None = None,
    api_key: str | _FromEnvironment | None = FROM_ENVIRONMENT,
    timeout: float = 120.0,
) -> Ask:
    """Return an ``ask(prompt)`` function backed by Ollama's chat API.

    Replies are deterministic (temperature 0), so calibration and live use agree. An
    empty reply is retried once.

    Args:
        model: Model name, e.g. ``"gpt-oss:20b"``.
        host: Server URL. Defaults to ``$OLLAMA_HOST``, else Ollama Cloud when an API
            key is available, else a local server.
        api_key: Bearer token. Read from ``$OLLAMA_API_KEY`` by default; pass ``None``
            to send no key (a local server needs none). It is never logged.
        timeout: Seconds to wait for each reply.
    """
    key = os.environ.get("OLLAMA_API_KEY") if isinstance(api_key, _FromEnvironment) else api_key
    url = host or os.environ.get("OLLAMA_HOST") or (_CLOUD_HOST if key else _LOCAL_HOST)
    endpoint = url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def chat(prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply: dict[str, Any] = json.loads(response.read())
        return str(reply["message"]["content"])

    def ask(prompt: str) -> str:
        # Reasoning models occasionally spend the whole reply on hidden "thinking" and
        # return empty content (about 4% of calls with gpt-oss:20b); one retry fixes it.
        return chat(prompt) or chat(prompt)

    return ask
