"""Loader for ``SWE-bench/SWE-smith-trajectories`` (Hugging Face, MIT), "tool" split.

SWE-agent runs by Claude 3.5/3.7 Sonnet and GPT-4o on synthetic bugs, labelled
``resolved``. Unlike `haltwise.datasets.swe_agent`, actions are OpenAI-style tool calls
(``bash``, ``str_replace_editor``, ``submit``), which makes this a second, quite
different test of haltwise's trace-only approach.

Only ``messages`` is used; ``patch`` reveals the outcome and is never read.

Dataset: https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories
"""

from __future__ import annotations

import ast
import json
import re
import urllib.request
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from haltwise.trace import Run, Step, StepStatus

DATASET_URL = "https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories"
SHARD_URL = (
    "https://huggingface.co/api/datasets/SWE-bench/SWE-smith-trajectories/parquet/default/tool/"
    "{shard}.parquet"
)
SHARD_COUNT = 8

ERROR_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "No replacement was performed",
    "did not appear verbatim",
    "No such file or directory",
    "command not found",
    "does not exist",
)

_COMMAND_SEPARATORS = re.compile(r"&&|;|\|\|")


def _decode(value: Any) -> Any:
    """Decode a field stored as JSON or as a Python literal; pass objects through."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _sub_command(arguments: Mapping[str, Any]) -> str | None:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    last_segment = _COMMAND_SEPARATORS.split(command)[-1].split()
    return last_segment[0] if last_segment else None


def _tool_name(function: Mapping[str, Any]) -> tuple[str, Any]:
    arguments = _decode(function.get("arguments") or "{}")
    name = str(function.get("name"))
    sub = _sub_command(arguments) if isinstance(arguments, Mapping) else None
    return (f"{name}:{sub}" if sub else name), arguments


def _content_text(content: Any) -> str:
    """Flatten message content (plain text or a list of text parts) to a string."""
    if isinstance(content, str) and content.startswith("[{"):
        try:
            content = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return str(content)
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def parse_swe_smith_record(
    record: Mapping[str, Any], *, run_id: str | None = None, keep_text: bool = False
) -> Run:
    """Convert one dataset row into a `Run`.

    Each tool call becomes a step named ``tool:sub-command`` (``bash:pytest``,
    ``str_replace_editor:view``...), fingerprinted by its full arguments. Each tool
    message becomes the matching result, marked as an error when it contains one of
    `ERROR_MARKERS`. Assistant messages without tool calls become `Step.llm`.
    System and user messages are skipped.
    """
    steps: list[Step] = []
    pending: deque[str] = deque()  # names of calls still waiting for their result
    for message in _decode(record["messages"]):
        role = message.get("role")
        if role == "assistant":
            calls = _decode(message.get("tool_calls")) or []
            if not calls:
                steps.append(Step.llm(_content_text(message.get("content")) if keep_text else None))
            for call in calls:
                name, arguments = _tool_name(call.get("function", {}))
                pending.append(name)
                steps.append(Step.tool_call(name, arguments))
        elif role == "tool":
            text = _content_text(message.get("content"))
            status = StepStatus.ERROR if any(m in text for m in ERROR_MARKERS) else StepStatus.OK
            tool = pending.popleft() if pending else None
            steps.append(Step.tool_result(tool, text if keep_text else None, status=status))

    return Run(steps, succeeded=bool(record["resolved"]), id=run_id, task=record.get("instance_id"))


def load_swe_smith(
    paths: str | Path | Sequence[str | Path],
    *,
    limit: int | None = None,
    models: Iterable[str] | None = None,
    keep_text: bool = False,
) -> list[Run]:
    """Load runs from downloaded parquet shards (needs ``haltwise[datasets]``).

    Args:
        paths: Shard file(s), e.g. from `download_swe_smith_shard`.
        limit: Stop after this many runs.
        models: Only keep runs from these ``model`` values, e.g.
            ``["claude-3-7-sonnet-20250219"]``. Useful for testing transfer.
        keep_text: Keep message text on the steps.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        raise ImportError(
            "Reading SWE-smith shards needs pyarrow: pip install 'haltwise[datasets]'"
        ) from error

    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    wanted = set(models) if models is not None else None
    runs: list[Run] = []
    for file in files:
        table = pq.read_table(file, columns=["messages", "instance_id", "resolved", "model"])
        for row, record in enumerate(table.to_pylist()):
            if wanted is not None and record["model"] not in wanted:
                continue
            runs.append(
                parse_swe_smith_record(
                    record, run_id=f"{Path(file).stem}:{row}", keep_text=keep_text
                )
            )
            if limit is not None and len(runs) >= limit:
                return runs
    return runs


def download_swe_smith_shard(shard: int = 1, directory: str | Path = "data") -> Path:
    """Download one parquet shard (~100-250 MB) unless it is already there."""
    if not 0 <= shard < SHARD_COUNT:
        raise ValueError(f"shard must be between 0 and {SHARD_COUNT - 1} (got {shard})")
    target = Path(directory) / f"swe_smith_tool_{shard}.parquet"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(SHARD_URL.format(shard=shard), partial)
        partial.rename(target)
    return target
