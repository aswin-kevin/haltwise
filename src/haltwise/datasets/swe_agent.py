"""Loader for ``nebius/SWE-agent-trajectories`` (Hugging Face, CC-BY-4.0).

About 80k SWE-agent runs on real GitHub issues, each labelled with whether the
generated patch resolved the issue (``target``). In each run, ``ai`` turns hold a
thought followed by one fenced shell command, and ``user`` turns hold what that
command printed.

Only the trajectory is used. ``exit_status``, ``generated_patch`` and ``eval_logs``
reveal how the run ended, so they are never read: using them would leak the answer
into the features.

Dataset: https://huggingface.co/datasets/nebius/SWE-agent-trajectories
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from haltwise.trace import Run, Step, StepStatus

DATASET_URL = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories"
SHARD_URL = (
    "https://huggingface.co/api/datasets/nebius/SWE-agent-trajectories/parquet/default/train/"
    "{shard}.parquet"
)
SHARD_COUNT = 12

_COMMAND_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Observation fragments that mean the previous command did not do what the agent wanted.
ERROR_MARKERS: tuple[str, ...] = (
    "Your proposed edit has introduced new syntax error",
    "Traceback (most recent call last)",
    "No such file or directory",
    "command not found",
    "No matches found",
    "not found",
    "Usage:",
)


def _command(text: str) -> str | None:
    blocks = _COMMAND_BLOCK.findall(text)
    if not blocks:
        return None
    command = blocks[-1].strip()
    return command or None


def _observation_status(text: str) -> StepStatus:
    return StepStatus.ERROR if any(m in text for m in ERROR_MARKERS) else StepStatus.OK


def _turns(trajectory: Any) -> list[Mapping[str, Any]]:
    # The dataset card documents the trajectory both as a list and as a JSON string.
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    return list(trajectory)


def parse_swe_agent_record(
    record: Mapping[str, Any], *, run_id: str | None = None, keep_text: bool = False
) -> Run:
    """Convert one dataset row into a `Run`.

    Each ``ai`` turn becomes a tool call named after its command word (``edit``,
    ``python``, ``search_dir``...), fingerprinted by the full command. Each ``user``
    turn after the first ``ai`` turn becomes that call's result, marked as an error
    when it contains one of `ERROR_MARKERS`. The opening system prompt and issue text
    are skipped.

    Args:
        record: One row of the dataset.
        run_id: Identifier for the run (the dataset has none of its own).
        keep_text: Keep message text on the steps. Off by default: the built-in scorers
            don't need it and it multiplies memory use.
    """
    steps: list[Step] = []
    last_tool: str | None = None
    for turn in _turns(record["trajectory"]):
        role, text = turn.get("role"), turn.get("text") or ""
        kept = text if keep_text else None
        if role == "ai":
            command = _command(text)
            if command is None:
                steps.append(Step.llm(kept))
                last_tool = None
                continue
            last_tool = command.split()[0]
            steps.append(Step.tool_call(last_tool, command, text=kept))
        elif role == "user" and steps:
            steps.append(Step.tool_result(last_tool, kept, status=_observation_status(text)))

    succeeded = record.get("target")  # Run accepts bools and "True"/"False" strings
    return Run(steps, succeeded=succeeded, id=run_id, task=record.get("instance_id"))


def load_swe_agent(
    paths: str | Path | Sequence[str | Path],
    *,
    limit: int | None = None,
    models: Iterable[str] | None = None,
    keep_text: bool = False,
) -> list[Run]:
    """Load runs from one or more downloaded parquet shards (needs ``haltwise[datasets]``).

    Args:
        paths: Shard file(s), e.g. from `download_swe_agent_shard`.
        limit: Stop after this many runs.
        models: Only keep runs from these ``model_name`` values, e.g.
            ``["swe-agent-llama-70b"]``. Useful for testing transfer between models.
        keep_text: See `parse_swe_agent_record`.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        raise ImportError(
            "Reading SWE-agent shards needs pyarrow: pip install 'haltwise[datasets]'"
        ) from error

    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    wanted = set(models) if models is not None else None
    runs: list[Run] = []
    for file in files:
        table = pq.read_table(file, columns=["instance_id", "model_name", "target", "trajectory"])
        for row, record in enumerate(table.to_pylist()):
            if wanted is not None and record["model_name"] not in wanted:
                continue
            run_id = f"{Path(file).stem}:{row}"
            runs.append(parse_swe_agent_record(record, run_id=run_id, keep_text=keep_text))
            if limit is not None and len(runs) >= limit:
                return runs
    return runs


def download_swe_agent_shard(shard: int = 0, directory: str | Path = "data") -> Path:
    """Download one parquet shard (~90 MB) unless it is already there; return its path."""
    if not 0 <= shard < SHARD_COUNT:
        raise ValueError(f"shard must be between 0 and {SHARD_COUNT - 1} (got {shard})")
    target = Path(directory) / f"swe_agent_{shard}.parquet"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(SHARD_URL.format(shard=shard), partial)
        partial.rename(target)
    return target
