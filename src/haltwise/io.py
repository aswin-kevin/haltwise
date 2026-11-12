"""Reading and writing runs as JSON Lines (one run per line)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from haltwise.trace import Run

# Field names used by PrefixGuard (arXiv 2605.06455), so its trace files load directly.
_ALIASES = {"trajectory_id": "id", "task_id": "task", "final_success": "succeeded"}


def _normalise(record: dict[str, Any]) -> dict[str, Any]:
    return {_ALIASES.get(key, key): value for key, value in record.items()}


def load_runs(path: str | Path) -> list[Run]:
    """Load runs from a JSON Lines file written by `save_runs` (or in PrefixGuard's layout).

    Each line is one run: ``{"id": ..., "task": ..., "succeeded": true, "steps": [...]}``.
    See `haltwise.Step.from_dict` for the step format.
    """
    runs = []
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                runs.append(Run.from_dict(_normalise(json.loads(line))))
    return runs


def save_runs(runs: Iterable[Run], path: str | Path) -> None:
    """Write runs to a JSON Lines file, one run per line."""
    with Path(path).open("w", encoding="utf-8") as out:
        for run in runs:
            out.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
