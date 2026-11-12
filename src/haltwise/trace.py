"""The trace vocabulary: what an agent did (`Step`) and how it ended (`Run`).

Everything in haltwise reads traces in this one shape. Adapters translate
framework-specific events (OpenTelemetry spans, SWE-agent logs, ...) into it.
Only *observable* behaviour is recorded, never model internals, so any model and any
framework works.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    """What kind of thing happened in one agent step."""

    LLM = "llm"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    USER = "user"
    OTHER = "other"


class StepStatus(str, Enum):
    """Whether the step went well."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


def digest_args(args: Any) -> str:
    """Return a short, stable fingerprint of tool-call arguments.

    Key order does not matter, so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` match.
    The fingerprint lets haltwise spot an agent repeating the exact same call
    without storing the (possibly sensitive) arguments themselves.
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _coerce_enum(enum_type: type[StepKind] | type[StepStatus], value: object, label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{label} must be one of: {allowed} (got {value!r})") from None


@dataclass(frozen=True)
class Step:
    """One observable event in an agent run.

    Build steps with the shortcuts (`Step.llm`, `Step.tool_call`, `Step.tool_result`,
    `Step.error`) or directly. ``kind`` and ``status`` accept plain strings.

    Attributes:
        kind: What happened.
        name: Tool or model name, if any.
        status: Whether it succeeded.
        args_digest: Fingerprint of tool-call arguments (see `digest_args`).
        text: Optional content. Built-in scorers never need it, so you can leave
            it out for privacy.
        tokens: Tokens consumed by this step, if known.
        cost: Money spent on this step, if known.
        meta: Anything else you want to keep alongside the step.

    """

    kind: StepKind
    name: str | None
    status: StepStatus
    args_digest: str | None
    text: str | None
    tokens: int | None
    cost: float | None
    meta: Mapping[str, Any] = field(compare=False)

    def __init__(
        self,
        kind: StepKind | str,
        *,
        name: str | None = None,
        status: StepStatus | str = StepStatus.OK,
        args_digest: str | None = None,
        text: str | None = None,
        tokens: int | None = None,
        cost: float | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        # Frozen dataclass: assign through object.__setattr__ once, at construction.
        assign = object.__setattr__
        assign(self, "kind", _coerce_enum(StepKind, kind, "kind"))
        assign(self, "name", name)
        assign(self, "status", _coerce_enum(StepStatus, status, "status"))
        assign(self, "args_digest", args_digest)
        assign(self, "text", text)
        assign(self, "tokens", tokens)
        assign(self, "cost", cost)
        assign(self, "meta", dict(meta or {}))

    @property
    def failed(self) -> bool:
        """True if the step errored or timed out."""
        return self.status in (StepStatus.ERROR, StepStatus.TIMEOUT)

    @property
    def call_signature(self) -> tuple[str | None, str | None] | None:
        """Identity of a tool call, ``(name, args_digest)``; ``None`` for other steps."""
        if self.kind is not StepKind.TOOL_CALL:
            return None
        return (self.name, self.args_digest)

    # ---- shortcuts -------------------------------------------------------------------

    @classmethod
    def llm(cls, text: str | None = None, *, name: str | None = None, **extra: Any) -> Step:
        """Create a step for a model call or message."""
        return cls(kind=StepKind.LLM, name=name, text=text, **extra)

    @classmethod
    def tool_call(cls, name: str, args: Any = None, **extra: Any) -> Step:
        """Create a step for the agent invoking a tool; ``args`` are fingerprinted, not stored."""
        digest = None if args is None else digest_args(args)
        return cls(kind=StepKind.TOOL_CALL, name=name, args_digest=digest, **extra)

    @classmethod
    def tool_result(cls, name: str | None = None, text: str | None = None, **extra: Any) -> Step:
        """Create a step for what a tool returned."""
        return cls(kind=StepKind.TOOL_RESULT, name=name, text=text, **extra)

    @classmethod
    def error(cls, text: str | None = None, **extra: Any) -> Step:
        """Create a step for something that went wrong (exception, failed command...)."""
        extra.setdefault("status", StepStatus.ERROR)
        return cls(kind=StepKind.ERROR, text=text, **extra)

    # ---- serialisation ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation; ``None`` fields are omitted."""
        data: dict[str, Any] = {
            "kind": self.kind.value,
            "name": self.name,
            "status": self.status.value,
            "args_digest": self.args_digest,
            "text": self.text,
            "tokens": self.tokens,
            "cost": self.cost,
            "meta": dict(self.meta) or None,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Inverse of `to_dict`. Raw ``args`` are accepted and fingerprinted."""
        digest = data.get("args_digest")
        if digest is None and data.get("args") is not None:
            digest = digest_args(data["args"])
        return cls(
            kind=data["kind"],
            name=data.get("name"),
            status=data.get("status", StepStatus.OK),
            args_digest=digest,
            text=data.get("text"),
            tokens=data.get("tokens"),
            cost=data.get("cost"),
            meta=data.get("meta"),
        )


_TRUE_WORDS = {"true", "1"}
_FALSE_WORDS = {"false", "0"}


def _outcome(value: object) -> bool | None:
    """Normalise an outcome label: bools, numpy scalars, 0/1, "true"/"false", "1"/"0".

    Anything else is rejected rather than guessed, because a misread label silently
    corrupts both training and the calibration guarantee.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in _TRUE_WORDS | _FALSE_WORDS:
        return value.strip().lower() in _TRUE_WORDS
    if getattr(value, "shape", ()) != ():  # arrays are not labels, even of size one
        raise ValueError(f"succeeded must be a single value, not an array (got {value!r})")
    try:
        if value in (0, 1):  # also covers numpy bools and 0/1 integers
            return bool(value == 1)
    except (TypeError, ValueError):
        pass
    raise ValueError(f"succeeded must be True, False, 1, 0 or None (got {value!r})")


@dataclass(frozen=True)
class Run:
    """A whole agent run: its steps and, once known, whether it succeeded.

    Attributes:
        steps: The steps, in order.
        succeeded: ``True``/``False`` once the outcome is known, else ``None``.
        id: Your identifier for the run.
        task: Identifier of the task the run attempted. Runs on the same task are
            kept on the same side of every train/calibration split, which avoids
            leaking task-specific patterns into the guarantee.

    """

    steps: tuple[Step, ...]
    succeeded: bool | None = None
    id: str | None = None
    task: str | None = None

    def __init__(
        self,
        steps: Iterable[Step],
        succeeded: bool | None = None,
        id: str | None = None,
        task: str | None = None,
    ) -> None:
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(self, "succeeded", _outcome(succeeded))
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "task", task)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def failed(self) -> bool:
        """True only when the run is known to have failed."""
        return self.succeeded is False

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation."""
        data: dict[str, Any] = {"id": self.id, "task": self.task, "succeeded": self.succeeded}
        data = {key: value for key, value in data.items() if value is not None}
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Run:
        """Inverse of `to_dict`."""
        return cls(
            steps=[Step.from_dict(step) for step in data.get("steps", [])],
            succeeded=data.get("succeeded"),
            id=data.get("id"),
            task=data.get("task"),
        )
