from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Step:
    tool_name: str
    args: dict


@dataclass(frozen=True)
class Observation:
    result: Any
    is_error: bool = False


@dataclass(frozen=True)
class HistoryEntry:
    step: Step
    observation: Observation


@dataclass(frozen=True)
class Decision:
    tool_name: Optional[str]
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
