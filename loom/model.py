from __future__ import annotations

from typing import Protocol

from loom.types import Decision, HistoryEntry, Tool


class ModelClient(Protocol):
    def decide(self, task: str, history: list[HistoryEntry], tools: list[Tool]) -> Decision: ...
