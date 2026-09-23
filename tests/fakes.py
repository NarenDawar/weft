from __future__ import annotations

from loom.types import Decision


class FakeModelClient:
    def __init__(self, decisions: list[Decision]):
        self._decisions = list(decisions)
        self.calls = 0

    def decide(self, task, history, tools) -> Decision:
        self.calls += 1
        return self._decisions.pop(0)


class FailingModelClient:
    def __init__(self, error: Exception):
        self._error = error
        self.calls = 0

    def decide(self, task, history, tools):
        self.calls += 1
        raise self._error
