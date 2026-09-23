from __future__ import annotations

from loom.storage import Storage
from loom.types import Decision, Observation, Step


class ReplayExhaustedError(Exception):
    pass


class ReplayModelClient:
    def __init__(self, storage: Storage, run_id: str):
        self.storage = storage
        self.run_id = run_id
        self._steps = storage.resolve_full_history(run_id)
        self._next_index = 0

    def decide(self, task: str, history, tools) -> Decision:
        if self._next_index >= len(self._steps):
            run = self.storage.get_run(self.run_id)
            if run.status == "complete":
                return Decision(tool_name=None, args={})
            raise ReplayExhaustedError(
                f"run {self.run_id!r} has no recorded step at index {self._next_index}"
            )
        entry = self._steps[self._next_index]
        self._next_index += 1
        return Decision(tool_name=entry.step.tool_name, args=entry.step.args)


class ReplayExecutor:
    def __init__(self, storage: Storage, run_id: str):
        self.storage = storage
        self.run_id = run_id
        self._steps = storage.resolve_full_history(run_id)
        self._next_index = 0

    def execute(self, step: Step) -> Observation:
        if self._next_index >= len(self._steps):
            raise ReplayExhaustedError(
                f"run {self.run_id!r} has no recorded step at index {self._next_index}"
            )
        entry = self._steps[self._next_index]
        self._next_index += 1
        return entry.observation
