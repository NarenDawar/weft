from __future__ import annotations

from loom.model import ModelClient
from loom.registry import ToolRegistry
from loom.storage import Storage, now_iso
from loom.types import Decision, HistoryEntry, Observation, Step


class RecordingModelClient:
    def __init__(self, real_client: ModelClient, storage: Storage, run_id: str):
        self.real_client = real_client
        self.storage = storage
        self.run_id = run_id

    def decide(self, task: str, history: list[HistoryEntry], tools) -> Decision:
        decision = self.real_client.decide(task, history, tools)
        if decision.tool_name is None:
            self.storage.mark_complete(self.run_id)
        return decision


class RecordingExecutor:
    def __init__(self, registry: ToolRegistry, storage: Storage, run_id: str, start_index: int = 0):
        self.registry = registry
        self.storage = storage
        self.run_id = run_id
        self._next_index = start_index

    def execute(self, step: Step) -> Observation:
        decided_at = now_iso()
        tool = self.registry.get(step.tool_name)
        try:
            result = tool.fn(**step.args)
            observation = Observation(result=result, is_error=False)
        except Exception as exc:
            observation = Observation(result=str(exc), is_error=True)
        executed_at = now_iso()
        self.storage.append_step(self.run_id, self._next_index, step, observation, decided_at, executed_at)
        self._next_index += 1
        return observation


def begin_recording(
    storage: Storage, task: str, real_client: ModelClient, registry: ToolRegistry
) -> tuple[str, RecordingModelClient, RecordingExecutor]:
    run_id = storage.create_run(task)
    model_client = RecordingModelClient(real_client, storage, run_id)
    executor = RecordingExecutor(registry, storage, run_id)
    return run_id, model_client, executor
