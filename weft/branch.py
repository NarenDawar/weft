from __future__ import annotations

from dataclasses import dataclass

from weft.model import ModelClient
from weft.recording import RecordingExecutor, RecordingModelClient
from weft.registry import ToolRegistry
from weft.replay import ReplayExecutor, ReplayModelClient
from weft.storage import Storage
from weft.types import Decision, Observation, Step


class InvalidForkPointError(Exception):
    pass


@dataclass
class BranchModelClient:
    replay: ReplayModelClient
    recording: RecordingModelClient
    fork_step: int

    def decide(self, task: str, history, tools) -> Decision:
        if self.replay._next_index < self.fork_step:
            return self.replay.decide(task, history, tools)
        return self.recording.decide(task, history, tools)


@dataclass
class BranchExecutor:
    replay: ReplayExecutor
    recording: RecordingExecutor
    fork_step: int

    def execute(self, step: Step) -> Observation:
        if self.replay._next_index < self.fork_step:
            return self.replay.execute(step)
        return self.recording.execute(step)


def branch(
    storage: Storage,
    parent_run_id: str,
    from_step: int,
    real_client: ModelClient,
    registry: ToolRegistry,
) -> tuple[str, BranchModelClient, BranchExecutor]:
    parent = storage.get_run(parent_run_id)
    parent_history = storage.resolve_full_history(parent_run_id)
    if from_step < 0 or from_step > len(parent_history):
        raise InvalidForkPointError(
            f"from_step={from_step} is out of range for run {parent_run_id!r} "
            f"({len(parent_history)} recorded steps)"
        )
    child_run_id = storage.create_run(
        parent.task, forked_from_run_id=parent_run_id, forked_from_step=from_step
    )
    replay_model = ReplayModelClient(storage, parent_run_id)
    replay_executor = ReplayExecutor(storage, parent_run_id)
    recording_model = RecordingModelClient(real_client, storage, child_run_id)
    recording_executor = RecordingExecutor(registry, storage, child_run_id, start_index=from_step)
    model_client = BranchModelClient(replay_model, recording_model, from_step)
    executor = BranchExecutor(replay_executor, recording_executor, from_step)
    return child_run_id, model_client, executor
