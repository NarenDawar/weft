from __future__ import annotations
from loom.recording import RecordingExecutor, RecordingModelClient, begin_recording
from loom.registry import ToolRegistry
from loom.storage import Storage
from loom.types import Decision, Step, Tool
from tests.fakes import FakeModelClient


def boom(**_kwargs):
    raise ValueError("tool broke")


def make_registry():
    read = Tool(
        name="read_file",
        description="",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        fn=lambda path: f"contents of {path}",
    )
    broken = Tool(name="broken", description="", parameters={"type": "object", "properties": {}, "required": []}, fn=boom)
    return ToolRegistry([read, broken])


def test_recording_model_client_forwards_and_returns_real_decision():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    real_client = FakeModelClient([Decision("read_file", {"path": "a.py"})])
    client = RecordingModelClient(real_client, storage, run_id)
    decision = client.decide("task", [], [])
    assert decision.tool_name == "read_file"
    assert real_client.calls == 1


def test_recording_model_client_marks_run_complete_on_none_decision():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    real_client = FakeModelClient([Decision(None, {})])
    client = RecordingModelClient(real_client, storage, run_id)
    client.decide("task", [], [])
    run = storage.get_run(run_id)
    assert run.status == "complete"


def test_recording_executor_logs_step_and_returns_observation():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    executor = RecordingExecutor(make_registry(), storage, run_id)
    obs = executor.execute(Step("read_file", {"path": "a.py"}))
    assert obs.result == "contents of a.py"
    logged = storage.get_own_steps(run_id)
    assert len(logged) == 1
    assert logged[0].step == Step("read_file", {"path": "a.py"})


def test_recording_executor_captures_tool_exception_as_error_observation():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    executor = RecordingExecutor(make_registry(), storage, run_id)
    obs = executor.execute(Step("broken", {}))
    assert obs.is_error is True
    assert "tool broke" in obs.result
    logged = storage.get_own_steps(run_id)
    assert logged[0].observation.is_error is True


def test_recording_executor_captures_unknown_tool_as_error_observation():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    executor = RecordingExecutor(make_registry(), storage, run_id)
    obs = executor.execute(Step("nonexistent_tool", {}))
    assert obs.is_error is True
    assert "nonexistent_tool" in obs.result
    logged = storage.get_own_steps(run_id)
    assert len(logged) == 1
    assert logged[0].observation.is_error is True
    assert logged[0].step == Step("nonexistent_tool", {})


def test_recording_executor_respects_start_index_for_branching():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    executor = RecordingExecutor(make_registry(), storage, run_id, start_index=3)
    executor.execute(Step("read_file", {"path": "a.py"}))
    row = storage.conn.execute("SELECT step_index FROM steps WHERE run_id = ?", (run_id,)).fetchone()
    assert row["step_index"] == 3


def test_begin_recording_creates_run_and_returns_wired_clients():
    storage = Storage(":memory:")
    real_client = FakeModelClient([Decision(None, {})])
    run_id, model_client, executor = begin_recording(storage, "do it", real_client, make_registry())
    run = storage.get_run(run_id)
    assert run.task == "do it"
    assert run.status == "recording"
    decision = model_client.decide("do it", [], [])
    assert decision.tool_name is None
