# tests/test_replay.py
from __future__ import annotations
import pytest
from loom.replay import ReplayExecutor, ReplayExhaustedError, ReplayModelClient
from loom.storage import Storage
from loom.types import Observation, Step


def make_storage_with_run(status="complete"):
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    storage.append_step(run_id, 0, Step("read_file", {"path": "a.py"}), Observation("hi"), "t0", "t1")
    storage.append_step(
        run_id, 1, Step("write_file", {"path": "b.py", "content": "x"}), Observation("ok"), "t2", "t3"
    )
    if status == "complete":
        storage.mark_complete(run_id)
    return storage, run_id


def test_replay_model_client_returns_recorded_decisions_in_order():
    storage, run_id = make_storage_with_run()
    client = ReplayModelClient(storage, run_id)
    d0 = client.decide("task", [], [])
    assert d0.tool_name == "read_file"
    assert d0.args == {"path": "a.py"}
    d1 = client.decide("task", [], [])
    assert d1.tool_name == "write_file"


def test_replay_model_client_returns_none_when_complete_and_exhausted():
    storage, run_id = make_storage_with_run(status="complete")
    client = ReplayModelClient(storage, run_id)
    client.decide("task", [], [])
    client.decide("task", [], [])
    d2 = client.decide("task", [], [])
    assert d2.tool_name is None


def test_replay_model_client_raises_when_exhausted_and_not_complete():
    storage, run_id = make_storage_with_run(status="recording")
    client = ReplayModelClient(storage, run_id)
    client.decide("task", [], [])
    client.decide("task", [], [])
    with pytest.raises(ReplayExhaustedError):
        client.decide("task", [], [])


def test_replay_executor_returns_recorded_observations_in_order():
    storage, run_id = make_storage_with_run()
    executor = ReplayExecutor(storage, run_id)
    obs0 = executor.execute(Step("read_file", {"path": "a.py"}))
    assert obs0 == Observation("hi", False)
    obs1 = executor.execute(Step("write_file", {"path": "b.py", "content": "x"}))
    assert obs1 == Observation("ok", False)


def test_replay_executor_raises_when_exhausted():
    storage, run_id = make_storage_with_run()
    executor = ReplayExecutor(storage, run_id)
    executor.execute(Step("read_file", {"path": "a.py"}))
    executor.execute(Step("write_file", {"path": "b.py", "content": "x"}))
    with pytest.raises(ReplayExhaustedError):
        executor.execute(Step("anything", {}))
