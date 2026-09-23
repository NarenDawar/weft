# tests/test_storage.py
from __future__ import annotations
import datetime
import pytest
from weft.storage import Storage, UnknownRunError
from weft.types import Observation, Step


def make_storage() -> Storage:
    return Storage(":memory:")


def test_create_run_and_list_runs():
    storage = make_storage()
    run_id = storage.create_run("do the task")
    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id
    assert runs[0].task == "do the task"
    assert runs[0].status == "recording"
    assert runs[0].forked_from_run_id is None


def test_append_step_and_get_own_steps_round_trip():
    storage = make_storage()
    run_id = storage.create_run("do the task")
    storage.append_step(
        run_id, 0, Step("read_file", {"path": "a.py"}), Observation("hello", False), "t0", "t1"
    )
    entries = storage.get_own_steps(run_id)
    assert len(entries) == 1
    assert entries[0].step == Step("read_file", {"path": "a.py"})
    assert entries[0].observation == Observation("hello", False)


def test_append_step_with_non_json_native_result_does_not_raise():
    storage = make_storage()
    run_id = storage.create_run("do the task")
    non_native_result = datetime.datetime(2024, 1, 1, 12, 30)
    storage.append_step(
        run_id, 0, Step("read_file", {"path": "a.py"}), Observation(non_native_result, False), "t0", "t1"
    )
    entries = storage.get_own_steps(run_id)
    assert len(entries) == 1
    assert entries[0].observation.result == repr(non_native_result)


def test_get_run_raises_for_unknown_run():
    storage = make_storage()
    with pytest.raises(UnknownRunError):
        storage.get_run("nope")


def test_resolve_full_history_for_root_run():
    storage = make_storage()
    run_id = storage.create_run("task")
    storage.append_step(run_id, 0, Step("a", {}), Observation("r0"), "t0", "t1")
    storage.append_step(run_id, 1, Step("b", {}), Observation("r1"), "t2", "t3")
    history = storage.resolve_full_history(run_id)
    assert [e.step.tool_name for e in history] == ["a", "b"]


def test_resolve_full_history_for_single_branch():
    storage = make_storage()
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("a", {}), Observation("r0"), "t0", "t1")
    storage.append_step(parent_id, 1, Step("b", {}), Observation("r1"), "t2", "t3")
    storage.append_step(parent_id, 2, Step("c", {}), Observation("r2"), "t4", "t5")

    child_id = storage.create_run("task", forked_from_run_id=parent_id, forked_from_step=2)
    storage.append_step(child_id, 2, Step("d", {}), Observation("r2b"), "t6", "t7")

    history = storage.resolve_full_history(child_id)
    assert [e.step.tool_name for e in history] == ["a", "b", "d"]


def test_resolve_full_history_for_multi_level_branch():
    storage = make_storage()
    root_id = storage.create_run("task")
    storage.append_step(root_id, 0, Step("a", {}), Observation("r0"), "t0", "t1")
    storage.append_step(root_id, 1, Step("b", {}), Observation("r1"), "t2", "t3")

    mid_id = storage.create_run("task", forked_from_run_id=root_id, forked_from_step=1)
    storage.append_step(mid_id, 1, Step("c", {}), Observation("r1b"), "t4", "t5")
    storage.append_step(mid_id, 2, Step("d", {}), Observation("r2b"), "t6", "t7")

    leaf_id = storage.create_run("task", forked_from_run_id=mid_id, forked_from_step=2)
    storage.append_step(leaf_id, 2, Step("e", {}), Observation("r2c"), "t8", "t9")

    history = storage.resolve_full_history(leaf_id)
    assert [e.step.tool_name for e in history] == ["a", "c", "e"]
