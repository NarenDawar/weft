from __future__ import annotations
import pytest
from loom.branch import InvalidForkPointError, branch
from loom.registry import ToolRegistry
from loom.storage import Storage
from loom.types import Decision, Observation, Step, Tool
from tests.fakes import FakeModelClient


def make_registry():
    read = Tool(
        name="read_file",
        description="",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        fn=lambda path: f"contents of {path}",
    )
    write = Tool(
        name="write_file",
        description="",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        fn=lambda path, content: "written",
    )
    return ToolRegistry([read, write])


def test_branch_replays_prefix_then_records_live():
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("original a"), "t0", "t1")
    storage.append_step(parent_id, 1, Step("read_file", {"path": "b.py"}), Observation("original b"), "t2", "t3")
    storage.mark_complete(parent_id)

    real_client = FakeModelClient(
        [Decision("write_file", {"path": "c.py", "content": "new"}), Decision(None, {})]
    )
    child_id, model_client, executor = branch(
        storage, parent_id, from_step=1, real_client=real_client, registry=make_registry()
    )

    d0 = model_client.decide("task", [], [])
    assert d0.tool_name == "read_file" and d0.args == {"path": "a.py"}
    obs0 = executor.execute(Step(d0.tool_name, d0.args))
    assert obs0 == Observation("original a", False)  # replayed, not re-executed for real

    d1 = model_client.decide("task", [], [])
    assert d1.tool_name == "write_file" and d1.args == {"path": "c.py", "content": "new"}
    obs1 = executor.execute(Step(d1.tool_name, d1.args))
    assert obs1 == Observation("written", False)  # actually executed this time

    d2 = model_client.decide("task", [], [])
    assert d2.tool_name is None

    full_history = storage.resolve_full_history(child_id)
    assert [e.step.tool_name for e in full_history] == ["read_file", "write_file"]
    assert full_history[0].observation.result == "original a"
    assert full_history[1].observation.result == "written"


def test_branch_raises_for_out_of_range_from_step():
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    with pytest.raises(InvalidForkPointError):
        branch(storage, parent_id, from_step=5, real_client=FakeModelClient([]), registry=make_registry())


def test_branch_child_run_has_correct_fork_pointer():
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    child_id, _, _ = branch(
        storage, parent_id, from_step=1, real_client=FakeModelClient([Decision(None, {})]), registry=make_registry()
    )
    child = storage.get_run(child_id)
    assert child.forked_from_run_id == parent_id
    assert child.forked_from_step == 1


def test_branch_at_step_zero_skips_replay_entirely():
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    storage.mark_complete(parent_id)

    real_client = FakeModelClient(
        [Decision("write_file", {"path": "b.py", "content": "y"}), Decision(None, {})]
    )
    child_id, model_client, executor = branch(
        storage, parent_id, from_step=0, real_client=real_client, registry=make_registry()
    )

    decision = model_client.decide("task", [], [])
    assert decision.tool_name == "write_file"
    assert real_client.calls == 1


def test_branch_at_step_equal_to_parent_length_replays_everything_then_goes_live():
    # from_step=N replays indices 0..N-1, then goes live at index N. With a
    # 1-step parent and from_step=1, that means: replay that one step first
    # (index 0 < fork_step 1), THEN go live (index 1 is no longer < fork_step
    # 1) — not skip replay entirely. from_step=0 (the previous test) is the
    # only value that skips replay outright.
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    storage.mark_complete(parent_id)

    real_client = FakeModelClient([Decision(None, {})])
    child_id, model_client, executor = branch(
        storage, parent_id, from_step=1, real_client=real_client, registry=make_registry()
    )

    d0 = model_client.decide("task", [], [])
    assert d0.tool_name == "read_file"
    assert real_client.calls == 0
    obs0 = executor.execute(Step(d0.tool_name, d0.args))
    assert obs0 == Observation("x", False)

    d1 = model_client.decide("task", [], [])
    assert d1.tool_name is None
    assert real_client.calls == 1
