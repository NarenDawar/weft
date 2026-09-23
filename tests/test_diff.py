from __future__ import annotations
from loom.diff import diff_runs
from loom.storage import Storage
from loom.types import Observation, Step


def make_run(storage, steps):
    run_id = storage.create_run("task")
    for i, (step, obs) in enumerate(steps):
        storage.append_step(run_id, i, step, obs, f"t{2 * i}", f"t{2 * i + 1}")
    return run_id


def test_diff_identical_runs_no_divergence():
    storage = Storage(":memory:")
    steps = [(Step("read_file", {"path": "a.py"}), Observation("x"))]
    run_a = make_run(storage, steps)
    run_b = make_run(storage, steps)
    result = diff_runs(storage, run_a, run_b)
    assert result.divergence_index is None
    assert result.shared_prefix_length == 1
    assert result.run_a_length == result.run_b_length == 1


def test_diff_finds_decision_divergence():
    storage = Storage(":memory:")
    run_a = make_run(storage, [(Step("read_file", {"path": "a.py"}), Observation("x"))])
    run_b = make_run(storage, [(Step("read_file", {"path": "b.py"}), Observation("y"))])
    result = diff_runs(storage, run_a, run_b)
    assert result.divergence_index == 0
    assert result.divergence_reason == "decision"


def test_diff_finds_observation_divergence_with_same_decision():
    storage = Storage(":memory:")
    run_a = make_run(storage, [(Step("read_file", {"path": "a.py"}), Observation("old content"))])
    run_b = make_run(storage, [(Step("read_file", {"path": "a.py"}), Observation("new content"))])
    result = diff_runs(storage, run_a, run_b)
    assert result.divergence_index == 0
    assert result.divergence_reason == "observation"


def test_diff_reports_length_difference_when_one_run_is_shorter():
    storage = Storage(":memory:")
    shared = (Step("read_file", {"path": "a.py"}), Observation("x"))
    run_a = make_run(storage, [shared])
    run_b = make_run(storage, [shared, (Step("write_file", {"path": "b.py", "content": "y"}), Observation("ok"))])
    result = diff_runs(storage, run_a, run_b)
    assert result.divergence_reason is None
    assert result.divergence_index == 1
    assert result.shared_prefix_length == 1
    assert result.run_a_length == 1
    assert result.run_b_length == 2
