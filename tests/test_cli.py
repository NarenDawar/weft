from __future__ import annotations
from loom.cli import main
from loom.storage import Storage
from loom.types import Observation, Step


def test_list_shows_no_runs_message_when_empty(tmp_path, capsys):
    db_path = str(tmp_path / "test.db")
    main(["--db", db_path, "list"])
    captured = capsys.readouterr()
    assert "No runs recorded yet" in captured.out


def test_list_shows_recorded_runs(tmp_path, capsys):
    db_path = str(tmp_path / "test.db")
    storage = Storage(db_path)
    run_id = storage.create_run("do the thing")
    storage.close()

    main(["--db", db_path, "list"])
    captured = capsys.readouterr()
    assert run_id in captured.out
    assert "do the thing" in captured.out


def test_show_prints_step_history(tmp_path, capsys):
    db_path = str(tmp_path / "test.db")
    storage = Storage(db_path)
    run_id = storage.create_run("task")
    storage.append_step(run_id, 0, Step("read_file", {"path": "a.py"}), Observation("hello"), "t0", "t1")
    storage.close()

    main(["--db", db_path, "show", run_id])
    captured = capsys.readouterr()
    assert "read_file" in captured.out
    assert "hello" in captured.out


def test_diff_prints_divergence_point(tmp_path, capsys):
    db_path = str(tmp_path / "test.db")
    storage = Storage(db_path)
    run_a = storage.create_run("task")
    storage.append_step(run_a, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    run_b = storage.create_run("task")
    storage.append_step(run_b, 0, Step("read_file", {"path": "b.py"}), Observation("y"), "t0", "t1")
    storage.close()

    main(["--db", db_path, "diff", run_a, run_b])
    captured = capsys.readouterr()
    assert "Diverges at step 0" in captured.out
    assert "different action taken" in captured.out
