from __future__ import annotations
from examples.demo_agent import run_demo


def test_demo_runs_end_to_end_and_produces_expected_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_demo()
    captured = capsys.readouterr()
    assert "Recorded run" in captured.out
    assert "Branched run" in captured.out
    assert "shared prefix: 1 step(s)" in captured.out
    assert "diverges at step 1 (decision)" in captured.out
