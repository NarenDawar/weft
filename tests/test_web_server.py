from __future__ import annotations
import json
import threading
import urllib.request

from loom.storage import Storage
from loom.types import Observation, Step
from loom.web.server import serve


def _get_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def test_api_runs_lists_recorded_runs():
    storage = Storage(":memory:")
    run_id = storage.create_run("do the thing")
    server = serve(storage, port=0)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = _get_json(f"http://127.0.0.1:{actual_port}/api/runs")
        assert len(data) == 1
        assert data[0]["run_id"] == run_id
        assert data[0]["task"] == "do the thing"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_api_run_history_returns_steps():
    storage = Storage(":memory:")
    run_id = storage.create_run("task")
    storage.append_step(run_id, 0, Step("read_file", {"path": "a.py"}), Observation("hi"), "t0", "t1")
    server = serve(storage, port=0)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = _get_json(f"http://127.0.0.1:{actual_port}/api/runs/{run_id}")
        assert len(data) == 1
        assert data[0]["tool_name"] == "read_file"
        assert data[0]["result"] == "hi"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_api_diff_returns_divergence():
    storage = Storage(":memory:")
    run_a = storage.create_run("task")
    storage.append_step(run_a, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    run_b = storage.create_run("task")
    storage.append_step(run_b, 0, Step("read_file", {"path": "b.py"}), Observation("y"), "t0", "t1")
    server = serve(storage, port=0)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = _get_json(f"http://127.0.0.1:{actual_port}/api/diff/{run_a}/{run_b}")
        assert data["divergence_index"] == 0
        assert data["divergence_reason"] == "decision"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_index_page_is_served():
    storage = Storage(":memory:")
    server = serve(storage, port=0)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{actual_port}/") as response:
            body = response.read().decode("utf-8")
            assert "<html" in body.lower()
    finally:
        server.shutdown()
        thread.join(timeout=5)
