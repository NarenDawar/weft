# Loom v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Loom v1 — a Python library and local tool that records every model decision and tool call in a tool-calling agent loop, then supports deterministic replay, branching a new run from any recorded step, diffing two runs to find where they diverged, and browsing all of this through a local web UI.

**Architecture:** Three client-side modes (Recording, Replay, Branch) all implement the same `ModelClient`/tool-execution interfaces a user's agent loop already depends on, so no control-flow changes are needed to switch between them. A SQLite storage layer holds two tables (`runs`, `steps`); a branched run stores only its own new steps and reconstructs full history by walking the fork chain. A diff algorithm and a stdlib-only local web server sit on top of the same storage layer.

**Tech Stack:** Python 3.10+, stdlib only for the core (`sqlite3`, `http.server`, `json`, `secrets`), `pytest` for tests. Zero third-party runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-loom-agent-time-travel-debugger-design.md`

## Global Constraints

- Core package (`loom/`) has zero required third-party dependencies — stdlib only (`sqlite3`, `http.server`, `json`, `secrets`, `datetime`).
- Python 3.10+ syntax throughout (`from __future__ import annotations`).
- Storage is a single SQLite file per project; hand-written schema and queries, no ORM.
- A branched run stores only its own new steps (never copies a parent's steps). Full history is reconstructed by walking the fork chain, recursively for multi-level branches.
- Replay past the end of a recording raises `ReplayExhaustedError` unless the run's `status` is `"complete"`, in which case it returns `Decision(tool_name=None)` — these are semantically different signals (an unexpected mismatch vs. genuine completion) and must never be conflated.
- `loom/web/` uses Python's stdlib `http.server` only — no web framework dependency.
- `pyproject.toml`'s package include list must ship only `loom*` — never `examples*` (they're demo code, not library code users install).

## Review Focus

- Multi-level branching (a branch of a branch) must reconstruct full history correctly by recursively walking the fork chain, not just resolving one level — owned by Task 2 (`test_resolve_full_history_for_multi_level_branch`).
- Replay must distinguish "genuinely done" (`status == "complete"`, returns `Decision(None)`) from "recording exhausted unexpectedly" (raises `ReplayExhaustedError`) — conflating them would hide real agent-code drift as if it were normal completion. Owned by Task 5 (`test_replay_model_client_returns_none_when_complete_and_exhausted`, `test_replay_model_client_raises_when_exhausted_and_not_complete`).
- Branch boundary values — `from_step=0` (branch immediately, nothing replayed) and `from_step == len(parent_history)` (continue live exactly where the parent left off) — must both be valid, not off-by-one errors. Owned by Task 6 (`test_branch_at_step_zero_skips_replay_entirely`, `test_branch_at_step_equal_to_parent_length_continues_live_immediately`).
- A tool that raises a real exception during recording must be captured as `Observation(is_error=True)`, not crash the recording session. Owned by Task 4 (`test_recording_executor_captures_tool_exception_as_error_observation`).
- Diffing two runs where one is a genuine prefix of the other (no divergence within the shared steps, only a length difference) must be reported distinctly from a decision/observation divergence. Owned by Task 7 (`test_diff_reports_length_difference_when_one_run_is_shorter`).

---

## File Structure

```
pyproject.toml
loom/
  __init__.py
  types.py
  registry.py
  storage.py
  model.py
  recording.py
  replay.py
  branch.py
  diff.py
  cli.py
  web/
    __init__.py
    server.py
    static/
      index.html
      app.js
      style.css
examples/
  __init__.py
  demo_agent.py
tests/
  __init__.py
  fakes.py
  test_registry.py
  test_storage.py
  test_model.py
  test_recording.py
  test_replay.py
  test_branch.py
  test_diff.py
  test_cli.py
  test_web_server.py
  test_demo_agent.py
```

---

### Task 1: Project scaffolding, core types, and tool registry

**Files:**
- Create: `pyproject.toml`
- Create: `loom/__init__.py`
- Create: `loom/types.py`
- Create: `loom/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `Step(tool_name: str, args: dict)`, `Observation(result: Any, is_error: bool = False)`, `HistoryEntry(step: Step, observation: Observation)`, `Decision(tool_name: str | None, args: dict = {})`, `Tool(name: str, description: str, parameters: dict, fn: Callable)` — all frozen dataclasses in `loom/types.py`.
- Produces: `ToolRegistry(tools: list[Tool])` with `.tools() -> list[Tool]`, `.get(name) -> Tool` (raises `UnknownToolError`); `UnknownToolError` exception, in `loom/registry.py`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "loom-agent-debugger"
version = "0.1.0"
description = "Time-travel debugger for tool-calling agents: record, replay, branch, and diff agent runs"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["loom*"]
```

- [ ] **Step 2: Install the package in editable/dev mode**

Run: `pip install -e ".[dev]"`

- [ ] **Step 3: Write the failing test**

```python
# tests/test_registry.py
from __future__ import annotations
import pytest
from loom.registry import ToolRegistry, UnknownToolError
from loom.types import Tool


def make_registry():
    read = Tool(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        fn=lambda path: f"contents of {path}",
    )
    return ToolRegistry([read])


def test_get_returns_registered_tool():
    registry = make_registry()
    assert registry.get("read_file").name == "read_file"


def test_get_raises_for_unknown_tool():
    registry = make_registry()
    with pytest.raises(UnknownToolError):
        registry.get("nope")


def test_tools_returns_all_registered_tools():
    registry = make_registry()
    names = [t.name for t in registry.tools()]
    assert names == ["read_file"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.registry'` (or similar import errors)

- [ ] **Step 5: Write `loom/__init__.py`**

```python
"""Loom: a time-travel debugger for tool-calling agents."""
```

- [ ] **Step 6: Write `loom/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Step:
    tool_name: str
    args: dict


@dataclass(frozen=True)
class Observation:
    result: Any
    is_error: bool = False


@dataclass(frozen=True)
class HistoryEntry:
    step: Step
    observation: Observation


@dataclass(frozen=True)
class Decision:
    tool_name: Optional[str]
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
```

- [ ] **Step 7: Write `loom/registry.py`**

```python
from __future__ import annotations

from loom.types import Tool


class UnknownToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml loom/__init__.py loom/types.py loom/registry.py tests/test_registry.py
git commit -m "feat: add core types and tool registry"
```

---

### Task 2: SQLite storage layer with fork-chain resolution

**Files:**
- Create: `loom/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `Step`, `Observation`, `HistoryEntry` from `loom/types.py` (Task 1).
- Produces: `UnknownRunError`; `generate_run_id() -> str`; `now_iso() -> str`; `RunRecord(run_id, task, started_at, forked_from_run_id, forked_from_step, status)` (frozen dataclass); `Storage(db_path: str)` with `.create_run(task, forked_from_run_id=None, forked_from_step=None, run_id=None) -> str`, `.mark_complete(run_id)`, `.append_step(run_id, step_index, step, observation, decided_at, executed_at)`, `.get_run(run_id) -> RunRecord` (raises `UnknownRunError`), `.list_runs() -> list[RunRecord]`, `.get_own_steps(run_id) -> list[HistoryEntry]`, `.resolve_full_history(run_id) -> list[HistoryEntry]`, `.close()`. `Storage(":memory:")` is a valid in-memory database, used throughout the test suite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from __future__ import annotations
import pytest
from loom.storage import Storage, UnknownRunError
from loom.types import Observation, Step


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.storage'`

- [ ] **Step 3: Write `loom/storage.py`**

```python
from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loom.types import HistoryEntry, Observation, Step


class UnknownRunError(Exception):
    pass


def generate_run_id() -> str:
    return secrets.token_hex(4)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    task: str
    started_at: str
    forked_from_run_id: Optional[str]
    forked_from_step: Optional[int]
    status: str


class Storage:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                started_at TEXT NOT NULL,
                forked_from_run_id TEXT,
                forked_from_step INTEGER,
                status TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                observation_json TEXT NOT NULL,
                is_error INTEGER NOT NULL,
                decided_at TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                PRIMARY KEY (run_id, step_index)
            )
            """
        )
        self.conn.commit()

    def create_run(
        self,
        task: str,
        forked_from_run_id: Optional[str] = None,
        forked_from_step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        run_id = run_id or generate_run_id()
        self.conn.execute(
            "INSERT INTO runs (run_id, task, started_at, forked_from_run_id, forked_from_step, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, task, now_iso(), forked_from_run_id, forked_from_step, "recording"),
        )
        self.conn.commit()
        return run_id

    def mark_complete(self, run_id: str) -> None:
        self.conn.execute("UPDATE runs SET status = 'complete' WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def append_step(
        self,
        run_id: str,
        step_index: int,
        step: Step,
        observation: Observation,
        decided_at: str,
        executed_at: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO steps (run_id, step_index, tool_name, args_json, observation_json, "
            "is_error, decided_at, executed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                step_index,
                step.tool_name,
                json.dumps(step.args),
                json.dumps(observation.result),
                1 if observation.is_error else 0,
                decided_at,
                executed_at,
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> RunRecord:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise UnknownRunError(run_id)
        return RunRecord(
            run_id=row["run_id"],
            task=row["task"],
            started_at=row["started_at"],
            forked_from_run_id=row["forked_from_run_id"],
            forked_from_step=row["forked_from_step"],
            status=row["status"],
        )

    def list_runs(self) -> list[RunRecord]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY started_at ASC").fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                task=r["task"],
                started_at=r["started_at"],
                forked_from_run_id=r["forked_from_run_id"],
                forked_from_step=r["forked_from_step"],
                status=r["status"],
            )
            for r in rows
        ]

    def get_own_steps(self, run_id: str) -> list[HistoryEntry]:
        rows = self.conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index ASC", (run_id,)
        ).fetchall()
        return [
            HistoryEntry(
                step=Step(tool_name=r["tool_name"], args=json.loads(r["args_json"])),
                observation=Observation(
                    result=json.loads(r["observation_json"]), is_error=bool(r["is_error"])
                ),
            )
            for r in rows
        ]

    def resolve_full_history(self, run_id: str) -> list[HistoryEntry]:
        run = self.get_run(run_id)
        own = self.get_own_steps(run_id)
        if run.forked_from_run_id is None:
            return own
        parent_history = self.resolve_full_history(run.forked_from_run_id)
        prefix = parent_history[: run.forked_from_step]
        return prefix + own

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage layer with fork-chain history resolution"
```

---

### Task 3: Model client protocol and test fakes

**Files:**
- Create: `loom/model.py`
- Create: `tests/__init__.py`
- Create: `tests/fakes.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Decision`, `HistoryEntry`, `Tool` from `loom/types.py` (Task 1).
- Produces: `ModelClient` protocol with `.decide(task: str, history: list[HistoryEntry], tools: list[Tool]) -> Decision`, in `loom/model.py`.
- Produces (test-only, imported by later tasks' tests): `FakeModelClient(decisions: list[Decision])` with `.decide(...)` popping decisions in order and `.calls` counter; `FailingModelClient(error: Exception)` with `.decide(...)` always raising, in `tests/fakes.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
from __future__ import annotations
import pytest
from loom.types import Decision
from tests.fakes import FakeModelClient, FailingModelClient


def test_fake_model_client_returns_decisions_in_order():
    client = FakeModelClient([Decision("a", {}), Decision("b", {})])
    assert client.decide("t", [], []).tool_name == "a"
    assert client.decide("t", [], []).tool_name == "b"
    assert client.calls == 2


def test_failing_model_client_raises():
    client = FailingModelClient(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        client.decide("t", [], [])
    assert client.calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fakes'`

- [ ] **Step 3: Write `loom/model.py`**

```python
from __future__ import annotations

from typing import Protocol

from loom.types import Decision, HistoryEntry, Tool


class ModelClient(Protocol):
    def decide(self, task: str, history: list[HistoryEntry], tools: list[Tool]) -> Decision: ...
```

- [ ] **Step 4: Write `tests/__init__.py`**

```python
```

- [ ] **Step 5: Write `tests/fakes.py`**

```python
from __future__ import annotations

from loom.types import Decision


class FakeModelClient:
    def __init__(self, decisions: list[Decision]):
        self._decisions = list(decisions)
        self.calls = 0

    def decide(self, task, history, tools) -> Decision:
        self.calls += 1
        return self._decisions.pop(0)


class FailingModelClient:
    def __init__(self, error: Exception):
        self._error = error
        self.calls = 0

    def decide(self, task, history, tools):
        self.calls += 1
        raise self._error
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add loom/model.py tests/__init__.py tests/fakes.py tests/test_model.py
git commit -m "feat: add ModelClient protocol and test fakes"
```

---

### Task 4: Recording — RecordingModelClient, RecordingExecutor, begin_recording

**Files:**
- Create: `loom/recording.py`
- Test: `tests/test_recording.py`

**Interfaces:**
- Consumes: `ModelClient` (Task 3), `ToolRegistry`/`UnknownToolError` (Task 1), `Storage`/`now_iso` (Task 2), `Step`/`Observation`/`Decision`/`HistoryEntry` (Task 1), `FakeModelClient` (Task 3).
- Produces: `RecordingModelClient(real_client: ModelClient, storage: Storage, run_id: str)` with `.decide(...)` — forwards to the real client, and calls `storage.mark_complete(run_id)` when the decision's `tool_name is None`. `RecordingExecutor(registry: ToolRegistry, storage: Storage, run_id: str, start_index: int = 0)` with `.execute(step: Step) -> Observation` — executes the real tool, captures exceptions as `Observation(is_error=True)`, and logs every attempt to storage regardless of success. `begin_recording(storage, task, real_client, registry) -> tuple[str, RecordingModelClient, RecordingExecutor]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recording.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recording.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.recording'`

- [ ] **Step 3: Write `loom/recording.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recording.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/recording.py tests/test_recording.py
git commit -m "feat: add recording wrappers that log every decision and tool call"
```

---

### Task 5: Replay — ReplayModelClient, ReplayExecutor

**Files:**
- Create: `loom/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `Decision`/`Observation`/`Step`/`HistoryEntry` (Task 1).
- Produces: `ReplayExhaustedError`; `ReplayModelClient(storage: Storage, run_id: str)` with `.decide(task, history, tools) -> Decision` — returns the recorded decision at the current index and advances; once past the recorded length, returns `Decision(tool_name=None)` if the run's status is `"complete"`, otherwise raises `ReplayExhaustedError`. `ReplayExecutor(storage: Storage, run_id: str)` with `.execute(step: Step) -> Observation` — returns the recorded observation at the current index and advances; raises `ReplayExhaustedError` once past the recorded length. Both classes independently track their own index; they stay in lockstep because any normal agent loop calls `decide()` then `execute()` alternately, exactly once each per step.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.replay'`

- [ ] **Step 3: Write `loom/replay.py`**

```python
from __future__ import annotations

from loom.storage import Storage
from loom.types import Decision, Observation, Step


class ReplayExhaustedError(Exception):
    pass


class ReplayModelClient:
    def __init__(self, storage: Storage, run_id: str):
        self.storage = storage
        self.run_id = run_id
        self._steps = storage.resolve_full_history(run_id)
        self._next_index = 0

    def decide(self, task: str, history, tools) -> Decision:
        if self._next_index >= len(self._steps):
            run = self.storage.get_run(self.run_id)
            if run.status == "complete":
                return Decision(tool_name=None, args={})
            raise ReplayExhaustedError(
                f"run {self.run_id!r} has no recorded step at index {self._next_index}"
            )
        entry = self._steps[self._next_index]
        self._next_index += 1
        return Decision(tool_name=entry.step.tool_name, args=entry.step.args)


class ReplayExecutor:
    def __init__(self, storage: Storage, run_id: str):
        self.storage = storage
        self.run_id = run_id
        self._steps = storage.resolve_full_history(run_id)
        self._next_index = 0

    def execute(self, step: Step) -> Observation:
        if self._next_index >= len(self._steps):
            raise ReplayExhaustedError(
                f"run {self.run_id!r} has no recorded step at index {self._next_index}"
            )
        entry = self._steps[self._next_index]
        self._next_index += 1
        return entry.observation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_replay.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/replay.py tests/test_replay.py
git commit -m "feat: add deterministic replay wrappers"
```

---

### Task 6: Branch — the branch() factory

**Files:**
- Create: `loom/branch.py`
- Test: `tests/test_branch.py`

**Interfaces:**
- Consumes: `RecordingModelClient`/`RecordingExecutor` (Task 4), `ReplayModelClient`/`ReplayExecutor` (Task 5), `Storage` (Task 2), `ModelClient` (Task 3), `ToolRegistry` (Task 1), `FakeModelClient` (Task 3).
- Produces: `InvalidForkPointError`; `BranchModelClient` and `BranchExecutor` (dataclasses wrapping a replay/recording pair plus the fork step, switching from replay to recording once the shared index reaches the fork point); `branch(storage, parent_run_id, from_step, real_client, registry) -> tuple[str, BranchModelClient, BranchExecutor]` — validates `from_step` is in `[0, len(parent_history)]` (raising `InvalidForkPointError` otherwise), creates the child run with the fork pointer, and returns wired clients.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_branch.py
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


def test_branch_at_step_equal_to_parent_length_continues_live_immediately():
    storage = Storage(":memory:")
    parent_id = storage.create_run("task")
    storage.append_step(parent_id, 0, Step("read_file", {"path": "a.py"}), Observation("x"), "t0", "t1")
    storage.mark_complete(parent_id)

    real_client = FakeModelClient([Decision(None, {})])
    child_id, model_client, executor = branch(
        storage, parent_id, from_step=1, real_client=real_client, registry=make_registry()
    )

    decision = model_client.decide("task", [], [])
    assert decision.tool_name is None
    assert real_client.calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_branch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.branch'`

- [ ] **Step 3: Write `loom/branch.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from loom.model import ModelClient
from loom.recording import RecordingExecutor, RecordingModelClient
from loom.registry import ToolRegistry
from loom.replay import ReplayExecutor, ReplayModelClient
from loom.storage import Storage
from loom.types import Decision, Observation, Step


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_branch.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/branch.py tests/test_branch.py
git commit -m "feat: add branch() to fork a new live run from a recorded step"
```

---

### Task 7: Diff algorithm

**Files:**
- Create: `loom/diff.py`
- Test: `tests/test_diff.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `HistoryEntry` (Task 1).
- Produces: `DiffResult(shared_prefix_length: int, divergence_index: int | None, divergence_reason: str | None, run_a_length: int, run_b_length: int)` (frozen dataclass; `divergence_reason` is `"decision"`, `"observation"`, or `None`); `diff_runs(storage, run_id_a, run_id_b) -> DiffResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diff.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diff.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.diff'`

- [ ] **Step 3: Write `loom/diff.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loom.storage import Storage


@dataclass(frozen=True)
class DiffResult:
    shared_prefix_length: int
    divergence_index: Optional[int]
    divergence_reason: Optional[str]
    run_a_length: int
    run_b_length: int


def diff_runs(storage: Storage, run_id_a: str, run_id_b: str) -> DiffResult:
    history_a = storage.resolve_full_history(run_id_a)
    history_b = storage.resolve_full_history(run_id_b)

    shortest = min(len(history_a), len(history_b))
    for i in range(shortest):
        entry_a = history_a[i]
        entry_b = history_b[i]
        if entry_a.step != entry_b.step:
            return DiffResult(
                shared_prefix_length=i,
                divergence_index=i,
                divergence_reason="decision",
                run_a_length=len(history_a),
                run_b_length=len(history_b),
            )
        if entry_a.observation != entry_b.observation:
            return DiffResult(
                shared_prefix_length=i,
                divergence_index=i,
                divergence_reason="observation",
                run_a_length=len(history_a),
                run_b_length=len(history_b),
            )

    divergence_index = shortest if len(history_a) != len(history_b) else None
    return DiffResult(
        shared_prefix_length=shortest,
        divergence_index=divergence_index,
        divergence_reason=None,
        run_a_length=len(history_a),
        run_b_length=len(history_b),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diff.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/diff.py tests/test_diff.py
git commit -m "feat: add run diff algorithm"
```

---

### Task 8: CLI — list, show, diff

**Files:**
- Create: `loom/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `diff_runs` (Task 7).
- Produces: `build_parser() -> argparse.ArgumentParser`; `main(argv: list[str] | None = None) -> int` — the CLI entry point, dispatching `list`/`show`/`diff` subcommands against a `--db` SQLite path (default `.loom.db`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.cli'`

- [ ] **Step 3: Write `loom/cli.py`**

```python
from __future__ import annotations

import argparse
import sys

from loom.diff import diff_runs
from loom.storage import Storage


def _open_storage(db_path: str) -> Storage:
    return Storage(db_path)


def cmd_list(args: argparse.Namespace) -> None:
    storage = _open_storage(args.db)
    runs = storage.list_runs()
    if not runs:
        print("No runs recorded yet.")
        return
    for run in runs:
        fork_note = (
            f" (forked from {run.forked_from_run_id} @ step {run.forked_from_step})"
            if run.forked_from_run_id
            else ""
        )
        print(f"{run.run_id}  {run.status:10}  {run.started_at}  {run.task}{fork_note}")


def cmd_show(args: argparse.Namespace) -> None:
    storage = _open_storage(args.db)
    run = storage.get_run(args.run_id)
    history = storage.resolve_full_history(args.run_id)
    print(f"Run {run.run_id}  ({run.status}, {len(history)} steps)")
    print(f"Task: {run.task}")
    print()
    for i, entry in enumerate(history):
        marker = " [ERROR]" if entry.observation.is_error else ""
        print(f"  [{i}] {entry.step.tool_name}({entry.step.args}) -> {entry.observation.result!r}{marker}")


def cmd_diff(args: argparse.Namespace) -> None:
    storage = _open_storage(args.db)
    result = diff_runs(storage, args.run_id_a, args.run_id_b)
    print(
        f"Comparing {args.run_id_a} ({result.run_a_length} steps) and "
        f"{args.run_id_b} ({result.run_b_length} steps)"
    )
    print(f"Shared prefix: {result.shared_prefix_length} step(s)")
    if result.divergence_reason == "decision":
        print(f"Diverges at step {result.divergence_index}: different action taken")
    elif result.divergence_reason == "observation":
        print(f"Diverges at step {result.divergence_index}: same action, different result")
    elif result.divergence_index is not None:
        print(f"Runs share a common prefix but differ in length starting at step {result.divergence_index}")
    else:
        print("No divergence: runs are identical")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loom", description="Time-travel debugger for agents.")
    parser.add_argument("--db", default=".loom.db", help="Path to the Loom SQLite database (default: .loom.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List all recorded runs")
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show a run's step-by-step history")
    show_parser.add_argument("run_id")
    show_parser.set_defaults(func=cmd_show)

    diff_parser = subparsers.add_parser("diff", help="Diff two runs")
    diff_parser.add_argument("run_id_a")
    diff_parser.add_argument("run_id_b")
    diff_parser.set_defaults(func=cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add loom/cli.py tests/test_cli.py
git commit -m "feat: add CLI (list, show, diff)"
```

---

### Task 9: Web UI — local server and timeline view, plus `loom serve`

**Files:**
- Create: `loom/web/__init__.py`
- Create: `loom/web/server.py`
- Create: `loom/web/static/index.html`
- Create: `loom/web/static/app.js`
- Create: `loom/web/static/style.css`
- Modify: `loom/cli.py` (add the `serve` subcommand)
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `diff_runs` (Task 7).
- Produces: `serve(storage: Storage, host: str = "127.0.0.1", port: int = 8420) -> http.server.ThreadingHTTPServer` — constructs and binds the server (the caller controls `.serve_forever()`); exposes `GET /api/runs`, `GET /api/runs/<run_id>`, `GET /api/diff/<a>/<b>`, and serves `index.html`/`app.js`/`style.css` as static files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_server.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loom.web'`

- [ ] **Step 3: Write `loom/web/__init__.py`**

```python
```

- [ ] **Step 4: Write `loom/web/server.py`**

```python
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from loom.diff import diff_runs
from loom.storage import RunRecord, Storage

STATIC_DIR = Path(__file__).parent / "static"


def _run_to_dict(run: RunRecord) -> dict:
    return {
        "run_id": run.run_id,
        "task": run.task,
        "started_at": run.started_at,
        "forked_from_run_id": run.forked_from_run_id,
        "forked_from_step": run.forked_from_step,
        "status": run.status,
    }


def _entry_to_dict(entry) -> dict:
    return {
        "tool_name": entry.step.tool_name,
        "args": entry.step.args,
        "result": entry.observation.result,
        "is_error": entry.observation.is_error,
    }


def make_handler(storage: Storage) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass

        def _send_json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]

            if parts == ["api", "runs"]:
                self._send_json([_run_to_dict(r) for r in storage.list_runs()])
                return

            if len(parts) == 3 and parts[0] == "api" and parts[1] == "runs":
                history = storage.resolve_full_history(parts[2])
                self._send_json([_entry_to_dict(e) for e in history])
                return

            if len(parts) == 4 and parts[0] == "api" and parts[1] == "diff":
                result = diff_runs(storage, parts[2], parts[3])
                self._send_json(
                    {
                        "shared_prefix_length": result.shared_prefix_length,
                        "divergence_index": result.divergence_index,
                        "divergence_reason": result.divergence_reason,
                        "run_a_length": result.run_a_length,
                        "run_b_length": result.run_b_length,
                    }
                )
                return

            if parsed.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html")
                return
            if parsed.path == "/app.js":
                self._send_file(STATIC_DIR / "app.js", "application/javascript")
                return
            if parsed.path == "/style.css":
                self._send_file(STATIC_DIR / "style.css", "text/css")
                return

            self._send_json({"error": "not found"}, status=404)

    return Handler


def serve(storage: Storage, host: str = "127.0.0.1", port: int = 8420) -> ThreadingHTTPServer:
    handler_cls = make_handler(storage)
    return ThreadingHTTPServer((host, port), handler_cls)
```

- [ ] **Step 5: Write `loom/web/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Loom</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <h1>Loom</h1>
  <div id="runs"></div>
  <div id="timeline"></div>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 6: Write `loom/web/static/app.js`**

```javascript
async function loadRuns() {
  const response = await fetch("/api/runs");
  const runs = await response.json();
  const container = document.getElementById("runs");
  container.innerHTML = "";
  const list = document.createElement("ul");
  runs.forEach((run) => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = `${run.run_id} — ${run.task} (${run.status})`;
    link.onclick = (event) => {
      event.preventDefault();
      loadTimeline(run.run_id);
    };
    item.appendChild(link);
    list.appendChild(item);
  });
  container.appendChild(list);
}

async function loadTimeline(runId) {
  const response = await fetch(`/api/runs/${runId}`);
  const steps = await response.json();
  const container = document.getElementById("timeline");
  container.innerHTML = `<h2>Run ${runId}</h2>`;
  const list = document.createElement("ol");
  steps.forEach((step) => {
    const item = document.createElement("li");
    item.className = step.is_error ? "step step-error" : "step";
    item.textContent = `${step.tool_name}(${JSON.stringify(step.args)}) -> ${JSON.stringify(step.result)}`;
    list.appendChild(item);
  });
  container.appendChild(list);
}

loadRuns();
```

- [ ] **Step 7: Write `loom/web/static/style.css`**

```css
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 800px;
  margin: 40px auto;
  padding: 0 16px;
  color: #222;
}

h1 {
  font-size: 20px;
}

#runs ul, #timeline ol {
  list-style: none;
  padding: 0;
}

#runs li {
  margin-bottom: 6px;
}

.step {
  font-family: monospace;
  font-size: 13px;
  padding: 6px 8px;
  border-left: 3px solid #4a9;
  margin-bottom: 4px;
  background: #f7f7f7;
}

.step-error {
  border-left-color: #e05252;
  background: #fdf0f0;
}
```

- [ ] **Step 8: Modify `loom/cli.py`** — add the `serve` subcommand

Add this import near the top, alongside the existing imports:

```python
from loom.web.server import serve
```

Add this function alongside the other `cmd_*` functions:

```python
def cmd_serve(args: argparse.Namespace) -> None:
    storage = _open_storage(args.db)
    server = serve(storage, host=args.host, port=args.port)
    print(f"Loom serving at http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
```

Add this to `build_parser()`, alongside the other subparsers:

```python
    serve_parser = subparsers.add_parser("serve", help="Start the local web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8420)
    serve_parser.set_defaults(func=cmd_serve)
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_web_server.py tests/test_cli.py -v`
Expected: PASS (8 tests: 4 web server + 4 CLI, no regressions in CLI from the import/subcommand addition)

- [ ] **Step 10: Commit**

```bash
git add loom/web/ loom/cli.py tests/test_web_server.py
git commit -m "feat: add local web UI (timeline, diff) and loom serve command"
```

---

### Task 10: Example — record, replay, branch, diff end to end

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/demo_agent.py`
- Test: `tests/test_demo_agent.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `begin_recording` (Task 4), `ReplayModelClient`/`ReplayExecutor` (Task 5), `branch` (Task 6), `diff_runs` (Task 7), `ToolRegistry` (Task 1), `Step`/`Decision`/`Tool` (Task 1).
- Produces: `FakeFilesystem`, `make_tools(fs) -> list[Tool]`, `ScriptedClient`, `run_demo() -> None` — a fully self-contained, runnable demonstration needing no real API key, since Loom's recording/replay/branch/diff mechanism is agnostic to whether decisions come from a real model or a script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_demo_agent.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples'`

- [ ] **Step 3: Write `examples/__init__.py`**

```python
```

- [ ] **Step 4: Write `examples/demo_agent.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

from loom.branch import branch
from loom.diff import diff_runs
from loom.recording import begin_recording
from loom.registry import ToolRegistry
from loom.replay import ReplayExecutor, ReplayModelClient
from loom.storage import Storage
from loom.types import Decision, Step, Tool


@dataclass
class FakeFilesystem:
    files: dict = field(default_factory=dict)

    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path: str, content: str) -> str:
        self.files[path] = content
        return f"wrote {path}"


def make_tools(fs: FakeFilesystem) -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            fn=fs.read_file,
        ),
        Tool(
            name="write_file",
            description="Write a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            fn=fs.write_file,
        ),
    ]


class ScriptedClient:
    """A fully scripted ModelClient standing in for a real model — Loom's
    recording/replay/branch/diff mechanism doesn't care whether decisions
    come from a real API or a script, so this demo needs no API key."""

    def __init__(self, decisions: list[Decision]):
        self._decisions = list(decisions)

    def decide(self, task, history, tools) -> Decision:
        return self._decisions.pop(0)


def run_demo() -> None:
    storage = Storage("demo.loom.db")
    fs = FakeFilesystem(files={"greet.py": "def greet():\n    return 'helo'\n"})
    registry = ToolRegistry(make_tools(fs))
    task = "fix the typo in greet.py"

    print("--- Recording an original run ---")
    client = ScriptedClient(
        [
            Decision("read_file", {"path": "greet.py"}),
            Decision("write_file", {"path": "greet.py", "content": "def greet():\n    return 'hello'\n"}),
            Decision(None, {}),
        ]
    )
    run_id, model_client, executor = begin_recording(storage, task, client, registry)
    while True:
        decision = model_client.decide(task, [], registry.tools())
        if decision.tool_name is None:
            break
        executor.execute(Step(decision.tool_name, decision.args))
    print(f"Recorded run {run_id}")

    print("--- Replaying it (no real work happens) ---")
    replay_model = ReplayModelClient(storage, run_id)
    replay_executor = ReplayExecutor(storage, run_id)
    while True:
        decision = replay_model.decide(task, [], registry.tools())
        if decision.tool_name is None:
            break
        obs = replay_executor.execute(Step(decision.tool_name, decision.args))
        print(f"  replayed: {decision.tool_name}({decision.args}) -> {obs.result!r}")

    print("--- Branching from step 1 with a different fix ---")
    branch_fs = FakeFilesystem(files={"greet.py": "def greet():\n    return 'helo'\n"})
    branch_registry = ToolRegistry(make_tools(branch_fs))
    branch_client = ScriptedClient(
        [
            Decision("write_file", {"path": "greet.py", "content": "def greet():\n    return 'hi'\n"}),
            Decision(None, {}),
        ]
    )
    branch_run_id, branch_model, branch_executor = branch(
        storage, run_id, from_step=1, real_client=branch_client, registry=branch_registry
    )
    while True:
        decision = branch_model.decide(task, [], branch_registry.tools())
        if decision.tool_name is None:
            break
        branch_executor.execute(Step(decision.tool_name, decision.args))
    print(f"Branched run {branch_run_id}")

    print("--- Diffing the two runs ---")
    result = diff_runs(storage, run_id, branch_run_id)
    print(f"  shared prefix: {result.shared_prefix_length} step(s)")
    print(f"  diverges at step {result.divergence_index} ({result.divergence_reason})")

    storage.close()


if __name__ == "__main__":
    run_demo()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_demo_agent.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests across every task)

- [ ] **Step 7: Commit**

```bash
git add examples/__init__.py examples/demo_agent.py tests/test_demo_agent.py
git commit -m "feat: add end-to-end record/replay/branch/diff demo"
```

---

## Self-Review Notes

- **Spec coverage:** core types + registry (Task 1), SQLite storage with fork-chain resolution (Task 2), the three client modes — Recording (Task 4), Replay (Task 5), Branch (Task 6) — the diff algorithm (Task 7), the CLI (Task 8), the local web UI (Task 9), and the record→replay→branch→diff example (Task 10) — every architecture component and the storage/diff/CLI/web-UI sections of the spec have an owning task.
- **Placeholder scan:** no TBD/TODO; every step has real, runnable code.
- **Type consistency:** `Step`, `Observation`, `HistoryEntry`, `Decision`, `Tool` are defined once in `loom/types.py` (Task 1) and used with identical field names throughout every later task; `RunRecord`/`Storage` are defined once in Task 2 and consumed as-is by Tasks 4–9.
- **Review Focus:** all five items (multi-level branch reconstruction, replay-complete-vs-exhausted distinction, branch boundary values, tool-exception capture during recording, run-length-difference diffing) have concrete owning tests, listed above.
