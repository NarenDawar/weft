from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from weft.types import HistoryEntry, Observation, Step


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
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
        # By the time this is called, the tool's real-world side effect has already
        # happened (see RecordingExecutor.execute), so serialization must not raise
        # and drop the record. `default=repr` falls back to repr() for any value
        # json can't natively serialize (datetime, bytes, custom objects, etc.).
        # Accepted limitation: such values are stored as their repr() string and
        # won't round-trip to their original type on replay (e.g. a tuple becomes
        # a string, not a list).
        self.conn.execute(
            "INSERT INTO steps (run_id, step_index, tool_name, args_json, observation_json, "
            "is_error, decided_at, executed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                step_index,
                step.tool_name,
                json.dumps(step.args, default=repr),
                json.dumps(observation.result, default=repr),
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
