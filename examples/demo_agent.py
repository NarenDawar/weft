from __future__ import annotations

from dataclasses import dataclass, field

from weft.branch import branch
from weft.diff import diff_runs
from weft.recording import begin_recording
from weft.registry import ToolRegistry
from weft.replay import ReplayExecutor, ReplayModelClient
from weft.storage import Storage
from weft.types import Decision, Step, Tool


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
    """A fully scripted ModelClient standing in for a real model — Weft's
    recording/replay/branch/diff mechanism doesn't care whether decisions
    come from a real API or a script, so this demo needs no API key."""

    def __init__(self, decisions: list[Decision]):
        self._decisions = list(decisions)

    def decide(self, task, history, tools) -> Decision:
        return self._decisions.pop(0)


def run_demo() -> None:
    storage = Storage("demo.weft.db")
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
