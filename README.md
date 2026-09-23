# Weft

[![tests](https://github.com/NarenDawar/weft/actions/workflows/test.yml/badge.svg)](https://github.com/NarenDawar/weft/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**A time-travel debugger for AI agents — git for agent execution.**

Agent runs are stochastic and multi-step, and today's tools show you a trace
after the fact but stop there. You can't ask "what would have happened if
step 3 had gone differently" without burning real API calls on a fresh,
*different* run. You can't tell exactly where two runs of the same task
diverged without reading two full traces side by side.

Weft fixes that by recording every model decision and tool call your agent
makes, then giving you three operations on top of the recording:

- **Replay** a run deterministically — no real API calls, no real side
  effects, instant.
- **Branch** a new run from any recorded step — the shared prefix replays
  for free, then it goes live from the fork point.
- **Diff** two runs to find the exact step where they diverged, and whether
  it was a different action or just a different result from the same one.

It ships with a CLI and a local web UI for browsing runs, their branch
points, and diffs between them.

## Why

Weft wraps two things any tool-calling agent loop already has: a model
client (something that turns history into a decision) and tool execution.
Nothing about your agent's own control-flow code changes — only which
client objects you hand it change, between a live run, a replay, and a
branch. There's no framework to adopt and no service to run; it's a local
SQLite file and a couple of Python classes.

## Install

```bash
pip install -e ".[dev]"
```

(Not yet published to PyPI — install from a checkout of this repo.) Zero
third-party dependencies for the core library — everything is Python
stdlib. `pytest` is only needed for running the test suite.

## Quickstart

Weft works with any object that looks like a model client — a real LLM API
wrapper, or (as below) something fully scripted. The recording/replay/branch
mechanism doesn't care which:

```python
from weft.recording import begin_recording
from weft.registry import ToolRegistry
from weft.storage import Storage
from weft.types import Decision, Step, Tool

# Your own tools, in your own environment
def read_file(path: str) -> str:
    return open(path).read()

def write_file(path: str, content: str) -> str:
    open(path, "w").write(content)
    return f"wrote {path}"

registry = ToolRegistry([
    Tool("read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, read_file),
    Tool("write_file", "Write a file", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write_file),
])

# Your own model client — just needs a `.decide(task, history, tools) -> Decision` method
model_client = my_model_client

storage = Storage(".weft.db")
run_id, recording_client, recording_executor = begin_recording(
    storage, "fix the typo in greet.py", model_client, registry
)

# Your existing agent loop, unchanged — just fed the recording wrappers
history = []
while True:
    decision = recording_client.decide("fix the typo in greet.py", history, registry.tools())
    if decision.tool_name is None:
        break
    step = Step(decision.tool_name, decision.args)
    observation = recording_executor.execute(step)
    history.append((step, observation))
```

That's it — every decision and tool call is now durably recorded in
`.weft.db`.

### Replay

```python
from weft.replay import ReplayModelClient, ReplayExecutor

replay_client = ReplayModelClient(storage, run_id)
replay_executor = ReplayExecutor(storage, run_id)
# Same loop shape as above, but decide()/execute() return the recorded
# values instead of calling anything real.
```

### Branch

```python
from weft.branch import branch

child_run_id, branch_client, branch_executor = branch(
    storage, run_id, from_step=1, real_client=model_client, registry=registry
)
# Steps 0..0 replay from the parent; step 1 onward runs live and records
# a new run, `child_run_id`, forked from `run_id` at step 1.
```

### Diff

```python
from weft.diff import diff_runs

result = diff_runs(storage, run_id, child_run_id)
print(result.shared_prefix_length, result.divergence_index, result.divergence_reason)
```

## CLI

```bash
weft list                    # every recorded run, with fork relationships
weft show <run_id>           # step-by-step history for one run
weft diff <run_id_a> <run_id_b>
weft serve                   # local web UI at http://127.0.0.1:8420
```

All four read from `.weft.db` in the current directory by default; pass
`--db <path>` to point at a different file.

## Web UI

`weft serve` starts a local, stdlib-only web server with a timeline view of
each run, its fork point relative to its parent (steps it replayed are
dimmed, its own new steps aren't), and a two-run diff view — check two runs
in the list and a "Compare" button appears, rendering both timelines side by
side with the divergence point highlighted.

## Try it without any setup

```bash
python examples/demo_agent.py
```

Runs a full record → replay → branch → diff cycle against an in-memory fake
filesystem with a fully scripted model client — no API key, no network
call, nothing to configure. It's the fastest way to see the whole mechanism
work end to end.

## How it works

Three client-side modes implement the same two interfaces — a `decide()`
method and a tool-executing `execute()` method:

- **Recording** wraps a real model client and real tool execution, logging
  every `(decision, tool call, result)` triple to SQLite before forwarding
  to the real thing.
- **Replay** wraps a `run_id` instead. `decide()`/`execute()` return the
  recorded result for the current step and advance — no real API call, no
  real side effect.
- **Branch** replays a parent run's steps up to a fork point, then
  transparently switches to recording live from there, writing a new run
  linked to the parent by `(forked_from_run_id, forked_from_step)`.

A branched run never copies its parent's steps into its own storage — it
stores only its own new steps starting at the fork point, and a run's full
history is reconstructed by walking the fork chain (recursively, for a
branch of a branch). This keeps storage from duplicating and means the
timeline UI's branch structure is the actual stored shape, not a display
trick.

Diffing two runs walks both resolved histories in parallel and reports the
first index where either the decision (tool name + args) or the observation
(the real result) differs — distinguishing "took a different action" from
"took the same action, got a different result" from "one run is just longer
than the other."

Full design rationale: [`docs/superpowers/specs/2026-09-23-weft-agent-time-travel-debugger-design.md`](docs/superpowers/specs/2026-09-23-weft-agent-time-travel-debugger-design.md).

## Current limitations

This is a v1. Known gaps, roughly in order of how much they'll bite you:

- **Tool results and args must be JSON-representable.** Anything that
  isn't (a `datetime`, a custom object) gets stored as its `repr()` string
  as a fallback — it won't replay back as its original type.
  Design/lookup-heavy tool args and results that are plain strings, numbers,
  dicts, and lists round-trip exactly.
- **A decision is only recorded once its tool call finishes**, not when the
  model makes it. If your process crashes mid-tool-call, that step is lost
  from the record entirely rather than partially captured.
- **No LangGraph (or other framework) adapter yet** — you wire the
  recording/replay/branch wrappers into your own loop directly. A LangGraph
  integration is the natural next step once the core mechanism has more
  mileage.
- **Local, single-user tool, not a hosted service.** `weft serve` binds to
  `127.0.0.1` by default and doesn't check the `Host` header — don't expose
  it on an untrusted network.
- **No redaction.** If your recorded args/observations contain secrets or
  PII, they're in `.weft.db` in plaintext. Don't commit that file or share
  it without checking what's in it first.

## License

MIT — see [LICENSE](LICENSE).
