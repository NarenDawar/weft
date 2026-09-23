# Loom: Time-Travel Debugger for Agents — Design

**Status:** Approved for planning
**Date:** 2026-09-23

## Summary

Loom is a Python library and local tool that records every model decision
and tool call a tool-calling agent makes, then lets you replay a recorded
run deterministically (no real API calls, no real side effects), branch a
new run from any step in a recorded run while reusing its prefix, and diff
two runs to find exactly where their behavior diverged. It's "git for agent
execution": agent runs are stochastic and multi-step, and existing
observability tools show you a trace after the fact but don't let you go
back, fork, or compare. Loom does all three.

## Motivation

Debugging a misbehaving agent today means staring at a trace log and
guessing. You can't ask "what would have happened if step 3 had gone
differently" without re-running the whole agent from scratch against a live
model, burning time and money, and getting a *different* non-deterministic
run instead of an answer. You can't compare two runs of the same task to
see exactly which decision or which tool result caused them to diverge
without manually reading two full traces side by side. Loom makes both of
these first-class operations.

## Goals

- Record every model decision and tool call in a tool-calling agent loop
  with zero changes to the agent's own control-flow code — only the client
  objects passed in change.
- Replay a recorded run deterministically: identical steps, no real API
  calls, no real side effects, instant.
- Branch a new run from any step of a recorded run, replaying the shared
  prefix for free and continuing live from the fork point.
- Diff two runs and report the first point where their decisions or
  observations differ.
- Ship a local web UI showing a run's timeline, branch points, and
  run-to-run diffs, in addition to a CLI for inspection.

## Non-goals (v1)

- Framework-agnostic HTTP-level interception (recording works by wrapping a
  `ModelClient`-shaped object and tool execution, matching a specific but
  common agent-loop shape — not by proxying arbitrary network traffic).
- Any dependency on Scry or its types. Loom defines its own minimal
  protocol shapes, independently, even though they resemble Scry's by
  design (a proven pattern for this domain).
- Multi-user / hosted storage. v1 is a local SQLite file per project.
- Automatic instrumentation of arbitrary existing agent frameworks beyond
  the explicit wrapper interfaces (a LangGraph adapter is the one
  integration target, matching Scry's approach, not a general auto-patcher).

## Architecture

Three client-side modes, all implementing the same two interfaces a user's
agent loop already depends on — a `ModelClient`-shaped object (`decide(task,
history, tools) -> Decision`) and tool execution:

- **Recording mode** (`RecordingModelClient`, `RecordingExecutor`): wraps a
  real model client and real tool execution. Every `decide()` call and
  every tool call is logged to SQLite with a monotonic step index *before*
  being forwarded to the real thing, so a run's log is complete even if a
  later step crashes.
- **Replay mode** (`ReplayModelClient`, `ReplayExecutor`): wraps a
  `run_id` instead of a real client. `decide()` and tool calls return the
  recorded result directly for the current step index — no real API call,
  no real side effect. Advancing past the end of what was recorded is an
  error, not a silent fallback, since it means the agent's code no longer
  matches what was recorded.
- **Branch mode**: a factory (`loom.branch(run_id, from_step=N)`) that
  returns a client pair which replays steps `0..N-1` from the recording,
  then transparently switches to recording live from step `N` onward,
  writing a new run whose `forked_from_run_id`/`forked_from_step` point at
  the parent.

A user's agent loop code is written once against the `ModelClient`/tool
interfaces and never needs to know which mode it's running in — only the
client objects passed into it change between a live run, a replay, and a
branch.

## Storage

One SQLite file per project. Two tables:

- **`runs`**: `run_id` (short random hex), `task`, `started_at`,
  `forked_from_run_id` (nullable), `forked_from_step` (nullable), `status`
  (`recording` / `complete`).
- **`steps`**: `run_id`, `step_index`, `tool_name`, `args_json`,
  `observation_json`, `is_error`, `decided_at`, `executed_at`. One row per
  (decision, tool call, result) triple — the atomic unit for diffing and
  branching, the same way a commit is git's atomic unit.

A branched run does **not** copy its parent's steps — it stores only its
own new steps starting at the fork point, plus the fork pointer.
Reconstructing a run's full history means walking the fork chain
(recursively, for multi-level branches) and concatenating parent steps
before the fork point with the run's own steps. This keeps storage
non-duplicated and makes the DAG structure the timeline UI displays the
actual stored structure, not a display-time reconstruction trick.

## Diff algorithm

Given two run IDs, resolve each run's full step history (via the fork-chain
walk above), then compare index by index:

1. If the decision (`tool_name` + `args`) at index `i` differs between the
   two runs, that's the divergence point — a different action was taken.
2. If the decision matches but the `observation` differs, that's still a
   divergence — the same action produced a different real-world result
   (e.g. a file changed between the two runs).
3. The first mismatch of either kind is reported as the divergence point;
   everything before it is the shared prefix, everything from it onward is
   run-specific. If one run is shorter, that's reported as well (ran fewer
   steps, not a mismatch in itself).

## CLI

Inspection only — branching is a Python API, not a CLI command, since
re-invoking a user's own agent loop code isn't something a generic CLI can
do:

- `loom list` — list all runs in the local `.loom.db`.
- `loom show <run_id>` — print the step-by-step summary for one run.
- `loom diff <run_id_a> <run_id_b>` — structural diff per the algorithm
  above.
- `loom serve` — starts the local web UI (see below).

## Web UI

`loom serve` starts a small local HTTP server (Python stdlib only, no web
framework — this is simple enough not to need one, matching Scry's
minimal-dependency philosophy) exposing a small JSON API (`/api/runs`,
`/api/runs/<id>`, `/api/diff/<a>/<b>`) and a static page that renders a
run's timeline, its branch points, and run-to-run diffs client-side with
vanilla JS.

## Error handling

- **Replay past the end of a recording**: raises a clear error naming the
  run and the step index — this means the agent's code no longer matches
  what was recorded, and silently falling through would hide that.
- **Branch with an out-of-range `from_step`** (negative or past the
  parent's recorded length): a validation error before anything runs.
- **Real tool/model exceptions during recording**: captured as an error
  observation (mirroring Scry's `Observation(is_error=True)` pattern) so a
  failed step is still part of the permanent record, not lost.

## Testing plan

- **Recording**: wrapper forwards to the real client/tool and logs the
  correct row.
- **Replay**: returns recorded values without touching the real
  client/tool; raises the documented error when asked to go past the
  recorded length.
- **Branch**: produces a run with the correct parent link; full-history
  reconstruction is correct across multiple levels of branching (a branch
  of a branch).
- **Diff**: correctly finds the first divergence by decision, by
  observation, and correctly reports runs of different lengths.

All tested against a temp-file SQLite database and deterministic fake
clients/tools, the same pattern as Scry's `tests/fakes.py`.

## Repository shape

```
loom/                       core package
  types.py                   Step, Observation, HistoryEntry, Decision
  model.py                   ModelClient protocol
  storage.py                 SQLite schema, run/step CRUD, fork-chain resolution
  recording.py                RecordingModelClient, RecordingExecutor
  replay.py                   ReplayModelClient, ReplayExecutor
  branch.py                   loom.branch() factory
  diff.py                     diff algorithm
  cli.py                       list / show / diff / serve
  web/
    server.py                  stdlib HTTP server + JSON API
    static/                     timeline HTML/JS/CSS
examples/                   a hand-rolled agent demonstrating record → replay → branch → diff
tests/
```

## Roadmap (explicitly out of scope for v1)

- LangGraph adapter (mirroring Scry's `scry/integrations/langgraph.py`
  pattern) once the core recording/replay/branch mechanism is proven.
- Multi-user / hosted storage for teams sharing recorded runs.
- Automatic redaction of sensitive data (API keys, PII) in recorded
  `args_json`/`observation_json` before a run is shared or exported.
