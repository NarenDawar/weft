from __future__ import annotations

import argparse
import sys

from loom.diff import diff_runs
from loom.storage import Storage
from loom.web.server import serve


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

    serve_parser = subparsers.add_parser("serve", help="Start the local web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8420)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
