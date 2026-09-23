from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from weft.diff import diff_runs
from weft.storage import RunRecord, Storage

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
