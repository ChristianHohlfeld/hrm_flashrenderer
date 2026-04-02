#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _handler_for(name: str, empty_sources: bool):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = _json_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/v1/health":
                self._write_json(404, {"ok": False, "error": "not found"})
                return
            if name == "router":
                self._write_json(200, {"ok": True})
                return
            self._write_json(
                200,
                {
                    "ok": True,
                    "backend": "deepseek_int8",
                    "deepseek_running": True,
                    "name": name,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/generate":
                self._write_json(404, {"ok": False, "error": "not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(body.decode("utf-8"))
            except Exception:
                self._write_json(400, {"ok": False, "error": "invalid json"})
                return

            if name != "router":
                self._write_json(200, {"ok": True, "text": "backend-ok", "sources": [{"sid": "b#1", "txt": "ok"}]})
                return

            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                self._write_json(400, {"ok": False, "error": "prompt required"})
                return

            sources = [] if empty_sources else [{"sid": "0001#s0001", "txt": "Mock source evidence."}]
            self._write_json(
                200,
                {
                    "ok": True,
                    "text": "Mock DeepSeek answer.",
                    "sources": sources,
                    "route": {
                        "selected": "nvlink_pair",
                        "prompt_tokens": 42,
                        "latency_ms": 7,
                    },
                },
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def _start_server(port: int, name: str, empty_sources: bool) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler_for(name=name, empty_sources=empty_sources))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock HRM flash stack servers for CI tests.")
    ap.add_argument("--solo-port", type=int, required=True)
    ap.add_argument("--nvlink-port", type=int, required=True)
    ap.add_argument("--solo3080-port", type=int, required=True)
    ap.add_argument("--router-port", type=int, required=True)
    ap.add_argument("--empty-sources", action="store_true")
    args = ap.parse_args()

    servers = [
        _start_server(args.solo_port, "solo_22gb", args.empty_sources),
        _start_server(args.nvlink_port, "nvlink_pair", args.empty_sources),
        _start_server(args.solo3080_port, "solo_3080", args.empty_sources),
        _start_server(args.router_port, "router", args.empty_sources),
    ]

    stop = threading.Event()

    def _shutdown(*_sig: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    stop.wait()
    for s in servers:
        s.shutdown()
        s.server_close()


if __name__ == "__main__":
    main()
