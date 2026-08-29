#!/usr/bin/env python3
"""Serve the app-web side-panel static export on a friendly host. Proxy /api and /files to 8873.

Usage (via portless):  portless panel python3 scripts/panel_server.py
PANEL_ROOT overrides the served export; default is this checkout's app-web/out.
Rebuild with `cd app-web && pnpm build` — the server serves whatever is on disk.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = os.environ.get(
    "PANEL_ROOT",
    str(Path(__file__).resolve().parents[1] / "app-web" / "out"),
)
UPSTREAM = os.environ.get("PANEL_UPSTREAM", "http://127.0.0.1:8873")
PORT = int(os.environ["PORT"])
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — base-class name
        sys.stderr.write("%s %s\n" % (self.command, format % args))

    def do_GET(self) -> None:
        if self._is_api():
            self._proxy()
        else:
            super().do_GET()

    def do_HEAD(self) -> None:
        if self._is_api():
            self._proxy()
        else:
            super().do_HEAD()

    def do_POST(self) -> None:
        self._proxy_or_405()

    def do_PUT(self) -> None:
        self._proxy_or_405()

    def do_PATCH(self) -> None:
        self._proxy_or_405()

    def do_DELETE(self) -> None:
        self._proxy_or_405()

    def do_OPTIONS(self) -> None:
        if self._is_api():
            self._proxy()
        else:
            self.send_error(405)

    def _is_api(self) -> bool:
        path = self.path.split("?", 1)[0]
        return path == "/files" or path.startswith("/files/") or path.startswith("/api/")

    def _proxy_or_405(self) -> None:
        if self._is_api():
            self._proxy()
        else:
            self.send_error(405)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        req = urllib.request.Request(
            UPSTREAM + self.path,
            data=body,
            method=self.command,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in HOP:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.flush()  # headers out now — SSE clients watch the stream from byte one
                if self.command == "HEAD":
                    return
                # read1 returns as soon as ANY bytes are available; plain read(64k) would
                # block until the buffer fills and strangle SSE (/api/events) to silence.
                read_some = getattr(resp, "read1", None) or resp.read
                while True:
                    chunk = read_some(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as err:
            data = err.read()
            self.send_response(err.code)
            for key, value in err.headers.items():
                if key.lower() in HOP:
                    continue
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD" and data:
                self.wfile.write(data)
        except Exception as exc:
            self.send_error(502, str(exc))


if __name__ == "__main__":
    if not os.path.isdir(os.path.join(ROOT, "panel")):
        sys.exit(f"panel export missing under {ROOT} — run: cd app-web && pnpm build")
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"panel static {ROOT} :{PORT} -> {UPSTREAM}", flush=True)
    httpd.serve_forever()
