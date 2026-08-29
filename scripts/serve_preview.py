#!/usr/bin/env python3
"""Dedicated local preview server for this repo's image galleries (Claude side).

Serves the loramake repo root over HTTP on 127.0.0.1:<port> (default 8770) so the
repo-root-relative URLs emitted by preview_gallery.py resolve. Kept on its OWN port and
pidfile so it never collides with the Hermes-side gallery server (127.0.0.1:18180) — the
two preview interfaces stay independent.

Started detached (start_new_session) with stdout/stderr to a logfile, so it survives the
agent shell that launched it and a broken pipe cannot wedge http.server — the failure mode
that orphaned the old shared :8765 server.

Usage:
    python3 scripts/serve_preview.py start     # spawn detached, print the preview URL
    python3 scripts/serve_preview.py status    # is it listening? + HTTP probe
    python3 scripts/serve_preview.py stop       # kill it (verified it is our http.server)
    python3 scripts/serve_preview.py restart

Env: LORAMAKE_PREVIEW_PORT (default 8770).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from config import ROOT

REPO = ROOT
PORT = (os.environ.get("LORAMAKE_PREVIEW_PORT", "8770").strip() or "8770")
HOST = "127.0.0.1"
PID_PATH = Path(f"/tmp/loramake-preview-{PORT}.pid")
LOG_PATH = Path(f"/tmp/loramake-preview-{PORT}.log")
PREVIEW_PATH = "/outputs/_preview/index.html"


def url() -> str:
    return f"http://{HOST}:{PORT}{PREVIEW_PATH}"


def listening_pid() -> int | None:
    """PID listening on PORT, or None. Uses lsof so a stale pidfile is never trusted."""
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out and out[0].isdigit() else None


def is_our_server(pid: int) -> bool:
    """True only if `pid` is an http.server bound to our port — so stop never kills a stranger
    that happens to hold the port."""
    cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                         capture_output=True, text=True).stdout
    return "http.server" in cmd and PORT in cmd


def http_probe() -> str:
    """HTTP status code for the preview URL (curl --noproxy, matching the repo convention)."""
    return subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "4",
         "--noproxy", HOST, url()], capture_output=True, text=True).stdout.strip()


def start() -> None:
    pid = listening_pid()
    if pid:
        who = "our server" if is_our_server(pid) else "a NON-preview process"
        print(f"already listening on {HOST}:{PORT} (pid {pid}, {who})")
        if is_our_server(pid):
            print(url())
        else:
            sys.exit(f"refusing to reuse {HOST}:{PORT}: held by {who}. Pick another "
                     f"LORAMAKE_PREVIEW_PORT.")
        return
    log = LOG_PATH.open("ab", buffering=0)
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", PORT, "--bind", HOST],
        cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    PID_PATH.write_text(f"{proc.pid}\n")
    time.sleep(0.6)
    if listening_pid():
        print(f"serving {REPO} on {HOST}:{PORT} (pid {proc.pid}, log {LOG_PATH})")
        print(url())
    else:
        sys.exit(f"failed to start on {HOST}:{PORT}; check {LOG_PATH}")


def stop() -> None:
    pid = listening_pid()
    if pid is None:
        PID_PATH.unlink(missing_ok=True)
        print(f"nothing listening on {HOST}:{PORT}")
        return
    if not is_our_server(pid):
        sys.exit(f"refusing to kill pid {pid} on {HOST}:{PORT}: not an http.server we own.")
    os.kill(pid, signal.SIGTERM)
    PID_PATH.unlink(missing_ok=True)
    print(f"stopped pid {pid} on {HOST}:{PORT}")


def status() -> None:
    pid = listening_pid()
    if pid is None:
        print(f"not listening on {HOST}:{PORT}")
        return
    who = "preview http.server" if is_our_server(pid) else "NON-preview process (foreign)"
    print(f"listening on {HOST}:{PORT} (pid {pid}, {who})")
    print(f"HTTP {http_probe()} for {url()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude-side preview server (own port, separate from Hermes)")
    ap.add_argument("action", choices=("start", "stop", "status", "restart"))
    args = ap.parse_args()
    if args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    elif args.action == "status":
        status()
    elif args.action == "restart":
        stop()
        start()


if __name__ == "__main__":
    main()
