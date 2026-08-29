from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GUARD_DIR = ROOT / "tools" / "ci_network_guard"


def _run_child(code: str, *, enabled: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GUARD_DIR)
    env["LORAMAKE_CI_BLOCK_NETWORK"] = "1" if enabled else "0"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_guard_blocks_non_loopback_connection_without_waiting() -> None:
    started = time.monotonic()
    result = _run_child(
        "import socket; socket.create_connection(('198.51.100.1', 9), timeout=8)",
        enabled=True,
    )
    elapsed = time.monotonic() - started

    assert result.returncode != 0
    assert elapsed < 2
    assert "CI network guard blocked non-loopback connection" in result.stderr


def test_guard_allows_loopback_connections() -> None:
    code = """
import socket
import threading

server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
port = server.getsockname()[1]

def accept_once():
    conn, _ = server.accept()
    with conn:
        conn.sendall(b"ok")
    server.close()

thread = threading.Thread(target=accept_once)
thread.start()
with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
    assert client.recv(2) == b"ok"
thread.join(timeout=2)
assert not thread.is_alive()
"""
    result = _run_child(code, enabled=True)
    assert result.returncode == 0, result.stderr


def test_guard_is_inert_without_explicit_ci_flag() -> None:
    result = _run_child(
        "import socket; assert not getattr(socket, '_loramake_ci_network_guard', False)",
        enabled=False,
    )
    assert result.returncode == 0, result.stderr
