#!/bin/zsh
# loramake app backend — idempotent start/stop/status (docs/remote_access.md "常驻").
#
# Why not launchd: uvicorn reads the repo under ~/Documents, which launchd children cannot
# needs Full Disk Access (the gacha TCC lesson, DAILY_GACHA.md). The server therefore starts
# from an FDA-carrying session (Claude Code / Terminal), then double-forks and calls setsid so
# automation-runner process-group cleanup does not reap uvicorn. A Claude Code scheduled task
# re-runs `start` periodically as the reboot watchdog.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Project venv (uv sync --extra dev --extra semantic). No cross-project venvs: start refuses
# with a clear message instead of silently borrowing another project's interpreter.
PY="$REPO/.venv/bin/python"
PORT=8873
LOG="$HOME/Library/Logs/loramake-app.log"
HEALTH="http://127.0.0.1:$PORT/api/health"
RDBG_MANIFEST="$REPO/tools/rdbg/Cargo.toml"
RDBG_LOCK="$REPO/tools/rdbg/Cargo.lock"
RDBG_BIN="$REPO/tools/rdbg/target/release/rdbg"

listener_pid() { lsof -ti tcp:$PORT -s tcp:LISTEN 2>/dev/null | head -1; }

is_ours() {  # recognize the uvicorn instance managed by this script
  ps -p "$1" -o command= 2>/dev/null | grep -q "uvicorn app.server:app"
}

status() {
  local pid; pid=$(listener_pid)
  if [ -z "$pid" ]; then echo "stopped"; return 1; fi
  if ! is_ours "$pid"; then echo "port $PORT held by foreign pid $pid"; return 2; fi
  if curl -s --noproxy '*' --max-time 5 "$HEALTH" | grep -q '"ok"'; then
    echo "running pid=$pid healthy"
  else
    echo "running pid=$pid but health check failed"; return 3
  fi
}

ensure_rdbg() {
  if [ -n "${LORAMAKE_RDBG_BIN:-}" ]; then
    [ -x "$LORAMAKE_RDBG_BIN" ] && return 0
    echo "LORAMAKE_RDBG_BIN is not executable: $LORAMAKE_RDBG_BIN"; return 4
  fi
  local needs_build=0
  [ -x "$RDBG_BIN" ] || needs_build=1
  if [ "$needs_build" -eq 0 ] && [ -n "$(
    find "$REPO/tools/rdbg/src" "$RDBG_MANIFEST" "$RDBG_LOCK" \
      -newer "$RDBG_BIN" -print -quit 2>/dev/null
  )" ]; then
    needs_build=1
  fi
  [ "$needs_build" -eq 0 ] && return 0
  command -v cargo >/dev/null 2>&1 || { echo "cargo is required to build tools/rdbg"; return 4; }
  echo "building tools/rdbg release binary"
  cargo build --release --locked --manifest-path "$RDBG_MANIFEST" || return 4
}

start() {
  if [ ! -x "$PY" ]; then
    echo "project venv missing: $PY — run: uv sync --extra dev --extra semantic" >&2
    return 1
  fi
  # The server only sees outputs under its own ROOT. Starting from a worktree serves
  # that sparse tree and the gallery looks wiped (2026-08-08 incident) — refuse unless
  # explicitly overridden. Production 8873 starts from the main checkout only.
  case "$REPO" in
    */.worktrees/*|*/worktrees/*)
      if [ "${LORAMAKE_ALLOW_WORKTREE_SERVER:-0}" != "1" ]; then
        echo "refusing to start: ROOT is a worktree ($REPO)"
        echo "use the main checkout's scripts/app_server.sh, or set LORAMAKE_ALLOW_WORKTREE_SERVER=1"
        return 2
      fi
      echo "warning: serving from worktree ROOT ($REPO)"
      ;;
  esac
  local pid; pid=$(listener_pid)
  if [ -n "$pid" ]; then
    if is_ours "$pid"; then
      ensure_rdbg || return $?
      # Report the live process's own cwd, not this invocation's REPO — after a
      # wrong-tree launch the two differ, and the running one is the truth.
      echo "already running pid=$pid (ROOT=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p'))"
      return 0
    fi
    echo "port $PORT held by foreign pid $pid — leaving untouched"; return 2
  fi
  ensure_rdbg || return $?
  echo "ROOT=$REPO"
  cd "$REPO" || return 1
  # A plain `nohup ... &` still shares the caller's process group. Non-interactive
  # automation runners may reap that group as soon as the command finishes, which makes
  # the server look healthy for one probe and then disappear. Double-fork + setsid gives
  # uvicorn its own session while preserving the FDA-carrying launch ancestry.
  "$PY" - "$REPO" "$LOG" "$PORT" <<'PY'
import os
import sys

repo, log_path, port = sys.argv[1:]
if os.fork() > 0:
    raise SystemExit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)

os.chdir(repo)
os.umask(0o077)
stdin_fd = os.open(os.devnull, os.O_RDONLY)

# Bounded local log lifecycle (next start only; this script never rotates a live fd).
log = os.path.abspath(log_path)
try:
    if os.path.getsize(log) > 10 * 1024 * 1024:
        oldest = f"{log}.5"
        if os.path.exists(oldest):
            os.unlink(oldest)
        for index in range(4, 0, -1):
            source = f"{log}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{log}.{index + 1}")
        os.replace(log, f"{log}.1")
except FileNotFoundError:
    pass
log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(stdin_fd, 0)
os.dup2(log_fd, 1)
os.dup2(log_fd, 2)
if stdin_fd > 2:
    os.close(stdin_fd)
if log_fd > 2:
    os.close(log_fd)

python = sys.executable
os.execv(
    python,
    [python, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", port],
)
PY
  for _ in {1..20}; do
    curl -s --noproxy '*' --max-time 2 "$HEALTH" | grep -q '"ok"' && { echo "started pid=$(listener_pid)"; return 0; }
    sleep 0.5
  done
  echo "did not become healthy in 10s — tail $LOG"; return 3
}

stop() {
  local pid; pid=$(listener_pid)
  [ -z "$pid" ] && { echo "not running"; return 0; }
  if ! is_ours "$pid"; then echo "port $PORT held by foreign pid $pid — leaving untouched"; return 2; fi
  kill "$pid" || return 1
  for _ in {1..10}; do  # wait for the port to clear so `restart` cannot race the old listener
    [ -z "$(listener_pid)" ] && { echo "stopped pid=$pid"; return 0; }
    sleep 0.3
  done
  echo "stopped pid=$pid (port still draining)"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 64 ;;
esac
