#!/bin/zsh
# loramake app backend — idempotent start/stop/status (docs/remote_access.md "常驻").
#
# Production persistence is LaunchAgent com.loramake.app (scripts/com.loramake.app.plist):
# foreground uvicorn on 127.0.0.1:8873, KeepAlive. This script is the manual/FDA helper
# and must not be the KeepAlive Program — it double-forks, so launchd would see an exit
# and spawn duplicates. `stop` kills the listener; if the agent is loaded, KeepAlive
# respawns it. Boot out com.loramake.app to stop for good.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Project venv (uv sync --extra dev --extra semantic). No cross-project venvs: start refuses
# with a clear message instead of silently borrowing another project's interpreter.
PY="$REPO/.venv/bin/python"
PORT=8873
LOG="$HOME/Library/Logs/loramake-app.log"
HEALTH="http://127.0.0.1:$PORT/api/health"
# Remote diagnostics run `rdbg doctor` from the rdbg release (app/remote_diagnostics.py).
RDBG_BIN="${LORAMAKE_RDBG_BIN:-$HOME/.local/bin/rdbg}"

listener_pid() { lsof -ti tcp:$PORT -s tcp:LISTEN 2>/dev/null | head -1; }

is_ours() {  # recognize the uvicorn instance managed by this script
  ps -p "$1" -o command= 2>/dev/null | grep -q "uvicorn app.server:app"
}

AGENT="gui/$(id -u)/com.loramake.app"

agent_loaded() { launchctl print "$AGENT" >/dev/null 2>&1; }

wait_healthy() {
  for _ in {1..20}; do
    curl -s --noproxy '*' --max-time 2 "$HEALTH" | grep -q '"ok"' && { echo "healthy pid=$(listener_pid)"; return 0; }
    sleep 0.5
  done
  echo "did not become healthy in 10s — tail $LOG"; return 3
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

# Diagnostics are optional: without rdbg 0.1.x the API answers 503, so only warn.
ensure_rdbg() {
  if [ -x "$RDBG_BIN" ] && "$RDBG_BIN" --version 2>/dev/null | grep -q '^rdbg 0\.1\.'; then
    return 0
  fi
  echo "warning: rdbg 0.1.x not found at $RDBG_BIN; /api/diagnostics will answer 503"
  echo "         install it as described in docs/remote_access.md (Remote diagnostics)"
}

start() {
  # launchd owns production 8873. A manual uvicorn cannot bind it, and every failed
  # respawn still imports app.server — which used to converge the live worker's jobs.
  # Hand the start to launchd instead of racing it.
  if agent_loaded; then
    echo "com.loramake.app is loaded — launchd owns port $PORT; kickstarting it instead"
    launchctl kickstart "$AGENT" || return 1
    wait_healthy
    return $?
  fi
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
  wait_healthy
}

stop() {
  local pid; pid=$(listener_pid)
  [ -z "$pid" ] && { echo "not running"; return 0; }
  if ! is_ours "$pid"; then echo "port $PORT held by foreign pid $pid — leaving untouched"; return 2; fi
  kill "$pid" || return 1
  local cleared=0
  for _ in {1..10}; do  # wait for the port to clear so `restart` cannot race the old listener
    if [ -z "$(listener_pid)" ]; then
      cleared=1
      break
    fi
    sleep 0.3
  done
  if [ "$cleared" = 1 ]; then
    echo "stopped pid=$pid"
  else
    echo "stopped pid=$pid (port still draining)"
  fi
  if agent_loaded; then
    echo "note: com.loramake.app KeepAlive will respawn; bootout to stop for good"
  fi
}

restart() {
  # stop+start hands the port to launchd's KeepAlive respawn, which then races this
  # script's own uvicorn: whichever loses the bind exits, and its aborted startup used
  # to converge the winner's in-flight jobs. kickstart -k keeps launchd the only starter.
  if agent_loaded; then
    echo "com.loramake.app is loaded — restarting through launchd"
    launchctl kickstart -k "$AGENT" || return 1
    wait_healthy
    return $?
  fi
  stop
  start
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 64 ;;
esac
