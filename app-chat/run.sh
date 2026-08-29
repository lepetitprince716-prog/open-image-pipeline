#!/bin/zsh
# app-chat sidecar — idempotent start/stop/status (chat agent for the workbench).
#
# Same lifecycle rationale as scripts/app_server.sh: not launchd (the server's
# Agent SDK child reads ~/.claude and the repo under ~/Documents, which launchd
# children cannot without Full Disk Access), started from an FDA-carrying
# session, then double-fork + setsid so automation-runner process-group cleanup
# does not reap it.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=8874
LOG="$HOME/Library/Logs/loramake-chat.log"
HEALTH="http://127.0.0.1:$PORT/"
# Claude egress contract (mirrors ~/.local/bin/claude-shadowrocket): the Agent
# SDK's CLI child reaches api.anthropic.com through the local ACL proxy, and
# NO_PROXY keeps the loramake calls (127.0.0.1:8873) direct.
PROXY_URL="http://127.0.0.1:11882"
NO_PROXY_HOSTS="localhost,127.0.0.1,::1"

listener_pid() { lsof -ti tcp:$PORT -s tcp:LISTEN 2>/dev/null | head -1; }

is_ours() {  # recognize the node instance managed by this script
  ps -p "$1" -o command= 2>/dev/null | grep -q "src/main.ts"
}

status() {
  local pid; pid=$(listener_pid)
  if [ -z "$pid" ]; then echo "stopped"; return 1; fi
  if ! is_ours "$pid"; then echo "port $PORT held by foreign pid $pid"; return 2; fi
  if curl -s --noproxy '*' --max-time 5 "$HEALTH" | grep -q "Z3Comfy Chat"; then
    echo "running pid=$pid healthy"
  else
    echo "running pid=$pid but health check failed"; return 3
  fi
}

start() {
  if [ ! -d "$ROOT/node_modules" ]; then
    echo "node_modules missing — run: cd $ROOT && pnpm install" >&2
    return 1
  fi
  if ! /usr/bin/nc -z -G 1 127.0.0.1 11882 >/dev/null 2>&1; then
    echo "local ACL proxy 127.0.0.1:11882 is not ready — the Agent SDK cannot reach Anthropic" >&2
    return 69
  fi
  local pid; pid=$(listener_pid)
  if [ -n "$pid" ]; then
    if is_ours "$pid"; then echo "already running pid=$pid"; return 0; fi
    echo "port $PORT held by foreign pid $pid — leaving untouched"; return 2
  fi
  local node_bin; node_bin="$(command -v node)"
  if [ -z "$node_bin" ]; then echo "node not found on PATH" >&2; return 1; fi

  export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL"
  export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL"
  export NO_PROXY="$NO_PROXY_HOSTS" no_proxy="$NO_PROXY_HOSTS"
  export NODE_USE_ENV_PROXY=1
  export NODE_ENV=production
  # Subscription credentials only: a stray API key would silently switch the
  # SDK to pay-per-token billing.
  unset ANTHROPIC_API_KEY

  echo "ROOT=$ROOT"
  /usr/bin/env python3 - "$ROOT" "$LOG" "$node_bin" <<'PY'
import os
import sys

root, log_path, node_bin = sys.argv[1:]
if os.fork() > 0:
    raise SystemExit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)

os.chdir(root)
os.umask(0o077)
stdin_fd = os.open(os.devnull, os.O_RDONLY)

# Bounded local log lifecycle (next start only; never rotates a live fd).
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

os.execv(
    node_bin,
    [node_bin, "--env-file-if-exists=.env", "--import", "tsx", "src/main.ts"],
)
PY
  for _ in {1..20}; do
    curl -s --noproxy '*' --max-time 2 "$HEALTH" | grep -q "Z3Comfy Chat" && { echo "started pid=$(listener_pid)"; return 0; }
    sleep 0.5
  done
  echo "did not become healthy in 10s — tail $LOG"; return 3
}

stop() {
  local pid; pid=$(listener_pid)
  [ -z "$pid" ] && { echo "not running"; return 0; }
  if ! is_ours "$pid"; then echo "port $PORT held by foreign pid $pid — leaving untouched"; return 2; fi
  kill "$pid" || return 1
  for _ in {1..10}; do
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
