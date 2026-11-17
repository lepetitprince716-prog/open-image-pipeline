"""Best-route selection for reaching the GPU boxes from this Mac.

Each box can be carried by more than one transport: the home LAN (fast, only
at home) and Cloudflare Access SSH (works anywhere the Mac has internet).
Consumers ask this module which route is live instead of hardcoding an ssh
alias or a host URL, so a box stays reachable when the Mac leaves the LAN.

Purity contract: importing this module never touches the network. Probing
happens only inside ``pick_route``/``probe_all``, and ``pick_route`` memoizes
its answer in a small JSON state file so the per-process cost is a file read
while the answer is fresh. When every route is down the canonical ``lan``
route is still reported, preserving the historical failure surface.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH_DEFAULT = ROOT / "app" / "data" / "net_routes.json"
STATE_ENV = "LORAMAKE_ROUTE_STATE"
TTL_ENV = "LORAMAKE_ROUTE_TTL_S"
TTL_DEFAULT_S = 90.0
SCHEMA = "loramake.net-routes.v1"
SSH_BIN = "/usr/bin/ssh"


@dataclass(frozen=True)
class Route:
    """One transport to a box. ``tcp`` is the fast liveness check used for
    picking; ``ssh_verify`` runs a full ssh roundtrip when tcp is None.
    ``proxy`` is the argv the ssh ProxyCommand execs to carry this route."""

    name: str
    ssh_alias: str
    comfy_host: str
    tcp: tuple[str, int] | None
    proxy: tuple[str, ...]
    ssh_timeout_s: float


ROUTES: dict[str, tuple[Route, ...]] = {
    "boxa": (
        Route(
            "lan",
            "boxa",
            "http://192.0.2.10:8188",
            ("192.0.2.10", 22),
            ("/usr/bin/nc", "192.0.2.10", "22"),
            4.0,
        ),
        Route(
            "cf",
            "boxa-cf",
            "http://127.0.0.1:18188",
            None,
            (
                str(Path.home() / ".cloudflared" / "boxa-cf-cloudflared-proxy.sh"),
                "ssh-boxa.example.com",
            ),
            15.0,
        ),
        # ponytail: ts ranks below cf on purpose. Away from the LAN, Tailscale is DERP-relayed and
        # measured 3.7 KB/s against Cloudflare's 461 KB/s (2026-09-22); at home lan wins anyway.
        # Promote it again only with a bandwidth probe, not by table order.
        # ComfyUI binds BOXA loopback only, so HTTP over ts still rides the ssh -L forward.
        Route(
            "ts",
            "boxa-ts",
            "http://127.0.0.1:18188",
            ("192.0.2.11", 22),
            ("/usr/bin/nc", "192.0.2.11", "22"),
            6.0,
        ),
    ),
    "boxb": (
        Route(
            "lan",
            "boxb",
            "http://192.0.2.20:8388",
            ("192.0.2.20", 22),
            ("/usr/bin/nc", "192.0.2.20", "22"),
            4.0,
        ),
        Route(
            "cf",
            "boxb-cf",
            "http://127.0.0.1:18388",
            None,
            (
                str(Path.home() / ".cloudflared" / "boxb-cf-cloudflared-proxy.sh"),
                "ssh-boxb.example.com",
            ),
            15.0,
        ),
    ),
}


def _routes_for(box: str) -> tuple[Route, ...]:
    try:
        return ROUTES[box]
    except KeyError:
        raise KeyError(box) from None


def state_path(env: dict | None = None) -> Path:
    values = os.environ if env is None else env
    raw = (values.get(STATE_ENV) or "").strip()
    return Path(raw).expanduser() if raw else STATE_PATH_DEFAULT


def _ttl_s(env: dict | None = None) -> float:
    values = os.environ if env is None else env
    raw = (values.get(TTL_ENV) or "").strip()
    try:
        return max(5.0, float(raw)) if raw else TTL_DEFAULT_S
    except ValueError:
        return TTL_DEFAULT_S


def _tcp_open(host: str, port: int, timeout: float) -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, (time.perf_counter() - started) * 1000
    except OSError:
        return False, (time.perf_counter() - started) * 1000


def _ssh_ok(alias: str, timeout: float) -> tuple[bool, float]:
    """One real ssh roundtrip (`exit` is a cmd builtin, so it works under both
    cmd.exe and PowerShell default shells). Control sockets are bypassed so the
    probe always pays for a fresh transport."""
    argv = [
        SSH_BIN,
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(timeout)}",
        "-o", "ControlMaster=no",
        "-o", "ControlPath=none",
        alias,
        "exit",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, capture_output=True, timeout=timeout + 4,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode == 0, (time.perf_counter() - started) * 1000
    except (OSError, subprocess.SubprocessError):
        return False, (time.perf_counter() - started) * 1000


def _http_ok(host: str, timeout: float = 6.0) -> tuple[bool, float, str | None]:
    """GET <host>/system_stats; returns (ok, rtt_ms, comfyui_version|None)."""
    started = time.perf_counter()
    try:
        with urlrequest.urlopen(f"{host}/system_stats", timeout=timeout) as resp:
            raw = resp.read(1 << 20)
            rtt = (time.perf_counter() - started) * 1000
            if resp.status != 200:
                return False, rtt, None
            version = None
            try:
                version = json.loads(raw).get("system", {}).get("comfyui_version")
            except (ValueError, AttributeError):
                version = None
            return True, rtt, version
    except (URLError, OSError):
        return False, (time.perf_counter() - started) * 1000, None


def _route_alive(route: Route) -> tuple[bool, float]:
    if route.tcp is not None:
        return _tcp_open(route.tcp[0], route.tcp[1], min(route.ssh_timeout_s, 3.0))
    return _ssh_ok(route.ssh_alias, route.ssh_timeout_s)


def _read_state(env: dict | None = None) -> dict | None:
    path = state_path(env)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return None
    if not isinstance(doc.get("ts"), (int, float)):
        return None
    if not isinstance(doc.get("boxes"), dict):
        return None
    return doc


def is_remote(box: str, env: dict | None = None) -> bool:
    """True unless the route cache says ``box`` is on a live LAN route (reads the cache, never probes).

    A cache that says every route is down (the picker then reports the canonical ``lan`` row with
    ``alive`` false) counts as remote too. No cache at all is never assumed remote.
    """
    row = ((_read_state(env) or {}).get("boxes") or {}).get(box) or {}
    return bool(row) and not (row.get("route") == "lan" and row.get("alive"))


def _write_state(box: str, route: Route, alive: bool, rtt_ms: float,
                 env: dict | None = None) -> None:
    path = state_path(env)
    doc = _read_state(env) or {"schema": SCHEMA, "boxes": {}}
    doc["ts"] = time.time()
    doc.setdefault("boxes", {})[box] = {
        "route": route.name,
        "ssh_alias": route.ssh_alias,
        "comfy_host": route.comfy_host,
        "alive": alive,
        "rtt_ms": round(rtt_ms, 1),
        "ts": time.time(),
    }
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def pick_route(box: str, *, refresh: bool = False, env: dict | None = None) -> str:
    """Return the name of the first live route for ``box`` (table order =
    preference). Cached in the state file; a stale or missing cache triggers a
    real probe. All-down still returns the canonical first route."""
    return _pick(box, refresh=refresh, env=env).name


def ssh_alias(box: str, *, refresh: bool = False, env: dict | None = None) -> str:
    """The ssh config alias that currently reaches ``box``."""
    return _pick(box, refresh=refresh, env=env).ssh_alias


def comfy_host(box: str, *, refresh: bool = False, env: dict | None = None) -> str:
    """The ComfyUI base URL that currently reaches ``box``."""
    return _pick(box, refresh=refresh, env=env).comfy_host


def proxy_argv(box: str, *, refresh: bool = False, env: dict | None = None) -> list[str]:
    """Argv the ssh ProxyCommand should exec to carry the live route."""
    return list(_pick(box, refresh=refresh, env=env).proxy)


def proxy_command_line(box: str, *, refresh: bool = False,
                       env: dict | None = None) -> str:
    """Shell-quoted ``proxy_argv`` for a ProxyCommand helper to eval/exec."""
    return " ".join(shlex.quote(a) for a in proxy_argv(box, refresh=refresh, env=env))


def _pick(box: str, *, refresh: bool, env: dict | None) -> Route:
    routes = _routes_for(box)
    if not refresh:
        doc = _read_state(env)
        if doc is not None:
            row = (doc.get("boxes") or {}).get(box)
            ts = row.get("ts") if isinstance(row, dict) else None
            if isinstance(ts, (int, float)) and (time.time() - ts) <= _ttl_s(env):
                for route in routes:
                    if route.name == row.get("route"):
                        return route
    return _probe_pick(box, routes, env)


def _probe_pick(box: str, routes: tuple[Route, ...], env: dict | None) -> Route:
    for route in routes:
        ok, rtt = _route_alive(route)
        if ok:
            _write_state(box, route, True, rtt, env)
            return route
    first = routes[0]
    _write_state(box, first, False, 0.0, env)
    return first


def probe_box(box: str) -> dict:
    """Probe every route to ``box`` without touching the cache."""
    rows = []
    for route in _routes_for(box):
        ok, rtt = _route_alive(route)
        http_ok, http_rtt, version = _http_ok(route.comfy_host)
        rows.append({
            "route": route.name,
            "ssh_alias": route.ssh_alias,
            "ssh_ok": ok,
            "ssh_rtt_ms": round(rtt, 1),
            "comfy_host": route.comfy_host,
            "http_ok": http_ok,
            "http_rtt_ms": round(http_rtt, 1),
            "comfyui_version": version,
        })
    return {"box": box, "routes": rows}


def probe_all() -> list[dict]:
    return [probe_box(box) for box in ROUTES]


# --- live ssh forwards ------------------------------------------------------------------
# The launchd forwards (127.0.0.1:181xx -> <box>-auto) pick their transport once, in the
# ProxyCommand, and then stick to it for as long as the ssh process lives. When the best
# route changes (LAN comes back, Tailscale degrades) nothing re-routes them, so `reroute`
# kills the ones on a stale transport and lets launchd KeepAlive respawn them on the pick.

TRANSPORT_MARKERS: tuple[tuple[str, str], ...] = (
    ("cloudflared access", "cf"),
    ("/usr/bin/nc 100.", "ts"),
    ("/usr/bin/nc 192.168.", "lan"),
)


def _ps_rows() -> list[tuple[int, int, str]]:
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def tunnel_routes(box: str, rows: list[tuple[int, int, str]]) -> dict[int, str | None]:
    """pid of every ``<box>-auto`` ssh port-forward -> the route its transport child rides
    (None when the child is missing or unrecognised)."""
    alias = f"{box}-auto"
    found: dict[int, str | None] = {
        pid: None for pid, _, cmd in rows
        if cmd.startswith(SSH_BIN) and " -L " in cmd and cmd.rstrip().endswith(" " + alias)
    }
    for _, ppid, cmd in rows:
        if ppid in found:
            for marker, name in TRANSPORT_MARKERS:
                if marker in cmd:
                    found[ppid] = name
    return found


def stale_tunnels(box: str, rows: list[tuple[int, int, str]], *, refresh: bool = False,
                  env: dict | None = None) -> tuple[str, list[int]]:
    """(picked route, pids of forwards riding a different, known transport)."""
    want = pick_route(box, refresh=refresh, env=env)
    row = ((_read_state(env) or {}).get("boxes") or {}).get(box) or {}
    if not row.get("alive"):  # all-down returns the canonical route; never kill a working forward for it
        return want, []
    stale = [pid for pid, name in tunnel_routes(box, rows).items()
             if name is not None and name != want]
    return want, stale


def reroute(box: str, *, refresh: bool = True, dry_run: bool = False,
            env: dict | None = None) -> dict:
    """Kill the forwards on a stale transport; launchd respawns them on the picked route."""
    want, stale = stale_tunnels(box, _ps_rows(), refresh=refresh, env=env)
    killed, failed = [], []
    for pid in stale:
        if dry_run:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError as exc:
            failed.append((pid, str(exc)))
    return {"box": box, "route": want, "stale": stale, "killed": killed, "failed": failed}
