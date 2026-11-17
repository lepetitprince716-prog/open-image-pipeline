#!/usr/bin/env python3
"""Thin, dependency-free driver helpers shared by the BOXA ComfyUI generation scripts.

This is the single home for the "driver side" primitives — HTTP to ComfyUI, image upload, PNG
validation, adaptive poll backoff, atomic sidecar writes, and asset preflight. It deliberately imports
only the standard library (no recipes, no catalog, no torch), so every gen script — including the locked
showcase line and lora_catalog (which boxa_quality_gen imports, so it cannot import back) — can reuse it
without a dependency cycle or heavy import cost.

None of this affects a generated image: it governs how the driver submits, polls, fetches, and writes.
"""
from __future__ import annotations

import difflib  # noqa: F401 — its only user moved to comfy_preflight; kept so dir() stays frozen
import json
import os
import shlex
import struct
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from loramake.config import NP, POLL_TIMEOUT, ROOT, BOXA_HOST, BOXB_HOST
# Leaf modules (C3). They import stdlib only; every dependency on this module's globals — the
# default cache path, the atomic writer, the /object_info probes — is passed in as a parameter.
# The names below are re-exported as the SAME objects because ~44 files and the
# scripts/boxa_comfy.py alias read them off this module.
from loramake import comfy_enum_cache as _enum_cache
from loramake import comfy_preflight as _preflight
from loramake.comfy_enum_cache import (
    _ENUM_DISK_LOCK, _ENUM_DISK_PRUNE_AGE, _ENUM_DISK_SCHEMA, _ENUM_DISK_TTL_DEFAULT,
    _enum_disk_enabled, _enum_disk_ttl,
)
from loramake.comfy_errors import _execution_error_detail, _submit_error_detail, classify_gen_error
from loramake.comfy_preflight import _suggest_suffix, missing_assets, suggest_assets


class QueueStateError(RuntimeError):
    """A ComfyUI queue response was unavailable or too ambiguous for a safe mutation."""


# Cloudflare Access mode (off-LAN): when COMFY_CF_ACCESS_TOKEN_FILE names a cloudflared
# browser-JWT file (from `cloudflared access login https://<host>`), every curl call goes
# through the system proxy to the Cloudflare edge instead of --noproxy LAN-direct.
_CF_TOKEN_FILE = os.environ.get("COMFY_CF_ACCESS_TOKEN_FILE", "")
# Service-token variant of CF mode (cloud sessions, headless runners): when both are set,
# requests authenticate with CF-Access-Client-Id/Secret instead of a browser JWT. The
# browser-JWT file wins when both are configured — it is the interactive path.
_CF_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
_CF_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
# Cloudflare's browser integrity check 403s (error 1010) bare curl/python user agents;
# a browser-family UA string is required alongside the Access JWT.
_CF_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) loramake-gen/1.0"
# Optional ComfyUI /ws completion wait (Studio / gen.py --run). Poll-only remains the default
# unless COMFY_WS_WAIT=1 or the queue worker exports GEN_EVENTS_OUT / LORAMAKE_ATTEMPT_ID.
# The handshake budget. It was 1.5s and, worse, further clamped to the poll delay
# (0.4s early in a run) — under the Cloudflare loopback's ~2s per request the
# connection could never be established. See _PromptWaiter.
_WS_OPEN_TIMEOUT_S = 5.0
# Transient connect failures worth retrying within one submission before polling.
_WS_CONNECT_ATTEMPTS = 3
_WS_CLOSE_TIMEOUT_S = 0.2
_WS_TERMINAL_TYPES = {"executed", "execution_success", "execution_error"}
_ENUM_CACHE: dict[tuple[str, str, str], tuple[float, frozenset[str]]] = {}
_ENUM_CACHE_LOCK = threading.Lock()
_ENUM_CACHE_TTL = 10.0
_ENUM_CACHE_MAX = 64
# Cross-process L2 for /object_info enums: every gen is a fresh process, so the in-process
# cache above never survives an invocation. The disk layer makes warm preflights free of
# network. Safety: preflight re-probes LIVE before failing on a missing name, so a stale
# entry can delay a rejection by one round-trip but can never wrongly reject a fresh model.
# Schema/TTL/prune-age constants and the file lock live in loramake.comfy_enum_cache and are
# re-exported above; only the L1 provenance set below is driver-local.
# Keys whose last object_info_enum answer came from the disk cache rather than a live
# probe. preflight_assets re-verifies exactly these before rejecting a missing name.
_ENUM_FROM_DISK: set[tuple[str, str, str]] = set()


def _enum_disk_path() -> Path:
    return _enum_cache.enum_disk_path(ROOT / "app" / "data" / "object_info_cache_v1.json")


def _enum_disk_load() -> dict:
    return _enum_cache.enum_disk_load(_enum_disk_path())


def _enum_disk_read(key: tuple[str, str, str]) -> set[str] | None:
    return _enum_cache.enum_disk_read(key, path=_enum_disk_path())


def _enum_disk_write(key: tuple[str, str, str], names: set[str]) -> None:
    """Best-effort read-modify-write with expired-entry pruning; never fails the caller."""
    _enum_cache.enum_disk_write(key, names, path=_enum_disk_path(), write_text=atomic_write_text)


def _is_direct_host(host: str) -> bool:
    """LAN, loopback (ssh forwards) and Tailscale hosts are reached directly, never via a proxy."""
    import ipaddress

    name = host.lower()
    if name == "localhost" or name.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr in ipaddress.ip_network("100.64.0.0/10"))


def _access_argv(host: str) -> list[str]:
    """Per-request transport argv fragment: LAN default bypasses proxies and hits the box
    directly; CF mode sends the Access credentials + UA and stays on the system proxy. The
    browser-JWT token file is re-read on every call so a refreshed `cloudflared access login`
    is picked up without restarting anything (same contract as boxa-comfy-access-proxy.mjs);
    the service-token env pair is the headless/cloud form of the same Access check.
    CF mode applies to tunnel hostnames only: a direct host keeps ``--noproxy`` even when
    credentials are set, so an exported token cannot route LAN traffic through the proxy."""
    if _is_direct_host(host):
        return ["--noproxy", host]
    if _CF_TOKEN_FILE:
        token = Path(_CF_TOKEN_FILE).read_text(encoding="utf-8").strip()
        return ["-H", f"cf-access-token: {token}", "-H", f"User-Agent: {_CF_UA}"]
    if _CF_CLIENT_ID and _CF_CLIENT_SECRET:
        return ["-H", f"CF-Access-Client-Id: {_CF_CLIENT_ID}",
                "-H", f"CF-Access-Client-Secret: {_CF_CLIENT_SECRET}",
                "-H", f"User-Agent: {_CF_UA}"]
    return ["--noproxy", host]


EventSink = Callable[[dict], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit_event(event: str, *, sink: EventSink | None = None, **fields) -> dict:
    """Emit one additive transport observation.

    ``GEN_EVENTS_OUT`` is the process boundary used by the app worker to learn a remote prompt id
    without parsing stdout.  The append is one ``O_APPEND`` write; POSIX mode is forced to 0600
    where supported, while Windows inherits the target directory's ACL.
    A caller-provided sink is best-effort and never changes the legacy transport return value.
    """
    record = {"event": event, "observed_at": _utc_now(), "epoch": time.time(), **fields}
    if sink is not None:
        try:
            sink(dict(record))
        except Exception as exc:
            print(f"  event sink WARN ({type(exc).__name__}: {exc})", file=sys.stderr)
    target = os.environ.get("GEN_EVENTS_OUT")
    if target:
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            print(f"  event file WARN ({type(exc).__name__}: {exc})", file=sys.stderr)
    return record


def _request_json(target: str, endpoint: str, *, payload: dict | None = None,
                  timeout: float = 30.0) -> dict:
    """Proxy-free JSON request with checked HTTP/curl/shape outcomes."""
    base = target.rstrip("/")
    host = urlparse(base).hostname or NP
    argv = ["curl", "-sS", "--fail", *_access_argv(host), "--max-time", str(timeout)]
    raw_payload = None
    if payload is not None:
        argv += ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"]
        raw_payload = json.dumps(payload)
    argv.append(f"{base}{endpoint}")
    try:
        # timeout= backstops a wedged curl process; --max-time only caps the transfer
        proc = subprocess.run(argv, input=raw_payload, capture_output=True, text=True,
                              timeout=timeout + 5)
    except subprocess.TimeoutExpired as exc:
        raise QueueStateError(f"{endpoint} curl hung past {timeout + 5:g}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "no response").strip()[-240:]
        raise QueueStateError(f"{endpoint} failed rc={proc.returncode}: {detail}")
    if not proc.stdout.strip():
        return {}
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise QueueStateError(f"{endpoint} returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise QueueStateError(f"{endpoint} returned {type(body).__name__}, expected object")
    return body


def _queue_prompt_ids(queue: object, key: str) -> set[str]:
    if not isinstance(queue, dict) or not isinstance(queue.get(key), list):
        raise QueueStateError(f"malformed /queue response: missing {key}")
    ids: set[str] = set()
    for entry in queue[key]:
        if isinstance(entry, dict):
            prompt_id = entry.get("prompt_id")
        elif isinstance(entry, (list, tuple)) and len(entry) > 1:
            prompt_id = entry[1]
        else:
            raise QueueStateError(f"malformed /queue {key} entry")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise QueueStateError(f"malformed prompt id in /queue {key}")
        ids.add(prompt_id)
    return ids


def observe_queue(host: str, *, timeout: float = 10.0) -> dict:
    """Return an exact, timestamped queue observation suitable for readiness/recovery decisions."""
    started = time.monotonic()
    body = _request_json(host, "/queue", timeout=timeout)
    running = _queue_prompt_ids(body, "queue_running")
    pending = _queue_prompt_ids(body, "queue_pending")
    return {
        "host": host.rstrip("/"),
        "running": sorted(running),
        "pending": sorted(pending),
        "observed_at": _utc_now(),
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
    }


def cancel_prompt(host: str, prompt_id: str, *, timeout: float = 20.0) -> dict:
    """Cancel only ``prompt_id`` and prove it disappeared.

    Pending work is deleted by id. ComfyUI's global ``/interrupt`` is used only when this prompt is
    the *sole* running prompt; visible foreign running prompts are never interrupted.
    """
    deadline = time.monotonic() + max(0.1, timeout)
    observation = observe_queue(host, timeout=min(10.0, timeout))
    running = set(observation["running"])
    pending = set(observation["pending"])
    action = "absent"
    if prompt_id in pending:
        _request_json(host, "/queue", payload={"delete": [prompt_id]}, timeout=min(10.0, timeout))
        action = "delete_pending"
    if prompt_id in running:
        if running != {prompt_id}:
            return {"ok": False, "action": "unsafe_running", "prompt_id": prompt_id,
                    "observation": observation,
                    "reason": "prompt is not the sole running item; global interrupt refused"}
        _request_json(host, "/interrupt", payload={}, timeout=min(10.0, timeout))
        action = "interrupt_unique_running"

    last = observation
    while True:
        last = observe_queue(host, timeout=min(10.0, max(0.1, deadline - time.monotonic())))
        if prompt_id not in set(last["running"]) | set(last["pending"]):
            return {"ok": True, "action": action, "prompt_id": prompt_id,
                    "observation": last, "reason": ""}
        if time.monotonic() >= deadline:
            return {"ok": False, "action": action, "prompt_id": prompt_id,
                    "observation": last, "reason": "prompt still present after cleanup timeout"}
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))


def __dir__():
    """Keep the legacy boxa_comfy dir() surface while still allowing explicit BOXB_HOST imports."""
    return sorted(n for n in globals() if n != "BOXB_HOST")


def _request_noproxy(args, host: str | None = None) -> str:
    """The hostname a curl call must bypass the ambient proxy for: the explicit ``host``, else the
    first URL in ``args`` (submit / history / view / object_info all carry their box URL), else
    the BOXA default.  Before 2026-09-03 every call bypassed only BOXA's hostname, so a BOXB run under
    an ambient proxy uploaded fine (upload_image derives its own) and then lost every later call."""
    candidate = host
    if not candidate:
        candidate = next((a for a in args if isinstance(a, str) and a.startswith(("http://", "https://"))), None)
    if candidate:
        parsed = urlparse(candidate).hostname
        if parsed:
            return parsed
    return NP


def curl(args, data=None, binary=False, *, host: str | None = None):
    """Run curl and return its stdout.

    Binary fetches (images) use ``--fail`` and return ``b""`` on any non-zero curl exit: a transfer
    cut mid-way (curl 18) or an HTTP error must never hand back partial bytes as an image. Text
    calls keep the historical contract (stdout as-is, error bodies included) because callers parse
    ComfyUI's JSON error bodies, and a cut JSON reply already fails to parse.
    """
    flags = ["-s", "--fail"] if binary else ["-s"]
    try:
        # timeout= backstops a wedged curl process; --max-time only caps the transfer
        proc = subprocess.run(["curl", *flags, *_access_argv(_request_noproxy(args, host)), "--max-time", "900"] + args,
                              input=data, capture_output=True, text=not binary,
                              timeout=905)
    except subprocess.TimeoutExpired:
        return "" if not binary else b""
    if binary and proc.returncode != 0:
        return b""
    return proc.stdout


def upload_image(path: str, host: str | None = None) -> str | None:
    """Upload a control/reference image to a box's ComfyUI input/ dir; return its server name.

    ``host`` selects the box (default BOXA); the no-proxy exemption follows that host.

    LORAMAKE_SKIP_UPLOAD=1 short-circuits the HTTP upload and returns the bare filename —
    for pre-staged files already sitting in the box's input/ dir (e.g. pushed via scp when
    the tunnel chokes on 6MB multipart POSTs).
    """
    import os
    if os.environ.get("LORAMAKE_SKIP_UPLOAD") == "1":
        from pathlib import Path as _P
        return _P(path).name
    host = (host or BOXA_HOST).rstrip("/")
    no_proxy = urlparse(host).hostname or NP
    out = subprocess.run(["curl", "-s", *_access_argv(no_proxy), "--max-time", "120",
                          "-F", f"image=@{path}", "-F", "overwrite=true", f"{host}/upload/image"],
                         capture_output=True, text=True).stdout
    try:
        j = json.loads(out)
        sub = j.get("subfolder") or ""
        return f"{sub}/{j['name']}" if sub else j["name"]
    except Exception as exc:
        print(f"upload_image failed for {path}: {exc!r} raw={out[:200]!r}", file=sys.stderr)
        return None


def png_size(path: str) -> tuple[int, int] | None:
    """Read (width, height) from a PNG's IHDR header, client-side (stdlib only). None if not a PNG."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    return None


def image_size(path: str) -> tuple[int, int] | None:
    """Read (width, height) from a PNG or JPEG header, client-side (stdlib only).

    None for unreadable files and other formats. JPEG dimensions live in the first
    SOF0-SOF15 frame marker (excluding DHT/JPG/DAC markers that share the range).
    """
    size = png_size(path)
    if size is not None:
        return size
    try:
        with open(path, "rb") as fh:
            if fh.read(2) != b"\xff\xd8":  # SOI
                return None
            while True:
                marker = fh.read(2)
                if len(marker) != 2 or marker[0] != 0xFF:
                    return None
                kind = marker[1]
                if kind == 0xD8 or 0xD0 <= kind <= 0xD7:  # standalone, no length
                    continue
                length_raw = fh.read(2)
                if len(length_raw) != 2:
                    return None
                length = struct.unpack(">H", length_raw)[0]
                if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):  # SOF
                    body = fh.read(5)
                    if len(body) != 5:
                        return None
                    h, w = struct.unpack(">HH", body[1:5])
                    return int(w), int(h)
                fh.seek(length - 2, 1)
    except OSError:
        return None


def poll_delay(elapsed: float) -> float:
    """Adaptive poll backoff: tight early (catch a fast job) then relax as the GPU job runs long.

    Early phase is 0.4s so a finished draft is noticed within ~0.4s; the curve is monotonic
    and the tail stays at the historical 5–10s so a long quality job does not hammer /history.
    """
    if elapsed < 25:
        return 0.4
    if elapsed < 90:
        return 1.0
    if elapsed < 300:
        return 5.0
    return 10.0


def _message_kinds(messages) -> list[str]:
    """First elements of /history status.messages entries (e.g. 'execution_cached')."""
    kinds = []
    for entry in messages or []:
        if isinstance(entry, (list, tuple)) and entry:
            kinds.append(str(entry[0]))
    return kinds


# How many recent /history entries a fully cached re-run searches for its graph's earlier run.
_CACHED_LOOKBACK = 200


def _earlier_outputs(pid: str, entry: dict, history: dict,
                     output_keys: Sequence[str]) -> tuple[str, dict] | None:
    """(prompt_id, outputs) of the newest other /history entry that ran the same graph as
    ``entry`` and listed an output under ``output_keys``.

    Re-submitting an identical graph is served entirely from ComfyUI's execution cache, so
    SaveImage writes no new file. ComfyUI 0.33.1 still lists the cached outputs (the files the
    earlier run saved); 0.27.1 drops them from /history for prompts submitted without a
    client_id. The earlier run of the same graph (``prompt[2]``) names the file that holds
    these pixels. ``history`` is a /history listing, oldest first."""
    def graph_of(item: dict) -> dict | None:
        prompt = item.get("prompt")
        return prompt[2] if isinstance(prompt, list) and len(prompt) > 2 else None

    graph = graph_of(entry)
    if graph is None:
        return None
    for other_pid, other in reversed(list(history.items())):
        if other_pid == pid or not isinstance(other, dict) or graph_of(other) != graph:
            continue
        outputs = other.get("outputs") or {}
        if any(node.get(key) for node in outputs.values() for key in output_keys):
            return other_pid, outputs
    return None


def _ws_connect():
    """Return websockets.sync.client.connect, or None when the optional dependency is absent."""
    try:
        from websockets.sync.client import connect
    except Exception:
        return None
    return connect


def _ws_completion_enabled() -> bool:
    setting = os.environ.get("COMFY_WS_WAIT", "").strip().lower()
    if setting in {"1", "true", "yes", "on", "auto"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("GEN_EVENTS_OUT") or os.environ.get("LORAMAKE_ATTEMPT_ID"))


def _ws_target_uri_and_headers(target: str) -> tuple[str | None, dict[str, str]]:
    parsed = urlparse(target.rstrip("/"))
    host = parsed.hostname
    if not host:
        return None, {}
    scheme = "wss" if parsed.scheme == "https" else "ws"
    port = parsed.port or (443 if scheme == "wss" else 80)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    client_id = f"loramake-wait-{uuid.uuid4().hex[:8]}"
    uri = f"{scheme}://{host}:{port}/ws?{urlencode({'clientId': client_id})}"
    headers: dict[str, str] = {}
    if _CF_TOKEN_FILE:
        try:
            token = Path(_CF_TOKEN_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            headers["cf-access-token"] = token
            headers["User-Agent"] = _CF_UA
    return uri, headers


def _read_until_terminal(websocket, prompt_id: str, deadline: float) -> bool:
    """True once a terminal frame for ``prompt_id`` arrives; False if ``deadline`` passes first.

    Frames for other prompts are skipped: ComfyUI broadcasts to every socket when a submission
    carries no client id, which is exactly why this driver does not send one — attaching one
    would make the server unicast to that sid and leave a separately-opened waiter deaf.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            raw = websocket.recv(timeout=remaining)
        except TimeoutError:
            return False
        if isinstance(raw, (bytes, bytearray)):
            continue  # binary latent previews
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(message, dict):
            continue
        data = message.get("data")
        if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
            continue
        if message.get("type") in _WS_TERMINAL_TYPES:
            return True


class _PromptWaiter:
    """One websocket, held for a whole submission, that wakes the poll loop on a terminal frame.

    The previous shape opened a fresh socket on every poll iteration and gave the handshake
    ``min(_WS_OPEN_TIMEOUT_S, poll_delay)`` to finish — and ``poll_delay`` is 0.4s for the first
    25 seconds of a run. Over the Cloudflare loopback a single request costs about two seconds,
    so the handshake could never complete, the first failure latched the fast path off for the
    rest of the run, and the optimisation was dead on precisely the link it exists for.

    So: connect once, keep it, and give the handshake a timeout of its own. A connect that fails
    because there is nothing to connect with (no ``websockets`` module, no usable uri) is
    permanent and never retried; a connect that times out is transient and retried a bounded
    number of times before this run gives up and polls.

    ``/history`` remains the authority for what happened. This only decides when to go look.
    """

    def __init__(self, prompt_id: str, target: str) -> None:
        self._prompt_id = prompt_id
        self._target = target
        self._socket = None
        self._attempts = 0
        self._permanent = False

    def __enter__(self) -> _PromptWaiter:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def _connect(self) -> bool:
        """True when ``self._socket`` is usable. Never raises."""
        if self._socket is not None:
            return True
        if self._permanent or not _ws_completion_enabled():
            return False
        connect = _ws_connect()
        if connect is None:
            self._permanent = True
            return False
        uri, headers = _ws_target_uri_and_headers(self._target)
        if uri is None:
            self._permanent = True
            return False
        if self._attempts >= _WS_CONNECT_ATTEMPTS:
            self._permanent = True
            return False
        self._attempts += 1
        try:
            self._socket = connect(uri, open_timeout=_WS_OPEN_TIMEOUT_S,
                                   close_timeout=_WS_CLOSE_TIMEOUT_S,
                                   additional_headers=headers or None)
        except Exception:
            self._socket = None
            return False
        return True

    def wait(self, timeout: float) -> bool | None:
        """True = terminal frame seen, go re-read /history. False = waited the full timeout, so
        the caller must not sleep again. None = no websocket; the caller owns the sleep."""
        if timeout <= 0 or not self._connect():
            return None
        deadline = time.monotonic() + timeout
        try:
            return _read_until_terminal(self._socket, self._prompt_id, deadline)
        except Exception:
            # The socket died mid-wait. Drop it, spend the rest of the interval so the caller
            # cannot busy-loop, and let the next wait() decide whether to reconnect.
            self.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            return False

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is None:
            return
        try:
            socket.close()
        except Exception:
            pass


def _wait_for_prompt_ws(prompt_id: str, target: str, wait_s: float) -> bool | None:
    """Single-shot form of :class:`_PromptWaiter`, kept for callers outside the poll loop.

    Same contract as before: True = re-read history, False = no extra sleep, None = keep the
    historical poll sleep. Never raises.
    """
    if wait_s <= 0:
        return None
    with _PromptWaiter(prompt_id, target) as waiter:
        return waiter.wait(wait_s)


PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"  # the fixed 12-byte trailer every complete PNG ends with


def looks_like_png(raw: bytes) -> bool:
    """True only for a non-trivial, complete PNG.

    Guards against /view returning an error body or 0 bytes, and against a transfer cut mid-way:
    a complete PNG ends with the IEND chunk (zero length, type IEND, fixed CRC).
    """
    return len(raw) > 1024 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[-12:] == PNG_IEND


def looks_like_webp(raw: bytes) -> bool:
    """True only for a non-trivial, complete WebP (RIFF....WEBP), the format of ComfyUI's /view previews.

    The RIFF size field (bytes 4..8, little-endian) counts everything after the first 8 bytes, so a
    truncated transfer shows up as a size that does not match the byte count.
    """
    return (len(raw) > 512 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
            and int.from_bytes(raw[4:8], "little") + 8 == len(raw))


PREVIEW_VIEW = "webp;80"  # ComfyUI /view re-encodes server-side: ~1/15 of the PNG over the wire


def _preview_fetch_box() -> str | None:
    """Box named by LORAMAKE_FETCH_PREVIEW (set by gen.py off the LAN), or None when the full PNG
    should cross the link. Needs Pillow to turn the preview back into a PNG; without it, stay PNG."""
    box = os.environ.get("LORAMAKE_FETCH_PREVIEW", "").strip().lower()
    if box in {"", "0", "false", "no", "off"}:
        return None
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        print(f"  NOTE: preview fetch requested for {box} but Pillow is missing; fetching full PNGs",
              file=sys.stderr)
        return None
    return box


def _webp_to_png(raw: bytes) -> bytes:
    """Lossless PNG re-encode of a fetched webp preview (the one lossy step already happened)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).save(buf, format="PNG")
    return buf.getvalue()


def submit_and_fetch(
    graph: dict,
    *,
    label: str = "",
    host: str | None = None,
    timeout: float | None = None,
    delay=None,
    output_keys: tuple = ("images",),
    validate=None,
    first_only: bool = True,
    event_sink: EventSink | None = None,
    view_params: dict[str, str] | None = None,
    allow_preview: bool = False,
) -> list[tuple[bytes, dict]] | None:
    """Submit a graph, poll /history, and fetch ComfyUI outputs.

    ``view_params`` are extra /view query params, e.g. ``{"preview": "webp;80"}`` to fetch ComfyUI's
    server-side webp preview instead of the full PNG (the caller's ``validate`` must accept it).

    ``allow_preview``: the caller records ``remote_original`` for items marked ``fetched_preview``,
    so off the LAN (``LORAMAKE_FETCH_PREVIEW``) only the webp preview is fetched. Callers that do
    not record it always get the full PNG, since otherwise the original would be unrecoverable.

    Returns ``[(raw, history_item), ...]`` on success, or ``None`` for submit failure, node error,
    timeout, or completion without a valid fetched output. Saving remains the caller's responsibility.

    A fully cached re-run (``execution_cached``, success, empty outputs; ComfyUI 0.27.1 without a
    client_id) is fetched from the newest earlier /history entry that ran the same graph. A local
    file cannot stand in for it: an output's name does not identify the graph that made it. With
    no such entry in the last ``_CACHED_LOOKBACK`` items, or none of its files left on the box,
    it emits ``prompt_error`` (``cached_without_output``) and returns ``None`` at once.
    """
    target = (host or BOXA_HOST).rstrip("/")
    budget = POLL_TIMEOUT if timeout is None else timeout
    delay_fn = poll_delay if delay is None else delay
    validate_fn = looks_like_png if validate is None else validate
    preview_box = (_preview_fetch_box()
                   if allow_preview and view_params is None and validate is None else None)
    if preview_box:
        view_params, validate_fn = {"preview": PREVIEW_VIEW}, looks_like_webp

    out = curl(["-X", "POST", f"{target}/prompt", "-H", "Content-Type: application/json",
                "--data-binary", "@-"], data=json.dumps({"prompt": graph}))
    try:
        pid = json.loads(out)["prompt_id"]
    except Exception as exc:
        detail = _submit_error_detail(out)
        if detail:
            print(f"  submit REJECTED [{detail['error_class']}] "
                  f"{detail['exception_type']}: {detail['exception_message']}", file=sys.stderr)
        else:
            print(f"  submit FAIL ({type(exc).__name__}: {exc}): {out[:400]}", file=sys.stderr)
        emit_event("prompt_submit_failed", sink=event_sink, host=target, label=label,
                   **(detail or {"error_class": type(exc).__name__}))
        return None
    print(f"  [{label}] pid={pid} ...", flush=True)
    emit_event("prompt_submitted", sink=event_sink, host=target, label=label, prompt_id=pid)

    # One websocket for the whole submission; see _PromptWaiter for why it is not
    # reopened per poll. /history stays the authority — this only wakes the loop.
    with _PromptWaiter(pid, target) as waiter:
        start = time.monotonic()
        consecutive_errs = 0

        def cached_failure(reason: str) -> None:
            # The entry is final: polling it again would only wait out the budget.
            message = (f"prompt {pid} completed from ComfyUI's execution cache without listing an "
                       f"output, and {reason}; re-run with a changed input such as the seed")
            print(f"    ERROR [cached_without_output]: {message}")
            emit_event("prompt_error", sink=event_sink, host=target, label=label, prompt_id=pid,
                       elapsed_s=round(time.monotonic() - start, 3), cached=True,
                       error_class="cached_without_output", exception_message=message)
            return None

        while time.monotonic() - start < budget:
            try:
                hj = json.loads(curl([f"{target}/history/{pid}"]) or "{}")
                consecutive_errs = 0
            except Exception as exc:
                consecutive_errs += 1
                if consecutive_errs >= 5 and consecutive_errs % 5 == 0:
                    print(f"    WARN: {consecutive_errs} consecutive /history errors (last: {exc!r}) "
                          "— box may be unreachable")
                time.sleep(delay_fn(time.monotonic() - start))
                continue

            entry = hj.get(pid) or {}
            status = entry.get("status") or {}
            outputs = entry.get("outputs")
            cached_fields: dict = {}
            if (not outputs and status.get("completed") and status.get("status_str") == "success"
                    and "execution_cached" in _message_kinds(status.get("messages"))):
                try:
                    listing = json.loads(curl([f"{target}/history?max_items={_CACHED_LOOKBACK}"]))
                except ValueError:  # "" from a dropped connection: poll again
                    time.sleep(delay_fn(time.monotonic() - start))
                    continue
                if not isinstance(listing, dict):
                    listing = {}
                earlier = _earlier_outputs(pid, entry, listing, output_keys)
                if earlier is None:
                    return cached_failure(
                        f"no earlier run of this graph is in the last {len(listing)} /history entries")
                earlier_pid, outputs = earlier
                cached_fields = {"cached": True}
                print(f"    CACHED: fully-cached completion -> fetching the output(s) of earlier "
                      f"prompt {earlier_pid}")

            if outputs:
                fetched: list[tuple[bytes, dict]] = []
                for node in outputs.values():
                    for key in output_keys:
                        for im in node.get(key, []):
                            extra = [a for k, v in (view_params or {}).items() for a in ("--data-urlencode", f"{k}={v}")]
                            raw = curl(["-G", f"{target}/view",
                                        "--data-urlencode", f"filename={im['filename']}",
                                        "--data-urlencode", f"subfolder={im.get('subfolder', '')}",
                                        "--data-urlencode", f"type={im.get('type', 'output')}", *extra], binary=True)
                            ok = validate_fn(raw)
                            if ok and preview_box:
                                # Only the small preview crossed the link; keep downstream PNG-only.
                                raw = _webp_to_png(raw)
                                im = {**im, "fetched_preview": PREVIEW_VIEW, "box": preview_box}
                                ok = looks_like_png(raw)
                            if ok:
                                fetched.append((raw, im))
                                if first_only:
                                    emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                                               prompt_id=pid,
                                               elapsed_s=round(time.monotonic() - start, 3), outputs=1,
                                               **cached_fields)
                                    return fetched
                            else:
                                print(f"    SKIP: /view returned {len(raw)} bytes (failed validation)")
                if cached_fields and not fetched:
                    return cached_failure(f"none of the outputs its earlier run {earlier_pid} listed "
                                          "is still on the box")
                emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                           prompt_id=pid, elapsed_s=round(time.monotonic() - start, 3),
                           outputs=len(fetched), **cached_fields)
                return fetched or None

            if hj.get(pid, {}).get("status", {}).get("status_str", "") == "error":
                detail = _execution_error_detail(hj[pid])
                if detail:
                    print(f"    ERROR [{detail['error_class']}] at node {detail['node_id']} "
                          f"({detail['node_type']}): {detail['exception_type']}: "
                          f"{detail['exception_message']}")
                else:
                    print("    ERROR (check ComfyUI log — possible OOM / bad node wiring)")
                emit_event("prompt_error", sink=event_sink, host=target, label=label, prompt_id=pid,
                           elapsed_s=round(time.monotonic() - start, 3), **(detail or {}))
                return None

            delay_s = delay_fn(time.monotonic() - start)
            ws_outcome = waiter.wait(delay_s)
            if ws_outcome is True:
                continue
            if ws_outcome is None:
                time.sleep(delay_s)

        elapsed = time.monotonic() - start
        print(f"    TIMEOUT after {int(elapsed)}s for pid={pid} (may still be running)")
        try:
            cancel = cancel_prompt(target, pid)
        except Exception as exc:
            cancel = {"ok": False, "action": "cancel_failed", "prompt_id": pid,
                      "reason": f"{type(exc).__name__}: {exc}"}
        action = cancel.get("action") or "cancel_failed"
        reason = cancel.get("reason") or ""
        print(f"    cancel pid={pid} action={action} ok={cancel.get('ok')}"
              + (f" ({reason})" if reason else ""), file=sys.stderr)
        emit_event("prompt_timeout", sink=event_sink, host=target, label=label, prompt_id=pid,
                   elapsed_s=round(elapsed, 3),
                   cancel_ok=bool(cancel.get("ok")),
                   cancel_action=str(action))
        return None


# ComfyUI v0.33 routes the two /free flags very differently (main.py): ``unload_models`` drops
# the models from VRAM, while ``free_memory`` additionally calls the executor's ``reset()`` and
# tears down the RAM_PRESSURE cache. The 2026-09-01 BOXA run recorded that teardown wedging the
# service (docs/handoffs/2026-09-01-comfyui-perf-opt/), and it also discards the warm cache that
# makes a same-seed re-run near-instant. "vram" hands the GPU back to the user's game without
# either cost; "full" is the historical payload, for when the cache genuinely must be cleared.
FREE_MODES = {
    "vram": {"unload_models": True, "free_memory": False},
    "full": {"unload_models": True, "free_memory": True},
}
FREE_MODE_DEFAULT = "vram"


def free_payload(mode: str | None = None) -> dict[str, bool]:
    """The POST /free body for ``mode``, else ``LORAMAKE_FREE_MODE``, else the default.

    An unrecognised value falls back to the default instead of raising: a typo in an env var
    must not fail a finished render at the VRAM-release step.
    """
    chosen = (mode if mode is not None else os.environ.get("LORAMAKE_FREE_MODE", "")).strip().lower()
    return dict(FREE_MODES.get(chosen, FREE_MODES[FREE_MODE_DEFAULT]))


def free_vram(host: str | None = None, *, clear_cache: bool = False) -> bool:
    """Best-effort POST /free; true only for curl success plus an observed HTTP 2xx status.

    ``clear_cache`` sends the "full" body regardless of the env: only the executor reset undoes
    nodes that patch the cached model in place (Anima IP-Adapter, verified on BOXA 2026-10-01).
    """
    try:
        target = (host or BOXA_HOST).rstrip("/")
        no_proxy = urlparse(target).hostname or NP
        payload = json.dumps(free_payload("full" if clear_cache else None))
        proc = subprocess.run([
            "curl", "-sS", "--fail", *_access_argv(no_proxy), "--max-time", "30",
            "--output", os.devnull, "--write-out", "%{http_code}",
            "-X", "POST", f"{target}/free", "-H", "Content-Type: application/json",
            "--data-binary", "@-",
        ], input=payload, capture_output=True, text=True)
        status = int(proc.stdout.strip()) if proc.stdout.strip().isdigit() else 0
        return proc.returncode == 0 and 200 <= status < 300
    except Exception:
        return False


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp sibling + atomic rename so a crash/disk-full cannot leave a partial file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def report_generated_file(path: str | Path) -> None:
    """Append one saved image to the per-run ``GEN_FILES_OUT`` manifest, best-effort."""
    files_out = os.environ.get("GEN_FILES_OUT")
    if not files_out:
        return
    line = (str(Path(path).resolve()) + "\n").encode("utf-8")
    try:
        fd = os.open(files_out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError:
        pass


# --- post_fx finishing pass ---------------------------------------------------------------
# The image-space finishing pass (scripts/post_fx.py: colour grade / halation / vignette /
# grain) needs PIL+numpy, which this stdlib-only module must not import. It runs as a
# subprocess against the generated PNG and writes a `<stem>_fx.png` sibling — the raw
# engine output is never touched, and a finishing-pass failure must never block a
# successful generation.

POST_FX_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "post_fx.py"
POST_FX_DEFAULT = "film"
_POST_FX_OFF = (None, "", "none", "off", "false", "0")


def resolve_post_fx_python() -> str | None:
    """Interpreter with PIL/numpy for post_fx.py: env override, then the project venv, then self."""
    repo_venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    for candidate in (
        os.environ.get("POST_FX_PYTHON"),
        str(repo_venv),
        sys.executable,
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def apply_post_fx(src: str | Path, preset: str, *, overrides: str = "",
                  timeout_s: int = 180) -> str | None:
    """Render the post_fx finishing pass next to `src` as `<stem>_fx.png`.

    ``overrides`` carries per-effect CLI overrides ("--grain 0.0 --vignette 0.15") that
    are appended after the preset. Returns the output path on success, or None on any
    failure (missing interpreter, bad preset, non-zero exit) — callers must treat None
    as best-effort, never as fatal.
    """
    if preset in _POST_FX_OFF:
        return None
    py = resolve_post_fx_python()
    if py is None or not POST_FX_SCRIPT.is_file():
        print(f"[post-fx] unavailable (no PIL interpreter / no script); skipping {Path(src).name}",
              file=sys.stderr)
        return None
    cmd = [py, str(POST_FX_SCRIPT), str(src), "--preset", str(preset)]
    if overrides:
        cmd += shlex.split(overrides)
    cmd.append("--overwrite")  # last, so card overrides can never drop it
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[post-fx] {Path(src).name}: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"[post-fx] {Path(src).name}: {proc.stderr.strip() or 'exit ' + str(proc.returncode)}",
              file=sys.stderr)
        return None
    out = proc.stdout.strip() or str(Path(src).with_name(f"{Path(src).stem}_fx.png"))
    print(f"[post-fx] {Path(src).name} -> {Path(out).name} ({preset})")
    return out


def _atomic_publish_no_clobber(path: Path, raw: bytes) -> tuple[int, int]:
    """Atomically publish complete bytes at an unoccupied path, returning its inode identity."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
            identity = (os.fstat(fh.fileno()).st_dev, os.fstat(fh.fileno()).st_ino)
        os.link(tmp, path)  # hard-link publication is atomic and fails if the final path already exists
        return identity
    finally:
        tmp.unlink(missing_ok=True)


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the artifact published by this call, never a later replacement."""
    try:
        stat = path.stat(follow_symlinks=False)
        if (stat.st_dev, stat.st_ino) == identity:
            path.unlink()
    except OSError:
        pass


def _png_add_itxt(raw: bytes, keyword: str, text: str) -> bytes:
    """Return `raw` PNG with one uncompressed iTXt chunk (UTF-8, so CJK prompts survive) inserted
    before IEND. iTXt layout: keyword\\0 compflag\\0 compmethod\\0 lang\\0 transkw\\0 text."""
    body = (keyword.encode("latin1")[:79] + b"\x00\x00\x00" + b"\x00" + b"\x00"
            + text.encode("utf-8"))
    crc = zlib.crc32(b"iTXt" + body) & 0xFFFFFFFF
    chunk = struct.pack(">I", len(body)) + b"iTXt" + body + struct.pack(">I", crc)
    pos = raw.rfind(b"IEND") - 4  # the 4-byte length field precedes the IEND type
    if pos < 8:
        raise ValueError("no IEND chunk")
    return raw[:pos] + chunk + raw[pos:]


def save_image_with_meta(path: Path, raw: bytes, meta: dict, *, sidecar: bool = True,
                         key: str = "loramake", overwrite: bool = True) -> None:
    """Write a generated PNG with the loramake curated metadata embedded as a UTF-8 iTXt chunk, so the
    params travel WITH the image even if the .meta.json sidecar is moved/lost (ComfyUI already embeds
    the full API graph in a separate 'prompt' chunk; this adds the curated loramake layer on top). By
    default still writes the .meta.json sidecar for back-compat. Best-effort embed: on any PNG-parse
    problem it falls back to writing the raw bytes unchanged (never loses the image)."""
    path = Path(path)
    try:
        out = _png_add_itxt(raw, key, json.dumps(meta, ensure_ascii=False)) if looks_like_png(raw) else raw
    except Exception:
        out = raw
    sidecar_path = Path(str(path) + ".meta.json")
    if overwrite:
        path.write_bytes(out)
        if sidecar:
            atomic_write_text(sidecar_path, json.dumps(meta, ensure_ascii=False, indent=2))
    else:
        image_identity = _atomic_publish_no_clobber(path, out)
        try:
            if sidecar:
                _atomic_publish_no_clobber(
                    sidecar_path,
                    json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                )
        except Exception:
            _unlink_if_identity(path, image_identity)
            raise
    # GEN_FILES_OUT protocol: when a driver (gen.py --run) exports this path, report every file
    # THIS process saved — a concurrent run writing the same outputs tree then cannot leak into
    # the driver's result (its directory-snapshot diff is only the fallback). Append-only, one
    # absolute path per line; reporting must never break the save itself.
    report_generated_file(path)


def object_info_enum(node_type: str, field: str, host: str | None = None, *,
                     refresh: bool = False) -> set[str]:
    """Filenames a ComfyUI loader node accepts (e.g. LoraLoader.lora_name). Empty set on any failure.

    ComfyUI serializes a combo field two ways across versions/nodes: the old `[[name, ...], {meta}]`
    (head is the name list) and the newer `["COMBO", {"options": [name, ...]}]`. Handle both; any
    other shape returns empty (safer than the char-set a bare `set(head)` would yield).

    Successful results are memoized per process for a short TTL, and persisted to a disk cache
    (LORAMAKE_ENUM_CACHE / _TTL / _PATH) so repeated CLI invocations skip the probe entirely.
    ``refresh=True`` bypasses both read paths (still writes through) — the preflight failure
    path uses it to re-verify a miss live before rejecting a run. Failures are never cached.
    """
    box = (host or BOXA_HOST).rstrip("/")
    key = (box, node_type, field)
    now = time.monotonic()
    if not refresh:
        with _ENUM_CACHE_LOCK:
            hit = _ENUM_CACHE.get(key)
            if hit is not None and now - hit[0] < _ENUM_CACHE_TTL:
                return set(hit[1])
        disk = _enum_disk_read(key)
        if disk is not None:
            with _ENUM_CACHE_LOCK:
                _ENUM_CACHE[key] = (time.monotonic(), frozenset(disk))
                _ENUM_FROM_DISK.add(key)
            return disk
    try:
        info = json.loads(curl([f"{box}/object_info/{node_type}"]) or "{}")
        spec = info[node_type]["input"]["required"][field]
        head = spec[0]
        if isinstance(head, list):
            result = set(head)
        elif head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            result = set(spec[1].get("options", []))
        else:
            return set()
    except Exception:
        return set()
    with _ENUM_CACHE_LOCK:
        _ENUM_CACHE[key] = (time.monotonic(), frozenset(result))
        _ENUM_FROM_DISK.discard(key)
        while len(_ENUM_CACHE) > _ENUM_CACHE_MAX:
            _ENUM_CACHE.pop(next(iter(_ENUM_CACHE)))
    _enum_disk_write(key, result)
    return result


def node_present(node_type: str, host: str | None = None) -> bool | None:
    """Whether the box registers a node class: True / False, or None when /object_info is unreachable.

    ComfyUI answers ``/object_info/<node>`` with ``{}`` for an unknown class, so an empty object is a
    definite "absent" while an empty or unparseable body is a transport failure. Callers that pick
    between alternative node packs (core vs custom) treat None as "assume the first choice and warn":
    the graph submission itself is the final arbiter and fails with the node name in the error.
    """
    box = (host or BOXA_HOST).rstrip("/")
    raw = curl([f"{box}/object_info/{node_type}"])
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    return isinstance(info, dict) and node_type in info


def object_info_enum_many(
    requests: Sequence[tuple[str, str]], host: str | None = None
) -> dict[tuple[str, str], set[str]]:
    """Parallel ``object_info_enum`` batch. Individual failures degrade to empty sets."""
    requests = list(dict.fromkeys(requests))
    if not requests:
        return {}
    with ThreadPoolExecutor(max_workers=min(6, len(requests))) as pool:
        return dict(pool.map(lambda req: (req, object_info_enum(req[0], req[1], host=host)), requests))


def preflight_assets(arch: str, ckpt: str, lora_names: list[str], *,
                     controlnets: "Sequence[str]" = (), upscalers: "Sequence[str]" = (),
                     clips: "Sequence[str]" = (), vaes: "Sequence[str]" = (),
                     model_patches: "Sequence[str]" = (),
                     host: str | None = None, lora_enums: set[str] | None = None) -> None:
    """Reject a run BEFORE any GPU work if the checkpoint or a LoRA name is absent on the box, turning a
    wasted GPU job + cryptic node error into an instant, clear message. Degrades to a no-op when the
    /object_info probe is unreachable, so it never blocks an otherwise-valid run.

    Optional loader-specific lists extend the same guard without changing existing callers:
    `controlnets` -> ControlNetLoader, `upscalers` -> UpscaleModelLoader, `clips` -> CLIPLoader, and
    `vaes` -> VAELoader, `model_patches` -> ModelPatchLoader (models/model_patches/, the Anima
    ControlNet-LLLite weights). Missing fixed recipe assets are hard stops with the exact loader
    field reported.
    `host` selects the box to probe (default BOXA).

    A miss answered from the disk enum cache is re-verified with one live probe before the
    hard stop, so a model added inside the cache TTL is never wrongly rejected. Rejections
    name the closest known loader entries when any are close.

    The decision itself lives in loramake.comfy_preflight; the three probe seams are bound here,
    per call, against this module's globals — so a `comfy.object_info_enum` monkeypatch still
    reaches them and the live re-verify keeps its mandatory refresh=True.
    """
    _preflight.preflight_assets(
        arch, ckpt, lora_names,
        controlnets=controlnets, upscalers=upscalers, clips=clips, vaes=vaes,
        model_patches=model_patches, lora_enums=lora_enums,
        box=(host or BOXA_HOST).rstrip("/"),
        enum_many=lambda probes, box: object_info_enum_many(probes, host=box),
        enum_live=lambda node, field, box: object_info_enum(node, field, host=box, refresh=True),
        answered_from_disk=lambda key: key in _ENUM_FROM_DISK,
    )
