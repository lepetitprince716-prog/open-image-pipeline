#!/usr/bin/env python3
"""Thin, dependency-free driver helpers shared by the Z3 ComfyUI generation scripts.

This is the single home for the "driver side" primitives — HTTP to ComfyUI, image upload, PNG
validation, adaptive poll backoff, atomic sidecar writes, and asset preflight. It deliberately imports
only the standard library (no recipes, no catalog, no torch), so every gen script — including the locked
showcase line and lora_catalog (which z3_quality_gen imports, so it cannot import back) — can reuse it
without a dependency cycle or heavy import cost.

None of this affects a generated image: it governs how the driver submits, polls, fetches, and writes.
"""
from __future__ import annotations

import difflib
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
from loramake.config import NP, POLL_TIMEOUT, ROOT, Z3_HOST, ZZZ_HOST


class QueueStateError(RuntimeError):
    """A ComfyUI queue response was unavailable or too ambiguous for a safe mutation."""


# Cloudflare Access mode (off-LAN): when COMFY_CF_ACCESS_TOKEN_FILE names a cloudflared
# browser-JWT file (from `cloudflared access login https://<host>`), every curl call goes
# through the system proxy to the Cloudflare edge instead of --noproxy LAN-direct.
_CF_TOKEN_FILE = os.environ.get("COMFY_CF_ACCESS_TOKEN_FILE", "")
# Cloudflare's browser integrity check 403s (error 1010) bare curl/python user agents;
# a browser-family UA string is required alongside the Access JWT.
_CF_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) loramake-gen/1.0"
# Optional ComfyUI /ws completion wait (Studio / gen.py --run). Poll-only remains the default
# unless COMFY_WS_WAIT=1 or the queue worker exports GEN_EVENTS_OUT / LORAMAKE_ATTEMPT_ID.
_WS_OPEN_TIMEOUT_S = 1.5
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
_ENUM_DISK_SCHEMA = "loramake.object-info-cache/v1"
_ENUM_DISK_TTL_DEFAULT = 900.0
_ENUM_DISK_PRUNE_AGE = 7 * 24 * 3600.0
_ENUM_DISK_LOCK = threading.Lock()
# Keys whose last object_info_enum answer came from the disk cache rather than a live
# probe. preflight_assets re-verifies exactly these before rejecting a missing name.
_ENUM_FROM_DISK: set[tuple[str, str, str]] = set()


def _enum_disk_enabled() -> bool:
    return os.environ.get("LORAMAKE_ENUM_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _enum_disk_ttl() -> float:
    try:
        return float(os.environ.get("LORAMAKE_ENUM_CACHE_TTL", str(_ENUM_DISK_TTL_DEFAULT)))
    except ValueError:
        return _ENUM_DISK_TTL_DEFAULT


def _enum_disk_path() -> Path:
    override = os.environ.get("LORAMAKE_ENUM_CACHE_PATH", "").strip()
    return Path(override).expanduser() if override else ROOT / "app" / "data" / "object_info_cache_v1.json"


def _enum_disk_load() -> dict:
    try:
        data = json.loads(_enum_disk_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != _ENUM_DISK_SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _enum_disk_read(key: tuple[str, str, str]) -> set[str] | None:
    if not _enum_disk_enabled():
        return None
    entry = _enum_disk_load().get("|".join(key))
    if not isinstance(entry, dict):
        return None
    names, observed_at = entry.get("names"), entry.get("observed_at")
    if not isinstance(names, list) or not isinstance(observed_at, (int, float)):
        return None
    if not (0.0 <= time.time() - observed_at <= _enum_disk_ttl()):
        return None
    return {n for n in names if isinstance(n, str)}


def _enum_disk_write(key: tuple[str, str, str], names: set[str]) -> None:
    """Best-effort read-modify-write with expired-entry pruning; never fails the caller."""
    if not _enum_disk_enabled():
        return
    path = _enum_disk_path()
    now = time.time()
    with _ENUM_DISK_LOCK:
        entries = _enum_disk_load()
        entries = {k: v for k, v in entries.items()
                   if isinstance(v, dict) and isinstance(v.get("observed_at"), (int, float))
                   and now - v["observed_at"] < _ENUM_DISK_PRUNE_AGE}
        entries["|".join(key)] = {"names": sorted(names), "observed_at": now}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(
                {"schema": _ENUM_DISK_SCHEMA, "entries": entries}, ensure_ascii=False))
            os.chmod(path, 0o600)
        except OSError:
            pass


def _access_argv(host: str) -> list[str]:
    """Per-request transport argv fragment: LAN default bypasses proxies and hits the box
    directly; CF mode sends the Access JWT + UA and stays on the system proxy. The token
    file is re-read on every call so a refreshed `cloudflared access login` is picked up
    without restarting anything (same contract as z3-comfy-access-proxy.mjs)."""
    if _CF_TOKEN_FILE:
        token = Path(_CF_TOKEN_FILE).read_text(encoding="utf-8").strip()
        return ["-H", f"cf-access-token: {token}", "-H", f"User-Agent: {_CF_UA}"]
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


def classify_gen_error(exception_type: str, message: str) -> str:
    """Coarse triage bucket for a ComfyUI failure: oom | model_missing | node_missing | other.

    Buckets match what the operator does next (free VRAM / fix the model name / install the
    node), not ComfyUI's exception taxonomy — keep them few and stable."""
    blob = f"{exception_type} {message}".lower()
    if "out of memory" in blob or "outofmemory" in blob or "cuda error" in blob:
        return "oom"
    if ("not in list" in blob or "value_not_in_list" in blob or "no such file" in blob
            or "safetensors" in blob and "not found" in blob):
        return "model_missing"
    if "does not exist" in blob or "invalid_prompt" in blob or "unknown node" in blob:
        return "node_missing"
    return "other"


def _execution_error_detail(item: dict) -> dict | None:
    """The execution_error payload from a /history item's status.messages, reduced to the
    fields an operator acts on. None when the item carries no parseable error message."""
    messages = (item.get("status") or {}).get("messages") or []
    for entry in messages:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        kind, payload = entry
        if kind != "execution_error" or not isinstance(payload, dict):
            continue
        exception_type = str(payload.get("exception_type") or "")
        exception_message = str(payload.get("exception_message") or "").strip()
        traceback_lines = payload.get("traceback") or []
        detail = {
            "node_id": str(payload.get("node_id") or ""),
            "node_type": str(payload.get("node_type") or ""),
            "exception_type": exception_type,
            "exception_message": exception_message[:1000],
            "error_class": classify_gen_error(exception_type, exception_message),
        }
        if isinstance(traceback_lines, list) and traceback_lines:
            detail["traceback_tail"] = "".join(str(ln) for ln in traceback_lines[-3:])[-1000:]
        return detail
    return None


def _submit_error_detail(body: str) -> dict | None:
    """Structured detail from a rejected POST /prompt response (ComfyUI returns
    {"error": {...}, "node_errors": {...}} on validation failure)."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
    node_errors = parsed.get("node_errors") if isinstance(parsed.get("node_errors"), dict) else {}
    message = str(error.get("message") or "")
    if not message and not node_errors:
        return None
    parts = []
    for node_id, node_err in list(node_errors.items())[:3]:
        if not isinstance(node_err, dict):
            continue
        node_type = str(node_err.get("class_type") or "")
        first = next(iter(node_err.get("errors") or []), None)
        first_msg = str(first.get("message") or "") if isinstance(first, dict) else ""
        parts.append(f"node {node_id} ({node_type}): {first_msg}".strip())
    summary = "; ".join(p for p in parts if p)
    exception_message = "; ".join(p for p in (message, summary) if p)[:1000]
    return {
        "exception_type": str(error.get("type") or "validation"),
        "exception_message": exception_message,
        "error_class": classify_gen_error(str(error.get("type") or ""), exception_message),
    }


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
    """Keep the legacy z3_comfy dir() surface while still allowing explicit ZZZ_HOST imports."""
    return sorted(n for n in globals() if n != "ZZZ_HOST")


def curl(args, data=None, binary=False):
    try:
        # timeout= backstops a wedged curl process; --max-time only caps the transfer
        return subprocess.run(["curl", "-s", *_access_argv(NP), "--max-time", "900"] + args,
                              input=data, capture_output=True, text=not binary,
                              timeout=905).stdout
    except subprocess.TimeoutExpired:
        return "" if not binary else b""


def upload_image(path: str) -> str | None:
    """Upload a control/reference image to the Z3 ComfyUI input/ dir; return its server name.

    LORAMAKE_SKIP_UPLOAD=1 short-circuits the HTTP upload and returns the bare filename —
    for pre-staged files already sitting in the box's input/ dir (e.g. pushed via scp when
    the tunnel chokes on 6MB multipart POSTs).
    """
    import os
    if os.environ.get("LORAMAKE_SKIP_UPLOAD") == "1":
        from pathlib import Path as _P
        return _P(path).name
    host = Z3_HOST.rstrip("/")
    out = subprocess.run(["curl", "-s", *_access_argv(NP), "--max-time", "120",
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


def _wait_for_prompt_ws(prompt_id: str, target: str, wait_s: float) -> bool | None:
    """Wait up to wait_s for a terminal /ws event. True=re-read history; False=no extra sleep;
    None=keep historical poll sleep. Never raises."""
    if wait_s <= 0 or not _ws_completion_enabled():
        return None
    connect = _ws_connect()
    if connect is None:
        return None
    uri, headers = _ws_target_uri_and_headers(target)
    if uri is None:
        return None
    open_timeout = min(_WS_OPEN_TIMEOUT_S, max(0.05, wait_s))
    try:
        websocket = connect(
            uri,
            open_timeout=open_timeout,
            close_timeout=_WS_CLOSE_TIMEOUT_S,
            additional_headers=headers or None,
        )
    except Exception:
        return None
    deadline = time.monotonic() + wait_s
    try:
        with websocket:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    raw = websocket.recv(timeout=remaining)
                except TimeoutError:
                    return False
                if isinstance(raw, (bytes, bytearray)):
                    continue
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
    except Exception:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return False
    return False


def looks_like_png(raw: bytes) -> bool:
    """True only for a non-trivial PNG — guards against /view returning an error body or 0 bytes."""
    return len(raw) > 1024 and raw[:8] == b"\x89PNG\r\n\x1a\n"


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
    cached_paths: Sequence[str] | None = None,
) -> list[tuple[bytes, dict]] | None:
    """Submit a graph, poll /history, and fetch ComfyUI outputs.

    Returns ``[(raw, history_item), ...]`` on success, or ``None`` for submit failure, node error,
    timeout, or completion without a valid fetched output. Saving remains the caller's responsibility.

    ``cached_paths``: local deterministic outputs for a fully-cached ComfyUI re-execution
    (``execution_cached`` + empty outputs). Without them the driver fails fast instead of
    hanging until the poll budget.
    """
    target = (host or Z3_HOST).rstrip("/")
    budget = POLL_TIMEOUT if timeout is None else timeout
    delay_fn = poll_delay if delay is None else delay
    validate_fn = looks_like_png if validate is None else validate

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

    start = time.monotonic()
    consecutive_errs = 0
    ws_unavailable = False
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

        if pid in hj and hj[pid].get("outputs"):
            fetched: list[tuple[bytes, dict]] = []
            for node in hj[pid]["outputs"].values():
                for key in output_keys:
                    for im in node.get(key, []):
                        raw = curl(["-G", f"{target}/view",
                                    "--data-urlencode", f"filename={im['filename']}",
                                    "--data-urlencode", f"subfolder={im.get('subfolder', '')}",
                                    "--data-urlencode", f"type={im.get('type', 'output')}"], binary=True)
                        if validate_fn(raw):
                            fetched.append((raw, im))
                            if first_only:
                                emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                                           prompt_id=pid,
                                           elapsed_s=round(time.monotonic() - start, 3), outputs=1)
                                return fetched
                        else:
                            print(f"    SKIP: /view returned {len(raw)} bytes (failed validation)")
            emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                       prompt_id=pid, elapsed_s=round(time.monotonic() - start, 3),
                       outputs=len(fetched))
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

        entry = hj.get(pid) or {}
        status = entry.get("status") or {}
        if (status.get("completed") and status.get("status_str") == "success"
                and "execution_cached" in _message_kinds(status.get("messages"))):
            if cached_paths:
                cached: list[tuple[bytes, dict]] = []
                for path_str in cached_paths:
                    local = Path(path_str)
                    try:
                        raw = local.read_bytes() if local.is_file() else b""
                    except OSError:
                        raw = b""
                    if validate_fn(raw):
                        cached.append((raw, {"filename": local.name, "subfolder": "",
                                             "type": "output", "source": "local_cached_output"}))
                        if first_only:
                            break
                if cached:
                    print(f"    CACHED: fully-cached completion -> reused {len(cached)} existing "
                          f"local output(s)")
                    emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                               prompt_id=pid, elapsed_s=round(time.monotonic() - start, 3),
                               outputs=len(cached), cached=True)
                    return cached
            print(f"    WARN: cached completion with no reusable outputs "
                  f"(cached_paths={list(cached_paths or [])})")
            emit_event("prompt_completed", sink=event_sink, host=target, label=label,
                       prompt_id=pid, elapsed_s=round(time.monotonic() - start, 3),
                       outputs=0, cached=True)
            return None

        delay_s = delay_fn(time.monotonic() - start)
        ws_outcome = None
        if not ws_unavailable and _ws_completion_enabled():
            ws_outcome = _wait_for_prompt_ws(pid, target, delay_s)
            if ws_outcome is None:
                ws_unavailable = True
        if ws_outcome is True:
            continue
        if ws_outcome is None:
            time.sleep(delay_s)

    elapsed = time.monotonic() - start
    print(f"    TIMEOUT after {int(elapsed)}s for pid={pid} (may still be running)")
    emit_event("prompt_timeout", sink=event_sink, host=target, label=label, prompt_id=pid,
               elapsed_s=round(elapsed, 3))
    return None


def free_vram(host: str | None = None) -> bool:
    """Best-effort POST /free; true only for curl success plus an observed HTTP 2xx status."""
    try:
        target = (host or Z3_HOST).rstrip("/")
        no_proxy = urlparse(target).hostname or NP
        payload = json.dumps({"unload_models": True, "free_memory": True})
        proc = subprocess.run([
            "curl", "-sS", "--fail", "--noproxy", no_proxy, "--max-time", "30",
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
    box = (host or Z3_HOST).rstrip("/")
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


def object_info_enum_many(
    requests: Sequence[tuple[str, str]], host: str | None = None
) -> dict[tuple[str, str], set[str]]:
    """Parallel ``object_info_enum`` batch. Individual failures degrade to empty sets."""
    requests = list(dict.fromkeys(requests))
    if not requests:
        return {}
    with ThreadPoolExecutor(max_workers=min(6, len(requests))) as pool:
        return dict(pool.map(lambda req: (req, object_info_enum(req[0], req[1], host=host)), requests))


def missing_assets(known: set[str], needed: list[str]) -> list[str]:
    """Names in `needed` absent from `known`. An empty `known` (probe failed) returns [] — skip the guard."""
    if not known:
        return []
    return [n for n in needed if n not in known]


def suggest_assets(name: str, known: set[str], limit: int = 3) -> list[str]:
    """Closest known loader names for an absent one, so a typo'd or subfolder-less name fails
    with actionable candidates. Basename containment beats fuzzy matching because loader
    names carry Windows subfolder prefixes the caller often omits."""
    def basename(n: str) -> str:
        return n.lower().replace("\\", "/").rsplit("/", 1)[-1]

    query = basename(name)
    stem = query.rsplit(".", 1)[0]
    contained = sorted(k for k in known
                       if stem and (stem in basename(k) or basename(k).rsplit(".", 1)[0] in query))
    if contained:
        return contained[:limit]
    # Fuzzy over extension-less stems: the shared ".safetensors" suffix would otherwise pull
    # unrelated names over the cutoff.
    by_stem = {basename(k).rsplit(".", 1)[0]: k for k in known}
    return [by_stem[c] for c in difflib.get_close_matches(stem, list(by_stem), n=limit, cutoff=0.6)]


def _suggest_suffix(names: list[str], known: set[str]) -> str:
    """`; closest: ...` appendix for a preflight rejection (empty when nothing is close)."""
    candidates: list[str] = []
    for name in names:
        for hit in suggest_assets(name, known):
            if hit not in candidates:
                candidates.append(hit)
    return f"; closest: {', '.join(candidates[:3])}" if candidates else ""


def preflight_assets(arch: str, ckpt: str, lora_names: list[str], *,
                     controlnets: "Sequence[str]" = (), upscalers: "Sequence[str]" = (),
                     clips: "Sequence[str]" = (), vaes: "Sequence[str]" = (),
                     host: str | None = None, lora_enums: set[str] | None = None) -> None:
    """Reject a run BEFORE any GPU work if the checkpoint or a LoRA name is absent on the box, turning a
    wasted GPU job + cryptic node error into an instant, clear message. Degrades to a no-op when the
    /object_info probe is unreachable, so it never blocks an otherwise-valid run.

    Optional loader-specific lists extend the same guard without changing existing callers:
    `controlnets` -> ControlNetLoader, `upscalers` -> UpscaleModelLoader, `clips` -> CLIPLoader, and
    `vaes` -> VAELoader. Missing fixed recipe assets are hard stops with the exact loader field reported.
    `host` selects the box to probe (default Z3).

    A miss answered from the disk enum cache is re-verified with one live probe before the
    hard stop, so a model added inside the cache TTL is never wrongly rejected. Rejections
    name the closest known loader entries when any are close."""
    box = (host or Z3_HOST).rstrip("/")
    ckpt_node, ckpt_field = (("CheckpointLoaderSimple", "ckpt_name") if arch == "sdxl"
                             else ("UNETLoader", "unet_name"))
    probes: list[tuple[str, str]] = [(ckpt_node, ckpt_field)]
    if lora_enums is None:
        probes.append(("LoraLoader", "lora_name"))
    if controlnets:
        probes.append(("ControlNetLoader", "control_net_name"))
    if upscalers:
        probes.append(("UpscaleModelLoader", "model_name"))
    if clips:
        probes.append(("CLIPLoader", "clip_name"))
    if vaes:
        probes.append(("VAELoader", "vae_name"))
    enums = object_info_enum_many(probes, host=box)

    def _known_and_missing(node: str, field: str, needed: list[str],
                           known: set[str] | None = None) -> tuple[set[str], list[str]]:
        if known is None:
            known = enums.get((node, field), set())
        miss = missing_assets(known, needed)
        if miss and (box, node, field) in _ENUM_FROM_DISK:
            live = object_info_enum(node, field, host=box, refresh=True)
            if live:
                known = live
                miss = missing_assets(known, needed)
        return known, miss

    known_loras_in = set(lora_enums) if lora_enums is not None else None
    known_ckpts, missing_ckpt = _known_and_missing(ckpt_node, ckpt_field, [ckpt])
    known_loras, missing_loras = _known_and_missing("LoraLoader", "lora_name", lora_names,
                                                    known=known_loras_in)
    if missing_ckpt:
        sys.exit(f"ERROR: checkpoint not found on this box ({box}): {missing_ckpt[0]!r} "
                 f"(checked {ckpt_node}.{ckpt_field}; {len(known_ckpts)} available"
                 f"{_suggest_suffix(missing_ckpt, known_ckpts)})")
    if missing_loras:
        sys.exit(f"ERROR: LoRA(s) not found on this box ({box}): {missing_loras} "
                 f"(checked LoraLoader.lora_name; {len(known_loras)} available"
                 f"{_suggest_suffix(missing_loras, known_loras)})")
    if controlnets:
        known, miss = _known_and_missing("ControlNetLoader", "control_net_name", list(controlnets))
        if miss:
            sys.exit(f"ERROR: ControlNet model(s) not found on this box ({box}): {miss} "
                     f"(checked ControlNetLoader.control_net_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
    if upscalers:
        known, miss = _known_and_missing("UpscaleModelLoader", "model_name", list(upscalers))
        if miss:
            sys.exit(f"ERROR: upscale model(s) not found on this box ({box}): {miss} "
                     f"(checked UpscaleModelLoader.model_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)}) — "
                     f"set Z3_UPSCALER to one of the available names")
    if clips:
        known, miss = _known_and_missing("CLIPLoader", "clip_name", list(clips))
        if miss:
            sys.exit(f"ERROR: CLIP model(s) not found on this box ({box}): {miss} "
                     f"(checked CLIPLoader.clip_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
    if vaes:
        known, miss = _known_and_missing("VAELoader", "vae_name", list(vaes))
        if miss:
            sys.exit(f"ERROR: VAE model(s) not found on this box ({box}): {miss} "
                     f"(checked VAELoader.vae_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
