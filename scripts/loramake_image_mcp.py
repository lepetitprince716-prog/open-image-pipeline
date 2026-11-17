#!/usr/bin/env python3
"""Hermes-facing image-generation facade.

One generation tool: ``studio_generate``. That name is the local ASCII-diffusion
trigger (``mcp__loramake_image__studio_generate``). xAI Imagine stays
``image_generate`` — do not reuse that name. New local edit/gen lines dispatch
*inside* this function; do not add sibling MCP tools for them.

``studio_job`` is the one exception and is not a generation line: it starts
nothing and spends no GPU. It exists because ``studio_generate(wait=false)``
returns a job id as soon as the job is queued — which is how an agent starts
several renders and collects them together instead of blocking on each in turn —
and that id needs somewhere to be read back. ``studio_jobs`` is its read-only
sibling: paged history queries (list/search), channel-scoped off-LAN.
``media_ingest`` renders nothing either: it uploads a local PNG/JPEG to
``/api/media/ingest`` and returns the opaque media id a brief references.

Fresh images are submitted to the Studio ``/api/jobs`` queue by default.  Studio
briefs also accept ``source_media_id`` / ``ref_media_id`` and ``rescue`` / ``promote``
tiers; this MCP passes ``source`` paths for direct follow-ups on the LAN, and opaque
job ids through the queue when off-LAN (cloud): the whole surface — fresh tiers,
promote edits, rescue, sequence/artistic refs — stays available there.
Whole-image follow-ups (``source``), region edits and rescue remain direct specialist
paths on the LAN.  Set ``LORAMAKE_IMAGE_DIRECT=1`` (or ``true``/``yes``) to bypass the queue
for an authorized local run; without it, Studio failures are surfaced as errors
and the script never silently falls back to direct execution.

The engines stay in ``scripts/gen.py``; NSFW fresh or NSFW-source edits use the
authoritative ``nsfw/gen_nsfw.py`` wrapper.  The ``region`` phases ``inspect``
and ``preview`` are pure local PIL work (overlays and mask compilation) and
deliberately never touch a GPU: the model looks, selects, and shows its
selection for confirmation before ``commit`` spends a render.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GEN_PY = REPO / "scripts" / "gen.py"
NSFW_GEN_PY = REPO / "nsfw" / "gen_nsfw.py"  # authoritative SFW/NSFW switch
FRESH_INTENT = "draft"
EDIT_INTENT = "promote"
# Qwen-Image line (image_generate / image_edit): natural-language prompts on the BOXA sidecar. The brief
# tier is fixed; which model answers is the recipe default in configs/qwen21/recipes.json.
QWEN21_TIER = "qwen21"
QWEN21_BOX = "boxa"
QWEN21_DEFAULT_RATIO, QWEN21_DEFAULT_SIZE = "2:3", "std"
# MOSS-Image line (scripts/moss_gen.py via the Studio queue): BOXA only, previews only.
MOSS_TIER = "moss"
BOXA_ONLY_TIERS = (QWEN21_TIER, MOSS_TIER)  # these tiers refuse box=auto
# Opaque ids app/brief.py resolves for ref_media_id: gallery (med_ + 24 hex) or job (12 hex, job_ optional).
MEDIA_OR_JOB_ID_RE = re.compile(r"^(?:med_[0-9a-f]{24}|(?:job_)?[0-9a-f]{12})$")
REGION_INTENT = "region"
RESCUE_INTENT = "rescue"
RESCUE_BOX = "boxb"  # pose lock is verified on BOXB/iLustMix only
RUN_TIMEOUT_S = 900

STUDIO_API = os.environ.get("LORAMAKE_STUDIO_API", "http://127.0.0.1:8873")
HTTP_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 3600.0 + HTTP_TIMEOUT_S  # cover the backend's full job timeout plus polling overhead
SUBMIT_RETRIES = 1  # exactly one retry: two total attempts for transport/5xx
RETRY_BACKOFF_S = 1.0  # patchable via module attribute
POLL_ERROR_TOLERANCE = 3  # consecutive transport/5xx poll failures before giving up
# Ceiling on a caller-chosen studio_job wait. An agent asking to block for an hour has
# given up the point of the non-blocking path; cap it and let them call again.
JOB_WAIT_MAX_S = 300.0
MAX_RESPONSE_BYTES = 1 << 20  # 1 MiB; misbehaving endpoint cannot exhaust memory
# Cloudflare's browser integrity check 403s (error 1010) bare python user agents; off-LAN
# requests ride the tunnel, so they carry the same browser-family UA as loramake.comfy.
_CF_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) loramake-gen/1.0"
# Off-LAN (cloud) sessions are scoped to one caller channel: this header rides every
# Studio request, the server stamps it into the brief at submit, and every read side
# (list/search/get/preview/files) filters by it — the channel sees only what it made.
CHANNEL_HEADER = "X-Loramake-Channel"
# The image block the agent actually looks at is base64-inlined into the tool result, so a
# 6 MB render costs about 8 MB of context to answer "did this work". The archival PNG on
# disk is untouched and stays what the envelope points at; only the inline copy shrinks.
PREVIEW_EDGE = int(os.environ.get("LORAMAKE_MCP_PREVIEW_EDGE", "1024") or 1024)
PREVIEW_QUALITY = int(os.environ.get("LORAMAKE_MCP_PREVIEW_Q", "85") or 85)

REGION_PHASES = ("inspect", "crops", "preview", "commit")
_TERMINAL_STATUSES = frozenset({"done", "error", "cancelled", "interrupted"})
# Session tag: studio_generate(group=...) accumulates jobs into one shared experiment
# (exp_<group>) for the /sessions/ page; LORAMAKE_SESSION_GROUP is the per-session
# default an agent's MCP config can inject. Same contract as app/brief.py.
SESSION_GROUP_ENV = "LORAMAKE_SESSION_GROUP"
_GROUP_RE = re.compile(r"[A-Za-z0-9_-]{1,40}\Z")

# Polling and request seams are patchable so tests can be instant and real-server-free.
_sleep = time.sleep
_urlopen = urllib.request.urlopen

REGION_HELP = (
    'region JSON: {"phase": "inspect"|"crops"|"preview"|"commit", '
    '"boxes": [[x,y,w,h], ...], "cells": ["E9", ...], '
    '"detect": [{"class": "feet", "index": 0}, ...], '
    '"keep_outs": [[x,y,w,h], ...], "mode": "refine"|"replace"|"polish"}. '
    "inspect returns a labelled coordinate grid (plus detector hits when "
    '"detect" names classes); "cells" are grid cell ids read off that overlay '
    "(each becomes one box); crops returns 2x zooms of the selected boxes so "
    "details (like toe count) can be verified up close; preview compiles the "
    "mask and returns a red overlay to confirm; commit runs the repaint."
)


def _studio_api() -> str:
    return STUDIO_API.rstrip("/")


def _cloud_mode() -> bool:
    """Off-LAN sessions reach Studio through the tunnel (https); LAN sessions never do."""
    return _studio_api().lower().startswith("https://")


def _channel() -> str:
    """The caller channel tag: only off-LAN (cloud) sessions carry one; "" on the LAN."""
    if not _cloud_mode():
        return ""
    return os.environ.get("LORAMAKE_CHANNEL", "cloud").strip() or "cloud"


def _request_headers(extra: dict | None = None) -> dict:
    headers = dict(extra or {})
    if _cloud_mode():
        headers.setdefault("User-Agent", _CF_UA)
        headers.setdefault(CHANNEL_HEADER, _channel())
    return headers


def _is_truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes"}


def _direct_enabled() -> bool:
    return _is_truthy_env("LORAMAKE_IMAGE_DIRECT")


def _lora_specs(primary: str | None, extras: list[str] | str | None) -> list[str]:
    """Combine a primary LoRA and an optional list of extras into one ordered list.

    A single string is one exact spec (filenames can contain commas); a list/sequence
    carries multiple specs.  Only surrounding whitespace is stripped.
    """
    out: list[str] = []
    if primary:
        out.append(primary.strip())
    if not extras:
        return out
    if isinstance(extras, str):
        out.append(extras.strip())
    else:
        for spec in extras:
            if isinstance(spec, str) and spec.strip():
                out.append(spec.strip())
    return out


def _style_lora_specs(value: str | list[str] | None) -> list[str]:
    """A single string is one exact style LoRA; a list carries multiple."""
    if not value:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def gen_script(
    source: str | None = None,
    gen_py: Path | None = None,
    nsfw: bool = False,
) -> Path:
    """Pick the existing router. Source under nsfw/outputs uses the authoritative wrapper."""
    if gen_py:
        return gen_py
    override = os.environ.get("LORAMAKE_GEN_PY")
    if override:
        return Path(override)
    if source:
        resolved = Path(source).expanduser().resolve()
        nsfw_root = (REPO / "nsfw" / "outputs").resolve()
        try:
            resolved.relative_to(nsfw_root)
            if NSFW_GEN_PY.is_file():
                return NSFW_GEN_PY
        except ValueError:
            pass
    if nsfw and NSFW_GEN_PY.is_file():
        return NSFW_GEN_PY
    return DEFAULT_GEN_PY


def build_gen_argv(
    prompt: str,
    source: str | None,
    json_out: Path,
    *,
    gen_py: Path | None = None,
    tier: str | None = None,
    box: str | None = None,
    base: str | None = None,
    lora: str | None = None,
    loras: list[str] | str | None = None,
    style_lora: str | list[str] | None = None,
    ref: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    res: str | None = None,
    nsfw: bool = False,
) -> list[str]:
    """Build a ``gen.py`` or ``nsfw/gen_nsfw.py`` argv from the new options.

    When no optional knobs are supplied the argv is exactly the historical
    ``draft`` / ``promote`` default shape.
    """
    script = gen_py or gen_script(source=source, nsfw=nsfw)
    intent = tier or (EDIT_INTENT if source else FRESH_INTENT)
    argv = [sys.executable, str(script), intent]
    if source:
        argv += ["--source", source]
    if box and box != "auto":
        argv += ["--box", box]
    if base:
        argv += ["--base", base]
    argv += ["--prompt", prompt]
    for spec in _lora_specs(lora, loras):
        argv += ["--lora", spec]
    if n is not None:
        argv += ["--n", str(n)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if ref:
        argv += ["--ref", ref]
    for sl in _style_lora_specs(style_lora):
        argv += ["--style-lora", sl]
    if intent == "sequence":
        argv.append("--ccip-keep")
    if intent == "artistic" and not nsfw:
        argv.append("--sfw")
    argv += ["--run", "--json-out", str(json_out)]
    if res:
        argv += ["--", "--res", res]
    return argv


def build_rescue_argv(
    prompt: str | None,
    source: str,
    json_out: Path,
    *,
    gen_py: Path | None = None,
) -> list[str]:
    """Structure rescue: gen.py rescue -> scripts/rescue_gen.py (one shot, fixed recipe)."""
    script = gen_py or gen_script(source=source)
    argv = [sys.executable, str(script), RESCUE_INTENT, "--source", source]
    if prompt:
        argv += ["--prompt", prompt]
    argv += ["--box", RESCUE_BOX, "--run", "--json-out", str(json_out)]
    return argv


def build_region_argv(
    prompt: str,
    source: str,
    spec_json: str,
    mode: str,
    json_out: Path,
    *,
    gen_py: Path | None = None,
) -> list[str]:
    """The commit-phase dispatch: gen.py region -> scripts/region_fix.py."""
    script = gen_py or gen_script(source=source)
    argv = [sys.executable, str(script), REGION_INTENT, "--source", source, "--prompt", prompt]
    argv += ["--run", "--json-out", str(json_out)]
    argv += ["--", "--spec-json", spec_json, "--mode", mode]
    return argv


def pick_image_path(result: dict) -> str | None:
    """One displayable path from a ``loramake.gen-result`` blob."""
    for key in ("best",):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("survivors", "new_files"):
        files = result.get(key)
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str) and item.strip():
                    return item
    return None


def display_envelope(path: str, result: dict, job_id: str | None = None,
                     group: str | None = None) -> dict:
    envelope = {
        "success": True,
        "host_image": path,
        "image": path,
        "intent": result.get("intent"),
        "box": result.get("box"),
        "rc": result.get("rc", 0),
    }
    if job_id is not None:
        envelope["job_id"] = job_id
    if group:
        envelope["group"] = group
    return envelope


def fail_envelope(error: str, **extra: object) -> dict:
    payload = {"success": False, "error": error}
    payload.update(extra)
    return payload


def pending_envelope(job_id: str, status: str, **extra: object) -> dict:
    """The job is accepted and still running. Not a success (there is no image) and not a
    failure (nothing went wrong), so it carries its own flag rather than overloading either --
    an agent that only reads ``success`` must not conclude an image exists.
    """
    payload = {
        "success": False,
        "pending": True,
        "job_id": job_id,
        "status": status,
        "note": "job is still running; call studio_job with this job_id for the result",
    }
    payload.update(extra)
    return payload


def _spawn_gen(argv: list[str], json_out: Path) -> dict:
    """Run one gen.py subprocess and translate its json-out into an envelope."""
    env = {**os.environ, "DRIVING_MODEL": os.environ.get("DRIVING_MODEL", "hermes")}
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
        raw = json_out.read_text(encoding="utf-8") if json_out.is_file() else ""
        result = json.loads(raw) if raw.strip() else {}
        if not isinstance(result, dict):
            result = {}
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"gen.py exited {proc.returncode}").strip()
            return fail_envelope(err[-2000:], rc=proc.returncode, intent=result.get("intent"))
        path = pick_image_path(result)
        if not path:
            return fail_envelope("gen.py produced no image", rc=proc.returncode, result=result)
        return display_envelope(path, result)
    except subprocess.TimeoutExpired:
        return fail_envelope(f"gen.py timed out after {RUN_TIMEOUT_S}s")
    except json.JSONDecodeError as exc:
        return fail_envelope(f"gen.py json-out unreadable: {exc}")
    finally:
        json_out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Studio API queue seam (urllib, stdlib only, no SSE)
# --------------------------------------------------------------------------- #


class _ResponseTooLargeError(Exception):
    """Raised when a response body exceeds MAX_RESPONSE_BYTES."""


def _read_limited(stream: object, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read at most ``limit`` bytes from a file-like response stream."""
    # Read one extra byte to detect over-limit without consuming the whole stream.
    chunk = stream.read(limit + 1)
    if len(chunk) > limit:
        raise _ResponseTooLargeError(f"response body exceeded {limit} bytes")
    return chunk


def _http_request(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = HTTP_TIMEOUT_S,
    data: bytes | None = None,
) -> tuple[int | None, str]:
    """Make a bounded JSON HTTP request and return (status, body) without secrets.

    Non-2xx HTTP status codes are returned, not raised; transport errors become
    (None, <safe message>) so callers decide whether to surface them.
    Response bodies are capped at MAX_RESPONSE_BYTES.  Only documented transport
    classes are caught; programming errors propagate.  ``data`` sends a prebuilt
    body (multipart) as-is; the caller sets its Content-Type.
    """
    req_headers = _request_headers(headers)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with _urlopen(req, timeout=timeout) as resp:
            body_bytes = _read_limited(resp, MAX_RESPONSE_BYTES)
            return resp.status, body_bytes.decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            body_bytes = _read_limited(exc, MAX_RESPONSE_BYTES)
            body_text = body_bytes.decode("utf-8", errors="replace")
        except _ResponseTooLargeError:
            body_text = "error response body too large"
        except (OSError, ValueError):
            body_text = ""
        return exc.code, body_text
    except _ResponseTooLargeError:
        # 2xx body too large is a shape/misbehaviour error, not a transient.
        return None, "Studio API response body too large"
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        # Never include the full URL or any secret env value in the returned text.
        return None, f"Studio API request failed: {type(exc).__name__}"


def _post_brief(url: str, brief: dict, idem: str) -> tuple[int | None, str]:
    """Submit a brief with the same idempotency key across any configured retries."""
    last_status: int | None = None
    last_body = ""
    for attempt in range(SUBMIT_RETRIES + 1):
        status, body = _http_request(
            "POST", url, body=brief, headers={"Idempotency-Key": idem}
        )
        last_status, last_body = status, body
        if status is None:
            # Transport error: retry if configured.
            if attempt < SUBMIT_RETRIES:
                _sleep(RETRY_BACKOFF_S)
                continue
            return last_status, last_body
        if status < 500:
            return last_status, last_body
        # 5xx: retry if configured.
        if attempt < SUBMIT_RETRIES:
            _sleep(RETRY_BACKOFF_S)
            continue
    return last_status, last_body


def _parse_json(text: str) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(text or "")
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "expected a JSON object"
    return data, None


def _api_path(path: str) -> str:
    return _studio_api() + path


def _resolve_group(group: str | None) -> tuple[str | None, str | None]:
    """Pick the session tag: explicit param wins, else the per-session env default.

    Returns (group, error). Invalid values come back as an error string instead of
    being dropped silently — the agent must learn its tag was refused.
    """
    value = group if group else os.environ.get(SESSION_GROUP_ENV) or None
    if value is None:
        return None, None
    value = value.strip()
    if not value:
        return None, None
    if not _GROUP_RE.fullmatch(value):
        return None, (
            f"group must match [A-Za-z0-9_-]{{1,40}} (got {value!r}"
            + ("" if group else f" from {SESSION_GROUP_ENV}")
            + ")"
        )
    return value, None


def build_brief(
    prompt: str,
    *,
    tier: str = "draft",
    box: str = "auto",
    provider: str = "local",
    base: str | None = None,
    lora: str | None = None,
    loras: list[str] | str | None = None,
    style_lora: str | None = None,
    ref: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    res: str | None = None,
    nsfw: bool = False,
    group: str | None = None,
    channel: str | None = None,
    source_media_id: str | None = None,
    ref_media_id: str | None = None,
) -> dict:
    """Build the ``/api/jobs`` brief JSON.  Defaults are draft / auto / local."""
    brief: dict = {
        "prompt": prompt,
        "tier": tier,
        "box": box,
        "provider": provider,
        "nsfw": bool(nsfw),
    }
    if channel:
        brief["channel"] = channel
    if base:
        brief["base"] = base
    if lora:
        brief["lora"] = lora
    extras = _lora_specs(None, loras)
    if extras:
        brief["loras"] = extras
    if style_lora:
        brief["style_lora"] = style_lora
    if ref:
        brief["ref"] = ref
    if source_media_id:
        brief["source_media_id"] = source_media_id
    if ref_media_id:
        brief["ref_media_id"] = ref_media_id
    if n is not None:
        brief["n"] = n
    if seed is not None:
        brief["seed"] = seed
    if res:
        brief["res"] = res
    if group:
        brief["group"] = group
    return brief


def _poll_job(job_id: str, wait_s: float | None = None) -> dict:
    """Poll ``/api/jobs/{id}`` until terminal or the budget runs out; never cancel the job.

    Tolerates a bounded number of consecutive transport/5xx blips after the job
    id is known; a successful 2xx poll resets the counter.  Malformed 2xx JSON,
    4xx, and exhaustion of the tolerance are immediate errors.

    ``wait_s`` bounds the wait. Running out of a *caller-chosen* budget is a pending
    result -- the job is fine, the caller just stopped waiting -- while running out of the
    full ``POLL_TIMEOUT_S`` is the timeout it always was. ``wait_s=0`` still reads the job
    once: the point of asking is to learn the state, and a poll that never looks would report
    "running" for a job that had already failed.
    """
    bounded = wait_s is not None
    url = _api_path(f"/api/jobs/{job_id}")
    deadline = time.monotonic() + (wait_s if bounded else POLL_TIMEOUT_S)
    consecutive_errors = 0
    last_status = ""
    last_group: str | None = None
    polled = False
    while True:
        if polled and time.monotonic() > deadline:
            if bounded:
                return pending_envelope(job_id, last_status or "running", group=last_group)
            return fail_envelope("Studio job timed out", job_id=job_id, status="timeout")
        polled = True
        status, body = _http_request("GET", url)
        if status is None or status >= 500:
            consecutive_errors += 1
            if consecutive_errors >= POLL_ERROR_TOLERANCE:
                return fail_envelope(
                    "Studio job poll failed after repeated transient errors",
                    job_id=job_id,
                    status="poll_error",
                )
            _sleep(POLL_INTERVAL_S)
            continue
        if status >= 400:
            detail, _ = _parse_json(body)
            msg = detail.get("detail", "") if isinstance(detail, dict) else ""
            return fail_envelope(
                f"Studio job poll failed: {msg or f'HTTP {status}'}",
                job_id=job_id,
                status="poll_error",
            )
        # Successful 2xx poll resets the blip counter.
        consecutive_errors = 0
        job, err = _parse_json(body)
        if err:
            return fail_envelope(err, job_id=job_id, status="poll_error")
        brief = job.get("brief")
        group = brief.get("group") if isinstance(brief, dict) else None
        if isinstance(group, str) and group:
            last_group = group
        st = job.get("status")
        last_status = str(st or "")
        if st in _TERMINAL_STATUSES:
            if st == "done":
                result = job.get("result") or {}
                path = pick_image_path(result)
                if not path:
                    return fail_envelope(
                        "Studio job finished but produced no image",
                        job_id=job_id,
                        status=st,
                        result=result,
                    )
                return display_envelope(path, result, job_id=job_id, group=group)
            error = job.get("error") or st
            return fail_envelope(str(error), job_id=job_id, status=st, group=group)
        _sleep(POLL_INTERVAL_S)


def _queue_generate(
    prompt: str,
    *,
    tier: str = "draft",
    box: str = "auto",
    base: str | None = None,
    lora: str | None = None,
    loras: list[str] | str | None = None,
    style_lora: str | None = None,
    ref: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    res: str | None = None,
    nsfw: bool = False,
    wait: bool = True,
    group: str | None = None,
    source_media_id: str | None = None,
    ref_media_id: str | None = None,
) -> dict:
    group, group_error = _resolve_group(group)
    if group_error:
        return fail_envelope(group_error)
    brief = build_brief(
        prompt,
        tier=tier,
        box=box,
        base=base,
        lora=lora,
        loras=loras,
        style_lora=style_lora,
        ref=ref,
        n=n,
        seed=seed,
        res=res,
        nsfw=nsfw,
        group=group,
        channel=_channel() or None,
        source_media_id=source_media_id,
        ref_media_id=ref_media_id,
    )
    return _submit_brief(brief, wait)


def _submit_brief(brief: dict, wait: bool = True) -> dict:
    """POST one already-built brief to the Studio queue; wait for the image or hand back the pending form."""
    idem = uuid.uuid4().hex
    url = _api_path("/api/jobs")
    status, body = _post_brief(url, brief, idem)
    if status is None:
        return fail_envelope(f"Studio API unavailable: {body}")
    if status >= 400:
        detail, _ = _parse_json(body)
        msg = detail.get("detail", "") if isinstance(detail, dict) else ""
        return fail_envelope(
            f"Studio submit failed: {msg or f'HTTP {status}'}",
            status_code=status,
        )
    job, err = _parse_json(body)
    if err or not isinstance(job, dict) or not job.get("id"):
        return fail_envelope(err or "Studio returned an unrecognised job shape")
    if not wait:
        return pending_envelope(job["id"], str(job.get("status") or "queued"), group=brief.get("group"))
    return _poll_job(job["id"])


def _civitai_fields(model: str | None, loras: list[str] | str | None, n: int | None,
                    seed: int | None, res: str | None, steps: int | None,
                    cfg: float | None) -> dict:
    """The Civitai knobs shared by a generate brief and a whatif quote (None = alias default)."""
    fields: dict = {}
    for key, value in (("base", model), ("n", n), ("seed", seed), ("res", res),
                       ("steps", steps), ("cfg", cfg)):
        if value is not None:
            fields[key] = value
    specs = _lora_specs(None, loras)
    if specs:
        fields["loras"] = specs
    return fields


def run_civitai_generate(prompt: str, *, model: str | None = None,
                         loras: list[str] | str | None = None, n: int | None = None,
                         seed: int | None = None, res: str | None = None,
                         steps: int | None = None, cfg: float | None = None,
                         wait: bool = True, group: str | None = None) -> dict:
    """Queue a Civitai cloud render through Studio. The standing Buzz budget, enforced in
    loramake.image_providers after the free whatif, is the spend authorization."""
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")
    group, group_error = _resolve_group(group)
    if group_error:
        return fail_envelope(group_error)
    brief = {
        "tier": "best",
        "prompt": text,
        "provider": "civitai",
        "confirm_paid_generation": True,
        "sfw": True,
        **_civitai_fields(model, loras, n, seed, res, steps, cfg),
    }
    if group:
        brief["group"] = group
    if _channel():
        brief["channel"] = _channel()
    return _submit_brief(brief, wait)


def run_civitai_quote(prompt: str, *, model: str | None = None,
                      loras: list[str] | str | None = None, n: int | None = None,
                      seed: int | None = None, res: str | None = None,
                      steps: int | None = None, cfg: float | None = None) -> dict:
    """Free whatif through Studio: cost, resolved model and whether the budget would allow it."""
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")
    body = {"provider": "civitai", "prompt": text,
            **_civitai_fields(model, loras, n, seed, res, steps, cfg)}
    status, raw = _http_request("POST", _api_path("/api/providers/whatif"), body=body)
    if status is None:
        return fail_envelope(f"Studio API unavailable: {raw}")
    data, err = _parse_json(raw)
    if status >= 400:
        detail = data.get("detail", "") if isinstance(data, dict) else ""
        return fail_envelope(f"Civitai quote failed: {detail or f'HTTP {status}'}", status_code=status)
    if err or data is None:
        return fail_envelope(err or "Studio returned an unrecognised quote")
    return {"success": True, **data}


def _direct_generate(
    prompt: str,
    *,
    source: str | None = None,
    tier: str = "draft",
    box: str | None = None,
    base: str | None = None,
    lora: str | None = None,
    loras: list[str] | str | None = None,
    style_lora: str | list[str] | None = None,
    ref: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    res: str | None = None,
    nsfw: bool = False,
) -> dict:
    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_gen_argv(
        prompt,
        source,
        json_out,
        tier=tier,
        box=box,
        base=base,
        lora=lora,
        loras=loras,
        style_lora=style_lora,
        ref=ref,
        n=n,
        seed=seed,
        res=res,
        nsfw=nsfw,
    )
    return _spawn_gen(argv, json_out)


def _source_incompatible(
    tier: str | None = None,
    base: str | None = None,
    ref: str | None = None,
    style_lora: object = None,
    res: str | None = None,
) -> str | None:
    """Return an error when fresh-only knobs are passed to a source edit."""
    if tier and tier != EDIT_INTENT:
        return f"source edits use the '{EDIT_INTENT}' tier, not '{tier}'"
    if base:
        return "source edits do not accept base"
    if ref:
        return "source edits do not accept ref"
    if style_lora:
        return "source edits do not accept style_lora"
    if res:
        return "source edits do not accept res"
    return None


def build_qwen21_brief(
    prompt: str,
    *,
    image: str | None = None,
    ratio: str | None = None,
    size: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    preview: bool | None = None,
) -> dict:
    """The qwen21 brief for image_generate (no image) or image_edit (image = path or gallery/job id).

    ``preview`` None means "webp preview when BOXA is not on the LAN": over Cloudflare or Tailscale the
    1-2 MB PNG is what makes a 5 s render take minutes; the PNG stays in the sidecar's output/.

    A local path is expanded and made absolute here, before it crosses HTTP: the Studio server resolves
    relative paths against its own working directory, not the caller's. ``ratio``/``size`` must travel
    as a pair, so a lone one is completed with the line default.
    """
    brief: dict = {"prompt": prompt, "tier": QWEN21_TIER, "box": QWEN21_BOX, "provider": "local", "nsfw": False}
    if image is not None:
        text = str(image).strip()
        if MEDIA_OR_JOB_ID_RE.fullmatch(text):
            brief["ref_media_id"] = text
        else:
            path = Path(text).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"image not found: {image}")
            brief["ref"] = str(path.resolve())
    elif ratio is not None or size is not None:
        brief["ratio"] = ratio or QWEN21_DEFAULT_RATIO
        brief["size"] = size or QWEN21_DEFAULT_SIZE
    for key, value in (("n", n), ("seed", seed), ("steps", steps)):
        if value is not None:
            brief[key] = value
    if preview is None:
        preview = boxa_is_remote()
    if preview:
        brief["preview"] = True
    return brief


def boxa_is_remote() -> bool:
    """True unless the route picker's cache says BOXA is on a live LAN route (never probes here).

    A cache that says every route is down (the picker then reports the canonical ``lan`` row with
    ``alive`` false) counts as remote too: the cheap webp fetch is the right default off the LAN.
    """
    try:
        from loramake import net_routes

        return net_routes.is_remote("boxa")
    except Exception:
        return False


def run_image_generate(prompt: str, *, ratio: str | None = None, size: str | None = None,
                       n: int | None = None, seed: int | None = None, steps: int | None = None,
                       preview: bool | None = None, wait: bool = True) -> dict:
    """Queue one Qwen-Image t2i brief (queue-only: the line has no direct in-process path)."""
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")
    brief = build_qwen21_brief(text, ratio=ratio, size=size, n=n, seed=seed, steps=steps, preview=preview)
    return _submit_brief(brief, wait)


def run_image_edit(image: str, prompt: str, *, n: int | None = None, seed: int | None = None,
                   steps: int | None = None, preview: bool | None = None, wait: bool = True) -> dict:
    """Queue one Qwen-Image edit brief for ``image`` (local path or gallery/job id)."""
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")
    if not str(image or "").strip():
        return fail_envelope("image_edit needs an image: a local file path or a gallery/job id")
    try:
        brief = build_qwen21_brief(text, image=image, n=n, seed=seed, steps=steps, preview=preview)
    except FileNotFoundError as exc:
        return fail_envelope(str(exc))
    return _submit_brief(brief, wait)


def _default_box(tier: str | None, box: str | None) -> str:
    """The box a queued brief names when the caller gave none; BOXA-only tiers refuse auto."""
    return box or (QWEN21_BOX if tier in BOXA_ONLY_TIERS else "auto")


def run_generate(
    prompt: str,
    source: str | None = None,
    *,
    tier: str | None = None,
    box: str | None = None,
    base: str | None = None,
    lora: str | None = None,
    loras: list[str] | str | None = None,
    style_lora: str | list[str] | None = None,
    ref: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    res: str | None = None,
    nsfw: bool = False,
    wait: bool = True,
    group: str | None = None,
) -> dict:
    """Fresh (queue by default) or whole-image follow-up (direct)."""
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")

    if _cloud_mode():
        # Off-LAN there is no local engine and no host filesystem: everything rides the
        # Studio queue. source references an earlier result by opaque id (promote tier);
        # ref anchors sequence/artistic by id. The server stamps the session channel and
        # refuses media outside it, so a cloud session only ever touches its own results.
        if _direct_enabled():
            return fail_envelope(
                "LORAMAKE_IMAGE_DIRECT is set but a cloud session has no local engine; "
                "only the Studio queue is reachable"
            )
        ref_media_id = None
        if ref:
            ref_media_id, ref_error = _cloud_media_ref(ref, "ref")
            if ref_error:
                return fail_envelope(ref_error)
        if source:
            source_id, source_error = _cloud_media_ref(source, "source")
            if source_error:
                return fail_envelope(source_error)
            if tier and tier != EDIT_INTENT:
                return fail_envelope(f"source edits use the '{EDIT_INTENT}' tier, not '{tier}'")
            if style_lora:
                return fail_envelope("source edits do not accept style_lora")
            if ref:
                return fail_envelope("pass either source or ref, not both")
            return _queue_generate(
                text,
                tier=EDIT_INTENT,
                box=box or "auto",
                base=base,
                lora=lora,
                loras=loras,
                n=n,
                seed=seed,
                res=res,
                nsfw=nsfw,
                wait=wait,
                group=group,
                source_media_id=source_id,
            )
        return _queue_generate(
            text,
            tier=tier or FRESH_INTENT,
            box=_default_box(tier, box),
            base=base,
            lora=lora,
            loras=loras,
            style_lora=style_lora,
            n=n,
            seed=seed,
            res=res,
            nsfw=nsfw,
            wait=wait,
            group=group,
            ref_media_id=ref_media_id,
        )

    if source:
        src = Path(source).expanduser()
        if not src.is_file():
            return fail_envelope(f"source not found: {source}")
        source = str(src)
        bad = _source_incompatible(
            tier=tier, base=base, ref=ref, style_lora=style_lora, res=res
        )
        if bad:
            return fail_envelope(bad)
        if group:
            # No queued job means nothing accumulates into the session experiment;
            # say so instead of dropping the tag.
            return fail_envelope("group is only valid for fresh queued generation")
        if not wait:
            # A source edit runs direct: a blocking subprocess with no job id to hand back.
            return fail_envelope(
                "wait=false is for fresh queued generation; a source edit runs direct and blocks"
            )
        return _direct_generate(
            text,
            source=source,
            tier=EDIT_INTENT,
            box=box,
            lora=lora,
            loras=loras,
            n=n,
            seed=seed,
            nsfw=nsfw,
        )

    if _direct_enabled():
        if tier == MOSS_TIER:
            return fail_envelope("the moss tier runs through the Studio queue only; "
                                 "unset LORAMAKE_IMAGE_DIRECT")
        if group:
            return fail_envelope(
                "group is only valid for fresh queued generation; "
                "LORAMAKE_IMAGE_DIRECT=1 runs in-process and records no job"
            )
        if not wait:
            # Direct mode is a blocking subprocess with no handle to hand back. Say so rather
            # than returning a job id that does not exist or silently waiting anyway.
            return fail_envelope(
                "wait=false needs the Studio queue; LORAMAKE_IMAGE_DIRECT=1 runs the engine "
                "in-process and has no job to poll"
            )
        return _direct_generate(
            text,
            source=None,
            tier=tier or FRESH_INTENT,
            box=box,
            base=base,
            lora=lora,
            loras=loras,
            style_lora=style_lora,
            ref=ref,
            n=n,
            seed=seed,
            res=res,
            nsfw=nsfw,
        )

    return _queue_generate(
        text,
        tier=tier or FRESH_INTENT,
        box=_default_box(tier, box),
        base=base,
        lora=lora,
        loras=loras,
        style_lora=style_lora,
        ref=ref,
        n=n,
        seed=seed,
        res=res,
        nsfw=nsfw,
        wait=wait,
        group=group,
    )


_JOB_OR_MEDIA_ID = re.compile(r"^(?:med_[0-9a-f]{24}|(?:job_)?[0-9a-f]{12})$")


def _cloud_media_ref(value: str, field: str) -> tuple[str | None, str | None]:
    """(opaque id, error): off-LAN, images are referenced by id — never by host path.

    The server re-validates scope (the id must belong to this session's channel), so a
    well-formed but foreign id fails there; only shape errors are caught here.
    """
    text = (value or "").strip()
    if not text:
        return None, f"{field} needs the job id of an earlier result from this channel"
    if "/" in text or "\\" in text or text.startswith("~"):
        return None, (
            f"{field} must be an opaque job id in a cloud session, not a path — host-side "
            "paths are unreachable from here; use the job_id of an earlier result"
        )
    if not _JOB_OR_MEDIA_ID.fullmatch(text):
        return None, f"{field} must be a gallery id (med_…) or job id, got {text!r}"
    return text, None


def run_rescue(prompt: str | None, source: str | None,
               *, wait: bool = True, group: str | None = None, nsfw: bool = False) -> dict:
    """Pose-locked re-render of a structurally broken image (extra limbs, fused
    objects, impossible joints). The prompt is optional: without it the source
    sidecar prompt is reused; pass one only to correct a content-level error
    (an OOD token literalised into an object)."""
    if _cloud_mode():
        # Off-LAN rescue rides the queue like any other channel job: the rescue tier keeps
        # its fixed recipe server-side (box auto resolves to the BOXB pose-lock line), and
        # the source is an opaque id from this session's channel.
        if not source:
            return fail_envelope("rescue needs a source: the job id of the broken render")
        source_id, source_error = _cloud_media_ref(source, "source")
        if source_error:
            return fail_envelope(source_error)
        return _queue_generate(
            (prompt or "").strip(),
            tier=RESCUE_INTENT,
            box="auto",
            nsfw=nsfw,
            wait=wait,
            group=group,
            source_media_id=source_id,
        )
    if not source:
        return fail_envelope("rescue needs a source image path")
    src = Path(source).expanduser()
    if not src.is_file():
        return fail_envelope(f"source not found: {source}")

    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_rescue_argv((prompt or "").strip() or None, str(src), json_out)
    envelope = _spawn_gen(argv, json_out)
    envelope.setdefault("mode", "rescue")
    return envelope


# --------------------------------------------------------------------------------------- #
# Job history queries (read-only; the channel header scopes them off-LAN)
# --------------------------------------------------------------------------------------- #


def _compact_job(job: dict) -> dict:
    """The list/search row an agent actually reads; full detail stays one studio_job away."""
    brief = job.get("brief") if isinstance(job.get("brief"), dict) else {}
    out: dict = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "tier": brief.get("tier"),
        "box": brief.get("box"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
    }
    prompt = brief.get("prompt")
    if isinstance(prompt, str) and prompt:
        out["prompt"] = prompt if len(prompt) <= 300 else prompt[:297] + "..."
    for key in ("nsfw", "group", "channel"):
        value = brief.get(key)
        if value not in (None, "", False):
            out[key] = value
    if job.get("error"):
        out["error"] = job["error"]
    result = job.get("result")
    if isinstance(result, dict):
        path = pick_image_path(result)
        if path:
            out["image"] = path
    return out


def run_jobs_query(q: str = "", status: str = "", limit: int = 20, offset: int = 0) -> dict:
    """Paged job history from the Studio queue; off-LAN the channel header scopes it."""
    try:
        limit = max(1, min(int(limit), 50))
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        return fail_envelope(f"limit/offset must be integers, got {limit!r}/{offset!r}")
    query = urllib.parse.urlencode(
        {"q": q or "", "status": status or "", "limit": limit, "offset": offset}
    )
    code, body = _http_request("GET", _api_path(f"/api/jobs/search?{query}"))
    if code is None:
        return fail_envelope(f"Studio API unavailable: {body}")
    if code >= 400:
        detail, _ = _parse_json(body)
        msg = detail.get("detail", "") if isinstance(detail, dict) else ""
        return fail_envelope(f"Studio jobs query failed: {msg or f'HTTP {code}'}",
                             status_code=code)
    data, err = _parse_json(body)
    if err or not isinstance(data.get("jobs") if isinstance(data, dict) else None, list):
        return fail_envelope(err or "Studio returned an unrecognised jobs shape")
    return {
        "success": True,
        "total": data.get("total", 0),
        "jobs": [_compact_job(j) for j in data["jobs"] if isinstance(j, dict)],
    }


INGEST_MAX_BYTES = 32 << 20  # app/media_ingest.MAX_BYTES; checked here so a huge file is never read
INGEST_TIMEOUT_S = 120.0
_INGEST_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _error_detail(body: str) -> str:
    """FastAPI ``detail``: a string for request errors, a list of ``{msg}`` for validation."""
    data, _ = _parse_json(body)
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg")) for d in detail if isinstance(d, dict) and d.get("msg"))
    return body[:200]


def run_media_ingest(path: str, label: str | None = None, source: str | None = None) -> dict:
    """Upload one local PNG/JPEG to ``POST /api/media/ingest``; the reply comes back as-is."""
    file = Path(path or "").expanduser()
    if not str(path or "").strip() or not file.is_file():
        return fail_envelope(f"media_ingest needs an existing image file, got {path!r}")
    if file.stat().st_size > INGEST_MAX_BYTES:
        return fail_envelope(f"file exceeds the {INGEST_MAX_BYTES >> 20} MB ingest limit: {file}")
    boundary = uuid.uuid4().hex
    fields = {k: v for k, v in (("label", label), ("source", source)) if v}
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    name = file.name.replace('"', "_")
    ctype = _INGEST_TYPES.get(file.suffix.lower(), "application/octet-stream")
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + file.read_bytes() + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    code, body = _http_request(
        "POST", _api_path("/api/media/ingest"), data=b"".join(parts), timeout=INGEST_TIMEOUT_S,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if code is None:
        return fail_envelope(f"Studio API unavailable: {body}")
    if code >= 400:
        return fail_envelope(f"media ingest failed: {_error_detail(body) or f'HTTP {code}'}",
                             status_code=code)
    data, err = _parse_json(body)
    if err or not isinstance(data.get("media_id"), str):
        return fail_envelope(err or "Studio ingest reply has no media_id")
    return data


# --------------------------------------------------------------------------------------- #
# Region phases
# --------------------------------------------------------------------------------------- #

def _scratch_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "loramake_region"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scratch_path(stem: str, kind: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return _scratch_dir() / f"{stem}_{kind}_{stamp}.png"


def _parse_detect_classes(value: object) -> list[str]:
    sys.path.insert(0, str(REPO))
    from loramake.mask_spec import DETECT_CLASSES
    from loramake.region_mask import MaskError

    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise MaskError('inspect "detect" must be a list of class names')
    unknown = [v for v in value if v not in DETECT_CLASSES]
    if unknown:
        raise MaskError(
            f"unknown detector classes {unknown}; choose from {', '.join(DETECT_CLASSES)}"
        )
    return list(value)


def _region_inspect(source: str, payload: dict) -> tuple[dict, str | None]:
    from PIL import Image

    from loramake import region_detect, region_overlay
    from loramake.region_mask import MaskError

    extra = set(payload) - {"phase", "detect"}
    if extra:
        raise MaskError(f"inspect accepts only \"detect\"; unknown keys: {sorted(extra)}")
    classes = _parse_detect_classes(payload.get("detect"))

    img = Image.open(source)
    img.load()
    overlay = region_overlay.grid_overlay(img)
    detections: list[dict] = []
    if classes:
        detections = [
            d for d in region_detect.detect_regions(source) if d["label"] in classes
        ]
        overlay = region_overlay.hits_overlay(overlay, detections)
    out_path = _scratch_path(Path(source).stem, "inspect")
    overlay.save(out_path)
    step_x, step_y = region_overlay.grid_steps((img.width, img.height))
    envelope = {
        "success": True,
        "phase": "inspect",
        "source": source,
        "size": [img.width, img.height],
        "grid_step": [step_x, step_y],
        "detections": detections,
        "overlay": str(out_path),
        "next": (
            "read the target region off the grid (cell ids like E9, or pixel boxes), then "
            "call phase=crops to verify the detail up close before phase=preview"
        ),
    }
    return envelope, str(out_path)


def _spec_fields(source: str, payload: dict) -> dict:
    """Region payload -> mask-spec fields: strip the phase, resolve cells to boxes.

    Cells are the coarse answer format for models that read the grid overlay but cannot
    report reliable pixel coordinates; the conversion must use the same grid step the
    overlay drew, so it goes through region_overlay.
    """
    from PIL import Image

    from loramake import region_overlay
    from loramake.region_mask import MaskError

    fields = {k: v for k, v in payload.items() if k != "phase"}
    cells = fields.pop("cells", None)
    if cells is not None:
        if not isinstance(cells, list) or not all(isinstance(c, str) for c in cells):
            raise MaskError('"cells" must be a list of grid cell ids like "E9"')
        with Image.open(source) as img:
            size = (img.width, img.height)
        try:
            boxes = region_overlay.cells_to_boxes(size, cells)
        except ValueError as exc:
            raise MaskError(str(exc)) from exc
        fields["boxes"] = list(fields.get("boxes") or []) + [list(b.as_tuple()) for b in boxes]
    fields.setdefault("source", source)
    return fields


def _compile_from_payload(source: str, payload: dict):
    """Shared preview/commit spec handling: parse, detect if needed, compile."""
    from loramake import mask_spec, region_detect

    spec = mask_spec.parse_spec(_spec_fields(source, payload))
    detections = region_detect.detect_regions(source) if spec.wants_detection() else []
    from PIL import Image

    with Image.open(source) as img:
        size = (img.width, img.height)
        src = img.convert("RGB")
    mask = mask_spec.compile_mask(spec, size, detections)
    return spec, src, mask, detections


def _region_crops(source: str, payload: dict) -> tuple[dict, str | None]:
    """2x zooms of the selected boxes, tiled into one sheet — the look-closer step.

    Fine judgments (a toe count, a finger boundary) are unreliable on a full frame even for
    strong vision models; on a 2x crop they are cheap. The protocol therefore counts on
    crops, never on the full image.
    """
    from PIL import Image

    from loramake import mask_spec, region_detect, region_overlay
    from loramake.region_mask import MaskError

    spec = mask_spec.parse_spec({**_spec_fields(source, payload), "keep_outs": []})
    with Image.open(source) as img:
        size = (img.width, img.height)
        src = img.convert("RGB")
    boxes = list(spec.boxes)
    if spec.detect:
        detections = region_detect.detect_regions(source)
        boxes += mask_spec.select_detections(spec, detections)
    if not boxes:
        raise MaskError("crops needs at least one boxes/cells/detect selection")

    pad = 24
    # Touching cells describe one region; crop the merged extent, not each cell,
    # or the subject gets cut at every cell boundary and counts become unanswerable.
    boxes = region_overlay.merge_boxes(boxes, gap=pad)
    tiles = []
    at_border: list[list[str]] = []
    for box in boxes:
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1 = min(size[0], box.x + box.width + pad)
        y1 = min(size[1], box.y + box.height + pad)
        # The enclosure gate needs to know which edges cannot be widened. The clamp above
        # already decides that, so report it instead of leaving the caller to judge it.
        at_border.append(
            [
                name
                for name, hit in (
                    ("left", x0 == 0),
                    ("top", y0 == 0),
                    ("right", x1 == size[0]),
                    ("bottom", y1 == size[1]),
                )
                if hit
            ]
        )
        crop = src.crop((x0, y0, x1, y1))
        tiles.append(crop.resize((crop.width * 2, crop.height * 2)))
    sheet_w = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    sheet_h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    out_path = _scratch_path(Path(source).stem, "crops")
    sheet.save(out_path)
    envelope = {
        "success": True,
        "phase": "crops",
        "source": source,
        "boxes": [list(b.as_tuple()) for b in boxes],
        "at_image_border": at_border,
        "crops": str(out_path),
        "next": (
            "first check enclosure on all four edges of each crop: the outermost content "
            "must be non-subject, and a crop whose subject touches an edge is invalid — "
            "add the neighbouring cell in that direction and re-run crops. The exception "
            'is an edge listed in "at_image_border": the source image ends there, so it '
            "encloses the subject and cannot be widened. Then verify the detail up close "
            "(e.g. count toes per foot twice, by tips and by toenails, per the delegation "
            "playbook; a healthy foot counts 5) — in a style that draws no consistent nail "
            "plate the tip count alone is authoritative and the nail count is void. Then "
            "call phase=preview with the confirmed selection"
        ),
    }
    return envelope, str(out_path)


def _region_preview(source: str, payload: dict) -> tuple[dict, str | None]:
    from loramake import mask_spec, region_overlay
    from loramake.region_mask import coverage

    spec, src, mask, _ = _compile_from_payload(source, payload)
    box = mask_spec.derive_bbox(mask)
    preview = region_overlay.mask_preview(
        src, mask, boxes=spec.boxes, keep_outs=spec.keep_outs, work_box=box
    )
    stem = Path(source).stem
    mask_path = _scratch_path(stem, "mask")
    preview_path = _scratch_path(stem, "preview")
    mask.save(mask_path)
    preview.save(preview_path)
    envelope = {
        "success": True,
        "phase": "preview",
        "source": source,
        "spec": json.loads(mask_spec.canonical_json(spec)),
        "bbox": list(box.as_tuple()),
        "coverage": round(coverage(mask), 4),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "mask_path": str(mask_path),
        "preview": str(preview_path),
        "next": (
            "if the red area is exactly what should be repainted, call again with the "
            "same selection and phase=commit; otherwise adjust boxes/keep_outs"
        ),
    }
    return envelope, str(preview_path)


def _region_commit(prompt: str, source: str, payload: dict) -> tuple[dict, str | None]:
    from loramake import mask_spec

    spec = mask_spec.parse_spec(_spec_fields(source, payload))

    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_region_argv(
        prompt, source, mask_spec.canonical_json(spec), spec.mode, json_out
    )
    envelope = _spawn_gen(argv, json_out)
    envelope.setdefault("phase", "commit")
    path = envelope.get("host_image")
    return envelope, path if envelope.get("success") else None


def run_region(prompt: str, source: str | None, region: str | dict) -> tuple[dict, str | None]:
    """Dispatch one region phase. Returns (envelope, image path to attach or None).

    ``region`` may arrive as a JSON string or as an already-decoded object:
    MCP clients are free to parse a JSON-looking argument before it reaches us.
    """
    sys.path.insert(0, str(REPO))
    from loramake.region_mask import MaskError

    if _cloud_mode():
        return (
            fail_envelope(
                "region phases read local image files and repaint on a local engine; "
                "not available from a cloud session",
                help=REGION_HELP,
            ),
            None,
        )

    if isinstance(region, dict):
        payload = dict(region)
    else:
        try:
            payload = json.loads(region)
        except (TypeError, json.JSONDecodeError) as exc:
            return fail_envelope(f"region is not valid JSON: {exc}", help=REGION_HELP), None
    if not isinstance(payload, dict):
        return fail_envelope("region must be a JSON object", help=REGION_HELP), None

    phase = payload.get("phase")
    if phase not in REGION_PHASES:
        return (
            fail_envelope(
                f"region phase must be one of {', '.join(REGION_PHASES)}", help=REGION_HELP
            ),
            None,
        )
    if not source:
        return fail_envelope("region editing needs a source image path", help=REGION_HELP), None
    src = Path(source).expanduser()
    if not src.is_file():
        return fail_envelope(f"source not found: {source}"), None
    source = str(src)

    try:
        if phase == "inspect":
            return _region_inspect(source, payload)
        if phase == "crops":
            return _region_crops(source, payload)
        if phase == "preview":
            return _region_preview(source, payload)
        text = (prompt or "").strip()
        if not text:
            return fail_envelope("commit needs a prompt describing the repaint"), None
        return _region_commit(text, source, payload)
    except MaskError as exc:
        return fail_envelope(str(exc), phase=phase, help=REGION_HELP), None


# --------------------------------------------------------------------------- #
# MCP surface
# --------------------------------------------------------------------------- #

def _has_non_default(
    *,
    tier: str | None,
    box: str | None,
    base: str | None,
    lora: str | None,
    loras: object,
    style_lora: object,
    ref: str | None,
    n: int | None,
    seed: int | None,
    res: str | None,
    nsfw: bool,
) -> list[str]:
    """List fresh/structure knobs that are set to non-default values."""
    bad: list[str] = []
    if tier:
        bad.append("tier")
    if box and box != "auto":
        bad.append("box")
    if base:
        bad.append("base")
    if lora:
        bad.append("lora")
    if loras:
        bad.append("loras")
    if style_lora:
        bad.append("style_lora")
    if ref:
        bad.append("ref")
    if n is not None:
        bad.append("n")
    if seed is not None:
        bad.append("seed")
    if res:
        bad.append("res")
    if nsfw:
        bad.append("nsfw")
    return bad


INSTRUCTIONS = (
    "Local BoxaComfy Studio image generation. Fresh images are queued to the "
    "Studio ``/api/jobs`` endpoint by default; set the environment variable "
    "``LORAMAKE_IMAGE_DIRECT=1`` only for an authorized direct local run. "
    "Call studio_generate for on-machine ComfyUI images and follow-up edits. "
    "Lint every fresh prompt with scripts/deai_lint.py prompt (image-deslop skill) until it "
    "reports zero errors. "
    "image_generate / image_edit are the Qwen-Image line: natural-language prompts, and edits of "
    "any local image file or earlier result, with size and format handled for you. "
    "xAI/Grok cloud images still use the built-in image_generate tool — "
    "do not replace that path with this one. "
    "Pass the previous image path as source when editing. "
    "For a localized repaint, keep source and walk the region phases: "
    "inspect (labelled grid + detector hits) -> preview (red mask "
    "overlay to confirm) -> commit (repaint only the masked area). "
    "For a structurally broken image (extra limbs, impossible joints, "
    "an object fused into the body, broken proportions) pass "
    "rescue=true with the source instead: one call re-renders it with "
    "its pose locked; add prompt only to correct a literalised token. "
    "Do not use terminal/gen.py for chat-visible local generation. "
    "A render takes minutes. To start several and collect them together, "
    "pass wait=false to get a job id back at once and then call studio_job "
    "for each; the default still blocks until the image is ready. "
    "Tag fresh queued renders with group (or set LORAMAKE_SESSION_GROUP) "
    "so the session's images accumulate into one experiment on /sessions/. "
    "When the Studio API is an https tunnel address (a cloud session), the whole "
    "queue surface stays available: fresh tiers, source edits (pass the earlier "
    "job id as source — the promote tier), rescue (rescue=true with the source "
    "job id), and sequence/artistic identity anchors (ref=<job id>). Jobs are "
    "tagged with the session channel: reads (studio_job, studio_jobs) see only "
    "this channel's results, media references resolve only inside it, and "
    "finished images come back as webp previews. Region phases still need local "
    "files and engines, so they stay LAN-only."
)


def _civitai_dispatch(prompt: str, *, local_only: dict, base: str | None, loras: list[str] | None,
                      n: int | None, seed: int | None, res: str | None, steps: int | None,
                      cfg: float | None, wait: bool, group: str | None, quote: bool) -> list:
    """studio_generate(provider="civitai"): refuse local-only knobs, then quote or queue."""
    bad = [name for name, value in local_only.items() if value]
    if bad:
        envelope = fail_envelope(f"provider=civitai does not accept {', '.join(bad)}")
        return [json.dumps(envelope, ensure_ascii=False)]
    if quote:
        envelope = run_civitai_quote(
            prompt, model=base, loras=loras, n=n, seed=seed, res=res, steps=steps, cfg=cfg,
        )
        return [json.dumps(envelope, ensure_ascii=False)]
    envelope = run_civitai_generate(
        prompt, model=base, loras=loras, n=n, seed=seed, res=res, steps=steps, cfg=cfg,
        wait=wait, group=group,
    )
    return _tool_blocks(envelope, envelope.get("host_image"))


def register(mcp) -> None:
    """Attach studio_generate / studio_job / image_generate / image_edit to an MCPServer (shared with scripts/loramake_mcp.py)."""

    @mcp.tool(name="studio_generate", structured_output=False)
    def studio_generate(
        prompt: str,
        source: str | None = None,
        region: str | dict | None = None,
        rescue: bool = False,
        tier: str | None = None,
        box: str | None = None,
        base: str | None = None,
        lora: str | None = None,
        loras: list[str] | None = None,
        style_lora: str | None = None,
        ref: str | None = None,
        n: int | None = None,
        seed: int | None = None,
        res: str | None = None,
        nsfw: bool = False,
        wait: bool = True,
        group: str | None = None,
        provider: str = "local",
        steps: int | None = None,
        cfg: float | None = None,
        quote: bool = False,
    ) -> list:
        """Generate or edit an image on the local Studio/ComfyUI pipeline, or on Civitai's cloud.

        Not the xAI image_generate tool. Fresh image: pass only ``prompt``; it
        is queued through the Studio ``/api/jobs`` queue with tier defaulting to
        ``draft`` and box defaulting to ``auto``. Set ``LORAMAKE_IMAGE_DIRECT=1``
        to bypass the queue for an authorized local run.

        Before a fresh prompt (image-deslop skill): fix a moment, a sourced light, a
        camera distance and angle, one artist voice and a muted palette with one
        accent, then run ``python3 scripts/deai_lint.py prompt --line LINE --base PROMPT``
        until it reports zero errors. LINE is ``anima`` for the draft tier and the
        Civitai ``anima`` / ``dreamland5`` bases, ``illustrious`` for best / artistic /
        sequence and the other Civitai bases.

        Whole-image follow-up edit: pass the previous image path as ``source``
        plus the new instruction in ``prompt``; this runs direct. Localized
        repaint: also pass ``region`` (a JSON object, or the same JSON as a
        string) and walk phases ``inspect`` -> ``preview`` -> ``commit``; each
        phase returns an image to look at before any GPU is spent. Structure rescue (extra limbs, impossible joints, object
        fused into the body, broken proportions): pass ``rescue=true`` with
        ``source`` — one call re-renders the image with its pose locked; leave
        ``prompt`` empty to reuse the source prompt, or pass a corrected prompt
        when a token was literalised into an object.

        Optional fresh-generation knobs: ``tier`` (draft|best|artistic|sequence|moss),
        ``box`` (auto|boxa|boxb), ``base`` (quirk-card stem), ``lora`` (primary),
        ``loras`` (extra, <=4), ``style_lora``, ``ref`` (identity anchor for
        sequence/artistic), ``n``, ``seed``, ``res`` (WxH), and ``nsfw``.
        ``tier="moss"`` is the MOSS-Image line: queued only, BOXA only (box defaults
        to boxa), SFW only, n <= 4, and only a webp preview comes back -- the original
        stays on BOXA. Its own knobs (ratio, negative, recipe, vary, upscale) live on
        the /moss/ page and the Studio API, not on this tool.

        Civitai cloud (paid in Buzz): ``provider="civitai"`` renders a fresh image through the
        same Studio queue. ``base`` is a model alias -- ``noobai`` (default), ``wai``
        (WAI-illustrious), ``anima`` (Anima aesthetic-v1.1), ``dreamland5`` (the Anima house look: Dreamland ANM plus
        its five style LoRAs and their triggers, added for you), ``flux1`` -- or a checkpoint AIR
        (``urn:air:sdxl:checkpoint:civitai:ID@VER``); each alias brings its own size, steps,
        cfg and negative prompt (configs/civitai_models.json), overridable with ``res``,
        ``steps``, ``cfg``. ``loras`` is a list of ``AIR[:strength]`` strings (default 0.8,
        negative allowed, at most 5) whose ecosystem must match the checkpoint. Anima prompts
        follow the Anima tag rules (no quality tags, artists as ``@name``). Every call gets a
        free Civitai quote first and runs only within the standing budget the user set (per
        call <= 30 Buzz, per day <= 150 Buzz); otherwise it fails with ``provider_budget`` and
        nothing is charged. ``quote=true`` returns that price, the resolved model and LoRAs and
        the budget verdict without submitting anything. SFW only; ``source``, ``region``,
        ``rescue``, ``box``, ``lora``, ``style_lora``, ``ref`` and non-best ``tier`` are local-only.

        ``group`` tags the job with a session label (``[A-Za-z0-9_-]{1,40}``) so
        all of this session's renders accumulate into one experiment on the
        /sessions/ page; when omitted, ``LORAMAKE_SESSION_GROUP`` from the MCP
        config supplies the default. Only queued generation accepts it — on the
        LAN that is fresh renders; edits and rescue run direct with no job to tag.

        ``wait`` defaults to true and blocks until the image is ready, which can be
        minutes. Pass ``wait=false`` to get ``{job_id, pending: true}`` back as soon
        as the job is queued, then poll with ``studio_job``; that is how you start
        several renders and collect them together instead of one at a time. It needs
        the Studio queue, so it is rejected for source edits, region phases, rescue
        and ``LORAMAKE_IMAGE_DIRECT=1``, all of which run direct with nothing to poll —
        except off-LAN (https Studio API), where source edits and rescue ride the queue
        too: pass the earlier job id as ``source`` (promote tier) or with
        ``rescue=true``, and an id as ``ref`` for sequence/artistic.
        """
        if provider == "civitai":
            return _civitai_dispatch(
                prompt, local_only={
                    "source": source, "region": region, "rescue": rescue, "box": box,
                    "lora": lora, "style_lora": style_lora, "ref": ref, "nsfw": nsfw,
                    "tier": tier not in (None, "best"),
                },
                base=base, loras=loras, n=n, seed=seed, res=res, steps=steps, cfg=cfg,
                wait=wait, group=group, quote=quote,
            )
        if provider != "local":
            return [json.dumps(fail_envelope("provider must be local or civitai"), ensure_ascii=False)]
        if quote or steps is not None or cfg is not None:
            envelope = fail_envelope("quote, steps and cfg apply to provider=civitai only")
            return [json.dumps(envelope, ensure_ascii=False)]

        if rescue and region is not None:
            envelope = fail_envelope("pass either rescue or region, not both")
            return [json.dumps(envelope, ensure_ascii=False)]

        # Region phases always run direct (LAN-only: local files + engine); off-LAN rescue
        # rides the Studio queue, so wait=false and group stay meaningful for it there.
        runs_direct = region is not None or (rescue and not _cloud_mode())
        if not wait and runs_direct:
            which = "region" if region is not None else "rescue"
            envelope = fail_envelope(
                f"wait=false is for fresh queued generation; {which} runs direct and blocks"
            )
            return [json.dumps(envelope, ensure_ascii=False)]

        if group and runs_direct:
            which = "region" if region is not None else "rescue"
            envelope = fail_envelope(
                f"group is only valid for fresh queued generation; {which} runs direct"
            )
            return [json.dumps(envelope, ensure_ascii=False)]

        if region is not None:
            bad = _has_non_default(
                tier=tier, box=box, base=base, lora=lora, loras=loras,
                style_lora=style_lora, ref=ref, n=n, seed=seed, res=res, nsfw=nsfw,
            )
            if bad:
                envelope = fail_envelope(
                    f"region does not accept {', '.join(bad)}; pass only prompt, source, and region"
                )
                return [json.dumps(envelope, ensure_ascii=False)]
            envelope, image_path = run_region(prompt, source, region)
            return _tool_blocks(envelope, image_path)

        if rescue:
            bad = _has_non_default(
                tier=tier, box=box, base=base, lora=lora, loras=loras,
                style_lora=style_lora, ref=ref, n=n, seed=seed, res=res, nsfw=nsfw,
            )
            if _cloud_mode():
                # On the queued path nsfw is the zone selector for the source id, not a knob.
                bad = [name for name in bad if name != "nsfw"]
            if bad:
                envelope = fail_envelope(
                    f"rescue does not accept {', '.join(bad)}; pass only prompt and source"
                )
                return [json.dumps(envelope, ensure_ascii=False)]
            envelope = run_rescue(prompt, source, wait=wait, group=group, nsfw=nsfw)
            return _tool_blocks(envelope, envelope.get("host_image"))

        # Whole-image source edit: reject fresh-only tier/resource knobs. Off-LAN the
        # queued promote tier accepts base/res, so run_generate's cloud branch validates.
        if source and not _cloud_mode():
            bad = _source_incompatible(
                tier=tier, base=base, ref=ref, style_lora=style_lora, res=res
            )
            if bad:
                envelope = fail_envelope(bad)
                return [json.dumps(envelope, ensure_ascii=False)]

        envelope = run_generate(
            prompt,
            source=source,
            tier=tier,
            box=box,
            base=base,
            lora=lora,
            loras=loras,
            style_lora=style_lora,
            ref=ref,
            n=n,
            seed=seed,
            res=res,
            nsfw=nsfw,
            wait=wait,
            group=group,
        )
        return _tool_blocks(envelope, envelope.get("host_image"))

    @mcp.tool(name="studio_job", structured_output=False)
    def studio_job(job_id: str, wait_s: float = 0.0) -> list:
        """Check a job started with ``studio_generate(wait=false)``.

        With ``wait_s`` at 0 this returns the job's current state immediately. With a
        positive ``wait_s`` it waits up to that many seconds for the job to finish before
        answering. Either way, a job that is still running comes back as
        ``{pending: true, status}`` -- that is not an error, just "not yet"; call again.
        A finished job returns the same envelope and image block ``studio_generate``
        would have. This never cancels anything.
        """
        jid = (job_id or "").strip()
        if not jid:
            return [json.dumps(fail_envelope("studio_job needs a job_id"), ensure_ascii=False)]
        try:
            budget = max(0.0, min(float(wait_s), JOB_WAIT_MAX_S))
        except (TypeError, ValueError):
            return [json.dumps(fail_envelope(f"wait_s must be a number, got {wait_s!r}"),
                               ensure_ascii=False)]
        envelope = _poll_job(jid, wait_s=budget)
        return _tool_blocks(envelope, envelope.get("host_image"))

    @mcp.tool(name="studio_jobs", structured_output=False)
    def studio_jobs(q: str = "", status: str = "", limit: int = 20, offset: int = 0) -> list:
        """Search this channel's Studio job history (read-only, starts nothing).

        ``q`` substring-matches the brief (prompt, LoRA names, tier), ``status``
        filters exactly (queued|running|done|error|cancelled|interrupted). Returns
        ``{total, jobs: [compact rows]}`` — call ``studio_job`` on a row's job_id for
        the full record and the image block. Off-LAN the session channel scopes the
        listing, so a cloud session only ever sees its own channel's jobs.
        """
        envelope = run_jobs_query(q=q, status=status, limit=limit, offset=offset)
        return [json.dumps(envelope, ensure_ascii=False)]

    @mcp.tool(name="media_ingest", structured_output=False)
    def media_ingest(path: str, label: str | None = None, source: str | None = None) -> list:
        """Put one local PNG/JPEG into the Studio SFW gallery (starts no job, spends no GPU).

        ``path`` is a file on this machine (max 32 MB). Returns the Studio reply unchanged:
        ``{media_id, zone, width, height}``. Send that ``med_…`` id as ``source_media_id``
        (promote / rescue) or ``ref_media_id`` (qwen21 / sequence / artistic) in a job brief,
        or pass it to ``image_edit`` as ``image``.
        Identical bytes come back with the same id. ``label`` is free text kept in the
        sidecar; ``source`` tags the provenance (default ``external``). Errors carry the
        backend's detail text.
        """
        return [json.dumps(run_media_ingest(path, label=label, source=source), ensure_ascii=False)]

    @mcp.tool(name="image_generate", structured_output=False)
    def image_generate(
        prompt: str,
        ratio: str | None = None,
        size: str | None = None,
        n: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        preview: bool | None = None,
        wait: bool = True,
    ) -> list:
        """Generate an image from a natural-language prompt on the local Qwen-Image line (BOXA).

        Write full sentences, not tags: subject, hair length, clothing, setting, light, and one
        medium sentence such as "The image is a detailed anime illustration."; no quality words.
        Before calling, fix a moment, a sourced light, a camera and a muted palette with one
        accent (image-deslop skill), then run
        ``python3 scripts/deai_lint.py prompt --line qwen21 --base PROMPT`` until it reports
        zero errors.
        ``ratio`` (1:1 | 2:3 | 3:2) and ``size`` (draft | std | large | 2k) pick a validated
        canvas; default 2:3 std = 832x1216. Valid pairs: 1:1 draft/std/2k, 2:3 std/large,
        3:2 std/large. ``n`` images share ``seed`` + index. Returns the saved PNG path plus a
        preview; pass that path to ``image_edit`` to change it. ``preview`` fetches the sidecar's
        small webp instead of the PNG (default: automatic, on whenever BOXA is not on the LAN; the
        PNG stays on BOXA). ``wait=false`` returns a job id for ``studio_job``. Runs through the
        Studio queue only.
        """
        envelope = run_image_generate(prompt, ratio=ratio, size=size, n=n, seed=seed, steps=steps,
                                      preview=preview, wait=wait)
        return _tool_blocks(envelope, envelope.get("host_image"))

    @mcp.tool(name="image_edit", structured_output=False)
    def image_edit(
        image: str,
        prompt: str,
        n: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        preview: bool | None = None,
        wait: bool = True,
    ) -> list:
        """Edit an existing image with a natural-language instruction on the local Qwen-Image line.

        ``image`` is a local file path (PNG / JPEG / WebP / GIF of any size: orientation, format and
        size are normalized for you) or a gallery / job id from an earlier result. An image pasted
        into chat has no path: save or drop it as a file first. Say what changes and what must stay
        (identity, pose, outfit), and state hair length. The result keeps the image's aspect ratio
        at about one megapixel; aspect ratios beyond about 1:3 are unverified and are not cropped.
        Returns the saved image path plus a preview; pass that path back as ``image`` to keep editing.
        ``preview`` as in ``image_generate`` (automatic when BOXA is not on the LAN). ``wait=false``
        returns a job id for ``studio_job``. Runs through the Studio queue only.
        """
        envelope = run_image_edit(image, prompt, n=n, seed=seed, steps=steps, preview=preview, wait=wait)
        return _tool_blocks(envelope, envelope.get("host_image"))


def main() -> None:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("loramake-image", instructions=INSTRUCTIONS)
    register(mcp)
    mcp.run(transport="stdio")


def _preview_enabled() -> bool:
    return os.environ.get("LORAMAKE_MCP_PREVIEW", "1").strip().lower() not in {"0", "false", "no", "off"}


def agent_preview(image_path: str) -> str:
    """A small webp copy for the inline image block, or the original path if anything goes wrong.

    Best-effort by design: a preview that cannot be made must degrade to showing the full image,
    never to showing nothing. The source file is only ever read.
    """
    if not _preview_enabled():
        return image_path
    try:
        from PIL import Image as PILImage

        source = Path(image_path)
        with PILImage.open(source) as img:
            img.load()
            if max(img.size) > PREVIEW_EDGE:
                img.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), PILImage.LANCZOS)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            dest = _scratch_dir() / f"{source.stem}_preview_{os.getpid()}.webp"
            img.save(dest, "WEBP", quality=PREVIEW_QUALITY, method=4)
        return str(dest) if dest.stat().st_size < source.stat().st_size else image_path
    except Exception:
        return image_path


def _cloud_outputs_dir() -> Path:
    """Where off-LAN previews land; the cloud env pins this outside the repo tree."""
    root = Path(os.environ.get("LORAMAKE_OUTPUTS", "") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fetch_job_preview(job_id: str) -> str | None:
    """Pull the server-side webp preview of a finished job; None on any failure.

    Off-LAN the archival PNG stays on the host, so this Studio endpoint is the only way
    the image block reaches the agent. Bytes are capped like any other response and must
    actually be webp before they touch the disk.
    """
    url = _api_path(f"/api/jobs/{job_id}/preview")
    req = urllib.request.Request(url, headers=_request_headers())
    try:
        with _urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            data = _read_limited(resp, MAX_RESPONSE_BYTES)
    except Exception:
        return None
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    dest = _cloud_outputs_dir() / f"job_{job_id}_preview.webp"
    try:
        dest.write_bytes(data)
    except OSError:
        return None
    return str(dest)


def _tool_blocks(envelope: dict, image_path: str | None) -> list:
    """The JSON envelope plus, when there is one, the image the agent looks at.

    The envelope keeps pointing at the archival PNG — that path is what a follow-up call passes
    as --source — while the block itself carries the shrunk copy. Off-LAN the PNG is not a
    local file, so the block falls back to the job's webp preview from Studio.
    """
    blocks: list = [json.dumps(envelope, ensure_ascii=False)]
    shown: str | None = None
    if image_path and Path(image_path).is_file():
        shown = agent_preview(image_path)
    elif envelope.get("success") and envelope.get("job_id"):
        shown = _fetch_job_preview(str(envelope["job_id"]))
    if shown:
        from mcp.server.mcpserver import Image

        blocks.append(Image(path=shown))
    return blocks


if __name__ == "__main__":
    main()
