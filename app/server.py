"""FastAPI backend for the loramake app.

Wraps gen.py subprocesses behind a persistent job queue and exposes the app API
through small APIRouter modules. Keep app orchestration and shared runtime state
here; endpoint groups live in app/routers/.

Run:  .venv/bin/python -m uvicorn app.server:app --port 8873   (project venv: uv sync --extra dev --extra semantic)
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import brief as brief_mod
from . import civitai_bridge as civitai_bridge_mod
from .box_status import BoxStatusCache
from . import experiments
from . import gallery_catalog
from . import jobs
from . import lifecycle
from . import lora_archive
from . import lora_sync
from . import meta as meta_mod
from . import nsfw_meta
from . import remote_diagnostics
from .network import enforcement as network_enforcement
from loramake.config import OUTPUT_CHARACTER

from .network import probes as network_probes
from .network import state as network_state
from .network import status as network_status
from .routers import (
    anima_pilot,
    assist,
    chargen as chargen_router,
    civitai_bridge as civitai_bridge_router,
    diagnostics,
    experiments as experiments_router,
    files,
    gallery,
    gallery_catalog as gallery_catalog_router,
    jobs as jobs_router,
    network as network_router,
    providers as providers_router,
    system,
    thumbs,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen  # noqa: E402  (scripts/gen.py — probe_idle/HOSTS; import has no side effects)

jobs.HOSTS = gen.HOSTS  # progress relay needs the box hosts; jobs.py stays gen-free
jobs.RESOLVE_BOX = gen.resolve_box  # box "auto" -> concrete box at launch (ZZZ-first policy)
# Network kill switches: submit-time (brief), launch-time (jobs), bridge seams,
# and box probe pause all consult the same persisted state.
brief_mod.NETWORK_PROBLEMS = network_enforcement.brief_problems
jobs.NETWORK_GATE = network_enforcement.launch_gate
civitai_bridge_mod.NETWORK_ENABLED = network_state.enabled
BOX_STATUS = BoxStatusCache(gen.HOSTS, lambda host: gen.probe_idle(host),
                            should_probe=network_state.enabled)
# Non-box service probes (lazy: the lambda resolves _lora_manager_url at call time)
NETWORK_PROBES = network_status.ProbeCache({
    "civitai": network_probes.probe_civitai,
    "novelai": network_probes.probe_novelai,
    "lora_manager": lambda: network_probes.probe_lora_manager(_lora_manager_url()),
})

WEB_OUT = ROOT / "app-web" / "out"  # Next static export (app-web/: pnpm build)
OUTPUTS_DIR = (ROOT / "outputs").resolve()
BY_MODEL_DIR = OUTPUTS_DIR / OUTPUT_CHARACTER / "by_model"
GACHA_DIR = BY_MODEL_DIR / "claude" / "daily_gacha"
# NSFW render path is a separate tree with separate endpoints (never mixed into /files,
# per the strict SFW/NSFW separation rule; see nsfw/APP_SWITCH_DESIGN.md).
NSFW_OUTPUTS = (ROOT / "nsfw" / "outputs").resolve()
NSFW_THUMB_CACHE = ROOT / "nsfw" / "app_cache" / "thumbs"  # NSFW derivatives stay under nsfw/
PREVIEW_CACHE = ROOT / "app" / "data" / "preview_cache"
NSFW_PREVIEW_CACHE = ROOT / "nsfw" / "app_cache" / "previews"
LM_PREVIEW_PREFIX = "/api/lm/previews?"  # LoRA-Manager (inside Z3's ComfyUI) preview endpoint
PREVIEW_MAX_EDGE = 384  # picker/models tiles render ~120-180px wide; 384 covers 2x DPR
THUMB_CACHE = ROOT / "app" / "data" / "thumb_cache"
THUMB_WIDTHS = (256, 480, 768)  # fixed steps so one image can't fan out into N cache entries
IMMUTABLE_CACHE = {"Cache-Control": "public, max-age=86400, immutable"}
GALLERY_CACHE_TTL_S = 10
_gallery_cache: dict = {"t": 0.0, "items": []}
_nsfw_gallery_cache: dict = {"t": 0.0, "items": []}
_preview_locks_guard = threading.Lock()
_preview_locks: dict[tuple[str, str], threading.Lock] = {}
_thumb_locks_guard = threading.Lock()
_thumb_locks: dict[tuple[str, int], threading.Lock] = {}

GROK_BIN = Path.home() / ".local" / "bin" / "grok-proxy"
ASSIST_TIMEOUT_S = 180  # grok single-turn measured ~66s wall on this machine (2026-07-03)
MAX_ASSIST_INPUT = 2000
NSFW_ASSIST_PROMPT = ROOT / "nsfw" / "assist_prompt.txt"  # content-owner supplied (optional)
ASSIST_TEMPLATE = (
    "You convert a natural-language image description (Chinese or English) into one "
    "comma-separated Danbooru-style tag prompt for an anime SDXL model (Illustrious). "
    "Output ONLY the tag line - no explanations, no quotes, no markdown. Keep any booru "
    "tags the user already wrote; translate the rest into concrete tags covering subject, "
    "outfit, pose, expression, lighting, composition and mood. Do not invent character names."
)


class StudioStaticFiles(StaticFiles):
    """No-cache HTML shell; immutable caching only for Next's fingerprinted assets."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        content_type = response.headers.get("content-type", "")
        cache_control = None
        if path.endswith(".html") or content_type.startswith("text/html"):
            cache_control = "no-cache"
        elif path.startswith("_next/static/"):
            cache_control = "public, max-age=31536000, immutable"
        if cache_control:
            # FileResponse.headers is a copied MutableHeaders view in this Starlette version;
            # mutate the ASGI header list that FileResponse.__call__ actually sends.
            response.raw_headers[:] = [
                (key, value) for key, value in response.raw_headers if key.lower() != b"cache-control"
            ]
            response.raw_headers.append((b"cache-control", cache_control.encode("ascii")))
            response.__dict__.pop("_headers", None)
        return response


def _fetch_preview(url: str) -> bytes:
    """LAN fetch of a LoRA-Manager preview (module-level so tests can stub it)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=8) as res:
        return res.read()


def _shrink_webp(data: bytes, max_edge: int) -> bytes:
    """Downscale an image payload to max_edge (longest side) and re-encode as WebP."""
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80)
    return buf.getvalue()


def _preview_zone(p: str) -> str:
    """Resolve the catalog rating without copying preview bytes across content zones."""
    for bucket in meta_mod.loras().values():
        for entry in bucket:
            if entry.get("preview") == p:
                return "nsfw" if entry.get("nsfw") else "sfw"
    return "sfw"  # legacy/unknown previews keep their historical public-cache behavior


def _preview_cache_path(p: str, zone: str | None = None) -> Path:
    resolved_zone = zone or _preview_zone(p)
    root = NSFW_PREVIEW_CACHE if resolved_zone == "nsfw" else PREVIEW_CACHE
    return root / f"{hashlib.sha1(p.encode()).hexdigest()}.webp"


def _ensure_preview(p: str) -> Path:
    """Fetch a preview lazily with per-key single-flight and atomic, zone-separated caching."""
    zone = _preview_zone(p)
    cached = _preview_cache_path(p, zone)
    if cached.is_file():
        return cached
    key = (zone, p)
    with _preview_locks_guard:
        lock = _preview_locks.setdefault(key, threading.Lock())
    try:
        with lock:
            if cached.is_file():
                return cached
            legacy = cached.with_suffix(".img")
            if legacy.is_file():
                data = legacy.read_bytes()
            else:
                data = _fetch_preview(f"{gen.HOSTS['z3']}{p}")
                # Valid tiny PNG/WebP previews can be well below 100 bytes and
                # compression size varies by Pillow/platform.  Image.open below
                # is the authoritative payload validation; only reject empties here.
                if not data:
                    raise ValueError("no preview")
            small = _shrink_webp(data, PREVIEW_MAX_EDGE)
            cached.parent.mkdir(parents=True, exist_ok=True)
            temp = cached.with_name(f".{cached.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                temp.write_bytes(small)
                temp.replace(cached)
            finally:
                temp.unlink(missing_ok=True)
    finally:
        with _preview_locks_guard:
            if not lock.locked():
                _preview_locks.pop(key, None)
    return cached


def _warm_previews() -> None:
    """Compatibility no-op: previews are intentionally fetched only on first request."""
    return None


def _startup_lora_refresh() -> None:
    """One drift check at boot; no polling afterwards."""
    try:
        out = lora_sync.refresh()
        if out.get("status") != "no_drift":
            print(f"[lora-sync] startup: {out}")
    except Exception as exc:
        print(f"[lora-sync] startup check failed: {exc}")


def _interrupt_box(host: str) -> bool:
    """POST /interrupt on a box (cancel its current execution). LAN call, proxy-free."""
    if not host:
        return False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        opener.open(urllib.request.Request(f"{host}/interrupt", method="POST"), timeout=8)
        return True
    except Exception:
        return False


# ---- standalone LoRA Manager control (scripts/z3_lora_manager.py owns the SSH/process
# details; the app only kicks it in a background thread and probes reachability) ----
LORA_MANAGER_PORT = 9000
LORA_MANAGER_SCRIPT = ROOT / "scripts" / "z3_lora_manager.py"
LORA_MANAGER_ACTION_TIMEOUT_S = 180
_lora_manager_lock = threading.Lock()
_lora_manager_op: dict = {"running": False, "action": None, "rc": None, "tail": "",
                          "finished_at": None}


def _lora_manager_url() -> str:
    host = urllib.parse.urlsplit(gen.HOSTS["z3"]).hostname or "z3"
    return f"http://{host}:{LORA_MANAGER_PORT}"


def lora_manager_status() -> dict:
    """Fast reachability probe of the Z3 standalone manager plus the last action outcome."""
    url = _lora_manager_url()
    enabled = network_state.enabled("lora_manager")
    if enabled:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LAN, proxy-free
        try:
            opener.open(url + "/", timeout=3)
            reachable = True
        except Exception:
            reachable = False
    else:
        reachable = False  # disabled: report unreachable without probing
    with _lora_manager_lock:
        op = dict(_lora_manager_op)
    return {"reachable": reachable, "enabled": enabled, "url": url, "op": op}


def lora_manager_action(action: str) -> dict:
    """Run z3_lora_manager.py start/stop in a background thread (SSH is slow)."""
    # Turning things off is always allowed; starting a disabled service is not.
    if action == "start" and not network_state.enabled("lora_manager"):
        raise HTTPException(status_code=409, detail="LoRA Manager 已在网络页停用")
    with _lora_manager_lock:
        if _lora_manager_op["running"]:
            return {"status": "busy", "op": dict(_lora_manager_op)}
        _lora_manager_op.update(running=True, action=action, rc=None, tail="")

    def run() -> None:
        try:
            proc = subprocess.run(
                [sys.executable, str(LORA_MANAGER_SCRIPT), action],
                capture_output=True, text=True, timeout=LORA_MANAGER_ACTION_TIMEOUT_S)
            rc, tail = proc.returncode, ((proc.stdout or "") + (proc.stderr or ""))[-400:]
        except Exception as exc:
            rc, tail = -1, str(exc)[-400:]
        with _lora_manager_lock:
            _lora_manager_op.update(running=False, rc=rc, tail=tail, finished_at=time.time())

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "action": action}


# ---- Forge (A1111 kernel on Z3 :7862) service control (scripts/forge_service.py owns
# the SSH/HTTP details; the app only kicks actions in a background thread and probes
# reachability through the 17862 loopback — docs/forge_line.md) ----
FORGE_SERVICE_SCRIPT = ROOT / "scripts" / "forge_service.py"
FORGE_ACTION_TIMEOUT_S = 300
FORGE_ACTIONS = ("start", "stop", "refresh", "unload")
_forge_lock = threading.Lock()
_forge_op: dict = {"running": False, "action": None, "rc": None, "tail": "",
                   "finished_at": None}


def forge_status() -> dict:
    """Fast reachability probe of the Z3 Forge API plus the last action outcome."""
    from loramake import forge as forge_mod
    from loramake.forge_endpoints import resolve_forge_host

    host = resolve_forge_host("z3")
    reachable, detail = forge_mod.ping(host, timeout=4.0)
    checkpoint = None
    if reachable:
        try:
            checkpoint = forge_mod.options(host, timeout=4.0).get("sd_model_checkpoint")
        except Exception:
            checkpoint = None
    with _forge_lock:
        op = dict(_forge_op)
    return {"reachable": reachable, "url": host, "checkpoint": checkpoint,
            "detail": detail, "op": op}


def forge_action(action: str) -> dict:
    """Run forge_service.py <action> in a background thread (SSH start/stop is slow)."""
    with _forge_lock:
        if _forge_op["running"]:
            return {"status": "busy", "op": dict(_forge_op)}
        _forge_op.update(running=True, action=action, rc=None, tail="")

    def run() -> None:
        try:
            proc = subprocess.run(
                [sys.executable, str(FORGE_SERVICE_SCRIPT), action],
                capture_output=True, text=True, timeout=FORGE_ACTION_TIMEOUT_S)
            rc, tail = proc.returncode, ((proc.stdout or "") + (proc.stderr or ""))[-400:]
        except Exception as exc:
            rc, tail = -1, str(exc)[-400:]
        with _forge_lock:
            _forge_op.update(running=False, rc=rc, tail=tail, finished_at=time.time())

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "action": action}


def network_status_snapshot() -> dict:
    """Compose the network dashboard payload from the existing caches (no remote I/O)."""
    with _lora_manager_lock:
        op = dict(_lora_manager_op)
    return network_status.compose_status(
        box_snapshot=BOX_STATUS.snapshot(),
        probe_snapshot=NETWORK_PROBES.snapshot(),
        enabled_map=network_state.all_enabled(),
        lm_url=_lora_manager_url(),
        lm_op=op,
    )


def _assist_llm(prompt: str) -> str:
    """Single-turn grok CLI call (module-level so tests can stub it)."""
    proc = subprocess.run([str(GROK_BIN), "-p", prompt],
                          capture_output=True, text=True, timeout=ASSIST_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"assist backend rc={proc.returncode}")
    return (proc.stdout or "").strip()


def _scan_gallery(root: Path, cache: dict, limit: int, offset: int, *, nsfw: bool = False) -> dict:
    """mtime scan of a gallery root with a short cache; trash/logs/frame dumps excluded."""
    now = time.time()
    if now - cache["t"] > GALLERY_CACHE_TTL_S:
        items: list[tuple[float, str]] = []
        if root.is_dir():
            for path in root.rglob("*.png"):
                sp = str(path)
                if "_trash" in sp or "_logs" in sp or "video_frames" in sp:
                    continue
                try:
                    items.append((path.stat().st_mtime, sp))
                except OSError:
                    continue
        items.sort(reverse=True)
        cache.update(t=now, items=items)
    offset = max(0, offset)
    window = cache["items"][offset:offset + max(1, min(limit, 200))]
    return {"total": len(cache["items"]),
            "items": [{"path": sp, "mtime": mt, "nsfw": nsfw} for mt, sp in window]}


def _sfw_outputs_path(p: str) -> Path:
    """Resolve + guard a client path against the SFW outputs tree."""
    path = Path(p)
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail="absolute path required")
    path = path.resolve()
    if not path.is_relative_to(OUTPUTS_DIR):
        raise HTTPException(status_code=403, detail="outside the outputs tree")
    if any("nsfw" in part.lower() for part in path.parts):
        raise HTTPException(status_code=403, detail="nsfw content is not served here")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such file")
    return path


def _nsfw_outputs_path(p: str) -> Path:
    """Resolve + guard a client path against the NSFW outputs tree."""
    path = Path(p)
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail="absolute path required")
    path = path.resolve()
    if not path.is_relative_to(NSFW_OUTPUTS):
        raise HTTPException(status_code=403, detail="outside the nsfw outputs tree")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such file")
    return path


def _thumb_response(path: Path, w: int, cache_dir: Path) -> FileResponse:
    """Downscaled WebP for a gallery grid; cache key includes mtime.

    Generation is single-flight per (path, width) so a prewarmer and the first browser
    request cannot publish a torn .webp.
    """
    w = min(THUMB_WIDTHS, key=lambda step: abs(step - w))
    key = hashlib.sha1(f"{path}|{w}|{int(path.stat().st_mtime)}".encode()).hexdigest()
    cached = cache_dir / f"{key}.webp"
    if not cached.is_file():
        lock_key = (str(cache_dir), key)
        with _thumb_locks_guard:
            lock = _thumb_locks.setdefault(lock_key, threading.Lock())
        try:
            with lock:
                if not cached.is_file():
                    try:
                        data = _shrink_webp(path.read_bytes(), w)
                    except Exception:
                        raise HTTPException(status_code=415, detail="not a decodable image")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temp = cached.with_name(f".{cached.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
                    try:
                        temp.write_bytes(data)
                        temp.replace(cached)
                    finally:
                        temp.unlink(missing_ok=True)
        finally:
            with _thumb_locks_guard:
                if not lock.locked():
                    _thumb_locks.pop(lock_key, None)
    return FileResponse(cached, media_type="image/webp", headers=IMMUTABLE_CACHE)


def _prewarm_job_media(job: dict) -> None:
    """Warm the result-card thumbnails for a just-finished job. Best-effort; never raises."""
    result = job.get("result") if isinstance(job, dict) else None
    brief = job.get("brief") if isinstance(job, dict) else None
    if not isinstance(result, dict) or not isinstance(brief, dict):
        return
    nsfw = brief.get("nsfw") is True
    primary = result.get("survivors")
    if primary is None:
        primary = result.get("images")
    if primary is None:
        primary = result.get("new_files")
    seen: set[str] = set()
    paths: list[str] = []
    for raw in (primary or []):
        if isinstance(raw, str) and raw and raw not in seen:
            seen.add(raw)
            paths.append(raw)
    for key in ("best", "contact_sheet"):
        raw = result.get(key)
        if isinstance(raw, str) and raw and raw not in seen:
            seen.add(raw)
            paths.append(raw)
    for raw in paths:
        try:
            if nsfw:
                path = _nsfw_outputs_path(raw)
                cache_dir = NSFW_THUMB_CACHE
            else:
                path = _sfw_outputs_path(raw)
                cache_dir = THUMB_CACHE
            for width in (256, 768):
                _thumb_response(path, width, cache_dir)
        except Exception:
            continue


jobs.PREWARM_RESULT_MEDIA = _prewarm_job_media


# SSE streams park a thread per open connection in 15s queue-get waits. On the
# asyncio default executor (min(32, cpu+4) threads) a handful of open tabs could
# exhaust it, so streaming waits get their own roomy pool.
def _fresh_sse_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=64, thread_name_prefix="sse")


SSE_EXECUTOR = _fresh_sse_executor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Recreated per lifespan: the previous one is shut down in `finally`, and a
    # second lifespan in the same process (repeated TestClient sessions) must not
    # inherit a dead executor.
    global SSE_EXECUTOR
    SSE_EXECUTOR = _fresh_sse_executor()
    lifecycle_stop = threading.Event()
    lifecycle_thread = threading.Thread(
        target=lifecycle.maintenance_loop,
        args=(lifecycle_stop,),
        kwargs={"interval_s": lifecycle.MAINTENANCE_INTERVAL_S},
        daemon=True,
        name="result-lifecycle",
    )
    jobs.start_worker()
    lifecycle_thread.start()
    threading.Thread(target=_startup_lora_refresh, daemon=True, name="lora-startup-check").start()
    try:
        yield
    finally:
        lifecycle_stop.set()
        lifecycle_thread.join(timeout=2.0)
        SSE_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="loramake", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
_state = sys.modules[__name__]
app.add_exception_handler(
    experiments.ExperimentError,
    lambda request, exc: experiments_router.problem_response(
        exc, request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:20]}"
    ),
)
app.include_router(jobs_router.create_router(_state))
app.include_router(anima_pilot.create_router(_state))
app.include_router(system.create_router(_state))
app.include_router(diagnostics.create_router(_state))
app.include_router(thumbs.create_router(_state))
app.include_router(gallery.create_router(_state))
app.include_router(assist.create_router(_state))
app.include_router(files.create_router(_state))
app.include_router(experiments_router.create_router(_state))
app.include_router(gallery_catalog_router.create_router(_state))
app.include_router(civitai_bridge_router.create_router(_state))
app.include_router(providers_router.create_router(_state))
app.include_router(network_router.create_router(_state))
app.include_router(chargen_router.create_router(_state))

BRIDGE_WEB = ROOT / "tools" / "civitai-bridge" / "web"
if BRIDGE_WEB.is_dir():
    # Phone / Z3 / any browser. Must mount before the Next export catch-all.
    app.mount(
        "/civitai-bridge",
        StaticFiles(directory=BRIDGE_WEB, html=True),
        name="civitai-bridge",
    )

CHARGEN_VIEWER_WEB = ROOT / "tools" / "chargen-viewer" / "web"
if CHARGEN_VIEWER_WEB.is_dir():
    # three.js viewer for 3d/chargen runs. Must mount before the Next catch-all.
    app.mount(
        "/chargen-viewer",
        StaticFiles(directory=CHARGEN_VIEWER_WEB, html=True),
        name="chargen-viewer",
    )

if WEB_OUT.is_dir():  # API routes above win; everything else falls through to the export
    app.mount("/", StudioStaticFiles(directory=WEB_OUT, html=True), name="web")
