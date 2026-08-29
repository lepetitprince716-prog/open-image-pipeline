#!/usr/bin/env python3
"""Driver helpers for the Forge (Stable Diffusion WebUI Forge) line on Z3.

The A1111-lineage sibling of ``loramake.comfy``: HTTP transport to Forge's
``/sdapi/v1`` API, enum probes with the same L1/L2 cache pattern, PNG
``parameters``-chunk embedding, and result saving through the shared
``save_image_with_meta`` (which feeds the gallery sidecar + the
``GEN_FILES_OUT`` manifest protocol).

Transport is curl-subprocess like the ComfyUI driver, because the Mac runs an
HTTPS proxy that must be bypassed for loopback calls (``_access_argv``).
Forge's txt2img is synchronous — one long-lived POST returns the finished
images — so there is no /history poll loop here; ``FORGE_TIMEOUT`` (default
1800s) bounds the request.  Endpoint selection lives in ``forge_endpoints``.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path
from urllib.parse import urlparse

from loramake.comfy import _access_argv, atomic_write_text, looks_like_png, save_image_with_meta
from loramake.config import ROOT


class ForgeError(RuntimeError):
    """A Forge API call failed or returned an unusable response."""


API = "/sdapi/v1"
ENDPOINT_MODELS = f"{API}/sd-models"
ENDPOINT_VAES = f"{API}/sd-vae"
ENDPOINT_LORAS = f"{API}/loras"
ENDPOINT_SAMPLERS = f"{API}/samplers"
ENDPOINT_SCHEDULERS = f"{API}/schedulers"
ENDPOINT_UPSCALERS = f"{API}/upscalers"
ENDPOINT_LATENT_MODES = f"{API}/latent-upscale-modes"
ENDPOINT_EMBEDDINGS = f"{API}/embeddings"
ENDPOINT_SCRIPTS = f"{API}/scripts"
ENDPOINT_OPTIONS = f"{API}/options"
ENDPOINT_TXT2IMG = f"{API}/txt2img"
# ADetailer extension surface (not core API); absence degrades to a warning.
ENDPOINT_AD_MODELS = "/adetailer/v1/ad_model"

_ENUM_DISK_SCHEMA = "loramake.forge-enum-cache/v1"
_ENUM_DISK_TTL_DEFAULT = 900.0
_ENUM_L1_TTL = 10.0
_ENUM_L1: dict[tuple[str, str], tuple[float, object]] = {}
_ENUM_LOCK = threading.Lock()

TUNNEL_HINT = (
    "is the com.loramake.forge-z3-loopback tunnel loaded and Forge running on Z3? "
    "(check: launchctl list | grep forge-z3; start Forge: scripts/forge_z3_control.ps1 "
    "via rdbg — see docs/forge_line.md)"
)


def default_timeout() -> float:
    try:
        return float(os.environ.get("FORGE_TIMEOUT", "1800"))
    except ValueError:
        return 1800.0


def _request(host: str, endpoint: str, *, payload: dict | None = None,
             timeout: float = 30.0) -> tuple[int, object]:
    """One curl round-trip. Returns (http_status, parsed_body).

    The body is parsed as JSON when possible, else returned as text.  curl-level
    failures (unreachable, hung) raise ForgeError; HTTP-level failures are the
    caller's to interpret — FastAPI's 422 detail is the diagnostic that matters.
    """
    base = host.rstrip("/")
    noproxy = urlparse(base).hostname or "127.0.0.1"
    fd, tmp = tempfile.mkstemp(prefix="forge_body_", suffix=".json")
    os.close(fd)
    argv = ["curl", "-sS", *_access_argv(noproxy), "--max-time", str(timeout),
            "-o", tmp, "-w", "%{http_code}"]
    raw_payload = None
    if payload is not None:
        argv += ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"]
        raw_payload = json.dumps(payload)
    argv.append(f"{base}{endpoint}")
    try:
        proc = subprocess.run(argv, input=raw_payload, capture_output=True, text=True,
                              timeout=timeout + 5)
        if proc.returncode != 0:
            detail = (proc.stderr or "no detail").strip()[-240:]
            raise ForgeError(f"{endpoint} unreachable (curl rc={proc.returncode}: {detail}) — {TUNNEL_HINT}")
        status = int(proc.stdout.strip()) if proc.stdout.strip().isdigit() else 0
        body_text = Path(tmp).read_text(encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise ForgeError(f"{endpoint} curl hung past {timeout + 5:g}s") from exc
    finally:
        Path(tmp).unlink(missing_ok=True)
    try:
        return status, json.loads(body_text)
    except json.JSONDecodeError:
        return status, body_text


def ping(host: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """GET /options as a liveness probe; the failure string names the tunnel."""
    try:
        status, body = _request(host, ENDPOINT_OPTIONS, timeout=timeout)
    except ForgeError as exc:
        return False, str(exc)
    if status == 200 and isinstance(body, dict):
        return True, f"Forge up at {host} (checkpoint: {body.get('sd_model_checkpoint')!r})"
    return False, f"Forge at {host} answered HTTP {status} on {ENDPOINT_OPTIONS} — {TUNNEL_HINT}"


# --- enum probes (L1 in-process + L2 disk, mirroring comfy.object_info_enum) --------------

def _enum_disk_enabled() -> bool:
    return os.environ.get("LORAMAKE_ENUM_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _enum_disk_ttl() -> float:
    try:
        return float(os.environ.get("LORAMAKE_FORGE_ENUM_TTL", str(_ENUM_DISK_TTL_DEFAULT)))
    except ValueError:
        return _ENUM_DISK_TTL_DEFAULT


def _enum_disk_path() -> Path:
    override = os.environ.get("LORAMAKE_FORGE_ENUM_CACHE_PATH", "").strip()
    return Path(override).expanduser() if override else ROOT / "app" / "data" / "forge_enum_cache_v1.json"


def _enum_disk_load() -> dict:
    try:
        data = json.loads(_enum_disk_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != _ENUM_DISK_SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _enum_disk_read(key: tuple[str, str]) -> object | None:
    if not _enum_disk_enabled():
        return None
    entry = _enum_disk_load().get("|".join(key))
    if not isinstance(entry, dict):
        return None
    observed_at = entry.get("observed_at")
    if not isinstance(observed_at, (int, float)):
        return None
    if not (0.0 <= time.time() - observed_at <= _enum_disk_ttl()):
        return None
    return entry.get("body")


def _enum_disk_write(key: tuple[str, str], body: object) -> None:
    if not _enum_disk_enabled():
        return
    path = _enum_disk_path()
    with _ENUM_LOCK:
        entries = _enum_disk_load()
        entries["|".join(key)] = {"body": body, "observed_at": time.time()}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(
                {"schema": _ENUM_DISK_SCHEMA, "entries": entries}, ensure_ascii=False))
            os.chmod(path, 0o600)
        except OSError:
            pass


def enum(host: str, endpoint: str, *, refresh: bool = False,
         timeout: float = 15.0) -> object | None:
    """Cached GET of one enum endpoint. None = unavailable (404 / unreachable).

    A 404 is cached like a success so a build without an optional endpoint
    (e.g. /schedulers on an older kernel) is probed once, not per call —
    callers must treat None as "cannot verify", never as "absent asset".
    """
    key = (host.rstrip("/"), endpoint)
    now = time.monotonic()
    if not refresh:
        with _ENUM_LOCK:
            hit = _ENUM_L1.get(key)
            if hit is not None and now - hit[0] < _ENUM_L1_TTL:
                return hit[1]
        disk = _enum_disk_read(key)
        if disk is not None:
            body = None if disk == "__unavailable__" else disk
            with _ENUM_LOCK:
                _ENUM_L1[key] = (time.monotonic(), body)
            return body
    try:
        status, body = _request(host, endpoint, timeout=timeout)
    except ForgeError:
        return None
    result: object | None = body if status == 200 else None
    with _ENUM_LOCK:
        _ENUM_L1[key] = (time.monotonic(), result)
    _enum_disk_write(key, result if result is not None else "__unavailable__")
    return result


def _dict_list(body: object) -> list[dict]:
    return [e for e in body if isinstance(e, dict)] if isinstance(body, list) else []


def sd_models(host: str, *, refresh: bool = False) -> list[dict]:
    return _dict_list(enum(host, ENDPOINT_MODELS, refresh=refresh))


def sd_vaes(host: str, *, refresh: bool = False) -> list[dict]:
    return _dict_list(enum(host, ENDPOINT_VAES, refresh=refresh))


def loras(host: str, *, refresh: bool = False) -> list[dict]:
    return _dict_list(enum(host, ENDPOINT_LORAS, refresh=refresh))


def sampler_names(host: str) -> set[str] | None:
    body = enum(host, ENDPOINT_SAMPLERS)
    if body is None:
        return None
    names: set[str] = set()
    for entry in _dict_list(body):
        if entry.get("name"):
            names.add(str(entry["name"]))
        names.update(str(a) for a in entry.get("aliases") or [])
    return names


def scheduler_names(host: str) -> set[str] | None:
    """Accepted scheduler identifiers: internal names + display labels."""
    body = enum(host, ENDPOINT_SCHEDULERS)
    if body is None:
        return None
    names: set[str] = set()
    for entry in _dict_list(body):
        for field in ("name", "label"):
            if entry.get(field):
                names.add(str(entry[field]))
    return names


def upscaler_names(host: str) -> set[str] | None:
    """GAN upscalers + latent modes — the vocabulary hr_upscaler accepts."""
    body = enum(host, ENDPOINT_UPSCALERS)
    if body is None:
        return None
    names = {str(e["name"]) for e in _dict_list(body) if e.get("name")}
    names.update(str(e["name"]) for e in _dict_list(enum(host, ENDPOINT_LATENT_MODES)) if e.get("name"))
    return names


def embedding_names(host: str) -> set[str] | None:
    body = enum(host, ENDPOINT_EMBEDDINGS)
    if not isinstance(body, dict):
        return None
    names: set[str] = set()
    for section in ("loaded", "skipped"):
        part = body.get(section)
        if isinstance(part, dict):
            names.update(str(n) for n in part)
    return names


def adetailer_available(host: str) -> bool | None:
    """True/False when the scripts enum answers; None when it cannot be probed."""
    body = enum(host, ENDPOINT_SCRIPTS)
    if not isinstance(body, dict):
        return None
    txt2img_scripts = body.get("txt2img")
    if not isinstance(txt2img_scripts, list):
        return None
    return any(str(s).lower() == "adetailer" for s in txt2img_scripts)


def adetailer_model_names(host: str) -> set[str] | None:
    body = enum(host, ENDPOINT_AD_MODELS)
    if isinstance(body, dict) and isinstance(body.get("ad_model"), list):
        return {str(n) for n in body["ad_model"]}
    return None


def options(host: str, *, timeout: float = 15.0) -> dict:
    """Live server options (never cached — carries mutable state like the loaded checkpoint)."""
    status, body = _request(host, ENDPOINT_OPTIONS, timeout=timeout)
    if status != 200 or not isinstance(body, dict):
        raise ForgeError(f"{ENDPOINT_OPTIONS} answered HTTP {status}")
    return body


def unload_checkpoint(host: str) -> bool:
    """POST /unload-checkpoint — release the loaded model's VRAM (Forge stays up)."""
    try:
        status, _ = _request(host, f"{API}/unload-checkpoint", payload={}, timeout=60.0)
    except ForgeError:
        return False
    return 200 <= status < 300


def refresh_models(host: str) -> None:
    """Ask Forge to re-scan checkpoints + LoRAs (needed after junctioning new files).

    Note the deployment caveat: newly junctioned *checkpoints* additionally need a Forge
    restart before they render correctly (docs/forge_line.md).
    """
    for endpoint in (f"{API}/refresh-checkpoints", f"{API}/refresh-loras"):
        try:
            _request(host, endpoint, payload={}, timeout=60.0)
        except ForgeError:
            pass
    with _ENUM_LOCK:
        _ENUM_L1.clear()
    for endpoint in (ENDPOINT_MODELS, ENDPOINT_LORAS):
        enum(host, endpoint, refresh=True)


# --- generation ---------------------------------------------------------------------------

def txt2img(host: str, payload: dict, *, timeout: float | None = None) -> tuple[list[bytes], dict]:
    """Synchronous POST /txt2img. Returns (png_bytes_list, info_dict).

    Non-200 responses raise ForgeError carrying the response body verbatim — the
    FastAPI 422 detail is exactly how payload-shape bugs (script_name vs
    alwayson_scripts) get diagnosed, so it is never swallowed.
    """
    budget = default_timeout() if timeout is None else timeout
    status, body = _request(host, ENDPOINT_TXT2IMG, payload=payload, timeout=budget)
    if status != 200:
        detail = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
        raise ForgeError(f"txt2img HTTP {status}: {detail[:2000]}")
    if not isinstance(body, dict) or not isinstance(body.get("images"), list):
        raise ForgeError("txt2img returned no images array")
    images: list[bytes] = []
    for encoded in body["images"]:
        try:
            raw = base64.b64decode(str(encoded).split(",", 1)[-1])
        except Exception as exc:
            raise ForgeError(f"txt2img image failed base64 decode: {exc}") from exc
        if not looks_like_png(raw):
            raise ForgeError(f"txt2img image is not a PNG ({len(raw)} bytes)")
        images.append(raw)
    info_raw = body.get("info")
    try:
        info = json.loads(info_raw) if isinstance(info_raw, str) else {}
    except json.JSONDecodeError:
        info = {}
    return images, info if isinstance(info, dict) else {}


# --- PNG parameters chunk -----------------------------------------------------------------

def _png_text_chunk(keyword: str, text: str) -> bytes:
    """One A1111-recognizable text chunk: tEXt when latin-1-safe, else uncompressed iTXt."""
    try:
        body = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
        kind = b"tEXt"
    except UnicodeEncodeError:
        body = keyword.encode("latin-1") + b"\x00\x00\x00\x00\x00" + text.encode("utf-8")
        kind = b"iTXt"
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def embed_parameters(raw: bytes, infotext: str) -> bytes:
    """Insert the A1111 ``parameters`` chunk before IEND (idempotent, best-effort).

    Forge's API-returned base64 PNGs do not reliably carry the infotext, so the
    driver re-embeds it from the response's ``info.infotexts[i]`` — that chunk is
    what makes the produced file metadata-round-trippable.
    """
    if not infotext or not looks_like_png(raw):
        return raw
    if b"tEXtparameters\x00" in raw or b"iTXtparameters\x00" in raw:
        return raw
    pos = raw.rfind(b"IEND") - 4  # the 4-byte length field precedes the IEND type
    if pos < 8:
        return raw
    return raw[:pos] + _png_text_chunk("parameters", infotext) + raw[pos:]


def save_results(images: list[bytes], info: dict, *, out_dir: Path, stem: str,
                 meta_base: dict) -> list[Path]:
    """Save each image with its own infotext + seed: parameters chunk first, then the
    loramake iTXt/sidecar layer via ``save_image_with_meta`` (which also appends to the
    GEN_FILES_OUT manifest)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    infotexts = info.get("infotexts") if isinstance(info.get("infotexts"), list) else []
    seeds = info.get("all_seeds") if isinstance(info.get("all_seeds"), list) else []
    saved: list[Path] = []
    for index, raw in enumerate(images):
        infotext = str(infotexts[index]) if index < len(infotexts) else ""
        seed = seeds[index] if index < len(seeds) else info.get("seed")
        path = out_dir / f"{stem}_{index:02d}.png"
        n = 1
        while path.exists():
            path = out_dir / f"{stem}_{index:02d}-{n}.png"
            n += 1
        meta = {**meta_base, "seed": seed, "infotext": infotext}
        save_image_with_meta(path, embed_parameters(raw, infotext), meta)
        saved.append(path)
    return saved
