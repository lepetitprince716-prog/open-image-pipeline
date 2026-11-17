"""Canonical per-image metadata (``loramake.image-meta.v1``): read, normalise, patch.

Every engine writes a ``<image>.png.meta.json`` sidecar next to the render (most of them
through ``loramake.comfy.save_image_with_meta``, which also embeds the same JSON in the PNG
as an iTXt ``loramake`` chunk).  The keys were never a schema: each engine grew its own
aliases (``unet`` / ``checkpoint`` / ``base_ckpt``, ``neg`` / ``negative``,
``res: [w, h]`` / ``resolution: "WxH"``, ``lora`` + ``lora_strength`` / ``content_loras``).
This module is the one place that knows those aliases, so every reader (gallery scan,
console inspector, batch manifests) sees the same canonical keys.

Normalisation is **additive**: the original keys are preserved untouched
(``scripts/anima_step2a_eval.py`` hash-binds ``unet`` and ``res`` literally), canonical
keys are added next to them.  Nothing here touches the network.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loramake.comfy_endpoints import CLAUDE_LOOPBACK_COMFY_HOSTS, DIRECT_COMFY_HOSTS

SCHEMA = "loramake.image-meta.v1"
SIDECAR_SUFFIX = ".meta.json"
PNG_ITXT_KEY = "loramake"
KNOWN_BOXES = tuple(DIRECT_COMFY_HOSTS)

# Canonical key order (also the order the console inspector groups them in).
CANONICAL_KEYS: tuple[str, ...] = (
    "schema",
    # 来源 / origin
    "driving_model", "gen_engine", "box", "host", "script", "argv", "intent", "attempt_id",
    "session", "surface",
    # 模型 / model
    "family", "arch", "base_ckpt", "vae", "content_loras",
    # 提示词 / prompt
    "prompt", "negative", "triggers_injected",
    # 采样 / sampling
    "seed", "sampler", "scheduler", "steps", "cfg", "clip_skip", "shift", "model_sampling",
    "denoise", "resolution",
    # 控制 / control
    "control", "ipadapter", "refine", "three_step", "hires_mode", "advanced", "init_image",
    "source_image",
    # 耗时 / 门禁
    "elapsed_s", "gates",
    # 策展 / curation
    "curated", "qc_score", "gacha_date", "section", "concept", "outfit_key",
)

# Keys that must never leave the machine through an API projection.
PRIVATE_KEYS: frozenset[str] = frozenset({"host"})

GATE_NAMES: tuple[str, ...] = ("wd14", "qc", "ccip")

_RESOLUTION_RE = re.compile(r"^\s*(\d{2,5})\s*[x×X*]\s*(\d{2,5})\s*$")


# ---------------------------------------------------------------------------
# small coercions
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _basename(value: str) -> str:
    """Last path segment across POSIX and Windows separators (BOXA writes backslashes)."""
    return re.split(r"[\\/]", value)[-1]


# ---------------------------------------------------------------------------
# alias resolution
# ---------------------------------------------------------------------------

def _resolution(meta: Mapping[str, Any]) -> str | None:
    raw = meta.get("resolution")
    if isinstance(raw, str):
        match = _RESOLUTION_RE.match(raw)
        if match:
            return f"{int(match.group(1))}x{int(match.group(2))}"
    res = meta.get("res")
    if isinstance(res, str):
        match = _RESOLUTION_RE.match(res)
        if match:
            return f"{int(match.group(1))}x{int(match.group(2))}"
    if isinstance(res, (list, tuple)) and len(res) == 2:
        width, height = _as_int(res[0]), _as_int(res[1])
        if width and height:
            return f"{width}x{height}"
    width, height = _as_int(meta.get("width")), _as_int(meta.get("height"))
    if width and height:
        return f"{width}x{height}"
    return None


def _lora_entry(entry: Any, strength: Any = None) -> dict[str, Any] | None:
    if isinstance(entry, Mapping):
        file = _as_text(entry.get("file")) or _as_text(entry.get("name")) or _as_text(entry.get("lora"))
        if not file:
            return None
        out: dict[str, Any] = {"file": file, "strength": _as_float(entry.get("strength"))}
        role = _as_text(entry.get("role"))
        if role:
            out["role"] = role
        return out
    file = _as_text(entry)
    if not file:
        return None
    return {"file": file, "strength": _as_float(strength)}


def _content_loras(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("content_loras", "loras"):
        raw = meta.get(key)
        if isinstance(raw, list) and raw:
            entries = [_lora_entry(item) for item in raw]
            return [entry for entry in entries if entry is not None]
    raw_lora = meta.get("lora")
    if not raw_lora:
        return []
    names = [raw_lora] if isinstance(raw_lora, str) else list(raw_lora) if isinstance(raw_lora, (list, tuple)) else []
    strengths = meta.get("lora_strength")
    if isinstance(strengths, (list, tuple)):
        pairs = zip(names, list(strengths) + [None] * max(0, len(names) - len(strengths)), strict=False)
    else:
        pairs = ((name, strengths) for name in names)
    entries = [_lora_entry(name, strength) for name, strength in pairs]
    return [entry for entry in entries if entry is not None]


def _sampler_pair(meta: Mapping[str, Any]) -> tuple[str | None, str | None]:
    sampler = _as_text(meta.get("sampler"))
    scheduler = _as_text(meta.get("scheduler"))
    if sampler and "/" in sampler and not scheduler:
        head, _, tail = sampler.partition("/")
        return _as_text(head), _as_text(tail)
    return sampler, scheduler


def _host_box_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for hosts in (DIRECT_COMFY_HOSTS, CLAUDE_LOOPBACK_COMFY_HOSTS):
        for box, url in hosts.items():
            parsed = urlparse(url)
            if parsed.hostname:
                table[f"{parsed.hostname}:{parsed.port or ''}"] = box
    return table


def derive_box(meta: Mapping[str, Any]) -> str | None:
    """Which ComfyUI box rendered this image: ``box`` → ``gen_engine`` suffix → ``host``."""
    box = _as_text(meta.get("box"))
    if box and box.lower() in KNOWN_BOXES:
        return box.lower()
    engine = _as_text(meta.get("gen_engine"))
    if engine and ":" in engine:
        candidate = engine.split(":", 1)[1].split("/", 1)[0].strip().lower()
        if candidate in KNOWN_BOXES:
            return candidate
    host = _as_text(meta.get("host"))
    if host:
        parsed = urlparse(host if "://" in host else f"http://{host}")
        if parsed.hostname:
            key = f"{parsed.hostname}:{parsed.port or ''}"
            return _host_box_table().get(key)
    return None


# ---------------------------------------------------------------------------
# normalise / context
# ---------------------------------------------------------------------------

def normalize(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``meta`` with the canonical keys filled from their aliases.

    Idempotent, additive, never raises on odd shapes: an unusable alias simply leaves the
    canonical key absent.
    """
    out: dict[str, Any] = dict(meta) if isinstance(meta, Mapping) else {}
    out["schema"] = SCHEMA

    base = _as_text(out.get("base_ckpt")) or _as_text(out.get("checkpoint")) or _as_text(out.get("unet"))
    if base:
        out["base_ckpt"] = base
    negative = out.get("negative") if isinstance(out.get("negative"), str) else out.get("neg")
    if isinstance(negative, str):
        out["negative"] = negative
    resolution = _resolution(out)
    if resolution:
        out["resolution"] = resolution
    loras = _content_loras(out)
    if loras or not isinstance(out.get("content_loras"), list):
        out["content_loras"] = loras
    sampler, scheduler = _sampler_pair(out)
    if sampler:
        out["sampler"] = sampler
    if scheduler:
        out["scheduler"] = scheduler

    for key in ("seed", "steps", "clip_skip"):
        value = _as_int(out.get(key))
        if value is not None:
            out[key] = value
    for key in ("cfg", "shift", "denoise", "elapsed_s", "qc_score"):
        value = _as_float(out.get(key))
        if value is not None:
            out[key] = value

    box = derive_box(out)
    if box:
        out["box"] = box
    else:
        # A writer may put a driver or model name in the box slot; the canonical record
        # only carries a real execution box (boxa / boxb) or nothing.
        out.pop("box", None)
    if "gates" in out and not isinstance(out.get("gates"), Mapping):
        out.pop("gates")
    return out


def gen_context(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run context the router (``scripts/gen.py``) hands engine subprocesses through env.

    ``LORAMAKE_GEN_CONTEXT`` is a JSON object (intent, box, attempt_id, script, argv,
    driving_model, session, surface); ``LORAMAKE_SESSION`` / ``LORAMAKE_SURFACE`` /
    ``DRIVING_MODEL`` fill the gaps for engines run directly.
    """
    values = os.environ if env is None else env
    context: dict[str, Any] = {}
    raw = values.get("LORAMAKE_GEN_CONTEXT", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            context.update({str(k): v for k, v in parsed.items()})
    for key, env_key in (("session", "LORAMAKE_SESSION"), ("surface", "LORAMAKE_SURFACE"),
                         ("driving_model", "DRIVING_MODEL"), ("attempt_id", "LORAMAKE_ATTEMPT_ID")):
        if not _as_text(context.get(key)) and _as_text(values.get(env_key)):
            context[key] = values[env_key].strip()
    return context


def stamp_context(meta: Mapping[str, Any] | None, env: Mapping[str, str] | None = None,
                  *, argv: list[str] | None = None) -> dict[str, Any]:
    """Writer-side: fill origin keys (driving_model, script, argv, intent, box, attempt_id,
    session, surface) from the process context, only where the engine left them empty."""
    out = normalize(meta)
    context = gen_context(env)
    for key in ("driving_model", "intent", "box", "attempt_id", "session", "surface", "script"):
        if not _as_text(out.get(key)) and _as_text(context.get(key)):
            out[key] = str(context[key]).strip()
    if not isinstance(out.get("argv"), list):
        raw_argv = context.get("argv")
        if isinstance(raw_argv, list) and all(isinstance(item, str) for item in raw_argv):
            out["argv"] = list(raw_argv)
        elif argv is not None:
            out["argv"] = list(argv)
    if not _as_text(out.get("script")):
        head = (argv or sys.argv or [""])[0]
        if head:
            out["script"] = _basename(head)
    if not _as_text(out.get("driving_model")):
        out["driving_model"] = "claude"
    return out


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------

def sidecar_path(image_path: str | Path) -> Path:
    return Path(f"{image_path}{SIDECAR_SUFFIX}")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def read_sidecar(image_path: str | Path) -> dict[str, Any] | None:
    """Raw (un-normalised) sidecar next to an image, or None."""
    return _read_json_object(sidecar_path(image_path))


def read_png_itxt(image_path: str | Path, key: str = PNG_ITXT_KEY) -> dict[str, Any] | None:
    """The ``loramake`` iTXt chunk of a PNG as a dict, or None (non-PNG / unreadable)."""
    path = Path(image_path)
    if path.suffix.lower() != ".png":
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            raw = image.info.get(key)
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_meta(image_path: str | Path, *, png_fallback: bool = True) -> dict[str, Any] | None:
    """Normalised metadata for an image: sidecar first, then the PNG iTXt copy."""
    raw = read_sidecar(image_path)
    source = "sidecar"
    if raw is None and png_fallback:
        raw = read_png_itxt(image_path)
        source = "png"
    if raw is None:
        return None
    out = normalize(raw)
    out["_source"] = source
    return out


def _merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """One-level deep merge: dict values (``gates``) merge, everything else replaces."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def update_sidecar(image_path: str | Path, patch: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically merge ``patch`` into the image's sidecar and return the normalised result.

    Creates the sidecar when missing.  Readers prefer the sidecar over the PNG iTXt copy,
    so a later patch (gate outcomes, curation) does not need to rewrite the image bytes.
    """
    target = sidecar_path(image_path)
    current = _read_json_object(target) or {}
    merged = normalize(_merge(current, patch))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return merged


# ---------------------------------------------------------------------------
# projections
# ---------------------------------------------------------------------------

def public_view(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """The API-safe projection: canonical, without machine addresses or private keys."""
    out = normalize(meta)
    for key in list(out):
        if key in PRIVATE_KEYS or key.startswith("_"):
            out.pop(key)
    return out


def gate_status(meta: Mapping[str, Any] | None, name: str) -> str | None:
    """``pass`` / ``fail`` / None for one gate (``wd14`` / ``qc`` / ``ccip``)."""
    if not isinstance(meta, Mapping):
        return None
    gates = meta.get("gates")
    if not isinstance(gates, Mapping):
        return None
    gate = gates.get(name)
    if not isinstance(gate, Mapping):
        return None
    passed = gate.get("pass")
    if isinstance(passed, bool):
        return "pass" if passed else "fail"
    return None
