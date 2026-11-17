"""Civitai Bridge — owned lookup, download, and recipe import.

The Chrome extension talks only to Studio. This module owns:

- a compact in-process index of what BOXA already has (sha256 + civitai ids)
- three-state classification: exact / other_version / missing
- Civitai Site REST (same surface as the Site MCP) for hashes and image meta
- LoRA-Manager calls on BOXA for download + recipe persist

Token is read from the canonical path and never returned in DTOs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

from loramake.config import BOXA_HOST

from app import recipe_brief, recipe_store

TOKEN_PATH = Path(os.path.expanduser(os.environ.get("CIVITAI_TOKEN_PATH", "~/.config/civitai/token")))
CIVITAI_COM = "https://civitai.com/api/v1"
CIVITAI_RED = "https://civitai.red/api/v1"
LM_STANDALONE = os.environ.get("BOXA_LM_URL", "http://192.0.2.10:9000")
INDEX_TTL_S = 60.0
NSFW_LEVEL_THRESHOLD = 4  # civitai bitmask: 4=R and above
MATURE_BROWSING_LEVEL = 28  # PG13+R+X+XXX, same as civitai-fetch-images.mjs

NSFW_HOSTS = {"civitai.red", "civitai.green"}
PAGE_HOSTS = {
    "civitai.com",
    "civitai.red",
    "civitai.green",
    "civarchive.com",
    "civitaiarchive.com",
}

_MODEL_PATH = re.compile(r"/models/(\d+)")
_IMAGE_PATH = re.compile(r"/images/(\d+)")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{8,64}$")
_VERSION_STEM = re.compile(r"^(\d{4,12})$")


def infer_version_id_from_filename(file_name: str | None) -> int | None:
    """Civitai downloads are often saved as '<versionId>.safetensors'."""
    if not file_name:
        return None
    stem = Path(str(file_name).replace("\\", "/")).stem
    if _VERSION_STEM.match(stem):
        return int(stem)
    return None


def normalize_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text or not _HASH_RE.match(text):
        return None
    return text


def _as_int(value: Any) -> int | None:
    if value is None or value is False:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def site_from_host(host: str) -> str:
    host = (host or "").lower().split(":")[0]
    if host in NSFW_HOSTS:
        return "red"
    if "civarchive" in host or host == "civitaiarchive.com":
        return "archive"
    return "com"


def parse_page_url(url: str) -> dict[str, Any]:
    """Extract kind / ids / site from a Civitai or CivArchive URL."""
    raw = (url or "").strip()
    if not raw:
        return {"kind": None, "modelId": None, "versionId": None, "imageId": None, "site": "com"}
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    site = site_from_host(host)
    qs = parse_qs(parsed.query)
    version_id = _as_int((qs.get("modelVersionId") or [None])[0])
    model_id = None
    image_id = None
    kind = None
    model_m = _MODEL_PATH.search(parsed.path or "")
    image_m = _IMAGE_PATH.search(parsed.path or "")
    if image_m:
        image_id = int(image_m.group(1))
        kind = "image"
    elif model_m:
        model_id = int(model_m.group(1))
        kind = "model"
    return {
        "kind": kind,
        "modelId": model_id,
        "versionId": version_id,
        "imageId": image_id,
        "site": site,
        "host": host,
    }


@dataclass(frozen=True)
class OwnedRecord:
    model_type: str
    file: str
    name: str
    sha256: str | None
    model_id: int | None
    version_id: int | None
    path: str | None = None


@dataclass
class OwnedIndex:
    by_sha: dict[str, OwnedRecord] = field(default_factory=dict)
    by_version: dict[int, OwnedRecord] = field(default_factory=dict)
    by_model: dict[int, list[OwnedRecord]] = field(default_factory=dict)
    records: list[OwnedRecord] = field(default_factory=list)
    built_at: float = 0.0
    source: str = ""

    def counts(self) -> dict[str, int]:
        return {
            "records": len(self.records),
            "with_sha": len(self.by_sha),
            "with_version": len(self.by_version),
            "with_model": len(self.by_model),
        }


def record_from_lm_item(item: dict, model_type: str = "lora") -> OwnedRecord | None:
    """Map one LoRA-Manager list item into an OwnedRecord. None if unusable."""
    if not isinstance(item, dict):
        return None
    civ = item.get("civitai") if isinstance(item.get("civitai"), dict) else {}
    sha = normalize_hash(item.get("sha256") or item.get("hash") or civ.get("SHA256") or civ.get("sha256"))
    version_id = _as_int(
        item.get("civitai_id")
        or item.get("modelVersionId")
        or civ.get("id")
        or civ.get("modelVersionId")
        or (civ.get("model") or {}).get("versionId")
    )
    model_id = _as_int(
        item.get("modelId")
        or item.get("model_id")
        or civ.get("modelId")
        or civ.get("model_id")
        or (civ.get("model") or {}).get("id")
    )
    file_name = (
        item.get("file_name")
        or item.get("fileName")
        or item.get("filename")
        or Path(str(item.get("file_path") or item.get("path") or "")).name
        or ""
    )
    name = (
        item.get("model_name")
        or civ.get("name")
        or item.get("name")
        or Path(str(file_name)).stem
        or ""
    )
    if version_id is None:
        version_id = infer_version_id_from_filename(file_name) or infer_version_id_from_filename(
            item.get("file_path") or item.get("path")
        )
    if not file_name and not sha and version_id is None and model_id is None:
        return None
    path = item.get("file_path") or item.get("path")
    return OwnedRecord(
        model_type=model_type,
        file=str(file_name),
        name=str(name),
        sha256=sha,
        model_id=model_id,
        version_id=version_id,
        path=str(path) if path else None,
    )


def build_index(items: Iterable[tuple[dict, str]], *, source: str = "", now: float | None = None) -> OwnedIndex:
    index = OwnedIndex(built_at=now if now is not None else time.time(), source=source)
    for item, model_type in items:
        rec = record_from_lm_item(item, model_type)
        if rec is None:
            continue
        index.records.append(rec)
        if rec.sha256 and rec.sha256 not in index.by_sha:
            index.by_sha[rec.sha256] = rec
        if rec.version_id is not None and rec.version_id not in index.by_version:
            index.by_version[rec.version_id] = rec
        if rec.model_id is not None:
            index.by_model.setdefault(rec.model_id, []).append(rec)
    return index


def _dto(status: str, rec: OwnedRecord | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": status,
        "file": rec.file if rec else None,
        "name": rec.name if rec else None,
        "modelType": rec.model_type if rec else None,
        "ownedVersionId": rec.version_id if rec else None,
        "ownedModelId": rec.model_id if rec else None,
    }
    out.update(extra)
    return out


def classify(
    index: OwnedIndex,
    *,
    model_id: int | None = None,
    version_id: int | None = None,
    sha256: str | None = None,
    hashes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Three-state owned check. Hash wins, then version id, then same-model other version."""
    for raw in [sha256, *(hashes or [])]:
        key = normalize_hash(raw)
        if key and key in index.by_sha:
            rec = index.by_sha[key]
            return _dto("exact", rec, match="sha256", modelId=model_id, versionId=version_id)
    if version_id is not None and version_id in index.by_version:
        rec = index.by_version[version_id]
        return _dto("exact", rec, match="version", modelId=model_id, versionId=version_id)
    if model_id is not None and model_id in index.by_model:
        recs = index.by_model[model_id]
        rec = recs[0]
        if version_id is None:
            return _dto("exact", rec, match="model", modelId=model_id, versionId=version_id)
        return _dto(
            "other_version",
            rec,
            match="model",
            modelId=model_id,
            versionId=version_id,
            ownedVersionIds=[r.version_id for r in recs if r.version_id is not None],
        )
    return _dto("missing", None, match=None, modelId=model_id, versionId=version_id)


def hashes_from_version_payload(payload: dict | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for file_info in payload.get("files") or []:
        if not isinstance(file_info, dict):
            continue
        hashes = file_info.get("hashes") or {}
        if isinstance(hashes, dict):
            for key in ("SHA256", "sha256", "AutoV2", "AutoV3"):
                norm = normalize_hash(hashes.get(key))
                if norm:
                    out.append(norm)
        for key in ("sha256", "SHA256", "hash"):
            norm = normalize_hash(file_info.get(key))
            if norm:
                out.append(norm)
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def resources_from_image_payload(payload: dict | None) -> list[dict[str, Any]]:
    """Normalize Civitai image meta into LM import-remote resource dicts."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("items") if isinstance(payload.get("items"), list) else None
    image = items[0] if items else payload
    if not isinstance(image, dict):
        return []
    meta = image.get("meta") if isinstance(image.get("meta"), dict) else {}
    raw_lists: list[Any] = []
    for candidate in (
        image.get("resources"),
        meta.get("resources"),
        meta.get("civitaiResources"),
    ):
        if isinstance(candidate, list):
            raw_lists.extend(candidate)
    resources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for raw in raw_lists:
        if not isinstance(raw, dict):
            continue
        rtype = str(raw.get("type") or raw.get("modelType") or "").lower()
        if rtype in {"model", "checkpoint", "ckpt"}:
            rtype = "checkpoint"
        elif rtype in {"lora", "locon", "lycoris", "loha", "lokr"}:
            rtype = "lora"
        elif rtype in {"embed", "embedding", "textualinversion", "textual_inversion"}:
            rtype = "embedding"
        else:
            # Civitai image resources often omit type for LoRAs
            if raw.get("weight") is not None or "lora" in str(raw.get("name") or "").lower():
                rtype = "lora"
            else:
                continue
        model_id = _as_int(raw.get("modelId") or raw.get("model_id"))
        version_id = _as_int(
            raw.get("modelVersionId") or raw.get("modelVersionID") or raw.get("versionId") or raw.get("id")
        )
        name = str(raw.get("modelName") or raw.get("name") or "")
        key = (rtype, model_id, version_id or name)
        if key in seen:
            continue
        seen.add(key)
        weight = raw.get("weight", 1.0)
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            weight_f = 1.0
        resources.append({
            "type": rtype,
            "modelId": model_id,
            "modelVersionId": version_id,
            "modelName": name,
            "modelVersionName": str(raw.get("modelVersionName") or raw.get("versionName") or ""),
            "weight": weight_f,
        })
    return resources


def gen_params_from_image_payload(payload: dict | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else None
    image = items[0] if items else payload
    if not isinstance(image, dict):
        return {}
    meta = image.get("meta") if isinstance(image.get("meta"), dict) else {}
    out: dict[str, Any] = {}
    mapping = {
        "prompt": ("prompt",),
        "negativePrompt": ("negativePrompt", "negative_prompt"),
        "sampler": ("sampler", "samplerName"),
        "scheduler": ("scheduler",),
        "steps": ("steps",),
        "cfgScale": ("cfgScale", "cfg"),
        "seed": ("seed",),
        "width": ("width",),
        "height": ("height",),
        "clipSkip": ("clipSkip", "clip_skip"),
    }
    for dest, keys in mapping.items():
        for key in keys:
            if meta.get(key) is not None:
                out[dest] = meta[key]
                break
    return out


def image_title(payload: dict | None, image_id: int | None) -> str:
    if isinstance(payload, dict):
        items = payload.get("items") if isinstance(payload.get("items"), list) else None
        image = items[0] if items else payload
        if isinstance(image, dict):
            for key in ("name", "title"):
                if image.get(key):
                    return str(image[key])[:120]
            meta = image.get("meta") if isinstance(image.get("meta"), dict) else {}
            prompt = str(meta.get("prompt") or "").strip()
            if prompt:
                return prompt.split(",")[0].strip()[:80] or f"civitai-{image_id}"
    return f"civitai-{image_id or 'image'}"


_ACTIVE_STATUSES = {"queued", "waiting", "downloading", "progress", "paused"}
_DONE_STATUSES = {"completed", "failed", "canceled", "cancelled", "already"}
_STATUS_MAP = {
    "progress": "downloading",
    "downloading": "downloading",
    "queued": "queued",
    "waiting": "queued",
    "paused": "paused",
    "completed": "completed",
    "already": "completed",
    "failed": "failed",
    "error": "failed",
    "canceled": "canceled",
    "cancelled": "canceled",
}


def normalize_status(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    return _STATUS_MAP.get(key, "unknown")


def _num(value: Any) -> float | None:
    if value is None or value is False:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_download_item(*parts: dict | None) -> dict[str, Any]:
    """Merge tracker / queue / progress-endpoint dicts into one DTO."""
    merged: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            merged.update(part)
    download_id = (
        merged.get("downloadId")
        or merged.get("download_id")
        or merged.get("id")
    )
    bytes_dl = _num(
        merged.get("bytesDownloaded")
        or merged.get("bytes_downloaded")
    )
    total = _num(merged.get("totalBytes") or merged.get("total_bytes"))
    progress = _num(merged.get("progress"))
    if progress is None and bytes_dl is not None and total and total > 0:
        progress = max(0.0, min(100.0, bytes_dl / total * 100.0))
    if progress is not None:
        progress = max(0.0, min(100.0, float(progress)))
    status = normalize_status(merged.get("status"))
    if status == "unknown" and progress is not None and progress >= 100:
        status = "completed"
    if status == "unknown" and progress is not None:
        status = "downloading"
    if merged.get("is_already_exists") or merged.get("already"):
        status = "completed"
        progress = 100.0
    speed = _num(merged.get("bytesPerSecond") or merged.get("bytes_per_second"))
    return {
        "downloadId": str(download_id) if download_id is not None else None,
        "modelId": _as_int(merged.get("modelId") or merged.get("model_id")),
        "versionId": _as_int(
            merged.get("versionId")
            or merged.get("modelVersionId")
            or merged.get("model_version_id")
        ),
        "name": merged.get("name") or merged.get("model_name") or merged.get("modelName"),
        "versionName": merged.get("versionName") or merged.get("version_name"),
        "status": status,
        "progress": progress,
        "bytesDownloaded": bytes_dl,
        "totalBytes": total,
        "bytesPerSecond": speed,
        "message": merged.get("message") or merged.get("error"),
        "active": status in {"queued", "downloading", "paused"},
    }


def merge_download_jobs(
    tracked: Iterable[dict],
    queue_items: Iterable[dict],
    progress_by_id: dict[str, dict],
    history_items: Iterable[dict] | None = None,
) -> list[dict[str, Any]]:
    """tracked + LM queue + per-id progress + recent history, keyed by downloadId."""
    by_id: dict[str, dict[str, Any]] = {}

    def _put(*parts: dict | None) -> None:
        dto = normalize_download_item(*[p for p in parts if p])
        key = dto.get("downloadId")
        if not key:
            return
        prev = by_id.get(key)
        by_id[key] = normalize_download_item(prev, dto) if prev else dto

    queue_ids = {
        str(item.get("download_id") or item.get("downloadId") or "")
        for item in queue_items
        if isinstance(item, dict)
    }
    queue_ids.discard("")
    for item in tracked:
        _put(item, progress_by_id.get(str(item.get("downloadId") or item.get("download_id") or "")))
    for item in queue_items:
        did = str(item.get("download_id") or item.get("downloadId") or "")
        _put(item, progress_by_id.get(did))
    for item in history_items or []:
        did = str(item.get("download_id") or item.get("downloadId") or "")
        live = did in progress_by_id or did in queue_ids
        if did in by_id and by_id[did]["active"] and live:
            continue
        _put(item, progress_by_id.get(did))
    jobs = list(by_id.values())
    rank = {"downloading": 0, "paused": 1, "queued": 2, "failed": 3, "completed": 4, "canceled": 5, "unknown": 6}
    jobs.sort(key=lambda j: (rank.get(j["status"], 9), -(j.get("progress") or 0)))
    return jobs


# progress() fan-out cap: the fastest frontend poll is 1.5s, so a shorter TTL buys
# nothing; without it every concurrent poller runs its own multi-curl LM sweep and
# the anyio worker pool starves (observed live as the SSE badge flapping).
PROGRESS_TTL_S = 1.5


class CivitaiBridge:
    """I/O boundary. Tests stub the fetchers."""

    def __init__(
        self,
        *,
        lm_get: Callable[..., dict | None] | None = None,
        lm_post: Callable[..., dict] | None = None,
        civitai_get: Callable[..., dict | None] | None = None,
        download_fn: Callable[..., dict] | None = None,
        now: Callable[[], float] | None = None,
        ttl_s: float = INDEX_TTL_S,
    ) -> None:
        self._lm_get = lm_get or _default_lm_get
        self._lm_post = lm_post or _default_lm_post
        self._civitai_get = civitai_get or _default_civitai_get
        self._download_fn = download_fn
        self._now = now or time.time
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._index: OwnedIndex | None = None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._hash_cache: dict[int, list[str]] = {}
        self._relink_attempted: set[str] = set()
        # Serializes the progress fan-out: waiters block until the in-flight build
        # lands, then read its cached value instead of starting their own sweep.
        self._progress_lock = threading.Lock()
        self._progress_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._scan_event: threading.Event | None = None
        self._recipes_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}
        self._recipes_ttl_s = 15.0

    def invalidate(self) -> None:
        with self._lock:
            self._index = None

    def index(self, *, force: bool = False) -> OwnedIndex:
        with self._lock:
            cached = self._index
            if (
                not force
                and cached is not None
                and (self._now() - cached.built_at) < self._ttl_s
            ):
                return cached
            waiter = self._scan_event
            owner = False
            if waiter is None:
                waiter = threading.Event()
                self._scan_event = waiter
                owner = True
        if not owner:
            waiter.wait()
            with self._lock:
                if self._index is not None:
                    return self._index
            return self._scan()
        try:
            built = self._scan()
            with self._lock:
                self._index = built
                return built
        finally:
            with self._lock:
                self._scan_event = None
                waiter.set()

    def health(self) -> dict[str, Any]:
        lm_ok, lm_src = self._probe_lm()
        try:
            idx = self.index()
            index_ok = True
            counts = idx.counts()
            source = idx.source
        except Exception as exc:  # noqa: BLE001 — health must never raise
            index_ok = False
            counts = {}
            source = ""
            lm_ok = False
            lm_src = str(exc)
        return {
            "ok": lm_ok and index_ok,
            "studio": True,
            "loraManager": {"ok": lm_ok, "source": lm_src},
            "index": {"ok": index_ok, **counts, "source": source},
        }

    def owned(self, queries: Iterable[dict], *, enrich: bool = False) -> list[dict[str, Any]]:
        idx = self.index()
        items = list(queries)
        hash_map: dict[int, list[str]] = {}
        if enrich:
            need: list[int] = []
            for q in items:
                vid = _as_int(q.get("versionId") or q.get("modelVersionId"))
                mid = _as_int(q.get("modelId"))
                local = classify(idx, model_id=mid, version_id=vid)
                if local["status"] != "exact" and vid is not None:
                    need.append(vid)
            hash_map = self._hashes_for_versions(need)
            # Hash hit against a file that LM never tagged with a Civitai id:
            # remember the mapping so later list checks don't need another API call,
            # and ask LM to fetch metadata so it stops showing base_model=Unknown.
            self._adopt_hash_hits(idx, items, hash_map)
        out: list[dict[str, Any]] = []
        for q in items:
            mid = _as_int(q.get("modelId"))
            vid = _as_int(q.get("versionId") or q.get("modelVersionId"))
            extra_hashes = list(q.get("hashes") or [])
            if vid is not None:
                extra_hashes.extend(hash_map.get(vid) or [])
            dto = classify(
                idx,
                model_id=mid,
                version_id=vid,
                sha256=q.get("sha256"),
                hashes=extra_hashes,
            )
            out.append(dto)
        return out

    def inspect(self, *, url: str = "", image_id: int | None = None, model_id: int | None = None,
                version_id: int | None = None, site: str | None = None) -> dict[str, Any]:
        parsed = parse_page_url(url) if url else {
            "kind": "image" if image_id else "model" if model_id else None,
            "modelId": model_id,
            "versionId": version_id,
            "imageId": image_id,
            "site": site or "com",
        }
        kind = parsed.get("kind")
        site_name = site or parsed.get("site") or "com"
        mid = _as_int(model_id if model_id is not None else parsed.get("modelId"))
        vid = _as_int(version_id if version_id is not None else parsed.get("versionId"))
        iid = _as_int(image_id if image_id is not None else parsed.get("imageId"))
        if kind == "image" and iid:
            payload = self._civitai_get(
                "/images",
                params={"imageId": iid, "nsfw": "X", "browsingLevel": MATURE_BROWSING_LEVEL},
                site=site_name,
            )
            resources = resources_from_image_payload(payload)
            queries = []
            for res in resources:
                queries.append({
                    "modelId": res.get("modelId"),
                    "versionId": res.get("modelVersionId"),
                })
            owned = self.owned(queries, enrich=True)
            for res, own in zip(resources, owned, strict=False):
                res["owned"] = own
            return {
                "kind": "image",
                "imageId": iid,
                "site": site_name,
                "title": image_title(payload, iid),
                "resources": resources,
                "genParams": gen_params_from_image_payload(payload),
                "recipeReady": bool(resources),
            }
        owned = self.owned([{"modelId": mid, "versionId": vid}], enrich=True)
        return {
            "kind": "model",
            "modelId": mid,
            "versionId": vid,
            "site": site_name,
            "owned": owned[0] if owned else classify(self.index(), model_id=mid, version_id=vid),
        }

    def download(self, *, model_id: int | None, version_id: int | None, site: str = "com",
                 allow_nsfw: bool | None = None) -> dict[str, Any]:
        if model_id is None and version_id is None:
            return {"ok": False, "error": "modelId or versionId required"}
        allow = bool(allow_nsfw) if allow_nsfw is not None else (site == "red")
        if self._download_fn is not None:
            result = self._download_fn(model_id=model_id, version_id=version_id, allow_nsfw=allow, site=site)
        else:
            result = self._download_via_lm(model_id=model_id, version_id=version_id, allow_nsfw=allow, site=site)
        if result.get("ok") and result.get("downloadId"):
            self._remember_job(result)
        if result.get("already"):
            self.invalidate()
        return result

    def progress(self, download_id: str | None = None) -> dict[str, Any]:
        key = str(download_id) if download_id else ""
        with self._progress_lock:
            cached = self._progress_cache.get(key)
            if cached is not None and (self._now() - cached[0]) < PROGRESS_TTL_S:
                return cached[1]
            value = self._progress_fetch(download_id)
            self._progress_cache[key] = (self._now(), value)
            return value

    def _progress_fetch(self, download_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            tracked = [dict(job) for job in self._jobs.values()]
        queue_payload = self._lm_get("/api/lm/downloads/queue") or {}
        history_payload = self._lm_get("/api/lm/downloads/history?limit=8") or {}
        queue_items = queue_payload.get("queue") or []
        history_items = history_payload.get("items") or history_payload.get("history") or []
        ids: list[str] = []
        for item in tracked:
            did = item.get("downloadId")
            if did:
                ids.append(str(did))
        for item in queue_items:
            if isinstance(item, dict) and item.get("download_id"):
                ids.append(str(item["download_id"]))
        if download_id:
            ids.append(str(download_id))
        progress_by_id: dict[str, dict] = {}
        for did in dict.fromkeys(ids):
            raw = self._lm_get(f"/api/lm/download-progress/{did}")
            if isinstance(raw, dict) and raw.get("success") is not False:
                progress_by_id[did] = raw
        jobs = merge_download_jobs(tracked, queue_items, progress_by_id, history_items)
        if download_id:
            jobs = [j for j in jobs if j.get("downloadId") == str(download_id)]
        done = {str(j.get("downloadId")) for j in jobs if j.get("status") in _DONE_STATUSES}
        if done:
            with self._lock:
                for did in done:
                    self._jobs.pop(did, None)
        if any(j.get("status") == "completed" for j in jobs):
            self.invalidate()
        return {
            "jobs": jobs,
            "active": [j for j in jobs if j.get("active")],
            "stats": queue_payload.get("stats") or {},
        }

    def _remember_job(self, result: dict[str, Any]) -> None:
        did = result.get("downloadId")
        if not did:
            return
        with self._lock:
            self._jobs[str(did)] = {
                "downloadId": str(did),
                "modelId": result.get("modelId"),
                "versionId": result.get("versionId"),
                "name": result.get("name"),
                "status": "queued",
                "progress": 0,
                "startedAt": self._now(),
            }

    # import-remote is GET-only upstream (recipe_route_registrar.py, verified 2026-08-25),
    # so a full prompt rides in the query string. Guard the encoded length; past the cap
    # the prompt is truncated rather than the whole save failing mid-flight.
    MAX_IMPORT_QS = 7000

    def save_recipe(self, *, image_id: int, url: str = "", site: str = "com",
                    name: str | None = None) -> dict[str, Any]:
        inspect = self.inspect(url=url, image_id=image_id, site=site)
        resources = inspect.get("resources") or []
        gen_params = inspect.get("genParams") or {}
        title = name or inspect.get("title") or f"civitai-{image_id}"
        image_url = url or f"https://{'civitai.red' if site == 'red' else 'civitai.com'}/images/{image_id}"
        if not resources:
            # No parseable resources from the REST payload, but LoRA Manager parses the
            # page itself — hand it the url rather than refusing the save. This is the
            # common path now, not a fallback: Civitai's public v1 image API stopped
            # returning `meta` (measured 2026-08-25, 10/10 top images came back with an
            # empty meta, with and without a token), so inspect usually has neither
            # resources nor gen params while LM's importer still succeeds.
            qs = urllib.parse.urlencode({"image_url": image_url})
            data = self._lm_get(f"/api/lm/recipes/import-from-url?{qs}")
            if not isinstance(data, dict) or data.get("error"):
                error = data.get("error") if isinstance(data, dict) else "LoRA Manager did not accept the recipe"
                return {"ok": False, "error": str(error or "LoRA Manager did not accept the recipe")}
            result = {"ok": True, "via": "import-from-url", "recipe": data, "title": title, "resources": []}
            self._capture_recipe_locally(result, image_id=image_id, site=site, inspect=inspect)
            return result
        params = {
            "image_url": image_url,
            "name": title,
            "resources": json.dumps([
                {
                    "type": r.get("type"),
                    "modelId": r.get("modelId"),
                    "modelVersionId": r.get("modelVersionId"),
                    "modelName": r.get("modelName"),
                    "modelVersionName": r.get("modelVersionName"),
                    "weight": r.get("weight", 1.0),
                }
                for r in resources
            ], ensure_ascii=False, separators=(",", ":")),
            "gen_params": json.dumps(gen_params, ensure_ascii=False, separators=(",", ":")),
        }
        qs = urllib.parse.urlencode(params)
        truncated = False
        if len(qs) > self.MAX_IMPORT_QS:
            trimmed = dict(gen_params)
            # Trim the two free-text fields until the encoded query fits; url-encoding
            # inflation means one pass by raw overflow is not enough.
            for field in ("prompt", "negativePrompt"):
                while len(qs) > self.MAX_IMPORT_QS and trimmed.get(field):
                    text = str(trimmed.get(field) or "")
                    overflow = len(qs) - self.MAX_IMPORT_QS
                    trimmed[field] = text[: max(0, len(text) - max(overflow // 3, 32))]
                    params["gen_params"] = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
                    qs = urllib.parse.urlencode(params)
                    truncated = True
            if len(qs) > self.MAX_IMPORT_QS:
                # The resources array alone is over the cap — nothing left to trim safely.
                return {"ok": False, "error": "recipe too large for LM import-remote (GET-only upstream)"}
        data = self._lm_get(f"/api/lm/recipes/import-remote?{qs}")
        if not isinstance(data, dict):
            return {"ok": False, "error": "LoRA Manager did not accept the recipe"}
        if data.get("error"):
            return {"ok": False, "error": str(data.get("error"))}
        result: dict[str, Any] = {"ok": True, "recipe": data, "title": title, "resources": resources}
        if truncated:
            result["truncated"] = True
        self._capture_recipe_locally(result, image_id=image_id, site=site, inspect=inspect)
        return result

    def _capture_recipe_locally(self, result: dict[str, Any], *, image_id: int, site: str,
                                inspect: dict[str, Any]) -> None:
        """Provenance is captured at import time — the raw Civitai view of the image the
        moment it was saved — so later fusion/lineage work can always trace back. Never
        lets a local write break the save itself."""
        try:
            lm_data = result.get("recipe") or {}
            lm_recipe = lm_data.get("recipe") if isinstance(lm_data.get("recipe"), dict) else lm_data
            recipe_id = str(
                (lm_recipe.get("id") if isinstance(lm_recipe, dict) else None)
                or f"civitai-{image_id}"
            )
            dto = recipe_brief.recipe_from_lm_item(lm_recipe) if isinstance(lm_recipe, dict) else {}
            if not dto.get("id"):
                dto = {
                    "id": recipe_id,
                    "title": result.get("title") or "",
                    "sourceUrl": f"https://{'civitai.red' if site == 'red' else 'civitai.com'}/images/{image_id}",
                    "previewNsfwLevel": 0,
                    "loras": [],
                    "genParams": inspect.get("genParams") or {},
                    "missingCount": 0,
                }
            record = recipe_store.upsert_from_lm(
                dto,
                site=site,
                provenance={
                    "civitaiImageId": image_id,
                    "site": site,
                    "resources": inspect.get("resources") or [],
                    "genParams": inspect.get("genParams") or {},
                    "capturedBy": "studio-save",
                },
                now=self._now(),
            )[0]
            result["localRecord"] = str(recipe_store.find_record_path(record["id"]) or "")
        except Exception:  # noqa: BLE001 — provenance capture is best-effort by design
            result["localRecord"] = None

    # --- recipe read/apply loop -------------------------------------------

    def recipes(self, *, page: int = 1, page_size: int = 20, search: str = "") -> dict[str, Any]:
        params = {"page": max(1, int(page)), "page_size": max(1, min(int(page_size), 100))}
        if search.strip():
            params["search"] = search.strip()
        cache_key = (params.get("page"), params.get("page_size"), params.get("search") or "")
        now = self._now()
        with self._lock:
            hit = self._recipes_cache.get(cache_key)
            if hit and now - hit[0] < self._recipes_ttl_s:
                return hit[1]
        qs = urllib.parse.urlencode(params)
        data = self._lm_get(f"/api/lm/recipes?{qs}")
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            return {"ok": False, "error": "LoRA Manager unreachable"}
        items = [recipe_brief.recipe_from_lm_item(item) for item in data["items"]]
        payload = {
            "ok": True,
            "items": items,
            "total": data.get("total"),
            "page": data.get("page"),
            "pageSize": data.get("page_size"),
            "totalPages": data.get("total_pages"),
            "source": data.get("_source"),
        }
        with self._lock:
            self._recipes_cache[cache_key] = (self._now(), payload)
        return payload

    def recipe_detail(self, recipe_id: str) -> dict[str, Any]:
        safe_id = urllib.parse.quote(str(recipe_id), safe="")
        data = self._lm_get(f"/api/lm/recipe/{safe_id}")
        # `reason` splits the two failures a caller must treat differently: no host
        # answered at all, versus LoRA Manager answering that it has no such recipe.
        if not isinstance(data, dict):
            return {"ok": False, "reason": "unreachable", "error": "LoRA Manager unreachable"}
        if data.get("error") or not data.get("id"):
            return {"ok": False, "reason": "not_found",
                    "error": str(data.get("error") or f"recipe not found: {recipe_id}")}
        dto = recipe_brief.recipe_from_lm_item(data)
        out: dict[str, Any] = {"ok": True, "recipe": dto}
        local = recipe_store.enrichment(dto["id"])
        if local:
            out["local"] = local
        return out

    def download_missing(self, *, recipe_id: str | None = None,
                         resources: list[dict] | None = None,
                         allow_nsfw: bool = False) -> dict[str, Any]:
        """Queue every missing LoRA of a saved recipe (or an explicit resource list)
        through the normal download path — NSFW gate and job tracker included.
        `allow_nsfw` mirrors the single-download button's allowNsfw (e.g. a civitai.red
        page); without it every NSFW-rated target lands in `skipped`."""
        targets: list[dict[str, Any]] = []
        if recipe_id:
            detail = self.recipe_detail(recipe_id)
            if not detail.get("ok"):
                return detail
            targets = recipe_brief.missing_loras(detail["recipe"])
        elif resources:
            for res in resources:
                if not isinstance(res, dict):
                    continue
                entry = {
                    "modelName": res.get("modelName") or res.get("name") or "",
                    "modelId": _as_int(res.get("modelId")),
                    "modelVersionId": _as_int(res.get("modelVersionId") or res.get("versionId")),
                }
                if entry["modelVersionId"] is None and entry["modelId"] is None:
                    entry["skipped"] = "no version id"
                targets.append(entry)
        else:
            return {"ok": False, "error": "recipeId or resources required"}
        queued: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        already: list[dict[str, Any]] = []
        for target in targets:
            name = target.get("modelName") or target.get("stem") or "?"
            if target.get("skipped"):
                skipped.append({"name": name, "reason": target["skipped"]})
                continue
            try:
                result = self.download(
                    model_id=target.get("modelId"),
                    version_id=target.get("modelVersionId"),
                    allow_nsfw=allow_nsfw or None,
                )
            # boxa_lora_download.resolve() raises SystemExit on a failed Civitai fetch;
            # one bad target must not abort the whole batch.
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                skipped.append({"name": name, "reason": str(exc) or "download failed"})
                continue
            if result.get("already"):
                already.append({"name": result.get("name") or name})
            elif result.get("ok"):
                queued.append({
                    "downloadId": result.get("downloadId"),
                    "name": result.get("name") or name,
                    "versionId": result.get("versionId") or target.get("modelVersionId"),
                })
            else:
                skipped.append({"name": name, "reason": str(result.get("error") or "download failed")})
        return {"ok": True, "queued": queued, "skipped": skipped, "already": already}

    def recipe_generate(self, recipe_id: str, overrides: dict | None = None) -> dict[str, Any]:
        detail = self.recipe_detail(recipe_id)
        if not detail.get("ok"):
            return detail
        job_brief, warnings = recipe_brief.brief_from_recipe(detail["recipe"], overrides)
        from app import brief as brief_mod  # local import keeps the module cycle-free

        problems = brief_mod.validate(job_brief)
        if problems:
            return {"ok": False, "problems": problems, "brief": job_brief, "warnings": warnings}
        from app import jobs as jobs_mod  # lazy: tests monkeypatch, CLI never imports this path

        submitted = jobs_mod.submit(job_brief)
        job_id = submitted.get("id") or submitted.get("jobId")
        # Usage tracking must not depend on an earlier import having built the record —
        # and once the job is queued, a bookkeeping failure must not turn the response
        # into an error for work that actually ran.
        try:
            recipe_store.upsert_from_lm(detail["recipe"], now=self._now())
            recipe_store.append_run(
                str(detail["recipe"].get("id") or recipe_id),
                {"jobId": job_id, "briefTier": job_brief.get("tier"), "via": "studio"},
                now=self._now(),
            )
        except Exception:  # noqa: BLE001 — run tracking is best-effort by design
            warnings = [*warnings, "run tracking failed; job queued anyway"]
        return {"ok": True, "jobId": job_id, "brief": job_brief, "warnings": warnings}

    def sync_recipes(self, *, max_pages: int = 50) -> dict[str, Any]:
        """Mirror the LM store into the local library (backfill for recipes saved
        before provenance capture existed, or saved LM-direct by the extension)."""
        created = 0
        updated = 0
        page = 1
        total: int | None = None
        while page <= max_pages:
            data = self._lm_get(f"/api/lm/recipes?page={page}&page_size=50")
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                if page == 1:
                    return {"ok": False, "error": "LoRA Manager unreachable"}
                break
            if total is None:
                total = _as_int(data.get("total"))
            batch = data["items"]
            if not batch:
                break
            for item in batch:
                dto = recipe_brief.recipe_from_lm_item(item)
                if not dto.get("id"):
                    continue
                _, was_created = recipe_store.upsert_from_lm(
                    dto,
                    provenance={"capturedBy": "sync"},
                    now=self._now(),
                )
                created += int(was_created)
                updated += int(not was_created)
            if total is not None and created + updated >= total:
                break
            page += 1
        return {"ok": True, "created": created, "updated": updated, "total": total}

    # --- internals ---------------------------------------------------------

    def _scan(self) -> OwnedIndex:
        pairs: list[tuple[dict, str]] = []
        source = ""
        last_err = ""
        for kind, model_type in (("loras", "lora"), ("checkpoints", "checkpoint"), ("embeddings", "embedding")):
            data, src, err = self._lm_list(kind)
            if src:
                source = src
            if err:
                last_err = err
            for item in data:
                pairs.append((item, model_type))
        if not pairs and last_err:
            raise RuntimeError(last_err)
        return build_index(pairs, source=source, now=self._now())

    def _lm_list(self, kind: str) -> tuple[list[dict], str, str]:
        items: list[dict] = []
        source = ""
        err = ""
        page, page_size, seen, total = 1, 100, 0, None
        while page <= 80:
            data = self._lm_get(f"/api/lm/{kind}/list?page={page}&page_size={page_size}")
            if not isinstance(data, dict):
                if page == 1:
                    err = f"LM {kind} list unavailable"
                break
            source = str(data.get("_source") or source or "lm")
            batch = data.get("items") or []
            if total is None:
                total = data.get("total")
            if not batch:
                break
            for item in batch:
                if isinstance(item, dict):
                    items.append(item)
            seen += len(batch)
            if total is not None and seen >= total:
                break
            page += 1
        return items, source, err

    def _probe_lm(self) -> tuple[bool, str]:
        data = self._lm_get("/api/lm/loras/list?page=1&page_size=1")
        if isinstance(data, dict) and ("items" in data or data.get("success") is not False):
            return True, str(data.get("_source") or "lm")
        return False, "unreachable"

    def _hashes_for_versions(self, version_ids: list[int]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for vid in dict.fromkeys(version_ids):
            if vid in self._hash_cache:
                out[vid] = self._hash_cache[vid]
                continue
            payload = self._civitai_get(f"/model-versions/{vid}", site="auto")
            hashes = hashes_from_version_payload(payload)
            if hashes:
                self._hash_cache[vid] = hashes
                out[vid] = hashes
        return out

    def _adopt_hash_hits(
        self, idx: OwnedIndex, queries: list[dict], hash_map: dict[int, list[str]]
    ) -> None:
        for q in queries:
            vid = _as_int(q.get("versionId") or q.get("modelVersionId"))
            mid = _as_int(q.get("modelId"))
            if vid is None:
                continue
            rec = None
            for raw in hash_map.get(vid) or []:
                key = normalize_hash(raw)
                if key and key in idx.by_sha:
                    rec = idx.by_sha[key]
                    break
            if rec is None:
                continue
            if rec.version_id is None and vid is not None:
                object.__setattr__(rec, "version_id", vid)
                idx.by_version[vid] = rec
            if rec.model_id is None and mid is not None:
                object.__setattr__(rec, "model_id", mid)
                idx.by_model.setdefault(mid, []).append(rec)
            if rec.path and rec.path not in self._relink_attempted:
                self._relink_attempted.add(rec.path)
                self._lm_fetch_civitai(rec)

    def _lm_fetch_civitai(self, rec: OwnedRecord) -> None:
        prefix = {"checkpoint": "checkpoints", "embedding": "embeddings"}.get(
            rec.model_type, "loras"
        )
        if rec.path:
            self._lm_post(f"/api/lm/{prefix}/fetch-civitai", {"file_path": rec.path})

    def relink_unlinked(self) -> dict[str, Any]:
        """Ask LM to resolve Unknown / missing Civitai ids by file hash."""
        idx = self.index()
        attempted = 0
        for rec in idx.records:
            if rec.version_id is not None or rec.model_id is not None:
                continue
            if not rec.path or not rec.sha256:
                continue
            if rec.path in self._relink_attempted:
                continue
            self._relink_attempted.add(rec.path)
            self._lm_fetch_civitai(rec)
            attempted += 1
        if attempted:
            self.invalidate()
        return {"attempted": attempted}

    def _download_via_lm(self, *, model_id: int | None, version_id: int | None,
                         allow_nsfw: bool, site: str) -> dict[str, Any]:
        # Import lazily so unit tests never need the CLI module unless they go through I/O.
        from boxa_lora_download import is_nsfw, resolve  # type: ignore

        info = resolve(model_id, version_id)
        if is_nsfw(info) and not allow_nsfw:
            return {"ok": False, "error": "model looks NSFW; pass allowNsfw or open it on civitai.red"}
        download_id = f"scb_{info['model_id']}_{info['version_id']}_{int(self._now())}"
        body = {
            "model_id": info["model_id"],
            "model_version_id": info["version_id"],
            "model_root": "",
            "relative_path": "",
            "use_default_paths": True,
            "download_id": download_id,
            "source": "civitai",
        }
        resp = self._lm_post("/api/lm/download-model", body)
        if resp.get("success"):
            return {
                "ok": True,
                "downloadId": download_id,
                "name": info.get("name"),
                "modelId": info.get("model_id"),
                "versionId": info.get("version_id"),
                "sizeMb": info.get("size_mb"),
            }
        err = str(resp.get("error") or "download failed")
        if "already exists" in err.lower():
            return {"ok": True, "already": True, "error": err, "name": info.get("name")}
        return {"ok": False, "error": err}


_bridge_singleton: CivitaiBridge | None = None
_bridge_lock = threading.Lock()


# Late-bound network kill switch (app/server.py wires it to app.network.state.enabled).
# None — the default for tests and standalone use — means everything is allowed.
NETWORK_ENABLED = None


def _service_enabled(service_id: str) -> bool:
    return NETWORK_ENABLED is None or bool(NETWORK_ENABLED(service_id))


def _gated_lm_get(path: str) -> dict | None:
    if not _service_enabled("lora_manager"):
        return None
    return _default_lm_get(path)


def _gated_lm_post(path: str, body: dict) -> dict:
    if not _service_enabled("lora_manager"):
        return {"success": False, "error": "LoRA Manager 已在网络页停用"}
    return _default_lm_post(path, body)


def _gated_civitai_get(path: str, *, params: dict | None = None, site: str = "com") -> dict | None:
    if not _service_enabled("civitai"):
        return None
    return _default_civitai_get(path, params=params, site=site)


def get_bridge() -> CivitaiBridge:
    global _bridge_singleton
    with _bridge_lock:
        if _bridge_singleton is None:
            # The gated seams consult the kill switch per call, so one singleton
            # serves both enabled and disabled states with zero I/O when off.
            _bridge_singleton = CivitaiBridge(
                lm_get=_gated_lm_get, lm_post=_gated_lm_post, civitai_get=_gated_civitai_get,
            )
        return _bridge_singleton


def _token() -> str | None:
    try:
        text = TOKEN_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _site_base(site: str) -> str:
    if site == "red":
        return CIVITAI_RED
    return CIVITAI_COM


def _default_civitai_get(path: str, *, params: dict | None = None, site: str = "com") -> dict | None:
    """GET Civitai Site REST. `site=auto` tries .red then .com for version lookups."""
    token = _token()
    sites = ["red", "com"] if site == "auto" else [site if site in {"red", "com"} else "com"]
    last: dict | None = None
    for which in sites:
        base = _site_base(which)
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        cmd = ["curl", "-sS", "--max-time", "25", "-H", "User-Agent: loramake-civitai-bridge/1.0", url]
        cfg = None
        extra: list[str] = []
        if token:
            extra = ["--config", "-"]
            cfg = f'header = "Authorization: Bearer {token}"\n'
        raw = _run_curl(cmd + extra, cfg)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("error"):
            last = data
            continue
        if isinstance(data, dict):
            return data
        last = data if isinstance(data, dict) else None
    return last


def _lm_hosts() -> list[str]:
    """Prefer ComfyUI :8188 (boxa_lora_download contract), then standalone :9000."""
    hosts = [BOXA_HOST.rstrip("/")]
    stand = LM_STANDALONE.rstrip("/")
    if stand and stand not in hosts:
        hosts.append(stand)
    return hosts


def _default_lm_get(path: str) -> dict | None:
    for host in _lm_hosts():
        noproxy = urlparse(host).hostname or ""
        cmd = ["curl", "-s", "--noproxy", noproxy, "--max-time", "20", f"{host}{path}"]
        try:
            # timeout= backstops a wedged curl itself; --max-time only caps the transfer
            raw = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
        except subprocess.TimeoutExpired:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = dict(data)
            data["_source"] = host
            return data
    return None


def _default_lm_post(path: str, body: dict) -> dict:
    last: dict = {"success": False, "error": "no LoRA Manager host answered"}
    for host in _lm_hosts():
        noproxy = urlparse(host).hostname or ""
        cmd = [
            "curl", "-s", "--noproxy", noproxy, "--max-time", "30",
            "-X", "POST", "-H", "Content-Type: application/json",
            "-d", json.dumps(body), f"{host}{path}",
        ]
        try:
            raw = subprocess.run(cmd, capture_output=True, text=True, timeout=35).stdout
        except subprocess.TimeoutExpired:
            last = {"success": False, "error": f"curl to {host} hung past 35s"}
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            last = {"success": False, "error": f"non-JSON from {host}: {raw[:160]}"}
            continue
        if isinstance(data, dict):
            data = dict(data)
            data["_source"] = host
            return data
    return last


def _run_curl(cmd: list[str], cfg_text: str | None) -> str:
    # Credentials travel via `--config -` on stdin: no argv exposure, and no
    # preexec_fn/fd plumbing (preexec_fn forces a full fork() of the server,
    # which is documented-unsafe under threads).
    try:
        proc = subprocess.run(cmd, input=cfg_text, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout or ""
