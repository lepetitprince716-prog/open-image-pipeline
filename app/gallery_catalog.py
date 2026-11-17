"""Zone-separated catalog for generated gallery media.

The legacy gallery scanner remains the metadata authority.  This adapter turns its
repo-relative records into a frontend-neutral API projection with opaque IDs and
safe media resolvers.  Write scope is limited to recoverable trash-delete: original
+ sidecar move into the legacy ``outputs/_trash/<run-stamp>/`` layout shared with
scripts/gallery_app.py, so the legacy gallery UI can restore them.  It deliberately
exposes no path leak, purge, restore, rating, or favourite operation.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gallery_scan  # noqa: E402  (metadata-only scanner; no server side effects)
from loramake import image_meta  # noqa: E402

SCAN_TTL_S = 30.0
# 64 是球幕画廊常驻层的档位：1447 张一次性铺满球面，256 档会让首屏下行量翻十几倍。
THUMB_WIDTHS = (64, 256, 480, 768)
THUMB_CACHE = ROOT / "app" / "data" / "gallery_thumb_cache"
NSFW_THUMB_CACHE = ROOT / "nsfw" / "app_cache" / "gallery_thumbs"
# Legacy trash convention (scripts/gallery_app.py): outputs/_trash/<run-stamp>/<repo-rel>.
TRASH_ROOT = ROOT / "outputs" / "_trash"
MAX_DELETE_BATCH = 200
IMMUTABLE_CACHE = {"Cache-Control": "private, max-age=86400, immutable"}
MAX_QUERY_LENGTH = 160

# Sort orders accepted by list_gallery. Offset pagination requires a deterministic
# total order, so every key ends with media_id. "newest" matches the snapshot build
# order (mtime desc, media_id tiebreak) and stays the backwards-compatible default.
SORT_ORDERS = ("newest", "oldest", "largest", "smallest", "name")

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


class GalleryError(Exception):
    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class _AssetFingerprint:
    size: int
    mtime_ns: int
    inode: int
    device: int

    @property
    def revision(self) -> str:
        raw = f"{self.size}:{self.mtime_ns}:{self.inode}:{self.device}".encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class _Record:
    media_id: str
    zone: str
    path: Path
    source_root: Path
    repo_rel: str
    fingerprint: _AssetFingerprint
    public: dict[str, Any]


@dataclass(frozen=True)
class _Snapshot:
    zone: str
    revision: str
    records: tuple[_Record, ...]
    by_id: dict[str, _Record]
    built_at: float
    index_stale: bool = False


logger = logging.getLogger(__name__)

_cache_lock = threading.RLock()
_snapshots: dict[str, _Snapshot] = {}
_last_scan_monotonic = 0.0
_rescan_in_flight = False
_rescan_failed = False
# media_id_for_path's negative cache: media_id -> the file state read just before a rescan that
# still did not index it. Inclusion depends only on the file, so while that state holds another
# rescan cannot change the answer. Bounded by clearing; a cleared entry costs one more rescan.
MISS_CACHE_MAX = 4096
_miss_states: dict[str, tuple[int, int, int, int]] = {}


def reset_cache() -> None:
    """Clear the process-local snapshot cache (primarily useful for tests)."""
    global _last_scan_monotonic, _rescan_in_flight, _rescan_failed
    with _cache_lock:
        _snapshots.clear()
        _miss_states.clear()
        _last_scan_monotonic = 0.0
        _rescan_in_flight = False
        _rescan_failed = False


def invalidate() -> None:
    """Expire the snapshot TTL so the next read rescans (the last snapshot keeps serving
    until then). ``jobs._finish`` calls it when a job lands as done; batch registration
    forces a rescan through ``refresh`` instead."""
    global _last_scan_monotonic
    with _cache_lock:
        _last_scan_monotonic = 0.0


def refresh(zone: str) -> _Snapshot:
    """Force a rescan now and return the fresh snapshot for ``zone``."""
    if zone not in {"sfw", "nsfw"}:
        raise GalleryError(400, "invalid_gallery_zone", "gallery zone is unsupported")
    return _scan_now(zone)


def _relative_to_source(path: Path, root: Path) -> PurePosixPath | None:
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return PurePosixPath(candidate.relative_to(base).as_posix())
        except ValueError:
            continue
    return None


def _file_state(path: Path) -> tuple[int, int, int, int] | None:
    """(size, mtime, ctime, inode) of ``path``, or None when it is not a readable file.

    ctime is in the key because a move into a source root keeps the mtime but changes the ctime.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


def media_id_for_path(zone: str, path: Path | str) -> str | None:
    """The catalog id of a file under one of ``zone``'s source roots, or None when it is not indexed.

    A miss rescans once before answering: a render that just landed may postdate the snapshot.
    A file that does not exist answers None without a scan, and a file a rescan already left out
    is not rescanned again until its state (size, mtime, ctime, inode) changes.
    """
    if zone not in {"sfw", "nsfw"}:
        raise GalleryError(400, "invalid_gallery_zone", "gallery zone is unsupported")
    target = Path(path)
    media_id = None
    for source in gallery_scan.SOURCES:
        if source["rating"] != zone:
            continue
        rel = _relative_to_source(target, Path(source["root"]))
        if rel is not None and rel.parts and ".." not in rel.parts:
            media_id = _media_id(zone, f"{source['prefix']}/{rel}")
            break
    if media_id is None:
        return None
    if media_id in _get_snapshot(zone).by_id:
        return media_id
    state = _file_state(target)
    if state is None:
        return None  # the scanner indexes existing files only
    with _cache_lock:
        if _miss_states.get(media_id) == state:
            return None
    snapshot = refresh(zone)
    if media_id in snapshot.by_id:
        return media_id
    if not snapshot.index_stale:  # a failed rescan proves nothing about this file
        with _cache_lock:
            if len(_miss_states) >= MISS_CACHE_MAX:
                _miss_states.clear()
            _miss_states[media_id] = state
    return None


def get_meta(zone: str, media_id: str) -> tuple[dict[str, Any], str]:
    """(``loramake.api.image-meta.v1`` payload, etag) for one catalog asset.

    The sidecar is read on demand rather than carried in every page; the PNG iTXt copy
    is the fallback when the sidecar was moved or never written.  ``meta`` is the
    public view (canonical keys, no machine addresses) or None when neither exists.
    """
    record = _validated_record(zone, media_id)
    meta = image_meta.read_meta(record.path)
    source = "none"
    if meta is not None:
        source = str(meta.get("_source") or "sidecar")
        meta = image_meta.public_view(meta)
    sidecar = image_meta.sidecar_path(record.path)
    try:
        sidecar_stamp = sidecar.stat().st_mtime_ns if sidecar.is_file() else 0
    except OSError:
        sidecar_stamp = 0
    etag = hashlib.sha256(
        f"{record.fingerprint.revision}:{sidecar_stamp}".encode()
    ).hexdigest()[:24]
    payload = {
        "schema": "loramake.api.image-meta.v1",
        "zone": zone,
        "media_id": media_id,
        "asset_revision": record.fingerprint.revision,
        "source": source,
        "meta": meta,
    }
    return payload, f'"meta-{etag}"'


def _media_id(zone: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{zone}\0{relative_path}".encode("utf-8")).hexdigest()
    return f"med_{digest[:24]}"


def _source_for_item(item: Any) -> tuple[Path, PurePosixPath]:
    raw_path = str(item.path)
    for source in gallery_scan.SOURCES:
        prefix = str(source["prefix"]).strip("/")
        marker = f"{prefix}/"
        if not raw_path.startswith(marker):
            continue
        rel_text = raw_path[len(marker) :]
        rel = PurePosixPath(rel_text)
        if (
            not rel.parts
            or rel.is_absolute()
            or any(part in {"", ".", ".."} for part in rel.parts)
        ):
            break
        return Path(source["root"]), rel
    raise GalleryError(400, "invalid_gallery_source", "gallery item is outside configured roots")


def _safe_regular_file(source_root: Path, rel: PurePosixPath) -> Path:
    """Resolve a scanner record without following any symlink below its source root."""
    try:
        root_resolved = source_root.resolve(strict=True)
    except OSError as exc:
        raise GalleryError(409, "gallery_source_changed", "gallery source is unavailable") from exc

    cursor = source_root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GalleryError(409, "unsafe_gallery_asset", "symlinked gallery assets are not served")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise GalleryError(409, "unsafe_gallery_asset", "gallery asset escaped its source root") from exc
    if not resolved.is_file():
        raise GalleryError(409, "gallery_asset_changed", "gallery asset is no longer a regular file")
    return resolved


def _fingerprint(path: Path) -> _AssetFingerprint:
    stat = path.stat()
    return _AssetFingerprint(
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
    )


def _recipe_sidecar_candidates(path: Path) -> tuple[Path, ...]:
    """Accepted recipe sidecar names next to a media file.

    Contract (nsfw/RECIPES.md): optional ``<image>.recipe.json``.  Support both
    ``foo.png.recipe.json`` and ``foo.recipe.json`` so either writer convention works.
    """
    return (
        path.with_name(f"{path.name}.recipe.json"),
        path.with_suffix(".recipe.json"),
    )


def _load_recipe_dict(path: Path) -> dict[str, Any] | None:
    for candidate in _recipe_sidecar_candidates(path):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == "nsfw-recipe-v1":
            return payload
    return None


def _path_character_hint(item: Any) -> str | None:
    """Infer character id from nsfw/outputs/<character>/… layout."""
    try:
        parts = PurePosixPath(str(item.path)).parts
    except (TypeError, ValueError):
        return None
    if len(parts) >= 3 and parts[0] == "nsfw" and parts[1] == "outputs":
        hint = parts[2].strip()
        if hint and hint not in {".", ".."} and not hint.startswith("_"):
            return hint
    return None


def _recipe_public_fields(item: Any, path: Path) -> dict[str, Any]:
    """Path-free recipe/plan facets for catalog filter + inspector.

    Both zones run the same fallback chain: recipe sidecar wins when present;
    character falls back to the nsfw path hint (naturally None for SFW); base
    falls back to the checkpoint stem — so SFW assets stay filterable by base
    without a sidecar, while tier/l1 remain NSFW-recipe taxonomy.
    """
    recipe = _load_recipe_dict(path)
    character = None
    l1_flags: list[str] = []
    leaf_ids: list[str] = []
    tier = None
    prompt_domain = None
    base = None
    recipe_id = None
    recipe_request = None

    if recipe is not None:
        raw_character = recipe.get("character")
        if isinstance(raw_character, str) and raw_character.strip():
            character = raw_character.strip()
        raw_flags = recipe.get("l1_flags")
        if isinstance(raw_flags, list):
            l1_flags = [str(x).strip() for x in raw_flags if str(x).strip()]
        raw_leaves = recipe.get("leaf_ids")
        if isinstance(raw_leaves, list):
            leaf_ids = [str(x).strip() for x in raw_leaves if str(x).strip()]
        raw_tier = recipe.get("tier")
        if isinstance(raw_tier, str) and raw_tier.strip():
            tier = raw_tier.strip().upper()
        raw_domain = recipe.get("prompt_domain")
        if isinstance(raw_domain, str) and raw_domain.strip():
            prompt_domain = raw_domain.strip()
        raw_base = recipe.get("base")
        if isinstance(raw_base, str) and raw_base.strip():
            base = raw_base.strip()
        raw_id = recipe.get("recipe_id")
        if isinstance(raw_id, str) and raw_id.strip():
            recipe_id = raw_id.strip()
        raw_request = recipe.get("request")
        if isinstance(raw_request, str) and raw_request.strip():
            recipe_request = raw_request.strip()[:200]

    if character is None:
        character = _path_character_hint(item)
    if base is None:
        ckpt = getattr(item, "checkpoint", None)
        if isinstance(ckpt, str) and ckpt.strip():
            base = Path(ckpt.replace("\\", "/")).stem

    return {
        "character": character,
        "l1_flags": l1_flags,
        "leaf_ids": leaf_ids,
        "tier": tier,
        "prompt_domain": prompt_domain,
        "base": base,
        "recipe_id": recipe_id,
        "recipe_request": recipe_request,
    }


def _public_item(
    item: Any,
    media_id: str,
    fingerprint: _AssetFingerprint,
    *,
    path: Path,
) -> dict[str, Any]:
    data = asdict(item)
    data.pop("path", None)
    data.pop("rating", None)
    # `box` (boxa / boxb) is public since 2026-09-03: the inspector shows where a render was
    # made and batch cards group by it. Machine addresses (`host`) never leave the process.
    data.pop("host", None)
    data.pop("git_tracked", None)
    data["media_id"] = media_id
    data["zone"] = str(item.rating)
    data["bytes"] = fingerprint.size
    data["asset_revision"] = fingerprint.revision
    data["variants"] = (
        ["thumb_64", "thumb_256", "thumb_480", "thumb_768", "original"]
        if item.media == "image"
        else ["original"]
    )
    data.update(_recipe_public_fields(item, path))
    return data


def _record_from_item(item: Any) -> _Record:
    zone = str(item.rating)
    if zone not in {"sfw", "nsfw"}:
        raise GalleryError(400, "invalid_gallery_zone", "gallery item has an unsupported zone")
    source_root, rel = _source_for_item(item)
    path = _safe_regular_file(source_root, rel)
    suffix = path.suffix.lower()
    expected_media = "video" if suffix in _VIDEO_MEDIA_TYPES else "image"
    if suffix not in {*_IMAGE_MEDIA_TYPES, *_VIDEO_MEDIA_TYPES} or item.media != expected_media:
        raise GalleryError(409, "unsupported_gallery_media", "gallery media type is unsupported")
    fingerprint = _fingerprint(path)
    media_id = _media_id(zone, str(item.path))
    return _Record(
        media_id=media_id,
        zone=zone,
        path=path,
        source_root=source_root.resolve(strict=True),
        repo_rel=str(item.path),
        fingerprint=fingerprint,
        public=_public_item(item, media_id, fingerprint, path=path),
    )


def _snapshot_for_zone(zone: str, records: list[_Record], *, built_at: float) -> _Snapshot:
    records.sort(key=lambda record: (-float(record.public["mtime"]), record.media_id))
    revision_hasher = hashlib.sha256()
    for record in records:
        revision_hasher.update(record.media_id.encode())
        revision_hasher.update(record.fingerprint.revision.encode())
    revision = revision_hasher.hexdigest()
    frozen_records = tuple(records)
    return _Snapshot(
        zone=zone,
        revision=revision,
        records=frozen_records,
        by_id={record.media_id: record for record in frozen_records},
        built_at=built_at,
    )


def _refresh_snapshots() -> None:
    """Rescan and swap in fresh snapshots.

    The scan is deliberately performed without holding ``_cache_lock`` so readers keep
    serving the previous snapshot for its full duration; only the swap-in is serialised.
    """
    global _last_scan_monotonic
    now = time.monotonic()
    scanned = gallery_scan.scan()
    by_zone: dict[str, list[_Record]] = {"sfw": [], "nsfw": []}
    for item in scanned:
        try:
            record = _record_from_item(item)
        except (GalleryError, OSError):
            # A broken or unsafe scanner record is excluded rather than weakening the
            # resolver boundary for the rest of the gallery.
            continue
        by_zone[record.zone].append(record)
    built_at = time.time()
    fresh = {
        zone: _snapshot_for_zone(zone, records, built_at=built_at)
        for zone, records in by_zone.items()
    }
    global _rescan_failed
    with _cache_lock:
        if _snapshots and now < _last_scan_monotonic:
            # A scan that started later already landed; discarding this older result keeps
            # an explicit refresh() from being overwritten by a background pass behind it.
            return
        _snapshots.update(fresh)
        _last_scan_monotonic = now
        _rescan_failed = False


def _background_rescan() -> None:
    """Refresh snapshots off the request path. A failed scan leaves the previous snapshot
    in place; the next expiry retries."""
    global _rescan_in_flight, _rescan_failed
    with _cache_lock:
        started_after = _last_scan_monotonic
    failed = False
    try:
        _refresh_snapshots()
    except Exception:
        failed = True
        logger.exception("gallery background rescan failed; the previous snapshot keeps serving")
    finally:
        with _cache_lock:
            _rescan_in_flight = False
            # Only flag staleness if nothing newer landed while this pass was failing;
            # otherwise a slow failure would re-mark a snapshot a later scan just refreshed.
            if failed and _last_scan_monotonic == started_after:
                _rescan_failed = True


def _spawn_rescan() -> None:
    """Start at most one background rescan. Caller must hold ``_cache_lock``."""
    global _rescan_in_flight
    if _rescan_in_flight:
        return
    _rescan_in_flight = True
    threading.Thread(target=_background_rescan, name="gallery-rescan", daemon=True).start()


def _scan_now(zone: str) -> _Snapshot:
    """Scan on the calling thread, degrading to the previous snapshot if the scan fails."""
    try:
        _refresh_snapshots()
    except Exception as exc:
        with _cache_lock:
            cached = _snapshots.get(zone)
        if cached is not None:
            return replace(cached, index_stale=True)
        raise GalleryError(503, "gallery_scan_failed", "gallery index is unavailable") from exc
    with _cache_lock:
        return _snapshots[zone]


def _get_snapshot(zone: str) -> _Snapshot:
    if zone not in {"sfw", "nsfw"}:
        raise GalleryError(400, "invalid_gallery_zone", "gallery zone is unsupported")
    now = time.monotonic()
    with _cache_lock:
        cached = _snapshots.get(zone)
        if cached is not None:
            # Stale-while-revalidate: an expired snapshot keeps serving while one
            # background rescan runs, so no request ever waits on a full scan.
            if now - _last_scan_monotonic >= SCAN_TTL_S:
                _spawn_rescan()
            return replace(cached, index_stale=True) if _rescan_failed else cached
    # ponytail: cold start is unlocked, so a startup burst can run a couple of scans in
    # parallel. Once per process, and serialising it would put the lock back on the
    # request path -- add a scan lock only if startup cost ever shows up.
    return _scan_now(zone)


def _search_blob(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "name",
        "collection",
        "driving_model",
        "line",
        "intent",
        "engine",
        "box",
        "checkpoint",
        "outfit",
        "concept",
        "prompt",
        "engine_line",
        "run_date",
        "character",
        "tier",
        "prompt_domain",
        "base",
        "recipe_id",
        "recipe_request",
    ):
        value = item.get(key)
        if value is not None:
            values.append(str(value))
    values.extend(str(value) for value in item.get("loras", []))
    values.extend(str(value) for value in item.get("l1_flags", []))
    values.extend(str(value) for value in item.get("leaf_ids", []))
    return "\n".join(values).casefold()


def _normalize_filter(value: str | None, *, max_len: int = 64) -> str:
    text = " ".join((value or "").split())
    if len(text) > max_len:
        raise GalleryError(400, "gallery_filter_too_long", "gallery filter value is too long")
    return text


def _cursor_signature(
    query: str,
    curated_only: bool,
    *,
    sort: str = "newest",
    character: str = "",
    tier: str = "",
    l1: str = "",
    base: str = "",
    recipe_id: str = "",
) -> str:
    raw = "\0".join(
        (
            query.casefold(),
            str(int(curated_only)),
            sort,
            character.casefold(),
            tier.casefold(),
            l1.casefold(),
            base.casefold(),
            recipe_id.casefold(),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalize_sort(value: str | None) -> str:
    sort = " ".join((value or "").split()).casefold() or "newest"
    if sort not in SORT_ORDERS:
        raise GalleryError(400, "invalid_gallery_sort", "gallery sort order is unsupported")
    return sort


def _sorted_records(records: list[_Record], sort: str) -> list[_Record]:
    if sort == "newest":
        # Snapshot order is already mtime desc with media_id tiebreak.
        return records
    if sort == "oldest":
        return sorted(records, key=lambda r: (float(r.public["mtime"]), r.media_id))
    if sort == "largest":
        return sorted(
            records,
            key=lambda r: (-int(r.public["bytes"]), -float(r.public["mtime"]), r.media_id),
        )
    if sort == "smallest":
        return sorted(
            records,
            key=lambda r: (int(r.public["bytes"]), -float(r.public["mtime"]), r.media_id),
        )
    return sorted(records, key=lambda r: (str(r.public["name"]).casefold(), r.media_id))


def _matches_recipe_filters(
    public: dict[str, Any],
    *,
    character: str,
    tier: str,
    l1: str,
    base: str,
    recipe_id: str,
) -> bool:
    if character:
        value = str(public.get("character") or "")
        if value.casefold() != character.casefold():
            return False
    if tier:
        value = str(public.get("tier") or "")
        if value.casefold() != tier.casefold():
            return False
    if l1:
        flags = [str(x).casefold() for x in (public.get("l1_flags") or [])]
        leaves = [str(x).casefold() for x in (public.get("leaf_ids") or [])]
        needle = l1.casefold()
        if needle not in flags and not any(needle == leaf or leaf.startswith(f"{needle}/") for leaf in leaves):
            return False
    if base:
        value = str(public.get("base") or public.get("checkpoint") or "").casefold()
        if base.casefold() not in value:
            return False
    if recipe_id:
        value = str(public.get("recipe_id") or "")
        if value.casefold() != recipe_id.casefold():
            return False
    return True


def _encode_cursor(revision: str, offset: int, signature: str) -> str:
    raw = json.dumps(
        {"revision": revision, "offset": offset, "signature": signature},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, *, revision: str, signature: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        offset = int(payload["offset"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise GalleryError(400, "invalid_gallery_cursor", "gallery cursor is invalid") from exc
    if payload.get("revision") != revision or payload.get("signature") != signature:
        raise GalleryError(409, "stale_gallery_cursor", "gallery changed; restart pagination")
    if offset < 0:
        raise GalleryError(400, "invalid_gallery_cursor", "gallery cursor is invalid")
    return offset


def list_gallery(
    *,
    zone: str,
    cursor: str | None,
    limit: int,
    query: str,
    curated_only: bool,
    sort: str | None = None,
    character: str | None = None,
    tier: str | None = None,
    l1: str | None = None,
    base: str | None = None,
    recipe_id: str | None = None,
) -> dict[str, Any]:
    snapshot = _get_snapshot(zone)
    normalized_query = " ".join(query.split())
    if len(normalized_query) > MAX_QUERY_LENGTH:
        raise GalleryError(400, "gallery_query_too_long", "gallery query is too long")
    sort_f = _normalize_sort(sort)
    character_f = _normalize_filter(character)
    tier_f = _normalize_filter(tier, max_len=8)
    l1_f = _normalize_filter(l1)
    base_f = _normalize_filter(base)
    recipe_id_f = _normalize_filter(recipe_id, max_len=96)
    needle = normalized_query.casefold()
    records = [
        record
        for record in snapshot.records
        if (not curated_only or bool(record.public["curated"]))
        and (not needle or needle in _search_blob(record.public))
        and _matches_recipe_filters(
            record.public,
            character=character_f,
            tier=tier_f,
            l1=l1_f,
            base=base_f,
            recipe_id=recipe_id_f,
        )
    ]
    records = _sorted_records(records, sort_f)
    signature = _cursor_signature(
        normalized_query,
        curated_only,
        sort=sort_f,
        character=character_f,
        tier=tier_f,
        l1=l1_f,
        base=base_f,
        recipe_id=recipe_id_f,
    )
    offset = (
        _decode_cursor(cursor, revision=snapshot.revision, signature=signature)
        if cursor
        else 0
    )
    page = records[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "schema": "loramake.api.gallery-page.v1",
        "zone": zone,
        "catalog_revision": snapshot.revision,
        "built_at": snapshot.built_at,
        "index_stale": snapshot.index_stale,
        "items": [record.public for record in page],
        "total": len(records),
        "next_cursor": (
            _encode_cursor(snapshot.revision, next_offset, signature)
            if next_offset < len(records)
            else None
        ),
    }


def _validated_record(zone: str, media_id: str) -> _Record:
    snapshot = _get_snapshot(zone)
    record = snapshot.by_id.get(media_id)
    if record is None:
        raise GalleryError(404, "gallery_asset_not_found", "gallery asset was not found")
    try:
        rel = record.path.relative_to(record.source_root)
    except ValueError as exc:
        raise GalleryError(409, "unsafe_gallery_asset", "gallery asset escaped its source root") from exc
    path = _safe_regular_file(record.source_root, PurePosixPath(rel.as_posix()))
    try:
        current = _fingerprint(path)
    except OSError as exc:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed") from exc
    if current != record.fingerprint:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed; refresh the catalog")
    return record


def asset_path(zone: str, media_id: str) -> Path:
    """Absolute source path of a catalog asset (validated against its snapshot).
    Used by the jobs domain to reverse-map a gallery asset to the job that made it."""
    return _validated_record(zone, media_id).path


def get_asset(
    zone: str,
    media_id: str,
    variant: str,
    *,
    expected_revision: str | None = None,
) -> tuple[Path, str, str]:
    record = _validated_record(zone, media_id)
    if expected_revision is not None and expected_revision != record.fingerprint.revision:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed; refresh the catalog")
    suffix = record.path.suffix.lower()
    if variant == "original":
        media_type = (_IMAGE_MEDIA_TYPES | _VIDEO_MEDIA_TYPES)[suffix]
        return record.path, media_type, f'"gallery-{record.fingerprint.revision}"'
    if variant not in {f"thumb_{width}" for width in THUMB_WIDTHS}:
        raise GalleryError(400, "invalid_gallery_variant", "gallery asset variant is unsupported")
    if record.public["media"] != "image":
        raise GalleryError(415, "gallery_preview_unavailable", "video thumbnails are not available")
    width = int(variant.removeprefix("thumb_"))
    path = thumbnail(record, width)
    return path, "image/webp", f'"gallery-{record.fingerprint.revision}-{width}"'


def thumbnail(record: _Record, width: int) -> Path:
    if width not in THUMB_WIDTHS:
        raise GalleryError(400, "invalid_gallery_variant", "gallery thumbnail width is unsupported")
    cache_root = NSFW_THUMB_CACHE if record.zone == "nsfw" else THUMB_CACHE
    key = hashlib.sha256(
        f"{record.media_id}:{record.fingerprint.revision}:{width}".encode()
    ).hexdigest()
    cached = cache_root / f"{key}.webp"
    if cached.is_file():
        return cached

    try:
        with Image.open(record.path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", quality=82, method=4)
    except (OSError, ValueError) as exc:
        raise GalleryError(422, "gallery_thumbnail_failed", "gallery image could not be decoded") from exc

    # Detect replacement during decoding before publishing a derivative.
    if _fingerprint(record.path) != record.fingerprint:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed; refresh the catalog")
    cache_root.mkdir(parents=True, exist_ok=True)
    temp = cache_root / f".{key}.{threading.get_ident()}.tmp"
    temp.write_bytes(buffer.getvalue())
    temp.replace(cached)
    return cached


def _trash_record(record: _Record, stamp: str) -> None:
    """Move one catalog asset (original + sidecars) into the legacy trash layout.

    Same convention as scripts/gallery_app.py's ``_delete``: the destination keeps
    the full repo-relative path under ``_trash/<run-stamp>/`` so SFW and NSFW stay
    separated inside the trash and the legacy restore maps straight back.  The
    asset is re-validated against its snapshot fingerprint before anything moves.
    Never hard-deletes.
    """
    try:
        rel = record.path.relative_to(record.source_root)
    except ValueError as exc:
        raise GalleryError(409, "unsafe_gallery_asset", "gallery asset escaped its source root") from exc
    path = _safe_regular_file(record.source_root, PurePosixPath(rel.as_posix()))
    try:
        current = _fingerprint(path)
    except OSError as exc:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed") from exc
    if current != record.fingerprint:
        raise GalleryError(409, "gallery_asset_changed", "gallery asset changed; refresh the catalog")

    repo_rel = PurePosixPath(record.repo_rel)
    if (
        not repo_rel.parts
        or repo_rel.is_absolute()
        or any(part in {"", ".", ".."} for part in repo_rel.parts)
    ):
        raise GalleryError(409, "unsafe_gallery_asset", "gallery asset escaped its source root")
    dst = TRASH_ROOT / stamp / Path(*repo_rel.parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # .meta.json is the legacy convention; .recipe.json variants ride along so a
    # trashed item keeps every sidecar it had (never orphaned, never hard-deleted).
    sidecars = [
        candidate
        for candidate in (
            path.with_name(f"{path.name}.meta.json"),
            *_recipe_sidecar_candidates(path),
        )
        if candidate.is_file() and not candidate.is_symlink()
    ]
    shutil.move(str(path), str(dst))
    for sidecar in sidecars:
        shutil.move(str(sidecar), str(dst.with_name(sidecar.name)))


def delete_media(*, zone: str, media_ids: list[str]) -> dict[str, Any]:
    """Batch trash-delete catalog media for one zone; per-item outcomes, never purge.

    IDs resolve through the zone's own snapshot, so a foreign-zone ID is simply
    ``not_found`` — a zone endpoint can never touch the other zone's media.  After
    any successful move the snapshot cache is invalidated and rebuilt so the
    returned revision already excludes the trashed items.
    """
    if zone not in {"sfw", "nsfw"}:
        raise GalleryError(400, "invalid_gallery_zone", "gallery zone is unsupported")
    if not media_ids:
        raise GalleryError(400, "gallery_delete_empty", "media_ids must contain at least one id")
    if len(media_ids) > MAX_DELETE_BATCH:
        raise GalleryError(
            400,
            "gallery_delete_too_many",
            f"media_ids is capped at {MAX_DELETE_BATCH} per request",
        )
    snapshot = _get_snapshot(zone)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    for media_id in dict.fromkeys(media_ids):  # dedupe, keep request order
        record = snapshot.by_id.get(media_id)
        if record is None:
            failed.append({"media_id": media_id, "reason": "not_found"})
            continue
        try:
            _trash_record(record, stamp)
        except GalleryError as exc:
            failed.append({"media_id": media_id, "reason": exc.code})
        except OSError:
            failed.append({"media_id": media_id, "reason": "delete_failed"})
        else:
            deleted.append(media_id)
    revision = snapshot.revision
    if deleted:
        # refresh() rescans outside _cache_lock. Clearing the cache under the lock and
        # rebuilding it there put a full scan back on every concurrent reader's path --
        # the same defect the read path was fixed for, reached through delete instead.
        revision = refresh(zone).revision
    return {
        "schema": "loramake.api.gallery-delete.v1",
        "zone": zone,
        "deleted": deleted,
        "failed": failed,
        "catalog_revision": revision,
    }
