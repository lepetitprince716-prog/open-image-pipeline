"""Review-batch engine behind the gallery's ``/review``, ``/view`` and ``/triage`` pages.

Extracted verbatim from ``scripts/gallery_app.py`` so the batch grammar, manifest
validation, verdict normalization and integrity checks can be read and tested apart from
the HTTP server. This module owns no repository layout: every filesystem root arrives as
a :class:`ReviewPaths` bundle built by the caller, which is why ``gallery_app`` builds a
fresh one per call from its own module globals.

Two on-disk files per batch, under ``<review root>/<batch id>/``:

``manifest.json``
    What was registered — schema v1 (pre-video, pre-caption, pre-mode) or v2. v1 is read,
    not migrated: ``media`` is inferred from the extension, ``caption`` defaults to empty
    and ``mode`` to ``review``.
``result.json``
    The verdicts, rewritten atomically on every decision: ``y`` yes, ``l`` love, ``n`` no,
    ``c`` comments, ``r`` rejection reasons. One verdict per item; reasons attach to
    rejections only.

Every failure is a :class:`ReviewError` carrying the HTTP status the API should answer
with, so the server layer stays a thin translation of exception to response.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

BATCH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
VIDEO_EXTS = {".mp4", ".webm"}
REVIEW_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
REVIEW_MEDIA_EXTS = REVIEW_IMAGE_EXTS | VIDEO_EXTS
REVIEW_MODES = {"review", "view"}
CAPTION_MAX = 200
REVIEW_BODY_MAX = 64 * 1024
REVIEW_LOCK = threading.Lock()
REASON_VOCAB = ("style", "nonsense", "face", "hands", "figure",
                "composition", "foreign_object")
# (path, size, mtime_ns) -> sha256, so streamed media is hashed once rather than once per
# Range request. See verify_review_media.
MEDIA_DIGESTS: dict[tuple[str, int, int], str] = {}
MEDIA_DIGEST_LOCK = threading.Lock()
MEDIA_DIGEST_MAX = 512


@dataclass(frozen=True)
class ReviewPaths:
    """The two filesystem roots the review layer is allowed to touch.

    Frozen and passed per call rather than read from module state: the caller owns the
    repository layout, and a caller that rebinds a root between calls (a test pointing the
    gallery at a scratch tree, for instance) must be able to steer this module without
    reimporting it.
    """

    review_root: Path
    repo: Path


class ReviewError(Exception):
    """A review request error with an HTTP status suitable for the API response."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def review_batch_id(value: object) -> str:
    """Validate the CLI/server shared batch-id grammar."""
    if not isinstance(value, str) or not BATCH_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ReviewError(400, "invalid batch")
    return value


def review_batch_dir(paths: ReviewPaths, batch: object) -> Path:
    """Resolve a batch directory while rejecting symlink/traversal escapes."""
    batch_id = review_batch_id(batch)
    root = paths.review_root.resolve()
    direct = root / batch_id
    if direct.is_symlink():
        raise ReviewError(400, "invalid batch")
    candidate = direct.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReviewError(400, "invalid batch") from exc
    return candidate


def review_file(paths: ReviewPaths, batch: object, name: str) -> Path:
    """Return a fixed file within a confined batch directory."""
    batch_dir = review_batch_dir(paths, batch)
    direct = batch_dir / name
    if direct.is_symlink():
        raise ReviewError(400, "invalid batch path")
    target = direct.resolve()
    try:
        target.relative_to(batch_dir)
    except ValueError as exc:
        raise ReviewError(400, "invalid batch path") from exc
    return target


def read_review_json(path: Path, *, missing_status: int, label: str) -> dict:
    if not path.is_file():
        raise ReviewError(missing_status, f"{label} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(500, f"invalid {label}") from exc
    if not isinstance(data, dict):
        raise ReviewError(500, f"invalid {label}")
    return data


def load_review_manifest(paths: ReviewPaths, batch: object) -> dict:
    """Load and minimally validate one batch manifest.

    Accepts both schema versions. v1 predates videos, captions and view mode, so its
    fields are filled in here rather than migrated on disk: ``media`` is inferred from
    the file extension, ``caption`` defaults to empty and ``mode`` to ``review``. Every
    v1 batch registered before this change therefore keeps working untouched.

    The browser never receives the returned raw objects; API serialization goes through
    ``review_batch_payload`` below, which deliberately omits ``path`` and
    ``source_path``.
    """
    batch_id = review_batch_id(batch)
    data = read_review_json(
        review_file(paths, batch_id, "manifest.json"), missing_status=404, label="batch"
    )
    if data.get("v") not in (1, 2) or data.get("batch") != batch_id \
            or not isinstance(data.get("items"), list):
        raise ReviewError(500, "invalid batch manifest")
    mode = data.get("mode", "review")
    if mode not in REVIEW_MODES:
        raise ReviewError(500, "invalid batch manifest")

    items: list[dict] = []
    seen: set[int] = set()
    for raw in data["items"]:
        if not isinstance(raw, dict):
            raise ReviewError(500, "invalid batch manifest")
        index = raw.get("i")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ReviewError(500, "invalid batch manifest")
        if any(not isinstance(raw.get(key), str) or not raw[key]
               for key in ("kind", "path", "name", "sha256", "source_path")):
            raise ReviewError(500, "invalid batch manifest")
        if raw["kind"] not in {"repo", "batch_asset"}:
            raise ReviewError(500, "invalid batch manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", raw["sha256"]):
            raise ReviewError(500, "invalid batch manifest")
        suffix = Path(raw["path"]).suffix.lower()
        media = raw.get("media", "video" if suffix in VIDEO_EXTS else "image")
        if media not in {"image", "video"}:
            raise ReviewError(500, "invalid batch manifest")
        caption = raw.get("caption", "")
        if not isinstance(caption, str) or len(caption) > CAPTION_MAX:
            raise ReviewError(500, "invalid batch manifest")
        seen.add(index)
        items.append({
            "i": index,
            "kind": raw["kind"],
            "path": raw["path"],
            "name": raw["name"],
            "media": media,
            "caption": caption,
            "sha256": raw["sha256"],
            "source_path": raw["source_path"],
        })
    items.sort(key=lambda item: item["i"])
    return {"v": 2, "batch": batch_id, "mode": mode, "items": items}


def empty_review_result(batch: str) -> dict:
    return {"v": 1, "batch": batch, "y": [], "l": [], "n": [], "c": {}, "r": {}}


def normalize_review_result(data: dict, batch: str, valid_indices: set[int] | None = None) -> dict:
    if data.get("v") != 1 or data.get("batch") != batch:
        raise ReviewError(500, "invalid review result")
    yes = data.get("y")
    no = data.get("n")
    comments = data.get("c")
    # "l" (love) postdates the first result.json shape, so a batch decided before it
    # existed simply has no such key. Default it rather than rejecting the file.
    love = data.get("l", [])
    # "r" (rejection reasons) postdates "l"; same default-to-empty treatment.
    raw_reasons = data.get("r", {})
    if not isinstance(yes, list) or not isinstance(no, list) or not isinstance(comments, dict) \
            or not isinstance(love, list) or not isinstance(raw_reasons, dict):
        raise ReviewError(500, "invalid review result")

    def indexes(values: list) -> set[int]:
        out: set[int] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReviewError(500, "invalid review result")
            if valid_indices is not None and value not in valid_indices:
                raise ReviewError(500, "invalid review result")
            out.add(value)
        return out

    y_set = indexes(yes)
    n_set = indexes(no)
    l_set = indexes(love)
    c_map: dict[int, str] = {}
    for key, value in comments.items():
        if not isinstance(key, str) or not re.fullmatch(r"(?:0|[1-9]\d*)", key):
            raise ReviewError(500, "invalid review result")
        index = int(key)
        if valid_indices is not None and index not in valid_indices:
            raise ReviewError(500, "invalid review result")
        if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 500:
            raise ReviewError(500, "invalid review result")
        c_map[index] = value
    r_map: dict[int, list[str]] = {}
    for key, value in raw_reasons.items():
        if not isinstance(key, str) or not re.fullmatch(r"(?:0|[1-9]\d*)", key):
            raise ReviewError(500, "invalid review result")
        index = int(key)
        if valid_indices is not None and index not in valid_indices:
            raise ReviewError(500, "invalid review result")
        if not isinstance(value, list):
            raise ReviewError(500, "invalid review result")
        clean: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or item not in REASON_VOCAB:
                raise ReviewError(500, "invalid review result")
            if item not in seen:
                seen.add(item)
                clean.append(item)
        if len(clean) > len(REASON_VOCAB):
            raise ReviewError(500, "invalid review result")
        r_map[index] = clean
    # One verdict per item: no index may appear in two of the four buckets.
    if len(y_set | l_set | n_set | c_map.keys()) != \
            len(y_set) + len(l_set) + len(n_set) + len(c_map):
        raise ReviewError(500, "invalid review result")
    # Reasons attach to rejections only.
    if not set(r_map).issubset(n_set):
        raise ReviewError(500, "invalid review result")
    return {
        "v": 1,
        "batch": batch,
        "y": sorted(y_set),
        "l": sorted(l_set),
        "n": sorted(n_set),
        "c": {str(index): c_map[index] for index in sorted(c_map)},
        "r": {str(index): r_map[index] for index in sorted(r_map)},
    }


def load_review_result(paths: ReviewPaths, batch: object,
                       valid_indices: set[int] | None = None) -> dict:
    batch_id = review_batch_id(batch)
    path = review_file(paths, batch_id, "result.json")
    if not path.exists():
        return empty_review_result(batch_id)
    data = read_review_json(path, missing_status=404, label="review result")
    return normalize_review_result(data, batch_id, valid_indices)


def atomic_write_review_result(paths: ReviewPaths, batch: str, result: dict) -> None:
    """Write result.json via a same-directory temporary and atomic replacement."""
    target = review_file(paths, batch, "result.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".result.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def review_batch_payload(paths: ReviewPaths, batch: object) -> dict:
    manifest = load_review_manifest(paths, batch)
    valid = {item["i"] for item in manifest["items"]}
    result = load_review_result(paths, manifest["batch"], valid)
    return {
        "v": 2,
        "batch": manifest["batch"],
        "mode": manifest["mode"],
        "items": [
            {key: item[key] for key in ("i", "name", "media", "caption")}
            for item in manifest["items"]
        ],
        "result": result,
    }


def review_media_path(paths: ReviewPaths, batch: object, index: object,
                      allowed_exts: set[str]) -> tuple[dict, Path]:
    """Resolve one batch item to a file on disk, confined to the batch or the repository.

    Path resolution only — no bytes are read here, so a video can be streamed by the
    server instead of being loaded whole. Integrity is checked separately by
    ``verify_review_media``.
    """
    manifest = load_review_manifest(paths, batch)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ReviewError(400, "invalid image index")
    item = next((item for item in manifest["items"] if item["i"] == index), None)
    if item is None:
        raise ReviewError(404, "image not found")

    raw_path = Path(item["path"])
    if raw_path.is_absolute():
        raise ReviewError(404, "image not found")
    try:
        if item["kind"] == "batch_asset":
            batch_dir = review_batch_dir(paths, manifest["batch"])
            assets_dir = batch_dir / "assets"
            if assets_dir.is_symlink():
                raise ReviewError(404, "image not found")
            allowed = assets_dir.resolve()
            allowed.relative_to(batch_dir)
            target = (batch_dir / raw_path).resolve()
            target.relative_to(allowed)
        else:
            repo = paths.repo.resolve()
            target = (repo / raw_path).resolve()
            target.relative_to(repo)
    except (OSError, ValueError) as exc:
        raise ReviewError(404, "image not found") from exc
    if target.suffix.lower() not in allowed_exts or not target.is_file():
        raise ReviewError(404, "image not found")
    return item, target


def verify_review_media(item: dict, target: Path) -> None:
    """Confirm the file still hashes to what registration recorded.

    Memoised on (path, size, mtime): a <video> element issues dozens of Range requests
    per clip, and re-hashing a hundred-megabyte file on each one would make playback
    unusable. Any write to the file changes size or mtime and forces a re-hash.
    """
    try:
        stat = target.stat()
    except OSError as exc:
        raise ReviewError(404, "image not found") from exc
    key = (str(target), stat.st_size, stat.st_mtime_ns)
    with MEDIA_DIGEST_LOCK:
        cached = MEDIA_DIGESTS.get(key)
    if cached is None:
        digest = hashlib.sha256()
        try:
            with target.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ReviewError(404, "image not found") from exc
        cached = digest.hexdigest()
        with MEDIA_DIGEST_LOCK:
            if len(MEDIA_DIGESTS) >= MEDIA_DIGEST_MAX:
                MEDIA_DIGESTS.clear()
            MEDIA_DIGESTS[key] = cached
    if cached != item["sha256"]:
        raise ReviewError(409, "image changed since registration")


def review_image(paths: ReviewPaths, batch: object, index: object) -> tuple[Path, bytes]:
    """Whole-file image read for the original /api/review/image endpoint."""
    item, target = review_media_path(paths, batch, index, REVIEW_IMAGE_EXTS)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ReviewError(404, "image not found") from exc
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ReviewError(409, "image changed since registration")
    return target, data


def validate_reasons(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise ReviewError(400, "reasons must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or value not in REASON_VOCAB:
            raise ReviewError(400, "unknown reason id")
        if value not in seen:
            seen.add(value)
            out.append(value)
    if len(out) > len(REASON_VOCAB):
        raise ReviewError(400, "too many reasons")
    return out


def review_decision(paths: ReviewPaths, batch: object, index: object, decision: object,
                    comment: object = None, reasons: object = None) -> dict:
    manifest = load_review_manifest(paths, batch)
    if manifest["mode"] == "view":
        raise ReviewError(400, "this batch was registered for viewing only")
    valid = {item["i"] for item in manifest["items"]}
    if isinstance(index, bool) or not isinstance(index, int) or index not in valid:
        raise ReviewError(400 if not isinstance(index, int) or isinstance(index, bool) else 404,
                          "invalid image index" if not isinstance(index, int) or isinstance(index, bool)
                          else "image not found")
    if decision not in {"yes", "love", "no", "comment", "reasons"}:
        raise ReviewError(400, "invalid decision")
    clean_comment: str | None = None
    if decision == "comment":
        if not isinstance(comment, str):
            raise ReviewError(400, "comment required")
        clean_comment = comment.strip()
        if not clean_comment:
            raise ReviewError(400, "comment required")
        if len(clean_comment) > 500:
            raise ReviewError(400, "comment too long")
    clean_reasons: list[str] = []
    if decision in {"no", "reasons"} and reasons is not None:
        clean_reasons = validate_reasons(reasons)

    with REVIEW_LOCK:
        result = load_review_result(paths, manifest["batch"], valid)
        if decision == "reasons":
            if index not in set(result["n"]):
                raise ReviewError(400, "reasons can only be set on a rejected item")
            if clean_reasons:
                result["r"][str(index)] = clean_reasons
            else:
                result["r"].pop(str(index), None)
        else:
            result["y"] = [value for value in result["y"] if value != index]
            result["l"] = [value for value in result["l"] if value != index]
            result["n"] = [value for value in result["n"] if value != index]
            result["c"].pop(str(index), None)
            # A repeated "no" without a reasons list keeps the chips already chosen;
            # any other verdict, or an explicit list, replaces them.
            if decision != "no" or reasons is not None:
                result["r"].pop(str(index), None)
            if decision == "yes":
                result["y"].append(index)
            elif decision == "love":
                result["l"].append(index)
            elif decision == "no":
                result["n"].append(index)
                if clean_reasons:
                    result["r"][str(index)] = clean_reasons
            else:
                result["c"][str(index)] = clean_comment
            result["y"].sort()
            result["l"].sort()
            result["n"].sort()
            result["c"] = {
                key: result["c"][key]
                for key in sorted(result["c"], key=int)
            }
        result["r"] = {
            key: result["r"][key]
            for key in sorted(result["r"], key=int)
        }
        atomic_write_review_result(paths, manifest["batch"], result)
    return result
