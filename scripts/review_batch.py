#!/usr/bin/env python3
"""Register a small, local-only media batch for the gallery at 127.0.0.1:8771.

Two modes share one registration format:

- ``review`` — the caller needs a per-item verdict. The batch opens at ``/review?batch=ID``
  (one item at a time, love/yes/no/comment) and decisions land in ``result.json``, which
  ``review_result.py`` reads back.
- ``view`` — the caller only needs the user to look. The batch opens at ``/view?batch=ID``
  (grid + lightbox) and no decision is ever written.

Items may be images or videos, may live inside or outside the repository, and may carry a
short caption explaining what the viewer is looking at. A path written as ``z3:...`` or
``zzz:...`` is fetched to a local cache first (see ``remote_media.py``), so output that
only exists on a Windows box can be shown without copying it by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

from config import OUTPUTS, ROOT


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MODES = ("review", "view")
BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
REVIEW_ROOT_PARTS = ("_gallery", "review_batches")
BUFFER_SIZE = 1024 * 1024
CAPTION_MAX = 200
# Items outside the repository are snapshotted into the batch so a later edit cannot
# change what the user reviewed. That copy is bounded: a multi-GB video belongs in the
# main gallery tree, served by /img/, not duplicated per batch.
SNAPSHOT_MAX_BYTES = 200 * 1024 * 1024


class ReviewBatchError(ValueError):
    """A user-facing registration error."""


def _generated_batch_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"review-{stamp}-{secrets.token_hex(4)}"


def _validate_batch_id(batch: str) -> str:
    if not BATCH_RE.fullmatch(batch):
        raise ReviewBatchError(
            "batch ID must be 1-128 ASCII characters, start with a letter or "
            "digit, and contain only letters, digits, '.', '_' or '-'"
        )
    return batch


def _clean_caption(raw: object, label: str) -> str:
    """Normalise one optional caption. Empty and absent are the same thing."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ReviewBatchError(f"caption must be a string: {label}")
    caption = raw.strip()
    if len(caption) > CAPTION_MAX:
        raise ReviewBatchError(
            f"caption longer than {CAPTION_MAX} characters: {label}"
        )
    return caption


def _source_record(raw: str, repo: Path, caption: str = "") -> dict[str, object]:
    source = Path(os.path.abspath(os.path.expanduser(raw)))
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ReviewBatchError(f"unsupported media extension: {raw}")
    try:
        actual = source.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ReviewBatchError(f"media does not exist: {raw}") from None
    if not actual.is_file():
        raise ReviewBatchError(f"media is not a file: {raw}")

    try:
        repo_relative = actual.relative_to(repo).as_posix()
    except ValueError:
        repo_relative = None
    # Only out-of-repo items get snapshotted, so only they are bounded by the cap.
    if repo_relative is None and actual.stat().st_size > SNAPSHOT_MAX_BYTES:
        limit_mb = SNAPSHOT_MAX_BYTES // (1024 * 1024)
        raise ReviewBatchError(
            f"media outside the repository is larger than {limit_mb}MB and would be "
            f"copied into the batch: {raw}. Move it under a gallery root and register "
            f"its repository-relative path instead."
        )
    return {
        "source": source,
        "actual": actual,
        "name": source.name,
        "repo_relative": repo_relative,
        "suffix": suffix,
        "media": "video" if suffix in VIDEO_EXTENSIONS else "image",
        "caption": caption,
    }


def _spec_entries(spec_path: str) -> list[tuple[str, str]]:
    """Read a JSON spec into (path, caption) pairs.

    The spec form exists so a caller registering many captioned items writes one file
    instead of an unreadable argv. Accepted shapes per entry: a bare path string, or an
    object with ``path`` and an optional ``caption``.
    """
    try:
        raw = Path(spec_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewBatchError(f"cannot read spec: {spec_path} ({exc.strerror})") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewBatchError(f"spec is not valid JSON: {spec_path} ({exc})") from None
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not data:
        raise ReviewBatchError(
            "spec must be a non-empty JSON array (or an object with a non-empty "
            '"items" array)'
        )

    entries: list[tuple[str, str]] = []
    for index, entry in enumerate(data):
        label = f"spec item {index}"
        if isinstance(entry, str):
            entries.append((entry, ""))
            continue
        if not isinstance(entry, dict):
            raise ReviewBatchError(f"{label} must be a string or an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ReviewBatchError(f"{label} is missing a non-empty \"path\"")
        entries.append((path, _clean_caption(entry.get("caption"), label)))
    return entries


def _localise(paths: Sequence[str], alias: str | None = None) -> list[str]:
    """Turn any ``z3:``/``zzz:`` references into local cached files, leaving the rest be.

    ``remote_media`` is imported inside the function, so merely importing this module
    never drags in the transport. (The import does still run for a purely local batch —
    only the network calls are skipped.)

    ``alias`` pins one SSH alias instead of letting the transport try each in turn. That
    matters when one leg of a box is down: the default order starts with Tailscale, so
    every reference pays a failed connection before falling through to the tunnel.
    """
    import remote_media

    if not any(remote_media.is_remote_ref(path) for path in paths):
        return list(paths)
    try:
        records = remote_media.resolve_all(paths, alias=alias)
    except remote_media.RemoteMediaError as exc:
        raise ReviewBatchError(str(exc)) from None
    return [str(record["path"]) for record in records]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        while chunk := src.read(BUFFER_SIZE):
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    shutil.copystat(source, destination, follow_symlinks=True)
    return digest.hexdigest()


def _manifest_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_json(path: Path, payload: object) -> None:
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _gallery_port() -> str:
    return os.environ.get("LORAMAKE_GALLERY_PORT", "8771").strip() or "8771"


def _batch_url(batch: str, mode: str) -> str:
    page = "view" if mode == "view" else "review"
    return f"http://127.0.0.1:{_gallery_port()}/{page}?batch={quote(batch, safe='')}"


def serving_tree_warning(port: str | None = None) -> str | None:
    """Warn when the gallery on that port serves a different tree than we wrote into.

    A batch registered here is only reachable from a server rooted at the same tree.
    Nothing used to check that, so a URL printed from one checkout against a gallery
    started from another exits 0 and 404s in the browser — which is exactly what
    happened on 2026-08-13. Best-effort: an unreachable or old server yields no warning
    rather than a false alarm.
    """
    port = port or _gallery_port()
    try:
        import gallery_app  # noqa: PLC0415 (optional, and it binds paths at import)
    except Exception:  # pragma: no cover - gallery_app is a sibling, but stay soft
        return None
    ours_root, ours_outputs = str(ROOT), str(OUTPUTS)
    fix = f"  fix: python3 {ROOT}/scripts/gallery_app.py restart"
    info = gallery_app.probe_health(port)
    if info is None:
        # Nothing listening is not worth a warning, but a gallery too old to answer
        # /api/health is: that is precisely the server that shadowed this tree on
        # 2026-08-13, and "cannot verify" must not read as "verified fine".
        if not gallery_app.is_legacy_gallery(port):
            return None
        return (
            f"warning: the gallery on 127.0.0.1:{port} predates /api/health, so it "
            f"cannot say which tree it serves.\n"
            f"  batch is in: root={ours_root} outputs={ours_outputs}\n" + fix
        )
    if info.get("root") == ours_root and info.get("outputs") == ours_outputs:
        return None
    return (
        f"warning: the gallery on 127.0.0.1:{port} serves a different tree, so this "
        f"batch will 404 there.\n"
        f"  it serves : root={info.get('root')} outputs={info.get('outputs')}\n"
        f"  batch is in: root={ours_root} outputs={ours_outputs}\n" + fix
    )


def register_batch(
    images: Sequence[str],
    batch: str | None = None,
    *,
    captions: Sequence[str] | None = None,
    mode: str = "review",
    alias: str | None = None,
) -> dict[str, object]:
    """Create one media batch and return its compact registration record."""
    if not images:
        raise ReviewBatchError("at least one media file is required")
    if mode not in MODES:
        raise ReviewBatchError(f"mode must be one of {', '.join(MODES)}")
    if captions is not None and len(captions) != len(images):
        raise ReviewBatchError("captions must line up one-to-one with media paths")

    batch_id = _validate_batch_id(batch if batch is not None else _generated_batch_id())
    repo = ROOT.resolve()
    review_root = (OUTPUTS / REVIEW_ROOT_PARTS[0] / REVIEW_ROOT_PARTS[1]).resolve()
    batch_dir = review_root / batch_id
    if batch_dir.exists():
        raise ReviewBatchError(f"review batch already exists: {batch_id}")

    texts = list(captions) if captions is not None else [""] * len(images)
    sources = [
        _source_record(raw, repo, _clean_caption(text, raw))
        for raw, text in zip(_localise(images, alias), texts)
    ]
    review_root.mkdir(parents=True, exist_ok=True)
    try:
        batch_dir.mkdir()
    except FileExistsError:
        raise ReviewBatchError(f"review batch already exists: {batch_id}") from None

    items: list[dict[str, object]] = []
    try:
        for index, record in enumerate(sources):
            actual = record["actual"]
            assert isinstance(actual, Path)
            repo_relative = record["repo_relative"]
            if repo_relative is not None:
                digest = _hash_file(actual)
                item_kind = "repo"
                item_path = repo_relative
            else:
                assets = batch_dir / "assets"
                assets.mkdir(exist_ok=True)
                suffix = record["suffix"]
                assert isinstance(suffix, str)
                fd, raw_tmp = tempfile.mkstemp(
                    prefix=f".{index:04d}.", suffix=f"{suffix}.tmp", dir=assets
                )
                os.close(fd)
                tmp = Path(raw_tmp)
                tmp.unlink()
                try:
                    digest = _copy_and_hash(actual, tmp)
                    snapshot = assets / f"{index:04d}-{digest[:12]}{suffix}"
                    os.replace(tmp, snapshot)
                finally:
                    tmp.unlink(missing_ok=True)
                item_kind = "batch_asset"
                item_path = snapshot.relative_to(batch_dir).as_posix()

            source = record["source"]
            assert isinstance(source, Path)
            name = record["name"]
            assert isinstance(name, str)
            items.append(
                {
                    "i": index,
                    "kind": item_kind,
                    "path": item_path,
                    "name": name,
                    "media": record["media"],
                    "caption": record["caption"],
                    "sha256": digest,
                    "source_path": source.as_posix(),
                }
            )

        manifest = {
            "v": 2,
            "batch": batch_id,
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": items,
        }
        _atomic_json(batch_dir / "manifest.json", manifest)
        # A view batch never collects decisions, so it gets no result file at all —
        # the page has nothing to write and review_result.py reports it as such.
        if mode == "review":
            _atomic_json(
                batch_dir / "result.json",
                {"v": 1, "batch": batch_id, "y": [], "l": [], "n": [], "c": {}},
            )
    except BaseException:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise

    return {
        "v": 2,
        "batch": batch_id,
        "mode": mode,
        "count": len(items),
        "url": _batch_url(batch_id, mode),
        "manifest": _manifest_path(batch_dir / "manifest.json", repo),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register images or videos for the local gallery at 127.0.0.1:8771"
    )
    parser.add_argument("--batch", help="explicit batch ID (must not already exist)")
    parser.add_argument(
        "--mode", choices=MODES, default="review",
        help="review: per-item yes/no/comment (default). view: look only, no decisions.",
    )
    parser.add_argument(
        "--spec", metavar="FILE.json",
        help='JSON array of "path" strings or {"path": ..., "caption": ...} objects',
    )
    parser.add_argument("--json", action="store_true", help="print one-line registration JSON")
    parser.add_argument(
        "--alias",
        help="force one SSH alias for z3:/zzz: references (e.g. z3-cf) instead of trying each",
    )
    parser.add_argument("media", nargs="*", metavar="MEDIA")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.spec) == bool(args.media):
        parser.exit(2, "review_batch.py: error: pass either --spec FILE.json or MEDIA paths\n")

    try:
        if args.spec:
            entries = _spec_entries(args.spec)
            paths = [path for path, _ in entries]
            captions = [caption for _, caption in entries]
        else:
            paths, captions = list(args.media), None
        registration = register_batch(
            paths, args.batch, captions=captions, mode=args.mode, alias=args.alias
        )
    except ReviewBatchError as exc:
        parser.exit(2, f"review_batch.py: error: {exc}\n")

    # The batch itself registered fine; it is the server that may be wrong. Warn on
    # stderr so --json consumers keep a clean stdout, and keep the exit code at 0.
    warning = serving_tree_warning()
    if warning:
        sys.stderr.write(warning + "\n")

    if args.json:
        print(json.dumps(registration, ensure_ascii=False, separators=(",", ":")))
    else:
        print(registration["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
