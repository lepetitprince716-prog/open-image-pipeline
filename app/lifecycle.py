"""Rollback-capable lifecycle management for Studio job metadata only.

Scope is deliberately narrow: direct ``app/data/results/*.json`` files that have no jobs row.
Generated images, references, Evidence and every other output tree are structurally unreachable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import jobs

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "app" / "data" / "results"
QUARANTINE_ROOT = ROOT / "app" / "data" / "quarantine" / "results"
MANIFEST_DIR = ROOT / "app" / "data" / "quarantine" / "manifests"
RETENTION_DAYS = 30
MAX_BYTES = 1 << 30
MAINTENANCE_INTERVAL_S = 6 * 60 * 60
_maintenance_lock = threading.Lock()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def runtime_paths(db_path: Path | None = None) -> tuple[Path, Path, Path, Path]:
    """Resolve lifecycle roots beside an isolated jobs DB, never back into production data."""
    db = Path(db_path if db_path is not None else jobs.DB_PATH)
    if db.resolve(strict=False) == jobs.DEFAULT_DB_PATH.resolve(strict=False):
        return db, RESULTS_DIR, QUARANTINE_ROOT, MANIFEST_DIR
    base = db.parent
    return db, base / "results", base / "quarantine" / "results", base / "quarantine" / "manifests"


def _atomic_json(path: Path, payload: dict) -> None:
    _private_dir(path.parent)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        path.chmod(0o600)
    finally:
        temp.unlink(missing_ok=True)


def _job_ids(db_path: Path) -> tuple[set[str], str]:
    if not db_path.is_file():
        return set(), "missing"
    conn = sqlite3.connect(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        ids = {str(row[0]) for row in conn.execute("SELECT id FROM jobs")}
        return ids, integrity
    finally:
        conn.close()


def find_orphans(*, db_path: Path | None = None, results_dir: Path | None = None) -> dict:
    db, default_results, _quarantine, _manifests = runtime_paths(db_path)
    results = Path(results_dir) if results_dir is not None else default_results
    job_ids, integrity = _job_ids(db)
    files = []
    if results.is_dir():
        for path in sorted(results.glob("*.json")):
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) or path.is_symlink():
                continue
            if path.stem not in job_ids:
                files.append(path)
    return {"integrity": integrity, "job_count": len(job_ids), "orphans": files,
            "orphan_bytes": sum(path.stat().st_size for path in files)}


def quarantine_orphans(*, db_path: Path | None = None, results_dir: Path | None = None,
                       quarantine_root: Path | None = None,
                       manifest_dir: Path | None = None,
                       now: float | None = None) -> dict:
    now = time.time() if now is None else now
    db, default_results, default_quarantine, default_manifests = runtime_paths(db_path)
    results = Path(results_dir) if results_dir is not None else default_results
    quarantine = Path(quarantine_root) if quarantine_root is not None else default_quarantine
    manifests = Path(manifest_dir) if manifest_dir is not None else default_manifests
    audit = find_orphans(db_path=db, results_dir=results)
    if audit["integrity"] != "ok":
        raise RuntimeError(f"jobs DB integrity is {audit['integrity']}; quarantine refused")
    paths: list[Path] = audit["orphans"]
    if not paths:
        return {"schema": "loramake.result-quarantine/v1", "status": "noop",
                "count": 0, "bytes": 0, "integrity": "ok"}
    batch_id = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    batch_dir = quarantine / batch_id
    _private_dir(quarantine)
    _private_dir(batch_dir)
    _private_dir(manifests)
    entries = []
    for source in paths:
        source = source.resolve(strict=True)
        if source.parent != results.resolve(strict=True):
            raise RuntimeError(f"orphan escaped direct results directory: {source}")
        stat_result = source.stat()
        target = batch_dir / source.name
        entries.append({"source": str(source), "quarantined": str(target),
                        "sha256": _sha256(source), "bytes": stat_result.st_size,
                        "mtime_ns": stat_result.st_mtime_ns, "moved": False})
    manifest = {
        "schema": "loramake.result-quarantine/v1",
        "batch_id": batch_id,
        "status": "planned",
        "created_at": _iso(now),
        "created_epoch": now,
        "expires_at": _iso(now + RETENTION_DAYS * 86400),
        "expires_epoch": now + RETENTION_DAYS * 86400,
        "integrity": audit["integrity"],
        "count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "scope": "app/data/results/*.json metadata only",
        "entries": entries,
    }
    manifest_path = manifests / f"{batch_id}.json"
    _atomic_json(manifest_path, manifest)  # hashes exist on disk before the first move
    for index, entry in enumerate(entries):
        source = Path(entry["source"])
        target = Path(entry["quarantined"])
        if target.exists():
            raise FileExistsError(target)
        os.replace(source, target)
        target.chmod(0o600)
        entry["moved"] = True
        entry["moved_at"] = _iso(time.time())
        # Persist every step: a crash leaves an exact, replayable partial manifest.
        manifest["moved_count"] = index + 1
        _atomic_json(manifest_path, manifest)
    manifest["status"] = "quarantined"
    manifest["completed_at"] = _iso(time.time())
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def restore_manifest(manifest_path: Path) -> dict:
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "loramake.result-quarantine/v1":
        raise ValueError("not a result quarantine manifest")
    restored = 0
    for entry in manifest.get("entries", []):
        source = Path(entry["source"])
        quarantined = Path(entry["quarantined"])
        if not quarantined.is_file():
            continue
        if _sha256(quarantined) != entry["sha256"]:
            raise RuntimeError(f"quarantine hash mismatch: {quarantined}")
        if source.exists():
            raise FileExistsError(f"restore target occupied: {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(quarantined, source)
        source.chmod(0o600)
        entry["restored_at"] = _iso(time.time())
        restored += 1
    manifest.update(status="restored", restored_count=restored, restored_at=_iso(time.time()))
    _atomic_json(path, manifest)
    return {"status": "restored", "restored": restored, "manifest": str(path)}


def purge_quarantine(*, manifest_dir: Path | None = None, now: float | None = None,
                     max_bytes: int = MAX_BYTES) -> dict:
    """Delete only manifest-bound quarantine files, oldest first when expired/over capacity."""
    now = time.time() if now is None else now
    manifests = Path(manifest_dir) if manifest_dir is not None else MANIFEST_DIR
    records = []
    for manifest_path in sorted(manifests.glob("*.json")) if manifests.is_dir() else []:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema") != "loramake.result-quarantine/v1":
            continue
        for entry in manifest.get("entries", []):
            path = Path(entry.get("quarantined", ""))
            if path.is_file() and _sha256(path) == entry.get("sha256"):
                records.append({"path": path, "entry": entry, "manifest": manifest,
                                "manifest_path": manifest_path,
                                "created": float(manifest.get("created_epoch") or path.stat().st_mtime),
                                "expires": float(manifest.get("expires_epoch") or 0),
                                "bytes": path.stat().st_size})
    records.sort(key=lambda record: (record["created"], str(record["path"])))
    total = sum(record["bytes"] for record in records)
    victims = [record for record in records if record["expires"] and now >= record["expires"]]
    victim_paths = {record["path"] for record in victims}
    remaining = total - sum(record["bytes"] for record in victims)
    for record in records:
        if remaining <= max_bytes:
            break
        if record["path"] in victim_paths:
            continue
        victims.append(record)
        victim_paths.add(record["path"])
        remaining -= record["bytes"]
    deleted = []
    touched: dict[Path, dict] = {}
    for record in victims:
        record["path"].unlink()
        record["entry"]["purged_at"] = _iso(now)
        deleted.append({"path": str(record["path"]), "sha256": record["entry"]["sha256"],
                        "bytes": record["bytes"]})
        touched[record["manifest_path"]] = record["manifest"]
        try:
            record["path"].parent.rmdir()
        except OSError:
            pass
    for path, manifest in touched.items():
        manifest["status"] = "purged" if all(
            not Path(entry.get("quarantined", "")).exists() for entry in manifest.get("entries", [])
        ) else "partially_purged"
        _atomic_json(path, manifest)
    audit = {"schema": "loramake.result-quarantine-purge/v1", "created_at": _iso(now),
             "deleted_count": len(deleted), "deleted_bytes": sum(item["bytes"] for item in deleted),
             "remaining_bytes": remaining, "max_bytes": max_bytes, "deleted": deleted}
    if deleted:
        audit_path = manifests / (
            "purge-" + datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        _atomic_json(audit_path, audit)
        audit["manifest"] = str(audit_path)
    return audit


def run_maintenance_once(*, db_path: Path | None = None, now: float | None = None,
                         max_bytes: int = MAX_BYTES) -> dict:
    """Quarantine new orphan metadata, then enforce retention/cap in one single-flight pass."""
    if not _maintenance_lock.acquire(blocking=False):
        return {"status": "already_running"}
    try:
        db, results, quarantine, manifests = runtime_paths(db_path)
        # Be independently safe on fresh/isolated runtimes even when a test or maintenance-only
        # process does not start the job worker first.
        conn = jobs._connect(db)
        conn.close()
        quarantined = quarantine_orphans(
            db_path=db,
            results_dir=results,
            quarantine_root=quarantine,
            manifest_dir=manifests,
            now=now,
        )
        purged = purge_quarantine(manifest_dir=manifests, now=now, max_bytes=max_bytes)
        return {
            "status": "ok",
            "quarantined_count": int(quarantined.get("count") or 0),
            "quarantined_bytes": int(quarantined.get("bytes") or 0),
            "deleted_count": int(purged.get("deleted_count") or 0),
            "deleted_bytes": int(purged.get("deleted_bytes") or 0),
            "remaining_bytes": int(purged.get("remaining_bytes") or 0),
        }
    finally:
        _maintenance_lock.release()


def maintenance_loop(stop: threading.Event, *, interval_s: float = MAINTENANCE_INTERVAL_S) -> None:
    """Run immediately and periodically; failures are observable but never stop the app."""
    interval = max(0.01, float(interval_s))
    while not stop.is_set():
        started = time.monotonic()
        try:
            result = run_maintenance_once()
            print(
                "[lifecycle] "
                f"status={result['status']} quarantined={result.get('quarantined_count', 0)} "
                f"deleted={result.get('deleted_count', 0)} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[lifecycle] maintenance failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if stop.wait(interval):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio result-metadata lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("quarantine")
    purge = sub.add_parser("purge")
    purge.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    restore = sub.add_parser("restore")
    restore.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit = find_orphans()
        out = {key: value for key, value in audit.items() if key != "orphans"}
        out["orphan_count"] = len(audit["orphans"])
    elif args.command == "quarantine":
        result = quarantine_orphans()
        out = {key: value for key, value in result.items() if key != "entries"}
    elif args.command == "purge":
        out = purge_quarantine(max_bytes=args.max_bytes)
    else:
        out = restore_manifest(args.manifest)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
