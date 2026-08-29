"""Frontend-neutral experiment catalog and append-only review ledger.

Source manifests and generated media remain the evidence of record.  This module
normalizes supported manifest schemas into a stable API projection, stores only
the projection/index and human review events in SQLite, and resolves media through
opaque asset IDs.  It never submits generation work or deletes evidence.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = (ROOT / "outputs" / "_analysis").resolve()
OUTPUTS_DIR = (ROOT / "outputs").resolve()
NSFW_OUTPUTS = (ROOT / "nsfw" / "outputs").resolve()
DB_PATH = ROOT / "app" / "data" / "experiments.sqlite3"
THUMB_CACHE = ROOT / "app" / "data" / "experiment_thumb_cache"
NSFW_THUMB_CACHE = ROOT / "nsfw" / "app_cache" / "experiment_thumbs"

SCAN_TTL_S = 5.0
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SUPPORT_BYTES = 16 * 1024 * 1024
MAX_ASSET_BYTES = 512 * 1024 * 1024
THUMB_WIDTHS = (256, 480, 768)
IMMUTABLE_CACHE = {"Cache-Control": "public, max-age=86400, immutable"}

_EXPERIMENT_ID_RE = re.compile(r"exp_[a-z0-9][a-z0-9_-]{2,63}\Z")
_CLIENT_ID_RE = re.compile(r"(?:web|hermes|claude|codex|other:[a-z0-9][a-z0-9_-]{0,47})\Z")
_ACTOR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_EVENT_ID_RE = re.compile(r"rev_[a-z0-9][a-z0-9_-]{2,63}\Z")
_ENTITY_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,63}\Z")
_SAMPLE_ID_RE = re.compile(r"smp_[a-z0-9][a-z0-9_-]{2,63}\Z")
_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
_REPORT_STATUSES = {"pass", "hold", "failed", "info"}
_REPORT_METRIC_STATUSES = {"pass", "fail", "info"}
# "required" keeps the historical contract: a render that is not on disk invalidates
# the manifest.  "local_only" is for evidence whose renders deliberately stay off the
# distributed tree (see .gitignore for outputs/_analysis/**/*.png); the recorded
# SHA-256 and byte count are the record, so the projection stays indexable and the
# missing render is reported instead of silently dropping the whole experiment.
_ASSET_AVAILABILITY_MODES = {"required", "local_only"}
_KNOWN_SCHEMAS = {
    "harem-complex-prompt-benchmark-manifest-v1",
    "loramake.harem_ab.v1",
    "loramake.experiment-manifest.v1",
}
_PUBLIC_STATUSES = {
    "running",
    "complete",
    "failed",
    "preserved_waiting_user_confirmation",
    "archived",
}
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  revision  INTEGER NOT NULL
);
INSERT OR IGNORE INTO catalog_state(singleton, revision) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS experiment_sources (
  experiment_id    TEXT PRIMARY KEY,
  zone             TEXT NOT NULL CHECK (zone IN ('sfw', 'nsfw')),
  source_rel       TEXT NOT NULL UNIQUE,
  source_schema    TEXT,
  source_revision  TEXT,
  source_mtime_ns  INTEGER NOT NULL,
  source_size      INTEGER NOT NULL,
  updated_at       TEXT NOT NULL,
  payload_json     TEXT,
  index_stale      INTEGER NOT NULL DEFAULT 0,
  error_code       TEXT,
  error_detail     TEXT
);

CREATE TABLE IF NOT EXISTS experiment_assets (
  experiment_id TEXT NOT NULL,
  asset_id      TEXT NOT NULL,
  zone          TEXT NOT NULL CHECK (zone IN ('sfw', 'nsfw')),
  path_rel      TEXT NOT NULL,
  media_type    TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  PRIMARY KEY (experiment_id, asset_id)
);

CREATE TABLE IF NOT EXISTS review_events (
  event_id             TEXT PRIMARY KEY,
  experiment_id        TEXT NOT NULL,
  zone                 TEXT NOT NULL CHECK (zone IN ('sfw', 'nsfw')),
  source_revision      TEXT NOT NULL,
  target_type          TEXT NOT NULL,
  target_id            TEXT NOT NULL,
  decision             TEXT NOT NULL CHECK (decision IN ('yes', 'no', 'comment')),
  comment              TEXT,
  actor_kind           TEXT NOT NULL CHECK (actor_kind IN ('human', 'agent')),
  actor_id             TEXT NOT NULL,
  client_id            TEXT NOT NULL,
  supersedes_event_id  TEXT,
  idempotency_key      TEXT NOT NULL,
  request_sha256       TEXT NOT NULL,
  created_at           TEXT NOT NULL,
  UNIQUE (client_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS catalog_events (
  revision        INTEGER PRIMARY KEY,
  zone            TEXT NOT NULL CHECK (zone IN ('sfw', 'nsfw')),
  event_type      TEXT NOT NULL,
  experiment_id  TEXT NOT NULL,
  review_event_id TEXT,
  created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_zone_updated
  ON experiment_sources(zone, updated_at DESC, experiment_id);
CREATE INDEX IF NOT EXISTS idx_reviews_experiment
  ON review_events(zone, experiment_id, created_at, event_id);
CREATE INDEX IF NOT EXISTS idx_catalog_events_zone
  ON catalog_events(zone, revision);
"""

_scan_lock = threading.RLock()
_last_scan: dict[tuple[str, str], float] = {}
_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


class ExperimentError(Exception):
    """Stable error contract used by the HTTP router."""

    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    try:
        os.chmod(path, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.exists():
                os.chmod(sidecar, 0o600)
    except OSError:
        pass
    return conn


def _catalog_revision(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT revision FROM catalog_state WHERE singleton = 1").fetchone()
    return int(row["revision"])


def _next_revision(
    conn: sqlite3.Connection,
    *,
    zone: str,
    event_type: str,
    experiment_id: str,
    review_event_id: str | None = None,
) -> dict[str, Any]:
    revision = _catalog_revision(conn) + 1
    created_at = _utc_now()
    conn.execute("UPDATE catalog_state SET revision = ? WHERE singleton = 1", (revision,))
    conn.execute(
        """INSERT INTO catalog_events
           (revision, zone, event_type, experiment_id, review_event_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (revision, zone, event_type, experiment_id, review_event_id, created_at),
    )
    return {
        "schema": "loramake.api.catalog-event.v1",
        "revision": revision,
        "zone": zone,
        "type": event_type,
        "experiment_id": experiment_id,
        "review_event_id": review_event_id,
        "created_at": created_at,
    }


def _publish(event: dict[str, Any]) -> None:
    with _subscribers_lock:
        for subscriber in list(_subscribers):
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass


def subscribe() -> queue.Queue:
    subscriber: queue.Queue = queue.Queue(maxsize=256)
    with _subscribers_lock:
        _subscribers.append(subscriber)
    return subscriber


def unsubscribe(subscriber: queue.Queue) -> None:
    with _subscribers_lock:
        if subscriber in _subscribers:
            _subscribers.remove(subscriber)


def _opaque_cursor(parts: list[Any]) -> str:
    raw = _json_bytes(parts)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None, expected: int) -> list[Any] | None:
    if not cursor:
        return None
    if len(cursor) > 512:
        raise ExperimentError(400, "invalid_cursor", "cursor is too long")
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise ExperimentError(400, "invalid_cursor", "cursor is malformed") from exc
    if not isinstance(value, list) or len(value) != expected:
        raise ExperimentError(400, "invalid_cursor", "cursor has the wrong shape")
    return value


def _slug(value: str, *, limit: int = 56) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (text or "unknown")[:limit].rstrip("_")


def _fallback_experiment_id(source_rel: str) -> str:
    return f"exp_{hashlib.sha256(source_rel.encode()).hexdigest()[:20]}"


def _legacy_experiment_id(source: Path, schema: str) -> str:
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source.parent.name)
    date = "".join(match.groups()) if match else hashlib.sha256(source.parent.name.encode()).hexdigest()[:8]
    prefix = "harem_complex" if schema.startswith("harem-complex") else "harem_ab"
    return f"exp_{prefix}_{date}"


def _model_basename(value: str) -> str:
    return PureWindowsPath(str(value)).name


def _recipe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("recipe must be an object")
    required = ("steps", "cfg", "sampler", "scheduler", "shift")
    if any(key not in value for key in required):
        raise ValueError("recipe is missing required fields")
    recipe = {
        "steps": int(value["steps"]),
        "cfg": float(value["cfg"]),
        "sampler": str(value["sampler"]),
        "scheduler": str(value["scheduler"]),
        "shift": float(value["shift"]),
    }
    if recipe["steps"] < 1 or recipe["cfg"] < 0:
        raise ValueError("recipe numeric fields are invalid")
    if not recipe["sampler"] or not recipe["scheduler"]:
        raise ValueError("recipe sampler fields are invalid")
    return recipe


def _load_support_json(source: Path, descriptor: Any) -> dict[str, Any]:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError("support-file descriptor is invalid")
    raw_path = Path(descriptor["path"])
    if raw_path.is_absolute():
        raise ValueError("absolute support-file path is forbidden")
    candidates = (ROOT / raw_path, source.parent / raw_path)
    target: Path | None = None
    source_root = source.parent.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(source_root) and resolved.is_file() and not resolved.is_symlink():
            target = resolved
            break
    if target is None:
        raise ValueError("support file is missing or outside the experiment directory")
    if target.stat().st_size > MAX_SUPPORT_BYTES:
        raise ValueError("support file exceeds the size limit")
    data = target.read_bytes()
    expected = descriptor.get("sha256")
    if expected and _sha256(data) != expected:
        raise ValueError("support-file sha256 mismatch")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("support file must contain a JSON object")
    return parsed


def _allowed_asset(path: Path, zone: str) -> bool:
    if zone == "nsfw":
        return path.is_relative_to(NSFW_OUTPUTS)
    if not path.is_relative_to(OUTPUTS_DIR):
        return False
    return not any("nsfw" in part.lower() for part in path.parts)


def _register_asset(
    *,
    experiment_id: str,
    zone: str,
    raw_path: str,
    expected_sha256: str,
    expected_bytes: int,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Project one declared asset, returning (public descriptor, delivery row).

    The delivery row is ``None`` when ``allow_missing`` is set and the render is not
    on disk: such an asset never enters ``experiment_assets``, so delivery keeps
    failing closed while the rest of the experiment stays indexable.
    """
    source_path = Path(raw_path)
    if source_path.is_absolute():
        raise ValueError("absolute asset path is forbidden")
    unresolved = ROOT / source_path
    if unresolved.is_symlink():
        raise ValueError("symlinked asset is forbidden")
    target = unresolved.resolve()
    if not _allowed_asset(target, zone):
        raise ValueError("asset is outside the allowed zone")
    media_type = _MEDIA_TYPES.get(target.suffix.lower())
    if media_type is None:
        raise ValueError("asset media type is unsupported")
    if not _SHA256_RE.fullmatch(str(expected_sha256)):
        raise ValueError("asset sha256 is invalid")
    if not target.is_file():
        if not allow_missing:
            raise ValueError("asset is missing")
        declared_bytes = int(expected_bytes)
        if declared_bytes < 1 or declared_bytes > MAX_ASSET_BYTES:
            raise ValueError("declared asset byte count is invalid")
        return {
            "asset_id": f"ast_{expected_sha256[:20]}",
            "media_type": media_type,
            "bytes": declared_bytes,
            "sha256": expected_sha256,
            "variants": [],
            "availability": "missing",
        }, None
    actual_size = target.stat().st_size
    if actual_size > MAX_ASSET_BYTES:
        raise ValueError("asset exceeds the size limit")
    if actual_size != int(expected_bytes):
        raise ValueError("asset byte count mismatch")
    data = target.read_bytes()
    if len(data) != int(expected_bytes):
        raise ValueError("asset byte count mismatch")
    if _sha256(data) != expected_sha256:
        raise ValueError("asset sha256 mismatch")
    asset_id = f"ast_{expected_sha256[:20]}"
    public = {
        "asset_id": asset_id,
        "media_type": media_type,
        "bytes": len(data),
        "sha256": expected_sha256,
        "variants": ["original", "thumb_256", "thumb_480", "thumb_768"],
        "availability": "available",
    }
    # Dimensions improve layout stability for clients, but image decoding is
    # deliberately advisory: a hash-valid image must remain indexable even if
    # Pillow cannot inspect it.  Asset delivery/thumbnail generation continues
    # to fail closed later for invalid bytes.
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        if width > 0 and height > 0:
            public["width"] = width
            public["height"] = height
    except (OSError, SyntaxError, ValueError):
        pass
    private = {
        "experiment_id": experiment_id,
        "asset_id": asset_id,
        "zone": zone,
        "path_rel": target.relative_to(ROOT.resolve()).as_posix(),
        "media_type": media_type,
        "bytes": len(data),
        "sha256": expected_sha256,
    }
    return public, private


def _register_report_source(source: Path, descriptor: Any) -> dict[str, Any]:
    """Verify a JSON report without exposing its path or unredacted contents."""
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError("canonical report source is invalid")
    raw_path = Path(descriptor["path"])
    if raw_path.is_absolute():
        raise ValueError("absolute report path is forbidden")
    unresolved = ROOT / raw_path
    if unresolved.is_symlink():
        raise ValueError("symlinked report is forbidden")
    target = unresolved.resolve()
    if not target.is_relative_to(source.parent.resolve()) or not target.is_file():
        raise ValueError("report is missing or outside the experiment directory")
    if target.suffix.lower() != ".json":
        raise ValueError("canonical report must be JSON")
    data = target.read_bytes()
    if len(data) > MAX_SUPPORT_BYTES:
        raise ValueError("report exceeds the size limit")
    expected_sha256 = str(descriptor.get("sha256", ""))
    expected_bytes = int(descriptor.get("bytes", 0))
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("report sha256 is invalid")
    if len(data) != expected_bytes:
        raise ValueError("report byte count mismatch")
    if _sha256(data) != expected_sha256:
        raise ValueError("report sha256 mismatch")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("report must contain a JSON object")
    return {
        "media_type": "application/json",
        "bytes": len(data),
        "sha256": expected_sha256,
    }


def _normalize_reports(
    source: Path,
    raw_reports: Any,
    *,
    experiment_id: str,
    zone: str,
    allow_missing_assets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if raw_reports is None:
        return [], []
    if not isinstance(raw_reports, list):
        raise ValueError("canonical reports must be an array")
    reports: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    report_ids: set[str] = set()
    for raw_report in raw_reports:
        if not isinstance(raw_report, dict):
            raise ValueError("canonical report is invalid")
        report_id = str(raw_report.get("report_id", ""))
        label = str(raw_report.get("label", ""))
        status = str(raw_report.get("status", ""))
        if (
            not _ENTITY_ID_RE.fullmatch(report_id)
            or report_id in report_ids
            or not label
            or len(label) > 120
            or status not in _REPORT_STATUSES
        ):
            raise ValueError("canonical report display fields are invalid")
        report_ids.add(report_id)
        source_binding = _register_report_source(source, raw_report.get("source"))

        raw_metrics = raw_report.get("metrics", [])
        if not isinstance(raw_metrics, list):
            raise ValueError("canonical report metrics must be an array")
        metrics: list[dict[str, Any]] = []
        metric_ids: set[str] = set()
        for raw_metric in raw_metrics:
            if not isinstance(raw_metric, dict):
                raise ValueError("canonical report metric is invalid")
            metric_id = str(raw_metric.get("metric_id", ""))
            metric_label = str(raw_metric.get("label", ""))
            display_value = str(raw_metric.get("display_value", ""))
            metric_status = str(raw_metric.get("status", "info"))
            if (
                not _ENTITY_ID_RE.fullmatch(metric_id)
                or metric_id in metric_ids
                or not metric_label
                or len(metric_label) > 120
                or not display_value
                or len(display_value) > 120
                or metric_status not in _REPORT_METRIC_STATUSES
            ):
                raise ValueError("canonical report metric fields are invalid")
            metric_ids.add(metric_id)
            metrics.append({
                "metric_id": metric_id,
                "label": metric_label,
                "display_value": display_value,
                "status": metric_status,
            })

        raw_assets = raw_report.get("assets", [])
        if not isinstance(raw_assets, list):
            raise ValueError("canonical report assets must be an array")
        report_assets: list[dict[str, Any]] = []
        roles: set[str] = set()
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, dict) or not isinstance(raw_asset.get("asset"), dict):
                raise ValueError("canonical report asset is invalid")
            role = str(raw_asset.get("role", ""))
            asset_label = str(raw_asset.get("label", ""))
            if (
                not _ENTITY_ID_RE.fullmatch(role)
                or role in roles
                or not asset_label
                or len(asset_label) > 120
            ):
                raise ValueError("canonical report asset fields are invalid")
            roles.add(role)
            descriptor = raw_asset["asset"]
            public, private = _register_asset(
                experiment_id=experiment_id,
                zone=zone,
                raw_path=str(descriptor.get("path", "")),
                expected_sha256=str(descriptor.get("sha256", "")),
                expected_bytes=int(descriptor.get("bytes", 0)),
                allow_missing=allow_missing_assets,
            )
            if private is not None:
                assets.append(private)
            report_assets.append({"role": role, "label": asset_label, "asset": public})

        notes = raw_report.get("notes", [])
        if (
            not isinstance(notes, list)
            or any(not isinstance(note, str) or not note or len(note) > 1000 for note in notes)
        ):
            raise ValueError("canonical report notes are invalid")
        reports.append({
            "report_id": report_id,
            "label": label,
            "status": status,
            "source": source_binding,
            "metrics": metrics,
            "assets": report_assets,
            "notes": notes,
        })
    return reports, assets


def _empty_human_review() -> dict[str, Any]:
    return {"event_count": 0, "latest_event_id": None, "current_by_target": {}}


def _policy_text(value: Any) -> str:
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict) and isinstance(value.get("note"), str):
        return "Each criterion is binary; ambiguous or partially satisfied is 0."
    return "Binary prompt-adherence criteria; ambiguous or partial satisfaction scores 0."


def _score(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("sample score must be an object or null")
    hits = int(value.get("hits"))
    total = int(value.get("total"))
    vector = value.get("vector")
    if (
        total < 1
        or hits < 0
        or hits > total
        or not isinstance(vector, list)
        or len(vector) != total
        or any(item not in (0, 1) for item in vector)
        or sum(vector) != hits
    ):
        raise ValueError("sample score is inconsistent")
    return {"hits": hits, "total": total, "vector": vector}


def _summary_by_arm(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("summary_by_arm must be an object")
    normalized: dict[str, Any] = {}
    for arm_id, summary in value.items():
        if not _ENTITY_ID_RE.fullmatch(str(arm_id)) or not isinstance(summary, dict):
            raise ValueError("automatic evaluation arm summary is invalid")
        raw_dimensions = summary.get("by_dimension", summary.get("per_dimension", {}))
        if not isinstance(raw_dimensions, dict):
            raise ValueError("automatic evaluation dimension summary is invalid")
        dimensions: dict[str, Any] = {}
        for dimension_id, item in raw_dimensions.items():
            if not _ENTITY_ID_RE.fullmatch(str(dimension_id)) or not isinstance(item, dict):
                raise ValueError("automatic evaluation dimension summary is invalid")
            dimension_summary = {
                "hits": int(item["hits"]),
                "total": int(item["total"]),
                "accuracy_pct": float(item["accuracy_pct"]),
            }
            if (
                dimension_summary["total"] < 1
                or not 0 <= dimension_summary["hits"] <= dimension_summary["total"]
                or not 0 <= dimension_summary["accuracy_pct"] <= 100
            ):
                raise ValueError("automatic evaluation dimension values are invalid")
            dimensions[str(dimension_id)] = dimension_summary
        arm_summary = {
            "hits": int(summary["hits"]),
            "total": int(summary["total"]),
            "accuracy_pct": float(summary["accuracy_pct"]),
            "median_warm_s": float(summary.get("median_warm_s", 0)),
            "by_dimension": dimensions,
        }
        if (
            arm_summary["total"] < 1
            or not 0 <= arm_summary["hits"] <= arm_summary["total"]
            or not 0 <= arm_summary["accuracy_pct"] <= 100
            or arm_summary["median_warm_s"] < 0
        ):
            raise ValueError("automatic evaluation arm values are invalid")
        normalized[str(arm_id)] = arm_summary
    return normalized


def _normalize_complex(
    source: Path,
    manifest: dict[str, Any],
    source_revision: str,
    updated_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_id = _legacy_experiment_id(source, str(manifest["schema"]))
    spec = _load_support_json(source, manifest.get("spec"))
    scoring = _load_support_json(source, manifest.get("scoring"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 24:
        raise ValueError("complex benchmark must contain 24 outputs")

    arm_labels = {
        "native_8step": "8-Step v1.0 · native",
        "control_8step": "8-Step v1.0 · strict shared recipe",
        "control_v15": "v1.5 · strict shared recipe",
    }
    arms: list[dict[str, Any]] = []
    for arm_id in ("native_8step", "control_8step", "control_v15"):
        matches = [item for item in outputs if item.get("arm") == arm_id]
        if not matches:
            raise ValueError(f"missing arm {arm_id}")
        arms.append({
            "arm_id": arm_id,
            "label": arm_labels[arm_id],
            "model": _model_basename(str(matches[0].get("model", ""))),
            "recipe": _recipe(matches[0].get("recipe")),
        })

    dimension_labels = {
        "pose": "姿势",
        "scene": "场景",
        "clothing": "服饰",
        "composition": "构图",
    }
    dimensions: list[dict[str, Any]] = []
    raw_dimensions = spec.get("dimensions")
    if not isinstance(raw_dimensions, list) or len(raw_dimensions) != 4:
        raise ValueError("complex benchmark must contain four dimensions")
    for dimension in raw_dimensions:
        dimension_id = str(dimension.get("id", ""))
        criteria = dimension.get("criteria")
        if dimension_id not in dimension_labels or not isinstance(criteria, list) or len(criteria) != 8:
            raise ValueError("dimension contract is invalid")
        dimensions.append({
            "dimension_id": dimension_id,
            "label": dimension_labels[dimension_id],
            "prompt": str(dimension.get("prompt", "")),
            "criteria": [str(value) for value in criteria],
        })

    assets: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for output in outputs:
        arm_id = str(output.get("arm", ""))
        dimension_id = str(output.get("dimension", ""))
        seed = int(output.get("seed"))
        if arm_id not in arm_labels or dimension_id not in dimension_labels:
            raise ValueError("sample references an unknown arm or dimension")
        vector = output.get("score_vector")
        if not isinstance(vector, list) or len(vector) != 8 or any(value not in (0, 1) for value in vector):
            raise ValueError("score vector is invalid")
        hits = int(output.get("criteria_hit"))
        if hits != sum(vector) or int(output.get("criteria_total")) != 8:
            raise ValueError("score aggregate does not match the vector")
        asset, private = _register_asset(
            experiment_id=experiment_id,
            zone="sfw",
            raw_path=str(output.get("local_path", "")),
            expected_sha256=str(output.get("sha256", "")),
            expected_bytes=int(output.get("bytes", 0)),
        )
        assets.append(private)
        samples.append({
            "sample_id": f"smp_{arm_id}_{dimension_id}_{seed}",
            "arm_id": arm_id,
            "dimension_id": dimension_id,
            "seed": seed,
            "asset": asset,
            "score": {"hits": hits, "total": 8, "vector": vector},
            "elapsed_s": float(output.get("elapsed_s", 0)),
        })

    raw_summary = manifest.get("scoring", {}).get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError("scoring summary is missing")
    summary_by_arm = _summary_by_arm(raw_summary)
    if set(summary_by_arm) != set(arm_labels):
        raise ValueError("scoring summary arms do not match experiment arms")

    status = str(manifest.get("status", "complete"))
    if status not in _PUBLIC_STATUSES:
        status = "complete"
    payload = {
        "schema": "loramake.api.experiment.v1",
        "experiment_id": experiment_id,
        "source_revision": source_revision,
        "catalog_revision": 0,
        "title": "Harem 8-Step v1.0 vs v1.5 复杂提示词基准",
        "zone": "sfw",
        "status": status,
        "provisional": True,
        "created_at": None,
        "updated_at": updated_at,
        "sample_count": len(samples),
        "arms": arms,
        "dimensions": dimensions,
        "samples": samples,
        "automatic_evaluation": {
            "policy": _policy_text(scoring.get("policy")),
            "summary_by_arm": summary_by_arm,
        },
        "human_review": _empty_human_review(),
        "preservation": {
            "cleanup_authorized": bool(manifest.get("cleanup_authorized", False)),
            "scene_preserved": status == "preserved_waiting_user_confirmation",
        },
        "warnings": [
            "Only two seeds were measured; all aggregate conclusions are provisional.",
            "Automatic prompt-adherence scoring is separate from final human aesthetic review.",
        ],
    }
    return payload, assets


def _normalize_harem_ab(
    source: Path,
    manifest: dict[str, Any],
    source_revision: str,
    updated_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_id = _legacy_experiment_id(source, str(manifest["schema"]))
    runs = manifest.get("runs")
    seeds = manifest.get("seeds")
    if not isinstance(runs, list) or not isinstance(seeds, list) or len(seeds) != 2:
        raise ValueError("legacy Harem A/B run shape is invalid")
    assets: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for run in runs:
        lane = _slug(str(run.get("lane", "")), limit=24)
        model_label = str(run.get("model", "model"))
        arm_id = f"{lane}_{_slug(model_label, limit=24)}"
        arms.append({
            "arm_id": arm_id,
            "label": f"{model_label} · {run.get('lane')}",
            "model": _model_basename(str(run.get("unet", ""))),
            "recipe": _recipe(run.get("recipe")),
        })
        files = run.get("files")
        if not isinstance(files, list) or len(files) != len(seeds):
            raise ValueError("legacy Harem A/B file count is invalid")
        for index, file_entry in enumerate(files):
            asset, private = _register_asset(
                experiment_id=experiment_id,
                zone="sfw",
                raw_path=str(file_entry.get("path", "")),
                expected_sha256=str(file_entry.get("sha256", "")),
                expected_bytes=int(file_entry.get("bytes", 0)),
            )
            assets.append(private)
            samples.append({
                "sample_id": f"smp_{arm_id}_{int(seeds[index])}",
                "arm_id": arm_id,
                "dimension_id": "visual_comparison",
                "seed": int(seeds[index]),
                "asset": asset,
                "score": None,
                "elapsed_s": 0.0,
            })
    status = str(manifest.get("status", "complete"))
    if status not in _PUBLIC_STATUSES:
        status = "complete"
    payload = {
        "schema": "loramake.api.experiment.v1",
        "experiment_id": experiment_id,
        "source_revision": source_revision,
        "catalog_revision": 0,
        "title": "Harem 8-Step v1.0 vs v1.5 初始 A/B",
        "zone": "sfw",
        "status": status,
        "provisional": True,
        "created_at": None,
        "updated_at": updated_at,
        "sample_count": len(samples),
        "arms": arms,
        "dimensions": [{
            "dimension_id": "visual_comparison",
            "label": "视觉对比",
            "prompt": str(manifest.get("prompt", "")),
            "criteria": [],
        }],
        "samples": samples,
        "automatic_evaluation": {
            "policy": "not_scored",
            "summary_by_arm": {},
        },
        "human_review": _empty_human_review(),
        "preservation": {
            "cleanup_authorized": bool(manifest.get("cleanup_authorized", False)),
            "scene_preserved": status == "preserved_waiting_user_confirmation",
        },
        "warnings": [
            "This legacy A/B run has no automatic criterion scores.",
            "Human visual review is required.",
        ],
    }
    return payload, assets


def _normalize_canonical(
    source: Path,
    manifest: dict[str, Any],
    source_revision: str,
    updated_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_id = str(manifest.get("experiment_id", ""))
    if not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError("canonical experiment_id is invalid")
    zone = str(manifest.get("zone", ""))
    if zone not in ("sfw", "nsfw"):
        raise ValueError("canonical zone is invalid")
    asset_availability = str(manifest.get("asset_availability", "required"))
    if asset_availability not in _ASSET_AVAILABILITY_MODES:
        raise ValueError("canonical asset_availability is invalid")
    allow_missing_assets = asset_availability == "local_only"
    raw_arms = manifest.get("arms")
    raw_dimensions = manifest.get("dimensions")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_arms, list) or not isinstance(raw_dimensions, list):
        raise ValueError("canonical arms and dimensions must be arrays")
    if not isinstance(raw_samples, list):
        raise ValueError("canonical samples must be an array")
    arms: list[dict[str, Any]] = []
    arm_ids: set[str] = set()
    for arm in raw_arms:
        if not isinstance(arm, dict):
            raise ValueError("canonical arm is invalid")
        arm_id = str(arm.get("arm_id", ""))
        if not _ENTITY_ID_RE.fullmatch(arm_id) or arm_id in arm_ids:
            raise ValueError("canonical arm_id is invalid or duplicated")
        label = str(arm.get("label", ""))
        model = _model_basename(str(arm.get("model", "")))
        if not label or len(label) > 120 or not model or len(model) > 200:
            raise ValueError("canonical arm display fields are invalid")
        arm_ids.add(arm_id)
        arms.append({
            "arm_id": arm_id,
            "label": label,
            "model": model,
            # Optional since 2026-08-10: auto-recorded studio jobs cannot see the
            # engine's effective sampling parameters; recipe-less arms stay indexable.
            "recipe": _recipe(arm["recipe"]) if arm.get("recipe") is not None else None,
        })
    dimensions: list[dict[str, Any]] = []
    dimension_ids: set[str] = set()
    for dimension in raw_dimensions:
        if not isinstance(dimension, dict):
            raise ValueError("canonical dimension is invalid")
        dimension_id = str(dimension.get("dimension_id", ""))
        criteria = dimension.get("criteria")
        if (
            not _ENTITY_ID_RE.fullmatch(dimension_id)
            or dimension_id in dimension_ids
            or not isinstance(criteria, list)
            or any(not isinstance(item, str) for item in criteria)
        ):
            raise ValueError("canonical dimension contract is invalid")
        label = str(dimension.get("label", ""))
        prompt = str(dimension.get("prompt", ""))
        if (
            not label
            or len(label) > 120
            or not prompt
            or any(not item or len(item) > 1000 for item in criteria)
        ):
            raise ValueError("canonical dimension display fields are invalid")
        dimension_ids.add(dimension_id)
        dimensions.append({
            "dimension_id": dimension_id,
            "label": label,
            "prompt": prompt[:16000],
            "criteria": criteria,
        })
    assets: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for sample in raw_samples:
        if not isinstance(sample, dict):
            raise ValueError("canonical sample is invalid")
        sample_id = str(sample.get("sample_id", ""))
        arm_id = str(sample.get("arm_id", ""))
        dimension_id = str(sample.get("dimension_id", ""))
        if (
            not _SAMPLE_ID_RE.fullmatch(sample_id)
            or sample_id in sample_ids
            or arm_id not in arm_ids
            or dimension_id not in dimension_ids
        ):
            raise ValueError("canonical sample references are invalid")
        sample_ids.add(sample_id)
        raw_asset = sample.get("asset")
        if not isinstance(raw_asset, dict):
            raise ValueError("canonical sample asset is invalid")
        asset, private = _register_asset(
            experiment_id=experiment_id,
            zone=zone,
            raw_path=str(raw_asset.get("path", "")),
            expected_sha256=str(raw_asset.get("sha256", "")),
            expected_bytes=int(raw_asset.get("bytes", 0)),
            allow_missing=allow_missing_assets,
        )
        if private is not None:
            assets.append(private)
        seed = int(sample.get("seed", 0))
        elapsed_s = float(sample.get("elapsed_s", 0))
        if seed < 0 or elapsed_s < 0:
            raise ValueError("canonical sample timing fields are invalid")
        samples.append({
            "sample_id": sample_id,
            "arm_id": arm_id,
            "dimension_id": dimension_id,
            "seed": seed,
            "asset": asset,
            "score": _score(sample.get("score")),
            "elapsed_s": elapsed_s,
        })
    reports, report_assets = _normalize_reports(
        source,
        manifest.get("reports"),
        experiment_id=experiment_id,
        zone=zone,
        allow_missing_assets=allow_missing_assets,
    )
    assets.extend(report_assets)
    status = str(manifest.get("status", "complete"))
    if status not in _PUBLIC_STATUSES:
        raise ValueError("canonical status is invalid")
    title = str(manifest.get("title", ""))
    warnings = manifest.get("warnings", [])
    if not title or len(title) > 200 or not isinstance(warnings, list):
        raise ValueError("canonical experiment display fields are invalid")
    if any(not isinstance(value, str) for value in warnings):
        raise ValueError("canonical warnings are invalid")
    warnings = list(warnings)
    declared = [sample["asset"] for sample in samples]
    declared += [entry["asset"] for report in reports for entry in report["assets"]]
    unavailable = sum(1 for entry in declared if entry["availability"] == "missing")
    if unavailable:
        warnings.append(
            f"{unavailable} of {len(declared)} declared renders are not present in this "
            "checkout; their recorded SHA-256 and byte counts remain the evidence of "
            "record and delivery of those assets stays refused."
        )
    payload = {
        "schema": "loramake.api.experiment.v1",
        "experiment_id": experiment_id,
        "source_revision": source_revision,
        "catalog_revision": 0,
        "title": title,
        "zone": zone,
        "status": status,
        "provisional": bool(manifest.get("provisional", False)),
        "created_at": manifest.get("created_at"),
        "updated_at": updated_at,
        "sample_count": len(samples),
        "arms": arms,
        "dimensions": dimensions,
        "samples": samples,
        "reports": reports,
        "automatic_evaluation": {
            "policy": _policy_text(
                manifest.get("automatic_evaluation", {}).get("policy", "not_scored")
                if isinstance(manifest.get("automatic_evaluation"), dict)
                else "not_scored"
            ),
            "summary_by_arm": _summary_by_arm(
                manifest.get("automatic_evaluation", {}).get("summary_by_arm", {})
                if isinstance(manifest.get("automatic_evaluation"), dict)
                else {}
            ),
        },
        "human_review": _empty_human_review(),
        "preservation": {
            "cleanup_authorized": bool(
                manifest.get("preservation", {}).get("cleanup_authorized", False)
                if isinstance(manifest.get("preservation"), dict)
                else False
            ),
            "scene_preserved": bool(
                manifest.get("preservation", {}).get("scene_preserved", False)
                if isinstance(manifest.get("preservation"), dict)
                else False
            ),
        },
        "warnings": warnings,
    }
    return payload, assets


def _normalize(
    source: Path,
    manifest: dict[str, Any],
    source_revision: str,
    updated_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = str(manifest.get("schema", ""))
    if schema == "harem-complex-prompt-benchmark-manifest-v1":
        return _normalize_complex(source, manifest, source_revision, updated_at)
    if schema == "loramake.harem_ab.v1":
        return _normalize_harem_ab(source, manifest, source_revision, updated_at)
    if schema == "loramake.experiment-manifest.v1":
        return _normalize_canonical(source, manifest, source_revision, updated_at)
    raise ValueError(f"unsupported schema: {schema or '<missing>'}")


def _error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "unsupported schema" in message:
        return "unsupported_schema"
    if "sha256 mismatch" in message or "byte count mismatch" in message:
        return "evidence_mismatch"
    if "outside" in message or "absolute" in message or "symlink" in message:
        return "unsafe_path"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    return "invalid_manifest"


def _index_failure(
    conn: sqlite3.Connection,
    *,
    source_rel: str,
    stat: os.stat_result,
    source_schema: str | None,
    exc: Exception,
) -> dict[str, Any] | None:
    existing = conn.execute(
        "SELECT * FROM experiment_sources WHERE source_rel = ?", (source_rel,)
    ).fetchone()
    code = _error_code(exc)
    detail = str(exc)[:500]
    if existing and (
        existing["source_mtime_ns"] == stat.st_mtime_ns
        and existing["source_size"] == stat.st_size
        and existing["error_code"] == code
        and existing["error_detail"] == detail
    ):
        return None
    experiment_id = existing["experiment_id"] if existing else _fallback_experiment_id(source_rel)
    zone = existing["zone"] if existing else "sfw"
    if existing:
        conn.execute(
            """UPDATE experiment_sources
               SET source_mtime_ns = ?, source_size = ?, updated_at = ?,
                   index_stale = 1, error_code = ?, error_detail = ?
               WHERE experiment_id = ?""",
            (stat.st_mtime_ns, stat.st_size, _utc_from_timestamp(stat.st_mtime), code, detail, experiment_id),
        )
    else:
        conn.execute(
            """INSERT INTO experiment_sources
               (experiment_id, zone, source_rel, source_schema, source_revision,
                source_mtime_ns, source_size, updated_at, payload_json,
                index_stale, error_code, error_detail)
               VALUES (?, ?, ?, ?, NULL, ?, ?, ?, NULL, 1, ?, ?)""",
            (
                experiment_id,
                zone,
                source_rel,
                source_schema,
                stat.st_mtime_ns,
                stat.st_size,
                _utc_from_timestamp(stat.st_mtime),
                code,
                detail,
            ),
        )
    event = _next_revision(
        conn,
        zone=zone,
        event_type="experiment.invalidated",
        experiment_id=experiment_id,
    )
    conn.commit()
    return event


def _index_one(
    conn: sqlite3.Connection,
    source: Path,
    source_rel: str,
    analysis_root: Path,
    force: bool,
) -> dict[str, Any] | None:
    try:
        stat = source.lstat()
    except OSError:
        # The source disappeared between enumeration and inspection. Treat the
        # scan as incomplete for this entry and retry on the next TTL pass.
        return None
    existing = conn.execute(
        "SELECT * FROM experiment_sources WHERE source_rel = ?", (source_rel,)
    ).fetchone()
    if (
        not force
        and existing
        and existing["source_mtime_ns"] == stat.st_mtime_ns
        and existing["source_size"] == stat.st_size
        and not existing["index_stale"]
    ):
        return None
    source_schema: str | None = None
    try:
        if source.is_symlink() or not source.resolve().is_relative_to(analysis_root):
            raise ValueError("manifest symlink or path escape is forbidden")
        if stat.st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds the size limit")
        data = source.read_bytes()
        manifest = json.loads(data)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must contain a JSON object")
        source_schema = str(manifest.get("schema", ""))
        if source_schema not in _KNOWN_SCHEMAS:
            raise ValueError(f"unsupported schema: {source_schema or '<missing>'}")
        source_revision = _sha256(data)
        updated_at = _utc_from_timestamp(stat.st_mtime)
        payload, assets = _normalize(source, manifest, source_revision, updated_at)
        experiment_id = payload["experiment_id"]
        zone = payload["zone"]
        if (
            existing
            and existing["payload_json"] is not None
            and existing["experiment_id"] != experiment_id
        ):
            raise ValueError("experiment_id cannot change for an indexed source")
        conflict = conn.execute(
            "SELECT source_rel FROM experiment_sources WHERE experiment_id = ? AND source_rel != ?",
            (experiment_id, source_rel),
        ).fetchone()
        if conflict:
            raise ValueError("experiment_id collides with another manifest")
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        unchanged = (
            existing
            and existing["source_revision"] == source_revision
            and existing["payload_json"] == payload_json
            and not existing["index_stale"]
        )
        conn.execute(
            """INSERT INTO experiment_sources
               (experiment_id, zone, source_rel, source_schema, source_revision,
                source_mtime_ns, source_size, updated_at, payload_json,
                index_stale, error_code, error_detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)
               ON CONFLICT(source_rel) DO UPDATE SET
                 experiment_id = excluded.experiment_id,
                 zone = excluded.zone,
                 source_schema = excluded.source_schema,
                 source_revision = excluded.source_revision,
                 source_mtime_ns = excluded.source_mtime_ns,
                 source_size = excluded.source_size,
                 updated_at = excluded.updated_at,
                 payload_json = excluded.payload_json,
                 index_stale = 0,
                 error_code = NULL,
                 error_detail = NULL""",
            (
                experiment_id,
                zone,
                source_rel,
                source_schema,
                source_revision,
                stat.st_mtime_ns,
                stat.st_size,
                updated_at,
                payload_json,
            ),
        )
        conn.execute("DELETE FROM experiment_assets WHERE experiment_id = ?", (experiment_id,))
        unique_assets = {asset["asset_id"]: asset for asset in assets}
        for asset in unique_assets.values():
            conn.execute(
                """INSERT INTO experiment_assets
                   (experiment_id, asset_id, zone, path_rel, media_type, bytes, sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset["experiment_id"],
                    asset["asset_id"],
                    asset["zone"],
                    asset["path_rel"],
                    asset["media_type"],
                    asset["bytes"],
                    asset["sha256"],
                ),
            )
        if unchanged:
            conn.commit()
            return None
        event = _next_revision(
            conn,
            zone=zone,
            event_type="experiment.indexed",
            experiment_id=experiment_id,
        )
        conn.commit()
        return event
    except Exception as exc:
        conn.rollback()
        return _index_failure(
            conn,
            source_rel=source_rel,
            stat=stat,
            source_schema=source_schema,
            exc=exc,
        )


def scan(*, force: bool = False, db_path: Path | None = None, analysis_root: Path | None = None) -> dict[str, Any]:
    """Incrementally scan supported manifests.  Per-file failures stay isolated."""
    db = Path(db_path or DB_PATH)
    root = Path(analysis_root or ANALYSIS_ROOT).resolve()
    cache_key = (str(db.resolve()), str(root))
    now = time.monotonic()
    with _scan_lock:
        if not force and now - _last_scan.get(cache_key, 0.0) < SCAN_TTL_S:
            conn = _connect(db)
            try:
                return {
                    "catalog_revision": _catalog_revision(conn),
                    "index_stale": bool(conn.execute(
                        """SELECT 1 FROM experiment_sources
                           WHERE index_stale = 1 AND payload_json IS NOT NULL LIMIT 1"""
                    ).fetchone()),
                }
            finally:
                conn.close()
        events: list[dict[str, Any]] = []
        conn = _connect(db)
        try:
            seen: set[str] = set()
            scan_complete = root.is_dir()
            sources: list[Path] = []
            if scan_complete:
                try:
                    sources = sorted(root.rglob("manifest.json"))
                except OSError:
                    scan_complete = False
            if scan_complete:
                for source in sources:
                    source_rel = source.relative_to(root).as_posix()
                    seen.add(source_rel)
                    event = _index_one(conn, source, source_rel, root, force)
                    if event:
                        events.append(event)
                existing = conn.execute(
                    "SELECT experiment_id, zone, source_rel FROM experiment_sources"
                ).fetchall()
                for row in existing:
                    if row["source_rel"] in seen:
                        continue
                    conn.execute("DELETE FROM experiment_assets WHERE experiment_id = ?", (row["experiment_id"],))
                    conn.execute("DELETE FROM experiment_sources WHERE experiment_id = ?", (row["experiment_id"],))
                    event = _next_revision(
                        conn,
                        zone=row["zone"],
                        event_type="experiment.removed",
                        experiment_id=row["experiment_id"],
                    )
                    events.append(event)
            conn.commit()
            revision = _catalog_revision(conn)
            stale = not scan_complete or bool(conn.execute(
                """SELECT 1 FROM experiment_sources
                   WHERE index_stale = 1 AND payload_json IS NOT NULL LIMIT 1"""
            ).fetchone())
        finally:
            conn.close()
        _last_scan[cache_key] = now
    for event in events:
        _publish(event)
    return {"catalog_revision": revision, "index_stale": stale}


def capabilities(*, db_path: Path | None = None, analysis_root: Path | None = None) -> dict[str, Any]:
    state = scan(db_path=db_path, analysis_root=analysis_root)
    return {
        "schema": "loramake.api.capabilities.v1",
        "catalog_revision": state["catalog_revision"],
        "experiment_schema": "loramake.api.experiment.v1",
        "review_schema": "loramake.api.review-event.v1",
        "features": {
            "experiments_read": True,
            "gallery_read": True,
            "reviews_write": True,
            "events_sse": True,
            "benchmark_run": False,
            "cleanup": False,
            "arbitrary_command": False,
        },
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": payload["experiment_id"],
        "title": payload["title"],
        "status": payload["status"],
        "provisional": payload["provisional"],
        "updated_at": payload["updated_at"],
        "sample_count": payload["sample_count"],
        "arm_count": len(payload["arms"]),
        "dimension_count": len(payload["dimensions"]),
        "source_revision": payload["source_revision"],
    }


def list_experiments(
    *,
    zone: str,
    cursor: str | None = None,
    limit: int = 30,
    status: str | None = None,
    model: str | None = None,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> dict[str, Any]:
    if zone not in ("sfw", "nsfw"):
        raise ExperimentError(400, "invalid_zone", "zone must be sfw or nsfw")
    state = scan(db_path=db_path, analysis_root=analysis_root)
    limit = max(1, min(int(limit), 100))
    after = _decode_cursor(cursor, 2)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT updated_at, experiment_id, payload_json
               FROM experiment_sources
               WHERE zone = ? AND payload_json IS NOT NULL
               ORDER BY updated_at DESC, experiment_id ASC""",
            (zone,),
        ).fetchall()
    finally:
        conn.close()
    items: list[tuple[str, str, dict[str, Any]]] = []
    model_query = (model or "").casefold().strip()
    for row in rows:
        if after:
            after_updated, after_id = str(after[0]), str(after[1])
            if row["updated_at"] > after_updated:
                continue
            if row["updated_at"] == after_updated and row["experiment_id"] <= after_id:
                continue
        payload = json.loads(row["payload_json"])
        if status and payload.get("status") != status:
            continue
        if model_query and not any(model_query in str(arm.get("model", "")).casefold() for arm in payload["arms"]):
            continue
        items.append((row["updated_at"], row["experiment_id"], _summary(payload)))
        if len(items) > limit:
            break
    has_more = len(items) > limit
    page = items[:limit]
    next_cursor = _opaque_cursor([page[-1][0], page[-1][1]]) if has_more and page else None
    return {
        "schema": "loramake.api.experiment-page.v1",
        "catalog_revision": state["catalog_revision"],
        "items": [entry[2] for entry in page],
        "next_cursor": next_cursor,
        "index_stale": state["index_stale"],
    }


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "schema": "loramake.api.review-event.v1",
        "event_id": row["event_id"],
        "experiment_id": row["experiment_id"],
        "source_revision": row["source_revision"],
        "target": {"type": row["target_type"], "id": row["target_id"]},
        "decision": row["decision"],
        "comment": row["comment"],
        "actor": {"kind": row["actor_kind"], "id": row["actor_id"]},
        "client_id": row["client_id"],
        "supersedes_event_id": row["supersedes_event_id"],
        "created_at": row["created_at"],
    }


def _review_projection(
    conn: sqlite3.Connection,
    experiment_id: str,
    source_revision: str,
) -> dict[str, Any]:
    rows = conn.execute(
        """SELECT * FROM review_events
           WHERE experiment_id = ?
           ORDER BY rowid ASC""",
        (experiment_id,),
    ).fetchall()
    current: dict[str, Any] = {}
    current_rows: list[sqlite3.Row] = []
    for row in rows:
        if row["source_revision"] != source_revision:
            continue
        current_rows.append(row)
        event = _event_dict(row)
        current[f"{row['target_type']}:{row['target_id']}"] = event
    return {
        "event_count": len(rows),
        "latest_event_id": current_rows[-1]["event_id"] if current_rows else None,
        "current_by_target": current,
    }


def _etag(source_revision: str, latest_event_id: str | None) -> str:
    review = latest_event_id or "none"
    digest = hashlib.sha256(f"{source_revision}:{review}".encode()).hexdigest()[:24]
    return f'"exp-{digest}"'


def get_experiment(
    experiment_id: str,
    *,
    zone: str,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    scan(db_path=db_path, analysis_root=analysis_root)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM experiment_sources
               WHERE experiment_id = ? AND zone = ? AND payload_json IS NOT NULL""",
            (experiment_id, zone),
        ).fetchone()
        if row is None:
            raise ExperimentError(404, "experiment_not_found", "experiment does not exist in this zone")
        payload = json.loads(row["payload_json"])
        review = _review_projection(conn, experiment_id, payload["source_revision"])
        payload["human_review"] = review
        payload["catalog_revision"] = _catalog_revision(conn)
        stale_warning = (
            "The source manifest could not be re-indexed; this is the last valid projection."
        )
        if row["index_stale"] and stale_warning not in payload["warnings"]:
            payload["warnings"].append(stale_warning)
        return payload, _etag(payload["source_revision"], review["latest_event_id"])
    finally:
        conn.close()


def list_reviews(
    experiment_id: str,
    *,
    zone: str,
    cursor: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> dict[str, Any]:
    get_experiment(experiment_id, zone=zone, db_path=db_path, analysis_root=analysis_root)
    after = _decode_cursor(cursor, 2)
    limit = max(1, min(int(limit), 100))
    conn = _connect(db_path)
    try:
        query = """SELECT rowid AS review_seq, * FROM review_events
                   WHERE experiment_id = ? AND zone = ?"""
        params: list[Any] = [experiment_id, zone]
        if after:
            if not isinstance(after[0], str) or not after[0]:
                raise ExperimentError(400, "invalid_cursor", "cursor created_at is invalid")
            try:
                review_seq = int(after[1])
            except (TypeError, ValueError) as exc:
                raise ExperimentError(400, "invalid_cursor", "cursor review sequence is invalid") from exc
            if review_seq < 1:
                raise ExperimentError(400, "invalid_cursor", "cursor review sequence is invalid")
            query += " AND (created_at < ? OR (created_at = ? AND rowid < ?))"
            params.extend([after[0], after[0], review_seq])
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit + 1)
        rows = conn.execute(query, params).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        source_row = conn.execute(
            """SELECT source_revision FROM experiment_sources
               WHERE experiment_id = ? AND zone = ? AND payload_json IS NOT NULL""",
            (experiment_id, zone),
        ).fetchone()
        if source_row is None:
            raise ExperimentError(404, "experiment_not_found", "experiment does not exist in this zone")
        projection = _review_projection(conn, experiment_id, source_row["source_revision"])
    finally:
        conn.close()
    next_cursor = (
        _opaque_cursor([page[-1]["created_at"], page[-1]["review_seq"]])
        if has_more and page else None
    )
    return {
        "schema": "loramake.api.review-page.v1",
        "events": [_event_dict(row) for row in page],
        "current_by_target": projection["current_by_target"],
        "next_cursor": next_cursor,
    }


def _validate_review_body(
    payload: dict[str, Any],
) -> tuple[str, str, str, str | None, str, str, str, str | None]:
    expected_keys = {
        "schema",
        "source_revision",
        "target",
        "decision",
        "comment",
        "actor",
        "supersedes_event_id",
    }
    if set(payload) != expected_keys:
        raise ExperimentError(400, "invalid_review", "review fields do not match the v1 contract")
    if payload.get("schema") != "loramake.api.review-write.v1":
        raise ExperimentError(400, "invalid_review", "review schema is invalid")
    source_revision = str(payload.get("source_revision", ""))
    if not _SHA256_RE.fullmatch(source_revision):
        raise ExperimentError(400, "invalid_review", "source_revision is invalid")
    target = payload.get("target")
    if not isinstance(target, dict) or set(target) != {"type", "id"}:
        raise ExperimentError(400, "invalid_review", "target must be an object")
    target_type = str(target.get("type", ""))
    target_id = str(target.get("id", ""))
    if target_type not in ("experiment", "dimension", "sample", "asset", "arm", "seed") or not target_id or len(target_id) > 128:
        raise ExperimentError(400, "invalid_review", "review target is invalid")
    decision = str(payload.get("decision", ""))
    comment = payload.get("comment")
    if decision == "comment":
        if not isinstance(comment, str) or not comment.strip() or len(comment) > 2000:
            raise ExperimentError(400, "invalid_review", "comment decision requires text")
        comment = comment.strip()
    elif decision in ("yes", "no"):
        if comment is not None:
            raise ExperimentError(400, "invalid_review", "yes/no decisions require comment=null")
    else:
        raise ExperimentError(400, "invalid_review", "decision is invalid")
    actor = payload.get("actor")
    if not isinstance(actor, dict) or set(actor) != {"kind", "id"}:
        raise ExperimentError(400, "invalid_review", "actor must be an object")
    actor_kind = str(actor.get("kind", ""))
    actor_id = str(actor.get("id", ""))
    if actor_kind not in ("human", "agent") or not _ACTOR_ID_RE.fullmatch(actor_id):
        raise ExperimentError(400, "invalid_review", "actor is invalid")
    supersedes = payload.get("supersedes_event_id")
    if supersedes is not None and not _EVENT_ID_RE.fullmatch(str(supersedes)):
        raise ExperimentError(400, "invalid_review", "supersedes_event_id is invalid")
    return source_revision, target_type, target_id, comment, decision, actor_kind, actor_id, supersedes


def _review_targets(detail: dict[str, Any], experiment_id: str) -> dict[str, set[str]]:
    """Return valid opaque target IDs for the currently indexed evidence."""
    samples = detail["samples"]
    return {
        "experiment": {experiment_id},
        "dimension": {item["dimension_id"] for item in detail["dimensions"]},
        "sample": {item["sample_id"] for item in samples},
        "asset": {item["asset"]["asset_id"] for item in samples},
        "arm": {item["arm_id"] for item in detail["arms"]},
        "seed": {str(item["seed"]) for item in samples},
    }


def create_review(
    experiment_id: str,
    *,
    zone: str,
    body: dict[str, Any],
    client_id: str,
    idempotency_key: str,
    if_match: str,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    if not _CLIENT_ID_RE.fullmatch(client_id):
        raise ExperimentError(400, "invalid_client", "X-Loramake-Client is invalid")
    if not 8 <= len(idempotency_key) <= 128:
        raise ExperimentError(400, "invalid_idempotency_key", "Idempotency-Key length is invalid")
    request_sha = _sha256(_json_bytes(body))
    conn = _connect(db_path)
    try:
        existing = conn.execute(
            "SELECT * FROM review_events WHERE client_id = ? AND idempotency_key = ?",
            (client_id, idempotency_key),
        ).fetchone()
        if existing:
            if (
                existing["request_sha256"] != request_sha
                or existing["experiment_id"] != experiment_id
                or existing["zone"] != zone
            ):
                raise ExperimentError(409, "idempotency_conflict", "Idempotency-Key was reused for another request")
            return _event_dict(existing), False
    finally:
        conn.close()

    detail, current_etag = get_experiment(
        experiment_id, zone=zone, db_path=db_path, analysis_root=analysis_root
    )
    if if_match != current_etag:
        raise ExperimentError(412, "stale_evidence", "experiment evidence or reviews changed")
    (
        source_revision,
        target_type,
        target_id,
        comment,
        decision,
        actor_kind,
        actor_id,
        supersedes,
    ) = _validate_review_body(body)
    if source_revision != detail["source_revision"]:
        raise ExperimentError(409, "stale_evidence", "source_revision does not match current evidence")

    valid_targets = _review_targets(detail, experiment_id)
    if target_id not in valid_targets[target_type]:
        raise ExperimentError(400, "invalid_target", "target does not exist in this experiment")

    _verify_experiment_assets(experiment_id, zone=zone, db_path=db_path)

    conn = _connect(db_path)
    event: dict[str, Any]
    catalog_event: dict[str, Any]
    try:
        conn.execute("BEGIN IMMEDIATE")
        duplicate = conn.execute(
            "SELECT * FROM review_events WHERE client_id = ? AND idempotency_key = ?",
            (client_id, idempotency_key),
        ).fetchone()
        if duplicate:
            if (
                duplicate["request_sha256"] != request_sha
                or duplicate["experiment_id"] != experiment_id
                or duplicate["zone"] != zone
            ):
                raise ExperimentError(409, "idempotency_conflict", "Idempotency-Key was reused for another request")
            conn.rollback()
            return _event_dict(duplicate), False

        source_row = conn.execute(
            """SELECT payload_json, source_revision, index_stale FROM experiment_sources
               WHERE experiment_id = ? AND zone = ? AND payload_json IS NOT NULL""",
            (experiment_id, zone),
        ).fetchone()
        if source_row is None:
            raise ExperimentError(404, "experiment_not_found", "experiment does not exist in this zone")
        if source_row["index_stale"]:
            raise ExperimentError(
                409,
                "evidence_unavailable",
                "the source changed but its replacement could not be verified",
            )
        projection = _review_projection(conn, experiment_id, source_row["source_revision"])
        locked_etag = _etag(source_row["source_revision"], projection["latest_event_id"])
        if if_match != locked_etag:
            raise ExperimentError(412, "stale_evidence", "experiment evidence or reviews changed")
        if source_revision != source_row["source_revision"]:
            raise ExperimentError(409, "stale_evidence", "source_revision does not match current evidence")
        locked_detail = json.loads(source_row["payload_json"])
        locked_targets = _review_targets(locked_detail, experiment_id)
        if target_id not in locked_targets[target_type]:
            raise ExperimentError(400, "invalid_target", "target does not exist in this experiment")
        current = projection["current_by_target"].get(f"{target_type}:{target_id}")
        if supersedes is not None:
            if current is None or current["event_id"] != supersedes:
                raise ExperimentError(
                    409,
                    "supersedes_conflict",
                    "supersedes_event_id is not the current target event",
                )
        elif current is not None:
            raise ExperimentError(
                409,
                "supersedes_required",
                "an existing target decision must be superseded explicitly",
            )

        event_id = f"rev_{uuid.uuid4().hex[:20]}"
        created_at = _utc_now()
        conn.execute(
            """INSERT INTO review_events
               (event_id, experiment_id, zone, source_revision, target_type, target_id,
                decision, comment, actor_kind, actor_id, client_id,
                supersedes_event_id, idempotency_key, request_sha256, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                experiment_id,
                zone,
                source_revision,
                target_type,
                target_id,
                decision,
                comment,
                actor_kind,
                actor_id,
                client_id,
                supersedes,
                idempotency_key,
                request_sha,
                created_at,
            ),
        )
        row = conn.execute("SELECT * FROM review_events WHERE event_id = ?", (event_id,)).fetchone()
        event = _event_dict(row)
        catalog_event = _next_revision(
            conn,
            zone=zone,
            event_type="review.created",
            experiment_id=experiment_id,
            review_event_id=event_id,
        )
        conn.commit()
    except ExperimentError:
        conn.rollback()
        raise
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ExperimentError(409, "review_conflict", "review could not be committed") from exc
    finally:
        conn.close()
    _publish(catalog_event)
    return event, True


def catalog_events_since(
    *,
    zone: str,
    since_revision: int = 0,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> list[dict[str, Any]]:
    scan(db_path=db_path, analysis_root=analysis_root)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM catalog_events
               WHERE zone = ? AND revision > ?
               ORDER BY revision ASC LIMIT 1000""",
            (zone, max(0, int(since_revision))),
        ).fetchall()
        return [{
            "schema": "loramake.api.catalog-event.v1",
            "revision": row["revision"],
            "zone": row["zone"],
            "type": row["event_type"],
            "experiment_id": row["experiment_id"],
            "review_event_id": row["review_event_id"],
            "created_at": row["created_at"],
        } for row in rows]
    finally:
        conn.close()


def get_asset(
    experiment_id: str,
    asset_id: str,
    *,
    zone: str,
    db_path: Path | None = None,
    analysis_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    get_experiment(experiment_id, zone=zone, db_path=db_path, analysis_root=analysis_root)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM experiment_assets
               WHERE experiment_id = ? AND asset_id = ? AND zone = ?""",
            (experiment_id, asset_id, zone),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ExperimentError(404, "asset_not_found", "asset does not exist in this experiment")
    path = _verify_asset_row(row, zone=zone)
    return path, dict(row)


def _verify_asset_row(row: sqlite3.Row | dict[str, Any], *, zone: str) -> Path:
    unresolved = ROOT / row["path_rel"]
    if unresolved.is_symlink():
        raise ExperimentError(409, "asset_changed", "asset became a symlink")
    path = unresolved.resolve()
    if not _allowed_asset(path, zone) or not path.is_file():
        raise ExperimentError(409, "asset_changed", "asset is missing or outside its zone")
    data = path.read_bytes()
    if len(data) != row["bytes"] or _sha256(data) != row["sha256"]:
        raise ExperimentError(409, "asset_changed", "asset changed since indexing")
    return path


def _verify_experiment_assets(
    experiment_id: str,
    *,
    zone: str,
    db_path: Path | None = None,
) -> None:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM experiment_assets
               WHERE experiment_id = ? AND zone = ?""",
            (experiment_id, zone),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ExperimentError(409, "evidence_unavailable", "experiment has no indexed assets")
    for row in rows:
        _verify_asset_row(row, zone=zone)


def thumbnail(path: Path, asset: dict[str, Any], width: int, *, zone: str) -> Path:
    width = min(THUMB_WIDTHS, key=lambda value: abs(value - int(width)))
    target_dir = NSFW_THUMB_CACHE if zone == "nsfw" else THUMB_CACHE
    target = target_dir / f"{asset['sha256']}_{width}.webp"
    if target.is_file():
        return target
    try:
        with Image.open(io.BytesIO(path.read_bytes())) as image:
            image = image.convert("RGB")
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", quality=82, method=4)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ExperimentError(
            415, "invalid_asset", "asset is not a decodable image"
        ) from exc
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target_dir, 0o700)
    except OSError:
        pass
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(buffer.getvalue())
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, target)
    return target
