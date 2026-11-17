#!/usr/bin/env python3
"""Offline DRAFT/HOLD dataset planner for Anima Step 2a.

The tool filters metadata, reconciles externally computed hashes, and plans a
human-review package. It has no image downloader and performs no network I/O.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = "anima-step2a-dataset-manifest-v1"
STATUS = "DRAFT_HOLD_PENDING_USER_FINAL_REVIEW"
FILTER_VERSION = "anima-step2a-draft-filter-v1"
REVIEW_VERSION = "anima-step2a-review-package-plan-v1"
DEFAULT_MIN_SCORE = 10.0
DEFAULT_PHASH_DISTANCE = 4
DEFAULT_VALIDATION_RATIO = 0.05
DEFAULT_SPLIT_SALT = "anima-step2a-draft-v1"
DEFAULT_MAX_INPUT_RECORDS = 10_000
MAX_INPUT_RECORDS = 100_000
STAGE_ORDER = [
    "metadata_filter",
    "md5_sha256_exact_dedup",
    "phash_near_dedup",
    "train_validation_split_after_dedup",
    "human_review_plan",
]

HARD_TAGS = frozenset(
    """
    baby bestiality child child_abuse elementary_school_student forced guro infant
    kindergartener loli lolicon middle_school_student minor molestation newborn
    preschooler rape sexual_abuse shota shotacon toddler underage zoophilia
    """.split()
)
SEXUAL_TAGS = frozenset({"explicit", "nsfw", "adult"})
WATCH_TAGS = frozenset(
    """
    age_regression aged_down childhood diaper elementary_school gym_uniform
    junior_high_school_student kindergarten middle_school pacifier randoseru
    school_swimsuit school_uniform serafuku young young-looking
    """.split()
)
RATINGS = {
    "g": "general",
    "general": "general",
    "safe": "general",
    "s": "sensitive",
    "sensitive": "sensitive",
    "q": "questionable",
    "questionable": "questionable",
    "e": "explicit",
    "explicit": "explicit",
}
FILTER_DECISIONS = {
    "DRAFT_ELIGIBLE",
    "HOLD_MISSING_METADATA",
    "HOLD_OPT_OUT_UNKNOWN",
    "HOLD_RATING_SENSITIVE",
    "HOLD_WATCHLIST_TAG",
    "REJECT_OPT_OUT",
    "REJECT_BLOCKED_TAG",
    "REJECT_RATING",
    "REJECT_SCORE",
}
FINAL_DECISIONS = FILTER_DECISIONS | {
    "HOLD_MISSING_DEDUP_HASH",
    "HOLD_HASH_CONFLICT",
    "REJECT_DUPLICATE_EXACT",
    "REJECT_DUPLICATE_PHASH",
}
HEX32 = re.compile(r"^[a-f0-9]{32}$")
HEX40_64 = re.compile(r"^[a-f0-9]{40,64}$")
HEX64 = re.compile(r"^[a-f0-9]{64}$")
PHASH = re.compile(r"^[a-f0-9]{16}$")


class PlanError(RuntimeError):
    pass


def _json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PlanError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _write_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON atomically; an existing target is never replaced."""

    path = path.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PlanError(f"cannot create output parent {path.parent}: {exc}") from exc
    temporary: Path | None = None
    directory_fd: int | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                "wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(
                    json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise PlanError(f"cannot stage complete output for {path}: {exc}") from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError as exc:
            raise PlanError(f"cannot open output parent for fsync: {path.parent}: {exc}") from exc
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PlanError(f"no-clobber: output already exists: {path}") from exc
        except OSError as exc:
            raise PlanError(f"atomic publish failed for {path}: {exc}") from exc
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            try:
                path.unlink()
            except OSError as rollback_exc:
                raise PlanError(
                    f"output parent fsync failed for {path.parent}: {exc}; "
                    f"published output rollback also failed: {rollback_exc}"
                ) from exc
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise PlanError(
                f"output parent fsync failed for {path.parent}: {exc}; "
                "published output was removed"
            ) from exc
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        if temporary:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlanError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise PlanError(f"{path}:{line_no}: JSONL row must be an object")
                yield row
    except (OSError, UnicodeDecodeError) as exc:
        raise PlanError(f"cannot decode JSONL {path}: {exc}") from exc


def _iter_parquet(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        raise PlanError(
            "Parquet input requires the optional pyarrow engine; engine unavailable, "
            "no output written. Use JSONL or provide pyarrow outside this change."
        ) from exc
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise PlanError(f"{path}: Parquet row must be an object")
                yield row
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError(f"{path}: Parquet decode failed closed: {exc}") from exc


def _read_rows(path: Path, max_records: int) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        iterator, kind = _iter_jsonl(path), "jsonl"
    elif suffix == ".parquet":
        iterator, kind = _iter_parquet(path, min(4096, max_records + 1)), "parquet"
    else:
        raise PlanError(f"{path}: expected .jsonl/.ndjson or .parquet")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iterator, 1):
        if index > max_records:
            raise PlanError(
                f"{path}: input exceeds --max-input-records={max_records}; "
                "selection is fail-closed and no output was written"
            )
        rows.append(row)
    if not rows:
        raise PlanError(f"{path}: zero input records")
    return rows, kind


def _first(row: dict[str, Any], names: Iterable[str]) -> tuple[Any, str]:
    for name in names:
        if row.get(name) is not None:
            return row[name], name
    return None, "missing"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value if value and "\n" not in value and "\r" not in value else None


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return []
    return sorted(
        {str(item).strip().lower().replace(" ", "_") for item in values if str(item).strip()}
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _opt_out(value: Any) -> str:
    if isinstance(value, bool):
        return "opted_out" if value else "not_opted_out"
    token = str(value).strip().lower() if value is not None else ""
    if token in {"1", "true", "yes", "opted_out", "excluded"}:
        return "opted_out"
    if token in {"0", "false", "no", "not_opted_out", "included"}:
        return "not_opted_out"
    return "unknown"


def _hex(value: Any, pattern: re.Pattern[str]) -> str | None:
    token = str(value).strip().lower().removeprefix("0x") if value is not None else ""
    return token if pattern.fullmatch(token) else None


def _phash(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**64:
        return f"{value:016x}"
    if value is None or not str(value).strip():
        return None
    token = str(value).strip().lower().removeprefix("0x")
    token = token.zfill(16)
    return token if PHASH.fullmatch(token) else None


def _filter(
    image_id: str | None,
    tags: list[str],
    rating: str,
    score: float | None,
    opted: str,
    min_score: float,
) -> tuple[str, list[str]]:
    blocked = sorted(set(tags) & (HARD_TAGS | SEXUAL_TAGS))
    watched = sorted(set(tags) & WATCH_TAGS)
    reasons: list[str] = []
    if not image_id:
        reasons.append("missing_image_id")
    if not tags:
        reasons.append("missing_tag_snapshot")
    if rating == "unknown":
        reasons.append("missing_or_unknown_rating")
    if score is None:
        reasons.append("missing_or_invalid_score")
    if opted != "not_opted_out":
        reasons.append(f"opt_out_status:{opted}")
    if blocked:
        reasons.append("blocked_tags:" + ",".join(blocked))
    if watched:
        reasons.append("watchlist_tags:" + ",".join(watched))
    if rating == "sensitive":
        reasons.append("sensitive_rating_requires_review")
    if rating in {"questionable", "explicit"}:
        reasons.append("rating_excluded:" + rating)
    if score is not None and score < min_score:
        reasons.append(f"score_below_minimum:{score:g}<{min_score:g}")

    if opted == "opted_out":
        return "REJECT_OPT_OUT", reasons
    if blocked:
        return "REJECT_BLOCKED_TAG", reasons
    if rating in {"questionable", "explicit"}:
        return "REJECT_RATING", reasons
    if score is not None and score < min_score:
        return "REJECT_SCORE", reasons
    if not image_id or not tags or rating == "unknown" or score is None:
        return "HOLD_MISSING_METADATA", reasons
    if opted == "unknown":
        return "HOLD_OPT_OUT_UNKNOWN", reasons
    if rating == "sensitive":
        return "HOLD_RATING_SENSITIVE", reasons
    if watched:
        return "HOLD_WATCHLIST_TAG", reasons
    return "DRAFT_ELIGIBLE", reasons


def _record(
    row: dict[str, Any],
    index: int,
    source: dict[str, str],
    filter_version: str,
    min_score: float,
) -> dict[str, Any]:
    if _text(row.get("source")) != source["name"]:
        raise PlanError(f"row {index}: source is missing or differs from locked source")
    if _text(row.get("source_revision")) != source["revision"]:
        raise PlanError(
            f"row {index}: source_revision is missing or differs from locked revision"
        )
    image_raw, _ = _first(row, ("image_id", "id", "post_id"))
    tags_raw, tags_field = _first(
        row, ("tags", "tag_string", "tag_string_general", "tag_snapshot")
    )
    rating_raw, _ = _first(row, ("rating", "rating_label"))
    score_raw, _ = _first(row, ("score",))
    opt_raw, _ = _first(row, ("opt_out", "is_opted_out", "opted_out", "opt_out_status"))
    license_raw, _ = _first(row, ("asset_license", "content_license", "license"))
    image_id, normalized_tags = _text(image_raw), _tags(tags_raw)
    rating = RATINGS.get(str(rating_raw).strip().lower(), "unknown")
    score, opted = _number(score_raw), _opt_out(opt_raw)
    decision, reasons = _filter(image_id, normalized_tags, rating, score, opted, min_score)
    key_id = image_id or f"__missing_row_{index}"
    perceptual, _ = _first(row, ("phash", "perceptual_hash"))
    return {
        "record_key": f"{source['name']}::{key_id}",
        "image_id": image_id,
        "source": source["name"],
        "source_revision": source["revision"],
        "source_license": source["license"],
        "asset_license": _text(license_raw) or "unknown",
        "opt_out_status": opted,
        "tag_snapshot": normalized_tags,
        "tag_snapshot_field": tags_field,
        "tag_snapshot_sha256": _digest(_json(normalized_tags)),
        "filter_version": filter_version,
        "rating": rating,
        "score": score,
        "hashes": {
            "md5": _hex(row.get("md5"), HEX32),
            "sha256": _hex(row.get("sha256"), HEX64),
            "phash": _phash(perceptual),
        },
        "filter_decision": decision,
        "filter_reasons": reasons,
        "decision": decision,
        "decision_reasons": list(reasons),
        "duplicate_of": None,
        "phash_distance": None,
        "split": "none",
        "review_selected": False,
    }


def _quality(record: dict[str, Any]) -> tuple[float, str]:
    score = record["score"]
    return (-(score if isinstance(score, (int, float)) else float("-inf")), record["record_key"])


def _phash_bands(value: int) -> Iterator[tuple[int, int]]:
    offset = 0
    for band, width in enumerate((13, 13, 13, 13, 12)):
        yield band, (value >> offset) & ((1 << width) - 1)
        offset += width


def _dedup_split(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    forced_conflicts: dict[str, str] | None = None,
) -> None:
    """Apply exact then pHash dedup; only survivors are split."""

    candidates: list[dict[str, Any]] = []
    forced_conflicts = forced_conflicts or {}
    for record in records:
        record.update(
            decision=record["filter_decision"],
            decision_reasons=list(record["filter_reasons"]),
            duplicate_of=None,
            phash_distance=None,
            split="none",
            review_selected=False,
        )
        if record["filter_decision"] != "DRAFT_ELIGIBLE":
            continue
        if record["record_key"] in forced_conflicts:
            record["decision"] = "HOLD_HASH_CONFLICT"
            record["decision_reasons"].append(
                "hash_conflict:" + forced_conflicts[record["record_key"]]
            )
            continue
        missing = sorted(name for name, value in record["hashes"].items() if value is None)
        if missing:
            record["decision"] = "HOLD_MISSING_DEDUP_HASH"
            record["decision_reasons"].append("missing_dedup_hashes:" + ",".join(missing))
            continue
        candidates.append(record)

    conflicts: set[str] = set()
    for field in ("md5", "sha256"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        other = "sha256" if field == "md5" else "md5"
        for record in candidates:
            groups[record["hashes"][field]].append(record)
        for digest, group in groups.items():
            if len({record["hashes"][other] for record in group}) > 1:
                for record in group:
                    conflicts.add(record["record_key"])
                    record["decision"] = "HOLD_HASH_CONFLICT"
                    record["decision_reasons"].append(
                        f"hash_conflict:{field}_{digest}_maps_to_multiple_{other}"
                    )

    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        if record["record_key"] not in conflicts:
            exact[(record["hashes"]["md5"], record["hashes"]["sha256"])].append(record)
    exact_survivors: list[dict[str, Any]] = []
    for group in exact.values():
        keeper, *duplicates = sorted(group, key=_quality)
        exact_survivors.append(keeper)
        for record in duplicates:
            record["decision"] = "REJECT_DUPLICATE_EXACT"
            record["duplicate_of"] = keeper["record_key"]
            record["decision_reasons"].append("duplicate_exact:" + keeper["record_key"])

    # Five disjoint bands make candidate lookup complete for Hamming distance <=4.
    threshold = policy["phash_distance"]
    buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
    keepers: dict[str, tuple[int, dict[str, Any]]] = {}
    survivors: list[dict[str, Any]] = []
    for record in sorted(exact_survivors, key=_quality):
        value = int(record["hashes"]["phash"], 16)
        possible: set[str] = set()
        for token in _phash_bands(value):
            possible.update(buckets[token])
        hits = sorted(
            ((value ^ keepers[key][0]).bit_count(), key)
            for key in possible
            if (value ^ keepers[key][0]).bit_count() <= threshold
        )
        if hits:
            distance, key = hits[0]
            record["decision"] = "REJECT_DUPLICATE_PHASH"
            record["duplicate_of"] = key
            record["phash_distance"] = distance
            record["decision_reasons"].append(f"duplicate_phash_{distance}:{key}")
            continue
        keepers[record["record_key"]] = (value, record)
        survivors.append(record)
        for token in _phash_bands(value):
            buckets[token].add(record["record_key"])

    order = sorted(
        survivors,
        key=lambda record: _digest(
            (
                policy["split_salt"]
                + "\0"
                + record["record_key"]
                + "\0"
                + record["hashes"]["sha256"]
            ).encode()
        ),
    )
    validation_count = (
        0
        if len(order) < 2
        else min(len(order) - 1, max(1, round(len(order) * policy["validation_ratio"])))
    )
    validation = {record["record_key"] for record in order[:validation_count]}
    for record in survivors:
        record["split"] = "validation" if record["record_key"] in validation else "train"


def _policy_digest(policy: dict[str, Any]) -> str:
    value = copy.deepcopy(policy)
    value.pop("policy_digest", None)
    return _digest(_json(value))


def _policy(args: argparse.Namespace) -> dict[str, Any]:
    changed = (
        args.min_score != DEFAULT_MIN_SCORE
        or args.phash_distance != DEFAULT_PHASH_DISTANCE
        or args.validation_ratio != DEFAULT_VALIDATION_RATIO
        or args.split_salt != DEFAULT_SPLIT_SALT
    )
    requested_version = _text(args.filter_version)
    if changed and (not requested_version or requested_version == FILTER_VERSION):
        raise PlanError(
            "non-default policy parameters require an explicit non-default --filter-version"
        )
    policy = {
        "status": "DRAFT_PENDING_USER_FINAL_REVIEW",
        "filter_version": requested_version or FILTER_VERSION,
        "policy_digest": "",
        "min_score": args.min_score,
        "ratings": {
            "eligible": ["general"],
            "hold": ["sensitive"],
            "reject": ["questionable", "explicit"],
        },
        "hard_blocked_tags": sorted(HARD_TAGS | SEXUAL_TAGS),
        "watchlist_tags": sorted(WATCH_TAGS),
        "phash_distance": args.phash_distance,
        "validation_ratio": args.validation_ratio,
        "split_salt": args.split_salt,
        "review": {"pilot_fraction": 1.0, "later_minimum": 100, "later_fraction": 0.01},
    }
    policy["policy_digest"] = _policy_digest(policy)
    return policy


def _input(
    path: Path, role: str, kind: str, count: int, max_input_records: int
) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "format": kind,
        "sha256": _file_sha(path),
        "record_count": count,
        "max_input_records": max_input_records,
    }


def _review_blank(
    records: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": REVIEW_VERSION,
        "mode": "not_planned",
        "status": "NOT_PLANNED",
        "eligible_population": sum(r["decision"] == "DRAFT_ELIGIBLE" for r in records),
        "minimum_sample": policy["review"]["later_minimum"],
        "sampling_fraction": policy["review"]["later_fraction"],
        "selected_record_keys": [],
        "held_reconsideration_record_keys": sorted(
            r["record_key"] for r in records if r["decision"].startswith("HOLD_")
        ),
        "actual_images_packaged": 0,
    }


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    records, review = manifest["records"], manifest["review_plan"]
    decisions = Counter(record["decision"] for record in records)
    return {
        "input_records": len(records),
        "draft_eligible_records": decisions["DRAFT_ELIGIBLE"],
        "held_records": sum(v for k, v in decisions.items() if k.startswith("HOLD_")),
        "rejected_records": sum(v for k, v in decisions.items() if k.startswith("REJECT_")),
        "train_records": sum(r["split"] == "train" for r in records),
        "validation_records": sum(r["split"] == "validation" for r in records),
        "review_selected_records": len(review["selected_record_keys"]),
        "decision_counts": dict(sorted(decisions.items())),
        "bulk_images_downloaded_by_tool": 0,
    }


def _manifest_identifier(manifest: dict[str, Any]) -> str:
    """Portable semantic identity: absolute paths and serialized-parent SHA are excluded."""

    identity = copy.deepcopy(manifest)
    identity.pop("created_at", None)
    identity.pop("manifest_id", None)
    identity["parent_manifest_sha256"] = None
    identity["source"].pop("metadata_path", None)
    for item in identity["inputs"]:
        item.pop("path", None)
        if item["role"] == "parent_manifest":
            item.pop("sha256", None)
    return "anima-step2a-v1-" + _digest(_json(identity))[:24]


def _finish(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["summary"] = _summary(manifest)
    manifest["manifest_id"] = _manifest_identifier(manifest)
    _validate(manifest)
    return manifest


def _base(
    operation: str,
    source: dict[str, Any],
    inputs: list[dict[str, Any]],
    policy: dict[str, Any],
    records: list[dict[str, Any]],
    parent_sha: str | None,
    parent_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "manifest_version": 1,
        "workflow_status": STATUS,
        "training_ready": False,
        "operation": operation,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_id": "",
        "parent_manifest_sha256": parent_sha,
        "parent_manifest_id": parent_id,
        "source": source,
        "inputs": inputs,
        "policy": policy,
        "stage_order": STAGE_ORDER,
        "network_activity": {
            "network_requests_performed": 0,
            "bulk_image_download_count": 0,
            "bulk_image_download_bytes": 0,
        },
        "review_plan": _review_blank(records, policy),
        "summary": {},
        "records": records,
    }


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PlanError(f"{label}: schema keys mismatch: {actual}")
    return value


def _is_hash(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _review_order(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (record for record in records if record["decision"] == "DRAFT_ELIGIBLE"),
        key=lambda record: _digest(
            (
                policy["split_salt"]
                + "\0review\0"
                + record["record_key"]
                + "\0"
                + record["hashes"]["sha256"]
            ).encode()
        ),
    )


def _validate(manifest: dict[str, Any], check_parent_file: bool = True) -> None:
    top = _exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_version",
            "workflow_status",
            "training_ready",
            "operation",
            "created_at",
            "manifest_id",
            "parent_manifest_sha256",
            "parent_manifest_id",
            "source",
            "inputs",
            "policy",
            "stage_order",
            "network_activity",
            "review_plan",
            "summary",
            "records",
        },
        "manifest",
    )
    if (
        top["schema_version"] != SCHEMA
        or type(top["manifest_version"]) is not int
        or top["manifest_version"] != 1
        or top["workflow_status"] != STATUS
        or top["training_ready"] is not False
        or top["operation"] not in {"plan", "reconcile", "review_package_plan"}
    ):
        raise PlanError("manifest: invalid DRAFT/HOLD v1 header")
    try:
        timestamp = datetime.fromisoformat(str(top["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanError("manifest: invalid created_at") from exc
    if timestamp.tzinfo is None:
        raise PlanError("manifest: created_at must include timezone")
    if top["operation"] == "plan":
        if top["parent_manifest_sha256"] is not None or top["parent_manifest_id"] is not None:
            raise PlanError("manifest: plan must not bind a parent")
    elif not (
        _is_hash(top["parent_manifest_sha256"], HEX64)
        and isinstance(top["parent_manifest_id"], str)
        and re.fullmatch(r"anima-step2a-v1-[a-f0-9]{24}", top["parent_manifest_id"])
    ):
        raise PlanError("manifest: child must bind parent SHA256 and manifest_id")

    source = _exact_keys(
        top["source"],
        {
            "name",
            "revision",
            "license",
            "metadata_path",
            "metadata_sha256",
            "opt_out_contract",
            "license_evidence",
        },
        "source",
    )
    if not all(
        isinstance(source[field], str) and source[field]
        for field in ("name", "license", "metadata_path")
    ):
        raise PlanError("source: name/license/path must be non-empty")
    if not _is_hash(source["revision"], HEX40_64) or not _is_hash(
        source["metadata_sha256"], HEX64
    ):
        raise PlanError("source: revision or metadata SHA256 is invalid")
    if source["opt_out_contract"] != "explicit per record; unknown is held":
        raise PlanError("source: opt-out contract drift")
    evidence = _exact_keys(
        source["license_evidence"],
        {"url", "revision", "sha256", "captured_date"},
        "source.license_evidence",
    )
    if (
        not isinstance(evidence["url"], str)
        or not evidence["url"].startswith("https://")
        or evidence["revision"] != source["revision"]
        or evidence["revision"] not in evidence["url"]
        or not _is_hash(evidence["sha256"], HEX64)
    ):
        raise PlanError("source.license_evidence: URL/revision/SHA binding is invalid")
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(evidence["captured_date"])):
            raise ValueError("not canonical ISO date")
        captured_date = datetime.strptime(
            str(evidence["captured_date"]), "%Y-%m-%d"
        ).date()
    except ValueError as exc:
        raise PlanError("source.license_evidence: captured_date must be YYYY-MM-DD") from exc
    if captured_date > timestamp.date():
        raise PlanError("source.license_evidence: captured_date is after manifest creation")

    inputs = top["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise PlanError("inputs: expected a non-empty array")
    for index, item in enumerate(inputs):
        entry = _exact_keys(
            item,
            {"role", "path", "format", "sha256", "record_count", "max_input_records"},
            f"inputs[{index}]",
        )
        if (
            entry["role"] not in {"metadata", "parent_manifest", "hash_inventory"}
            or entry["format"] not in {"jsonl", "parquet", "json"}
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or not _is_hash(entry["sha256"], HEX64)
            or type(entry["record_count"]) is not int
            or type(entry["max_input_records"]) is not int
            or not 1 <= entry["record_count"] <= entry["max_input_records"] <= MAX_INPUT_RECORDS
        ):
            raise PlanError(f"inputs[{index}]: invalid provenance or record bound")
    metadata_inputs = [item for item in inputs if item["role"] == "metadata"]
    if len(metadata_inputs) != 1 or not any(
        item["path"] == source["metadata_path"]
        and item["sha256"] == source["metadata_sha256"]
        for item in metadata_inputs
    ):
        raise PlanError("inputs: metadata descriptor is not bound to source")
    if top["operation"] != "plan" and not any(
        item["role"] == "parent_manifest"
        and item["sha256"] == top["parent_manifest_sha256"]
        for item in inputs
    ):
        raise PlanError("inputs: child lacks the bound parent descriptor")
    roles = [item["role"] for item in inputs]
    if top["operation"] == "plan" and roles != ["metadata"]:
        raise PlanError("inputs: plan may contain only its metadata descriptor")
    if top["operation"] == "reconcile" and roles[-2:] != [
        "parent_manifest",
        "hash_inventory",
    ]:
        raise PlanError("inputs: reconcile must end with parent + hash inventory")
    if top["operation"] == "review_package_plan" and roles[-1:] != [
        "parent_manifest"
    ]:
        raise PlanError("inputs: review plan must end with its parent descriptor")
    if top["operation"] != "plan" and check_parent_file:
        parent_entry = [item for item in inputs if item["role"] == "parent_manifest"][-1]
        parent_path = Path(parent_entry["path"])
        if (
            parent_entry["sha256"] != top["parent_manifest_sha256"]
            or _file_sha(parent_path) != top["parent_manifest_sha256"]
        ):
            raise PlanError("inputs: immediate parent file SHA256 binding failed")
        try:
            parent_value = json.loads(parent_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanError(f"inputs: cannot decode immediate parent: {exc}") from exc
        if not isinstance(parent_value, dict):
            raise PlanError("inputs: immediate parent root must be an object")
        _validate(parent_value, check_parent_file=False)
        if parent_value["manifest_id"] != top["parent_manifest_id"]:
            raise PlanError("inputs: parent_manifest_id does not match immediate parent")

    policy = _exact_keys(
        top["policy"],
        {
            "status",
            "filter_version",
            "policy_digest",
            "min_score",
            "ratings",
            "hard_blocked_tags",
            "watchlist_tags",
            "phash_distance",
            "validation_ratio",
            "split_salt",
            "review",
        },
        "policy",
    )
    ratings = _exact_keys(policy["ratings"], {"eligible", "hold", "reject"}, "policy.ratings")
    review_policy = _exact_keys(
        policy["review"],
        {"pilot_fraction", "later_minimum", "later_fraction"},
        "policy.review",
    )
    changed = (
        policy["min_score"] != DEFAULT_MIN_SCORE
        or policy["phash_distance"] != DEFAULT_PHASH_DISTANCE
        or policy["validation_ratio"] != DEFAULT_VALIDATION_RATIO
        or policy["split_salt"] != DEFAULT_SPLIT_SALT
    )
    if (
        policy["status"] != "DRAFT_PENDING_USER_FINAL_REVIEW"
        or not isinstance(policy["filter_version"], str)
        or not policy["filter_version"]
        or (changed and policy["filter_version"] == FILTER_VERSION)
        or policy["policy_digest"] != _policy_digest(policy)
        or not isinstance(policy["min_score"], (int, float))
        or isinstance(policy["min_score"], bool)
        or not math.isfinite(policy["min_score"])
        or ratings
        != {
            "eligible": ["general"],
            "hold": ["sensitive"],
            "reject": ["questionable", "explicit"],
        }
        or policy["hard_blocked_tags"] != sorted(HARD_TAGS | SEXUAL_TAGS)
        or policy["watchlist_tags"] != sorted(WATCH_TAGS)
        or type(policy["phash_distance"]) is not int
        or not 0 <= policy["phash_distance"] <= 4
        or not isinstance(policy["validation_ratio"], (int, float))
        or isinstance(policy["validation_ratio"], bool)
        or not 0 < policy["validation_ratio"] <= 0.5
        or not isinstance(policy["split_salt"], str)
        or not policy["split_salt"]
        or review_policy
        != {"pilot_fraction": 1.0, "later_minimum": 100, "later_fraction": 0.01}
    ):
        raise PlanError("policy: schema, digest, version, or parameter binding is invalid")

    if top["stage_order"] != STAGE_ORDER:
        raise PlanError("stage_order: dedup/split order drift")
    network = _exact_keys(
        top["network_activity"],
        {"network_requests_performed", "bulk_image_download_count", "bulk_image_download_bytes"},
        "network_activity",
    )
    if any(type(value) is not int for value in network.values()) or network != {
        "network_requests_performed": 0,
        "bulk_image_download_count": 0,
        "bulk_image_download_bytes": 0,
    }:
        raise PlanError("network_activity: all counters must remain zero")

    records = top["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_INPUT_RECORDS:
        raise PlanError("records: expected 1..100000 rows")
    if not any(item["record_count"] == len(records) for item in metadata_inputs):
        raise PlanError("inputs: metadata record_count does not match manifest records")
    record_fields = {
        "record_key",
        "image_id",
        "source",
        "source_revision",
        "source_license",
        "asset_license",
        "opt_out_status",
        "tag_snapshot",
        "tag_snapshot_field",
        "tag_snapshot_sha256",
        "filter_version",
        "rating",
        "score",
        "hashes",
        "filter_decision",
        "filter_reasons",
        "decision",
        "decision_reasons",
        "duplicate_of",
        "phash_distance",
        "split",
        "review_selected",
    }
    by_key: dict[str, dict[str, Any]] = {}
    for index, record_value in enumerate(records):
        record = _exact_keys(record_value, record_fields, f"records[{index}]")
        key = record["record_key"]
        if not isinstance(key, str) or not key or key in by_key:
            raise PlanError(f"records[{index}]: missing or duplicate record_key")
        by_key[key] = record
        if record["image_id"] is None:
            if not key.startswith(source["name"] + "::__missing_row_"):
                raise PlanError(f"{key}: missing image key is malformed")
        elif not isinstance(record["image_id"], str) or key != (
            source["name"] + "::" + record["image_id"]
        ):
            raise PlanError(f"{key}: record_key/image_id binding failed")
        if (
            record["source"] != source["name"]
            or record["source_revision"] != source["revision"]
            or record["source_license"] != source["license"]
            or record["filter_version"] != policy["filter_version"]
            or not isinstance(record["asset_license"], str)
            or not record["asset_license"]
            or record["opt_out_status"] not in {"not_opted_out", "opted_out", "unknown"}
            or record["rating"]
            not in {"general", "sensitive", "questionable", "explicit", "unknown"}
        ):
            raise PlanError(f"{key}: top-level provenance/filter binding failed")
        tags = record["tag_snapshot"]
        if (
            not isinstance(tags, list)
            or tags != sorted(set(tags))
            or not all(isinstance(tag, str) and tag for tag in tags)
            or record["tag_snapshot_sha256"] != _digest(_json(tags))
            or not isinstance(record["tag_snapshot_field"], str)
            or not record["tag_snapshot_field"]
        ):
            raise PlanError(f"{key}: tag snapshot or tag hash is invalid")
        score = record["score"]
        if score is not None and (
            not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score)
        ):
            raise PlanError(f"{key}: score is invalid")
        hashes = _exact_keys(record["hashes"], {"md5", "sha256", "phash"}, f"{key}.hashes")
        if not (
            hashes["md5"] is None or _is_hash(hashes["md5"], HEX32)
        ) or not (
            hashes["sha256"] is None or _is_hash(hashes["sha256"], HEX64)
        ) or not (
            hashes["phash"] is None or _is_hash(hashes["phash"], PHASH)
        ):
            raise PlanError(f"{key}: hash shape is invalid")
        expected_filter, expected_reasons = _filter(
            record["image_id"],
            tags,
            record["rating"],
            score,
            record["opt_out_status"],
            policy["min_score"],
        )
        if (
            record["filter_decision"] != expected_filter
            or record["filter_reasons"] != expected_reasons
            or record["filter_decision"] not in FILTER_DECISIONS
            or record["decision"] not in FINAL_DECISIONS
            or not isinstance(record["decision_reasons"], list)
            or not all(isinstance(reason, str) and reason for reason in record["decision_reasons"])
            or type(record["review_selected"]) is not bool
        ):
            raise PlanError(f"{key}: filter/decision semantics are invalid")
        if expected_filter != "DRAFT_ELIGIBLE" and record["decision"] != expected_filter:
            raise PlanError(f"{key}: filtered record decision was altered")
        if expected_filter == "DRAFT_ELIGIBLE" and record["decision"] not in {
            "DRAFT_ELIGIBLE",
            "HOLD_MISSING_DEDUP_HASH",
            "HOLD_HASH_CONFLICT",
            "REJECT_DUPLICATE_EXACT",
            "REJECT_DUPLICATE_PHASH",
        }:
            raise PlanError(f"{key}: eligible filter result has impossible final decision")
        if record["decision"] == expected_filter and record["decision_reasons"] != expected_reasons:
            raise PlanError(f"{key}: decision reasons drifted")
        if record["decision"] == "DRAFT_ELIGIBLE":
            if not (
                _is_hash(hashes["md5"], HEX32)
                and _is_hash(hashes["sha256"], HEX64)
                and _is_hash(hashes["phash"], PHASH)
                and record["split"] in {"train", "validation"}
                and record["duplicate_of"] is None
                and record["phash_distance"] is None
            ):
                raise PlanError(f"{key}: eligible record lacks hashes/post-dedup split")
        elif record["split"] != "none":
            raise PlanError(f"{key}: non-eligible record appears in split")
        if record["phash_distance"] is not None and type(
            record["phash_distance"]
        ) is not int:
            raise PlanError(f"{key}: pHash distance must be an integer")
        if record["decision"] == "HOLD_MISSING_DEDUP_HASH":
            missing = sorted(name for name, value in hashes.items() if value is None)
            if not missing or record["decision_reasons"] != expected_reasons + [
                "missing_dedup_hashes:" + ",".join(missing)
            ]:
                raise PlanError(f"{key}: missing-hash hold semantics are invalid")
        if record["decision"] == "HOLD_HASH_CONFLICT" and not any(
            reason.startswith("hash_conflict:") for reason in record["decision_reasons"]
        ):
            raise PlanError(f"{key}: hash-conflict hold lacks provenance")

    for key, record in by_key.items():
        duplicate = record["duplicate_of"]
        if record["decision"] in {"REJECT_DUPLICATE_EXACT", "REJECT_DUPLICATE_PHASH"}:
            if duplicate not in by_key or duplicate == key:
                raise PlanError(f"{key}: duplicate reference is invalid")
            target = by_key[duplicate]
            if record["decision"] == "REJECT_DUPLICATE_PHASH" and target[
                "decision"
            ] != "DRAFT_ELIGIBLE":
                raise PlanError(f"{key}: duplicate target is not eligible")
            if record["decision"] == "REJECT_DUPLICATE_EXACT" and target[
                "decision"
            ] not in {"DRAFT_ELIGIBLE", "REJECT_DUPLICATE_PHASH"}:
                raise PlanError(f"{key}: exact duplicate target is invalid")
            if record["decision"] == "REJECT_DUPLICATE_EXACT" and (
                record["hashes"]["md5"] != target["hashes"]["md5"]
                or record["hashes"]["sha256"] != target["hashes"]["sha256"]
                or record["phash_distance"] is not None
                or record["decision_reasons"]
                != record["filter_reasons"] + ["duplicate_exact:" + duplicate]
            ):
                raise PlanError(f"{key}: exact duplicate binding is invalid")
            if record["decision"] == "REJECT_DUPLICATE_PHASH":
                if not (
                    _is_hash(record["hashes"]["phash"], PHASH)
                    and _is_hash(target["hashes"]["phash"], PHASH)
                ):
                    raise PlanError(f"{key}: pHash duplicate lacks hashes")
                distance = (
                    int(record["hashes"]["phash"], 16)
                    ^ int(target["hashes"]["phash"], 16)
                ).bit_count()
                if (
                    record["phash_distance"] != distance
                    or distance > policy["phash_distance"]
                    or record["decision_reasons"]
                    != record["filter_reasons"]
                    + [f"duplicate_phash_{distance}:{duplicate}"]
                ):
                    raise PlanError(f"{key}: pHash duplicate distance is invalid")
        elif duplicate is not None or record["phash_distance"] is not None:
            raise PlanError(f"{key}: non-duplicate carries duplicate metadata")

    eligible = [record for record in records if record["decision"] == "DRAFT_ELIGIBLE"]
    exact_seen: set[tuple[str, str]] = set()
    phash_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for record in eligible:
        exact_key = (record["hashes"]["md5"], record["hashes"]["sha256"])
        if exact_key in exact_seen:
            raise PlanError("eligible records contain an exact duplicate")
        exact_seen.add(exact_key)
        value = int(record["hashes"]["phash"], 16)
        possible = {
            prior
            for token in _phash_bands(value)
            for prior in phash_buckets[token]
        }
        if any((value ^ prior).bit_count() <= policy["phash_distance"] for prior in possible):
            raise PlanError("eligible records contain a pHash near duplicate")
        for token in _phash_bands(value):
            phash_buckets[token].append(value)
    split_order = sorted(
        eligible,
        key=lambda record: _digest(
            (
                policy["split_salt"]
                + "\0"
                + record["record_key"]
                + "\0"
                + record["hashes"]["sha256"]
            ).encode()
        ),
    )
    validation_count = (
        0
        if len(split_order) < 2
        else min(
            len(split_order) - 1,
            max(1, round(len(split_order) * policy["validation_ratio"])),
        )
    )
    expected_validation = {record["record_key"] for record in split_order[:validation_count]}
    if any(
        record["split"]
        != ("validation" if record["record_key"] in expected_validation else "train")
        for record in eligible
    ):
        raise PlanError("records: train/validation split is not reproducible")

    review = _exact_keys(
        top["review_plan"],
        {
            "version",
            "mode",
            "status",
            "eligible_population",
            "minimum_sample",
            "sampling_fraction",
            "selected_record_keys",
            "held_reconsideration_record_keys",
            "actual_images_packaged",
        },
        "review_plan",
    )
    selected = review["selected_record_keys"]
    held = review["held_reconsideration_record_keys"]
    eligible_keys = {record["record_key"] for record in eligible}
    expected_held = sorted(
        record["record_key"] for record in records if record["decision"].startswith("HOLD_")
    )
    if (
        review["version"] != REVIEW_VERSION
        or review["mode"] not in {"not_planned", "pilot", "batch"}
        or review["status"] not in {"NOT_PLANNED", "PLAN_ONLY_NO_IMAGES"}
        or type(review["eligible_population"]) is not int
        or review["eligible_population"] != len(eligible)
        or type(review["minimum_sample"]) is not int
        or review["minimum_sample"] < 0
        or not isinstance(review["sampling_fraction"], (int, float))
        or isinstance(review["sampling_fraction"], bool)
        or not 0 < review["sampling_fraction"] <= 1
        or not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or not set(selected) <= eligible_keys
        or held != expected_held
        or type(review["actual_images_packaged"]) is not int
        or review["actual_images_packaged"] != 0
        or any(
            record["review_selected"] != (record["record_key"] in set(selected))
            for record in records
        )
    ):
        raise PlanError("review_plan: schema, references, or record binding is invalid")
    ordered_review = _review_order(records, policy)
    if review["mode"] == "not_planned":
        if (
            review["status"] != "NOT_PLANNED"
            or selected
            or review["minimum_sample"] != policy["review"]["later_minimum"]
            or review["sampling_fraction"] != policy["review"]["later_fraction"]
        ):
            raise PlanError("review_plan: not_planned parameters drifted")
    elif review["mode"] == "pilot":
        if (
            review["status"] != "PLAN_ONLY_NO_IMAGES"
            or review["minimum_sample"] != len(eligible)
            or review["sampling_fraction"] != 1.0
            or selected != [record["record_key"] for record in ordered_review]
        ):
            raise PlanError("review_plan: pilot is not a full deterministic review")
    else:
        count = min(
            len(ordered_review),
            max(
                review["minimum_sample"],
                math.ceil(len(ordered_review) * review["sampling_fraction"]),
            ),
        )
        if (
            review["status"] != "PLAN_ONLY_NO_IMAGES"
            or selected != [record["record_key"] for record in ordered_review[:count]]
        ):
            raise PlanError("review_plan: batch sample is not reproducible")

    summary = _exact_keys(
        top["summary"],
        {
            "input_records",
            "draft_eligible_records",
            "held_records",
            "rejected_records",
            "train_records",
            "validation_records",
            "review_selected_records",
            "decision_counts",
            "bulk_images_downloaded_by_tool",
        },
        "summary",
    )
    integer_summary_fields = {
        "input_records",
        "draft_eligible_records",
        "held_records",
        "rejected_records",
        "train_records",
        "validation_records",
        "review_selected_records",
        "bulk_images_downloaded_by_tool",
    }
    if (
        any(type(summary[field]) is not int for field in integer_summary_fields)
        or not isinstance(summary["decision_counts"], dict)
        or any(type(value) is not int for value in summary["decision_counts"].values())
        or summary != _summary(top)
    ):
        raise PlanError("summary: counts do not match records/review plan")
    if top["manifest_id"] != _manifest_identifier(top):
        raise PlanError("manifest_id: portable semantic identity mismatch")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot decode manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError("manifest root must be an object")
    try:
        _validate(value)
    except PlanError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PlanError(f"manifest semantic validation failed closed: {exc}") from exc
    return value


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.input)
    rows, kind = _read_rows(path, args.max_input_records)
    policy = _policy(args)
    if args.license_evidence_revision != args.source_revision:
        raise PlanError("license evidence revision must equal --source-revision")
    source = {
        "name": args.source,
        "revision": args.source_revision,
        "license": args.source_license,
        "metadata_path": str(path.resolve()),
        "metadata_sha256": _file_sha(path),
        "opt_out_contract": "explicit per record; unknown is held",
        "license_evidence": {
            "url": args.license_evidence_url,
            "revision": args.license_evidence_revision,
            "sha256": args.license_evidence_sha256,
            "captured_date": args.license_evidence_date,
        },
    }
    records = [
        _record(row, index, source, policy["filter_version"], policy["min_score"])
        for index, row in enumerate(rows, 1)
    ]
    if len({record["record_key"] for record in records}) != len(records):
        raise PlanError("duplicate source/image_id keys make reconciliation ambiguous")
    _dedup_split(records, policy)
    return _finish(
        _base(
            "plan",
            source,
            [_input(path, "metadata", kind, len(rows), args.max_input_records)],
            policy,
            records,
            None,
            None,
        )
    )


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, inventory_path = Path(args.manifest), Path(args.inventory)
    parent = _load_manifest(manifest_path)
    rows, kind = _read_rows(inventory_path, args.max_input_records)
    source_name = parent["source"]["name"]
    source_revision = parent["source"]["revision"]
    inventory: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, 1):
        image_raw, _ = _first(row, ("image_id", "id", "post_id"))
        image_id = _text(image_raw)
        if (
            not image_id
            or _text(row.get("source")) != source_name
            or _text(row.get("source_revision")) != source_revision
        ):
            raise PlanError(
                f"inventory row {index} lacks exact source/source_revision/image_id binding"
            )
        key = f"{source_name}::{image_id}"
        if key in inventory:
            raise PlanError(f"duplicate inventory key: {key}")
        phash_raw, _ = _first(row, ("phash", "perceptual_hash"))
        hashes = {
            "md5": _hex(row.get("md5"), HEX32),
            "sha256": _hex(row.get("sha256"), HEX64),
            "phash": _phash(phash_raw),
        }
        if any(value is None for value in hashes.values()):
            raise PlanError(f"{key}: inventory requires valid MD5, SHA256, and pHash")
        inventory[key] = hashes  # type: ignore[assignment]

    records = copy.deepcopy(parent["records"])
    known = {record["record_key"] for record in records}
    if set(inventory) - known:
        raise PlanError("inventory contains keys absent from parent manifest")
    forced: dict[str, str] = {}
    for record in records:
        incoming = inventory.get(record["record_key"])
        if not incoming:
            continue
        for field in ("md5", "sha256", "phash"):
            old = record["hashes"].get(field)
            if old and old != incoming[field]:
                forced[record["record_key"]] = f"metadata_{field}_does_not_match_inventory"
        record["hashes"] = incoming
    _dedup_split(records, parent["policy"], forced)
    inputs = copy.deepcopy(parent["inputs"]) + [
        _input(
            manifest_path,
            "parent_manifest",
            "json",
            len(parent["records"]),
            MAX_INPUT_RECORDS,
        ),
        _input(
            inventory_path,
            "hash_inventory",
            kind,
            len(rows),
            args.max_input_records,
        ),
    ]
    return _finish(
        _base(
            "reconcile",
            copy.deepcopy(parent["source"]),
            inputs,
            copy.deepcopy(parent["policy"]),
            records,
            _file_sha(manifest_path),
            parent["manifest_id"],
        )
    )


def review_package(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.manifest)
    parent = _load_manifest(path)
    records = copy.deepcopy(parent["records"])
    eligible = [record for record in records if record["decision"] == "DRAFT_ELIGIBLE"]
    if not eligible:
        raise PlanError("zero post-dedup eligible records; reconcile hashes first")
    ordered = _review_order(records, parent["policy"])
    if args.mode == "pilot":
        selected = ordered
        minimum_sample = len(ordered)
        sampling_fraction = 1.0
    else:
        count = max(args.minimum_sample, math.ceil(len(ordered) * args.sampling_fraction))
        selected = ordered[: min(len(ordered), count)]
        minimum_sample = args.minimum_sample
        sampling_fraction = args.sampling_fraction
    selected_keys = {record["record_key"] for record in selected}
    for record in records:
        record["review_selected"] = record["record_key"] in selected_keys
    result = _base(
        "review_package_plan",
        copy.deepcopy(parent["source"]),
        copy.deepcopy(parent["inputs"])
        + [
            _input(
                path,
                "parent_manifest",
                "json",
                len(records),
                MAX_INPUT_RECORDS,
            )
        ],
        copy.deepcopy(parent["policy"]),
        records,
        _file_sha(path),
        parent["manifest_id"],
    )
    result["review_plan"] = {
        "version": REVIEW_VERSION,
        "mode": args.mode,
        "status": "PLAN_ONLY_NO_IMAGES",
        "eligible_population": len(eligible),
        "minimum_sample": minimum_sample,
        "sampling_fraction": sampling_fraction,
        "selected_record_keys": [record["record_key"] for record in selected],
        "held_reconsideration_record_keys": sorted(
            record["record_key"] for record in records if record["decision"].startswith("HOLD_")
        ),
        "actual_images_packaged": 0,
    }
    return _finish(result)


def _revision(value: str) -> str:
    value = value.strip().lower()
    if not HEX40_64.fullmatch(value):
        raise argparse.ArgumentTypeError("revision must be immutable 40-64 hex")
    return value


def _ratio(value: str) -> float:
    try:
        value_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio must be numeric") from exc
    if not 0 < value_float <= 0.5:
        raise argparse.ArgumentTypeError("ratio must be in (0, 0.5]")
    return value_float


def _fraction(value: str) -> float:
    try:
        value_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fraction must be numeric") from exc
    if not 0 < value_float <= 1:
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return value_float


def _record_limit(value: str) -> int:
    try:
        value_int = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("record limit must be an integer") from exc
    if not 1 <= value_int <= MAX_INPUT_RECORDS:
        raise argparse.ArgumentTypeError(
            f"record limit must be in [1, {MAX_INPUT_RECORDS}]"
        )
    return value_int


def _sha256_arg(value: str) -> str:
    value = value.strip().lower()
    if not HEX64.fullmatch(value):
        raise argparse.ArgumentTypeError("value must be a 64-character SHA256")
    return value


def _https_url(value: str) -> str:
    value = value.strip()
    if not value.startswith("https://"):
        raise argparse.ArgumentTypeError("license evidence URL must use https://")
    return value


def _date_arg(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Offline Anima Step 2a DRAFT/HOLD metadata and review planner"
    )
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--input", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--source", required=True)
    plan.add_argument("--source-revision", required=True, type=_revision)
    plan.add_argument("--source-license", required=True)
    plan.add_argument("--license-evidence-url", required=True, type=_https_url)
    plan.add_argument("--license-evidence-revision", required=True, type=_revision)
    plan.add_argument("--license-evidence-sha256", required=True, type=_sha256_arg)
    plan.add_argument("--license-evidence-date", required=True, type=_date_arg)
    plan.add_argument("--filter-version")
    plan.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    plan.add_argument(
        "--phash-distance",
        type=int,
        choices=range(5),
        default=DEFAULT_PHASH_DISTANCE,
    )
    plan.add_argument(
        "--validation-ratio",
        type=_ratio,
        default=DEFAULT_VALIDATION_RATIO,
    )
    plan.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    plan.add_argument(
        "--max-input-records",
        type=_record_limit,
        default=DEFAULT_MAX_INPUT_RECORDS,
    )
    plan.set_defaults(handler=build_plan)

    rec = commands.add_parser("reconcile")
    rec.add_argument("--manifest", required=True)
    rec.add_argument("--inventory", required=True)
    rec.add_argument("--output", required=True)
    rec.add_argument(
        "--max-input-records",
        type=_record_limit,
        default=DEFAULT_MAX_INPUT_RECORDS,
    )
    rec.set_defaults(handler=reconcile)

    review = commands.add_parser("review-package")
    review.add_argument("--manifest", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--mode", choices=("pilot", "batch"), required=True)
    review.add_argument("--minimum-sample", type=int, default=100)
    review.add_argument("--sampling-fraction", type=_fraction, default=0.01)
    review.set_defaults(handler=review_package)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        if not all(
            str(value).strip()
            for value in (args.source, args.source_license, args.split_salt)
        ):
            print("ERROR: fail-closed: source/license/split fields must be non-empty", file=sys.stderr)
            return 2
        if not math.isfinite(args.min_score):
            print("ERROR: fail-closed: min-score must be finite", file=sys.stderr)
            return 2
    if args.command == "review-package" and args.minimum_sample < 100:
        print("ERROR: fail-closed: minimum review sample must be at least 100", file=sys.stderr)
        return 2
    try:
        payload = args.handler(args)
        _write_no_clobber(Path(args.output), payload)
    except PlanError as exc:
        print(f"ERROR: fail-closed: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {Path(args.output).resolve()} "
        f"(DRAFT/HOLD; bulk_image_downloads=0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
