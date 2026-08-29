#!/usr/bin/env python3
"""Approved-policy, offline dataset planner for Anima Step 2a.

This v2 contract reads only local metadata, hash inventories, approval
evidence, proposals, and prior manifests. It has no downloader or HTTP client
and records tool activity separately from externally acquired inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import anima_step2a_dataset as v1

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "anima-step2a-dataset-manifest-v2"
STATUS = "APPROVED_POLICY_PENDING_SOURCE_ACCESS_AND_G1_REVIEW"
FILTER_VERSION = "anima-step2a-pilot-filter-v1"
REVIEW_VERSION = "anima-step2a-review-package-plan-v2"
ELIGIBLE = "POLICY_ELIGIBLE_PENDING_RECONCILIATION_REVIEW"
V1_ELIGIBLE = "DRAFT" + "_ELIGIBLE"
DEFAULT_MIN_SCORE = 10.0
DEFAULT_PHASH_DISTANCE = 4
DEFAULT_VALIDATION_RATIO = 0.05
DEFAULT_SPLIT_SALT = "anima-step2a-pilot-v1"
DEFAULT_MAX_INPUT_RECORDS = v1.DEFAULT_MAX_INPUT_RECORDS
MAX_INPUT_RECORDS = v1.MAX_INPUT_RECORDS
STAGE_ORDER = list(v1.STAGE_ORDER)

APPROVAL_SCHEMA = "anima-step2a-policy-approval-record-v1"
APPROVAL_ID = "anima-step2a-policy-approval-20260718-v1"
APPROVED_DATE = "2026-07-18"
APPROVAL_SOURCE = {
    "user_message": "next",
    "context": (
        "User instructed continuation immediately after receiving the exact Phase A "
        "policy checklist and the metadata-only next-step boundary."
    ),
}
PROPOSAL_RELATIVE_PATH = "docs/lora_training/anima_step2a/COMPLIANCE_PROPOSAL_APPROVAL_SNAPSHOT_2026-07-18.md"
PROPOSAL_SHA256 = "4c375d021f883755fe8ed05d4865eddf3983bebe7d67500a81565f49f2cfa018"
APPROVED_POLICY_DIGEST = "b037bd6804b08bceb9684365509a2784f2b1a66cf34a64da240fd79546cb0bc9"
APPROVED_SCOPE = {
    "phase_a_policy_approved": True,
    "bounded_metadata_pilot_authorized": True,
    "bulk_image_download_authorized": False,
    "pilot_scale_up_authorized": False,
    "phase_d_training_authorized": False,
    "g4_release_authorized": False,
}
APPROVED_SOURCE_NAME = "deepghs/danbooru2024-webp-4Mpixel"
APPROVED_SOURCE_REVISION = "60ca7e02a844bccbe6c8344d1a18b7ea280b787f"
APPROVED_SOURCE_LICENSE = "MIT License-Minimal-Modified; original-repository opt-out applies"
APPROVED_LICENSE_EVIDENCE_URL = (
    f"https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel/blob/{APPROVED_SOURCE_REVISION}/README.md"
)
APPROVED_LICENSE_EVIDENCE_SHA256 = "d2be56b31ac956601cb3880e940461524377dcdcc5768e43e0642ed89091636b"
APPROVED_LICENSE_EVIDENCE_DATE = "2026-07-18"

HARD_TAGS = v1.HARD_TAGS
SEXUAL_TAGS = v1.SEXUAL_TAGS
WATCH_TAGS = v1.WATCH_TAGS
FILTER_DECISIONS = (v1.FILTER_DECISIONS - {V1_ELIGIBLE}) | {ELIGIBLE}
FINAL_DECISIONS = (v1.FINAL_DECISIONS - {V1_ELIGIBLE}) | {ELIGIBLE}
HEX32 = v1.HEX32
HEX40_64 = v1.HEX40_64
HEX64 = v1.HEX64
PHASH = v1.PHASH
PlanError = v1.PlanError

_json = v1._json
_digest = v1._digest
_file_sha = v1._file_sha
_write_no_clobber = v1._write_no_clobber
_read_rows = v1._read_rows
_input = v1._input
_first = v1._first
_text = v1._text
_hex = v1._hex
_phash = v1._phash
_exact_keys = v1._exact_keys
_is_hash = v1._is_hash


def _policy_digest(policy: dict[str, Any]) -> str:
    value = copy.deepcopy(policy)
    value["policy_digest"] = ""
    return _digest(_json(value))


def _policy() -> dict[str, Any]:
    policy = {
        "status": "APPROVED_BY_USER",
        "filter_version": FILTER_VERSION,
        "policy_digest": "",
        "min_score": DEFAULT_MIN_SCORE,
        "ratings": {
            "eligible": ["general"],
            "hold": ["sensitive"],
            "reject": ["questionable", "explicit"],
        },
        "hard_blocked_tags": sorted(HARD_TAGS | SEXUAL_TAGS),
        "watchlist_tags": sorted(WATCH_TAGS),
        "phash_distance": DEFAULT_PHASH_DISTANCE,
        "validation_ratio": DEFAULT_VALIDATION_RATIO,
        "split_salt": DEFAULT_SPLIT_SALT,
        "review": {
            "pilot_fraction": 1.0,
            "later_minimum": 100,
            "later_fraction": 0.01,
        },
    }
    policy["policy_digest"] = _policy_digest(policy)
    if policy["policy_digest"] != APPROVED_POLICY_DIGEST:
        raise PlanError("approved policy constants do not reproduce the frozen digest")
    return policy


def _approval_record(value: Any) -> dict[str, Any]:
    record = _exact_keys(
        value,
        {
            "schema",
            "approval_id",
            "approved_date",
            "approval_source",
            "proposal",
            "approved_policy",
            "scope",
        },
        "approval_record",
    )
    source = _exact_keys(
        record["approval_source"],
        {"user_message", "context"},
        "approval_record.approval_source",
    )
    proposal = _exact_keys(
        record["proposal"],
        {"path", "sha256"},
        "approval_record.proposal",
    )
    approved_policy = _exact_keys(
        record["approved_policy"],
        {"filter_version", "policy_digest"},
        "approval_record.approved_policy",
    )
    scope = _exact_keys(
        record["scope"],
        set(APPROVED_SCOPE),
        "approval_record.scope",
    )
    if (
        record["schema"] != APPROVAL_SCHEMA
        or record["approval_id"] != APPROVAL_ID
        or record["approved_date"] != APPROVED_DATE
        or source != APPROVAL_SOURCE
        or proposal
        != {
            "path": PROPOSAL_RELATIVE_PATH,
            "sha256": PROPOSAL_SHA256,
        }
        or approved_policy
        != {
            "filter_version": FILTER_VERSION,
            "policy_digest": APPROVED_POLICY_DIGEST,
        }
        or scope != APPROVED_SCOPE
        or any(type(scope[key]) is not bool for key in APPROVED_SCOPE)
    ):
        raise PlanError("approval_record: frozen approval semantics do not match")
    if _policy()["policy_digest"] != approved_policy["policy_digest"]:
        raise PlanError("approval_record: approved policy digest does not reproduce")
    return record


def _load_approval(record_path: Path, proposal_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot decode approval record {record_path}: {exc}") from exc
    record = _approval_record(value)
    expected_proposal = (ROOT / record["proposal"]["path"]).resolve()
    actual_proposal = proposal_path.resolve()
    if actual_proposal != expected_proposal:
        raise PlanError("proposal path differs from the project-relative path in the approval record")
    if _file_sha(actual_proposal) != record["proposal"]["sha256"]:
        raise PlanError("proposal SHA256 differs from the approved proposal")
    return {
        "record_schema": record["schema"],
        "approval_id": record["approval_id"],
        "approved_date": record["approved_date"],
        "record_path": str(record_path.resolve()),
        "record_sha256": _file_sha(record_path),
        "proposal_path": str(actual_proposal),
        "proposal_sha256": record["proposal"]["sha256"],
        "policy_digest": record["approved_policy"]["policy_digest"],
    }


def _validate_approved_source_contract(
    source: Any,
    context: str,
) -> None:
    evidence = source.get("license_evidence") if isinstance(source, dict) else None
    actual = {
        "name": source.get("name") if isinstance(source, dict) else None,
        "revision": source.get("revision") if isinstance(source, dict) else None,
        "license": source.get("license") if isinstance(source, dict) else None,
        "license_evidence": {
            "url": evidence.get("url") if isinstance(evidence, dict) else None,
            "revision": (evidence.get("revision") if isinstance(evidence, dict) else None),
            "sha256": evidence.get("sha256") if isinstance(evidence, dict) else None,
            "captured_date": (evidence.get("captured_date") if isinstance(evidence, dict) else None),
        },
    }
    expected = {
        "name": APPROVED_SOURCE_NAME,
        "revision": APPROVED_SOURCE_REVISION,
        "license": APPROVED_SOURCE_LICENSE,
        "license_evidence": {
            "url": APPROVED_LICENSE_EVIDENCE_URL,
            "revision": APPROVED_SOURCE_REVISION,
            "sha256": APPROVED_LICENSE_EVIDENCE_SHA256,
            "captured_date": APPROVED_LICENSE_EVIDENCE_DATE,
        },
    }
    if actual != expected:
        raise PlanError(f"{context}: source and README evidence must match the approved snapshot")


def _filter(
    image_id: str | None,
    tags: list[str],
    rating: str,
    score: float | None,
    opted: str,
    min_score: float,
) -> tuple[str, list[str]]:
    decision, reasons = v1._filter(image_id, tags, rating, score, opted, min_score)
    return (ELIGIBLE if decision == V1_ELIGIBLE else decision), reasons


def _record(
    row: dict[str, Any],
    index: int,
    source: dict[str, str],
    filter_version: str,
    min_score: float,
) -> dict[str, Any]:
    record = v1._record(row, index, source, filter_version, min_score)
    if record["filter_decision"] == V1_ELIGIBLE:
        record["filter_decision"] = ELIGIBLE
        record["decision"] = ELIGIBLE
    return record


def _dedup_split(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    forced_conflicts: dict[str, str] | None = None,
) -> None:
    for record in records:
        if record["filter_decision"] == ELIGIBLE:
            record["filter_decision"] = V1_ELIGIBLE
        if record["decision"] == ELIGIBLE:
            record["decision"] = V1_ELIGIBLE
    try:
        v1._dedup_split(records, policy, forced_conflicts)
    finally:
        for record in records:
            if record["filter_decision"] == V1_ELIGIBLE:
                record["filter_decision"] = ELIGIBLE
            if record["decision"] == V1_ELIGIBLE:
                record["decision"] = ELIGIBLE


def _external_inputs(path: Path) -> dict[str, Any]:
    try:
        metadata_bytes = path.stat().st_size
    except OSError as exc:
        raise PlanError(f"cannot stat metadata input {path}: {exc}") from exc
    return {
        "metadata": {
            "path": str(path.resolve()),
            "sha256": _file_sha(path),
            "bytes": metadata_bytes,
            "acquired_by_tool": False,
        },
        "images": {
            "count": 0,
            "bytes": 0,
            "acquired_by_tool": False,
        },
    }


def _review_blank(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": REVIEW_VERSION,
        "mode": "not_planned",
        "status": "NOT_PLANNED",
        "eligible_population": sum(record["decision"] == ELIGIBLE for record in records),
        "minimum_sample": policy["review"]["later_minimum"],
        "sampling_fraction": policy["review"]["later_fraction"],
        "selected_record_keys": [],
        "held_reconsideration_record_keys": sorted(
            record["record_key"] for record in records if record["decision"].startswith("HOLD_")
        ),
        "actual_images_packaged": 0,
    }


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    review = manifest["review_plan"]
    decisions = Counter(record["decision"] for record in records)
    return {
        "input_records": len(records),
        "policy_eligible_records": decisions[ELIGIBLE],
        "held_records": sum(value for key, value in decisions.items() if key.startswith("HOLD_")),
        "rejected_records": sum(value for key, value in decisions.items() if key.startswith("REJECT_")),
        "train_records": sum(record["split"] == "train" for record in records),
        "validation_records": sum(record["split"] == "validation" for record in records),
        "review_selected_records": len(review["selected_record_keys"]),
        "decision_counts": dict(sorted(decisions.items())),
    }


def _manifest_identifier(manifest: dict[str, Any]) -> str:
    identity = copy.deepcopy(manifest)
    identity.pop("created_at", None)
    identity.pop("manifest_id", None)
    identity["parent_manifest_sha256"] = None
    identity["source"].pop("metadata_path", None)
    identity["approval"].pop("record_path", None)
    identity["approval"].pop("proposal_path", None)
    identity["external_inputs"]["metadata"].pop("path", None)
    for item in identity["inputs"]:
        item.pop("path", None)
        if item["role"] == "parent_manifest":
            item.pop("sha256", None)
    return "anima-step2a-v2-" + _digest(_json(identity))[:24]


def _base(
    operation: str,
    source: dict[str, Any],
    inputs: list[dict[str, Any]],
    approval: dict[str, Any],
    external_inputs: dict[str, Any],
    policy: dict[str, Any],
    records: list[dict[str, Any]],
    parent_sha: str | None,
    parent_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "manifest_version": 2,
        "workflow_status": STATUS,
        "training_ready": False,
        "operation": operation,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_id": "",
        "parent_manifest_sha256": parent_sha,
        "parent_manifest_id": parent_id,
        "approval": approval,
        "source": source,
        "inputs": inputs,
        "policy": policy,
        "stage_order": STAGE_ORDER,
        "tool_activity": {
            "network_requests_performed": 0,
            "metadata_download_count": 0,
            "metadata_download_bytes": 0,
            "image_download_count": 0,
            "image_download_bytes": 0,
        },
        "external_inputs": external_inputs,
        "review_plan": _review_blank(records, policy),
        "summary": {},
        "records": records,
    }


def _to_v1_proxy(manifest: dict[str, Any]) -> dict[str, Any]:
    proxy = copy.deepcopy(manifest)
    proxy.pop("approval")
    proxy.pop("external_inputs")
    tool_activity = proxy.pop("tool_activity")
    proxy["schema_version"] = v1.SCHEMA
    proxy["manifest_version"] = 1
    proxy["workflow_status"] = v1.STATUS
    proxy["network_activity"] = {
        "network_requests_performed": tool_activity["network_requests_performed"],
        "bulk_image_download_count": tool_activity["image_download_count"],
        "bulk_image_download_bytes": tool_activity["image_download_bytes"],
    }
    proxy["policy"]["status"] = "DRAFT_PENDING_USER_FINAL_REVIEW"
    proxy["policy"]["policy_digest"] = v1._policy_digest(proxy["policy"])
    proxy["review_plan"]["version"] = v1.REVIEW_VERSION
    for record in proxy["records"]:
        if record["filter_decision"] == ELIGIBLE:
            record["filter_decision"] = V1_ELIGIBLE
        if record["decision"] == ELIGIBLE:
            record["decision"] = V1_ELIGIBLE
    if proxy["parent_manifest_id"] is not None:
        suffix = proxy["parent_manifest_id"].removeprefix("anima-step2a-v2-")
        proxy["parent_manifest_id"] = "anima-step2a-v1-" + suffix
    proxy["summary"] = v1._summary(proxy)
    proxy["manifest_id"] = v1._manifest_identifier(proxy)
    return proxy


def _validate_approval_binding(value: Any) -> dict[str, Any]:
    approval = _exact_keys(
        value,
        {
            "record_schema",
            "approval_id",
            "approved_date",
            "record_path",
            "record_sha256",
            "proposal_path",
            "proposal_sha256",
            "policy_digest",
        },
        "approval",
    )
    if (
        approval["record_schema"] != APPROVAL_SCHEMA
        or approval["approval_id"] != APPROVAL_ID
        or approval["approved_date"] != APPROVED_DATE
        or not isinstance(approval["record_path"], str)
        or not approval["record_path"]
        or not _is_hash(approval["record_sha256"], HEX64)
        or not isinstance(approval["proposal_path"], str)
        or not approval["proposal_path"]
        or approval["proposal_sha256"] != PROPOSAL_SHA256
        or approval["policy_digest"] != APPROVED_POLICY_DIGEST
    ):
        raise PlanError("approval: schema, identity, path, or digest binding is invalid")
    loaded = _load_approval(
        Path(approval["record_path"]),
        Path(approval["proposal_path"]),
    )
    if loaded != approval:
        raise PlanError("approval: manifest binding differs from the approval record")
    return approval


def _validate_external_inputs(
    value: Any,
    source: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> None:
    external = _exact_keys(value, {"metadata", "images"}, "external_inputs")
    metadata = _exact_keys(
        external["metadata"],
        {"path", "sha256", "bytes", "acquired_by_tool"},
        "external_inputs.metadata",
    )
    images = _exact_keys(
        external["images"],
        {"count", "bytes", "acquired_by_tool"},
        "external_inputs.images",
    )
    metadata_inputs = [item for item in inputs if item.get("role") == "metadata"]
    if (
        len(metadata_inputs) != 1
        or metadata["path"] != source["metadata_path"]
        or metadata["sha256"] != source["metadata_sha256"]
        or metadata["path"] != metadata_inputs[0].get("path")
        or metadata["sha256"] != metadata_inputs[0].get("sha256")
        or type(metadata["bytes"]) is not int
        or metadata["bytes"] < 1
        or metadata["acquired_by_tool"] is not False
        or images != {"count": 0, "bytes": 0, "acquired_by_tool": False}
        or any(type(images[key]) is not int for key in ("count", "bytes"))
    ):
        raise PlanError("external_inputs: metadata/image provenance is invalid")
    metadata_path = Path(metadata["path"])
    try:
        actual_bytes = metadata_path.stat().st_size
    except OSError as exc:
        raise PlanError(f"external_inputs: cannot stat bound metadata file {metadata_path}: {exc}") from exc
    if actual_bytes != metadata["bytes"] or _file_sha(metadata_path) != metadata["sha256"]:
        raise PlanError("external_inputs: bound metadata bytes or SHA256 changed")


def _validate_parent_semantics(
    child: dict[str, Any],
    parent: dict[str, Any],
    parent_path: Path,
) -> None:
    if child["parent_manifest_id"] != parent["manifest_id"]:
        raise PlanError("parent_manifest_id does not match the immediate parent")
    if child["inputs"][: len(parent["inputs"])] != parent["inputs"]:
        raise PlanError("inputs: child lineage does not preserve the parent input chain")
    for field in ("source", "approval", "external_inputs", "policy", "stage_order"):
        if child[field] != parent[field]:
            raise PlanError(f"parent binding: child {field} differs from the immediate parent")
    appended = child["inputs"][len(parent["inputs"]) :]
    parent_descriptor = appended[0] if appended else {}
    if (
        parent_descriptor.get("role") != "parent_manifest"
        or Path(str(parent_descriptor.get("path", ""))).resolve() != parent_path.resolve()
        or parent_descriptor.get("format") != "json"
        or parent_descriptor.get("sha256") != child["parent_manifest_sha256"]
        or parent_descriptor.get("record_count") != len(parent["records"])
    ):
        raise PlanError("inputs: immediate parent descriptor is invalid")

    child_keys = [record["record_key"] for record in child["records"]]
    parent_keys = [record["record_key"] for record in parent["records"]]
    if child_keys != parent_keys:
        raise PlanError("parent binding: record set/order differs from the immediate parent")
    if child["operation"] == "review_package_plan":
        if parent["operation"] not in {"plan", "reconcile"} or len(appended) != 1:
            raise PlanError("parent binding: invalid review-package lineage")
        for child_record, parent_record in zip(child["records"], parent["records"]):
            comparable = copy.deepcopy(child_record)
            comparable["review_selected"] = parent_record["review_selected"]
            if comparable != parent_record:
                raise PlanError("parent binding: review package changed non-review record state")
    elif child["operation"] == "reconcile":
        if parent["operation"] not in {"plan", "reconcile"} or len(appended) != 2:
            raise PlanError("parent binding: invalid reconcile lineage")
        if appended[1].get("role") != "hash_inventory":
            raise PlanError("parent binding: reconcile lacks its hash inventory")
        mutable = {
            "hashes",
            "decision",
            "decision_reasons",
            "duplicate_of",
            "phash_distance",
            "split",
            "review_selected",
        }
        for child_record, parent_record in zip(child["records"], parent["records"]):
            for field in set(parent_record) - mutable:
                if child_record[field] != parent_record[field]:
                    raise PlanError(f"parent binding: reconcile changed immutable record field {field}")


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
            "approval",
            "source",
            "inputs",
            "policy",
            "stage_order",
            "tool_activity",
            "external_inputs",
            "review_plan",
            "summary",
            "records",
        },
        "manifest",
    )
    if (
        top["schema_version"] != SCHEMA
        or type(top["manifest_version"]) is not int
        or top["manifest_version"] != 2
        or top["workflow_status"] != STATUS
        or top["training_ready"] is not False
        or top["operation"] not in {"plan", "reconcile", "review_package_plan"}
    ):
        raise PlanError("manifest: invalid approved-policy v2 header")
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
        and re.fullmatch(r"anima-step2a-v2-[a-f0-9]{24}", top["parent_manifest_id"])
    ):
        raise PlanError("manifest: child must bind v2 parent SHA256 and manifest_id")

    approval = _validate_approval_binding(top["approval"])
    if approval["policy_digest"] != top["policy"].get("policy_digest"):
        raise PlanError("approval: approved digest differs from manifest policy")
    if top["policy"] != _policy() or _policy_digest(top["policy"]) != APPROVED_POLICY_DIGEST:
        raise PlanError("policy: approved snapshot or strict digest reproduction failed")

    tool = _exact_keys(
        top["tool_activity"],
        {
            "network_requests_performed",
            "metadata_download_count",
            "metadata_download_bytes",
            "image_download_count",
            "image_download_bytes",
        },
        "tool_activity",
    )
    expected_tool = {
        "network_requests_performed": 0,
        "metadata_download_count": 0,
        "metadata_download_bytes": 0,
        "image_download_count": 0,
        "image_download_bytes": 0,
    }
    if tool != expected_tool or any(type(tool[field]) is not int for field in expected_tool):
        raise PlanError("tool_activity: all network/download counters must remain zero")

    if not isinstance(top["inputs"], list):
        raise PlanError("inputs: expected an array")
    _validate_approved_source_contract(top["source"], "source")
    _validate_external_inputs(top["external_inputs"], top["source"], top["inputs"])
    if top["stage_order"] != STAGE_ORDER:
        raise PlanError("stage_order: dedup/split order drift")

    if not isinstance(top["records"], list):
        raise PlanError("records: expected an array")
    for index, record in enumerate(top["records"]):
        if not isinstance(record, dict):
            raise PlanError(f"records[{index}]: expected an object")
        if (
            record.get("filter_decision") not in FILTER_DECISIONS
            or record.get("decision") not in FINAL_DECISIONS
            or any(
                V1_ELIGIBLE in value
                for value in (record.get("filter_decision"), record.get("decision"))
                if isinstance(value, str)
            )
        ):
            raise PlanError(f"records[{index}]: v2 decision vocabulary is invalid")

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
    if review["version"] != REVIEW_VERSION:
        raise PlanError("review_plan: v2 version is invalid")

    summary = _exact_keys(
        top["summary"],
        {
            "input_records",
            "policy_eligible_records",
            "held_records",
            "rejected_records",
            "train_records",
            "validation_records",
            "review_selected_records",
            "decision_counts",
        },
        "summary",
    )
    if (
        any(type(summary[field]) is not int for field in set(summary) - {"decision_counts"})
        or not isinstance(summary["decision_counts"], dict)
        or any(type(value) is not int for value in summary["decision_counts"].values())
        or summary != _summary(top)
    ):
        raise PlanError("summary: counts do not match records/review plan")

    v1._validate(_to_v1_proxy(top), check_parent_file=False)

    if top["operation"] != "plan" and check_parent_file:
        parent_entries = [item for item in top["inputs"] if item.get("role") == "parent_manifest"]
        if not parent_entries:
            raise PlanError("inputs: child lacks an immediate parent descriptor")
        parent_path = Path(parent_entries[-1]["path"])
        if _file_sha(parent_path) != top["parent_manifest_sha256"]:
            raise PlanError("inputs: immediate parent file SHA256 binding failed")
        parent = _load_manifest(parent_path, check_parent_file=False)
        _validate_parent_semantics(top, parent, parent_path)

    if top["manifest_id"] != _manifest_identifier(top):
        raise PlanError("manifest_id: portable semantic identity mismatch")


def _finish(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["summary"] = _summary(manifest)
    manifest["manifest_id"] = _manifest_identifier(manifest)
    _validate(manifest)
    return manifest


def _load_manifest(path: Path, check_parent_file: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot decode manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError("manifest root must be an object")
    try:
        _validate(value, check_parent_file=check_parent_file)
    except PlanError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PlanError(f"manifest semantic validation failed closed: {exc}") from exc
    return value


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    _validate_approved_source_contract(
        {
            "name": args.source,
            "revision": args.source_revision,
            "license": args.source_license,
            "license_evidence": {
                "url": args.license_evidence_url,
                "revision": args.license_evidence_revision,
                "sha256": args.license_evidence_sha256,
                "captured_date": args.license_evidence_date,
            },
        },
        "plan arguments",
    )
    approval = _load_approval(Path(args.approval_record), Path(args.proposal))
    path = Path(args.input)
    rows, kind = _read_rows(path, args.max_input_records)
    policy = _policy()
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
        _record(row, index, source, policy["filter_version"], policy["min_score"]) for index, row in enumerate(rows, 1)
    ]
    if len({record["record_key"] for record in records}) != len(records):
        raise PlanError("duplicate source/image_id keys make reconciliation ambiguous")
    _dedup_split(records, policy)
    return _finish(
        _base(
            "plan",
            source,
            [_input(path, "metadata", kind, len(rows), args.max_input_records)],
            approval,
            _external_inputs(path),
            policy,
            records,
            None,
            None,
        )
    )


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    inventory_path = Path(args.inventory)
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
            raise PlanError(f"inventory row {index} lacks exact source/source_revision/image_id binding")
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
            copy.deepcopy(parent["approval"]),
            copy.deepcopy(parent["external_inputs"]),
            copy.deepcopy(parent["policy"]),
            records,
            _file_sha(manifest_path),
            parent["manifest_id"],
        )
    )


def _review_order(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (record for record in records if record["decision"] == ELIGIBLE),
        key=lambda record: _digest(
            (policy["split_salt"] + "\0review\0" + record["record_key"] + "\0" + record["hashes"]["sha256"]).encode()
        ),
    )


def review_package(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.manifest)
    parent = _load_manifest(path)
    records = copy.deepcopy(parent["records"])
    eligible = [record for record in records if record["decision"] == ELIGIBLE]
    if not eligible:
        raise PlanError("zero post-dedup policy-eligible records; reconcile hashes first")
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
        copy.deepcopy(parent["approval"]),
        copy.deepcopy(parent["external_inputs"]),
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
    return v1._revision(value)


def _record_limit(value: str) -> int:
    return v1._record_limit(value)


def _sha256_arg(value: str) -> str:
    return v1._sha256_arg(value)


def _https_url(value: str) -> str:
    return v1._https_url(value)


def _date_arg(value: str) -> str:
    return v1._date_arg(value)


def _fraction(value: str) -> float:
    return v1._fraction(value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Offline Anima Step 2a approved-policy metadata and review planner")
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--input", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--approval-record", required=True)
    plan.add_argument("--proposal", required=True)
    plan.add_argument("--source", required=True)
    plan.add_argument("--source-revision", required=True, type=_revision)
    plan.add_argument("--source-license", required=True)
    plan.add_argument("--license-evidence-url", required=True, type=_https_url)
    plan.add_argument("--license-evidence-revision", required=True, type=_revision)
    plan.add_argument("--license-evidence-sha256", required=True, type=_sha256_arg)
    plan.add_argument("--license-evidence-date", required=True, type=_date_arg)
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
    if args.command == "plan" and not all(str(value).strip() for value in (args.source, args.source_license)):
        print(
            "ERROR: fail-closed: source/license fields must be non-empty",
            file=sys.stderr,
        )
        return 2
    if args.command == "review-package" and args.minimum_sample < 100:
        print(
            "ERROR: fail-closed: minimum review sample must be at least 100",
            file=sys.stderr,
        )
        return 2
    try:
        payload = args.handler(args)
        _write_no_clobber(Path(args.output), payload)
    except PlanError as exc:
        print(f"ERROR: fail-closed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {Path(args.output).resolve()} (approved policy; training_ready=false; tool downloads=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
