#!/usr/bin/env python3
"""Offline draft intake contract for Anima Step 2a.

This tool intentionally performs no network access, column-payload fetches,
image downloads, or training.  It provides three fail-closed operations:

* ``sample-plan`` ranks bounded Parquet page bundles from a separately proven
  OffsetIndex inventory without using physical first-N order.
* ``derive-provenance`` preserves the target's native ``source`` value while
  explicitly writing canonical source/revision provenance into every row.
* ``ledger-join`` applies an immutable full-key removal ledger.  A delta-only
  ledger never turns a non-match into an allow decision.

The input semantics remain DRAFT/G1 HOLD and are not covered by the existing
v1 policy approval record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import anima_step2a_parquet_footer_plan as footer_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_SCHEMA = "anima-step2a-intake-manifest-v1"
PAGE_INVENTORY_SCHEMA = "anima-step2a-page-inventory-v1"
PAGE_INDEX_PROBE_SCHEMA = "anima-step2a-page-index-probe-v1"
LEDGER_SCHEMA = "anima-step2a-removal-ledger-v1"
INPUT_POLICY_VERSION = "anima-step2a-input-policy-v2-draft"
STATUS = "DRAFT_G1_HOLD"
SAMPLER_VERSION = "sha256-page-bundle-rank-v1"
PROVENANCE_MODE = "derived-from-immutable-file-binding-v1"
DEFAULT_SELECTION_SALT = "anima-step2a-page-selection-20260718-v1"

TARGET_SOURCE = "deepghs/danbooru2024-webp-4Mpixel"
TARGET_REVISION = "60ca7e02a844bccbe6c8344d1a18b7ea280b787f"
TARGET_FILENAME = "metadata.parquet"
TARGET_SIZE_BYTES = 6_846_458_425
TARGET_FOOTER_LENGTH = 139_521
TARGET_FOOTER_SHA256 = "0c3d48949974d9e6870304290a2649a0d2b9f762d3453374ad6defb3b71fb276"
APPROVED_FILTER_VERSION = "anima-step2a-pilot-filter-v1"

APPROVAL_RECORD_RELATIVE_PATH = "docs/lora_training/anima_step2a/APPROVAL_RECORD_v1.json"
APPROVAL_RECORD_SHA256 = "388922c14c093093c4f10ea80cfde6789d40dc840af31321ac9154b38cf60cc2"
APPROVAL_SNAPSHOT_RELATIVE_PATH = "docs/lora_training/anima_step2a/COMPLIANCE_PROPOSAL_APPROVAL_SNAPSHOT_2026-07-18.md"
APPROVAL_SNAPSHOT_SHA256 = "4c375d021f883755fe8ed05d4865eddf3983bebe7d67500a81565f49f2cfa018"
FOOTER_PLAN_RELATIVE_PATH = "outputs/_analysis/anima_step2a_footer_plan_2026-07-18/footer_selected_column_plan_v1.json"
FOOTER_PLAN_SHA256 = "067eeca1ffdbc08391e1911c16931991550b1605d6b7484b4082b9367a01e36a"
REMOVAL_LEDGER_CHAIN_ID = "anima-step2a-removal-chain-20260718-v1"
REMOVAL_LEDGER_GENESIS_ID = "anima-step2a-removal-genesis-20260718-v1"
REMOVAL_LEDGER_GENESIS_RELATIVE_PATH = "docs/lora_training/anima_step2a/REMOVAL_LEDGER_GENESIS_v1.json"
REMOVAL_LEDGER_GENESIS_SHA256 = "8a1269cfc40b77cb806d6ca10428133600a03b338a13b11477738ae441b6de47"
REMOVAL_LEDGER_HEAD_ID = "anima-step2a-removal-head-20260718-v1"
REMOVAL_LEDGER_HEAD_RELATIVE_PATH = "docs/lora_training/anima_step2a/REMOVAL_LEDGER_HEAD_v1.json"
REMOVAL_LEDGER_HEAD_SHA256 = "484bb088e9bfee209f8ed6073fe854ed6dc551928abfa3a0d3d459df455fc088"

REQUIRED_PROJECTED_COLUMNS = (
    "score",
    "source",
    "md5",
    "tag_string",
    "tag_string_general",
    "rating",
    "id",
)

DEFAULT_PILOT_RECORD_CAP = 10_000
HARD_PILOT_RECORD_CAP = 100_000
DEFAULT_MINIMUM_RECORDS = 1_000
DEFAULT_MAX_TOUCHED_ROWS = 10_000
HARD_MAX_TOUCHED_ROWS = 100_000
DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
HARD_MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_PAYLOAD_REQUESTS = 128
HARD_MAX_PAYLOAD_REQUESTS = 4096
MAX_METADATA_RANGE_REQUESTS = 128
MAX_METADATA_RANGE_BYTES = 32 * 1024 * 1024
MAX_RAW_FOOTER_EVIDENCE_BYTES = 1024 * 1024
MAX_OFFSET_INDEX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_LEDGER_COVERAGE_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
MAX_JSONL_TOTAL_BYTES = 64 * 1024 * 1024
MAX_COMPACT_DEPTH = 32
MAX_COMPACT_VALUES = 2_000_000
MAX_COMPACT_ITEMS = 1_000_000
PARENT_ROLE_MAX_BYTES = {
    "page_index_inventory": MAX_JSON_BYTES,
    "page_index_probe": MAX_JSON_BYTES,
    "raw_fixed_footer": MAX_RAW_FOOTER_EVIDENCE_BYTES,
    "offset_index_evidence": MAX_OFFSET_INDEX_EVIDENCE_BYTES,
    "selection_manifest": MAX_JSON_BYTES,
    "bounded_projected_rows": MAX_JSONL_TOTAL_BYTES,
    "derived_provenance_manifest": MAX_JSON_BYTES,
    "removal_ledger": MAX_JSON_BYTES,
    "ledger_coverage_evidence": MAX_LEDGER_COVERAGE_EVIDENCE_BYTES,
    "trusted_removal_ledger_head": MAX_JSON_BYTES,
    "previous_removal_snapshot": MAX_JSON_BYTES,
}
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class IntakeError(RuntimeError):
    """Fail-closed contract error."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntakeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise IntakeError(f"non-finite JSON number: {value}")


def _json_loads(raw: str | bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except IntakeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeError("invalid JSON") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_read_handle(path: Path) -> Any:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not directory or os.open not in os.supports_dir_fd:
        raise IntakeError("platform lacks race-resistant no-symlink input open")
    absolute = _absolute_lexical(path)
    parts = absolute.parts
    if len(parts) < 2 or not absolute.anchor:
        raise IntakeError(f"input path has no file component: {path}")
    directory_flags = os.O_RDONLY | directory | nofollow | cloexec
    final_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | nofollow | cloexec
    parent_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in parts[1:-1]:
            child_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        descriptor = os.open(
            parts[-1],
            final_flags,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _file_sha256(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> str:
    digest = hashlib.sha256()
    try:
        with _open_read_handle(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IntakeError(f"input is not a regular file: {path}")
            if before.st_size <= 0 or before.st_size > max_bytes:
                raise IntakeError(f"input size is outside 1..{max_bytes}: {path}")
            total = 0
            while True:
                read_size = min(1024 * 1024, max_bytes - total + 1)
                chunk = handle.read(read_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise IntakeError(f"input size is outside 1..{max_bytes}: {path}")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except IntakeError:
        raise
    except OSError as exc:
        raise IntakeError(f"failed to hash input: {path}") from exc
    if total != before.st_size or _stable_file_identity(after) != _stable_file_identity(before):
        raise IntakeError(f"input changed while being hashed: {path}")
    return digest.hexdigest()


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        with _open_read_handle(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IntakeError(f"{label} input is not a regular file: {path}")
            if before.st_size <= 0 or before.st_size > max_bytes:
                raise IntakeError(f"{label} input size is outside 1..{max_bytes}: {path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = handle.read(min(1024 * 1024, max_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise IntakeError(f"{label} input size is outside 1..{max_bytes}: {path}")
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except IntakeError:
        raise
    except OSError as exc:
        raise IntakeError(f"failed to read {label} input: {path}") from exc
    if total != before.st_size or _stable_file_identity(after) != _stable_file_identity(before):
        raise IntakeError(f"{label} input changed while being read: {path}")
    return b"".join(chunks)


def _load_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    return _json_loads(_read_bounded_bytes(path, max_bytes=max_bytes, label="JSON"))


def _load_json_with_sha256(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> tuple[Any, str]:
    raw = _read_bounded_bytes(path, max_bytes=max_bytes, label="JSON")
    return _json_loads(raw), _digest(raw)


def _expect_keys(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise IntakeError(f"{label}: expected keys {sorted(expected)}, got {actual}")
    return value


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _required_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not _is_int(value, minimum=minimum):
        raise IntakeError(f"{label}: expected integer >= {minimum}")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntakeError(f"{label}: expected non-empty string")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _required_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or UTC_TIMESTAMP.fullmatch(value) is None:
        raise IntakeError(f"{label}: expected UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise IntakeError(f"{label}: invalid UTC calendar timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise IntakeError(f"{label}: non-canonical UTC timestamp")
    return value


def _required_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise IntakeError(f"{label}: invalid lowercase hexadecimal digest")
    return value


CT_STOP = 0x00
CT_TRUE = 0x01
CT_FALSE = 0x02
CT_BYTE = 0x03
CT_I16 = 0x04
CT_I32 = 0x05
CT_I64 = 0x06
CT_DOUBLE = 0x07
CT_BINARY = 0x08
CT_LIST = 0x09
CT_SET = 0x0A
CT_MAP = 0x0B
CT_STRUCT = 0x0C
CT_UUID = 0x0D


class _CompactReader:
    """Bounded Apache Thrift Compact reader for Parquet OffsetIndex bytes."""

    def __init__(self, payload: bytes) -> None:
        if not payload or len(payload) > MAX_OFFSET_INDEX_EVIDENCE_BYTES:
            raise IntakeError("OffsetIndex payload size is outside the evidence cap")
        self.payload = payload
        self.offset = 0
        self.values_read = 0

    def _read(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise IntakeError("truncated OffsetIndex compact payload")
        result = self.payload[self.offset : self.offset + size]
        self.offset += size
        return result

    def _read_u8(self) -> int:
        return self._read(1)[0]

    def _read_varint(self) -> int:
        result = 0
        shift = 0
        for index in range(10):
            byte = self._read_u8()
            if index == 9 and byte > 1:
                raise IntakeError("OffsetIndex compact varint overflow")
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                return result
            shift += 7
        raise IntakeError("OffsetIndex compact varint overflow")

    def _read_zigzag(self) -> int:
        value = self._read_varint()
        return (value >> 1) ^ -(value & 1)

    def _count(self) -> None:
        self.values_read += 1
        if self.values_read > MAX_COMPACT_VALUES:
            raise IntakeError("OffsetIndex compact value cap exceeded")

    def _read_collection(self, depth: int) -> tuple[int, list[Any]]:
        header = self._read_u8()
        size = header >> 4
        element_type = header & 0x0F
        if size == 15:
            size = self._read_varint()
        if size > MAX_COMPACT_ITEMS:
            raise IntakeError("OffsetIndex compact collection cap exceeded")
        return element_type, [self._read_value(element_type, depth + 1) for _ in range(size)]

    def _read_value(self, type_code: int, depth: int) -> Any:
        if depth > MAX_COMPACT_DEPTH:
            raise IntakeError("OffsetIndex compact nesting cap exceeded")
        self._count()
        if type_code in {CT_TRUE, CT_FALSE}:
            encoded = self._read_u8()
            if encoded not in {CT_TRUE, CT_FALSE}:
                raise IntakeError("invalid OffsetIndex compact boolean")
            return encoded == CT_TRUE
        if type_code == CT_BYTE:
            value = self._read_u8()
            return value if value < 128 else value - 256
        if type_code in {CT_I16, CT_I32, CT_I64}:
            return self._read_zigzag()
        if type_code == CT_DOUBLE:
            return struct.unpack("<d", self._read(8))[0]
        if type_code == CT_BINARY:
            size = self._read_varint()
            if size > MAX_OFFSET_INDEX_EVIDENCE_BYTES:
                raise IntakeError("OffsetIndex compact binary cap exceeded")
            return self._read(size)
        if type_code in {CT_LIST, CT_SET}:
            return self._read_collection(depth)
        if type_code == CT_STRUCT:
            return self._read_struct(depth)
        if type_code == CT_UUID:
            return self._read(16)
        if type_code == CT_MAP:
            raise IntakeError("OffsetIndex compact map is outside the frozen schema")
        raise IntakeError("unknown OffsetIndex compact type")

    def _read_struct(self, depth: int) -> dict[int, tuple[int, Any]]:
        if depth > MAX_COMPACT_DEPTH:
            raise IntakeError("OffsetIndex compact nesting cap exceeded")
        fields: dict[int, tuple[int, Any]] = {}
        last_field_id = 0
        while True:
            header = self._read_u8()
            if header == CT_STOP:
                return fields
            delta = header >> 4
            type_code = header & 0x0F
            field_id = last_field_id + delta if delta else self._read_zigzag()
            if not 1 <= field_id <= 32_767 or field_id in fields:
                raise IntakeError("invalid OffsetIndex compact field id")
            last_field_id = field_id
            if type_code == CT_TRUE:
                value: Any = True
            elif type_code == CT_FALSE:
                value = False
            else:
                value = self._read_value(type_code, depth + 1)
            fields[field_id] = (type_code, value)

    def parse_root(self) -> dict[int, tuple[int, Any]]:
        result = self._read_struct(0)
        if self.offset != len(self.payload):
            raise IntakeError("trailing bytes after OffsetIndex compact struct")
        return result


def _parse_offset_index(payload: bytes) -> list[dict[str, int]]:
    root = _CompactReader(payload).parse_root()
    if not set(root) <= {1, 2} or 1 not in root:
        raise IntakeError("OffsetIndex fields are outside the frozen Apache schema")
    page_type, page_collection = root[1]
    if page_type != CT_LIST or not isinstance(page_collection, tuple):
        raise IntakeError("OffsetIndex.page_locations is not a compact list")
    element_type, locations = page_collection
    if element_type != CT_STRUCT or not locations:
        raise IntakeError("OffsetIndex.page_locations must contain structs")
    result: list[dict[str, int]] = []
    for ordinal, location in enumerate(locations):
        if not isinstance(location, dict) or set(location) != {1, 2, 3}:
            raise IntakeError("PageLocation fields are outside the frozen Apache schema")
        if location[1][0] != CT_I64 or location[2][0] != CT_I32 or location[3][0] != CT_I64:
            raise IntakeError("PageLocation compact types do not match Apache schema")
        offset = location[1][1]
        compressed_page_size = location[2][1]
        first_row_index = location[3][1]
        if (
            type(offset) is not int
            or type(compressed_page_size) is not int
            or type(first_row_index) is not int
            or offset < 0
            or compressed_page_size <= 0
            or first_row_index < 0
        ):
            raise IntakeError("PageLocation contains an invalid integer")
        result.append(
            {
                "page_ordinal": ordinal,
                "first_row_index": first_row_index,
                "offset": offset,
                "compressed_page_size": compressed_page_size,
            }
        )
    if 2 in root:
        size_type, size_collection = root[2]
        if size_type != CT_LIST or not isinstance(size_collection, tuple):
            raise IntakeError("OffsetIndex byte-array sizes are not a compact list")
        size_element_type, sizes = size_collection
        if size_element_type != CT_I64 or len(sizes) != len(result):
            raise IntakeError("OffsetIndex byte-array size count does not match page locations")
        if any(type(item) is not int or item < 0 for item in sizes):
            raise IntakeError("OffsetIndex byte-array sizes contain an invalid integer")
    for previous, current in zip(result, result[1:]):
        if current["offset"] <= previous["offset"] or current["first_row_index"] <= previous["first_row_index"]:
            raise IntakeError("OffsetIndex page locations are not strictly ordered")
    return result


def _portable_path(path: Path) -> str:
    resolved = _absolute_lexical(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_bytes(path: Path, *, max_bytes: int) -> bytes:
    return _read_bounded_bytes(path, max_bytes=max_bytes, label="binary")


def _load_bytes_with_sha256(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, str]:
    raw = _read_bounded_bytes(path, max_bytes=max_bytes, label="binary")
    return raw, _digest(raw)


def _write_no_clobber(path: Path, value: Any) -> None:
    parent = path.parent.resolve()
    if not parent.is_dir():
        raise IntakeError(f"output parent does not exist: {parent}")
    payload = _canonical(value) + b"\n"
    if len(payload) > MAX_JSON_BYTES:
        raise IntakeError(f"output exceeds {MAX_JSON_BYTES}-byte artifact cap")
    descriptor = -1
    temporary: Path | None = None
    published = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise IntakeError(f"output already exists: {path}") from exc
        published = True
        temporary.unlink()
        temporary = None
        _fsync_directory(parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
            try:
                _fsync_directory(parent)
            except OSError:
                pass
        raise


def _source_binding() -> dict[str, Any]:
    approval_path = PROJECT_ROOT / APPROVAL_RECORD_RELATIVE_PATH
    snapshot_path = PROJECT_ROOT / APPROVAL_SNAPSHOT_RELATIVE_PATH
    footer_plan_path = PROJECT_ROOT / FOOTER_PLAN_RELATIVE_PATH
    ledger_genesis_path = PROJECT_ROOT / REMOVAL_LEDGER_GENESIS_RELATIVE_PATH
    ledger_head_path = PROJECT_ROOT / REMOVAL_LEDGER_HEAD_RELATIVE_PATH
    approval, approval_sha256 = _load_json_with_sha256(approval_path)
    footer_plan, footer_plan_sha256 = _load_json_with_sha256(footer_plan_path)
    if approval_sha256 != APPROVAL_RECORD_SHA256:
        raise IntakeError("approval record byte binding mismatch")
    if footer_plan_sha256 != FOOTER_PLAN_SHA256:
        raise IntakeError("footer plan byte binding mismatch")
    bindings = (
        (snapshot_path, APPROVAL_SNAPSHOT_SHA256, "approval snapshot"),
        (ledger_genesis_path, REMOVAL_LEDGER_GENESIS_SHA256, "removal-ledger genesis"),
        (ledger_head_path, REMOVAL_LEDGER_HEAD_SHA256, "removal-ledger head"),
    )
    for path, expected, label in bindings:
        if _file_sha256(path) != expected:
            raise IntakeError(f"{label} byte binding mismatch")

    try:
        valid = (
            approval["schema"] == "anima-step2a-policy-approval-record-v1"
            and approval["approved_policy"]["filter_version"] == APPROVED_FILTER_VERSION
            and approval["proposal"]["sha256"] == APPROVAL_SNAPSHOT_SHA256
            and approval["scope"]["bounded_metadata_pilot_authorized"] is True
            and approval["scope"]["bulk_image_download_authorized"] is False
            and approval["scope"]["phase_d_training_authorized"] is False
            and footer_plan["source"] == TARGET_SOURCE
            and footer_plan["revision"] == TARGET_REVISION
            and footer_plan["filename"] == TARGET_FILENAME
            and footer_plan["size_bytes"] == TARGET_SIZE_BYTES
            and footer_plan["prior_raw_footer_probe"]["footer_sha256"] == TARGET_FOOTER_SHA256
            and footer_plan["selected_column_plan"]["column_payload_requests_performed"] == 0
            and footer_plan["selected_column_plan"]["column_payload_bytes_received"] == 0
            and footer_plan["activity"]["image_files_downloaded"] == 0
            and footer_plan["activity"]["training_runs_started"] == 0
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise IntakeError("checked-in approval/footer lineage is semantically inconsistent")

    result = {
        "source": TARGET_SOURCE,
        "source_revision": TARGET_REVISION,
        "filename": TARGET_FILENAME,
        "size_bytes": TARGET_SIZE_BYTES,
        "footer_sha256": TARGET_FOOTER_SHA256,
        "footer_plan": {
            "path": FOOTER_PLAN_RELATIVE_PATH,
            "sha256": FOOTER_PLAN_SHA256,
        },
        "prior_policy_approval": {
            "record_path": APPROVAL_RECORD_RELATIVE_PATH,
            "record_sha256": APPROVAL_RECORD_SHA256,
            "snapshot_path": APPROVAL_SNAPSHOT_RELATIVE_PATH,
            "snapshot_sha256": APPROVAL_SNAPSHOT_SHA256,
            "approved_filter_version": APPROVED_FILTER_VERSION,
            "approves_v2_input_semantics": False,
        },
        "removal_ledger_trust": {
            "chain_id": REMOVAL_LEDGER_CHAIN_ID,
            "genesis_path": REMOVAL_LEDGER_GENESIS_RELATIVE_PATH,
            "genesis_sha256": REMOVAL_LEDGER_GENESIS_SHA256,
            "head_path": REMOVAL_LEDGER_HEAD_RELATIVE_PATH,
            "head_sha256": REMOVAL_LEDGER_HEAD_SHA256,
            "head_id": REMOVAL_LEDGER_HEAD_ID,
        },
        "input_policy_version": INPUT_POLICY_VERSION,
    }
    result["source_binding_sha256"] = _digest(result)
    return result


def _footer_projection_truth() -> dict[int, dict[str, Any]]:
    footer_plan_path = PROJECT_ROOT / FOOTER_PLAN_RELATIVE_PATH
    footer_plan, footer_plan_sha256 = _load_json_with_sha256(footer_plan_path)
    if footer_plan_sha256 != FOOTER_PLAN_SHA256:
        raise IntakeError("footer plan byte binding mismatch")
    try:
        file_metadata = footer_plan["file_metadata"]
        selected_plan = footer_plan["selected_column_plan"]
        selected_paths = [item["source_path"] for item in selected_plan["selected_columns"]]
        candidates = selected_plan["whole_row_group_candidates"]
    except (KeyError, TypeError) as exc:
        raise IntakeError("footer plan projection truth is malformed") from exc
    if (
        file_metadata.get("row_group_count") != 8
        or file_metadata.get("num_rows") != 8_005_010
        or set(selected_paths) != set(REQUIRED_PROJECTED_COLUMNS)
        or len(selected_paths) != len(REQUIRED_PROJECTED_COLUMNS)
        or not isinstance(candidates, list)
        or len(candidates) != 8
    ):
        raise IntakeError("footer plan projection universe drift")
    result: dict[int, dict[str, Any]] = {}
    for expected_index, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, dict)
            or candidate.get("index") != expected_index
            or not _is_int(candidate.get("rows"), minimum=1)
            or not isinstance(candidate.get("ranges"), list)
        ):
            raise IntakeError("footer plan row-group truth drift")
        columns: dict[str, tuple[int, int]] = {}
        for range_value in candidate["ranges"]:
            if not isinstance(range_value, dict):
                raise IntakeError("footer plan column range is malformed")
            path = range_value.get("source_path")
            start = range_value.get("start")
            end = range_value.get("end")
            byte_count = range_value.get("bytes")
            if (
                path not in REQUIRED_PROJECTED_COLUMNS
                or path in columns
                or not _is_int(start)
                or not _is_int(end)
                or not _is_int(byte_count, minimum=1)
                or end < start
                or byte_count != end - start + 1
                or range_value.get("row_group_index") != expected_index
            ):
                raise IntakeError("footer plan column projection truth drift")
            columns[path] = (start, end)
        if set(columns) != set(REQUIRED_PROJECTED_COLUMNS):
            raise IntakeError("footer plan row group lacks a frozen projected column")
        result[expected_index] = {
            "row_count": candidate["rows"],
            "columns": columns,
        }
    if sum(item["row_count"] for item in result.values()) != file_metadata["num_rows"]:
        raise IntakeError("footer plan row counts do not match file metadata")
    return result


def _parent_max_bytes(role: str) -> int:
    try:
        return PARENT_ROLE_MAX_BYTES[role]
    except KeyError as exc:
        raise IntakeError(f"parent role has no bounded-file contract: {role}") from exc


def _parent(
    path: Path,
    role: str,
    semantic_id: str | None = None,
    *,
    byte_sha256: str | None = None,
) -> dict[str, Any]:
    if byte_sha256 is not None:
        _required_hex(byte_sha256, HEX64, f"{role} parent sha256")
    return {
        "role": role,
        "path": _portable_path(path),
        "sha256": byte_sha256 or _file_sha256(path, max_bytes=_parent_max_bytes(role)),
        "semantic_id": semantic_id,
    }


def _activity() -> dict[str, Any]:
    return {
        "network_requests_performed": 0,
        "metadata_range_requests_performed": 0,
        "metadata_range_bytes_received": 0,
        "column_payload_requests_performed": 0,
        "column_payload_bytes_received": 0,
        "image_files_downloaded": 0,
        "image_bytes_downloaded": 0,
        "training_runs_started": 0,
    }


def _manifest_identifier(manifest: dict[str, Any]) -> str:
    identity = copy.deepcopy(manifest)
    identity.pop("manifest_id", None)
    for parent in identity["parents"]:
        parent.pop("path", None)
        parent.pop("sha256", None)
    return "anima-step2a-intake-" + _digest(identity)[:24]


def _base_manifest(
    operation: str,
    parents: list[dict[str, Any]],
    limits: dict[str, Any],
    selection: dict[str, Any] | None,
    records: list[dict[str, Any]],
    ledger: dict[str, Any] | None,
    reasons: list[str],
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_version": INPUT_POLICY_VERSION,
        "operation": operation,
        "status": STATUS,
        "training_ready": False,
        "manifest_id": "",
        "source_binding": _source_binding(),
        "parents": parents,
        "limits": limits,
        "selection": selection,
        "records": records,
        "ledger": ledger,
        "gate": {
            "g1": "HOLD",
            "reasons": sorted(set(reasons)),
        },
        "activity": _activity(),
    }
    manifest["manifest_id"] = _manifest_identifier(manifest)
    _validate_manifest(manifest)
    return manifest


def _validate_parent(value: Any, label: str) -> None:
    parent = _expect_keys(value, {"role", "path", "sha256", "semantic_id"}, label)
    _required_text(parent["role"], f"{label}.role")
    _required_text(parent["path"], f"{label}.path")
    _required_hex(parent["sha256"], HEX64, f"{label}.sha256")
    if parent["semantic_id"] is not None:
        _required_text(parent["semantic_id"], f"{label}.semantic_id")


def _validate_source_binding(value: Any) -> None:
    if value != _source_binding():
        raise IntakeError("manifest source binding drift")


def _validate_record(value: Any, index: int) -> None:
    label = f"records[{index}]"
    record = _expect_keys(
        value,
        {
            "record_key",
            "source",
            "source_revision",
            "native_source",
            "native_source_revision",
            "source_locator",
            "image_id",
            "md5",
            "sha256",
            "raw_row_sha256",
            "projected_metadata",
            "provenance",
            "ledger_status",
        },
        label,
    )
    if record["source"] != TARGET_SOURCE or record["source_revision"] != TARGET_REVISION:
        raise IntakeError(f"{label}: canonical source binding mismatch")
    _optional_text(record["native_source"], f"{label}.native_source")
    if record["native_source_revision"] not in {None, TARGET_REVISION}:
        raise IntakeError(f"{label}: conflicting native source_revision")
    image_id = _required_text(record["image_id"], f"{label}.image_id")
    if record["record_key"] != f"{TARGET_SOURCE}::{image_id}":
        raise IntakeError(f"{label}: record key mismatch")
    _required_hex(record["md5"], HEX32, f"{label}.md5")
    if record["sha256"] is not None:
        _required_hex(record["sha256"], HEX64, f"{label}.sha256")
    _required_hex(record["raw_row_sha256"], HEX64, f"{label}.raw_row_sha256")
    locator = _expect_keys(
        record["source_locator"],
        {
            "row_group_index",
            "row_index_in_group",
            "anchor_page_ordinal",
            "selection_key_sha256",
        },
        f"{label}.source_locator",
    )
    _required_int(locator["row_group_index"], f"{label}.source_locator.row_group_index")
    _required_int(locator["row_index_in_group"], f"{label}.source_locator.row_index_in_group")
    _required_int(locator["anchor_page_ordinal"], f"{label}.source_locator.anchor_page_ordinal")
    _required_hex(
        locator["selection_key_sha256"],
        HEX64,
        f"{label}.source_locator.selection_key_sha256",
    )
    metadata = _expect_keys(
        record["projected_metadata"],
        {
            "tag_string",
            "tag_string_general",
            "rating",
            "score",
            "asset_license",
        },
        f"{label}.projected_metadata",
    )
    for field in ("tag_string", "tag_string_general", "rating", "asset_license"):
        _optional_text(metadata[field], f"{label}.projected_metadata.{field}")
    if metadata["score"] is not None and (
        isinstance(metadata["score"], bool)
        or not isinstance(metadata["score"], (int, float))
        or not math.isfinite(metadata["score"])
    ):
        raise IntakeError(f"{label}.projected_metadata.score: expected finite number or null")
    provenance = _expect_keys(
        record["provenance"],
        {
            "mode",
            "source_binding_sha256",
            "selection_manifest_sha256",
            "input_row_sha256",
        },
        f"{label}.provenance",
    )
    if provenance["mode"] != PROVENANCE_MODE:
        raise IntakeError(f"{label}: provenance mode drift")
    _required_hex(provenance["source_binding_sha256"], HEX64, f"{label}.provenance.source_binding_sha256")
    _required_hex(
        provenance["selection_manifest_sha256"],
        HEX64,
        f"{label}.provenance.selection_manifest_sha256",
    )
    if provenance["input_row_sha256"] != record["raw_row_sha256"]:
        raise IntakeError(f"{label}: raw-row provenance mismatch")
    if record["ledger_status"] not in {
        "HOLD_LEDGER_HASH_PENDING",
        "HOLD_OPT_OUT_UNKNOWN",
        "NOT_OPTED_OUT",
        "REJECT_OPT_OUT",
    }:
        raise IntakeError(f"{label}: invalid ledger status")


def _validate_limits(value: Any) -> dict[str, int]:
    limits = _expect_keys(
        value,
        {
            "pilot_record_cap",
            "minimum_records",
            "max_touched_rows",
            "max_payload_bytes",
            "max_payload_requests",
        },
        "manifest.limits",
    )
    pilot_cap = _required_int(
        limits["pilot_record_cap"],
        "manifest.limits.pilot_record_cap",
        minimum=1,
    )
    minimum_records = _required_int(
        limits["minimum_records"],
        "manifest.limits.minimum_records",
        minimum=1,
    )
    touched_cap = _required_int(
        limits["max_touched_rows"],
        "manifest.limits.max_touched_rows",
        minimum=1,
    )
    payload_cap = _required_int(
        limits["max_payload_bytes"],
        "manifest.limits.max_payload_bytes",
        minimum=1,
    )
    request_cap = _required_int(
        limits["max_payload_requests"],
        "manifest.limits.max_payload_requests",
        minimum=1,
    )
    if pilot_cap > HARD_PILOT_RECORD_CAP:
        raise IntakeError("manifest.limits.pilot_record_cap exceeds hard cap")
    if minimum_records > HARD_PILOT_RECORD_CAP:
        raise IntakeError("manifest.limits.minimum_records exceeds hard cap")
    if touched_cap > HARD_MAX_TOUCHED_ROWS:
        raise IntakeError("manifest.limits.max_touched_rows exceeds hard cap")
    if payload_cap > HARD_MAX_PAYLOAD_BYTES:
        raise IntakeError("manifest.limits.max_payload_bytes exceeds hard cap")
    if request_cap > HARD_MAX_PAYLOAD_REQUESTS:
        raise IntakeError("manifest.limits.max_payload_requests exceeds hard cap")
    if minimum_records > pilot_cap or pilot_cap > touched_cap:
        raise IntakeError("manifest limits are internally inconsistent")
    return limits


def _validate_selection(value: Any, limits: dict[str, int]) -> dict[str, Any]:
    selection = _expect_keys(
        value,
        {
            "algorithm_version",
            "selection_salt",
            "inventory_status",
            "anchor_column",
            "candidate_count",
            "eligible_candidate_count",
            "selected_windows",
            "selected_rows",
            "touched_rows_upper_bound",
            "planned_payload_requests",
            "planned_payload_bytes",
            "planned_ranges",
            "selection_ready",
            "execution_authorized",
            "hold_reasons",
        },
        "manifest.selection",
    )
    if selection["algorithm_version"] != SAMPLER_VERSION:
        raise IntakeError("manifest.selection algorithm version drift")
    salt = _required_text(selection["selection_salt"], "manifest.selection.selection_salt")
    if selection["inventory_status"] not in {
        "COMPLETE",
        "OFFSET_INDEX_UNAVAILABLE",
        "INCOMPLETE",
        "UNKNOWN",
    }:
        raise IntakeError("manifest.selection inventory status is invalid")
    if selection["anchor_column"] != "id":
        raise IntakeError("manifest.selection anchor column drift")
    candidate_count = _required_int(
        selection["candidate_count"],
        "manifest.selection.candidate_count",
    )
    eligible_count = _required_int(
        selection["eligible_candidate_count"],
        "manifest.selection.eligible_candidate_count",
    )
    if eligible_count > candidate_count:
        raise IntakeError("manifest.selection eligible candidates exceed candidates")

    if not isinstance(selection["selected_windows"], list):
        raise IntakeError("manifest.selection.selected_windows must be a list")
    windows: list[dict[str, Any]] = []
    seen_windows: set[tuple[int, int]] = set()
    logical_intervals: dict[int, list[tuple[int, int]]] = {}
    for index, window_value in enumerate(selection["selected_windows"]):
        label = f"manifest.selection.selected_windows[{index}]"
        window = _expect_keys(
            window_value,
            {
                "row_group_index",
                "anchor_page_ordinal",
                "first_row_index",
                "row_count",
                "selection_key_sha256",
            },
            label,
        )
        group_index = _required_int(window["row_group_index"], f"{label}.row_group_index")
        page_ordinal = _required_int(window["anchor_page_ordinal"], f"{label}.anchor_page_ordinal")
        first_row = _required_int(window["first_row_index"], f"{label}.first_row_index")
        row_count = _required_int(window["row_count"], f"{label}.row_count", minimum=1)
        expected_key = _selection_key(
            group_index,
            {
                "page_ordinal": page_ordinal,
                "first_row_index": first_row,
                "row_count": row_count,
            },
            salt,
        )
        if window["selection_key_sha256"] != expected_key:
            raise IntakeError(f"{label}: selection-key digest mismatch")
        identity = (group_index, page_ordinal)
        if identity in seen_windows:
            raise IntakeError(f"{label}: duplicate selected anchor page")
        seen_windows.add(identity)
        interval = (first_row, first_row + row_count)
        for prior in logical_intervals.setdefault(group_index, []):
            if interval[0] < prior[1] and prior[0] < interval[1]:
                raise IntakeError(f"{label}: selected logical windows overlap")
        logical_intervals[group_index].append(interval)
        windows.append(window)
    expected_window_order = sorted(
        windows,
        key=lambda item: (
            item["selection_key_sha256"],
            item["row_group_index"],
            item["anchor_page_ordinal"],
        ),
    )
    if windows != expected_window_order:
        raise IntakeError("manifest.selection selected windows are not hash-ranked")
    if len(windows) > eligible_count:
        raise IntakeError("manifest.selection selected windows exceed eligible candidates")

    selected_rows = _required_int(
        selection["selected_rows"],
        "manifest.selection.selected_rows",
    )
    if selected_rows != sum(window["row_count"] for window in windows):
        raise IntakeError("manifest.selection selected-row count mismatch")
    touched_rows = _required_int(
        selection["touched_rows_upper_bound"],
        "manifest.selection.touched_rows_upper_bound",
    )
    planned_requests = _required_int(
        selection["planned_payload_requests"],
        "manifest.selection.planned_payload_requests",
    )
    planned_bytes = _required_int(
        selection["planned_payload_bytes"],
        "manifest.selection.planned_payload_bytes",
    )
    if (
        selected_rows > limits["pilot_record_cap"]
        or touched_rows > limits["max_touched_rows"]
        or planned_requests > limits["max_payload_requests"]
        or planned_bytes > limits["max_payload_bytes"]
        or touched_rows < selected_rows
    ):
        raise IntakeError("manifest.selection exceeds a frozen limit")

    if not isinstance(selection["planned_ranges"], list):
        raise IntakeError("manifest.selection.planned_ranges must be a list")
    ranges: list[dict[str, Any]] = []
    selected_groups = {window["row_group_index"] for window in windows}
    data_columns_by_group: dict[int, set[str]] = {}
    for index, range_value in enumerate(selection["planned_ranges"]):
        label = f"manifest.selection.planned_ranges[{index}]"
        planned_range = _expect_keys(
            range_value,
            {
                "row_group_index",
                "column",
                "kind",
                "page_ordinal",
                "start",
                "end",
                "bytes",
            },
            label,
        )
        group_index = _required_int(planned_range["row_group_index"], f"{label}.row_group_index")
        if group_index not in selected_groups:
            raise IntakeError(f"{label}: range has no selected row-group window")
        if planned_range["column"] not in REQUIRED_PROJECTED_COLUMNS:
            raise IntakeError(f"{label}: projected column drift")
        if planned_range["kind"] not in {"data_page", "column_prefix_prerequisite"}:
            raise IntakeError(f"{label}: range kind is invalid")
        if planned_range["kind"] == "data_page":
            _required_int(planned_range["page_ordinal"], f"{label}.page_ordinal")
            data_columns_by_group.setdefault(group_index, set()).add(planned_range["column"])
        elif planned_range["page_ordinal"] is not None:
            raise IntakeError(f"{label}: dictionary range must not carry a page ordinal")
        start = _required_int(planned_range["start"], f"{label}.start")
        end = _required_int(planned_range["end"], f"{label}.end")
        byte_count = _required_int(planned_range["bytes"], f"{label}.bytes", minimum=1)
        if end < start or byte_count != end - start + 1:
            raise IntakeError(f"{label}: byte range arithmetic mismatch")
        ranges.append(planned_range)
    if ranges != sorted(ranges, key=lambda item: (item["start"], item["end"], item["column"])):
        raise IntakeError("manifest.selection planned ranges are not physically sorted")
    for previous, current in zip(ranges, ranges[1:]):
        if current["start"] <= previous["end"]:
            raise IntakeError("manifest.selection planned physical ranges overlap")
    if planned_requests != len(ranges) or planned_bytes != sum(item["bytes"] for item in ranges):
        raise IntakeError("manifest.selection planned-range counters mismatch")
    for group_index in selected_groups:
        if data_columns_by_group.get(group_index) != set(REQUIRED_PROJECTED_COLUMNS):
            raise IntakeError("manifest.selection lacks a projected data page for a selected row group")

    if type(selection["selection_ready"]) is not bool or selection["execution_authorized"] is not False:
        raise IntakeError("manifest.selection execution gate drift")
    reasons = selection["hold_reasons"]
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or not all(isinstance(item, str) and item for item in reasons)
    ):
        raise IntakeError("manifest.selection hold reasons must be unique sorted strings")
    expected_ready = bool(windows) and not reasons
    if selection["selection_ready"] is not expected_ready:
        raise IntakeError("manifest.selection readiness is internally inconsistent")
    if expected_ready:
        if selection["inventory_status"] != "COMPLETE" or selected_rows < limits["minimum_records"]:
            raise IntakeError("manifest.selection ready state lacks complete bounded coverage")
    elif any((windows, ranges, selected_rows, touched_rows, planned_requests, planned_bytes)):
        raise IntakeError("manifest.selection non-ready state must carry zero executable plan")
    return selection


def _validate_ledger_summary(value: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _expect_keys(
        value,
        {
            "ledger_id",
            "ledger_sha256",
            "coverage_mode",
            "coverage_as_of",
            "coverage_evidence_sha256",
            "previous_snapshot_id",
            "entry_count",
            "matched_removals",
            "hash_pending_records",
            "unknown_opt_out_records",
        },
        "manifest.ledger",
    )
    if (
        not isinstance(summary["ledger_id"], str)
        or re.fullmatch(
            r"anima-step2a-removal-[0-9a-f]{24}",
            summary["ledger_id"],
        )
        is None
    ):
        raise IntakeError("manifest.ledger ledger_id is invalid")
    _required_hex(summary["ledger_sha256"], HEX64, "manifest.ledger.ledger_sha256")
    if summary["coverage_mode"] not in {"delta_only", "authoritative_complete_snapshot"}:
        raise IntakeError("manifest.ledger coverage mode is invalid")
    _required_utc_timestamp(summary["coverage_as_of"], "manifest.ledger.coverage_as_of")
    _required_hex(
        summary["coverage_evidence_sha256"],
        HEX64,
        "manifest.ledger.coverage_evidence_sha256",
    )
    _required_text(
        summary["previous_snapshot_id"],
        "manifest.ledger.previous_snapshot_id",
    )
    entry_count = _required_int(summary["entry_count"], "manifest.ledger.entry_count")
    matched = _required_int(summary["matched_removals"], "manifest.ledger.matched_removals")
    hash_pending = _required_int(
        summary["hash_pending_records"],
        "manifest.ledger.hash_pending_records",
    )
    unknown = _required_int(
        summary["unknown_opt_out_records"],
        "manifest.ledger.unknown_opt_out_records",
    )
    status_counts = {
        status: sum(record["ledger_status"] == status for record in records)
        for status in {
            "HOLD_LEDGER_HASH_PENDING",
            "HOLD_OPT_OUT_UNKNOWN",
            "NOT_OPTED_OUT",
            "REJECT_OPT_OUT",
        }
    }
    if (
        matched != status_counts["REJECT_OPT_OUT"]
        or hash_pending != status_counts["HOLD_LEDGER_HASH_PENDING"]
        or unknown != status_counts["HOLD_OPT_OUT_UNKNOWN"]
        or matched > entry_count
    ):
        raise IntakeError("manifest.ledger status counters mismatch")
    if summary["coverage_mode"] == "delta_only" and status_counts["NOT_OPTED_OUT"]:
        raise IntakeError("delta ledger must not emit NOT_OPTED_OUT")
    if summary["coverage_mode"] == "authoritative_complete_snapshot" and status_counts["HOLD_OPT_OUT_UNKNOWN"]:
        raise IntakeError("complete ledger must not leave hashed records at unknown opt-out")
    return summary


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    value = _expect_keys(
        manifest,
        {
            "schema",
            "contract_version",
            "operation",
            "status",
            "training_ready",
            "manifest_id",
            "source_binding",
            "parents",
            "limits",
            "selection",
            "records",
            "ledger",
            "gate",
            "activity",
        },
        "manifest",
    )
    if (
        value["schema"] != MANIFEST_SCHEMA
        or value["contract_version"] != INPUT_POLICY_VERSION
        or value["operation"] not in {"sample_plan", "derive_provenance", "ledger_join"}
        or value["status"] != STATUS
        or value["training_ready"] is not False
    ):
        raise IntakeError("manifest fixed contract fields drift")
    _validate_source_binding(value["source_binding"])
    if not isinstance(value["parents"], list) or not value["parents"]:
        raise IntakeError("manifest.parents must be non-empty")
    for index, parent in enumerate(value["parents"]):
        _validate_parent(parent, f"parents[{index}]")
    limits = _validate_limits(value["limits"])
    selection = None
    if value["selection"] is not None:
        selection = _validate_selection(value["selection"], limits)
    if not isinstance(value["records"], list):
        raise IntakeError("manifest.records must be a list")
    if len(value["records"]) > limits["pilot_record_cap"]:
        raise IntakeError("manifest.records exceed pilot record cap")
    for index, record in enumerate(value["records"]):
        _validate_record(record, index)
    records = value["records"]
    record_order = sorted(
        records,
        key=lambda record: (
            record["source_locator"]["row_group_index"],
            record["source_locator"]["row_index_in_group"],
            record["record_key"],
        ),
    )
    if records != record_order:
        raise IntakeError("manifest.records are not in canonical locator order")
    if len({record["record_key"] for record in records}) != len(records):
        raise IntakeError("manifest.records contain duplicate record keys")
    locators = {
        (
            record["source_locator"]["row_group_index"],
            record["source_locator"]["row_index_in_group"],
        )
        for record in records
    }
    if len(locators) != len(records):
        raise IntakeError("manifest.records contain duplicate row locators")
    if selection is not None:
        for index, record in enumerate(records):
            window = _row_window(
                record["source_locator"]["row_group_index"],
                record["source_locator"]["row_index_in_group"],
                selection,
            )
            if (
                record["source_locator"]["anchor_page_ordinal"] != window["anchor_page_ordinal"]
                or record["source_locator"]["selection_key_sha256"] != window["selection_key_sha256"]
            ):
                raise IntakeError(f"records[{index}]: source locator does not match selection")
            if record["provenance"]["source_binding_sha256"] != value["source_binding"]["source_binding_sha256"]:
                raise IntakeError(f"records[{index}]: source-binding provenance mismatch")
    if records and len({record["provenance"]["selection_manifest_sha256"] for record in records}) != 1:
        raise IntakeError("manifest.records disagree on selection-manifest provenance")

    parent_roles = [parent["role"] for parent in value["parents"]]
    expected_reasons: set[str]
    if value["operation"] == "sample_plan":
        expected_parent_roles = (
            [
                "page_index_inventory",
                "page_index_probe",
                "raw_fixed_footer",
                "offset_index_evidence",
            ]
            if selection is not None and selection["inventory_status"] == "COMPLETE"
            else ["page_index_inventory"]
        )
        if parent_roles != expected_parent_roles or selection is None or records or value["ledger"] is not None:
            raise IntakeError("sample_plan manifest shape drift")
        expected_reasons = {
            "INPUT_SEMANTICS_V2_NOT_APPROVED",
            "NATIVE_SOURCE_VALUE_SEMANTICS_UNVERIFIED",
            "REMOVAL_LEDGER_JOIN_PENDING",
            "HUMAN_REVIEW_PENDING",
            *selection["hold_reasons"],
        }
    elif value["operation"] == "derive_provenance":
        if (
            parent_roles != ["selection_manifest", "bounded_projected_rows"]
            or selection is None
            or selection["selection_ready"] is not True
            or len(records) != selection["selected_rows"]
            or value["ledger"] is not None
            or value["parents"][1]["semantic_id"] is not None
        ):
            raise IntakeError("derive_provenance manifest shape drift")
        selection_parent = value["parents"][0]
        if any(record["provenance"]["selection_manifest_sha256"] != selection_parent["sha256"] for record in records):
            raise IntakeError("derive_provenance selection parent binding mismatch")
        for index, record in enumerate(records):
            expected_status = "HOLD_LEDGER_HASH_PENDING" if record["sha256"] is None else "HOLD_OPT_OUT_UNKNOWN"
            if record["ledger_status"] != expected_status:
                raise IntakeError(f"records[{index}]: pre-ledger status mismatch")
        expected_reasons = {
            "INPUT_SEMANTICS_V2_NOT_APPROVED",
            "REMOVAL_LEDGER_JOIN_PENDING",
            "HUMAN_REVIEW_PENDING",
        }
        if any(record["sha256"] is None for record in records):
            expected_reasons.add("HASH_RECONCILIATION_PENDING")
    else:
        if (
            parent_roles
            != [
                "derived_provenance_manifest",
                "removal_ledger",
                "ledger_coverage_evidence",
                "trusted_removal_ledger_head",
                "previous_removal_snapshot",
            ]
            or selection is None
            or selection["selection_ready"] is not True
            or len(records) != selection["selected_rows"]
            or value["ledger"] is None
            or value["parents"][2]["semantic_id"] is not None
        ):
            raise IntakeError("ledger_join manifest shape drift")
        _validate_ledger_summary(value["ledger"], records)
        expected_reasons = {
            "INPUT_SEMANTICS_V2_NOT_APPROVED",
            "HUMAN_REVIEW_PENDING",
        }
        if any(record["ledger_status"] == "HOLD_LEDGER_HASH_PENDING" for record in records):
            expected_reasons.add("HASH_RECONCILIATION_PENDING")
        if any(record["ledger_status"] == "HOLD_OPT_OUT_UNKNOWN" for record in records):
            expected_reasons.add("OPT_OUT_COVERAGE_INCOMPLETE")
    gate = _expect_keys(value["gate"], {"g1", "reasons"}, "manifest.gate")
    if gate["g1"] != "HOLD" or not isinstance(gate["reasons"], list) or not gate["reasons"]:
        raise IntakeError("manifest.gate must remain HOLD with reasons")
    if gate["reasons"] != sorted(set(gate["reasons"])) or not all(
        isinstance(item, str) and item for item in gate["reasons"]
    ):
        raise IntakeError("manifest.gate reasons must be unique sorted strings")
    if set(gate["reasons"]) != expected_reasons:
        raise IntakeError("manifest.gate reasons do not match operation state")
    if value["activity"] != _activity():
        raise IntakeError("manifest activity must remain zero")
    if value["manifest_id"] != _manifest_identifier(value):
        raise IntakeError("manifest semantic identifier mismatch")
    return value


def _resolve_manifest_parent(parent: dict[str, Any]) -> Path:
    path = Path(parent["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _absolute_lexical(path)


def _verify_manifest_lineage(
    value: dict[str, Any],
    manifest_path: Path,
    stack: tuple[Path, ...],
) -> None:
    parent_paths: list[Path] = []
    observed_parent_sha256s: list[str] = []
    for index, parent in enumerate(value["parents"]):
        parent_path = _resolve_manifest_parent(parent)
        max_bytes = _parent_max_bytes(parent["role"])
        observed_sha256 = _file_sha256(parent_path, max_bytes=max_bytes)
        if observed_sha256 != parent["sha256"]:
            raise IntakeError(f"parents[{index}]: parent byte binding mismatch")
        parent_paths.append(parent_path)
        observed_parent_sha256s.append(observed_sha256)

    if value["operation"] == "sample_plan":
        inventory_path = parent_paths[0]
        inventory_value, inventory_sha256 = _load_json_with_sha256(inventory_path)
        if inventory_sha256 != value["parents"][0]["sha256"]:
            raise IntakeError("sample_plan inventory changed between binding and validation")
        inventory = _validate_inventory(inventory_value, inventory_path)
        if value["parents"][0]["semantic_id"] != _inventory_semantic_id(inventory):
            raise IntakeError("sample_plan inventory semantic binding mismatch")
        limits = value["limits"]
        expected_selection = _build_selection(
            inventory,
            salt=value["selection"]["selection_salt"],
            pilot_record_cap=limits["pilot_record_cap"],
            minimum_records=limits["minimum_records"],
            max_touched_rows=limits["max_touched_rows"],
            max_payload_bytes=limits["max_payload_bytes"],
            max_payload_requests=limits["max_payload_requests"],
        )
        if value["selection"] != expected_selection:
            raise IntakeError("sample_plan selection does not recompute from inventory")
        probe_path = _inventory_probe_path(inventory, inventory_path)
        evidence_path = _inventory_evidence_path(inventory, inventory_path)
        if probe_path is None or evidence_path is None:
            if len(parent_paths) != 1:
                raise IntakeError("non-complete sample_plan has unexpected probe/evidence parents")
        else:
            probe_value, probe_sha256 = _load_json_with_sha256(probe_path)
            if probe_sha256 != inventory["page_index_probe"]["sha256"]:
                raise IntakeError("sample_plan page-index probe changed during lineage replay")
            probe = _validate_page_index_probe(probe_value, probe_path)
            raw_footer_path = _resolve_relative_reference(
                probe_path,
                probe["raw_footer_evidence"]["path"],
            )
            if (
                len(parent_paths) != 4
                or parent_paths[1] != probe_path
                or parent_paths[2] != raw_footer_path
                or parent_paths[3] != evidence_path
                or value["parents"][1]["sha256"] != inventory["page_index_probe"]["sha256"]
                or value["parents"][2]["sha256"] != probe["raw_footer_evidence"]["sha256"]
                or value["parents"][3]["sha256"] != inventory["offset_index_evidence"]["sha256"]
                or value["parents"][1]["semantic_id"] != inventory["page_index_probe"]["probe_id"]
                or value["parents"][2]["semantic_id"] != f"{probe['probe_id']}::raw-footer-evidence"
                or value["parents"][3]["semantic_id"] != f"{_inventory_semantic_id(inventory)}::offset-index-evidence"
            ):
                raise IntakeError("sample_plan page-index/raw-footer/evidence parent binding mismatch")
    elif value["operation"] == "derive_provenance":
        selection_parent, selection_parent_sha256 = _load_manifest_with_sha256(
            parent_paths[0],
            "sample_plan",
            _stack=stack,
        )
        if (
            selection_parent_sha256 != value["parents"][0]["sha256"]
            or value["parents"][0]["semantic_id"] != selection_parent["manifest_id"]
        ):
            raise IntakeError("derive_provenance selection semantic binding mismatch")
        if (
            value["source_binding"] != selection_parent["source_binding"]
            or value["limits"] != selection_parent["limits"]
            or value["selection"] != selection_parent["selection"]
        ):
            raise IntakeError("derive_provenance inherited contract drift")
        rows, rows_sha256 = _read_rows_with_sha256(
            parent_paths[1],
            value["selection"]["selected_rows"],
        )
        if rows_sha256 != value["parents"][1]["sha256"]:
            raise IntakeError("derive_provenance bounded rows changed between binding and validation")
        expected_records = [
            _derive_record(
                row,
                value["selection"],
                value["parents"][0]["sha256"],
                value["source_binding"]["source_binding_sha256"],
            )
            for row in rows
        ]
        expected_records.sort(
            key=lambda record: (
                record["source_locator"]["row_group_index"],
                record["source_locator"]["row_index_in_group"],
                record["record_key"],
            )
        )
        if value["records"] != expected_records:
            raise IntakeError("derive_provenance records do not recompute from bounded rows")
    elif value["operation"] == "ledger_join":
        derived_parent, derived_parent_sha256 = _load_manifest_with_sha256(
            parent_paths[0],
            "derive_provenance",
            _stack=stack,
        )
        if (
            derived_parent_sha256 != value["parents"][0]["sha256"]
            or value["parents"][0]["semantic_id"] != derived_parent["manifest_id"]
        ):
            raise IntakeError("ledger_join derived-manifest semantic binding mismatch")
        if (
            value["source_binding"] != derived_parent["source_binding"]
            or value["limits"] != derived_parent["limits"]
            or value["selection"] != derived_parent["selection"]
        ):
            raise IntakeError("ledger_join inherited contract drift")
        ledger, ledger_sha256 = _load_ledger_with_sha256(parent_paths[1])
        if ledger_sha256 != value["parents"][1]["sha256"] or value["parents"][1]["semantic_id"] != ledger["ledger_id"]:
            raise IntakeError("ledger_join removal-ledger semantic binding mismatch")
        if (
            value["parents"][2]["semantic_id"] is not None
            or observed_parent_sha256s[2] != ledger["coverage"]["evidence"]["sha256"]
            or value["parents"][2]["sha256"] != ledger["coverage"]["evidence"]["sha256"]
        ):
            raise IntakeError("ledger_join coverage evidence binding mismatch")
        trusted_head = _trusted_ledger_head()
        if (
            parent_paths[3] != _absolute_lexical(PROJECT_ROOT / REMOVAL_LEDGER_HEAD_RELATIVE_PATH)
            or value["parents"][3]["sha256"] != REMOVAL_LEDGER_HEAD_SHA256
            or value["parents"][3]["semantic_id"] != trusted_head["head_id"]
        ):
            raise IntakeError("ledger_join trusted-head parent binding mismatch")
        _verify_ledger_is_trusted_head(
            parent_paths[1],
            ledger,
            trusted_head,
            ledger_sha256=ledger_sha256,
        )
        previous_path = parent_paths[4]
        previous = _verify_previous_snapshot(
            ledger,
            str(previous_path),
        )
        if value["parents"][4]["sha256"] != ledger["previous_snapshot"]["sha256"] or value["parents"][4][
            "semantic_id"
        ] != _snapshot_id(previous):
            raise IntakeError("ledger_join previous-snapshot semantic binding mismatch")
        expected_records, expected_summary, expected_reasons = _compute_ledger_join(
            derived_parent,
            value["parents"][0]["sha256"],
            ledger,
            value["parents"][1]["sha256"],
            previous,
        )
        if (
            value["records"] != expected_records
            or value["ledger"] != expected_summary
            or value["gate"]["reasons"] != sorted(set(expected_reasons))
        ):
            raise IntakeError("ledger_join result does not recompute from bound parents")


def _load_manifest_with_sha256(
    path: Path,
    expected_operation: str | None = None,
    *,
    _stack: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], str]:
    resolved = _absolute_lexical(path)
    if resolved in _stack:
        raise IntakeError("manifest parent cycle detected")
    raw_value, byte_sha256 = _load_json_with_sha256(resolved)
    value = _validate_manifest(raw_value)
    if expected_operation is not None and value["operation"] != expected_operation:
        raise IntakeError(f"expected parent operation {expected_operation}")
    _verify_manifest_lineage(value, resolved, (*_stack, resolved))
    return value, byte_sha256


def _load_manifest(
    path: Path,
    expected_operation: str | None = None,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    return _load_manifest_with_sha256(
        path,
        expected_operation,
        _stack=_stack,
    )[0]


def _projected_chunk_universe_sha256() -> str:
    truth = _footer_projection_truth()
    return _digest(
        [
            {
                "row_group_index": group_index,
                "row_count": truth[group_index]["row_count"],
                "columns": [
                    {
                        "path": path,
                        "start": truth[group_index]["columns"][path][0],
                        "end": truth[group_index]["columns"][path][1],
                    }
                    for path in REQUIRED_PROJECTED_COLUMNS
                ],
            }
            for group_index in sorted(truth)
        ]
    )


def _parse_raw_footer_offset_index_truth(
    raw_footer: bytes,
) -> dict[tuple[int, str], tuple[int, int]]:
    try:
        metadata = footer_plan._parse_parquet_metadata(
            raw_footer,
            file_size=TARGET_SIZE_BYTES,
        )
    except footer_plan.FooterPlanError as exc:
        raise IntakeError("raw footer Compact Protocol parse failed") from exc
    footer_truth = _footer_projection_truth()
    if metadata.version != 2 or metadata.num_rows != 8_005_010 or len(metadata.row_groups) != 8:
        raise IntakeError("raw footer target universe drift")

    result: dict[tuple[int, str], tuple[int, int]] = {}
    index_ranges: list[tuple[int, int, tuple[int, str]]] = []
    all_chunk_ranges = [(column.start, column.end) for group in metadata.row_groups for column in group.columns]
    footer_start = TARGET_SIZE_BYTES - 8 - len(raw_footer)
    for expected_group_index, group in enumerate(metadata.row_groups):
        if group.index != expected_group_index or group.num_rows != footer_truth[expected_group_index]["row_count"]:
            raise IntakeError("raw footer row-group universe drift")
        projected: dict[str, footer_plan.ColumnChunk] = {}
        for column in group.columns:
            path = ".".join(column.path)
            if path in REQUIRED_PROJECTED_COLUMNS:
                if path in projected:
                    raise IntakeError("raw footer contains a duplicate projected column")
                projected[path] = column
        if set(projected) != set(REQUIRED_PROJECTED_COLUMNS):
            raise IntakeError("raw footer lacks a required projected column")
        for path in REQUIRED_PROJECTED_COLUMNS:
            column = projected[path]
            coordinate = (expected_group_index, path)
            if (column.start, column.end) != footer_truth[expected_group_index]["columns"][path]:
                raise IntakeError("raw footer projected chunk does not match frozen footer plan")
            offset = column.offset_index_offset
            length = column.offset_index_length
            if offset is None or length is None:
                raise IntakeError("raw footer required ColumnChunk lacks fields 4-5")
            end = offset + length - 1
            if offset < 4 or length <= 0 or end >= footer_start:
                raise IntakeError("raw footer OffsetIndex coordinate is outside the target file")
            result[coordinate] = (offset, length)
            index_ranges.append((offset, end, coordinate))

    expected_coordinates = {(group_index, path) for group_index in range(8) for path in REQUIRED_PROJECTED_COLUMNS}
    if set(result) != expected_coordinates:
        raise IntakeError("raw footer OffsetIndex coordinate universe is incomplete")
    ordered_indexes = sorted(index_ranges)
    for previous, current in zip(ordered_indexes, ordered_indexes[1:]):
        if current[0] <= previous[1]:
            raise IntakeError("raw footer OffsetIndex ranges overlap")
    for index_start, index_end, _ in index_ranges:
        for chunk_start, chunk_end in all_chunk_ranges:
            if index_start <= chunk_end and chunk_start <= index_end:
                raise IntakeError("raw footer OffsetIndex range overlaps projected column data")
    return result


def _load_raw_footer_offset_index_truth(
    probe: dict[str, Any],
    probe_path: Path,
) -> tuple[Path, dict[tuple[int, str], tuple[int, int]]]:
    descriptor = _expect_keys(
        probe["raw_footer_evidence"],
        {"path", "sha256", "size_bytes", "serialization"},
        "page-index probe.raw_footer_evidence",
    )
    _required_text(descriptor["path"], "page-index probe.raw_footer_evidence.path")
    if (
        descriptor["sha256"] != TARGET_FOOTER_SHA256
        or descriptor["size_bytes"] != TARGET_FOOTER_LENGTH
        or descriptor["serialization"] != "Apache Parquet FileMetaData TCompactProtocol raw footer"
    ):
        raise IntakeError("page-index probe raw-footer descriptor drift")
    raw_footer_path = _resolve_relative_reference(probe_path, descriptor["path"])
    raw_footer, raw_footer_sha256 = _load_bytes_with_sha256(
        raw_footer_path,
        max_bytes=MAX_RAW_FOOTER_EVIDENCE_BYTES,
    )
    if len(raw_footer) != TARGET_FOOTER_LENGTH or raw_footer_sha256 != TARGET_FOOTER_SHA256:
        raise IntakeError("page-index probe raw-footer byte binding mismatch")
    return raw_footer_path, _parse_raw_footer_offset_index_truth(raw_footer)


def _page_index_probe_identifier(probe: dict[str, Any]) -> str:
    identity = copy.deepcopy(probe)
    identity.pop("probe_id", None)
    identity["raw_footer_evidence"].pop("path", None)
    identity["offset_index_evidence"].pop("path", None)
    return "anima-step2a-page-index-probe-" + _digest(identity)[:24]


def _resolve_relative_reference(owner_path: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = _absolute_lexical(owner_path).parent / path
    return _absolute_lexical(path)


def _validate_page_index_probe(
    value: Any,
    probe_path: Path,
) -> dict[str, Any]:
    probe = _expect_keys(
        value,
        {
            "schema",
            "probe_id",
            "status",
            "source",
            "footer_binding",
            "projected_columns",
            "parser",
            "range_transport",
            "raw_footer_evidence",
            "offset_index_evidence",
            "offset_index_ranges",
            "activity",
        },
        "page-index probe",
    )
    if (
        probe["schema"] != PAGE_INDEX_PROBE_SCHEMA
        or probe["status"] != "COMPLETE"
        or probe["parser"] != "bounded-raw-footer-fields-4-5-and-offset-index-v1"
        or probe["range_transport"] != "HTTP 206 exact Content-Range v1"
    ):
        raise IntakeError("page-index probe fixed contract fields drift")
    source = _expect_keys(
        probe["source"],
        {"name", "revision", "filename", "size_bytes"},
        "page-index probe.source",
    )
    if source != {
        "name": TARGET_SOURCE,
        "revision": TARGET_REVISION,
        "filename": TARGET_FILENAME,
        "size_bytes": TARGET_SIZE_BYTES,
    }:
        raise IntakeError("page-index probe source binding mismatch")
    footer_binding = _expect_keys(
        probe["footer_binding"],
        {
            "footer_plan_v1_sha256",
            "footer_sha256",
            "row_group_count",
            "num_rows",
            "projected_chunk_universe_sha256",
        },
        "page-index probe.footer_binding",
    )
    if footer_binding != {
        "footer_plan_v1_sha256": FOOTER_PLAN_SHA256,
        "footer_sha256": TARGET_FOOTER_SHA256,
        "row_group_count": 8,
        "num_rows": 8_005_010,
        "projected_chunk_universe_sha256": _projected_chunk_universe_sha256(),
    }:
        raise IntakeError("page-index probe footer binding mismatch")
    if probe["projected_columns"] != list(REQUIRED_PROJECTED_COLUMNS):
        raise IntakeError("page-index probe projected-column contract drift")
    raw_footer_path, footer_index_truth = _load_raw_footer_offset_index_truth(
        probe,
        probe_path,
    )
    descriptor = _expect_keys(
        probe["offset_index_evidence"],
        {"path", "sha256", "size_bytes", "serialization"},
        "page-index probe.offset_index_evidence",
    )
    _required_text(descriptor["path"], "page-index probe.offset_index_evidence.path")
    expected_aggregate_sha = _required_hex(
        descriptor["sha256"],
        HEX64,
        "page-index probe.offset_index_evidence.sha256",
    )
    expected_aggregate_size = _required_int(
        descriptor["size_bytes"],
        "page-index probe.offset_index_evidence.size_bytes",
        minimum=1,
    )
    if (
        expected_aggregate_size > MAX_OFFSET_INDEX_EVIDENCE_BYTES
        or descriptor["serialization"] != "Apache Thrift TCompactProtocol OffsetIndex concatenation"
    ):
        raise IntakeError("page-index probe OffsetIndex evidence descriptor drift")
    evidence_path = _resolve_relative_reference(probe_path, descriptor["path"])
    if evidence_path == raw_footer_path:
        raise IntakeError("page-index probe evidence roles share one input path")
    evidence_bytes = _load_bytes(
        evidence_path,
        max_bytes=MAX_OFFSET_INDEX_EVIDENCE_BYTES,
    )
    if len(evidence_bytes) != expected_aggregate_size or _digest(evidence_bytes) != expected_aggregate_sha:
        raise IntakeError("page-index probe aggregate evidence byte binding mismatch")

    if not isinstance(probe["offset_index_ranges"], list) or len(probe["offset_index_ranges"]) != 56:
        raise IntakeError("page-index probe must contain exactly 56 OffsetIndex ranges")
    expected_coordinates = [(group_index, path) for group_index in range(8) for path in REQUIRED_PROJECTED_COLUMNS]
    evidence_ranges: list[tuple[int, int]] = []
    remote_ranges: list[tuple[int, int]] = []
    seen_coordinates: set[tuple[int, str]] = set()
    total_bytes = 0
    for index, range_value in enumerate(probe["offset_index_ranges"]):
        label = f"page-index probe.offset_index_ranges[{index}]"
        item = _expect_keys(
            range_value,
            {
                "row_group_index",
                "column",
                "offset",
                "length",
                "evidence_offset",
                "sha256",
                "http_range",
            },
            label,
        )
        group_index = _required_int(
            item["row_group_index"],
            f"{label}.row_group_index",
        )
        column_path = _required_text(item["column"], f"{label}.column")
        coordinate = (group_index, column_path)
        if coordinate in seen_coordinates:
            raise IntakeError(f"{label}: duplicate range coordinate")
        seen_coordinates.add(coordinate)
        if coordinate != expected_coordinates[index]:
            raise IntakeError(f"{label}: range coordinate order drift")
        offset = _required_int(item["offset"], f"{label}.offset")
        length = _required_int(item["length"], f"{label}.length", minimum=1)
        if (offset, length) != footer_index_truth[coordinate]:
            raise IntakeError(f"{label}: range does not match raw footer fields 4-5")
        evidence_offset = _required_int(item["evidence_offset"], f"{label}.evidence_offset")
        slice_sha = _required_hex(item["sha256"], HEX64, f"{label}.sha256")
        remote_end = offset + length - 1
        if remote_end >= TARGET_SIZE_BYTES - 8 - TARGET_FOOTER_LENGTH:
            raise IntakeError(f"{label}: remote OffsetIndex range escapes pre-footer bytes")
        if evidence_offset + length > len(evidence_bytes):
            raise IntakeError(f"{label}: aggregate evidence slice escapes file")
        raw_index = evidence_bytes[evidence_offset : evidence_offset + length]
        if _digest(raw_index) != slice_sha:
            raise IntakeError(f"{label}: aggregate evidence slice digest mismatch")
        _parse_offset_index(raw_index)
        response = _expect_keys(
            item["http_range"],
            {
                "status_code",
                "content_range",
                "content_length",
                "bytes_received",
                "sha256",
            },
            f"{label}.http_range",
        )
        if response != {
            "status_code": 206,
            "content_range": f"bytes {offset}-{remote_end}/{TARGET_SIZE_BYTES}",
            "content_length": length,
            "bytes_received": length,
            "sha256": slice_sha,
        }:
            raise IntakeError(f"{label}: HTTP Range evidence mismatch")
        evidence_ranges.append((evidence_offset, evidence_offset + length))
        remote_ranges.append((offset, remote_end))
        total_bytes += length
    for ranges, label in (
        (sorted(evidence_ranges), "aggregate evidence"),
        (sorted(remote_ranges), "remote OffsetIndex"),
    ):
        for previous, current in zip(ranges, ranges[1:]):
            overlap = current[0] < previous[1] if label == "aggregate evidence" else current[0] <= previous[1]
            if overlap:
                raise IntakeError(f"page-index probe {label} ranges overlap")
    ordered_evidence = sorted(evidence_ranges)
    if (
        ordered_evidence[0][0] != 0
        or ordered_evidence[-1][1] != len(evidence_bytes)
        or any(current[0] != previous[1] for previous, current in zip(ordered_evidence, ordered_evidence[1:]))
    ):
        raise IntakeError("page-index probe aggregate evidence slices contain gaps")

    expected_activity = {
        "metadata_range_requests_performed": 59,
        "metadata_range_bytes_received": TARGET_FOOTER_LENGTH + 12 + total_bytes,
        "column_payload_requests_performed": 0,
        "column_payload_bytes_received": 0,
        "image_files_downloaded": 0,
        "training_runs_started": 0,
    }
    if probe["activity"] != expected_activity:
        raise IntakeError("page-index probe activity accounting mismatch")
    if expected_activity["metadata_range_bytes_received"] > MAX_METADATA_RANGE_BYTES:
        raise IntakeError("page-index probe metadata bytes exceed frozen cap")
    if probe["probe_id"] != _page_index_probe_identifier(probe):
        raise IntakeError("page-index probe semantic identifier mismatch")
    return probe


def _inventory_evidence_path(
    inventory: dict[str, Any],
    inventory_path: Path | None,
) -> Path | None:
    descriptor = inventory["offset_index_evidence"]
    if descriptor is None:
        return None
    if inventory_path is None:
        raise IntakeError("complete page inventory requires its file path for evidence binding")
    raw_path = Path(descriptor["path"])
    if not raw_path.is_absolute():
        raw_path = _absolute_lexical(inventory_path).parent / raw_path
    return _absolute_lexical(raw_path)


def _inventory_probe_path(
    inventory: dict[str, Any],
    inventory_path: Path | None,
) -> Path | None:
    descriptor = inventory["page_index_probe"]
    if descriptor is None:
        return None
    if inventory_path is None:
        raise IntakeError("complete page inventory requires its file path for probe binding")
    return _resolve_relative_reference(inventory_path, descriptor["path"])


def _validate_inventory(
    value: Any,
    inventory_path: Path | None = None,
) -> dict[str, Any]:
    inventory = _expect_keys(
        value,
        {
            "schema",
            "status",
            "source",
            "footer_plan_sha256",
            "footer_sha256",
            "page_index_probe",
            "offset_index_evidence",
            "anchor_column",
            "projected_columns",
            "row_groups",
            "activity",
        },
        "page inventory",
    )
    if inventory["schema"] != PAGE_INVENTORY_SCHEMA:
        raise IntakeError("page inventory schema mismatch")
    if inventory["status"] not in {
        "COMPLETE",
        "OFFSET_INDEX_UNAVAILABLE",
        "INCOMPLETE",
        "UNKNOWN",
    }:
        raise IntakeError("page inventory status is invalid")
    source = _expect_keys(
        inventory["source"],
        {"name", "revision", "filename", "size_bytes"},
        "page inventory.source",
    )
    if source != {
        "name": TARGET_SOURCE,
        "revision": TARGET_REVISION,
        "filename": TARGET_FILENAME,
        "size_bytes": TARGET_SIZE_BYTES,
    }:
        raise IntakeError("page inventory source binding mismatch")
    if inventory["footer_plan_sha256"] != FOOTER_PLAN_SHA256 or inventory["footer_sha256"] != TARGET_FOOTER_SHA256:
        raise IntakeError("page inventory footer binding mismatch")
    if inventory["anchor_column"] != "id":
        raise IntakeError("page inventory anchor column must be id")
    if inventory["projected_columns"] != list(REQUIRED_PROJECTED_COLUMNS):
        raise IntakeError("page inventory projected-column contract drift")
    expected_activity = {
        "metadata_range_requests_performed": 0,
        "metadata_range_bytes_received": 0,
        "column_payload_requests_performed": 0,
        "column_payload_bytes_received": 0,
        "image_files_downloaded": 0,
        "training_runs_started": 0,
    }
    activity = _expect_keys(inventory["activity"], expected_activity, "page inventory.activity")
    for key, item in activity.items():
        _required_int(item, f"page inventory.activity.{key}")
    if any(activity[key] != 0 for key in expected_activity if not key.startswith("metadata_range_")):
        raise IntakeError("page inventory records forbidden payload/image/training activity")
    if (
        activity["metadata_range_requests_performed"] > MAX_METADATA_RANGE_REQUESTS
        or activity["metadata_range_bytes_received"] > MAX_METADATA_RANGE_BYTES
    ):
        raise IntakeError("page inventory metadata activity exceeds frozen caps")
    if not isinstance(inventory["row_groups"], list):
        raise IntakeError("page inventory.row_groups must be a list")

    evidence_bytes: bytes | None = None
    evidence_path: Path | None = None
    probe: dict[str, Any] | None = None
    probe_path: Path | None = None
    if inventory["status"] == "COMPLETE":
        if (
            not inventory["row_groups"]
            or inventory["page_index_probe"] is None
            or inventory["offset_index_evidence"] is None
        ):
            raise IntakeError("complete page inventory lacks row groups, probe, or index evidence")
        probe_descriptor = _expect_keys(
            inventory["page_index_probe"],
            {"path", "sha256", "probe_id"},
            "page inventory.page_index_probe",
        )
        _required_text(probe_descriptor["path"], "page inventory.page_index_probe.path")
        _required_hex(
            probe_descriptor["sha256"],
            HEX64,
            "page inventory.page_index_probe.sha256",
        )
        _required_text(
            probe_descriptor["probe_id"],
            "page inventory.page_index_probe.probe_id",
        )
        probe_path = _inventory_probe_path(inventory, inventory_path)
        assert probe_path is not None
        probe_value, probe_sha256 = _load_json_with_sha256(probe_path)
        if probe_sha256 != probe_descriptor["sha256"]:
            raise IntakeError("page-index probe parent byte binding mismatch")
        probe = _validate_page_index_probe(probe_value, probe_path)
        if probe["probe_id"] != probe_descriptor["probe_id"]:
            raise IntakeError("page-index probe parent semantic binding mismatch")
        descriptor = _expect_keys(
            inventory["offset_index_evidence"],
            {"path", "sha256", "size_bytes", "serialization"},
            "page inventory.offset_index_evidence",
        )
        _required_text(descriptor["path"], "page inventory.offset_index_evidence.path")
        _required_hex(
            descriptor["sha256"],
            HEX64,
            "page inventory.offset_index_evidence.sha256",
        )
        evidence_size = _required_int(
            descriptor["size_bytes"],
            "page inventory.offset_index_evidence.size_bytes",
            minimum=1,
        )
        if evidence_size > MAX_OFFSET_INDEX_EVIDENCE_BYTES:
            raise IntakeError("OffsetIndex evidence descriptor exceeds hard cap")
        if descriptor["serialization"] != "Apache Thrift TCompactProtocol OffsetIndex concatenation":
            raise IntakeError("OffsetIndex evidence serialization drift")
        evidence_path = _inventory_evidence_path(inventory, inventory_path)
        assert evidence_path is not None
        probe_evidence_path = _resolve_relative_reference(
            probe_path,
            probe["offset_index_evidence"]["path"],
        )
        if evidence_path != probe_evidence_path or {
            key: value for key, value in descriptor.items() if key != "path"
        } != {key: value for key, value in probe["offset_index_evidence"].items() if key != "path"}:
            raise IntakeError("inventory OffsetIndex evidence does not match probe parent")
        evidence_bytes = _load_bytes(
            evidence_path,
            max_bytes=MAX_OFFSET_INDEX_EVIDENCE_BYTES,
        )
        if len(evidence_bytes) != evidence_size or _digest(evidence_bytes) != descriptor["sha256"]:
            raise IntakeError("OffsetIndex aggregate evidence byte binding mismatch")
        if (
            activity["metadata_range_requests_performed"] != 59
            or activity["metadata_range_bytes_received"] != TARGET_FOOTER_LENGTH + 12 + evidence_size
        ):
            raise IntakeError("complete inventory metadata accounting does not match evidence")
        if activity != probe["activity"]:
            raise IntakeError("inventory metadata activity does not match page-index probe")
    else:
        if (
            inventory["row_groups"]
            or inventory["page_index_probe"] is not None
            or inventory["offset_index_evidence"] is not None
        ):
            raise IntakeError("non-complete page inventory must not carry executable evidence")

    footer_truth = _footer_projection_truth()
    seen_group_indexes: set[int] = set()
    global_chunk_ranges: list[tuple[int, int, str]] = []
    global_index_ranges: list[tuple[int, int, str]] = []
    evidence_ranges: list[tuple[int, int, str]] = []
    footer_start = TARGET_SIZE_BYTES - 8 - TARGET_FOOTER_LENGTH
    for group_position, group_value in enumerate(inventory["row_groups"]):
        group = _expect_keys(
            group_value,
            {"row_group_index", "row_count", "columns"},
            f"page inventory.row_groups[{group_position}]",
        )
        group_index = _required_int(
            group["row_group_index"],
            f"page inventory.row_groups[{group_position}].row_group_index",
        )
        if group_index in seen_group_indexes:
            raise IntakeError("duplicate row-group index")
        seen_group_indexes.add(group_index)
        if group_index not in footer_truth:
            raise IntakeError("row-group index is outside footer-bound target universe")
        row_count = _required_int(
            group["row_count"],
            f"page inventory.row_groups[{group_position}].row_count",
            minimum=1,
        )
        if row_count != footer_truth[group_index]["row_count"]:
            raise IntakeError("row-group row count does not match footer-bound truth")
        if not isinstance(group["columns"], list):
            raise IntakeError("page inventory columns must be a list")
        paths = []
        chunk_ranges: list[tuple[int, int]] = []
        for column_position, column_value in enumerate(group["columns"]):
            label = f"page inventory.row_groups[{group_position}].columns[{column_position}]"
            column = _expect_keys(
                column_value,
                {
                    "path",
                    "chunk_start",
                    "chunk_end",
                    "offset_index",
                    "prerequisite_ranges",
                    "pages",
                },
                label,
            )
            path = _required_text(column["path"], f"{label}.path")
            paths.append(path)
            chunk_start = _required_int(column["chunk_start"], f"{label}.chunk_start")
            chunk_end = _required_int(column["chunk_end"], f"{label}.chunk_end")
            if chunk_end < chunk_start or chunk_end >= footer_start:
                raise IntakeError(f"{label}: invalid column-chunk range")
            if (chunk_start, chunk_end) != footer_truth[group_index]["columns"].get(path):
                raise IntakeError(f"{label}: column chunk does not match footer-bound truth")
            chunk_ranges.append((chunk_start, chunk_end))
            global_chunk_ranges.append((chunk_start, chunk_end, label))
            offset_index = _expect_keys(
                column["offset_index"],
                {"offset", "length", "evidence_offset", "sha256"},
                f"{label}.offset_index",
            )
            offset = _required_int(offset_index["offset"], f"{label}.offset_index.offset")
            length = _required_int(offset_index["length"], f"{label}.offset_index.length", minimum=1)
            evidence_offset = _required_int(
                offset_index["evidence_offset"],
                f"{label}.offset_index.evidence_offset",
            )
            expected_index_sha = _required_hex(
                offset_index["sha256"],
                HEX64,
                f"{label}.offset_index.sha256",
            )
            if offset + length > footer_start:
                raise IntakeError(f"{label}: OffsetIndex range exceeds target file")
            global_index_ranges.append((offset, offset + length - 1, label))
            if evidence_bytes is None or evidence_offset + length > len(evidence_bytes):
                raise IntakeError(f"{label}: OffsetIndex evidence slice escapes aggregate bytes")
            raw_index = evidence_bytes[evidence_offset : evidence_offset + length]
            if _digest(raw_index) != expected_index_sha:
                raise IntakeError(f"{label}: OffsetIndex slice digest mismatch")
            assert probe is not None
            probe_item = probe["offset_index_ranges"][
                group_index * len(REQUIRED_PROJECTED_COLUMNS) + REQUIRED_PROJECTED_COLUMNS.index(path)
            ]
            if {
                "row_group_index": group_index,
                "column": path,
                "offset": offset,
                "length": length,
                "evidence_offset": evidence_offset,
                "sha256": expected_index_sha,
            } != {
                key: probe_item[key]
                for key in {
                    "row_group_index",
                    "column",
                    "offset",
                    "length",
                    "evidence_offset",
                    "sha256",
                }
            }:
                raise IntakeError(f"{label}: OffsetIndex range does not match page-index probe")
            evidence_ranges.append((evidence_offset, evidence_offset + length, label))
            parsed_pages = _parse_offset_index(raw_index)
            for page_index, parsed_page in enumerate(parsed_pages):
                next_first = (
                    parsed_pages[page_index + 1]["first_row_index"] if page_index + 1 < len(parsed_pages) else row_count
                )
                parsed_page["row_count"] = next_first - parsed_page["first_row_index"]
                if parsed_page["row_count"] <= 0:
                    raise IntakeError(f"{label}: OffsetIndex row coverage exceeds row group")
            if not isinstance(column["prerequisite_ranges"], list):
                raise IntakeError(f"{label}.prerequisite_ranges must be a list")
            prerequisites: list[tuple[int, int]] = []
            for prerequisite_index, prerequisite_value in enumerate(column["prerequisite_ranges"]):
                prerequisite = _expect_keys(
                    prerequisite_value,
                    {"kind", "start", "end"},
                    f"{label}.prerequisite_ranges[{prerequisite_index}]",
                )
                if prerequisite["kind"] != "column_prefix_prerequisite":
                    raise IntakeError(f"{label}: unsupported prerequisite range kind")
                start = _required_int(prerequisite["start"], f"{label}.prerequisite.start")
                end = _required_int(prerequisite["end"], f"{label}.prerequisite.end")
                if end < start or start < chunk_start or end > chunk_end:
                    raise IntakeError(f"{label}: prerequisite range escapes column chunk")
                prerequisites.append((start, end))
            if not isinstance(column["pages"], list) or column["pages"] != parsed_pages:
                raise IntakeError(f"{label}.pages do not match raw OffsetIndex evidence")
            if not column["pages"]:
                raise IntakeError(f"{label}.pages must be a non-empty list")
            expected_row = 0
            previous_page_end = -1
            page_ranges: list[tuple[int, int]] = []
            for page_position, page_value in enumerate(column["pages"]):
                page = _expect_keys(
                    page_value,
                    {
                        "page_ordinal",
                        "first_row_index",
                        "row_count",
                        "offset",
                        "compressed_page_size",
                    },
                    f"{label}.pages[{page_position}]",
                )
                if page["page_ordinal"] != page_position:
                    raise IntakeError(f"{label}: page ordinals must be contiguous from zero")
                first_row = _required_int(page["first_row_index"], f"{label}.page.first_row_index")
                rows = _required_int(page["row_count"], f"{label}.page.row_count", minimum=1)
                page_offset = _required_int(page["offset"], f"{label}.page.offset")
                page_size = _required_int(
                    page["compressed_page_size"],
                    f"{label}.page.compressed_page_size",
                    minimum=1,
                )
                page_end = page_offset + page_size - 1
                if first_row != expected_row:
                    raise IntakeError(f"{label}: page row coverage is not contiguous")
                if page_offset < chunk_start or page_end > chunk_end or page_offset <= previous_page_end:
                    raise IntakeError(f"{label}: page range is unordered or outside column chunk")
                expected_row += rows
                previous_page_end = page_end
                page_ranges.append((page_offset, page_end))
            if expected_row != row_count:
                raise IntakeError(f"{label}: pages do not cover the complete row group")
            expected_prerequisites = [] if page_ranges[0][0] == chunk_start else [(chunk_start, page_ranges[0][0] - 1)]
            if prerequisites != expected_prerequisites:
                raise IntakeError(f"{label}: column prefix prerequisite is incomplete")
            all_ranges = sorted(prerequisites + page_ranges)
            if all_ranges[0][0] != chunk_start or all_ranges[-1][1] != chunk_end:
                raise IntakeError(f"{label}: page ranges do not span the footer-bound column chunk")
            for previous, current in zip(all_ranges, all_ranges[1:]):
                if current[0] != previous[1] + 1:
                    raise IntakeError(f"{label}: page/prerequisite ranges are not byte-contiguous")
        if paths != list(REQUIRED_PROJECTED_COLUMNS):
            raise IntakeError("each row group must carry all projected columns in frozen order")
        for previous, current in zip(sorted(chunk_ranges), sorted(chunk_ranges)[1:]):
            if current[0] <= previous[1]:
                raise IntakeError("column chunks overlap inside a row group")
    if inventory["status"] == "COMPLETE" and seen_group_indexes != set(footer_truth):
        raise IntakeError("complete inventory does not cover the footer-bound row-group universe")
    if evidence_bytes is not None:
        ordered_evidence = sorted(evidence_ranges)
        if not ordered_evidence or ordered_evidence[0][0] != 0 or ordered_evidence[-1][1] != len(evidence_bytes):
            raise IntakeError("OffsetIndex aggregate evidence has uncovered bytes")
        for previous, current in zip(ordered_evidence, ordered_evidence[1:]):
            if current[0] != previous[1]:
                raise IntakeError("OffsetIndex aggregate evidence slices overlap or contain gaps")
    physical_ranges = sorted(global_chunk_ranges + global_index_ranges)
    for previous, current in zip(physical_ranges, physical_ranges[1:]):
        if current[0] <= previous[1]:
            raise IntakeError(f"physical ranges overlap: {previous[2]} and {current[2]}")
    return inventory


def _inventory_groups(inventory: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        group["row_group_index"]: {
            "row_count": group["row_count"],
            "columns": {column["path"]: column for column in group["columns"]},
        }
        for group in inventory["row_groups"]
    }


def _inventory_semantic_id(inventory: dict[str, Any]) -> str:
    identity = copy.deepcopy(inventory)
    identity["row_groups"].sort(key=lambda item: item["row_group_index"])
    if identity["page_index_probe"] is not None:
        identity["page_index_probe"] = {
            "probe_id": identity["page_index_probe"]["probe_id"],
        }
    if identity["offset_index_evidence"] is not None:
        identity["offset_index_evidence"] = {
            "serialization": identity["offset_index_evidence"]["serialization"],
        }
        for group in identity["row_groups"]:
            for column in group["columns"]:
                column["offset_index"].pop("evidence_offset", None)
    return "anima-step2a-page-inventory-" + _digest(identity)[:24]


def _overlaps(first: int, count: int, other_first: int, other_count: int) -> bool:
    return first < other_first + other_count and other_first < first + count


def _interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _selection_key(group_index: int, page: dict[str, Any], salt: str) -> str:
    return _digest(
        [
            SAMPLER_VERSION,
            salt,
            TARGET_SOURCE,
            TARGET_REVISION,
            TARGET_FILENAME,
            TARGET_FOOTER_SHA256,
            INPUT_POLICY_VERSION,
            group_index,
            page["page_ordinal"],
            page["first_row_index"],
            page["row_count"],
        ]
    )


def _materialize_selection(
    inventory: dict[str, Any],
    windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    groups = _inventory_groups(inventory)
    page_refs: set[tuple[int, str, int]] = set()
    for window in windows:
        group_index = window["row_group_index"]
        for path, column in groups[group_index]["columns"].items():
            for page in column["pages"]:
                if _overlaps(
                    window["first_row_index"],
                    window["row_count"],
                    page["first_row_index"],
                    page["row_count"],
                ):
                    page_refs.add((group_index, path, page["page_ordinal"]))

    ranges: list[dict[str, Any]] = []
    touched_by_group: dict[int, list[tuple[int, int]]] = {}
    columns_with_pages: set[tuple[int, str]] = set()
    for group_index, path, ordinal in sorted(page_refs):
        column = groups[group_index]["columns"][path]
        page = column["pages"][ordinal]
        columns_with_pages.add((group_index, path))
        start = page["offset"]
        end = start + page["compressed_page_size"] - 1
        ranges.append(
            {
                "row_group_index": group_index,
                "column": path,
                "kind": "data_page",
                "page_ordinal": ordinal,
                "start": start,
                "end": end,
                "bytes": end - start + 1,
            }
        )
        touched_by_group.setdefault(group_index, []).append(
            (page["first_row_index"], page["first_row_index"] + page["row_count"])
        )
    for group_index, path in sorted(columns_with_pages):
        column = groups[group_index]["columns"][path]
        for prerequisite in column["prerequisite_ranges"]:
            ranges.append(
                {
                    "row_group_index": group_index,
                    "column": path,
                    "kind": prerequisite["kind"],
                    "page_ordinal": None,
                    "start": prerequisite["start"],
                    "end": prerequisite["end"],
                    "bytes": prerequisite["end"] - prerequisite["start"] + 1,
                }
            )

    ranges.sort(key=lambda item: (item["start"], item["end"], item["column"]))
    deduplicated: list[dict[str, Any]] = []
    for item in ranges:
        if deduplicated and item["start"] <= deduplicated[-1]["end"]:
            previous = deduplicated[-1]
            if item["start"] == previous["start"] and item["end"] == previous["end"]:
                continue
            raise IntakeError("planned physical payload ranges overlap")
        deduplicated.append(item)
    touched_rows = sum(_interval_length(intervals) for intervals in touched_by_group.values())
    payload_bytes = sum(item["bytes"] for item in deduplicated)
    return deduplicated, touched_rows, payload_bytes


def _build_selection(
    inventory: dict[str, Any],
    *,
    salt: str,
    pilot_record_cap: int,
    minimum_records: int,
    max_touched_rows: int,
    max_payload_bytes: int,
    max_payload_requests: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    candidates: list[dict[str, Any]] = []
    if inventory["status"] == "COMPLETE":
        for group in inventory["row_groups"]:
            anchor = next(column for column in group["columns"] if column["path"] == inventory["anchor_column"])
            for page in anchor["pages"]:
                candidates.append(
                    {
                        "row_group_index": group["row_group_index"],
                        "anchor_page_ordinal": page["page_ordinal"],
                        "first_row_index": page["first_row_index"],
                        "row_count": page["row_count"],
                        "selection_key_sha256": _selection_key(group["row_group_index"], page, salt),
                    }
                )
    else:
        reasons.append(f"PAGE_INVENTORY_{inventory['status']}")
    candidates.sort(
        key=lambda item: (
            item["selection_key_sha256"],
            item["row_group_index"],
            item["anchor_page_ordinal"],
        )
    )

    selected: list[dict[str, Any]] = []
    eligible_candidate_count = 0
    for candidate in candidates:
        if candidate["row_count"] > pilot_record_cap or candidate["row_count"] > max_touched_rows:
            continue
        eligible_candidate_count += 1
        proposed = selected + [candidate]
        logical_rows = sum(item["row_count"] for item in proposed)
        if logical_rows > pilot_record_cap:
            continue
        ranges, touched_rows, payload_bytes = _materialize_selection(inventory, proposed)
        if touched_rows > max_touched_rows or payload_bytes > max_payload_bytes or len(ranges) > max_payload_requests:
            continue
        selected = proposed

    ranges: list[dict[str, Any]] = []
    touched_rows = 0
    payload_bytes = 0
    selected_rows = sum(item["row_count"] for item in selected)
    if selected:
        ranges, touched_rows, payload_bytes = _materialize_selection(inventory, selected)
    if selected_rows < minimum_records:
        selected = []
        ranges = []
        touched_rows = 0
        payload_bytes = 0
        selected_rows = 0
        reasons.append("MINIMUM_RECORDS_NOT_REACHED")
    if not candidates and inventory["status"] == "COMPLETE":
        reasons.append("NO_ANCHOR_PAGE_CANDIDATES")
    if candidates and not selected and "MINIMUM_RECORDS_NOT_REACHED" not in reasons:
        reasons.append("NO_CANDIDATE_FITS_FROZEN_CAPS")
    selection_ready = bool(selected) and not reasons
    return {
        "algorithm_version": SAMPLER_VERSION,
        "selection_salt": salt,
        "inventory_status": inventory["status"],
        "anchor_column": inventory["anchor_column"],
        "candidate_count": len(candidates),
        "eligible_candidate_count": eligible_candidate_count,
        "selected_windows": selected,
        "selected_rows": selected_rows,
        "touched_rows_upper_bound": touched_rows,
        "planned_payload_requests": len(ranges),
        "planned_payload_bytes": payload_bytes,
        "planned_ranges": ranges,
        "selection_ready": selection_ready,
        "execution_authorized": False,
        "hold_reasons": sorted(set(reasons)),
    }


def sample_plan(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(args.inventory)
    inventory_value, inventory_sha256 = _load_json_with_sha256(inventory_path)
    inventory = _validate_inventory(inventory_value, inventory_path)
    limits = {
        "pilot_record_cap": args.pilot_record_cap,
        "minimum_records": args.minimum_records,
        "max_touched_rows": args.max_touched_rows,
        "max_payload_bytes": args.max_payload_bytes,
        "max_payload_requests": args.max_payload_requests,
    }
    selection = _build_selection(
        inventory,
        salt=args.selection_salt,
        pilot_record_cap=args.pilot_record_cap,
        minimum_records=args.minimum_records,
        max_touched_rows=args.max_touched_rows,
        max_payload_bytes=args.max_payload_bytes,
        max_payload_requests=args.max_payload_requests,
    )
    reasons = [
        "INPUT_SEMANTICS_V2_NOT_APPROVED",
        "NATIVE_SOURCE_VALUE_SEMANTICS_UNVERIFIED",
        "REMOVAL_LEDGER_JOIN_PENDING",
        "HUMAN_REVIEW_PENDING",
    ]
    reasons.extend(selection["hold_reasons"])
    parents = [
        _parent(
            inventory_path,
            "page_index_inventory",
            _inventory_semantic_id(inventory),
            byte_sha256=inventory_sha256,
        )
    ]
    probe_path = _inventory_probe_path(inventory, inventory_path)
    evidence_path = _inventory_evidence_path(inventory, inventory_path)
    if probe_path is not None and evidence_path is not None:
        probe_value, probe_sha256 = _load_json_with_sha256(probe_path)
        if probe_sha256 != inventory["page_index_probe"]["sha256"]:
            raise IntakeError("page-index probe changed during sample-plan assembly")
        probe = _validate_page_index_probe(probe_value, probe_path)
        raw_footer_path = _resolve_relative_reference(
            probe_path,
            probe["raw_footer_evidence"]["path"],
        )
        parents.append(
            _parent(
                probe_path,
                "page_index_probe",
                inventory["page_index_probe"]["probe_id"],
                byte_sha256=inventory["page_index_probe"]["sha256"],
            )
        )
        parents.append(
            _parent(
                raw_footer_path,
                "raw_fixed_footer",
                f"{probe['probe_id']}::raw-footer-evidence",
                byte_sha256=probe["raw_footer_evidence"]["sha256"],
            )
        )
        parents.append(
            _parent(
                evidence_path,
                "offset_index_evidence",
                f"{_inventory_semantic_id(inventory)}::offset-index-evidence",
                byte_sha256=inventory["offset_index_evidence"]["sha256"],
            )
        )
    return _base_manifest(
        "sample_plan",
        parents,
        limits,
        selection,
        [],
        None,
        reasons,
    )


def _read_rows_with_sha256(
    path: Path,
    maximum: int,
) -> tuple[list[dict[str, Any]], str]:
    if maximum < 0 or maximum > HARD_PILOT_RECORD_CAP:
        raise IntakeError("row-read maximum is outside the hard cap")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        with _open_read_handle(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IntakeError(f"JSONL input is not a regular file: {path}")
            if before.st_size <= 0 or before.st_size > MAX_JSONL_TOTAL_BYTES:
                raise IntakeError(f"JSONL input size is outside 1..{MAX_JSONL_TOTAL_BYTES}: {path}")
            line_number = 0
            total_read = 0
            while raw := handle.readline(
                min(
                    MAX_JSONL_LINE_BYTES + 1,
                    MAX_JSONL_TOTAL_BYTES - total_read + 1,
                )
            ):
                line_number += 1
                total_read += len(raw)
                if total_read > MAX_JSONL_TOTAL_BYTES:
                    raise IntakeError(f"JSONL input size is outside 1..{MAX_JSONL_TOTAL_BYTES}: {path}")
                digest.update(raw)
                if len(raw) > MAX_JSONL_LINE_BYTES:
                    raise IntakeError(f"input row {line_number} exceeds line-byte cap")
                if not raw.strip():
                    raise IntakeError(f"input row {line_number} is blank")
                if len(rows) >= maximum:
                    raise IntakeError(f"input exceeds selected row count {maximum}")
                row = _json_loads(raw)
                if not isinstance(row, dict):
                    raise IntakeError(f"input row {line_number} is not an object")
                rows.append(row)
            after = os.fstat(handle.fileno())
    except IntakeError:
        raise
    except OSError as exc:
        raise IntakeError(f"failed to read JSONL input: {path}") from exc
    if total_read != before.st_size or _stable_file_identity(after) != _stable_file_identity(before):
        raise IntakeError(f"JSONL input changed while being read: {path}")
    if len(rows) != maximum:
        raise IntakeError(f"input row count {len(rows)} does not equal selected row count {maximum}")
    return rows, digest.hexdigest()


def _read_rows(path: Path, maximum: int) -> list[dict[str, Any]]:
    return _read_rows_with_sha256(path, maximum)[0]


def _row_window(row_group_index: int, row_index: int, selection: dict[str, Any]) -> dict[str, Any]:
    matches = [
        window
        for window in selection["selected_windows"]
        if window["row_group_index"] == row_group_index
        and window["first_row_index"] <= row_index < window["first_row_index"] + window["row_count"]
    ]
    if len(matches) != 1:
        raise IntakeError("row locator is outside or ambiguous within selected windows")
    return matches[0]


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, bool):
        raise IntakeError("image id must not be boolean")
    if type(value) is int and value >= 0:
        return str(value)
    return _required_text(value, "image id")


def _derive_record(
    row: dict[str, Any],
    selection: dict[str, Any],
    selection_manifest_sha256: str,
    source_binding_sha256: str,
) -> dict[str, Any]:
    allowed = {
        "row_group_index",
        "row_index_in_group",
        "id",
        "source",
        "source_revision",
        "md5",
        "sha256",
        "tag_string",
        "tag_string_general",
        "rating",
        "score",
        "asset_license",
    }
    required = {
        "row_group_index",
        "row_index_in_group",
        "id",
        "source",
        "md5",
        "tag_string",
        "tag_string_general",
        "rating",
        "score",
    }
    if not required <= set(row) or not set(row) <= allowed:
        raise IntakeError("bounded row keys do not match the frozen projection")
    row_group_index = _required_int(row["row_group_index"], "row_group_index")
    row_index = _required_int(row["row_index_in_group"], "row_index_in_group")
    window = _row_window(row_group_index, row_index, selection)
    image_id = _normalize_image_id(row["id"])
    native_source = _optional_text(row["source"], "native source")
    native_revision = row.get("source_revision")
    if native_revision is not None:
        native_revision = _required_text(native_revision, "native source_revision")
        if native_revision != TARGET_REVISION:
            raise IntakeError("native source_revision conflicts with immutable binding")
    md5 = _required_hex(row["md5"], HEX32, "row md5")
    sha256 = row.get("sha256")
    if sha256 is not None:
        sha256 = _required_hex(sha256, HEX64, "row sha256")
    for field in ("tag_string", "tag_string_general", "rating", "asset_license"):
        if field in row:
            _optional_text(row[field], field)
    score = row["score"]
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
    ):
        raise IntakeError("score must be a finite number or null")
    raw_row_sha256 = _digest(row)
    return {
        "record_key": f"{TARGET_SOURCE}::{image_id}",
        "source": TARGET_SOURCE,
        "source_revision": TARGET_REVISION,
        "native_source": native_source,
        "native_source_revision": native_revision,
        "source_locator": {
            "row_group_index": row_group_index,
            "row_index_in_group": row_index,
            "anchor_page_ordinal": window["anchor_page_ordinal"],
            "selection_key_sha256": window["selection_key_sha256"],
        },
        "image_id": image_id,
        "md5": md5,
        "sha256": sha256,
        "raw_row_sha256": raw_row_sha256,
        "projected_metadata": {
            "tag_string": row["tag_string"],
            "tag_string_general": row["tag_string_general"],
            "rating": row["rating"],
            "score": score,
            "asset_license": row.get("asset_license"),
        },
        "provenance": {
            "mode": PROVENANCE_MODE,
            "source_binding_sha256": source_binding_sha256,
            "selection_manifest_sha256": selection_manifest_sha256,
            "input_row_sha256": raw_row_sha256,
        },
        "ledger_status": "HOLD_LEDGER_HASH_PENDING" if sha256 is None else "HOLD_OPT_OUT_UNKNOWN",
    }


def derive_provenance(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection_manifest)
    parent, selection_sha256 = _load_manifest_with_sha256(
        selection_path,
        "sample_plan",
    )
    selection = parent["selection"]
    if selection is None or selection["selection_ready"] is not True or not selection["selected_windows"]:
        raise IntakeError("selection manifest has no bounded deterministic selection")
    rows_path = Path(args.input)
    rows, rows_sha256 = _read_rows_with_sha256(
        rows_path,
        selection["selected_rows"],
    )
    source_binding_sha256 = parent["source_binding"]["source_binding_sha256"]
    records = [
        _derive_record(
            row,
            selection,
            selection_sha256,
            source_binding_sha256,
        )
        for row in rows
    ]
    records.sort(
        key=lambda record: (
            record["source_locator"]["row_group_index"],
            record["source_locator"]["row_index_in_group"],
            record["record_key"],
        )
    )
    locators = {
        (record["source_locator"]["row_group_index"], record["source_locator"]["row_index_in_group"])
        for record in records
    }
    record_keys = {record["record_key"] for record in records}
    if len(locators) != len(records) or len(record_keys) != len(records):
        raise IntakeError("duplicate row locator or source/image id")
    reasons = [
        "INPUT_SEMANTICS_V2_NOT_APPROVED",
        "REMOVAL_LEDGER_JOIN_PENDING",
        "HUMAN_REVIEW_PENDING",
    ]
    if any(record["sha256"] is None for record in records):
        reasons.append("HASH_RECONCILIATION_PENDING")
    return _base_manifest(
        "derive_provenance",
        [
            _parent(
                selection_path,
                "selection_manifest",
                parent["manifest_id"],
                byte_sha256=selection_sha256,
            ),
            _parent(
                rows_path,
                "bounded_projected_rows",
                None,
                byte_sha256=rows_sha256,
            ),
        ],
        copy.deepcopy(parent["limits"]),
        copy.deepcopy(selection),
        records,
        None,
        reasons,
    )


def _full_key(record: dict[str, Any]) -> list[str]:
    if record["sha256"] is None:
        raise IntakeError("full removal key requires SHA256")
    return [
        record["source"],
        record["image_id"],
        record["md5"],
        record["sha256"],
        record["source_revision"],
    ]


def _ledger_identifier(ledger: dict[str, Any]) -> str:
    identity = copy.deepcopy(ledger)
    identity.pop("ledger_id", None)
    identity["coverage"]["evidence"].pop("path", None)
    return "anima-step2a-removal-" + _digest(identity)[:24]


def _validate_ledger(value: Any) -> dict[str, Any]:
    ledger = _expect_keys(
        value,
        {
            "schema",
            "ledger_id",
            "source",
            "coverage",
            "previous_snapshot",
            "entries",
        },
        "ledger",
    )
    if ledger["schema"] != LEDGER_SCHEMA:
        raise IntakeError("ledger schema mismatch")
    source = _expect_keys(ledger["source"], {"name", "revision"}, "ledger.source")
    if source != {"name": TARGET_SOURCE, "revision": TARGET_REVISION}:
        raise IntakeError("ledger source/revision mismatch")
    coverage = _expect_keys(
        ledger["coverage"],
        {
            "mode",
            "as_of",
            "authority",
            "evidence",
            "covered_population_sha256",
        },
        "ledger.coverage",
    )
    if coverage["mode"] not in {"delta_only", "authoritative_complete_snapshot"}:
        raise IntakeError("ledger coverage mode is invalid")
    _required_utc_timestamp(coverage["as_of"], "ledger.coverage.as_of")
    _required_text(coverage["authority"], "ledger.coverage.authority")
    evidence = _expect_keys(coverage["evidence"], {"path", "sha256"}, "ledger.coverage.evidence")
    _required_text(evidence["path"], "ledger.coverage.evidence.path")
    _required_hex(evidence["sha256"], HEX64, "ledger.coverage.evidence.sha256")
    if coverage["mode"] == "delta_only":
        if coverage["covered_population_sha256"] is not None:
            raise IntakeError("delta ledger must not claim complete population coverage")
    else:
        _required_hex(
            coverage["covered_population_sha256"],
            HEX64,
            "ledger.coverage.covered_population_sha256",
        )
    previous = _expect_keys(
        ledger["previous_snapshot"],
        {"kind", "snapshot_id", "sha256", "entry_count"},
        "ledger.previous_snapshot",
    )
    if previous["kind"] not in {"genesis", "ledger"}:
        raise IntakeError("ledger.previous_snapshot.kind is invalid")
    _required_text(previous["snapshot_id"], "ledger.previous_snapshot.snapshot_id")
    _required_hex(previous["sha256"], HEX64, "ledger.previous_snapshot.sha256")
    _required_int(previous["entry_count"], "ledger.previous_snapshot.entry_count")
    if previous["kind"] == "genesis" and previous != {
        "kind": "genesis",
        "snapshot_id": REMOVAL_LEDGER_GENESIS_ID,
        "sha256": REMOVAL_LEDGER_GENESIS_SHA256,
        "entry_count": 0,
    }:
        raise IntakeError("ledger genesis predecessor descriptor is not frozen")
    if not isinstance(ledger["entries"], list):
        raise IntakeError("ledger entries must be a list")
    seen_full_keys: set[str] = set()
    seen_image_ids: set[str] = set()
    for index, entry_value in enumerate(ledger["entries"]):
        label = f"ledger.entries[{index}]"
        entry = _expect_keys(
            entry_value,
            {
                "sequence",
                "action",
                "source",
                "source_revision",
                "image_id",
                "md5",
                "sha256",
                "full_key_sha256",
                "reason_code",
                "effective_at",
                "evidence_sha256",
            },
            label,
        )
        if entry["sequence"] != index + 1 or entry["action"] != "REMOVE":
            raise IntakeError(f"{label}: sequence/action contract drift")
        if entry["source"] != TARGET_SOURCE or entry["source_revision"] != TARGET_REVISION:
            raise IntakeError(f"{label}: source binding mismatch")
        image_id = _required_text(entry["image_id"], f"{label}.image_id")
        if image_id in seen_image_ids:
            raise IntakeError(f"{label}: duplicate image_id with conflicting or repeated hash identity")
        seen_image_ids.add(image_id)
        _required_hex(entry["md5"], HEX32, f"{label}.md5")
        _required_hex(entry["sha256"], HEX64, f"{label}.sha256")
        expected_key = _digest(
            [
                entry["source"],
                entry["image_id"],
                entry["md5"],
                entry["sha256"],
                entry["source_revision"],
            ]
        )
        if entry["full_key_sha256"] != expected_key:
            raise IntakeError(f"{label}: full-key digest mismatch")
        if expected_key in seen_full_keys:
            raise IntakeError(f"{label}: duplicate full removal key")
        seen_full_keys.add(expected_key)
        _required_text(entry["reason_code"], f"{label}.reason_code")
        _required_utc_timestamp(entry["effective_at"], f"{label}.effective_at")
        if entry["effective_at"] > coverage["as_of"]:
            raise IntakeError(f"{label}: effective_at is after coverage as_of")
        _required_hex(entry["evidence_sha256"], HEX64, f"{label}.evidence_sha256")
    if ledger["ledger_id"] != _ledger_identifier(ledger):
        raise IntakeError("ledger semantic identifier mismatch")
    return ledger


def _load_ledger(path: Path) -> dict[str, Any]:
    return _load_ledger_with_sha256(path)[0]


def _load_ledger_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    value, byte_sha256 = _load_json_with_sha256(path)
    return _validate_ledger(value), byte_sha256


def _validate_ledger_genesis(value: Any) -> dict[str, Any]:
    genesis = _expect_keys(
        value,
        {
            "schema",
            "genesis_id",
            "chain_id",
            "source",
            "created_at",
            "entry_count",
            "entries_sha256",
        },
        "removal-ledger genesis",
    )
    if (
        genesis["schema"] != "anima-step2a-removal-ledger-genesis-v1"
        or genesis["genesis_id"] != REMOVAL_LEDGER_GENESIS_ID
        or genesis["chain_id"] != REMOVAL_LEDGER_CHAIN_ID
        or genesis["source"] != {"name": TARGET_SOURCE, "revision": TARGET_REVISION}
        or genesis["entry_count"] != 0
        or genesis["entries_sha256"] != _digest([])
    ):
        raise IntakeError("removal-ledger genesis contract drift")
    _required_utc_timestamp(genesis["created_at"], "removal-ledger genesis.created_at")
    return genesis


def _snapshot_id(snapshot: dict[str, Any]) -> str:
    if snapshot.get("schema") == LEDGER_SCHEMA:
        return snapshot["ledger_id"]
    return snapshot["genesis_id"]


def _trusted_relative_path(value: Any, label: str) -> Path:
    raw = _required_text(value, label)
    path = Path(raw)
    if path.is_absolute():
        raise IntakeError(f"{label}: expected repository-relative path")
    resolved = _absolute_lexical(PROJECT_ROOT / path)
    try:
        resolved.relative_to(_absolute_lexical(PROJECT_ROOT))
    except ValueError as exc:
        raise IntakeError(f"{label}: path escapes repository") from exc
    return resolved


def _trusted_ledger_head() -> dict[str, Any]:
    head_path = PROJECT_ROOT / REMOVAL_LEDGER_HEAD_RELATIVE_PATH
    head_value, head_sha256 = _load_json_with_sha256(head_path)
    if head_sha256 != REMOVAL_LEDGER_HEAD_SHA256:
        raise IntakeError("removal-ledger trusted head byte binding mismatch")
    head = _expect_keys(
        head_value,
        {"schema", "head_id", "chain_id", "source", "current"},
        "removal-ledger trusted head",
    )
    if (
        head["schema"] != "anima-step2a-removal-ledger-head-v1"
        or head["head_id"] != REMOVAL_LEDGER_HEAD_ID
        or head["chain_id"] != REMOVAL_LEDGER_CHAIN_ID
        or head["source"] != {"name": TARGET_SOURCE, "revision": TARGET_REVISION}
    ):
        raise IntakeError("removal-ledger trusted head contract drift")
    current = _expect_keys(
        head["current"],
        {"kind", "snapshot_id", "path", "sha256", "entry_count"},
        "removal-ledger trusted head.current",
    )
    if current["kind"] not in {"genesis", "ledger"}:
        raise IntakeError("removal-ledger trusted head kind is invalid")
    _required_text(current["snapshot_id"], "removal-ledger trusted head.current.snapshot_id")
    _required_hex(current["sha256"], HEX64, "removal-ledger trusted head.current.sha256")
    _required_int(current["entry_count"], "removal-ledger trusted head.current.entry_count")
    current_path = _trusted_relative_path(
        current["path"],
        "removal-ledger trusted head.current.path",
    )
    current_value, current_sha256 = _load_json_with_sha256(current_path)
    if current_sha256 != current["sha256"]:
        raise IntakeError("removal-ledger trusted head snapshot byte binding mismatch")
    if current["kind"] == "genesis":
        if (
            current_path != _absolute_lexical(PROJECT_ROOT / REMOVAL_LEDGER_GENESIS_RELATIVE_PATH)
            or current["sha256"] != REMOVAL_LEDGER_GENESIS_SHA256
        ):
            raise IntakeError("removal-ledger trusted head genesis binding mismatch")
        snapshot = _validate_ledger_genesis(current_value)
        entry_count = 0
    else:
        snapshot = _validate_ledger(current_value)
        entry_count = len(snapshot["entries"])
    if current["snapshot_id"] != _snapshot_id(snapshot) or current["entry_count"] != entry_count:
        raise IntakeError("removal-ledger trusted head semantic binding mismatch")
    return head


def _verify_previous_snapshot(
    ledger: dict[str, Any],
    previous_path_value: str | None,
) -> dict[str, Any]:
    descriptor = ledger["previous_snapshot"]
    if previous_path_value is None:
        raise IntakeError("append-only ledger requires --previous-ledger")
    previous_path = _absolute_lexical(Path(previous_path_value))
    previous_value, previous_sha256 = _load_json_with_sha256(previous_path)
    if descriptor["kind"] == "genesis":
        if previous_path != _absolute_lexical(PROJECT_ROOT / REMOVAL_LEDGER_GENESIS_RELATIVE_PATH):
            raise IntakeError("ledger genesis predecessor path is not frozen")
        previous = _validate_ledger_genesis(previous_value)
        previous_entry_count = 0
        previous_as_of = previous["created_at"]
        previous_mode = None
    else:
        previous = _validate_ledger(previous_value)
        previous_entry_count = len(previous["entries"])
        previous_as_of = previous["coverage"]["as_of"]
        previous_mode = previous["coverage"]["mode"]
    if (
        descriptor["snapshot_id"] != _snapshot_id(previous)
        or descriptor["sha256"] != previous_sha256
        or descriptor["entry_count"] != previous_entry_count
        or (descriptor["kind"] == "ledger" and ledger["entries"][:previous_entry_count] != previous["entries"])
    ):
        raise IntakeError("ledger append-only prefix binding failed")
    if ledger["coverage"]["as_of"] < previous_as_of:
        raise IntakeError("ledger coverage as_of moved backwards")
    if (
        previous_mode == "authoritative_complete_snapshot"
        and ledger["coverage"]["mode"] != "authoritative_complete_snapshot"
    ):
        raise IntakeError("ledger coverage mode regressed from authoritative complete to delta")
    return previous


def _verify_ledger_is_trusted_head(
    ledger_path: Path,
    ledger: dict[str, Any],
    trusted_head: dict[str, Any],
    *,
    ledger_sha256: str | None = None,
) -> None:
    current = trusted_head["current"]
    if current["kind"] != "ledger":
        raise IntakeError("trusted removal-ledger head has no accepted ledger snapshot")
    trusted_path = _trusted_relative_path(
        current["path"],
        "removal-ledger trusted head.current.path",
    )
    if ledger_sha256 is None:
        observed_ledger, ledger_sha256 = _load_ledger_with_sha256(ledger_path)
        if observed_ledger != ledger:
            raise IntakeError("removal ledger changed between validation and trusted-head check")
    if (
        _absolute_lexical(ledger_path) != trusted_path
        or current["snapshot_id"] != ledger["ledger_id"]
        or current["sha256"] != ledger_sha256
        or current["entry_count"] != len(ledger["entries"])
    ):
        raise IntakeError("removal ledger is not the frozen accepted head")


def _compute_ledger_join(
    parent: dict[str, Any],
    parent_sha256: str,
    ledger: dict[str, Any],
    ledger_sha256: str,
    previous: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    coverage = ledger["coverage"]
    if coverage["mode"] == "authoritative_complete_snapshot" and coverage["covered_population_sha256"] != parent_sha256:
        raise IntakeError("complete ledger coverage is not bound to this population manifest")
    removals = {entry["full_key_sha256"] for entry in ledger["entries"]}
    removals_by_image_id = {entry["image_id"]: entry["full_key_sha256"] for entry in ledger["entries"]}
    records = copy.deepcopy(parent["records"])
    matched = 0
    unknown = 0
    hash_pending = 0
    for record in records:
        if record["sha256"] is None:
            record["ledger_status"] = "HOLD_LEDGER_HASH_PENDING"
            hash_pending += 1
            continue
        key = _digest(_full_key(record))
        if record["image_id"] in removals_by_image_id and removals_by_image_id[record["image_id"]] != key:
            raise IntakeError("ledger image_id/hash identity conflicts with derived population")
        if key in removals:
            record["ledger_status"] = "REJECT_OPT_OUT"
            matched += 1
        elif coverage["mode"] == "authoritative_complete_snapshot":
            record["ledger_status"] = "NOT_OPTED_OUT"
        else:
            record["ledger_status"] = "HOLD_OPT_OUT_UNKNOWN"
            unknown += 1
    ledger_summary = {
        "ledger_id": ledger["ledger_id"],
        "ledger_sha256": ledger_sha256,
        "coverage_mode": coverage["mode"],
        "coverage_as_of": coverage["as_of"],
        "coverage_evidence_sha256": coverage["evidence"]["sha256"],
        "previous_snapshot_id": _snapshot_id(previous),
        "entry_count": len(ledger["entries"]),
        "matched_removals": matched,
        "hash_pending_records": hash_pending,
        "unknown_opt_out_records": unknown,
    }
    reasons = [
        "INPUT_SEMANTICS_V2_NOT_APPROVED",
        "HUMAN_REVIEW_PENDING",
    ]
    if hash_pending:
        reasons.append("HASH_RECONCILIATION_PENDING")
    if unknown:
        reasons.append("OPT_OUT_COVERAGE_INCOMPLETE")
    return records, ledger_summary, reasons


def ledger_join(args: argparse.Namespace) -> dict[str, Any]:
    trusted_head = _trusted_ledger_head()
    trusted_head_path = PROJECT_ROOT / REMOVAL_LEDGER_HEAD_RELATIVE_PATH
    manifest_path = Path(args.manifest)
    parent, parent_sha256 = _load_manifest_with_sha256(
        manifest_path,
        "derive_provenance",
    )
    ledger_path = Path(args.ledger)
    ledger, ledger_sha256 = _load_ledger_with_sha256(ledger_path)
    coverage_evidence_path = Path(args.coverage_evidence)
    coverage_evidence_sha256 = _file_sha256(
        coverage_evidence_path,
        max_bytes=MAX_LEDGER_COVERAGE_EVIDENCE_BYTES,
    )
    if coverage_evidence_sha256 != ledger["coverage"]["evidence"]["sha256"]:
        raise IntakeError("ledger coverage evidence byte binding mismatch")
    previous = _verify_previous_snapshot(ledger, args.previous_ledger)
    _verify_ledger_is_trusted_head(
        ledger_path,
        ledger,
        trusted_head,
        ledger_sha256=ledger_sha256,
    )
    records, ledger_summary, reasons = _compute_ledger_join(
        parent,
        parent_sha256,
        ledger,
        ledger_sha256,
        previous,
    )
    parents = [
        _parent(
            manifest_path,
            "derived_provenance_manifest",
            parent["manifest_id"],
            byte_sha256=parent_sha256,
        ),
        _parent(
            ledger_path,
            "removal_ledger",
            ledger["ledger_id"],
            byte_sha256=ledger_sha256,
        ),
        _parent(
            coverage_evidence_path,
            "ledger_coverage_evidence",
            None,
            byte_sha256=coverage_evidence_sha256,
        ),
        _parent(
            trusted_head_path,
            "trusted_removal_ledger_head",
            trusted_head["head_id"],
            byte_sha256=REMOVAL_LEDGER_HEAD_SHA256,
        ),
        _parent(
            Path(args.previous_ledger),
            "previous_removal_snapshot",
            _snapshot_id(previous),
            byte_sha256=ledger["previous_snapshot"]["sha256"],
        ),
    ]
    return _base_manifest(
        "ledger_join",
        parents,
        copy.deepcopy(parent["limits"]),
        copy.deepcopy(parent["selection"]),
        records,
        ledger_summary,
        reasons,
    )


def _bounded_int(
    value: str,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be within {minimum}..{maximum}")
    return parsed


def _pilot_cap(value: str) -> int:
    return _bounded_int(value, label="pilot record cap", minimum=1, maximum=HARD_PILOT_RECORD_CAP)


def _minimum_records(value: str) -> int:
    return _bounded_int(value, label="minimum records", minimum=1, maximum=HARD_PILOT_RECORD_CAP)


def _touched_cap(value: str) -> int:
    return _bounded_int(value, label="touched-row cap", minimum=1, maximum=HARD_MAX_TOUCHED_ROWS)


def _payload_cap(value: str) -> int:
    return _bounded_int(value, label="payload byte cap", minimum=1, maximum=HARD_MAX_PAYLOAD_BYTES)


def _request_cap(value: str) -> int:
    return _bounded_int(value, label="payload request cap", minimum=1, maximum=HARD_MAX_PAYLOAD_REQUESTS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Offline Anima Step 2a draft intake contract")
    commands = result.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample-plan")
    sample.add_argument("--inventory", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--selection-salt", default=DEFAULT_SELECTION_SALT)
    sample.add_argument("--pilot-record-cap", type=_pilot_cap, default=DEFAULT_PILOT_RECORD_CAP)
    sample.add_argument("--minimum-records", type=_minimum_records, default=DEFAULT_MINIMUM_RECORDS)
    sample.add_argument("--max-touched-rows", type=_touched_cap, default=DEFAULT_MAX_TOUCHED_ROWS)
    sample.add_argument("--max-payload-bytes", type=_payload_cap, default=DEFAULT_MAX_PAYLOAD_BYTES)
    sample.add_argument(
        "--max-payload-requests",
        type=_request_cap,
        default=DEFAULT_MAX_PAYLOAD_REQUESTS,
    )
    sample.set_defaults(handler=sample_plan)

    derive = commands.add_parser("derive-provenance")
    derive.add_argument("--input", required=True)
    derive.add_argument("--selection-manifest", required=True)
    derive.add_argument("--output", required=True)
    derive.set_defaults(handler=derive_provenance)

    join = commands.add_parser("ledger-join")
    join.add_argument("--manifest", required=True)
    join.add_argument("--ledger", required=True)
    join.add_argument("--coverage-evidence", required=True)
    join.add_argument("--previous-ledger", required=True)
    join.add_argument("--output", required=True)
    join.set_defaults(handler=ledger_join)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists() or output_path.is_symlink():
        print(f"ERROR: fail-closed: output already exists: {output_path}", file=sys.stderr)
        return 2
    if args.command == "sample-plan":
        if not isinstance(args.selection_salt, str) or not args.selection_salt.strip():
            print("ERROR: fail-closed: selection salt must be non-empty", file=sys.stderr)
            return 2
        if args.minimum_records > args.pilot_record_cap:
            print("ERROR: fail-closed: minimum records exceed pilot cap", file=sys.stderr)
            return 2
        if args.pilot_record_cap > args.max_touched_rows:
            print("ERROR: fail-closed: pilot cap exceeds touched-row cap", file=sys.stderr)
            return 2
    try:
        _source_binding()
        manifest = args.handler(args)
        _write_no_clobber(output_path, manifest)
    except (IntakeError, OSError) as exc:
        print(f"ERROR: fail-closed: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {Path(args.output).resolve()} "
        "(draft input semantics; training_ready=false; tool network/payload/images/training=0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
