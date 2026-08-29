#!/usr/bin/env python3
"""Parse the fixed Anima Step 2a Parquet footer and plan bounded column ranges.

The target file is immutable and gated.  This follow-on reuses the credential
and HTTP Range protections from ``anima_step2a_parquet_range_probe``.  Footer
bytes are parsed in memory with a deliberately small, dependency-free reader
for the Apache Thrift Compact Protocol used by Parquet metadata.

The tool never reads column payloads.  Its only persistent output is a
no-clobber JSON record containing safe schema identifiers, row-group counts,
and exact byte ranges for a later selected-column metadata pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import anima_step2a_parquet_range_probe as range_probe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA = "anima-step2a-parquet-footer-plan-evidence-v1"
RAW_PROBE_RELATIVE_PATH = "outputs/_analysis/anima_step2a_source_access_2026-07-18/parquet_footer_probe_v1.json"
RAW_PROBE_PATH = PROJECT_ROOT / RAW_PROBE_RELATIVE_PATH
RAW_PROBE_SHA256 = "127e33acd7cf32eb8e25a7752854380363a13de1efb686ff470131d1f244fcc3"
ACCESS_RECORD_SHA256 = "fa007d01257c0f3d62414fc623838884f702bea2a231a1c0607388e7ddb7be67"
RAW_FOOTER_LENGTH = 139_521
RAW_FOOTER_SHA256 = "0c3d48949974d9e6870304290a2649a0d2b9f762d3453374ad6defb3b71fb276"

DEFAULT_PILOT_RECORDS = 10_000
HARD_MAX_PILOT_RECORDS = 100_000
DEFAULT_MAX_PLANNED_COLUMN_BYTES = 256 * 1024 * 1024
HARD_MAX_PLANNED_COLUMN_BYTES = 1024 * 1024 * 1024
MAX_PRIOR_EVIDENCE_BYTES = 64 * 1024
MAX_COMPACT_DEPTH = 64
MAX_COMPACT_VALUES = 1_000_000
MAX_CONTAINER_ITEMS = 250_000
MAX_STRUCT_FIELDS = 4096
MAX_SCHEMA_ELEMENTS = 4096
MAX_ROW_GROUPS = 100_000
MAX_BINARY_BYTES = range_probe.DEFAULT_MAX_FOOTER_BYTES

SAFE_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")

PHYSICAL_TYPES = {
    0: "BOOLEAN",
    1: "INT32",
    2: "INT64",
    3: "INT96",
    4: "FLOAT",
    5: "DOUBLE",
    6: "BYTE_ARRAY",
    7: "FIXED_LEN_BYTE_ARRAY",
}
REPETITION_TYPES = {0: "REQUIRED", 1: "OPTIONAL", 2: "REPEATED"}
CONVERTED_TYPES = {
    0: "UTF8",
    1: "MAP",
    2: "MAP_KEY_VALUE",
    3: "LIST",
    4: "ENUM",
    5: "DECIMAL",
    6: "DATE",
    7: "TIME_MILLIS",
    8: "TIME_MICROS",
    9: "TIMESTAMP_MILLIS",
    10: "TIMESTAMP_MICROS",
    11: "UINT_8",
    12: "UINT_16",
    13: "UINT_32",
    14: "UINT_64",
    15: "INT_8",
    16: "INT_16",
    17: "INT_32",
    18: "INT_64",
    19: "JSON",
    20: "BSON",
    21: "INTERVAL",
}
LOGICAL_TYPES = {
    1: "STRING",
    2: "MAP",
    3: "LIST",
    4: "ENUM",
    5: "DECIMAL",
    6: "DATE",
    7: "TIME",
    8: "TIMESTAMP",
    10: "INTEGER",
    11: "UNKNOWN",
    12: "JSON",
    13: "BSON",
    14: "UUID",
    15: "FLOAT16",
    16: "VARIANT",
    17: "GEOMETRY",
    18: "GEOGRAPHY",
}
COMPRESSION_CODECS = {
    0: "UNCOMPRESSED",
    1: "SNAPPY",
    2: "GZIP",
    3: "LZO",
    4: "BROTLI",
    5: "LZ4",
    6: "ZSTD",
    7: "LZ4_RAW",
}

CORE_POLICY_ALIASES = {
    "source": ("source",),
    "source_revision": ("source_revision",),
    "image_id": ("image_id", "id", "post_id"),
    "tags": ("tags", "tag_string", "tag_string_general", "tag_snapshot"),
    "rating": ("rating", "rating_label"),
    "score": ("score",),
    "md5": ("md5",),
}
OPTIONAL_DATASET_ALIASES = {
    "asset_license": ("asset_license", "content_license", "license"),
    "sha256": ("sha256",),
    "phash": ("phash", "perceptual_hash"),
}
EXPLICIT_OPT_OUT_ALIASES = (
    "opt_out",
    "is_opted_out",
    "opted_out",
    "opt_out_status",
)
OPT_OUT_RELATED_PATTERN = re.compile(
    r"(?:opt_?out|withdraw|takedown|remov|delet|ban)",
    re.IGNORECASE,
)

# Apache Thrift Compact Protocol type nibbles.
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
VALID_VALUE_TYPES = frozenset(
    {
        CT_TRUE,
        CT_FALSE,
        CT_BYTE,
        CT_I16,
        CT_I32,
        CT_I64,
        CT_DOUBLE,
        CT_BINARY,
        CT_LIST,
        CT_SET,
        CT_MAP,
        CT_STRUCT,
        CT_UUID,
    }
)


class FooterPlanError(RuntimeError):
    """Base class for deterministic, fail-closed planner errors."""


class PriorProbeEvidenceError(FooterPlanError):
    pass


class AccessRecordBindingError(FooterPlanError):
    pass


class FooterEvidenceMismatchError(FooterPlanError):
    pass


class CompactProtocolError(FooterPlanError):
    pass


class ParquetMetadataError(FooterPlanError):
    pass


class SelectionPlanError(FooterPlanError):
    pass


@dataclass(frozen=True)
class CompactField:
    type_code: int
    value: Any


@dataclass(frozen=True)
class CompactStruct:
    fields: dict[int, CompactField]


@dataclass(frozen=True)
class CompactList:
    element_type: int
    values: tuple[Any, ...]


@dataclass(frozen=True)
class CompactMap:
    key_type: int
    value_type: int
    entries: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class SchemaNode:
    name: str
    physical_type: int | None
    repetition_type: int | None
    num_children: int | None
    converted_type: int | None
    logical_type: str | None


@dataclass(frozen=True)
class SchemaLeaf:
    index: int
    path: tuple[str, ...]
    physical_type: int
    repetition_type: int
    converted_type: int | None
    logical_type: str | None


@dataclass(frozen=True)
class ColumnChunk:
    row_group_index: int
    leaf_index: int
    path: tuple[str, ...]
    codec: int
    num_values: int
    total_compressed_size: int
    total_uncompressed_size: int
    start: int
    end: int
    offset_index_offset: int | None
    offset_index_length: int | None
    column_index_offset: int | None
    column_index_length: int | None


@dataclass(frozen=True)
class RowGroup:
    index: int
    num_rows: int
    total_byte_size: int
    total_compressed_size: int
    columns: tuple[ColumnChunk, ...]


@dataclass(frozen=True)
class ParsedParquetMetadata:
    version: int
    root_name: str
    num_rows: int
    leaves: tuple[SchemaLeaf, ...]
    row_groups: tuple[RowGroup, ...]


class CompactReader:
    """Bounded reader for the value subset used by Parquet file metadata."""

    def __init__(self, payload: bytes) -> None:
        if not payload or len(payload) > range_probe.DEFAULT_MAX_FOOTER_BYTES:
            raise CompactProtocolError
        self.payload = payload
        self.offset = 0
        self.values_read = 0

    def _count_value(self) -> None:
        self.values_read += 1
        if self.values_read > MAX_COMPACT_VALUES:
            raise CompactProtocolError

    def _read(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise CompactProtocolError
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
                raise CompactProtocolError
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                return result
            shift += 7
        raise CompactProtocolError

    def _read_zigzag(self) -> int:
        value = self._read_varint()
        return (value >> 1) ^ -(value & 1)

    def _read_bool_value(self) -> bool:
        encoded = self._read_u8()
        if encoded == CT_TRUE:
            return True
        if encoded == CT_FALSE:
            return False
        raise CompactProtocolError

    def _read_collection(self, depth: int) -> CompactList:
        header = self._read_u8()
        size = header >> 4
        element_type = header & 0x0F
        if size == 15:
            size = self._read_varint()
        if element_type not in VALID_VALUE_TYPES or size > MAX_CONTAINER_ITEMS:
            raise CompactProtocolError
        values = tuple(self._read_value(element_type, depth + 1) for _ in range(size))
        return CompactList(element_type=element_type, values=values)

    def _read_map(self, depth: int) -> CompactMap:
        size = self._read_varint()
        if size > MAX_CONTAINER_ITEMS:
            raise CompactProtocolError
        if size == 0:
            return CompactMap(key_type=CT_STOP, value_type=CT_STOP, entries=())
        types = self._read_u8()
        key_type = types >> 4
        value_type = types & 0x0F
        if key_type not in VALID_VALUE_TYPES or value_type not in VALID_VALUE_TYPES:
            raise CompactProtocolError
        entries = tuple(
            (
                self._read_value(key_type, depth + 1),
                self._read_value(value_type, depth + 1),
            )
            for _ in range(size)
        )
        return CompactMap(
            key_type=key_type,
            value_type=value_type,
            entries=entries,
        )

    def _read_value(self, type_code: int, depth: int) -> Any:
        if depth > MAX_COMPACT_DEPTH:
            raise CompactProtocolError
        self._count_value()
        if type_code in {CT_TRUE, CT_FALSE}:
            return self._read_bool_value()
        if type_code == CT_BYTE:
            value = self._read_u8()
            return value if value < 128 else value - 256
        if type_code in {CT_I16, CT_I32, CT_I64}:
            return self._read_zigzag()
        if type_code == CT_DOUBLE:
            return struct.unpack("<d", self._read(8))[0]
        if type_code == CT_BINARY:
            size = self._read_varint()
            if size > MAX_BINARY_BYTES:
                raise CompactProtocolError
            return self._read(size)
        if type_code in {CT_LIST, CT_SET}:
            return self._read_collection(depth)
        if type_code == CT_MAP:
            return self._read_map(depth)
        if type_code == CT_STRUCT:
            return self._read_struct(depth)
        if type_code == CT_UUID:
            return self._read(16)
        raise CompactProtocolError

    def _read_struct(self, depth: int) -> CompactStruct:
        if depth > MAX_COMPACT_DEPTH:
            raise CompactProtocolError
        fields: dict[int, CompactField] = {}
        last_field_id = 0
        while True:
            header = self._read_u8()
            if header == CT_STOP:
                return CompactStruct(fields=fields)
            delta = header >> 4
            type_code = header & 0x0F
            if type_code not in VALID_VALUE_TYPES:
                raise CompactProtocolError
            field_id = last_field_id + delta if delta else self._read_zigzag()
            if not 1 <= field_id <= 32_767 or field_id in fields or len(fields) >= MAX_STRUCT_FIELDS:
                raise CompactProtocolError
            last_field_id = field_id
            if type_code == CT_TRUE:
                value: Any = True
            elif type_code == CT_FALSE:
                value = False
            else:
                value = self._read_value(type_code, depth + 1)
            fields[field_id] = CompactField(type_code=type_code, value=value)

    def parse_root_struct(self) -> CompactStruct:
        result = self._read_struct(0)
        if self.offset != len(self.payload):
            raise CompactProtocolError
        return result


def _field(
    value: CompactStruct,
    field_id: int,
    expected_type: int,
    *,
    required: bool = False,
) -> Any | None:
    field = value.fields.get(field_id)
    if field is None:
        if required:
            raise ParquetMetadataError
        return None
    if field.type_code != expected_type:
        raise ParquetMetadataError
    return field.value


def _int_field(
    value: CompactStruct,
    field_id: int,
    expected_type: int,
    *,
    required: bool = False,
    minimum: int | None = None,
) -> int | None:
    result = _field(
        value,
        field_id,
        expected_type,
        required=required,
    )
    if result is None:
        return None
    if type(result) is not int or (minimum is not None and result < minimum):
        raise ParquetMetadataError
    return result


def _struct_field(
    value: CompactStruct,
    field_id: int,
    *,
    required: bool = False,
) -> CompactStruct | None:
    result = _field(value, field_id, CT_STRUCT, required=required)
    if result is None:
        return None
    if not isinstance(result, CompactStruct):
        raise ParquetMetadataError
    return result


def _list_field(
    value: CompactStruct,
    field_id: int,
    element_type: int,
    *,
    required: bool = False,
) -> CompactList | None:
    result = _field(value, field_id, CT_LIST, required=required)
    if result is None:
        return None
    if not isinstance(result, CompactList) or result.element_type != element_type:
        raise ParquetMetadataError
    return result


def _safe_text(raw: Any) -> str:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= 128:
        raise ParquetMetadataError
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParquetMetadataError from exc
    if not SAFE_SCHEMA_NAME.fullmatch(value):
        raise ParquetMetadataError
    return value


def _schema_node(value: CompactStruct) -> SchemaNode:
    name = _safe_text(_field(value, 4, CT_BINARY, required=True))
    physical_type = _int_field(value, 1, CT_I32, minimum=0)
    if physical_type is not None and physical_type not in PHYSICAL_TYPES:
        raise ParquetMetadataError
    repetition_type = _int_field(value, 3, CT_I32, minimum=0)
    if repetition_type is not None and repetition_type not in REPETITION_TYPES:
        raise ParquetMetadataError
    num_children = _int_field(value, 5, CT_I32, minimum=1)
    converted_type = _int_field(value, 6, CT_I32, minimum=0)
    if converted_type is not None and converted_type not in CONVERTED_TYPES:
        raise ParquetMetadataError
    logical_struct = _struct_field(value, 10)
    logical_type: str | None = None
    if logical_struct is not None:
        if len(logical_struct.fields) != 1:
            raise ParquetMetadataError
        logical_id = next(iter(logical_struct.fields))
        logical_type = LOGICAL_TYPES.get(logical_id)
        if logical_type is None:
            raise ParquetMetadataError
    if (physical_type is None) == (num_children is None):
        raise ParquetMetadataError
    return SchemaNode(
        name=name,
        physical_type=physical_type,
        repetition_type=repetition_type,
        num_children=num_children,
        converted_type=converted_type,
        logical_type=logical_type,
    )


def _schema_leaves(
    schema_values: CompactList,
) -> tuple[str, tuple[SchemaLeaf, ...]]:
    if not 1 <= len(schema_values.values) <= MAX_SCHEMA_ELEMENTS:
        raise ParquetMetadataError
    nodes: list[SchemaNode] = []
    for value in schema_values.values:
        if not isinstance(value, CompactStruct):
            raise ParquetMetadataError
        nodes.append(_schema_node(value))
    root = nodes[0]
    if root.physical_type is not None or root.repetition_type not in {None, 0} or root.num_children is None:
        raise ParquetMetadataError

    leaves: list[SchemaLeaf] = []

    def consume(index: int, path: tuple[str, ...]) -> int:
        if index >= len(nodes):
            raise ParquetMetadataError
        node = nodes[index]
        if node.num_children is None:
            if node.physical_type is None or node.repetition_type is None:
                raise ParquetMetadataError
            leaves.append(
                SchemaLeaf(
                    index=len(leaves),
                    path=path + (node.name,),
                    physical_type=node.physical_type,
                    repetition_type=node.repetition_type,
                    converted_type=node.converted_type,
                    logical_type=node.logical_type,
                )
            )
            return index + 1
        if node.physical_type is not None:
            raise ParquetMetadataError
        next_index = index + 1
        child_path = path + (node.name,)
        for _ in range(node.num_children):
            next_index = consume(next_index, child_path)
        return next_index

    next_index = 1
    for _ in range(root.num_children):
        next_index = consume(next_index, ())
    if next_index != len(nodes) or not leaves or len({leaf.path for leaf in leaves}) != len(leaves):
        raise ParquetMetadataError
    return root.name, tuple(leaves)


def _path_list(value: CompactList) -> tuple[str, ...]:
    if not value.values or len(value.values) > 64:
        raise ParquetMetadataError
    return tuple(_safe_text(item) for item in value.values)


def _column_chunk(
    value: CompactStruct,
    *,
    row_group_index: int,
    leaf: SchemaLeaf,
    footer_start: int,
) -> ColumnChunk:
    if _field(value, 1, CT_BINARY) is not None:
        raise ParquetMetadataError
    _int_field(value, 2, CT_I64, required=True)
    metadata = _struct_field(value, 3, required=True)
    if metadata is None or 8 in value.fields or 9 in value.fields:
        raise ParquetMetadataError

    physical_type = _int_field(metadata, 1, CT_I32, required=True, minimum=0)
    path = _list_field(metadata, 3, CT_BINARY, required=True)
    codec = _int_field(metadata, 4, CT_I32, required=True, minimum=0)
    num_values = _int_field(metadata, 5, CT_I64, required=True, minimum=0)
    total_uncompressed = _int_field(
        metadata,
        6,
        CT_I64,
        required=True,
        minimum=0,
    )
    total_compressed = _int_field(
        metadata,
        7,
        CT_I64,
        required=True,
        minimum=1,
    )
    data_page_offset = _int_field(
        metadata,
        9,
        CT_I64,
        required=True,
        minimum=4,
    )
    index_page_offset = _int_field(metadata, 10, CT_I64, minimum=4)
    dictionary_page_offset = _int_field(metadata, 11, CT_I64, minimum=4)
    offset_index_offset = _int_field(value, 4, CT_I64, minimum=4)
    offset_index_length = _int_field(value, 5, CT_I32, minimum=1)
    column_index_offset = _int_field(value, 6, CT_I64, minimum=4)
    column_index_length = _int_field(value, 7, CT_I32, minimum=1)

    if (
        physical_type is None
        or physical_type != leaf.physical_type
        or path is None
        or _path_list(path) != leaf.path
        or codec is None
        or codec not in COMPRESSION_CODECS
        or num_values is None
        or total_uncompressed is None
        or total_compressed is None
        or data_page_offset is None
        or (offset_index_offset is None) != (offset_index_length is None)
        or (column_index_offset is None) != (column_index_length is None)
    ):
        raise ParquetMetadataError
    if dictionary_page_offset is not None and dictionary_page_offset > data_page_offset:
        raise ParquetMetadataError
    page_offsets = [
        offset
        for offset in (
            data_page_offset,
            dictionary_page_offset,
            index_page_offset,
        )
        if offset is not None
    ]
    start = min(page_offsets)
    end = start + total_compressed - 1
    if end < start or end >= footer_start or any(offset < start or offset > end for offset in page_offsets):
        raise ParquetMetadataError
    for auxiliary_offset, auxiliary_length in (
        (offset_index_offset, offset_index_length),
        (column_index_offset, column_index_length),
    ):
        if (
            auxiliary_offset is not None
            and auxiliary_length is not None
            and auxiliary_offset + auxiliary_length > footer_start
        ):
            raise ParquetMetadataError
    return ColumnChunk(
        row_group_index=row_group_index,
        leaf_index=leaf.index,
        path=leaf.path,
        codec=codec,
        num_values=num_values,
        total_compressed_size=total_compressed,
        total_uncompressed_size=total_uncompressed,
        start=start,
        end=end,
        offset_index_offset=offset_index_offset,
        offset_index_length=offset_index_length,
        column_index_offset=column_index_offset,
        column_index_length=column_index_length,
    )


def _row_group(
    value: CompactStruct,
    *,
    index: int,
    leaves: tuple[SchemaLeaf, ...],
    footer_start: int,
) -> RowGroup:
    columns_value = _list_field(value, 1, CT_STRUCT, required=True)
    total_byte_size = _int_field(
        value,
        2,
        CT_I64,
        required=True,
        minimum=0,
    )
    num_rows = _int_field(value, 3, CT_I64, required=True, minimum=0)
    total_compressed_declared = _int_field(value, 6, CT_I64, minimum=0)
    ordinal = _int_field(value, 7, CT_I16, minimum=0)
    if (
        columns_value is None
        or len(columns_value.values) != len(leaves)
        or total_byte_size is None
        or num_rows is None
        or (ordinal is not None and ordinal != index)
    ):
        raise ParquetMetadataError
    columns: list[ColumnChunk] = []
    for leaf, column_value in zip(leaves, columns_value.values):
        if not isinstance(column_value, CompactStruct):
            raise ParquetMetadataError
        columns.append(
            _column_chunk(
                column_value,
                row_group_index=index,
                leaf=leaf,
                footer_start=footer_start,
            )
        )
    total_compressed = sum(column.total_compressed_size for column in columns)
    if total_compressed_declared is not None and total_compressed_declared != total_compressed:
        raise ParquetMetadataError
    return RowGroup(
        index=index,
        num_rows=num_rows,
        total_byte_size=total_byte_size,
        total_compressed_size=total_compressed,
        columns=tuple(columns),
    )


def _parse_parquet_metadata(
    footer: bytes,
    *,
    file_size: int,
) -> ParsedParquetMetadata:
    root = CompactReader(footer).parse_root_struct()
    version = _int_field(root, 1, CT_I32, required=True, minimum=1)
    schema_values = _list_field(root, 2, CT_STRUCT, required=True)
    num_rows = _int_field(root, 3, CT_I64, required=True, minimum=0)
    row_group_values = _list_field(root, 4, CT_STRUCT, required=True)
    if (
        version is None
        or schema_values is None
        or num_rows is None
        or row_group_values is None
        or not 1 <= len(row_group_values.values) <= MAX_ROW_GROUPS
        or 8 in root.fields
        or 9 in root.fields
    ):
        raise ParquetMetadataError

    root_name, leaves = _schema_leaves(schema_values)
    footer_start = file_size - 8 - len(footer)
    row_groups: list[RowGroup] = []
    for index, value in enumerate(row_group_values.values):
        if not isinstance(value, CompactStruct):
            raise ParquetMetadataError
        row_groups.append(
            _row_group(
                value,
                index=index,
                leaves=leaves,
                footer_start=footer_start,
            )
        )
    if sum(group.num_rows for group in row_groups) != num_rows:
        raise ParquetMetadataError

    chunks = sorted(
        (column for group in row_groups for column in group.columns),
        key=lambda column: (column.start, column.end),
    )
    for previous, current in zip(chunks, chunks[1:]):
        if previous.end >= current.start:
            raise ParquetMetadataError

    return ParsedParquetMetadata(
        version=version,
        root_name=root_name,
        num_rows=num_rows,
        leaves=leaves,
        row_groups=tuple(row_groups),
    )


def _load_exact_json(
    path: Path,
    *,
    expected_sha256: str,
    byte_cap: int,
    error_type: type[FooterPlanError],
) -> tuple[dict[str, Any], bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(byte_cap + 1)
    except OSError as exc:
        raise error_type from exc
    if not raw or len(raw) > byte_cap or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise error_type
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_type from exc
    if not isinstance(value, dict):
        raise error_type
    return value, raw


def _validate_access_record_binding(path: Path) -> None:
    _load_exact_json(
        path,
        expected_sha256=ACCESS_RECORD_SHA256,
        byte_cap=range_probe.MAX_ACCESS_RECORD_BYTES,
        error_type=AccessRecordBindingError,
    )


def _load_prior_probe() -> dict[str, Any]:
    value, _ = _load_exact_json(
        RAW_PROBE_PATH,
        expected_sha256=RAW_PROBE_SHA256,
        byte_cap=MAX_PRIOR_EVIDENCE_BYTES,
        error_type=PriorProbeEvidenceError,
    )
    expected_requests = [
        {"start": 0, "end": 3, "bytes": 4},
        {
            "start": range_probe.FIXED_SIZE_BYTES - 8,
            "end": range_probe.FIXED_SIZE_BYTES - 1,
            "bytes": 8,
        },
        {
            "start": range_probe.FIXED_SIZE_BYTES - 8 - RAW_FOOTER_LENGTH,
            "end": range_probe.FIXED_SIZE_BYTES - 9,
            "bytes": RAW_FOOTER_LENGTH,
        },
    ]
    try:
        valid = (
            value["schema"] == range_probe.EVIDENCE_SCHEMA
            and value["source"] == range_probe.FIXED_SOURCE
            and value["revision"] == range_probe.FIXED_REVISION
            and value["filename"] == range_probe.FIXED_FILENAME
            and value["size_bytes"] == range_probe.FIXED_SIZE_BYTES
            and value["footer"]
            == {
                "length_bytes": RAW_FOOTER_LENGTH,
                "sha256": RAW_FOOTER_SHA256,
            }
            and value["ranges"]["request_count"] == 3
            and value["ranges"]["bytes_received"] == RAW_FOOTER_LENGTH + 12
            and value["ranges"]["footer_byte_cap"] == range_probe.DEFAULT_MAX_FOOTER_BYTES
            and value["ranges"]["requests"] == expected_requests
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise PriorProbeEvidenceError
    return value


def _resolve_policy_fields(
    metadata: ParsedParquetMetadata,
) -> dict[str, Any]:
    top_level = {leaf.path[0]: leaf for leaf in metadata.leaves if len(leaf.path) == 1}
    resolved: dict[str, tuple[SchemaLeaf, ...]] = {}
    missing: list[str] = []
    for canonical, aliases in CORE_POLICY_ALIASES.items():
        matches = tuple(top_level[alias] for alias in aliases if alias in top_level)
        if not matches:
            missing.append(canonical)
        else:
            resolved[canonical] = matches

    optional = {
        canonical: tuple(top_level[alias] for alias in aliases if alias in top_level)
        for canonical, aliases in OPTIONAL_DATASET_ALIASES.items()
    }
    optional = {canonical: matches for canonical, matches in optional.items() if matches}
    opt_out = tuple(top_level[name] for name in EXPLICIT_OPT_OUT_ALIASES if name in top_level)
    explicit_paths = {leaf.path[0] for leaf in opt_out}
    related = sorted(name for name in top_level if OPT_OUT_RELATED_PATTERN.search(name) and name not in explicit_paths)
    return {
        "resolved_core": resolved,
        "resolved_optional": optional,
        "missing_core": missing,
        "opt_out": opt_out,
        "related": related,
    }


def _selected_column_plan(
    metadata: ParsedParquetMetadata,
    *,
    pilot_records: int,
    max_planned_bytes: int,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    resolved_core: dict[str, tuple[SchemaLeaf, ...]] = resolution["resolved_core"]
    resolved_optional: dict[str, tuple[SchemaLeaf, ...]] = resolution["resolved_optional"]
    missing_core: list[str] = resolution["missing_core"]
    opt_out: tuple[SchemaLeaf, ...] = resolution["opt_out"]
    selections: list[tuple[str, SchemaLeaf]] = []
    for canonical, leaves in {
        **resolved_core,
        **resolved_optional,
    }.items():
        selections.extend((canonical, leaf) for leaf in leaves)
    selections.extend(("explicit_per_record_opt_out", leaf) for leaf in opt_out)
    selections.sort(key=lambda item: item[1].index)

    selected_columns = [
        {
            "canonical_field": canonical,
            "source_path": ".".join(leaf.path),
            "leaf_index": leaf.index,
        }
        for canonical, leaf in selections
    ]
    candidates: list[dict[str, Any]] = []
    if selections:
        for group in metadata.row_groups:
            candidate_ranges = []
            for canonical, leaf in selections:
                chunk = group.columns[leaf.index]
                candidate_ranges.append(
                    {
                        "row_group_index": group.index,
                        "canonical_field": canonical,
                        "source_path": ".".join(chunk.path),
                        "start": chunk.start,
                        "end": chunk.end,
                        "bytes": chunk.total_compressed_size,
                    }
                )
            candidate_bytes = sum(item["bytes"] for item in candidate_ranges)
            candidates.append(
                {
                    "index": group.index,
                    "rows": group.num_rows,
                    "selected_column_bytes": candidate_bytes,
                    "within_pilot_record_cap": group.num_rows <= pilot_records,
                    "within_planned_byte_cap": candidate_bytes <= max_planned_bytes,
                    "ranges": candidate_ranges,
                }
            )

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["within_pilot_record_cap"] and candidate["within_planned_byte_cap"]
    ]
    selected_candidate = eligible_candidates[0] if not missing_core and opt_out and eligible_candidates else None
    ranges = selected_candidate["ranges"] if selected_candidate is not None else []
    planned_bytes = selected_candidate["selected_column_bytes"] if selected_candidate is not None else 0
    rows_planned = selected_candidate["rows"] if selected_candidate is not None else 0
    plan_ready = selected_candidate is not None
    return {
        "pilot_record_cap": pilot_records,
        "hard_pilot_record_cap": HARD_MAX_PILOT_RECORDS,
        "max_planned_column_bytes": max_planned_bytes,
        "row_selection_contract": (
            "whole row groups only; no truncation, implicit first-N, or partial-row-group consumption"
        ),
        "selected_columns": selected_columns,
        "whole_row_group_candidates": candidates,
        "candidate_row_group_count": len(candidates),
        "minimum_candidate_rows": (min(candidate["rows"] for candidate in candidates) if candidates else None),
        "minimum_candidate_selected_column_bytes": (
            min(candidate["selected_column_bytes"] for candidate in candidates) if candidates else None
        ),
        "selected_row_group_index": (selected_candidate["index"] if selected_candidate is not None else None),
        "rows_planned": rows_planned,
        "request_count": len(ranges),
        "bytes_planned": planned_bytes,
        "ranges": ranges,
        "range_inventory_ready": not missing_core and bool(candidates),
        "plan_ready": plan_ready,
        "column_payload_requests_performed": 0,
        "column_payload_bytes_received": 0,
        "column_payload_files_persisted": 0,
    }


def _schema_evidence(metadata: ParsedParquetMetadata) -> list[dict[str, Any]]:
    return [
        {
            "leaf_index": leaf.index,
            "path": ".".join(leaf.path),
            "physical_type": PHYSICAL_TYPES[leaf.physical_type],
            "repetition_type": REPETITION_TYPES[leaf.repetition_type],
            "converted_type": (CONVERTED_TYPES[leaf.converted_type] if leaf.converted_type is not None else None),
            "logical_type": leaf.logical_type,
        }
        for leaf in metadata.leaves
    ]


def _build_evidence(
    *,
    discovery: range_probe.FooterDiscovery,
    prior_probe: dict[str, Any],
    metadata: ParsedParquetMetadata,
    pilot_records: int,
    max_planned_bytes: int,
) -> dict[str, Any]:
    if discovery.evidence != prior_probe:
        raise FooterEvidenceMismatchError
    if len(discovery.footer) != RAW_FOOTER_LENGTH or hashlib.sha256(discovery.footer).hexdigest() != RAW_FOOTER_SHA256:
        raise FooterEvidenceMismatchError

    resolution = _resolve_policy_fields(metadata)
    plan = _selected_column_plan(
        metadata,
        pilot_records=pilot_records,
        max_planned_bytes=max_planned_bytes,
        resolution=resolution,
    )
    resolved_core: dict[str, tuple[SchemaLeaf, ...]] = resolution["resolved_core"]
    resolved_optional: dict[str, tuple[SchemaLeaf, ...]] = resolution["resolved_optional"]
    missing_core: list[str] = resolution["missing_core"]
    opt_out: tuple[SchemaLeaf, ...] = resolution["opt_out"]

    gate_reasons: list[str] = []
    if missing_core:
        gate_reasons.append("MISSING_CORE_POLICY_COLUMNS")
    if not opt_out:
        gate_reasons.append("EXPLICIT_PER_RECORD_OPT_OUT_FIELD_NOT_OBSERVED")
    else:
        gate_reasons.append("OPT_OUT_VALUE_AND_REMOVAL_LEDGER_PROOF_PENDING")
    candidates = plan["whole_row_group_candidates"]
    if candidates and not any(candidate["within_pilot_record_cap"] for candidate in candidates):
        gate_reasons.append("ALL_ROW_GROUPS_EXCEED_PILOT_RECORD_CAP")
    if candidates and not any(candidate["within_planned_byte_cap"] for candidate in candidates):
        gate_reasons.append("ALL_ROW_GROUPS_EXCEED_PLANNED_BYTE_CAP")
    if not plan["plan_ready"]:
        gate_reasons.append("BOUNDED_SELECTED_COLUMN_PLAN_NOT_READY")

    return {
        "schema": EVIDENCE_SCHEMA,
        "source": range_probe.FIXED_SOURCE,
        "revision": range_probe.FIXED_REVISION,
        "filename": range_probe.FIXED_FILENAME,
        "size_bytes": range_probe.FIXED_SIZE_BYTES,
        "authorization": discovery.evidence["authorization"],
        "prior_raw_footer_probe": {
            "path": RAW_PROBE_RELATIVE_PATH,
            "sha256": RAW_PROBE_SHA256,
            "footer_length_bytes": RAW_FOOTER_LENGTH,
            "footer_sha256": RAW_FOOTER_SHA256,
        },
        "footer_refetch": {
            "request_count": discovery.evidence["ranges"]["request_count"],
            "bytes_received": discovery.evidence["ranges"]["bytes_received"],
            "footer_byte_cap": discovery.evidence["ranges"]["footer_byte_cap"],
            "raw_footer_persisted": False,
            "dependency_changes_performed": False,
        },
        "parser": {
            "serialization": "Apache Thrift TCompactProtocol",
            "implementation": "bounded Python standard-library reader",
            "full_parquet_file_required": False,
        },
        "file_metadata": {
            "parquet_version": metadata.version,
            "root_name": metadata.root_name,
            "num_rows": metadata.num_rows,
            "row_group_count": len(metadata.row_groups),
            "leaf_column_count": len(metadata.leaves),
            "columns": _schema_evidence(metadata),
        },
        "policy_field_resolution": {
            "core_fields": {
                canonical: [".".join(leaf.path) for leaf in leaves]
                for canonical, leaves in sorted(resolved_core.items())
            },
            "optional_fields": {
                canonical: [".".join(leaf.path) for leaf in leaves]
                for canonical, leaves in sorted(resolved_optional.items())
            },
            "missing_core_fields": sorted(missing_core),
            "explicit_per_record_opt_out_field_observed": bool(opt_out),
            "explicit_per_record_opt_out_fields": [".".join(leaf.path) for leaf in opt_out],
            "related_but_not_explicit_fields": resolution["related"],
            "proof_status": (
                "NAME_OBSERVED_VALUE_AND_REMOVAL_LEDGER_PROOF_PENDING" if opt_out else "ABSENT_FROM_SCHEMA"
            ),
        },
        "selected_column_plan": plan,
        "gate": {
            "g1": "HOLD",
            "reasons": gate_reasons,
        },
        "activity": {
            "metadata_range_requests_performed": discovery.evidence["ranges"]["request_count"],
            "metadata_range_bytes_received": discovery.evidence["ranges"]["bytes_received"],
            "full_metadata_file_downloaded": False,
            "image_files_downloaded": 0,
            "image_bytes_downloaded": 0,
            "training_runs_started": 0,
        },
        "status": "FOOTER_PARSED_SELECTED_COLUMN_PLAN_EVALUATED_G1_HOLD",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = range_probe._SafeArgumentParser(
        description=(
            "Parse the fixed Parquet footer in memory and emit a bounded "
            "selected-column byte-range plan without reading column payloads."
        )
    )
    parser.add_argument("--access-record", required=True, type=Path)
    parser.add_argument(
        "--pilot-records",
        type=range_probe._positive_int,
        default=DEFAULT_PILOT_RECORDS,
    )
    parser.add_argument(
        "--max-planned-column-bytes",
        type=range_probe._positive_int,
        default=DEFAULT_MAX_PLANNED_COLUMN_BYTES,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=range_probe._timeout,
        default=30.0,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _probe_config(args: argparse.Namespace) -> range_probe.ProbeConfig:
    if args.pilot_records > HARD_MAX_PILOT_RECORDS or args.max_planned_column_bytes > HARD_MAX_PLANNED_COLUMN_BYTES:
        raise range_probe.SafeArgumentError
    _validate_access_record_binding(args.access_record)
    return range_probe._config_from_args(
        argparse.Namespace(
            access_record=args.access_record,
            max_footer_bytes=range_probe.DEFAULT_MAX_FOOTER_BYTES,
            timeout_seconds=args.timeout_seconds,
            output=Path(os.path.abspath(args.output)),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        config = _probe_config(args)
        range_probe._assert_output_available(config.output)
        prior_probe = _load_prior_probe()
        discovery = range_probe._discover_with_footer(config)
        metadata = _parse_parquet_metadata(
            discovery.footer,
            file_size=config.expected_size,
        )
        evidence = _build_evidence(
            discovery=discovery,
            prior_probe=prior_probe,
            metadata=metadata,
            pilot_records=args.pilot_records,
            max_planned_bytes=args.max_planned_column_bytes,
        )
        range_probe._write_no_clobber(config.output, evidence)
    except Exception as exc:
        range_probe._emit_safe_error(exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
