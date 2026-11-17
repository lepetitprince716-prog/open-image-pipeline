#!/usr/bin/env python3
"""Credential-safe, byte-capped Parquet footer discovery for Anima Step 2a.

The probe authenticates only the Hugging Face metadata request.  It then reads
the signed metadata ``location`` without an Authorization header and fetches
exactly three HTTP ranges: the four-byte file magic, the eight-byte trailer,
and the Thrift footer containing the Parquet row-group directory.

Parquet bytes remain in memory.  The only persistent artifact is a compact,
no-clobber JSON evidence record containing safe identifiers, hashes, and range
accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

ACCESS_SCHEMA = "anima-step2a-source-access-authorization-v1"
EVIDENCE_SCHEMA = "anima-step2a-parquet-range-probe-evidence-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_SOURCE = "deepghs/danbooru2024-webp-4Mpixel"
FIXED_REVISION = "60ca7e02a844bccbe6c8344d1a18b7ea280b787f"
FIXED_FILENAME = "metadata.parquet"
FIXED_SIZE_BYTES = 6_846_458_425
FIXED_LICENSE_SHA256 = "d9827d69b4d5e9acb79871b589b09f62bc8e711798f17f78740923045815338c"
FIXED_APPROVAL_ID = "anima-step2a-policy-approval-20260718-v1"
FIXED_APPROVAL_RECORD_SHA256 = "388922c14c093093c4f10ea80cfde6789d40dc840af31321ac9154b38cf60cc2"
FIXED_POLICY_DIGEST = "b037bd6804b08bceb9684365509a2784f2b1a66cf34a64da240fd79546cb0bc9"
APPROVAL_RECORD_PATH = PROJECT_ROOT / "docs/lora_training/anima_step2a/APPROVAL_RECORD_v1.json"
DEFAULT_MAX_FOOTER_BYTES = 16 * 1024 * 1024
HARD_MAX_FOOTER_BYTES = 64 * 1024 * 1024
MAX_ACCESS_RECORD_BYTES = 16 * 1024
MAX_APPROVAL_RECORD_BYTES = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024
PARQUET_MAGIC = b"PAR1"
ACCESS_FIELDS = frozenset(
    {
        "schema",
        "source",
        "revision",
        "filename",
        "size_bytes",
        "license_sha256",
        "user_accepted_date",
        "approval_id",
        "approval_record_sha256",
        "policy_digest",
    }
)
SAFE_ERROR_CLASS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class ProbeError(RuntimeError):
    """Base class for deterministic, fail-closed probe errors."""


class SafeArgumentError(ProbeError):
    pass


class AccessRecordError(ProbeError):
    pass


class ApprovalRecordError(ProbeError):
    pass


class NoClobberError(ProbeError):
    pass


class OutputPublishError(ProbeError):
    pass


class MissingCredentialError(ProbeError):
    pass


class MetadataRevisionError(ProbeError):
    pass


class MetadataSizeError(ProbeError):
    pass


class MetadataLocationError(ProbeError):
    pass


class RangeStatusError(ProbeError):
    def __init__(self, status_code: int) -> None:
        super().__init__()
        self.status_code = status_code


class RangeIgnoredError(RangeStatusError):
    pass


class RangeHeaderError(ProbeError):
    pass


class TruncatedRangeError(ProbeError):
    pass


class OverlongRangeError(ProbeError):
    pass


class BadParquetMagicError(ProbeError):
    pass


class InvalidFooterLengthError(ProbeError):
    pass


class FooterTooLargeError(ProbeError):
    pass


class RedirectPolicyError(ProbeError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    """Never echo command-line values from parse failures."""

    def error(self, message: str) -> None:  # noqa: ARG002
        raise SafeArgumentError


class _HTTPSOnlyRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Keep signed redirect material on encrypted transport."""

    def redirect_request(
        self,
        req: urlrequest.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urlrequest.Request | None:
        try:
            _validate_location(newurl)
        except MetadataLocationError as exc:
            raise RedirectPolicyError from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class ProbeConfig:
    source: str
    revision: str
    filename: str
    expected_size: int
    license_sha256: str
    user_accepted_date: str
    approval_id: str
    approval_record_sha256: str
    policy_digest: str
    max_footer_bytes: int
    timeout_seconds: float
    output: Path


@dataclass(frozen=True)
class FooterDiscovery:
    """In-memory footer bytes plus the safe evidence derived from them."""

    evidence: dict[str, Any]
    footer: bytes


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError
    return parsed


def _timeout(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError from exc
    if not 0.1 <= parsed <= 300.0:
        raise argparse.ArgumentTypeError
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Fetch only the bounded Parquet header, trailer, and row-group "
            "directory footer from an immutable Hugging Face dataset revision."
        )
    )
    parser.add_argument("--access-record", required=True, type=Path)
    parser.add_argument(
        "--max-footer-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_FOOTER_BYTES,
    )
    parser.add_argument("--timeout-seconds", type=_timeout, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _is_exact_iso_date(value: str) -> bool:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AccessRecordError
        result[key] = value
    return result


def _load_access_record(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_ACCESS_RECORD_BYTES + 1)
    except OSError as exc:
        raise AccessRecordError from exc
    if not raw or len(raw) > MAX_ACCESS_RECORD_BYTES:
        raise AccessRecordError
    try:
        record = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AccessRecordError) as exc:
        raise AccessRecordError from exc
    if not isinstance(record, dict):
        raise AccessRecordError
    return record


def _validate_access_record(record: dict[str, Any]) -> str:
    if set(record) != ACCESS_FIELDS:
        raise AccessRecordError
    expected = {
        "schema": ACCESS_SCHEMA,
        "source": FIXED_SOURCE,
        "revision": FIXED_REVISION,
        "filename": FIXED_FILENAME,
        "size_bytes": FIXED_SIZE_BYTES,
        "license_sha256": FIXED_LICENSE_SHA256,
        "approval_id": FIXED_APPROVAL_ID,
        "approval_record_sha256": FIXED_APPROVAL_RECORD_SHA256,
        "policy_digest": FIXED_POLICY_DIGEST,
    }
    if type(record["size_bytes"]) is not int:
        raise AccessRecordError
    if any(type(record[key]) is not str for key in ACCESS_FIELDS if key != "size_bytes"):
        raise AccessRecordError
    if any(record[key] != value for key, value in expected.items()):
        raise AccessRecordError
    accepted_date = record["user_accepted_date"]
    if not _is_exact_iso_date(accepted_date):
        raise AccessRecordError
    return accepted_date


def _validate_checked_in_approval_record() -> None:
    try:
        with APPROVAL_RECORD_PATH.open("rb") as handle:
            raw = handle.read(MAX_APPROVAL_RECORD_BYTES + 1)
    except OSError as exc:
        raise ApprovalRecordError from exc
    if not raw or len(raw) > MAX_APPROVAL_RECORD_BYTES:
        raise ApprovalRecordError
    if hashlib.sha256(raw).hexdigest() != FIXED_APPROVAL_RECORD_SHA256:
        raise ApprovalRecordError


def _config_from_args(args: argparse.Namespace) -> ProbeConfig:
    if args.max_footer_bytes > HARD_MAX_FOOTER_BYTES:
        raise SafeArgumentError
    record = _load_access_record(args.access_record)
    accepted_date = _validate_access_record(record)
    _validate_checked_in_approval_record()
    return ProbeConfig(
        source=FIXED_SOURCE,
        revision=FIXED_REVISION,
        filename=FIXED_FILENAME,
        expected_size=FIXED_SIZE_BYTES,
        license_sha256=FIXED_LICENSE_SHA256,
        user_accepted_date=accepted_date,
        approval_id=FIXED_APPROVAL_ID,
        approval_record_sha256=FIXED_APPROVAL_RECORD_SHA256,
        policy_digest=FIXED_POLICY_DIGEST,
        max_footer_bytes=args.max_footer_bytes,
        timeout_seconds=args.timeout_seconds,
        output=Path(os.path.abspath(args.output)),
    )


def _assert_output_available(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise NoClobberError from exc
    raise NoClobberError


def _write_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing path."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputPublishError from exc

    temporary: Path | None = None
    directory_fd: int | None = None
    published = False
    try:
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                serialized = (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as exc:
            raise OutputPublishError from exc

        try:
            os.link(temporary, path)
            published = True
        except FileExistsError as exc:
            raise NoClobberError from exc
        except OSError as exc:
            raise OutputPublishError from exc

        if os.name != "nt":
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                os.fsync(directory_fd)
            except OSError as exc:
                try:
                    path.unlink()
                    published = False
                except OSError:
                    pass
                raise OutputPublishError from exc
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published:
            # ``published`` is intentionally retained only as local state.  The
            # evidence file itself never contains temp paths or source URLs.
            pass


def _validate_location(location: Any) -> str:
    if not isinstance(location, str) or not location or len(location) > 32_768:
        raise MetadataLocationError
    try:
        parts = urlsplit(location)
        hostname = parts.hostname
        username = parts.username
        password = parts.password
        parts.port
    except ValueError as exc:
        raise MetadataLocationError from exc
    if (
        parts.scheme.lower() != "https"
        or not hostname
        or username is not None
        or password is not None
        or parts.fragment
    ):
        raise MetadataLocationError
    return location


def _status_code(response: Any) -> int | None:
    status = getattr(response, "status", None)
    if type(status) is int:
        return status
    try:
        code = response.getcode()
    except Exception:
        return None
    return code if type(code) is int else None


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _read_exact_bounded(response: Any, expected_bytes: int) -> bytes:
    payload = bytearray()
    while len(payload) < expected_bytes:
        request_bytes = min(READ_CHUNK_BYTES, expected_bytes - len(payload))
        chunk = response.read(request_bytes)
        if not isinstance(chunk, bytes):
            raise TruncatedRangeError
        if not chunk:
            raise TruncatedRangeError
        payload.extend(chunk)
        if len(payload) > expected_bytes:
            raise OverlongRangeError
    extra = response.read(1)
    if not isinstance(extra, bytes):
        raise OverlongRangeError
    if extra:
        raise OverlongRangeError
    return bytes(payload)


def _fetch_range(
    opener: Any,
    location: str,
    *,
    start: int,
    end: int,
    file_size: int,
    timeout_seconds: float,
) -> bytes:
    expected_bytes = end - start + 1
    request = urlrequest.Request(
        location,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "loramake-anima-step2a-parquet-range-probe/1.0",
        },
        method="GET",
    )
    response: Any | None = None
    try:
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except urlerror.HTTPError as exc:
            try:
                exc.close()
            finally:
                status = exc.code if type(exc.code) is int else 0
            raise RangeStatusError(status) from None

        status = _status_code(response)
        if status == 200:
            # Do not call read: a Range-ignoring service could otherwise stream
            # the complete multi-gigabyte Parquet object.
            raise RangeIgnoredError(200)
        if status != 206:
            raise RangeStatusError(status if status is not None else 0)

        if _header(response, "Content-Range") != f"bytes {start}-{end}/{file_size}":
            raise RangeHeaderError
        if _header(response, "Content-Length") != str(expected_bytes):
            raise RangeHeaderError
        encoding = _header(response, "Content-Encoding")
        if encoding not in {None, "identity"}:
            raise RangeHeaderError
        return _read_exact_bounded(response, expected_bytes)
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _build_opener() -> Any:
    return urlrequest.build_opener(_HTTPSOnlyRedirectHandler())


def _metadata_values(metadata: Any, config: ProbeConfig) -> tuple[str, str | None]:
    commit = getattr(metadata, "commit_hash", None)
    if not isinstance(commit, str) or commit != config.revision:
        raise MetadataRevisionError

    size = getattr(metadata, "size", None)
    if type(size) is not int or size != config.expected_size:
        raise MetadataSizeError

    location = _validate_location(getattr(metadata, "location", None))
    etag = getattr(metadata, "etag", None)
    if etag is not None and not isinstance(etag, str):
        raise MetadataLocationError
    return location, etag


def _discover_with_footer(
    config: ProbeConfig,
    *,
    opener: Any | None = None,
) -> FooterDiscovery:
    secret = get_token()
    if not isinstance(secret, str) or not secret.strip():
        raise MissingCredentialError

    try:
        hub_url = hf_hub_url(
            repo_id=config.source,
            filename=config.filename,
            repo_type="dataset",
            revision=config.revision,
        )
        metadata = get_hf_file_metadata(
            hub_url,
            token=secret,
            timeout=config.timeout_seconds,
        )
    finally:
        # Avoid retaining the credential after the only authenticated call.
        secret = None

    location, etag = _metadata_values(metadata, config)
    active_opener = opener if opener is not None else _build_opener()

    header = _fetch_range(
        active_opener,
        location,
        start=0,
        end=3,
        file_size=config.expected_size,
        timeout_seconds=config.timeout_seconds,
    )
    if header != PARQUET_MAGIC:
        raise BadParquetMagicError

    trailer_start = config.expected_size - 8
    trailer = _fetch_range(
        active_opener,
        location,
        start=trailer_start,
        end=config.expected_size - 1,
        file_size=config.expected_size,
        timeout_seconds=config.timeout_seconds,
    )
    if trailer[4:] != PARQUET_MAGIC:
        raise BadParquetMagicError

    footer_length = int.from_bytes(trailer[:4], byteorder="little", signed=False)
    if footer_length <= 0:
        raise InvalidFooterLengthError
    if footer_length > config.max_footer_bytes:
        raise FooterTooLargeError

    footer_start = config.expected_size - 8 - footer_length
    footer_end = config.expected_size - 9
    if footer_start < 4 or footer_end < footer_start:
        raise InvalidFooterLengthError

    footer = _fetch_range(
        active_opener,
        location,
        start=footer_start,
        end=footer_end,
        file_size=config.expected_size,
        timeout_seconds=config.timeout_seconds,
    )

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "source": config.source,
        "revision": config.revision,
        "filename": config.filename,
        "size_bytes": config.expected_size,
        "authorization": {
            "license_sha256": config.license_sha256,
            "user_accepted_date": config.user_accepted_date,
            "approval_id": config.approval_id,
            "approval_record_sha256": config.approval_record_sha256,
            "policy_digest": config.policy_digest,
        },
        "etag_sha256": (hashlib.sha256(etag.encode("utf-8")).hexdigest() if etag is not None else None),
        "footer": {
            "length_bytes": footer_length,
            "sha256": hashlib.sha256(footer).hexdigest(),
        },
        "ranges": {
            "request_count": 3,
            "bytes_received": 4 + 8 + footer_length,
            "footer_byte_cap": config.max_footer_bytes,
            "requests": [
                {"start": 0, "end": 3, "bytes": 4},
                {
                    "start": trailer_start,
                    "end": config.expected_size - 1,
                    "bytes": 8,
                },
                {
                    "start": footer_start,
                    "end": footer_end,
                    "bytes": footer_length,
                },
            ],
        },
    }
    return FooterDiscovery(evidence=evidence, footer=footer)


def _discover(config: ProbeConfig, *, opener: Any | None = None) -> dict[str, Any]:
    """Preserve the v1 evidence-only API while follow-ons parse bytes in memory."""

    return _discover_with_footer(config, opener=opener).evidence


def _exception_status(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if type(direct) is int:
        return direct
    direct = getattr(exc, "code", None)
    if type(direct) is int:
        return direct
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if type(status) is int else None


def _emit_safe_error(exc: Exception) -> None:
    error_class = type(exc).__name__
    if not SAFE_ERROR_CLASS.fullmatch(error_class):
        error_class = "Exception"
    payload: dict[str, Any] = {"error_class": error_class}
    status = _exception_status(exc)
    if status is not None:
        payload["status"] = status
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        config = _config_from_args(args)
        _assert_output_available(config.output)
        evidence = _discover(config)
        _write_no_clobber(config.output, evidence)
    except Exception as exc:
        _emit_safe_error(exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
