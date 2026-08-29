#!/usr/bin/env python3
"""Keeper-only per-shot finishing: RIFE 2x -> SeedVR2 2x -> 9:16 conform.

``plan`` is read-only unless ``--verify`` is selected.  Verification runs only local
version/model probes and a tiny synthetic adapter canary, then records the evidence in
the plan.  Keeper admission is bound to an immutable review record and the exact clip
bytes.  ``run`` re-verifies current local readiness, re-derives the execution basis from
the unchanged source manifest, and then executes only explicitly configured argv arrays with
``shell=False``.  It has no import or call path into ComfyUI/generation graphs.

Examples:

  python3 scripts/video_finish.py plan finish.json --out finish.plan.json --verify --json
  python3 scripts/video_finish.py run finish.plan.json --json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake.video_edit import VideoEditError, probe_media, sha256_file  # noqa: E402
from loramake.video_direction import (  # noqa: E402
    VideoDirectionError,
    frame_timecode,
    iter_shots,
    validate_direction,
)

FINISH_SCHEMA = "loramake-video-finish-v1"
FINISH_PLAN_SCHEMA = "loramake-video-finish-plan-v1"
FINISH_REPORT_SCHEMA = "loramake-video-finish-report-v1"
EXECUTION_POLICY = "keeper_only_local_adapters"
TARGET = {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "aspect_ratio": "9:16",
    "rife_multiplier": 2,
    "seedvr2_scale": 2,
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX64 = re.compile(r"^[a-f0-9]{64}$")
SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]+$")
ALLOWED_PLACEHOLDERS = {
    "input", "output", "multiplier", "scale", "shot_id", "target_width", "target_height", "target_fps",
}
REVIEW_SCHEMA = "loramake-video-review-v1"
REVIEW_DIMENSIONS = (
    "motion", "camera", "beat", "identity", "continuity", "flicker", "line_boiling", "palette",
)
REVIEW_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
REVIEW_TIMECODE_RE = re.compile(r"^[0-9]{2,}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}$")
VERSION_RE = re.compile(r"(?<![0-9])(\d+)\.(\d+)(?:\.(\d+))?")
READINESS_METHOD = "local-version-model-canary-v1"


class VideoFinishError(RuntimeError):
    """Manifest, readiness, execution, or no-clobber failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VideoFinishError(message)


def _identifier(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(ID_RE.fullmatch(value)), f"{label} is not a safe identifier")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    _require(not missing, f"{label} missing required fields: {', '.join(missing)}")
    _require(not extra, f"{label} has unsupported fields: {', '.join(extra)}")


def _local_path(value: object, *, base: Path, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty local path")
    _require(not value.startswith(("http://", "https://")), f"{label} must be a local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VideoFinishError(f"{label} not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise VideoFinishError(f"{label} is not valid JSON: {source}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return source, payload


def _placeholders(argv: Sequence[str]) -> set[str]:
    fields: set[str] = set()
    formatter = string.Formatter()
    for token in argv:
        try:
            parsed = formatter.parse(token)
            for _, field, _, _ in parsed:
                if field:
                    fields.add(field)
        except ValueError as exc:
            raise VideoFinishError(f"adapter argv contains invalid braces: {token!r}") from exc
    return fields


def _resolve_executable(token: str, *, base: Path) -> tuple[str, bool]:
    path = Path(token).expanduser()
    if path.is_absolute() or "/" in token or "\\" in token:
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        return str(path), path.is_file() and os.access(path, os.X_OK)
    located = shutil.which(token)
    return (str(Path(located).resolve()), True) if located else (token, False)


def _validate_local_argv(value: object, *, base: Path, label: str) -> list[str]:
    argv = _array(value, label)
    _require(bool(argv) and all(isinstance(token, str) and token for token in argv),
             f"{label} must contain non-empty strings")
    _require(not any(token.startswith(("http://", "https://")) for token in argv),
             f"{label} may reference local resources only")
    executable, _ = _resolve_executable(argv[0], base=base)
    return [executable, *argv[1:]]


def _validate_adapter(name: str, raw: object, *, base: Path) -> dict[str, Any]:
    adapter = copy.deepcopy(_object(raw, f"adapters.{name}"))
    required = {"argv", "output_extension", "check_paths", "version_argv"}
    if name == "seedvr2":
        required.add("model_files")
    _exact_keys(adapter, required=required, optional=set(),
                label=f"adapters.{name}")
    argv = _validate_local_argv(adapter["argv"], base=base, label=f"adapters.{name}.argv")
    placeholders = _placeholders(argv)
    _require(placeholders <= ALLOWED_PLACEHOLDERS,
             f"adapters.{name}.argv has unsupported placeholders: {sorted(placeholders - ALLOWED_PLACEHOLDERS)}")
    _require({"input", "output"} <= placeholders,
             f"adapters.{name}.argv must contain {{input}} and {{output}}")
    extension = adapter["output_extension"]
    _require(isinstance(extension, str) and bool(SAFE_EXTENSION.fullmatch(extension)),
             f"adapters.{name}.output_extension must look like .mp4")
    check_paths = _array(adapter["check_paths"], f"adapters.{name}.check_paths")
    resolved_checks: list[str] = []
    for index, item in enumerate(check_paths):
        resolved_checks.append(str(_local_path(item, base=base, label=f"adapters.{name}.check_paths[{index}]")))
    version_argv = _validate_local_argv(
        adapter["version_argv"], base=base, label=f"adapters.{name}.version_argv"
    )
    _require(not _placeholders(version_argv), f"adapters.{name}.version_argv must not contain placeholders")

    model_files: list[dict[str, Any]] = []
    if name == "seedvr2":
        raw_models = _array(adapter["model_files"], "adapters.seedvr2.model_files")
        _require(bool(raw_models), "adapters.seedvr2.model_files must not be empty")
        for index, raw_model in enumerate(raw_models):
            model = copy.deepcopy(_object(raw_model, f"adapters.seedvr2.model_files[{index}]"))
            _exact_keys(
                model,
                required={"path", "sha256", "bytes"},
                optional=set(),
                label=f"adapters.seedvr2.model_files[{index}]",
            )
            model_path = _local_path(
                model["path"], base=base, label=f"adapters.seedvr2.model_files[{index}].path"
            )
            _require(isinstance(model["sha256"], str) and bool(HEX64.fullmatch(model["sha256"])),
                     f"adapters.seedvr2.model_files[{index}].sha256 must be 64 lowercase hex characters")
            _require(isinstance(model["bytes"], int) and not isinstance(model["bytes"], bool)
                     and model["bytes"] > 0,
                     f"adapters.seedvr2.model_files[{index}].bytes must be a positive integer")
            model["path"] = str(model_path)
            model_files.append(model)
    executable, executable_ready = _resolve_executable(argv[0], base=base)
    argv[0] = executable
    adapter["argv"] = argv
    adapter["check_paths"] = resolved_checks
    adapter["version_argv"] = version_argv
    if name == "seedvr2":
        adapter["model_files"] = model_files
    adapter["readiness"] = {
        "configured": True,
        "executable": executable,
        "executable_found": executable_ready,
        "check_paths": [
            {"path": path, "exists": Path(path).is_file()} for path in resolved_checks
        ],
        "version_probe": {"status": "not_run", "argv": version_argv},
        "compatible": False,
        "model_files": [
            {
                "path": model["path"],
                "expected_sha256": model["sha256"],
                "expected_bytes": model["bytes"],
                "status": "not_checked",
            }
            for model in model_files
        ],
        "canary": {"status": "not_run"},
        "verified": False,
        "ready": False,
    }
    return adapter


def _validate_review_record(
    *,
    review_path: Path,
    declared_hash: str,
    media: Path,
    media_hash: str,
    project_id: str,
    shot_id: str,
) -> dict[str, Any]:
    _require(review_path.is_file(), f"keeper review record not found: {review_path}")
    _require(isinstance(declared_hash, str) and bool(HEX64.fullmatch(declared_hash)),
             f"keeper {shot_id}.review.sha256 must be 64 lowercase hex characters")
    _require(sha256_file(review_path) == declared_hash, f"keeper {shot_id} review record hash mismatch")
    _, review = _load_object(review_path, f"keeper {shot_id} review record")
    _exact_keys(
        review,
        required={
            "schema", "record_id", "project_id", "direction", "clip", "scene_id", "shot_id",
            "timecode", "reviewer", "verdicts", "summary", "created_at",
        },
        optional=set(),
        label=f"keeper {shot_id} review record",
    )
    _require(review.get("schema") == REVIEW_SCHEMA, f"keeper {shot_id} review schema must be {REVIEW_SCHEMA!r}")
    _require(
        isinstance(review.get("record_id"), str) and bool(REVIEW_ID_RE.fullmatch(review["record_id"])),
        f"keeper {shot_id} review.record_id violates the review schema",
    )
    _require(review.get("project_id") == project_id, f"keeper {shot_id} review project_id mismatch")
    _require(
        isinstance(review.get("scene_id"), str) and bool(REVIEW_ID_RE.fullmatch(review["scene_id"])),
        f"keeper {shot_id} review.scene_id violates the review schema",
    )
    _require(review.get("shot_id") == shot_id, f"keeper {shot_id} review shot_id mismatch")

    direction_ref = _object(review.get("direction"), f"keeper {shot_id} review.direction")
    _exact_keys(
        direction_ref,
        required={"path", "sha256"},
        optional=set(),
        label=f"keeper {shot_id} review.direction",
    )
    direction_path = _local_path(
        direction_ref["path"], base=review_path.parent, label=f"keeper {shot_id} review.direction.path"
    )
    direction_hash = direction_ref.get("sha256")
    _require(
        isinstance(direction_hash, str) and bool(HEX64.fullmatch(direction_hash)),
        f"keeper {shot_id} review.direction.sha256 must be 64 lowercase hex characters",
    )
    _require(direction_path.is_file(), f"keeper {shot_id} review direction not found: {direction_path}")
    _require(sha256_file(direction_path) == direction_hash, f"keeper {shot_id} review direction hash mismatch")
    _, direction = _load_object(direction_path, f"keeper {shot_id} review direction")
    try:
        validate_direction(direction)
    except VideoDirectionError as exc:
        raise VideoFinishError(f"keeper {shot_id} review direction is invalid: {exc}") from None
    _require(
        direction.get("project", {}).get("id") == project_id,
        f"keeper {shot_id} review direction project_id mismatch",
    )
    reviewed_scene = None
    reviewed_shot = None
    for scene, shot in iter_shots(direction):
        if shot.get("id") == shot_id:
            reviewed_scene, reviewed_shot = scene, shot
            break
    _require(reviewed_shot is not None, f"keeper {shot_id} review direction has no matching shot")
    _require(
        reviewed_scene.get("id") == review.get("scene_id"),
        f"keeper {shot_id} review scene_id mismatch",
    )

    timecode = _object(review.get("timecode"), f"keeper {shot_id} review.timecode")
    _exact_keys(
        timecode,
        required={"value", "frame", "relative_frame"},
        optional=set(),
        label=f"keeper {shot_id} review.timecode",
    )
    frame = timecode.get("frame")
    relative_frame = timecode.get("relative_frame")
    _require(
        isinstance(timecode.get("value"), str) and bool(REVIEW_TIMECODE_RE.fullmatch(timecode["value"])),
        f"keeper {shot_id} review.timecode.value violates the review schema",
    )
    _require(isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0,
             f"keeper {shot_id} review.timecode.frame must be a non-negative integer")
    _require(isinstance(relative_frame, int) and not isinstance(relative_frame, bool) and relative_frame >= 0,
             f"keeper {shot_id} review.timecode.relative_frame must be a non-negative integer")
    start_frame = reviewed_shot["timeline"]["start_frame"]
    end_frame = reviewed_shot["timeline"]["end_frame"]
    _require(start_frame <= frame < end_frame, f"keeper {shot_id} review timecode falls outside the shot")
    _require(relative_frame == frame - start_frame, f"keeper {shot_id} review relative_frame mismatch")
    _require(
        timecode["value"] == frame_timecode(frame, direction["project"]["fps"]),
        f"keeper {shot_id} review timecode is not canonical",
    )

    _require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()),
             f"keeper {shot_id} review.reviewer must be a non-empty string")
    _require(isinstance(review.get("summary"), str), f"keeper {shot_id} review.summary must be a string")
    created_at = review.get("created_at")
    _require(isinstance(created_at, str) and bool(created_at),
             f"keeper {shot_id} review.created_at must be an ISO-8601 timestamp")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        raise VideoFinishError(
            f"keeper {shot_id} review.created_at must be an ISO-8601 timestamp"
        ) from None
    _require(parsed_created_at.tzinfo is not None,
             f"keeper {shot_id} review.created_at must include a timezone")

    clip = _object(review.get("clip"), f"keeper {shot_id} review.clip")
    _exact_keys(clip, required={"path", "sha256", "bytes"}, optional=set(),
                label=f"keeper {shot_id} review.clip")
    clip_path = _local_path(clip["path"], base=review_path.parent, label=f"keeper {shot_id} review.clip.path")
    _require(clip_path == media, f"keeper {shot_id} review clip path does not match keeper")
    _require(clip.get("sha256") == media_hash, f"keeper {shot_id} review clip hash does not match keeper")
    _require(clip.get("bytes") == media.stat().st_size,
             f"keeper {shot_id} review clip byte count does not match keeper")
    verdicts = _object(review.get("verdicts"), f"keeper {shot_id} review.verdicts")
    _require(set(verdicts) == set(REVIEW_DIMENSIONS),
             f"keeper {shot_id} review must contain exactly all review dimensions")
    for dimension in REVIEW_DIMENSIONS:
        verdict = _object(verdicts[dimension], f"keeper {shot_id} review.verdicts.{dimension}")
        _exact_keys(
            verdict,
            required={"verdict", "note"},
            optional=set(),
            label=f"keeper {shot_id} review.verdicts.{dimension}",
        )
        _require(verdict.get("verdict") == "pass",
                 f"keeper {shot_id} review dimension {dimension} is not pass")
        _require(isinstance(verdict.get("note"), str),
                 f"keeper {shot_id} review dimension {dimension} note must be a string")
    return {"path": str(review_path), "sha256": declared_hash, "schema": REVIEW_SCHEMA}


def validate_finish_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, payload = _load_object(path, "finish manifest")
    manifest = copy.deepcopy(payload)
    _exact_keys(
        manifest,
        required={
            "schema_version", "project_id", "finish_id", "driving_model", "output_root", "target", "keepers",
            "adapters",
        },
        optional=set(),
        label="finish manifest",
    )
    _require(manifest["schema_version"] == FINISH_SCHEMA, f"schema_version must be {FINISH_SCHEMA!r}")
    _identifier(manifest["project_id"], "project_id")
    _identifier(manifest["finish_id"], "finish_id")
    _identifier(manifest["driving_model"], "driving_model")
    target = _object(manifest["target"], "target")
    _require(target == TARGET, f"target must be exactly {TARGET}")
    output_root = _local_path(manifest["output_root"], base=source.parent, label="output_root")
    manifest["output_root"] = str(output_root)

    keepers = _array(manifest["keepers"], "keepers")
    _require(bool(keepers), "keepers must not be empty")
    shot_ids: set[str] = set()
    normalized_keepers: list[dict[str, Any]] = []
    for index, raw in enumerate(keepers):
        keeper = copy.deepcopy(_object(raw, f"keepers[{index}]"))
        _exact_keys(keeper, required={"shot_id", "path", "sha256", "keeper", "review"}, optional=set(),
                    label=f"keepers[{index}]")
        shot_id = _identifier(keeper["shot_id"], f"keepers[{index}].shot_id")
        _require(shot_id not in shot_ids, f"duplicate keeper shot_id: {shot_id}")
        shot_ids.add(shot_id)
        _require(keeper["keeper"] is True, f"shot {shot_id} is not marked keeper")
        media = _local_path(keeper["path"], base=source.parent, label=f"keeper {shot_id}.path")
        _require(media.is_file(), f"keeper source not found: {media}")
        declared_hash = keeper["sha256"]
        _require(isinstance(declared_hash, str) and bool(HEX64.fullmatch(declared_hash)),
                 f"keeper {shot_id}.sha256 must be 64 lowercase hex characters")
        _require(sha256_file(media) == declared_hash, f"keeper {shot_id} hash mismatch")
        review_ref = _object(keeper["review"], f"keepers[{index}].review")
        _exact_keys(review_ref, required={"path", "sha256"}, optional=set(), label=f"keepers[{index}].review")
        review_path = _local_path(review_ref["path"], base=source.parent, label=f"keeper {shot_id}.review.path")
        keeper["review"] = _validate_review_record(
            review_path=review_path,
            declared_hash=review_ref["sha256"],
            media=media,
            media_hash=declared_hash,
            project_id=manifest["project_id"],
            shot_id=shot_id,
        )
        keeper["path"] = str(media)
        normalized_keepers.append(keeper)
    manifest["keepers"] = normalized_keepers

    adapters_raw = _object(manifest["adapters"], "adapters")
    _require(set(adapters_raw) <= {"rife", "seedvr2"}, "adapters may contain only rife and seedvr2")
    normalized_adapters: dict[str, dict[str, Any]] = {}
    for name in ("rife", "seedvr2"):
        if name in adapters_raw:
            normalized_adapters[name] = _validate_adapter(name, adapters_raw[name], base=source.parent)
    manifest["adapters"] = normalized_adapters
    return source, manifest


def _command_probe(argv: Sequence[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, shell=False)
    except (FileNotFoundError, PermissionError) as exc:
        return {"status": "failed", "argv": list(argv), "returncode": None, "output": str(exc)}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "argv": list(argv), "returncode": None, "output": "timeout"}
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()[-4000:]
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "argv": list(argv),
        "returncode": result.returncode,
        "output": combined,
    }


def _version_tuple(output: str) -> tuple[int, int, int] | None:
    match = VERSION_RE.search(output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _verify_models(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(str(entry["path"]))
        exists = path.is_file()
        observed_bytes = path.stat().st_size if exists else None
        observed_hash = sha256_file(path) if exists and observed_bytes == entry["expected_bytes"] else None
        ok = (
            exists
            and observed_bytes == entry["expected_bytes"]
            and observed_hash == entry["expected_sha256"]
        )
        checked.append({
            **dict(entry),
            "exists": exists,
            "observed_bytes": observed_bytes,
            "observed_sha256": observed_hash,
            "status": "passed" if ok else "failed",
        })
    return checked


def _canary(
    *, name: str, adapter: Mapping[str, Any], ffmpeg: str, ffprobe: str, target: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix=f"loramake-{name}-canary-") as temp_text:
            temp = Path(temp_text)
            source = temp / "input.mkv"
            output = temp / f"output{adapter['output_extension']}"
            generated = _command_probe([
                ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=2:duration=1",
                "-frames:v", "2", "-an", "-map_metadata", "-1", "-c:v", "ffv1", str(source),
            ], timeout=30)
            if generated["status"] != "passed" or not source.is_file():
                return {"status": "failed", "phase": "fixture", "result": generated}
            argv = _format_adapter_argv(
                adapter, input_path=source, output_path=output, shot_id="canary", target=target
            )
            executed = _command_probe(argv, timeout=120)
            if executed["status"] != "passed" or not output.is_file() or output.stat().st_size <= 0:
                return {"status": "failed", "phase": "adapter", "result": executed}
            source_probe = probe_media(source, ffprobe=ffprobe)
            output_probe = probe_media(output, ffprobe=ffprobe)
            source_video = _video_stream(source_probe)
            output_video = _video_stream(output_probe)
            if name == "rife":
                checks = {
                    "fps_2x": abs(float(output_video.get("fps") or 0) - 2 * float(source_video.get("fps") or 0)) < 0.05,
                    "dimensions_preserved": (
                        output_video.get("width") == source_video.get("width")
                        and output_video.get("height") == source_video.get("height")
                    ),
                }
            else:
                checks = {
                    "dimensions_2x": (
                        output_video.get("width") == 2 * int(source_video.get("width") or 0)
                        and output_video.get("height") == 2 * int(source_video.get("height") or 0)
                    ),
                    "fps_preserved": abs(float(output_video.get("fps") or 0) - float(source_video.get("fps") or 0)) < 0.05,
                }
            return {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source": source_probe,
                "output": output_probe,
                "output_sha256": sha256_file(output),
                "output_bytes": output.stat().st_size,
            }
    except (VideoFinishError, VideoEditError, OSError, subprocess.SubprocessError) as exc:
        return {"status": "failed", "phase": "verification", "error": str(exc)}


def _readiness(
    manifest: Mapping[str, Any], *, ffmpeg: str, ffprobe: str, verify: bool
) -> dict[str, Any]:
    adapters = manifest["adapters"]
    readiness: dict[str, Any] = {}
    for name in ("rife", "seedvr2"):
        if name in adapters:
            readiness[name] = copy.deepcopy(adapters[name]["readiness"])
        else:
            readiness[name] = {
                "configured": False,
                "executable": None,
                "executable_found": False,
                "check_paths": [],
                "version_probe": {"status": "not_run", "argv": []},
                "compatible": False,
                "model_files": [],
                "canary": {"status": "not_run"},
                "verified": False,
                "ready": False,
            }
    for tool_name, token in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        executable, found = _resolve_executable(token, base=Path.cwd())
        readiness[tool_name] = {
            "configured": True,
            "executable": executable,
            "executable_found": found,
            "version_probe": {"status": "not_run", "argv": [executable, "-version"]},
            "verified": False,
            "ready": False,
        }
    if not verify:
        return readiness

    for tool_name in ("ffmpeg", "ffprobe"):
        item = readiness[tool_name]
        item["version_probe"] = _command_probe(item["version_probe"]["argv"])
        item["verified"] = True
        item["ready"] = item["executable_found"] and item["version_probe"]["status"] == "passed"
    tools_ready = readiness["ffmpeg"]["ready"] and readiness["ffprobe"]["ready"]
    for name in ("rife", "seedvr2"):
        item = readiness[name]
        if not item["configured"]:
            continue
        item["version_probe"] = _command_probe(item["version_probe"]["argv"])
        parsed = _version_tuple(str(item["version_probe"].get("output") or ""))
        if parsed is not None:
            item["version_probe"]["parsed_version"] = ".".join(str(part) for part in parsed)
        item["compatible"] = parsed is not None and (name != "rife" or parsed >= (4, 7, 0))
        if name == "seedvr2":
            item["model_files"] = _verify_models(item["model_files"])
        models_ready = name != "seedvr2" or (
            bool(item["model_files"]) and all(model["status"] == "passed" for model in item["model_files"])
        )
        prerequisites = (
            item["executable_found"]
            and all(check["exists"] for check in item["check_paths"])
            and item["version_probe"]["status"] == "passed"
            and item["compatible"]
            and models_ready
            and tools_ready
        )
        if prerequisites:
            item["canary"] = _canary(
                name=name,
                adapter=manifest["adapters"][name],
                ffmpeg=readiness["ffmpeg"]["executable"],
                ffprobe=readiness["ffprobe"]["executable"],
                target=manifest["target"],
            )
        item["verified"] = True
        item["ready"] = prerequisites and item["canary"]["status"] == "passed"
    return readiness


def _format_adapter_argv(
    adapter: Mapping[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    shot_id: str,
    target: Mapping[str, Any],
) -> list[str]:
    values = {
        "input": str(input_path),
        "output": str(output_path),
        "multiplier": str(target["rife_multiplier"]),
        "scale": str(target["seedvr2_scale"]),
        "shot_id": shot_id,
        "target_width": str(target["width"]),
        "target_height": str(target["height"]),
        "target_fps": str(target["fps"]),
    }
    try:
        return [token.format_map(values) for token in adapter["argv"]]
    except (KeyError, ValueError) as exc:
        raise VideoFinishError(f"failed to format adapter argv: {exc}") from exc


def _video_stream(observed: Mapping[str, Any]) -> Mapping[str, Any]:
    for stream in observed.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise VideoFinishError("keeper/adaptor output has no video stream")


def _fmt(value: float | int) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"


def _conform_argv(
    *,
    ffmpeg: str,
    input_path: Path,
    work_output: Path,
    duration_s: float,
    frames: int,
    target: Mapping[str, Any],
) -> list[str]:
    width = int(target["width"])
    height = int(target["height"])
    fps = int(target["fps"])
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,fps={fps}:round=near,format=yuv422p10le"
    )
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(input_path),
        "-vf", vf, "-frames:v", str(frames), "-fps_mode", "cfr", "-r", str(fps),
        "-t", _fmt(duration_s), "-an", "-map_metadata", "-1", "-map_chapters", "-1",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        "-metadata", f"comment={FINISH_SCHEMA}",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le", str(work_output),
    ]


def build_finish_plan(
    manifest_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    verify_readiness: bool = False,
) -> dict[str, Any]:
    source, manifest = validate_finish_manifest(manifest_path)
    target = manifest["target"]
    readiness = _readiness(manifest, ffmpeg=ffmpeg, ffprobe=ffprobe, verify=verify_readiness)
    ffmpeg_executable = readiness["ffmpeg"]["executable"]
    output_root = Path(manifest["output_root"])
    shots: list[dict[str, Any]] = []

    for keeper in manifest["keepers"]:
        shot_id = keeper["shot_id"]
        source_media = Path(keeper["path"])
        observed = probe_media(source_media, ffprobe=ffprobe)
        video = _video_stream(observed)
        source_duration = float(video.get("duration_s") or observed.get("duration_s") or 0.0)
        _require(source_duration > 0 and float(video.get("fps") or 0) > 0,
                 f"keeper {shot_id} has invalid duration/fps")
        target_frames = max(1, round(source_duration * int(target["fps"])))
        duration_s = target_frames / int(target["fps"])
        shot_root = output_root / shot_id

        rife_extension = manifest["adapters"].get("rife", {}).get("output_extension", ".mkv")
        seed_extension = manifest["adapters"].get("seedvr2", {}).get("output_extension", ".mkv")
        rife_output = shot_root / f"01_rife2x{rife_extension}"
        rife_work = shot_root / f".01_rife2x.part{rife_extension}"
        seed_output = shot_root / f"02_seedvr2_2x{seed_extension}"
        seed_work = shot_root / f".02_seedvr2_2x.part{seed_extension}"
        final_output = shot_root / "03_1080x1920_30fps_prores_lt.mov"
        final_work = shot_root / ".03_1080x1920_30fps_prores_lt.part.mov"

        stages: list[dict[str, Any]] = []
        rife_adapter = manifest["adapters"].get("rife")
        stages.append({
            "id": "rife2x",
            "kind": "adapter",
            "adapter": "rife",
            "input": str(source_media),
            "work_output": str(rife_work),
            "output": str(rife_output),
            "argv": _format_adapter_argv(
                rife_adapter,
                input_path=source_media,
                output_path=rife_work,
                shot_id=shot_id,
                target=target,
            ) if rife_adapter else None,
            "expected": {"fps_multiplier": 2},
        })
        seed_adapter = manifest["adapters"].get("seedvr2")
        stages.append({
            "id": "seedvr2_2x",
            "kind": "adapter",
            "adapter": "seedvr2",
            "input": str(rife_output),
            "work_output": str(seed_work),
            "output": str(seed_output),
            "argv": _format_adapter_argv(
                seed_adapter,
                input_path=rife_output,
                output_path=seed_work,
                shot_id=shot_id,
                target=target,
            ) if seed_adapter else None,
            "expected": {"scale": 2},
        })
        stages.append({
            "id": "conform_9x16_1080x1920_30fps",
            "kind": "ffmpeg",
            "adapter": None,
            "input": str(seed_output),
            "work_output": str(final_work),
            "output": str(final_output),
            "argv": _conform_argv(
                ffmpeg=ffmpeg_executable,
                input_path=seed_output,
                work_output=final_work,
                duration_s=duration_s,
                frames=target_frames,
                target=target,
            ),
            "expected": {
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "frames": target_frames,
                "codec": "prores",
                "profile": "LT",
                "pix_fmt": "yuv422p10le",
                "audio": False,
            },
        })
        shots.append({
            "shot_id": shot_id,
            "source": {
                "path": str(source_media),
                "sha256": keeper["sha256"],
                "bytes": source_media.stat().st_size,
                "keeper": True,
                "review": copy.deepcopy(keeper["review"]),
            },
            "source_ffprobe": observed,
            "source_duration_s": source_duration,
            "duration_s": duration_s,
            "target_frames": target_frames,
            "stages": stages,
        })

    verification_completed = verify_readiness and all(item["verified"] for item in readiness.values())
    runnable = verification_completed and all(item["ready"] for item in readiness.values())
    return {
        "schema_version": FINISH_PLAN_SCHEMA,
        "project_id": manifest["project_id"],
        "finish_id": manifest["finish_id"],
        "driving_model": manifest["driving_model"],
        "execution_policy": EXECUTION_POLICY,
        "source_manifest": {"path": str(source), "sha256": sha256_file(source)},
        "target": copy.deepcopy(target),
        "readiness_verification": {
            "requested": verify_readiness,
            "completed": verification_completed,
            "method": READINESS_METHOD,
        },
        "readiness": readiness,
        "runnable": runnable,
        "shots": shots,
        "report_path": str(output_root / "finish_report.json"),
    }


def write_json_no_clobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError as exc:
        raise VideoFinishError(f"no-clobber: file already exists: {target}") from exc
    return target


def _run_argv(argv: Sequence[str], *, timeout: float, label: str) -> subprocess.CompletedProcess[str]:
    _require(bool(argv) and all(isinstance(token, str) and token for token in argv), f"{label}: invalid argv")
    _require(not any(token.startswith(("http://", "https://")) for token in argv), f"{label}: local argv only")
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, shell=False)
    except FileNotFoundError as exc:
        raise VideoFinishError(f"{label}: executable not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoFinishError(f"{label}: timed out after {timeout:g}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise VideoFinishError(f"{label}: exit {result.returncode}: {detail}")
    return result


def _publish_work(work: Path, output: Path) -> None:
    _require(work.is_file() and work.stat().st_size > 0, f"stage produced no file: {work}")
    try:
        os.link(work, output)
    except FileExistsError as exc:
        raise VideoFinishError(f"no-clobber: stage output already exists: {output}") from exc
    except OSError as exc:
        raise VideoFinishError(f"failed to publish stage output {output}: {exc}") from exc
    work.unlink()


def _verify_stage(
    stage: Mapping[str, Any],
    *,
    observed: Mapping[str, Any],
    prior_observed: Mapping[str, Any],
) -> dict[str, Any]:
    video = _video_stream(observed)
    prior_video = _video_stream(prior_observed)
    checks: dict[str, bool] = {"video_stream": True}
    if stage["id"] == "rife2x":
        checks["fps_2x"] = abs(float(video.get("fps") or 0) - 2 * float(prior_video.get("fps") or 0)) < 0.05
        checks["dimensions_preserved"] = (
            video.get("width") == prior_video.get("width") and video.get("height") == prior_video.get("height")
        )
    elif stage["id"] == "seedvr2_2x":
        checks["dimensions_2x"] = (
            video.get("width") == 2 * int(prior_video.get("width") or 0)
            and video.get("height") == 2 * int(prior_video.get("height") or 0)
        )
        checks["fps_preserved"] = abs(float(video.get("fps") or 0) - float(prior_video.get("fps") or 0)) < 0.05
    else:
        expected = stage["expected"]
        observed_frames = int(video.get("nb_frames") or 0)
        if observed_frames <= 0:
            observed_frames = round(float(observed.get("duration_s") or 0.0) * float(video.get("fps") or 0.0))
        checks.update({
            "width": video.get("width") == expected["width"],
            "height": video.get("height") == expected["height"],
            "fps": abs(float(video.get("fps") or 0) - expected["fps"]) < 0.01,
            "codec": video.get("codec_name") == expected["codec"],
            "profile": str(video.get("profile") or "").upper() == expected["profile"],
            "pixel_format": video.get("pix_fmt") == expected["pix_fmt"],
            "frames": observed_frames == expected["frames"],
            "no_audio": not any(stream.get("codec_type") == "audio" for stream in observed.get("streams", [])),
        })
    _require(all(checks.values()), f"stage {stage['id']} verification failed: {checks}")
    return {"ok": True, "checks": checks}


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    _exact_keys(
        plan,
        required={
            "schema_version", "project_id", "finish_id", "driving_model", "execution_policy", "source_manifest",
            "target", "readiness_verification", "readiness", "runnable", "shots", "report_path",
        },
        optional=set(),
        label="finish plan",
    )
    _require(plan["schema_version"] == FINISH_PLAN_SCHEMA, f"plan schema must be {FINISH_PLAN_SCHEMA!r}")
    _require(plan["execution_policy"] == EXECUTION_POLICY, f"plan execution_policy must be {EXECUTION_POLICY!r}")
    _require(plan["target"] == TARGET, f"plan target must be exactly {TARGET}")
    _require(isinstance(plan["shots"], list) and bool(plan["shots"]), "finish plan must contain shots")
    verification = _object(plan["readiness_verification"], "readiness_verification")
    _exact_keys(verification, required={"requested", "completed", "method"}, optional=set(),
                label="readiness_verification")
    _require(verification["method"] == READINESS_METHOD,
             f"readiness_verification.method must be {READINESS_METHOD!r}")


def _execution_basis(plan: Mapping[str, Any]) -> dict[str, Any]:
    basis = copy.deepcopy(dict(plan))
    for field in ("readiness", "readiness_verification", "runnable"):
        basis.pop(field, None)
    return basis


def run_finish_plan(
    plan_path: str | Path,
    *,
    ffmpeg: str | None = None,
    ffprobe: str = "ffprobe",
    timeout: float = 3600.0,
) -> dict[str, Any]:
    source, plan = _load_object(plan_path, "finish plan")
    _validate_plan_shape(plan)
    source_manifest = _object(plan["source_manifest"], "source_manifest")
    manifest_path = Path(source_manifest.get("path", "")).expanduser().resolve()
    _require(manifest_path.is_file(), f"source finish manifest not found: {manifest_path}")
    _require(sha256_file(manifest_path) == source_manifest.get("sha256"), "source finish manifest hash changed")

    ffmpeg_for_rebuild = ffmpeg or str(plan["readiness"]["ffmpeg"]["executable"])
    verification = plan["readiness_verification"]
    _require(
        verification["requested"] is True and verification["completed"] is True and plan["runnable"] is True,
        "finish plan has no completed runnable readiness verification",
    )
    rebuilt = build_finish_plan(
        manifest_path,
        ffmpeg=ffmpeg_for_rebuild,
        ffprobe=ffprobe,
        verify_readiness=True,
    )
    _require(_execution_basis(rebuilt) == _execution_basis(plan),
             "finish plan does not match the unchanged explicit adapter manifest")
    _require(rebuilt["runnable"] is True,
             f"current local RIFE/SeedVR2 readiness verification failed: {rebuilt['readiness']}")

    report_path = Path(plan["report_path"])
    all_paths: list[Path] = [report_path]
    for shot in plan["shots"]:
        for stage in shot["stages"]:
            all_paths.extend([Path(stage["work_output"]), Path(stage["output"])])
    existing = [str(path) for path in all_paths if path.exists()]
    _require(not existing, f"no-clobber preflight found existing finish outputs: {existing}")
    for shot in plan["shots"]:
        for stage in shot["stages"]:
            Path(stage["output"]).parent.mkdir(parents=True, exist_ok=True)

    shot_reports: list[dict[str, Any]] = []
    for shot in plan["shots"]:
        previous_observed = shot["source_ffprobe"]
        stage_reports: list[dict[str, Any]] = []
        for stage in shot["stages"]:
            argv = stage["argv"]
            _require(isinstance(argv, list), f"stage {stage['id']} has no configured argv")
            work = Path(stage["work_output"])
            output = Path(stage["output"])
            try:
                _run_argv(argv, timeout=timeout, label=f"{shot['shot_id']}:{stage['id']}")
                observed = probe_media(work, ffprobe=ffprobe)
                verification = _verify_stage(stage, observed=observed, prior_observed=previous_observed)
                _publish_work(work, output)
            except (VideoFinishError, VideoEditError):
                work.unlink(missing_ok=True)
                raise
            published_observed = probe_media(output, ffprobe=ffprobe)
            stage_reports.append({
                "id": stage["id"],
                "argv": argv,
                "output": str(output),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "ffprobe": published_observed,
                "verification": verification,
            })
            previous_observed = published_observed
        shot_reports.append({"shot_id": shot["shot_id"], "stages": stage_reports})

    report = {
        "schema_version": FINISH_REPORT_SCHEMA,
        "project_id": plan["project_id"],
        "finish_id": plan["finish_id"],
        "driving_model": plan["driving_model"],
        "execution_policy": EXECUTION_POLICY,
        "plan_path": str(source),
        "plan_sha256": sha256_file(source),
        "target": copy.deepcopy(plan["target"]),
        "shots": shot_reports,
        "status": "complete",
    }
    write_json_no_clobber(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keeper-only RIFE/SeedVR2 per-shot finishing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write exact keeper-only plan; never invoke adapters")
    plan.add_argument("manifest", help="loramake-video-finish-v1 JSON")
    plan.add_argument("--out", required=True, help="no-clobber finish plan JSON path")
    plan.add_argument("--ffmpeg", default="ffmpeg", help="local ffmpeg executable")
    plan.add_argument("--ffprobe", default="ffprobe", help="local ffprobe executable")
    plan.add_argument(
        "--verify",
        action="store_true",
        help="run explicit local version/model probes and tiny synthetic adapter canaries",
    )
    plan.add_argument("--json", action="store_true", help="print the plan JSON")
    run = subparsers.add_parser("run", help="execute the unchanged plan through explicit local adapters")
    run.add_argument("plan", help="loramake-video-finish-plan-v1 JSON")
    run.add_argument("--ffmpeg", help="override local ffmpeg path; plan is re-derived and must still match")
    run.add_argument("--ffprobe", default="ffprobe", help="local ffprobe executable")
    run.add_argument("--timeout", type=float, default=3600.0, help="per-stage timeout seconds")
    run.add_argument("--json", action="store_true", help="print the report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_finish_plan(
                args.manifest,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                verify_readiness=args.verify,
            )
            plan_path = write_json_no_clobber(args.out, plan)
            if args.json:
                print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            else:
                print(plan_path)
                print("READY" if plan["runnable"] else "NOT_READY")
            return 0 if plan["runnable"] else 3
        report = run_finish_plan(args.plan, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, timeout=args.timeout)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(Path(report["plan_path"]))
            print("complete")
        return 0
    except (VideoFinishError, VideoEditError) as exc:
        print(f"video_finish: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
