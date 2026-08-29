"""Offline contracts for deterministic video direction and human review records.

The module is deliberately standard-library-only and side-effect free at import time.
It normalizes a compact JSON brief into a frame-addressed direction contract; execution,
media inspection, networking, and GPU ownership stay in their existing adapters.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shlex
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from loramake.video_spec import WAN22_I2V

DIRECTION_SCHEMA = "loramake-video-direction-v1"
REVIEW_SCHEMA = "loramake-video-review-v1"
STORYBOARD_SCHEMA = "loramake-video-storyboard-v1"

REVIEW_DIMENSIONS = (
    "motion",
    "camera",
    "beat",
    "identity",
    "continuity",
    "flicker",
    "line_boiling",
    "palette",
)
REVIEW_VERDICTS = ("pass", "partial", "fail", "uncertain", "not_applicable")
IMPLEMENTATIONS = ("wan_i2v", "still_motion", "title_card", "external")
STILL_MOTION_PRESETS = ("hold", "slow_push_in", "impact_jolt", "status_reveal")

_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_RES_RE = re.compile(r"^(\d+)x(\d+)$")
_TIMECODE_RE = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$")


class VideoDirectionError(ValueError):
    """Raised when a brief or review record violates the public contract."""


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VideoDirectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: str | Path, *, label: str = "JSON") -> tuple[Path, dict[str, Any]]:
    """Read one UTF-8 JSON object while rejecting duplicate object keys."""
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except OSError as exc:
        raise VideoDirectionError(f"could not read {label}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise VideoDirectionError(f"invalid {label}: {exc}") from None
    if not isinstance(value, dict):
        raise VideoDirectionError(f"{label} root must be a JSON object")
    return source, value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VideoDirectionError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "an"
        raise VideoDirectionError(f"{label} must be {qualifier} JSON array")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "" if allow_empty else " non-empty"
        raise VideoDirectionError(f"{label} must be a{suffix} string")
    return value.strip() if not allow_empty else value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label)
    if not _ID_RE.fullmatch(text):
        raise VideoDirectionError(f"{label} must match {_ID_RE.pattern!r}")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise VideoDirectionError(f"{label} must be an integer {bound}")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise VideoDirectionError(f"{label} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise VideoDirectionError(f"{label} must be a finite number") from None
    if not result.is_finite():
        raise VideoDirectionError(f"{label} must be a finite number")
    return result


def _bounded_float(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    number = _decimal(value, label)
    if number < Decimal(str(minimum)) or number > Decimal(str(maximum)):
        raise VideoDirectionError(f"{label} must be within {minimum:g}..{maximum:g}")
    return float(number)


def _seconds_to_frame(value: Any, fps: int, label: str) -> int:
    seconds = _decimal(value, label)
    if seconds < 0:
        raise VideoDirectionError(f"{label} must be >= 0")
    frames = seconds * fps
    integral = frames.to_integral_value(rounding=ROUND_HALF_UP)
    if frames != integral:
        raise VideoDirectionError(f"{label} must land on an exact frame at {fps} fps")
    return int(integral)


def frame_timecode(frame: int, fps: int) -> str:
    """Render a frame index as a stable millisecond timecode."""
    _integer(frame, "frame")
    _integer(fps, "fps", minimum=1)
    millis = int((Decimal(frame) * 1000 / fps).to_integral_value(rounding=ROUND_HALF_UP))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        match = _RES_RE.fullmatch(value.strip().lower())
        if not match:
            raise VideoDirectionError("project.resolution must be WIDTHxHEIGHT")
        width, height = int(match.group(1)), int(match.group(2))
    elif isinstance(value, dict):
        width = _integer(value.get("width"), "project.resolution.width", minimum=1)
        height = _integer(value.get("height"), "project.resolution.height", minimum=1)
    else:
        raise VideoDirectionError("project.resolution must be WIDTHxHEIGHT or an object")
    if width > 16_384 or height > 16_384:
        raise VideoDirectionError("project.resolution exceeds 16384 pixels on one edge")
    return width, height


def _ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _asset_path(value: Any, label: str, base_dir: Path) -> str | None:
    text = _optional_text(value, label)
    if text is None:
        return None
    path = Path(text).expanduser()
    return str((base_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    rows = _array(value, label, nonempty=nonempty)
    return [_text(row, f"{label}[{index}]") for index, row in enumerate(rows)]


def _default_wan_length(duration_frames: int, timeline_fps: int) -> int:
    seconds = Decimal(duration_frames) / timeline_fps
    raw = max(5, int((seconds * WAN22_I2V.fps).to_integral_value(rounding=ROUND_CEILING)))
    length = raw + ((1 - raw) % 4)
    if length > WAN22_I2V.max_length:
        raise VideoDirectionError(
            f"Wan shot duration needs {length} source frames; split the shot or set generation.length <= 81"
        )
    return length


def _normalize_still_motion(value: Any, *, implementation: str, shot_id: str) -> dict[str, Any] | None:
    label = f"shot {shot_id}.generation.still_motion"
    if implementation != "still_motion":
        if value is not None:
            raise VideoDirectionError(f"{label} is only valid for still_motion shots")
        return None
    data = _mapping(value, label)
    unknown = sorted(set(data) - {"preset", "strength", "region", "color"})
    if unknown:
        raise VideoDirectionError(f"{label} has unsupported fields: {', '.join(unknown)}")
    preset = _text(data.get("preset"), f"{label}.preset")
    if preset not in STILL_MOTION_PRESETS:
        raise VideoDirectionError(
            f"{label}.preset must be one of {', '.join(STILL_MOTION_PRESETS)}"
        )
    default_strength = {"hold": 0.0, "slow_push_in": 0.045, "impact_jolt": 0.012,
                        "status_reveal": 0.85}[preset]
    strength = _bounded_float(
        data.get("strength", default_strength), f"{label}.strength", minimum=0.0, maximum=1.0
    )
    region_value = data.get("region")
    region: dict[str, float] | None = None
    if region_value is not None:
        region_data = _mapping(region_value, f"{label}.region")
        if set(region_data) != {"x", "y", "width", "height"}:
            raise VideoDirectionError(f"{label}.region must contain x, y, width and height")
        region = {
            key: _bounded_float(region_data[key], f"{label}.region.{key}", minimum=0.0, maximum=1.0)
            for key in ("x", "y", "width", "height")
        }
        if region["width"] <= 0 or region["height"] <= 0:
            raise VideoDirectionError(f"{label}.region width and height must be positive")
        if region["x"] + region["width"] > 1 or region["y"] + region["height"] > 1:
            raise VideoDirectionError(f"{label}.region must stay inside the frame")
    color_value = data.get("color")
    color = None if color_value is None else _text(color_value, f"{label}.color")
    if color is not None and not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        raise VideoDirectionError(f"{label}.color must be #RRGGBB")
    if preset == "status_reveal" and (region is None or color is None or strength <= 0):
        raise VideoDirectionError(
            f"{label} status_reveal requires a region, #RRGGBB color and positive strength"
        )
    if preset != "status_reveal" and (region is not None or color is not None):
        raise VideoDirectionError(f"{label} region/color are reserved for status_reveal")
    if preset == "hold" and strength != 0:
        raise VideoDirectionError(f"{label}.strength must be 0 for hold")
    if preset in {"slow_push_in", "impact_jolt"} and strength <= 0:
        raise VideoDirectionError(f"{label}.strength must be positive for {preset}")
    return {"preset": preset, "strength": strength, "region": region, "color": color}


def _normalize_generation(
    raw: Any,
    *,
    implementation: str,
    duration_frames: int,
    timeline_fps: int,
    base_dir: Path,
    project_id: str,
    shot_id: str,
    default_res: str,
) -> dict[str, Any]:
    data = _mapping(raw if raw is not None else {}, f"shot {shot_id}.generation")
    seed_image = _asset_path(data.get("seed_image"), f"shot {shot_id}.generation.seed_image", base_dir)
    identity_ref = _asset_path(data.get("identity_ref"), f"shot {shot_id}.generation.identity_ref", base_dir)
    prompt = _text(data.get("prompt", ""), f"shot {shot_id}.generation.prompt", allow_empty=True)
    seed = data.get("seed")
    res = data.get("res")
    length = data.get("length")
    pos_suffix = data.get("pos_suffix")
    negative = data.get("negative")
    anime_lora = data.get("anime_lora")
    still_motion = _normalize_still_motion(
        data.get("still_motion"), implementation=implementation, shot_id=shot_id
    )

    if implementation == "wan_i2v":
        if seed_image is None:
            raise VideoDirectionError(f"Wan shot {shot_id} requires generation.seed_image")
        if not prompt.strip():
            raise VideoDirectionError(f"Wan shot {shot_id} requires generation.prompt")
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(f"{project_id}:{shot_id}".encode()).digest()[:4], "big")
        seed = _integer(seed, f"shot {shot_id}.generation.seed", maximum=2**63 - 1)
        length = _default_wan_length(duration_frames, timeline_fps) if length is None else _integer(
            length, f"shot {shot_id}.generation.length", minimum=5, maximum=WAN22_I2V.max_length
        )
        resolved_res = res or default_res
        try:
            width, height = WAN22_I2V.validate_shape(length, resolved_res)
        except ValueError as exc:
            raise VideoDirectionError(f"shot {shot_id}.generation: {exc}") from None
        res = f"{width}x{height}"
        pos_suffix = WAN22_I2V.pos_suffix if pos_suffix is None else _text(
            pos_suffix, f"shot {shot_id}.generation.pos_suffix", allow_empty=True
        )
        negative = WAN22_I2V.negative if negative is None else _text(
            negative, f"shot {shot_id}.generation.negative", allow_empty=True
        )
        anime_lora = _optional_text(anime_lora, f"shot {shot_id}.generation.anime_lora")
    else:
        if seed is not None:
            seed = _integer(seed, f"shot {shot_id}.generation.seed", maximum=2**63 - 1)
        if res is not None:
            match = _RES_RE.fullmatch(_text(res, f"shot {shot_id}.generation.res").lower())
            if not match:
                raise VideoDirectionError(f"shot {shot_id}.generation.res must be WIDTHxHEIGHT")
            res = f"{int(match.group(1))}x{int(match.group(2))}"
        if length is not None:
            length = _integer(length, f"shot {shot_id}.generation.length", minimum=1)
        pos_suffix = _optional_text(pos_suffix, f"shot {shot_id}.generation.pos_suffix")
        negative = _optional_text(negative, f"shot {shot_id}.generation.negative")
        anime_lora = _optional_text(anime_lora, f"shot {shot_id}.generation.anime_lora")

    return {
        "seed_image": seed_image,
        "identity_ref": identity_ref,
        "seed": seed,
        "prompt": prompt,
        "res": res,
        "length": length,
        "source_fps": WAN22_I2V.fps if implementation == "wan_i2v" else None,
        "pos_suffix": pos_suffix,
        "negative": negative,
        "anime_lora": anime_lora,
        "still_motion": still_motion,
    }


def _normalize_sound_cues(
    value: Any, *, shot_id: str, start_frame: int, end_frame: int, fps: int
) -> list[dict[str, Any]]:
    rows = _array(value if value is not None else [], f"shot {shot_id}.sound_cues")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        cue = _mapping(raw, f"shot {shot_id}.sound_cues[{index}]")
        frame = _seconds_to_frame(cue.get("at"), fps, f"shot {shot_id}.sound_cues[{index}].at")
        if frame < start_frame or frame >= end_frame:
            raise VideoDirectionError(f"sound cue {index} for shot {shot_id} falls outside the shot")
        result.append(
            {
                "frame": frame,
                "timecode": frame_timecode(frame, fps),
                "kind": _text(cue.get("kind"), f"shot {shot_id}.sound_cues[{index}].kind"),
                "description": _text(
                    cue.get("description"), f"shot {shot_id}.sound_cues[{index}].description"
                ),
            }
        )
    return result


def normalize_direction(brief: Any, *, base_dir: str | Path = ".") -> dict[str, Any]:
    """Normalize a compact JSON brief into ``loramake-video-direction-v1``.

    Times in the brief are seconds and must resolve to exact project frames. Relative
    asset paths are anchored to ``base_dir``; the function performs no existence check,
    network call, subprocess invocation, or GPU import.
    """
    root = _mapping(brief, "brief")
    project_in = _mapping(root.get("project"), "project")
    project_id = _identifier(project_in.get("id"), "project.id")
    title = _text(project_in.get("title"), "project.title")
    description = _text(project_in.get("description", ""), "project.description", allow_empty=True)
    driver = _identifier(project_in.get("driver", "codex"), "project.driver")
    asset_namespace = _identifier(project_in.get("asset_namespace", project_id), "project.asset_namespace")
    width, height = _resolution(project_in.get("resolution"))
    fps = _integer(project_in.get("fps"), "project.fps", minimum=1, maximum=120)
    aspect_ratio = _ratio(width, height)
    declared_ratio = project_in.get("aspect_ratio")
    if declared_ratio is not None and _text(declared_ratio, "project.aspect_ratio") != aspect_ratio:
        raise VideoDirectionError(
            f"project.aspect_ratio must match resolution ({aspect_ratio}, got {declared_ratio!r})"
        )

    base = Path(base_dir).expanduser().resolve()
    scenes_in = _array(root.get("scenes"), "scenes", nonempty=True)
    scenes: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    shot_ids: set[str] = set()
    timeline_rows: list[tuple[int, int, str]] = []
    max_end = 0

    for scene_index, scene_raw in enumerate(scenes_in, 1):
        scene_in = _mapping(scene_raw, f"scenes[{scene_index - 1}]")
        scene_id = _identifier(scene_in.get("id"), f"scenes[{scene_index - 1}].id")
        if scene_id in scene_ids:
            raise VideoDirectionError(f"duplicate scene id: {scene_id}")
        scene_ids.add(scene_id)
        shots_in = _array(scene_in.get("shots"), f"scene {scene_id}.shots", nonempty=True)
        shots: list[dict[str, Any]] = []
        for shot_index, shot_raw in enumerate(shots_in, 1):
            shot_in = _mapping(shot_raw, f"scene {scene_id}.shots[{shot_index - 1}]")
            shot_id = _identifier(shot_in.get("id"), f"scene {scene_id}.shots[{shot_index - 1}].id")
            if shot_id in shot_ids:
                raise VideoDirectionError(f"duplicate shot id: {shot_id}")
            shot_ids.add(shot_id)
            start_frame = _seconds_to_frame(
                shot_in.get("start", shot_in.get("start_seconds")), fps, f"shot {shot_id}.start"
            )
            end_frame = _seconds_to_frame(
                shot_in.get("end", shot_in.get("end_seconds")), fps, f"shot {shot_id}.end"
            )
            if end_frame <= start_frame:
                raise VideoDirectionError(f"shot {shot_id}.end must be after start")
            implementation = _text(shot_in.get("implementation"), f"shot {shot_id}.implementation")
            if implementation not in IMPLEMENTATIONS:
                raise VideoDirectionError(
                    f"shot {shot_id}.implementation must be one of {', '.join(IMPLEMENTATIONS)}"
                )
            duration_frames = end_frame - start_frame
            generation = _normalize_generation(
                shot_in.get("generation"),
                implementation=implementation,
                duration_frames=duration_frames,
                timeline_fps=fps,
                base_dir=base,
                project_id=project_id,
                shot_id=shot_id,
                default_res="480x832" if height > width else "832x480",
            )
            shots.append(
                {
                    "id": shot_id,
                    "order": shot_index,
                    "title": _text(shot_in.get("title"), f"shot {shot_id}.title"),
                    "timeline": {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "duration_frames": duration_frames,
                        "start_timecode": frame_timecode(start_frame, fps),
                        "end_timecode": frame_timecode(end_frame, fps),
                    },
                    "shot_size": _text(shot_in.get("shot_size"), f"shot {shot_id}.shot_size"),
                    "visual": _text(shot_in.get("visual"), f"shot {shot_id}.visual"),
                    "implementation": implementation,
                    "start_state": _text(shot_in.get("start_state"), f"shot {shot_id}.start_state"),
                    "end_state": _text(shot_in.get("end_state"), f"shot {shot_id}.end_state"),
                    "action": _text(shot_in.get("action"), f"shot {shot_id}.action"),
                    "camera": _text(shot_in.get("camera"), f"shot {shot_id}.camera"),
                    "screen_direction": _text(
                        shot_in.get("screen_direction"), f"shot {shot_id}.screen_direction"
                    ),
                    "continuity_locks": _string_list(
                        shot_in.get("continuity_locks"), f"shot {shot_id}.continuity_locks", nonempty=True
                    ),
                    "generation": generation,
                    "sound_cues": _normalize_sound_cues(
                        shot_in.get("sound_cues"),
                        shot_id=shot_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        fps=fps,
                    ),
                    "acceptance": _string_list(
                        shot_in.get("acceptance"), f"shot {shot_id}.acceptance", nonempty=True
                    ),
                }
            )
            timeline_rows.append((start_frame, end_frame, shot_id))
            max_end = max(max_end, end_frame)
        scenes.append(
            {
                "id": scene_id,
                "order": scene_index,
                "title": _text(scene_in.get("title"), f"scene {scene_id}.title"),
                "shots": shots,
            }
        )

    prior_end = -1
    prior_id = ""
    for start_frame, end_frame, shot_id in sorted(timeline_rows):
        if start_frame < prior_end:
            raise VideoDirectionError(f"shot {shot_id} overlaps shot {prior_id}")
        prior_end, prior_id = end_frame, shot_id

    duration_value = project_in.get("duration_seconds")
    duration_frames = max_end if duration_value is None else _seconds_to_frame(
        duration_value, fps, "project.duration_seconds"
    )
    if duration_frames < max_end:
        raise VideoDirectionError("project duration ends before the last shot")

    result = {
        "schema": DIRECTION_SCHEMA,
        "project": {
            "id": project_id,
            "title": title,
            "description": description,
            "driver": driver,
            "asset_namespace": asset_namespace,
            "aspect_ratio": aspect_ratio,
            "resolution": {"width": width, "height": height},
            "fps": fps,
            "duration_frames": duration_frames,
            "duration_timecode": frame_timecode(duration_frames, fps),
        },
        "scenes": scenes,
    }
    validate_direction(result)
    return result


def iter_shots(direction: dict[str, Any]):
    for scene in direction["scenes"]:
        for shot in scene["shots"]:
            yield scene, shot


def json_bytes(value: Any) -> bytes:
    """Canonical repository JSON rendering used for hashes and publication."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _decimal_seconds(frames: int, fps: int) -> str:
    value = Decimal(frames) / fps
    rendered = format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")
    return rendered.rstrip("0").rstrip(".") or "0"


def _fmt_filter(value: float) -> str:
    rendered = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _wan_argv(shot: dict[str, Any], output_dir: Path, repo_root: Path) -> list[str]:
    generation = shot["generation"]
    argv = [
        "python3",
        str((repo_root / "scripts" / "z3_video.py").resolve()),
        "--image",
        generation["seed_image"],
        "--prompt",
        generation["prompt"],
        "--length",
        str(generation["length"]),
        "--out",
        str(output_dir),
        "--seed",
        str(generation["seed"]),
        "--res",
        generation["res"],
        "--pos-suffix",
        generation["pos_suffix"],
        "--neg",
        generation["negative"],
    ]
    if generation["anime_lora"]:
        argv.extend(["--anime-lora", generation["anime_lora"]])
    return argv


def _still_argv(
    shot: dict[str, Any], output_dir: Path, project: dict[str, Any]
) -> list[str] | None:
    source = shot["generation"]["seed_image"]
    if source is None:
        return None
    width = project["resolution"]["width"]
    height = project["resolution"]["height"]
    fps = project["fps"]
    frames = shot["timeline"]["duration_frames"]
    duration = _decimal_seconds(frames, fps)
    output = output_dir / f"{shot['id']}_source.mp4"
    motion = shot["generation"]["still_motion"]
    if not isinstance(motion, dict):
        raise VideoDirectionError(f"still_motion shot {shot['id']} has no motion contract")
    preset = motion["preset"]
    strength = float(motion["strength"])
    base = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    if preset == "slow_push_in":
        denominator = max(1, frames - 1)
        video_filter = (
            f"{base},zoompan=z='1+{_fmt_filter(strength)}*on/{denominator}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps}"
        )
    elif preset == "impact_jolt":
        margin = max(4, round(min(width, height) * strength))
        padded_width = width + 2 * margin
        padded_height = height + 2 * margin
        video_filter = (
            f"scale={padded_width}:{padded_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={padded_width}:{padded_height},"
            f"crop={width}:{height}:"
            f"x='{margin}+if(eq(n,1),{margin},if(eq(n,2),-{margin},if(eq(n,3),{margin // 2},0)))':"
            f"y='{margin}+if(eq(n,0),-{margin // 2},if(eq(n,1),{margin},"
            f"if(eq(n,2),-{margin},if(eq(n,3),{margin // 2},0))))',fps={fps}"
        )
    elif preset == "status_reveal":
        region = motion["region"]
        if not isinstance(region, dict):
            raise VideoDirectionError(f"still_motion shot {shot['id']} status_reveal lacks region")
        x = round(width * float(region["x"]))
        y = round(height * float(region["y"]))
        box_width = max(1, round(width * float(region["width"])))
        box_height = max(8, round(height * float(region["height"])))
        color = str(motion["color"]).removeprefix("#")
        segments = 8
        filters = [base, f"fps={fps}"]
        for segment in range(segments):
            # Reveal bottom-to-top over the middle 60% of the shot, then hold.
            bottom_index = segments - 1 - segment
            segment_y = y + math.floor(box_height * bottom_index / segments)
            segment_bottom = y + math.ceil(box_height * (bottom_index + 1) / segments)
            segment_h = max(1, segment_bottom - segment_y)
            reveal_time = float(duration) * (0.2 + 0.6 * segment / (segments - 1))
            filters.append(
                f"drawbox=x={x}:y={segment_y}:w={box_width}:h={segment_h}:"
                f"color=0x{color}@{_fmt_filter(strength)}:t=fill:"
                f"enable='gte(t,{_fmt_filter(reveal_time)})'"
            )
        video_filter = ",".join(filters)
    elif preset == "hold":
        video_filter = f"{base},fps={fps}"
    else:  # normalized directions never reach this branch
        raise VideoDirectionError(f"unsupported still motion preset: {preset}")
    video_filter += ",setsar=1,format=yuv420p"
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-loop",
        "1",
        "-i",
        source,
        "-t",
        duration,
        "-frames:v",
        str(frames),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def build_storyboard_manifest(
    direction: dict[str, Any],
    *,
    direction_path: str | Path,
    direction_sha256: str,
    repo_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile direction shots into exact argv without executing any command."""
    validate_direction(direction)
    if not re.fullmatch(r"[a-f0-9]{64}", direction_sha256):
        raise VideoDirectionError("direction_sha256 must be a lowercase SHA-256")
    project = direction["project"]
    root = Path(repo_root).resolve()
    clip_root = (
        Path(output_root).resolve()
        / project["asset_namespace"]
        / "by_model"
        / project["driver"]
        / "video"
        / project["id"]
        / "shots"
    )
    shots: list[dict[str, Any]] = []
    global_order = 0
    for scene, shot in iter_shots(direction):
        global_order += 1
        output_dir = clip_root / shot["id"]
        seed_image = shot["generation"]["seed_image"]
        identity_ref = shot["generation"]["identity_ref"]
        seed_available = bool(seed_image and Path(seed_image).is_file())
        identity_available = identity_ref is None or Path(identity_ref).is_file()
        if shot["implementation"] == "wan_i2v":
            argv = _wan_argv(shot, output_dir, root)
            state = "ready" if seed_available and identity_available else "needs_source_asset"
            touches_gpu = True
        elif shot["implementation"] == "still_motion":
            argv = _still_argv(shot, output_dir, project)
            state = "ready" if argv and seed_available else "needs_source_asset"
            touches_gpu = False
        else:
            argv = None
            state = "deferred_to_edit"
            touches_gpu = False
        env = {"DRIVING_MODEL": project["driver"]}
        prepare_argv = ["mkdir", "-p", str(output_dir)] if argv else None
        shell = None
        if argv:
            prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
            shell = f"{prefix} {shlex.join(argv)}"
        shots.append(
            {
                "scene_id": scene["id"],
                "shot_id": shot["id"],
                "order": global_order,
                "implementation": shot["implementation"],
                "timeline": shot["timeline"],
                "seed_image": seed_image,
                "identity_ref": identity_ref,
                "asset_state": (
                    "available" if seed_available and identity_available
                    else ("not_applicable" if seed_image is None else "planned")
                ),
                "output_dir": str(output_dir),
                "env": env,
                "prepare_argv": prepare_argv,
                "argv": argv,
                "shell": shell,
                "touches_gpu_if_executed": touches_gpu,
                "command_state": state,
            }
        )
    return {
        "schema": STORYBOARD_SCHEMA,
        "project_id": project["id"],
        "direction": {"path": str(Path(direction_path).resolve()), "sha256": direction_sha256},
        "clip_root": str(clip_root),
        "shots": shots,
    }


def render_shot_list(direction: dict[str, Any]) -> str:
    """Render a compact human review surface from the normalized contract."""
    validate_direction(direction)
    project = direction["project"]
    lines = [
        f"# {project['title']} — Shot list",
        "",
        f"- Project: `{project['id']}`",
        f"- Canvas: `{project['resolution']['width']}x{project['resolution']['height']}` "
        f"(`{project['aspect_ratio']}`, {project['fps']} fps)",
        f"- Duration: `{project['duration_timecode']}` / `{project['duration_frames']}` frames",
        f"- Driver: `{project['driver']}`",
        "",
        "| # | Scene / Shot | Time | Size | Implementation | Visual |",
        "|---:|---|---|---|---|---|",
    ]
    count = 0
    for scene, shot in iter_shots(direction):
        count += 1
        timeline = shot["timeline"]
        visual = shot["visual"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {count} | `{scene['id']}/{shot['id']}` | "
            f"{timeline['start_timecode']}–{timeline['end_timecode']} | "
            f"{shot['shot_size']} | {shot['implementation']} | {visual} |"
        )
    for scene, shot in iter_shots(direction):
        lines.extend(
            [
                "",
                f"## {scene['id']} / {shot['id']} — {shot['title']}",
                "",
                f"- Start state: {shot['start_state']}",
                f"- End state: {shot['end_state']}",
                f"- Action: {shot['action']}",
                f"- Camera: {shot['camera']}",
                f"- Screen direction: {shot['screen_direction']}",
                f"- Continuity locks: {'; '.join(shot['continuity_locks'])}",
                f"- Acceptance: {'; '.join(shot['acceptance'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def render_commands(manifest: dict[str, Any], *, repo_root: str | Path) -> str:
    """Render the manifest's exact argv as a reviewable POSIX shell script."""
    lines = ["#!/bin/sh", "set -eu", f"cd {shlex.quote(str(Path(repo_root).resolve()))}", ""]
    for shot in manifest["shots"]:
        lines.append(
            f"# {shot['scene_id']}/{shot['shot_id']} [{shot['implementation']}] {shot['command_state']}"
        )
        if shot["shell"] and shot["command_state"] == "ready":
            lines.append(shlex.join(shot["prepare_argv"]))
            lines.append(shot["shell"])
        elif shot["shell"]:
            lines.append(f"# blocked exact command: {shot['shell']}")
        else:
            lines.append(f"# command deferred: {shot['command_state']}")
        lines.append("")
    return "\n".join(lines)


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise VideoDirectionError(
            f"{label} fields do not match the contract; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def validate_direction(direction: Any) -> None:
    """Perform dependency-free semantic validation for a normalized direction."""
    data = _mapping(direction, "direction")
    _exact_keys(data, {"schema", "project", "scenes"}, "direction")
    if data.get("schema") != DIRECTION_SCHEMA:
        raise VideoDirectionError(f"direction.schema must be {DIRECTION_SCHEMA}")
    project = _mapping(data.get("project"), "direction.project")
    _exact_keys(
        project,
        {
            "id",
            "title",
            "description",
            "driver",
            "asset_namespace",
            "aspect_ratio",
            "resolution",
            "fps",
            "duration_frames",
            "duration_timecode",
        },
        "direction.project",
    )
    _identifier(project.get("id"), "direction.project.id")
    _text(project.get("title"), "direction.project.title")
    _text(project.get("description"), "direction.project.description", allow_empty=True)
    _identifier(project.get("driver"), "direction.project.driver")
    _identifier(project.get("asset_namespace"), "direction.project.asset_namespace")
    resolution = _mapping(project.get("resolution"), "direction.project.resolution")
    _exact_keys(resolution, {"width", "height"}, "direction.project.resolution")
    width = _integer(resolution.get("width"), "direction.project.resolution.width", minimum=1, maximum=16_384)
    height = _integer(
        resolution.get("height"), "direction.project.resolution.height", minimum=1, maximum=16_384
    )
    if project.get("aspect_ratio") != _ratio(width, height):
        raise VideoDirectionError("direction.project.aspect_ratio does not match resolution")
    fps = _integer(project.get("fps"), "direction.project.fps", minimum=1, maximum=120)
    duration = _integer(project.get("duration_frames"), "direction.project.duration_frames", minimum=1)
    if project.get("duration_timecode") != frame_timecode(duration, fps):
        raise VideoDirectionError("direction.project.duration_timecode is not canonical")

    known_scenes: set[str] = set()
    known_shots: set[str] = set()
    timeline_rows: list[tuple[int, int, str]] = []
    for scene_index, scene in enumerate(_array(data.get("scenes"), "direction.scenes", nonempty=True), 1):
        scene_data = _mapping(scene, "direction.scene")
        _exact_keys(scene_data, {"id", "order", "title", "shots"}, "direction.scene")
        scene_id = _identifier(scene_data.get("id"), "direction.scene.id")
        if scene_id in known_scenes:
            raise VideoDirectionError(f"duplicate scene id: {scene_id}")
        known_scenes.add(scene_id)
        if scene_data.get("order") != scene_index:
            raise VideoDirectionError(f"scene {scene_id}.order must be {scene_index}")
        _text(scene_data.get("title"), f"scene {scene_id}.title")
        for shot_index, shot in enumerate(
            _array(scene_data.get("shots"), "direction.scene.shots", nonempty=True), 1
        ):
            shot_data = _mapping(shot, "direction.shot")
            _exact_keys(
                shot_data,
                {
                    "id",
                    "order",
                    "title",
                    "timeline",
                    "shot_size",
                    "visual",
                    "implementation",
                    "start_state",
                    "end_state",
                    "action",
                    "camera",
                    "screen_direction",
                    "continuity_locks",
                    "generation",
                    "sound_cues",
                    "acceptance",
                },
                "direction.shot",
            )
            shot_id = _identifier(shot_data.get("id"), "direction.shot.id")
            if shot_id in known_shots:
                raise VideoDirectionError(f"duplicate shot id: {shot_id}")
            known_shots.add(shot_id)
            if shot_data.get("order") != shot_index:
                raise VideoDirectionError(f"shot {shot_id}.order must be {shot_index}")
            for key in (
                "title",
                "shot_size",
                "visual",
                "start_state",
                "end_state",
                "action",
                "camera",
                "screen_direction",
            ):
                _text(shot_data.get(key), f"shot {shot_id}.{key}")
            implementation = _text(shot_data.get("implementation"), f"shot {shot_id}.implementation")
            if implementation not in IMPLEMENTATIONS:
                raise VideoDirectionError(f"shot {shot_id}.implementation is unsupported")
            locks = _string_list(
                shot_data.get("continuity_locks"), f"shot {shot_id}.continuity_locks", nonempty=True
            )
            if len(set(locks)) != len(locks):
                raise VideoDirectionError(f"shot {shot_id}.continuity_locks must be unique")
            acceptance = _string_list(
                shot_data.get("acceptance"), f"shot {shot_id}.acceptance", nonempty=True
            )
            if len(set(acceptance)) != len(acceptance):
                raise VideoDirectionError(f"shot {shot_id}.acceptance must be unique")
            timeline = _mapping(shot_data.get("timeline"), f"shot {shot_id}.timeline")
            _exact_keys(
                timeline,
                {"start_frame", "end_frame", "duration_frames", "start_timecode", "end_timecode"},
                f"shot {shot_id}.timeline",
            )
            start = _integer(timeline.get("start_frame"), f"shot {shot_id}.start_frame")
            end = _integer(timeline.get("end_frame"), f"shot {shot_id}.end_frame", minimum=1)
            if end <= start or end > duration:
                raise VideoDirectionError(f"shot {shot_id} has an invalid timeline")
            if timeline.get("duration_frames") != end - start:
                raise VideoDirectionError(f"shot {shot_id} duration does not match its endpoints")
            if timeline.get("start_timecode") != frame_timecode(start, fps):
                raise VideoDirectionError(f"shot {shot_id} start_timecode is not canonical")
            if timeline.get("end_timecode") != frame_timecode(end, fps):
                raise VideoDirectionError(f"shot {shot_id} end_timecode is not canonical")

            generation = _mapping(shot_data.get("generation"), f"shot {shot_id}.generation")
            _exact_keys(
                generation,
                {
                    "seed_image",
                    "identity_ref",
                    "seed",
                    "prompt",
                    "res",
                    "length",
                    "source_fps",
                    "pos_suffix",
                    "negative",
                    "anime_lora",
                    "still_motion",
                },
                f"shot {shot_id}.generation",
            )
            for key in ("seed_image", "identity_ref", "anime_lora"):
                _optional_text(generation.get(key), f"shot {shot_id}.generation.{key}")
            _text(generation.get("prompt"), f"shot {shot_id}.generation.prompt", allow_empty=True)
            seed = generation.get("seed")
            if seed is not None:
                _integer(seed, f"shot {shot_id}.generation.seed", maximum=2**63 - 1)
            source_fps = generation.get("source_fps")
            if source_fps is not None:
                _integer(source_fps, f"shot {shot_id}.generation.source_fps", minimum=1)
            for key in ("pos_suffix", "negative"):
                value = generation.get(key)
                if value is not None:
                    _text(value, f"shot {shot_id}.generation.{key}", allow_empty=True)
            res = generation.get("res")
            length = generation.get("length")
            still_motion = generation.get("still_motion")
            if implementation == "wan_i2v":
                if not generation.get("seed_image") or seed is None or not generation.get("prompt"):
                    raise VideoDirectionError(f"Wan shot {shot_id} is missing seed_image, seed, or prompt")
                if source_fps != WAN22_I2V.fps:
                    raise VideoDirectionError(f"Wan shot {shot_id}.source_fps must be {WAN22_I2V.fps}")
                if generation.get("pos_suffix") is None or generation.get("negative") is None:
                    raise VideoDirectionError(f"Wan shot {shot_id} must freeze pos_suffix and negative")
                try:
                    WAN22_I2V.validate_shape(length, res)
                except (TypeError, ValueError) as exc:
                    raise VideoDirectionError(f"Wan shot {shot_id} has an invalid shape: {exc}") from None
                if still_motion is not None:
                    raise VideoDirectionError(f"Wan shot {shot_id} must not define still_motion")
            elif implementation == "still_motion":
                if not generation.get("seed_image"):
                    raise VideoDirectionError(f"still_motion shot {shot_id} requires seed_image")
                # Reuse the brief normalizer as the single semantic validator for the
                # normalized preset surface, then require canonical equality.
                normalized_motion = _normalize_still_motion(
                    still_motion, implementation=implementation, shot_id=shot_id
                )
                if normalized_motion != still_motion:
                    raise VideoDirectionError(f"still_motion shot {shot_id} is not canonical")
            else:
                if still_motion is not None:
                    raise VideoDirectionError(f"shot {shot_id} must not define still_motion")
                if length is not None:
                    _integer(length, f"shot {shot_id}.generation.length", minimum=1)
                if res is not None and not _RES_RE.fullmatch(_text(res, f"shot {shot_id}.generation.res")):
                    raise VideoDirectionError(f"shot {shot_id}.generation.res must be WIDTHxHEIGHT")

            cues = _array(shot_data.get("sound_cues"), f"shot {shot_id}.sound_cues")
            for cue_index, cue in enumerate(cues):
                cue_data = _mapping(cue, f"shot {shot_id}.sound_cues[{cue_index}]")
                _exact_keys(
                    cue_data,
                    {"frame", "timecode", "kind", "description"},
                    f"shot {shot_id}.sound_cues[{cue_index}]",
                )
                cue_frame = _integer(cue_data.get("frame"), f"shot {shot_id}.sound_cues[{cue_index}].frame")
                if cue_frame < start or cue_frame >= end:
                    raise VideoDirectionError(f"shot {shot_id}.sound_cues[{cue_index}] is outside the shot")
                if cue_data.get("timecode") != frame_timecode(cue_frame, fps):
                    raise VideoDirectionError(f"shot {shot_id}.sound_cues[{cue_index}] timecode is not canonical")
                _text(cue_data.get("kind"), f"shot {shot_id}.sound_cues[{cue_index}].kind")
                _text(cue_data.get("description"), f"shot {shot_id}.sound_cues[{cue_index}].description")
            timeline_rows.append((start, end, shot_id))

    prior_end = -1
    prior_id = ""
    for start, end, shot_id in sorted(timeline_rows):
        if start < prior_end:
            raise VideoDirectionError(f"shot {shot_id} overlaps shot {prior_id}")
        prior_end, prior_id = end, shot_id


def parse_timecode(value: str, fps: int) -> int:
    """Parse HH:MM:SS.mmm or decimal seconds and resolve to the nearest named frame."""
    text = _text(value, "timecode")
    match = _TIMECODE_RE.fullmatch(text)
    if match:
        hours, minutes, seconds = (int(match.group(index)) for index in (1, 2, 3))
        if minutes >= 60 or seconds >= 60:
            raise VideoDirectionError("timecode minutes and seconds must be below 60")
        fraction = (match.group(4) or "").ljust(3, "0")
        total = Decimal(hours * 3600 + minutes * 60 + seconds) + Decimal(int(fraction or "0")) / 1000
    else:
        total = _decimal(text, "timecode")
    if total < 0:
        raise VideoDirectionError("timecode must be >= 0")
    exact = total * fps
    frame = int(exact.to_integral_value(rounding=ROUND_HALF_UP))
    # Millisecond timecodes cannot name every 24/30 fps frame exactly. Accept at most
    # half of one displayed millisecond of rounding error.
    tolerance = Decimal(fps) / 2000
    if abs(exact - frame) > tolerance:
        raise VideoDirectionError(f"timecode {value!r} does not identify a frame at {fps} fps")
    return frame


def build_review_record(
    direction: dict[str, Any],
    *,
    direction_path: str | Path,
    shot_id: str,
    timecode: str,
    clip_path: str | Path,
    verdicts: dict[str, Any],
    notes: dict[str, str] | None = None,
    reviewer: str = "human",
    summary: str = "",
    record_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable, frame-addressed event bound to the exact reviewed clip."""
    validate_direction(direction)
    if set(verdicts) != set(REVIEW_DIMENSIONS):
        missing = sorted(set(REVIEW_DIMENSIONS) - set(verdicts))
        extra = sorted(set(verdicts) - set(REVIEW_DIMENSIONS))
        raise VideoDirectionError(f"verdicts must contain exactly the review dimensions; missing={missing}, extra={extra}")
    note_map = {} if notes is None else _mapping(notes, "notes")
    extra_notes = set(note_map) - set(REVIEW_DIMENSIONS)
    if extra_notes:
        raise VideoDirectionError(f"unknown review note dimensions: {sorted(extra_notes)}")
    direction_file = Path(direction_path).expanduser().resolve()
    if not direction_file.is_file():
        raise VideoDirectionError(f"direction file not found: {direction_file}")
    _, bound_direction = load_json_object(direction_file, label="direction JSON")
    validate_direction(bound_direction)
    if bound_direction != direction:
        raise VideoDirectionError("direction object does not match direction_path")
    clip_file = Path(clip_path).expanduser().resolve()
    try:
        clip_stat = clip_file.stat()
    except OSError as exc:
        raise VideoDirectionError(f"review clip not found: {clip_file} ({type(exc).__name__})") from None
    if not clip_file.is_file() or clip_stat.st_size <= 0:
        raise VideoDirectionError(f"review clip must be a non-empty file: {clip_file}")

    target_scene = None
    target_shot = None
    for scene, shot in iter_shots(direction):
        if shot["id"] == shot_id:
            target_scene, target_shot = scene, shot
            break
    if target_shot is None:
        raise VideoDirectionError(f"unknown shot id: {shot_id}")
    fps = direction["project"]["fps"]
    frame = parse_timecode(timecode, fps)
    start_frame = target_shot["timeline"]["start_frame"]
    end_frame = target_shot["timeline"]["end_frame"]
    if frame < start_frame or frame >= end_frame:
        raise VideoDirectionError(
            f"timecode {timecode!r} (frame {frame}) is outside shot {shot_id} "
            f"[{start_frame}, {end_frame})"
        )

    resolved_id = record_id or f"review-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"
    resolved_id = _identifier(resolved_id, "record_id")
    actor = _text(reviewer, "reviewer")
    resolved_created_at = created_at or utc_now()
    try:
        parsed_created_at = datetime.fromisoformat(resolved_created_at.replace("Z", "+00:00"))
    except ValueError:
        raise VideoDirectionError("created_at must be an ISO-8601 timestamp") from None
    if parsed_created_at.tzinfo is None:
        raise VideoDirectionError("created_at must include a timezone")

    verdict_entries: dict[str, dict[str, str]] = {}
    for dimension in REVIEW_DIMENSIONS:
        verdict = _text(verdicts[dimension], f"verdicts.{dimension}")
        if verdict not in REVIEW_VERDICTS:
            raise VideoDirectionError(
                f"verdicts.{dimension} must be one of {', '.join(REVIEW_VERDICTS)}"
            )
        note = note_map.get(dimension, "")
        if not isinstance(note, str):
            raise VideoDirectionError(f"notes.{dimension} must be a string")
        verdict_entries[dimension] = {"verdict": verdict, "note": note}

    return {
        "schema": REVIEW_SCHEMA,
        "record_id": resolved_id,
        "project_id": direction["project"]["id"],
        "direction": {"path": str(direction_file), "sha256": sha256_file(direction_file)},
        "clip": {
            "path": str(clip_file),
            "sha256": sha256_file(clip_file),
            "bytes": clip_stat.st_size,
        },
        "scene_id": target_scene["id"],
        "shot_id": target_shot["id"],
        "timecode": {
            "value": frame_timecode(frame, fps),
            "frame": frame,
            "relative_frame": frame - start_frame,
        },
        "reviewer": actor,
        "verdicts": verdict_entries,
        "summary": _text(summary, "summary", allow_empty=True),
        "created_at": resolved_created_at,
    }


def atomic_publish_no_clobber(path: str | Path, raw: bytes) -> tuple[int, int]:
    """Publish one complete file atomically, failing if its final name already exists."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    temp = target.with_name(f".{target.name}.{token}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, target)
        stat = target.stat()
        return stat.st_dev, stat.st_ino
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def atomic_publish_bundle(files: dict[str | Path, bytes]) -> None:
    """Publish a small package as no-clobber files, cleaning only owned partials."""
    targets = [Path(path) for path in files]
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError(f"output already exists: {target}")
    owned: list[tuple[Path, tuple[int, int]]] = []
    try:
        for target, raw in ((Path(path), value) for path, value in files.items()):
            owned.append((target, atomic_publish_no_clobber(target, raw)))
    except Exception:
        for target, identity in reversed(owned):
            try:
                stat = target.stat()
                if (stat.st_dev, stat.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
