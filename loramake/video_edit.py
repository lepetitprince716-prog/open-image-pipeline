"""Deterministic, evidence-bearing FFmpeg editing for ``loramake-video-edit-v1``.

The module deliberately stays below the generation layer.  It consumes already-created
local media, verifies every declared SHA-256, builds one bounded video timeline, mixes
local audio, applies optional titles, and publishes a render plus an ffprobe evidence
sidecar without replacing an existing file.

Only a small v1 surface is accepted:

* one primary video track;
* hard cuts and crossfades of at most :data:`MAX_CROSSFADE_S`;
* any number of audio tracks mixed to 48 kHz stereo;
* local font assets for deterministic ``drawtext`` titles;
* ``work``, ``master`` and ``delivery`` encoder profiles.

No ComfyUI, GPU-generation graph, network client, or optional Python dependency is
imported here.  FFmpeg/ffprobe are the only runtime executables.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

EDIT_SCHEMA = "loramake-video-edit-v1"
EVIDENCE_SCHEMA = "loramake-video-edit-evidence-v1"
MAX_CROSSFADE_S = 0.5
DEFAULT_LOUDNESS = {"enabled": True, "integrated_lufs": -14.0, "true_peak_db": -1.0, "lra": 11.0}
LOUDNESS_TRUE_PEAK_HEADROOM_DB = 0.1
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class VideoEditError(RuntimeError):
    """A contract, tool, render, or publication error with a user-facing message."""


@dataclass(frozen=True, slots=True)
class Asset:
    id: str
    kind: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class EditContract:
    """Validated edit contract with local paths resolved against the spec directory."""

    raw: dict[str, Any]
    source_path: Path | None
    base_dir: Path
    assets: Mapping[str, Asset]
    width: int
    height: int
    fps: int
    duration_s: float
    total_frames: int
    profile: str
    output: Path

    @property
    def video_clips(self) -> list[dict[str, Any]]:
        return self.raw["tracks"]["video"][0]["clips"]

    @property
    def audio_clips(self) -> list[dict[str, Any]]:
        return [clip for track in self.raw["tracks"]["audio"] for clip in track["clips"]]


@dataclass(frozen=True, slots=True)
class RenderResult:
    output: Path
    evidence: Path
    observed: dict[str, Any]
    stream_contract: dict[str, Any]


PROFILE_ARGS: dict[str, tuple[str, ...]] = {
    "work": (
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
    ),
    "master": (
        "-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le",
        "-c:a", "pcm_s24le",
    ),
    "delivery": (
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
    ),
}
PROFILE_SUFFIX = {"work": ".mp4", "master": ".mov", "delivery": ".mp4"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VideoEditError(message)


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} must be a finite number")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be a finite number")
    if minimum is not None:
        _require(number >= minimum, f"{label} must be >= {minimum:g}")
    return number


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return int(value)


def _identifier(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(ID_RE.fullmatch(value)), f"{label} is not a safe identifier")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    _require(not missing, f"{label} missing required fields: {', '.join(missing)}")
    _require(not extra, f"{label} has unsupported fields: {', '.join(extra)}")


def _resolve_local_path(value: object, *, base_dir: Path, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty local path")
    _require(not value.startswith(("http://", "https://")), f"{label} must be a local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _aligned_duration(value: float, fps: int, label: str) -> int:
    frames = round(value * fps)
    _require(frames > 0 and abs(frames / fps - value) <= 1e-6,
             f"{label} must align to the {fps} fps timeline")
    return frames


def load_edit_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VideoEditError(f"edit spec not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise VideoEditError(f"edit spec is not valid JSON: {source}: {exc}") from exc
    _require(isinstance(payload, dict), "edit spec must be a JSON object")
    return payload


def validate_edit_spec(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    source_path: str | Path | None = None,
    verify_hashes: bool = True,
    profile_override: str | None = None,
    output_override: str | Path | None = None,
) -> EditContract:
    """Validate and resolve a ``loramake-video-edit-v1`` document.

    Validation is intentionally stricter than FFmpeg: durations are timeline-frame
    aligned, all source bytes are hash-bound, and a v1 edit has exactly one video
    track.  The returned ``raw`` document is a defensive copy.
    """

    _require(isinstance(payload, Mapping), "edit spec must be an object")
    raw: dict[str, Any] = copy.deepcopy(dict(payload))
    _exact_keys(
        raw,
        required={
            "schema_version", "project_id", "edit_id", "driving_model", "assets", "timeline",
            "tracks", "titles", "profile", "output", "audio_normalization",
        },
        optional=set(),
        label="edit spec",
    )
    _require(raw["schema_version"] == EDIT_SCHEMA, f"schema_version must be {EDIT_SCHEMA!r}")
    _identifier(raw["project_id"], "project_id")
    _identifier(raw["edit_id"], "edit_id")
    _identifier(raw["driving_model"], "driving_model")

    base = Path(base_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve() if source_path is not None else None

    assets_raw = _list(raw["assets"], "assets")
    _require(bool(assets_raw), "assets must contain at least one asset")
    assets: dict[str, Asset] = {}
    for index, item_raw in enumerate(assets_raw):
        item = _object(item_raw, f"assets[{index}]")
        _exact_keys(item, required={"id", "kind", "path", "sha256"}, optional=set(), label=f"assets[{index}]")
        asset_id = _identifier(item["id"], f"assets[{index}].id")
        _require(asset_id not in assets, f"duplicate asset id: {asset_id}")
        kind = item["kind"]
        _require(kind in {"video", "image", "audio", "font"}, f"asset {asset_id}: unsupported kind {kind!r}")
        path = _resolve_local_path(item["path"], base_dir=base, label=f"asset {asset_id}.path")
        _require(path.is_file(), f"asset {asset_id} not found or not a regular file: {path}")
        declared_hash = item["sha256"]
        _require(isinstance(declared_hash, str) and bool(HEX64.fullmatch(declared_hash)),
                 f"asset {asset_id}.sha256 must be 64 lowercase hex characters")
        if verify_hashes:
            observed_hash = sha256_file(path)
            _require(observed_hash == declared_hash, f"asset {asset_id} hash mismatch")
        assets[asset_id] = Asset(asset_id, kind, path, declared_hash)

    timeline = _object(raw["timeline"], "timeline")
    _exact_keys(timeline, required={"width", "height", "fps", "duration_s", "background"}, optional=set(),
                label="timeline")
    width = _integer(timeline["width"], "timeline.width", minimum=16)
    height = _integer(timeline["height"], "timeline.height", minimum=16)
    _require(width % 2 == 0 and height % 2 == 0, "timeline width and height must be even")
    fps = _integer(timeline["fps"], "timeline.fps", minimum=1)
    _require(fps <= 120, "timeline.fps must be <= 120")
    duration_s = _number(timeline["duration_s"], "timeline.duration_s", minimum=1 / fps)
    total_frames = _aligned_duration(duration_s, fps, "timeline.duration_s")
    _require(isinstance(timeline["background"], str) and bool(timeline["background"]),
             "timeline.background must be a color string")

    tracks = _object(raw["tracks"], "tracks")
    _exact_keys(tracks, required={"video", "audio"}, optional=set(), label="tracks")
    video_tracks = _list(tracks["video"], "tracks.video")
    audio_tracks = _list(tracks["audio"], "tracks.audio")
    _require(len(video_tracks) == 1, "v1 requires exactly one primary video track")
    video_track = _object(video_tracks[0], "tracks.video[0]")
    _exact_keys(video_track, required={"id", "clips"}, optional=set(), label="tracks.video[0]")
    _identifier(video_track["id"], "tracks.video[0].id")
    clips = _list(video_track["clips"], "tracks.video[0].clips")
    _require(bool(clips), "primary video track must contain clips")

    clip_ids: set[str] = set()
    summed_duration = 0.0
    transitions_total = 0.0
    clip_durations: list[float] = []
    for index, clip_raw in enumerate(clips):
        clip = _object(clip_raw, f"video clip {index}")
        _exact_keys(clip, required={"id", "asset_id", "in_s", "out_s", "transition_to_next"}, optional=set(),
                    label=f"video clip {index}")
        clip_id = _identifier(clip["id"], f"video clip {index}.id")
        _require(clip_id not in clip_ids, f"duplicate video clip id: {clip_id}")
        clip_ids.add(clip_id)
        asset_id = _identifier(clip["asset_id"], f"video clip {clip_id}.asset_id")
        _require(asset_id in assets and assets[asset_id].kind in {"video", "image"},
                 f"video clip {clip_id} must reference a video or image asset")
        in_s = _number(clip["in_s"], f"video clip {clip_id}.in_s", minimum=0.0)
        out_s = _number(clip["out_s"], f"video clip {clip_id}.out_s", minimum=0.0)
        _require(out_s > in_s, f"video clip {clip_id}.out_s must be greater than in_s")
        if assets[asset_id].kind == "image":
            _require(in_s == 0.0, f"image clip {clip_id}.in_s must be 0")
        clip_duration = out_s - in_s
        _aligned_duration(clip_duration, fps, f"video clip {clip_id} duration")
        clip_durations.append(clip_duration)
        summed_duration += clip_duration
        transition = _object(clip["transition_to_next"], f"video clip {clip_id}.transition_to_next")
        _exact_keys(transition, required={"type", "duration_s"}, optional=set(),
                    label=f"video clip {clip_id}.transition_to_next")
        transition_type = transition["type"]
        _require(transition_type in {"hard_cut", "crossfade"},
                 f"video clip {clip_id}: transition must be hard_cut or crossfade")
        transition_duration = _number(
            transition["duration_s"], f"video clip {clip_id}.transition_to_next.duration_s", minimum=0.0
        )
        if transition_type == "hard_cut":
            _require(transition_duration == 0.0, f"video clip {clip_id}: hard_cut duration must be 0")
        else:
            _require(index < len(clips) - 1, "the final clip cannot transition to another clip")
            _require(1 / fps <= transition_duration <= MAX_CROSSFADE_S,
                     f"video clip {clip_id}: crossfade must be 1 frame..{MAX_CROSSFADE_S:g}s")
            _aligned_duration(transition_duration, fps, f"video clip {clip_id} crossfade")
            transitions_total += transition_duration
        if index == len(clips) - 1:
            _require(transition_type == "hard_cut" and transition_duration == 0.0,
                     "the final video clip transition must be hard_cut with duration 0")

    for index, clip in enumerate(clips[:-1]):
        transition = clip["transition_to_next"]
        if transition["type"] == "crossfade":
            transition_duration = float(transition["duration_s"])
            _require(transition_duration < clip_durations[index] and transition_duration < clip_durations[index + 1],
                     f"video clip {clip['id']}: crossfade must be shorter than both adjacent clips")

    computed_duration = summed_duration - transitions_total
    _require(abs(computed_duration - duration_s) <= 1e-6,
             f"timeline.duration_s={duration_s:g} does not match clips/transitions ({computed_duration:g})")

    audio_track_ids: set[str] = set()
    audio_clip_ids: set[str] = set()
    for track_index, track_raw in enumerate(audio_tracks):
        track = _object(track_raw, f"tracks.audio[{track_index}]")
        _exact_keys(track, required={"id", "clips"}, optional=set(), label=f"tracks.audio[{track_index}]")
        track_id = _identifier(track["id"], f"tracks.audio[{track_index}].id")
        _require(track_id not in audio_track_ids, f"duplicate audio track id: {track_id}")
        audio_track_ids.add(track_id)
        for clip_index, clip_raw in enumerate(_list(track["clips"], f"tracks.audio[{track_index}].clips")):
            clip = _object(clip_raw, f"audio clip {track_index}:{clip_index}")
            _exact_keys(
                clip,
                required={"id", "asset_id", "in_s", "out_s", "start_s", "gain_db", "fade_in_s", "fade_out_s"},
                optional=set(),
                label=f"audio clip {track_index}:{clip_index}",
            )
            clip_id = _identifier(clip["id"], f"audio clip {track_index}:{clip_index}.id")
            _require(clip_id not in audio_clip_ids, f"duplicate audio clip id: {clip_id}")
            audio_clip_ids.add(clip_id)
            asset_id = _identifier(clip["asset_id"], f"audio clip {clip_id}.asset_id")
            _require(asset_id in assets and assets[asset_id].kind == "audio",
                     f"audio clip {clip_id} must reference an audio asset")
            in_s = _number(clip["in_s"], f"audio clip {clip_id}.in_s", minimum=0.0)
            out_s = _number(clip["out_s"], f"audio clip {clip_id}.out_s", minimum=0.0)
            start_s = _number(clip["start_s"], f"audio clip {clip_id}.start_s", minimum=0.0)
            gain_db = _number(clip["gain_db"], f"audio clip {clip_id}.gain_db")
            fade_in = _number(clip["fade_in_s"], f"audio clip {clip_id}.fade_in_s", minimum=0.0)
            fade_out = _number(clip["fade_out_s"], f"audio clip {clip_id}.fade_out_s", minimum=0.0)
            _require(out_s > in_s, f"audio clip {clip_id}.out_s must be greater than in_s")
            clip_duration = out_s - in_s
            _require(start_s < duration_s and start_s + clip_duration <= duration_s + 1e-6,
                     f"audio clip {clip_id} lies outside the edit timeline")
            _require(fade_in + fade_out <= clip_duration + 1e-6,
                     f"audio clip {clip_id} fades exceed its duration")
            _require(-96.0 <= gain_db <= 24.0, f"audio clip {clip_id}.gain_db must be -96..24")

    titles = _list(raw["titles"], "titles")
    title_ids: set[str] = set()
    for index, title_raw in enumerate(titles):
        title = _object(title_raw, f"titles[{index}]")
        _exact_keys(
            title,
            required={
                "id", "text", "font_asset_id", "start_s", "end_s", "font_size", "color", "align",
                "valign", "margin_x", "margin_y", "box", "box_color", "box_border_w",
            },
            optional=set(),
            label=f"titles[{index}]",
        )
        title_id = _identifier(title["id"], f"titles[{index}].id")
        _require(title_id not in title_ids, f"duplicate title id: {title_id}")
        title_ids.add(title_id)
        _require(isinstance(title["text"], str) and 0 < len(title["text"]) <= 500,
                 f"title {title_id}.text must contain 1..500 characters")
        font_id = _identifier(title["font_asset_id"], f"title {title_id}.font_asset_id")
        _require(font_id in assets and assets[font_id].kind == "font",
                 f"title {title_id} must reference a font asset")
        start_s = _number(title["start_s"], f"title {title_id}.start_s", minimum=0.0)
        end_s = _number(title["end_s"], f"title {title_id}.end_s", minimum=0.0)
        _require(0 <= start_s < end_s <= duration_s + 1e-6, f"title {title_id} lies outside the timeline")
        _aligned_duration(start_s if start_s else 1 / fps, fps, f"title {title_id}.start_s") if start_s else None
        _aligned_duration(end_s, fps, f"title {title_id}.end_s")
        _integer(title["font_size"], f"title {title_id}.font_size", minimum=6)
        _require(title["align"] in {"left", "center", "right"}, f"title {title_id}.align is invalid")
        _require(title["valign"] in {"top", "center", "bottom"}, f"title {title_id}.valign is invalid")
        _integer(title["margin_x"], f"title {title_id}.margin_x", minimum=0)
        _integer(title["margin_y"], f"title {title_id}.margin_y", minimum=0)
        _require(type(title["box"]) is bool, f"title {title_id}.box must be a boolean")
        _integer(title["box_border_w"], f"title {title_id}.box_border_w", minimum=0)
        _require(isinstance(title["color"], str) and bool(title["color"]), f"title {title_id}.color is invalid")
        _require(isinstance(title["box_color"], str) and bool(title["box_color"]),
                 f"title {title_id}.box_color is invalid")

    normalization = _object(raw["audio_normalization"], "audio_normalization")
    _exact_keys(normalization, required={"enabled", "integrated_lufs", "true_peak_db", "lra"}, optional=set(),
                label="audio_normalization")
    _require(type(normalization["enabled"]) is bool, "audio_normalization.enabled must be a boolean")
    integrated = _number(normalization["integrated_lufs"], "audio_normalization.integrated_lufs")
    true_peak = _number(normalization["true_peak_db"], "audio_normalization.true_peak_db")
    lra = _number(normalization["lra"], "audio_normalization.lra", minimum=1.0)
    _require(-70 <= integrated <= -5, "audio_normalization.integrated_lufs must be -70..-5")
    _require(-9 <= true_peak <= 0, "audio_normalization.true_peak_db must be -9..0")
    _require(lra <= 50, "audio_normalization.lra must be <= 50")

    profile = profile_override or raw["profile"]
    _require(profile in PROFILE_ARGS, f"profile must be one of {', '.join(PROFILE_ARGS)}")
    raw["profile"] = profile
    if output_override is not None:
        raw["output"] = str(output_override)
    output = _resolve_local_path(raw["output"], base_dir=base, label="output")
    _require(output.suffix.lower() == PROFILE_SUFFIX[profile],
             f"{profile} profile output must use {PROFILE_SUFFIX[profile]}")

    return EditContract(
        raw=raw,
        source_path=source,
        base_dir=base,
        assets=assets,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        total_frames=total_frames,
        profile=profile,
        output=output,
    )


def load_and_validate_edit(
    path: str | Path,
    *,
    profile_override: str | None = None,
    output_override: str | Path | None = None,
    verify_hashes: bool = True,
) -> EditContract:
    source = Path(path).expanduser().resolve()
    return validate_edit_spec(
        load_edit_spec(source),
        base_dir=source.parent,
        source_path=source,
        verify_hashes=verify_hashes,
        profile_override=profile_override,
        output_override=output_override,
    )


def _rate(value: object) -> float:
    numerator, _, denominator = str(value or "0/1").partition("/")
    try:
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_media(path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    media = Path(path)
    command = [
        ffprobe, "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,"
        "nb_frames,sample_rate,channels,channel_layout,duration:format=format_name,duration,size",
        "-of", "json", str(media),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise VideoEditError(f"ffprobe executable not found: {ffprobe}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoEditError(f"ffprobe timed out: {media}") from exc
    if result.returncode != 0:
        raise VideoEditError(f"ffprobe failed for {media}: {result.stderr.strip()[:500]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoEditError(f"ffprobe returned invalid JSON for {media}") from exc
    streams = payload.get("streams")
    fmt = payload.get("format")
    _require(isinstance(streams, list) and isinstance(fmt, dict), f"ffprobe returned no streams for {media}")
    normalized_streams: list[dict[str, Any]] = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        item = dict(stream)
        if item.get("codec_type") == "video":
            item["fps"] = _rate(item.get("avg_frame_rate") or item.get("r_frame_rate"))
            for key in ("width", "height", "nb_frames"):
                if key in item:
                    try:
                        item[key] = int(item[key])
                    except (TypeError, ValueError):
                        item[key] = 0
        elif item.get("codec_type") == "audio":
            for key in ("sample_rate", "channels"):
                if key in item:
                    try:
                        item[key] = int(item[key])
                    except (TypeError, ValueError):
                        item[key] = 0
        if "duration" in item:
            try:
                item["duration_s"] = float(item.pop("duration"))
            except (TypeError, ValueError):
                item["duration_s"] = 0.0
        normalized_streams.append(item)
    try:
        duration_s = float(fmt.get("duration") or 0.0)
        size = int(fmt.get("size") or media.stat().st_size)
    except (TypeError, ValueError, OSError):
        duration_s = 0.0
        size = media.stat().st_size if media.is_file() else 0
    return {
        "format_name": str(fmt.get("format_name") or ""),
        "duration_s": duration_s,
        "bytes": size,
        "streams": normalized_streams,
    }


def _source_duration(asset: Asset, *, ffprobe: str) -> float | None:
    if asset.kind not in {"video", "audio"}:
        return None
    observed = probe_media(asset.path, ffprobe=ffprobe)
    streams = [s for s in observed["streams"] if s.get("codec_type") == asset.kind]
    _require(bool(streams), f"asset {asset.id} contains no {asset.kind} stream")
    stream_duration = float(streams[0].get("duration_s") or 0.0)
    return stream_duration or float(observed.get("duration_s") or 0.0)


def verify_asset_streams(contract: EditContract, *, ffprobe: str = "ffprobe") -> dict[str, dict[str, Any]]:
    """Probe referenced media and prove every in/out range lies inside the source."""

    observations: dict[str, dict[str, Any]] = {}
    max_out: dict[str, float] = {}
    for clip in contract.video_clips:
        max_out[clip["asset_id"]] = max(max_out.get(clip["asset_id"], 0.0), float(clip["out_s"]))
    for clip in contract.audio_clips:
        max_out[clip["asset_id"]] = max(max_out.get(clip["asset_id"], 0.0), float(clip["out_s"]))
    for asset_id, requested_out in max_out.items():
        asset = contract.assets[asset_id]
        if asset.kind == "image":
            continue
        observations[asset_id] = probe_media(asset.path, ffprobe=ffprobe)
        duration = _source_duration(asset, ffprobe=ffprobe)
        _require(duration is not None and duration + 0.02 >= requested_out,
                 f"asset {asset_id} is shorter than requested out_s={requested_out:g}")
    return observations


def _fmt(value: float | int) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"


def _partition_hard_cut_groups(clips: Sequence[Mapping[str, Any]]) -> tuple[list[list[int]], list[float]]:
    groups: list[list[int]] = []
    crossfades: list[float] = []
    current: list[int] = []
    for index, clip in enumerate(clips):
        current.append(index)
        transition = clip["transition_to_next"]
        if transition["type"] == "crossfade":
            groups.append(current)
            current = []
            crossfades.append(float(transition["duration_s"]))
    if current:
        groups.append(current)
    return groups, crossfades


def _video_inputs_and_filters(
    contract: EditContract,
    *,
    title_images: Sequence[Path],
) -> tuple[list[str], list[str], str]:
    working_pix_fmt = "yuv422p10le" if contract.profile == "master" else "yuv420p"
    inputs: list[str] = []
    filters: list[str] = []
    durations: list[float] = []
    for index, clip in enumerate(contract.video_clips):
        asset = contract.assets[clip["asset_id"]]
        duration = float(clip["out_s"]) - float(clip["in_s"])
        durations.append(duration)
        if asset.kind == "image":
            inputs.extend(["-loop", "1", "-framerate", str(contract.fps), "-t", _fmt(duration), "-i", str(asset.path)])
            trim = f"trim=duration={_fmt(duration)}"
        else:
            inputs.extend(["-i", str(asset.path)])
            trim = f"trim=start={_fmt(clip['in_s'])}:end={_fmt(clip['out_s'])}"
        filters.append(
            f"[{index}:v:0]{trim},setpts=PTS-STARTPTS,"
            f"scale={contract.width}:{contract.height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={contract.width}:{contract.height},fps={contract.fps}:round=near,"
            f"format={working_pix_fmt},setsar=1,settb=AVTB"
            f"[vc{index}]"
        )

    groups, crossfades = _partition_hard_cut_groups(contract.video_clips)
    group_labels: list[str] = []
    group_durations: list[float] = []
    for group_index, group in enumerate(groups):
        labels = "".join(f"[vc{clip_index}]" for clip_index in group)
        label = f"vg{group_index}"
        if len(group) == 1:
            filters.append(f"{labels}null[{label}]")
        else:
            filters.append(f"{labels}concat=n={len(group)}:v=1:a=0[{label}]")
        group_labels.append(label)
        group_durations.append(sum(durations[index] for index in group))

    current_label = group_labels[0]
    current_duration = group_durations[0]
    for group_index in range(1, len(group_labels)):
        crossfade = crossfades[group_index - 1]
        output_label = f"vx{group_index}"
        offset = current_duration - crossfade
        filters.append(
            f"[{current_label}][{group_labels[group_index]}]"
            f"xfade=transition=fade:duration={_fmt(crossfade)}:offset={_fmt(offset)}[{output_label}]"
        )
        current_label = output_label
        current_duration += group_durations[group_index] - crossfade

    filters.append(
        f"[{current_label}]trim=duration={_fmt(contract.duration_s)},setpts=PTS-STARTPTS[vbase]"
    )
    _require(len(title_images) == len(contract.raw["titles"]), "title image count does not match the edit contract")
    for title_image in title_images:
        inputs.extend([
            "-loop", "1", "-framerate", str(contract.fps), "-t", _fmt(contract.duration_s),
            "-i", str(title_image),
        ])
    title_label = "vbase"
    for index, title in enumerate(contract.raw["titles"]):
        output_label = f"vtitle{index}"
        title_input = len(contract.video_clips) + index
        filters.append(
            f"[{title_label}][{title_input}:v:0]overlay=x=0:y=0:format=auto:"
            f"enable='gte(t,{_fmt(title['start_s'])})*lt(t,{_fmt(title['end_s'])})'[{output_label}]"
        )
        title_label = output_label
    filters.append(f"[{title_label}]format={working_pix_fmt}[vout]")
    return inputs, filters, "vout"


def _rgba_color(value: str) -> tuple[int, int, int, int]:
    """Parse common Pillow colors plus FFmpeg's ``name@opacity`` notation."""

    from PIL import ImageColor

    color, marker, opacity_text = value.partition("@")
    try:
        rgb = ImageColor.getrgb(color)
        opacity = float(opacity_text) if marker else 1.0
    except (ValueError, TypeError) as exc:
        raise VideoEditError(f"invalid title color: {value!r}") from exc
    _require(0.0 <= opacity <= 1.0, f"title color opacity must be 0..1: {value!r}")
    if len(rgb) == 4:
        return int(rgb[0]), int(rgb[1]), int(rgb[2]), round(int(rgb[3]) * opacity)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), round(255 * opacity)


def render_title_images(contract: EditContract, destination: str | Path) -> list[Path]:
    """Rasterize local-font titles to transparent full-canvas overlays.

    Homebrew's default FFmpeg build may omit ``drawtext``/libfreetype.  Pillow is
    already a project dependency, so rasterizing once and using FFmpeg's ubiquitous
    ``overlay`` filter keeps title rendering local, deterministic, and portable.
    """

    from PIL import Image, ImageDraw, ImageFont

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, title in enumerate(contract.raw["titles"]):
        image = Image.new("RGBA", (contract.width, contract.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font_path = contract.assets[title["font_asset_id"]].path
        try:
            font = ImageFont.truetype(str(font_path), int(title["font_size"]))
        except (OSError, ValueError) as exc:
            raise VideoEditError(f"failed to load title font {font_path}: {exc}") from exc
        text = str(title["text"])
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align=str(title["align"]))
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        margin_x = int(title["margin_x"])
        margin_y = int(title["margin_y"])
        x = {
            "left": margin_x - bbox[0],
            "center": (contract.width - text_width) / 2 - bbox[0],
            "right": contract.width - margin_x - text_width - bbox[0],
        }[str(title["align"])]
        y = {
            "top": margin_y - bbox[1],
            "center": (contract.height - text_height) / 2 - bbox[1],
            "bottom": contract.height - margin_y - text_height - bbox[1],
        }[str(title["valign"])]
        if title["box"]:
            border = int(title["box_border_w"])
            rect = (
                math.floor(x + bbox[0] - border),
                math.floor(y + bbox[1] - border),
                math.ceil(x + bbox[2] + border),
                math.ceil(y + bbox[3] + border),
            )
            draw.rectangle(rect, fill=_rgba_color(str(title["box_color"])))
        draw.multiline_text(
            (x, y), text, font=font, fill=_rgba_color(str(title["color"])), spacing=4,
            align=str(title["align"]),
        )
        path = root / f"title_{index:02d}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        outputs.append(path)
    return outputs


def _audio_inputs_and_filters(
    contract: EditContract,
    *,
    first_input_index: int,
    loudnorm_filter: str | None,
) -> tuple[list[str], list[str], str, bool]:
    inputs: list[str] = []
    filters: list[str] = []
    clips = contract.audio_clips
    if not clips:
        inputs.extend([
            "-f", "lavfi", "-t", _fmt(contract.duration_s), "-i", "anullsrc=r=48000:cl=stereo",
        ])
        filters.append(
            f"[{first_input_index}:a:0]atrim=duration={_fmt(contract.duration_s)},"
            "asetpts=PTS-STARTPTS,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[apre]"
        )
        has_program_audio = False
    else:
        labels: list[str] = []
        for offset, clip in enumerate(clips):
            asset = contract.assets[clip["asset_id"]]
            input_index = first_input_index + offset
            inputs.extend(["-i", str(asset.path)])
            duration = float(clip["out_s"]) - float(clip["in_s"])
            chain = [
                f"[{input_index}:a:0]atrim=start={_fmt(clip['in_s'])}:end={_fmt(clip['out_s'])}",
                "asetpts=PTS-STARTPTS",
                "aresample=48000",
                "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
                f"volume={_fmt(clip['gain_db'])}dB",
            ]
            if float(clip["fade_in_s"]) > 0:
                chain.append(f"afade=t=in:st=0:d={_fmt(clip['fade_in_s'])}")
            if float(clip["fade_out_s"]) > 0:
                fade_start = duration - float(clip["fade_out_s"])
                chain.append(f"afade=t=out:st={_fmt(fade_start)}:d={_fmt(clip['fade_out_s'])}")
            delay_ms = round(float(clip["start_s"]) * 1000)
            if delay_ms:
                chain.append(f"adelay={delay_ms}:all=1")
            label = f"ac{offset}"
            filters.append(",".join(chain) + f"[{label}]")
            labels.append(label)
        if len(labels) == 1:
            filters.append(f"[{labels[0]}]anull[amix]")
        else:
            filters.append(
                "".join(f"[{label}]" for label in labels)
                + f"amix=inputs={len(labels)}:duration=longest:normalize=0:dropout_transition=0[amix]"
            )
        filters.append(
            f"[amix]apad=pad_dur={_fmt(contract.duration_s)},atrim=duration={_fmt(contract.duration_s)},"
            "asetpts=PTS-STARTPTS,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[apre]"
        )
        has_program_audio = True
    if loudnorm_filter:
        filters.append(f"[apre]{loudnorm_filter},aresample=48000,aformat=channel_layouts=stereo[aout]")
    else:
        filters.append("[apre]aresample=48000,aformat=channel_layouts=stereo[aout]")
    return inputs, filters, "aout", has_program_audio


def _loudnorm_measure_filter(normalization: Mapping[str, Any]) -> str:
    return (
        f"loudnorm=I={_fmt(normalization['integrated_lufs'])}:"
        f"TP={_fmt(normalization['true_peak_db'])}:LRA={_fmt(normalization['lra'])}:print_format=json"
    )


def _normalization_with_peak_headroom(normalization: Mapping[str, Any]) -> dict[str, Any]:
    """Leave encoder/measurement rounding below the declared true-peak ceiling.

    FFmpeg reports loudnorm true peak to two decimal places and AAC/MP4 encoding can
    land a few hundredths above the filter target.  The production target therefore
    uses a deterministic 0.1 dB margin while verification remains strict against the
    public ceiling (normally -1 dBTP).
    """

    effective = dict(normalization)
    effective["true_peak_db"] = (
        float(normalization["true_peak_db"]) - LOUDNESS_TRUE_PEAK_HEADROOM_DB
    )
    return effective


def _parse_loudnorm(stderr: str) -> dict[str, str]:
    candidates = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    _require(bool(candidates), "FFmpeg loudnorm pass 1 returned no measurement JSON")
    try:
        payload = json.loads(candidates[-1])
    except json.JSONDecodeError as exc:
        raise VideoEditError("FFmpeg loudnorm pass 1 returned invalid measurement JSON") from exc
    required = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    _require(required <= set(payload), "FFmpeg loudnorm pass 1 measurement is incomplete")
    for key in required:
        try:
            _require(math.isfinite(float(payload[key])), f"FFmpeg loudnorm measurement {key} is not finite")
        except (TypeError, ValueError) as exc:
            raise VideoEditError(f"FFmpeg loudnorm measurement {key} is not numeric") from exc
    return {str(key): str(value) for key, value in payload.items()}


def _loudnorm_apply_filter(normalization: Mapping[str, Any], measured: Mapping[str, str]) -> str:
    return (
        f"loudnorm=I={_fmt(normalization['integrated_lufs'])}:"
        f"TP={_fmt(normalization['true_peak_db'])}:LRA={_fmt(normalization['lra'])}:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )


def _run(command: Sequence[str], *, timeout: float, label: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(list(command), capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise VideoEditError(f"{label} executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoEditError(f"{label} timed out after {timeout:g}s") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-2000:]
        raise VideoEditError(f"{label} failed with exit {result.returncode}: {tail}")
    return result


def measure_loudness(
    contract: EditContract,
    *,
    ffmpeg: str = "ffmpeg",
    timeout: float = 600.0,
) -> dict[str, Any]:
    normalization = contract.raw["audio_normalization"]
    if not contract.audio_clips:
        return {"status": "not_applicable", "reason": "no_program_audio"}
    if not normalization["enabled"]:
        return {"status": "disabled"}
    effective = _normalization_with_peak_headroom(normalization)
    inputs, filters, label, _ = _audio_inputs_and_filters(
        contract, first_input_index=0, loudnorm_filter=_loudnorm_measure_filter(effective)
    )
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "info", *inputs,
        "-filter_complex", ";".join(filters), "-map", f"[{label}]", "-f", "null", "-",
    ]
    result = _run(command, timeout=timeout, label="FFmpeg loudnorm pass 1")
    measured = _parse_loudnorm(result.stderr)
    return {
        "status": "measured",
        "target": {
            "integrated_lufs": float(normalization["integrated_lufs"]),
            "true_peak_db": float(normalization["true_peak_db"]),
            "lra": float(normalization["lra"]),
            "render_true_peak_db": float(effective["true_peak_db"]),
        },
        "measured": measured,
    }


def verify_output_loudness(
    media: str | Path,
    normalization: Mapping[str, Any],
    *,
    ffmpeg: str = "ffmpeg",
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Audit the encoded output against the requested LUFS/true-peak ceiling."""

    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-i", str(media), "-map", "0:a:0",
        "-af", _loudnorm_measure_filter(normalization), "-f", "null", "-",
    ]
    result = _run(command, timeout=timeout, label="FFmpeg loudness verification")
    measured = _parse_loudnorm(result.stderr)
    integrated_lufs = float(measured["input_i"])
    true_peak_db = float(measured["input_tp"])
    integrated_error = abs(integrated_lufs - float(normalization["integrated_lufs"]))
    peak_within_ceiling = true_peak_db <= float(normalization["true_peak_db"])
    checks = {
        "integrated_lufs_within_0_6": integrated_error <= 0.6,
        "true_peak_at_or_below_ceiling": peak_within_ceiling,
    }
    return {
        "passed": all(checks.values()),
        "integrated_lufs": integrated_lufs,
        "true_peak_db": true_peak_db,
        "integrated_error_lu": integrated_error,
        "checks": checks,
        "raw": measured,
    }


def build_render_command(
    contract: EditContract,
    *,
    temporary_output: str | Path,
    loudness: Mapping[str, Any],
    title_images: Sequence[str | Path] | None = None,
    ffmpeg: str = "ffmpeg",
) -> tuple[list[str], str]:
    resolved_title_images = [Path(path) for path in (title_images or [])]
    video_inputs, video_filters, video_label = _video_inputs_and_filters(
        contract, title_images=resolved_title_images
    )
    normalization = contract.raw["audio_normalization"]
    loudnorm_filter: str | None = None
    if loudness.get("status") == "measured":
        loudnorm_filter = _loudnorm_apply_filter(
            _normalization_with_peak_headroom(normalization), loudness["measured"]
        )
    audio_inputs, audio_filters, audio_label, _ = _audio_inputs_and_filters(
        contract,
        first_input_index=len(contract.video_clips) + len(resolved_title_images),
        loudnorm_filter=loudnorm_filter,
    )
    filter_graph = ";".join([*video_filters, *audio_filters])
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
        *video_inputs, *audio_inputs,
        "-filter_complex", filter_graph,
        "-map", f"[{video_label}]", "-map", f"[{audio_label}]",
        "-frames:v", str(contract.total_frames), "-fps_mode", "cfr", "-r", str(contract.fps),
        "-t", _fmt(contract.duration_s), "-ar", "48000", "-ac", "2",
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        "-metadata", f"comment={EDIT_SCHEMA}",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        *PROFILE_ARGS[contract.profile],
        str(temporary_output),
    ]
    return command, filter_graph


def _first_stream(observed: Mapping[str, Any], stream_type: str) -> Mapping[str, Any] | None:
    for stream in observed.get("streams", []):
        if isinstance(stream, Mapping) and stream.get("codec_type") == stream_type:
            return stream
    return None


def verify_render_stream(contract: EditContract, observed: Mapping[str, Any]) -> dict[str, Any]:
    video = _first_stream(observed, "video")
    audio = _first_stream(observed, "audio")
    observed_duration = float(observed.get("duration_s") or 0.0)
    video_duration = float(video.get("duration_s") or 0.0) if video else 0.0
    audio_duration = float(audio.get("duration_s") or 0.0) if audio else 0.0
    video_duration_observed = video_duration > 0.0
    audio_duration_observed = audio_duration > 0.0
    frame_tolerance = 1 / contract.fps + 1e-6
    checks = {
        "one_video_stream": sum(1 for s in observed.get("streams", []) if s.get("codec_type") == "video") == 1,
        "one_audio_stream": sum(1 for s in observed.get("streams", []) if s.get("codec_type") == "audio") == 1,
        "width": bool(video and video.get("width") == contract.width),
        "height": bool(video and video.get("height") == contract.height),
        "fps": bool(video and abs(float(video.get("fps") or 0.0) - contract.fps) < 0.01),
        "frames": bool(
            video
            and (
                int(video.get("nb_frames") or 0) == contract.total_frames
                if int(video.get("nb_frames") or 0) > 0
                else round(float(observed.get("duration_s") or 0.0) * contract.fps) == contract.total_frames
            )
        ),
        "duration_within_one_frame": abs(observed_duration - contract.duration_s) <= frame_tolerance,
        "video_duration_observed": video_duration_observed,
        "audio_duration_observed": audio_duration_observed,
        "video_duration_within_one_frame": bool(
            video and video_duration_observed and abs(video_duration - contract.duration_s) <= frame_tolerance
        ),
        "audio_duration_within_one_frame": bool(
            audio and audio_duration_observed and abs(audio_duration - contract.duration_s) <= frame_tolerance
        ),
        "av_alignment_within_one_frame": bool(
            video
            and audio
            and video_duration_observed
            and audio_duration_observed
            and abs(video_duration - audio_duration) <= frame_tolerance
        ),
        "audio_sample_rate": bool(audio and audio.get("sample_rate") == 48000),
        "audio_channels": bool(audio and audio.get("channels") == 2),
    }
    if contract.profile in {"work", "delivery"}:
        checks["video_codec"] = bool(video and video.get("codec_name") == "h264")
        checks["audio_codec"] = bool(audio and audio.get("codec_name") == "aac")
        checks["pixel_format"] = bool(video and video.get("pix_fmt") == "yuv420p")
    else:
        checks["video_codec"] = bool(video and video.get("codec_name") == "prores")
        checks["prores_profile"] = bool(video and str(video.get("profile") or "").upper() == "LT")
        checks["audio_codec"] = bool(audio and audio.get("codec_name") == "pcm_s24le")
        checks["pixel_format"] = bool(video and video.get("pix_fmt") == "yuv422p10le")
    if contract.profile == "delivery":
        checks["h264_profile"] = bool(video and str(video.get("profile") or "").lower() == "high")
    return {"ok": all(checks.values()), "checks": checks}


def _ffmpeg_version(ffmpeg: str) -> str:
    result = _run([ffmpeg, "-version"], timeout=15, label="FFmpeg version probe")
    return result.stdout.splitlines()[0] if result.stdout else ""


def _sanitized_command(
    command: Sequence[str],
    temporary_output: Path,
    output: Path,
    title_images: Sequence[Path],
) -> list[str]:
    replacements = {str(temporary_output): str(output)}
    replacements.update({str(path): f"<TITLE_OVERLAY_{index:02d}.png>" for index, path in enumerate(title_images)})
    return [replacements.get(str(item), str(item)) for item in command]


def _write_json_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _publish_pair(media_tmp: Path, output: Path, evidence_tmp: Path, evidence: Path) -> None:
    """Atomically publish two same-filesystem hard links, rolling back on pair failure."""

    published_media = False
    try:
        os.link(media_tmp, output)
        published_media = True
        os.link(evidence_tmp, evidence)
    except FileExistsError as exc:
        if published_media:
            output.unlink(missing_ok=True)
        raise VideoEditError(f"no-clobber: output or evidence already exists ({exc.filename})") from exc
    except OSError as exc:
        if published_media:
            output.unlink(missing_ok=True)
        raise VideoEditError(f"failed to publish render atomically: {exc}") from exc


def render_edit(
    contract: EditContract,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout: float = 3600.0,
) -> RenderResult:
    """Render, verify, and no-clobber publish an edit and evidence sidecar."""

    output = contract.output
    evidence = output.with_suffix(output.suffix + ".evidence.json")
    _require(not output.exists(), f"no-clobber: output already exists: {output}")
    _require(not evidence.exists(), f"no-clobber: evidence already exists: {evidence}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_observations = verify_asset_streams(contract, ffprobe=ffprobe)
    loudness = measure_loudness(contract, ffmpeg=ffmpeg, timeout=min(timeout, 900.0))

    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.render-", dir=output.parent) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        media_tmp = temp_dir / f"render{output.suffix}"
        evidence_tmp = temp_dir / "evidence.json"
        title_images = render_title_images(contract, temp_dir / "titles")
        command, filter_graph = build_render_command(
            contract,
            temporary_output=media_tmp,
            loudness=loudness,
            title_images=title_images,
            ffmpeg=ffmpeg,
        )
        _run(command, timeout=timeout, label="FFmpeg edit render")
        _require(media_tmp.is_file() and media_tmp.stat().st_size > 0, "FFmpeg produced no render")
        observed = probe_media(media_tmp, ffprobe=ffprobe)
        stream_contract = verify_render_stream(contract, observed)
        _require(stream_contract["ok"], f"render stream contract failed: {stream_contract['checks']}")
        if loudness.get("status") == "measured":
            loudness = dict(loudness)
            loudness["verification"] = verify_output_loudness(
                media_tmp,
                contract.raw["audio_normalization"],
                ffmpeg=ffmpeg,
                timeout=min(timeout, 900.0),
            )
            _require(loudness["verification"]["passed"],
                     f"render loudness verification failed: {loudness['verification']['checks']}")

        spec_sha = sha256_file(contract.source_path) if contract.source_path else canonical_json_sha256(contract.raw)
        evidence_payload: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "edit_schema_version": EDIT_SCHEMA,
            "project_id": contract.raw["project_id"],
            "edit_id": contract.raw["edit_id"],
            "driving_model": contract.raw["driving_model"],
            "spec_path": str(contract.source_path) if contract.source_path else None,
            "spec_sha256": spec_sha,
            "effective_spec_sha256": canonical_json_sha256(contract.raw),
            "profile": contract.profile,
            "output": {
                "path": str(output),
                "sha256": sha256_file(media_tmp),
                "bytes": media_tmp.stat().st_size,
            },
            "timeline": {
                "width": contract.width,
                "height": contract.height,
                "fps": contract.fps,
                "duration_s": contract.duration_s,
                "frames": contract.total_frames,
            },
            "assets": [
                {"id": asset.id, "kind": asset.kind, "path": str(asset.path), "sha256": asset.sha256}
                for asset in contract.assets.values()
            ],
            "source_ffprobe": source_observations,
            "audio_normalization": loudness,
            "ffmpeg": {
                "version": _ffmpeg_version(ffmpeg),
                "argv": _sanitized_command(command, media_tmp, output, title_images),
                "filter_complex_sha256": hashlib.sha256(filter_graph.encode("utf-8")).hexdigest(),
            },
            "ffprobe": observed,
            "stream_contract": stream_contract,
            "no_clobber": True,
        }
        _write_json_temp(evidence_tmp, evidence_payload)
        _publish_pair(media_tmp, output, evidence_tmp, evidence)

    return RenderResult(output=output, evidence=evidence, observed=observed, stream_contract=stream_contract)


def shell_join(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


__all__ = [
    "DEFAULT_LOUDNESS",
    "EDIT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "EditContract",
    "MAX_CROSSFADE_S",
    "PROFILE_ARGS",
    "PROFILE_SUFFIX",
    "RenderResult",
    "VideoEditError",
    "build_render_command",
    "canonical_json_sha256",
    "load_and_validate_edit",
    "load_edit_spec",
    "measure_loudness",
    "probe_media",
    "render_edit",
    "render_title_images",
    "sha256_file",
    "shell_join",
    "validate_edit_spec",
    "verify_asset_streams",
    "verify_output_loudness",
    "verify_render_stream",
]
