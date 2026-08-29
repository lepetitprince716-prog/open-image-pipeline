"""Frozen Wan 2.2 I2V production contract shared by producer and auditor.

This Module is intentionally side-effect free at import time: no ComfyUI, subprocess,
ffprobe, or PIL import.  ``z3_video`` owns graph/transport execution; ``video_probe``
owns planning and review.  Both consume this one recipe and metadata contract so a
parameter change cannot silently drift between generation and release audit.
"""
from __future__ import annotations

import hashlib
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

ARTIFACT_SCHEMA = "loramake-video-artifact-v1"
LEGACY_PLAN_SCHEMA = "loramake_video_probe_plan_v2"
LEGACY_REPORT_SCHEMA = "loramake_video_probe_report_v2"
PLAN_SCHEMA = "loramake_video_probe_plan_v3"
REPORT_SCHEMA = "loramake_video_probe_report_v3"
AUDIT_SCHEMA = "loramake_video_probe_audit_v3"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for_image(path: str | Path) -> int:
    return zlib.crc32(Path(path).read_bytes()) & 0x7FFFFFFF


def sniff_mp4_container(raw: bytes) -> str | None:
    """Early-reject non-MP4 response bodies before authoritative ffprobe."""
    return "mp4" if len(raw) > 1024 and len(raw) >= 12 and raw[4:8] == b"ftyp" else None


def parse_lora(spec: str | None) -> tuple[str, float] | None:
    """Parse FILE[:STRENGTH] with a 1.0 default."""
    if not spec:
        return None
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        if tail.replace(".", "", 1).isdigit():
            return head, float(tail)
    return spec, 1.0


def image_provenance(path: str | Path) -> dict:
    """Freeze one decodable image's path/hash/size/dimensions through one Interface."""
    image = Path(path).resolve()
    if not image.is_file():
        raise ValueError(f"seed image not found or not a regular file: {image}")
    try:
        from PIL import Image

        with Image.open(image) as im:
            dimensions = [int(im.width), int(im.height)]
            im.verify()
    except Exception as exc:
        raise ValueError(f"seed image is not decodable: {image} ({type(exc).__name__})") from exc
    return {
        "path": str(image),
        "sha256": sha256_file(image),
        "bytes": image.stat().st_size,
        "dimensions": dimensions,
    }


def provenance_matches(record: object) -> bool:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return False
    try:
        return image_provenance(record["path"]) == record
    except (OSError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class Wan22I2VSpec:
    """Immutable value object for the production recipe and artifact contract."""

    high_unet: str = "Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf"
    low_unet: str = "Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    text_encoder: str = "umt5-xxl-encoder-Q5_K_M.gguf"
    vae: str = "Wan2.1_VAE.safetensors"
    lightx2v_high: str = "wan22_i2v_lightx2v_4step_high.safetensors"
    lightx2v_low: str = "wan22_i2v_lightx2v_4step_low.safetensors"
    default_length: int = 81
    default_width: int = 832
    default_height: int = 480
    fps: int = 16
    shift: float = 5.0
    steps: int = 4
    boundary: int = 2
    cfg: float = 1.0
    pos_suffix: str = "flat 2D anime, cel shading, clean lineart, limited animation, subtle motion"
    negative: str = ("(((realistic))), ((photograph)), 3d, semi-realistic, depth of field, blurry, "
                     "worst quality, low quality, jpeg artifacts, watermark, text, deformed")
    poll_timeout: float = 2400.0
    min_length: int = 5
    max_length: int = 81
    max_dimension: int = 1024
    engine: str = "wan2.2_i2v_a14b"

    @property
    def max_pixels(self) -> int:
        return self.default_width * self.default_height

    @property
    def default_resolution(self) -> str:
        return f"{self.default_width}x{self.default_height}"

    def resolution(self, value: str) -> tuple[int, int]:
        try:
            parts = value.lower().split("x")
            if len(parts) != 2:
                raise ValueError
            width, height = (int(part) for part in parts)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f"--res must be WxH (got {value!r})") from None
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError(f"--res must be positive multiples of 16 (got {width}x{height})")
        if max(width, height) > self.max_dimension or width * height > self.max_pixels:
            raise ValueError(
                "--res exceeds the frozen 480p safety envelope "
                f"(max edge {self.max_dimension}, max {self.max_pixels} pixels; got {width}x{height})"
            )
        return width, height

    def validate_request(self, image: str | Path, length: int, res: str) -> tuple[int, int]:
        image_provenance(image)
        return self.validate_shape(length, res)

    def validate_shape(self, length: int, res: str) -> tuple[int, int]:
        if length < self.min_length or length > self.max_length or length % 4 != 1:
            raise ValueError(
                f"--length must be {self.min_length}..{self.max_length} and 4n+1 "
                f"(got {length}); try 41, 61, 81"
            )
        return self.resolution(res)

    def recipe(self, *, seed: int, length: int, res: str,
               pos_suffix: str | None = None, anime_lora: str | None = None,
               negative: str | None = None) -> dict:
        width, height = self.validate_shape(length, res)
        suffix = self.pos_suffix if pos_suffix is None else pos_suffix
        resolved_negative = self.negative if negative is None else negative
        if type(seed) is not int or not isinstance(suffix, str) or not isinstance(resolved_negative, str):
            raise ValueError("seed must be an int; pos_suffix and negative must be strings")
        if anime_lora is not None and not isinstance(anime_lora, str):
            raise ValueError("anime_lora must be a string or None")
        return {
            "seed": seed,
            "length": length,
            "res": f"{width}x{height}",
            "fps": self.fps,
            "steps": self.steps,
            "boundary": self.boundary,
            "cfg": self.cfg,
            "shift": self.shift,
            "pos_suffix": suffix,
            "negative": resolved_negative,
            "anime_lora": anime_lora,
        }

    def recipe_is_locked(self, recipe: object) -> bool:
        if not isinstance(recipe, Mapping):
            return False
        try:
            length = recipe["length"]
            self.resolution(recipe["res"])
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            type(recipe.get("seed")) is int
            and type(length) is int
            and self.min_length <= length <= self.max_length
            and length % 4 == 1
            and recipe.get("fps") == self.fps
            and recipe.get("steps") == self.steps
            and recipe.get("boundary") == self.boundary
            and recipe.get("cfg") == self.cfg
            and recipe.get("shift") == self.shift
            and isinstance(recipe.get("pos_suffix"), str)
            and isinstance(recipe.get("negative"), str)
            and (recipe.get("anime_lora") is None or isinstance(recipe.get("anime_lora"), str))
        )

    def stream_matches(self, observed: object, *, width: int, height: int,
                       frames: int, size: int) -> bool:
        if not isinstance(observed, Mapping):
            return False
        fps = observed.get("fps")
        return bool(
            observed.get("container") == "mp4"
            and observed.get("codec") == "h264"
            and observed.get("width") == width
            and observed.get("height") == height
            and observed.get("frames") == frames
            and not isinstance(fps, bool)
            and isinstance(fps, (int, float))
            and math.isfinite(fps)
            and abs(float(fps) - self.fps) < 0.01
            and observed.get("bytes") == size
        )

    def expected_prompt(self, prompt: str, recipe: Mapping) -> str:
        return ", ".join(value for value in (prompt, recipe.get("pos_suffix")) if value)

    def expected_anime_lora(self, recipe: Mapping) -> str | None:
        spec = recipe.get("anime_lora")
        anime = parse_lora(spec) if isinstance(spec, str) else None
        return f"{anime[0]}:{anime[1]}" if anime else None

    def metadata_matches(self, meta: object, *, recipe: Mapping, prompt: str) -> bool:
        if not isinstance(meta, Mapping) or not self.recipe_is_locked(recipe):
            return False
        return bool(
            meta.get("prompt") == self.expected_prompt(prompt, recipe)
            and meta.get("negative") == recipe.get("negative")
            and meta.get("seed") == recipe.get("seed")
            and meta.get("steps") == self.steps
            and meta.get("boundary") == self.boundary
            and meta.get("cfg") == self.cfg
            and meta.get("shift") == self.shift
            and meta.get("engine") == self.engine
            and meta.get("high_unet") == self.high_unet
            and meta.get("low_unet") == self.low_unet
            and meta.get("lightx2v") == [self.lightx2v_high, self.lightx2v_low]
            and meta.get("anime_lora") == self.expected_anime_lora(recipe)
        )


WAN22_I2V = Wan22I2VSpec()

# Seamless-loop variant: identical models and sampler constants, FLF conditioning with the
# same image at both ends.  The suffix keeps the camera anchored so the motion can return
# to the starting pose; the extra negatives suppress the drift modes that break a loop.
LOOP_POS_SUFFIX = (
    WAN22_I2V.pos_suffix
    + ", seamless loop, motion returns to the starting pose, static camera"
)
LOOP_NEGATIVE = WAN22_I2V.negative + ", jerky motion, camera shake, zoom, pan"
WAN22_FLF_LOOP = replace(
    WAN22_I2V,
    engine="wan2.2_flf2v_loop",
    pos_suffix=LOOP_POS_SUFFIX,
    negative=LOOP_NEGATIVE,
)

# Mac-side loop encode defaults (gif long-edge cap controls file size; fps stays native).
GIF_MAX_WIDTH = 480
GIF_FPS = 16
