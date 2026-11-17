#!/usr/bin/env python3
"""Image-space finishing pass: turns a clean SDXL render into something that reads as hand-crafted
"artwork" rather than a flat AI render. Pure post-processing (PIL + numpy) — no GPU, no ComfyUI.

This is the cheap "render -> artwork" layer the artistic line was missing: a colour grade (split-tone
+ contrast/saturation), subtle highlight halation, a vignette to pull focus, and fine film grain. All
effects are gentle by default and strength-tunable; the order matches the research-backed chain
(grade -> halation -> vignette -> grain).

Usage:
  post_fx.py in.png                          # default 'film' look -> in_fx.png
  post_fx.py in.png --preset teal-orange --out graded.png
  post_fx.py in.png --preset muted --grain 0.0 --vignette 0.15   # preset + per-effect overrides
  post_fx.py in.png --preset none --halation 0.4                 # start neutral, add one effect
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class Grade:
    """A finishing look. Tints are small per-channel RGB offsets (already scaled); the rest are
    multipliers/strengths. contrast/saturation are 1.0 = no change."""
    name: str
    temp: float = 0.0          # -1..1 warm(+)/cool(-) shift
    tint: float = 0.0          # -1..1 green(+)/magenta(-) shift
    contrast: float = 1.0      # 0..2 around mid-grey
    saturation: float = 1.0    # 0..2, 1 neutral
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)     # RGB offsets in -1..1, shadows
    highlight_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB offsets in -1..1, highlights
    halation: float = 0.0      # 0..1 highlight glow
    vignette: float = 0.0      # 0..1 corner darkening
    grain: float = 0.0         # 0..0.15 luminance-noise sigma


PRESETS: dict[str, Grade] = {
    "none": Grade("none"),
    "film": Grade("film", temp=0.10, contrast=1.06, saturation=1.04,
                  shadow_tint=(0.015, 0.005, -0.015), highlight_tint=(0.015, 0.010, -0.005),
                  halation=0.22, vignette=0.18, grain=0.020),
    "teal-orange": Grade("teal-orange", temp=0.06, contrast=1.10, saturation=1.06,
                         shadow_tint=(-0.02, 0.0, 0.03), highlight_tint=(0.04, 0.02, -0.03),
                         halation=0.20, vignette=0.20, grain=0.018),
    "muted": Grade("muted", temp=0.02, contrast=0.96, saturation=0.82,
                   shadow_tint=(0.01, 0.01, 0.0), highlight_tint=(0.01, 0.0, -0.01),
                   halation=0.10, vignette=0.14, grain=0.024),
    "warm": Grade("warm", temp=0.18, contrast=1.05, saturation=1.08,
                  shadow_tint=(0.03, 0.01, -0.02), highlight_tint=(0.04, 0.03, -0.02),
                  halation=0.26, vignette=0.16, grain=0.016),
}
DEFAULT_PRESET = "film"

_SCALAR_RANGES: dict[str, tuple[float, float]] = {
    "temp": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "contrast": (0.0, 2.0),
    "saturation": (0.0, 2.0),
    "halation": (0.0, 1.0),
    "vignette": (0.0, 1.0),
    "grain": (0.0, 0.15),
}


def _validate_grade(grade: Grade, seed: int = 0) -> None:
    """Reject malformed or out-of-contract grades before any pixels are changed."""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    for name, (lo, hi) in _SCALAR_RANGES.items():
        value = getattr(grade, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    for name in ("shadow_tint", "highlight_tint"):
        value = getattr(grade, name)
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly 3 RGB offsets, got {value!r}")
        if not all(np.isfinite(component) for component in value):
            raise ValueError(f"{name} components must be finite, got {value!r}")
        if not all(-1.0 <= component <= 1.0 for component in value):
            raise ValueError(f"{name} components must be in [-1, 1], got {value!r}")


def _split_tone(arr: np.ndarray, shadow: tuple, highlight: tuple) -> np.ndarray:
    lum = arr @ _LUMA
    s = np.asarray(shadow, dtype=np.float32)
    h = np.asarray(highlight, dtype=np.float32)
    return arr + (1.0 - lum)[..., None] * s + lum[..., None] * h


def _halation(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    lum = arr @ _LUMA
    mask = np.clip((lum - 0.70) / 0.30, 0.0, 1.0)[..., None]  # only the brightest ~30%
    bright = (arr * mask * 255.0).clip(0, 255).astype(np.uint8)
    radius = max(2.0, 0.01 * max(arr.shape[:2]))
    blurred = np.asarray(Image.fromarray(bright).filter(ImageFilter.GaussianBlur(radius)),
                         dtype=np.float32) / 255.0
    return 1.0 - (1.0 - arr) * (1.0 - blurred * strength)  # screen blend the glow back in


def _vignette(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    h, w = arr.shape[:2]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = (xx ** 2 + yy ** 2) / 2.0  # ~1 at the corners
    return arr * (1.0 - strength * r2)[..., None]


def grade_image(img: Image.Image, grade: Grade, seed: int = 0) -> Image.Image:
    """Apply a finishing grade. Deterministic for a given (grade, seed)."""
    _validate_grade(grade, seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    if grade.temp:
        arr[..., 0] *= 1.0 + 0.10 * grade.temp
        arr[..., 2] *= 1.0 - 0.10 * grade.temp
    if grade.tint:
        arr[..., 1] *= 1.0 + 0.10 * grade.tint
    if grade.contrast != 1.0:
        arr = (arr - 0.5) * grade.contrast + 0.5
    if grade.saturation != 1.0:
        lum = (arr @ _LUMA)[..., None]
        arr = lum + (arr - lum) * grade.saturation
    if any(grade.shadow_tint) or any(grade.highlight_tint):
        arr = _split_tone(arr, grade.shadow_tint, grade.highlight_tint)
    arr = _halation(np.clip(arr, 0.0, 1.0), grade.halation)
    arr = _vignette(arr, grade.vignette)
    if grade.grain > 0:
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(0.0, grade.grain, size=(arr.shape[0], arr.shape[1], 1)).astype(np.float32)

    out = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out)  # 3-channel uint8 -> RGB inferred


def _grade_from_args(args: argparse.Namespace) -> Grade:
    """Start from a preset, then apply any per-effect CLI overrides."""
    g = PRESETS[args.preset]
    overrides = {k: v for k, v in (
        ("temp", args.temp), ("tint", args.tint), ("contrast", args.contrast),
        ("saturation", args.saturation), ("halation", args.halation),
        ("vignette", args.vignette), ("grain", args.grain)) if v is not None}
    return replace(g, **overrides) if overrides else g


def _publish_image(image: Image.Image, src: Path, out: Path, overwrite: bool) -> None:
    """Publish a fully-rendered image atomically; default mode never replaces a directory entry."""
    for _ in range(100):
        temp = out.with_name(f".{out.stem}.tmp-{uuid.uuid4().hex}{out.suffix}")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        break
    else:
        raise FileExistsError(f"could not allocate a unique sibling temp file for {out}")
    try:
        image.save(temp)
        if overwrite:
            if out.resolve() == src.resolve() or (out.exists() and out.samefile(src)):
                raise ValueError("output path resolves to the input image; choose a different --out")
            if out.is_file() and not out.is_symlink():
                temp.chmod(stat.S_IMODE(out.stat().st_mode))
            os.replace(temp, out)
        else:
            try:
                os.link(temp, out)
            except FileExistsError as exc:
                raise ValueError(f"output appeared during rendering: {out}; rendered image was not published") \
                    from exc
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Image-space finishing pass (colour grade / halation / "
                                             "vignette / film grain) — render -> artwork.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--out", default=None, help="output path (default: <stem>_fx.png next to input)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output file (the input image is never replaceable)")
    ap.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET,
                    help=f"base look (default {DEFAULT_PRESET})")
    ap.add_argument("--seed", type=int, default=0, help="grain seed (deterministic)")
    for name in ("temp", "tint", "contrast", "saturation", "halation", "vignette", "grain"):
        ap.add_argument(f"--{name}", type=float, default=None, help=f"override preset {name}")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_fx.png")
    out_exists = out.exists()
    if out.resolve() == src.resolve() or (out_exists and out.samefile(src)):
        sys.exit("output path resolves to the input image; choose a different --out")
    if out.is_symlink() and not out_exists:
        sys.exit(f"output path is a broken symlink: {out}; choose a different --out")
    if out_exists and not args.overwrite:
        sys.exit(f"output already exists: {out} (pass --overwrite to replace it)")
    try:
        grade = _grade_from_args(args)
        with Image.open(src) as source_image:
            rendered = grade_image(source_image, grade, args.seed)
        _publish_image(rendered, src, out, args.overwrite)
    except ValueError as exc:
        sys.exit(str(exc))
    print(out)


if __name__ == "__main__":
    main()
