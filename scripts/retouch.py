#!/usr/bin/env python3
"""Corrective retouch pass: fixes defects (noise, colour cast, dull contrast, blown/crushed levels,
soft focus) rather than adding a stylistic look. Counterpart to post_fx.py's grade_image — post_fx
adds a look to a clean render, this cleans up a render before (or instead of) grading it. Pure
post-processing (PIL + numpy); cv2 is an optional denoise accelerant with a PIL fallback, so the
script still runs without it.

Apply order (classic corrective chain): denoise -> auto_wb -> wb_temp/tint -> auto_levels -> manual
levels -> curve -> scurve -> shadows -> highlights -> vibrance -> saturation -> clarity -> sharpen.

Usage:
  retouch.py in.png                              # default 'fix' preset -> in_rt.png
  retouch.py in.png --preset crisp --out sharp.png
  retouch.py in.png --preset none --auto_wb 0.6 --sharpen 0.3   # start neutral, add one correction
  retouch.py in.png --preset fix --curve "0:0,128:145,255:255"
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

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    cv2 = None
    _HAVE_CV2 = False

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class Retouch:
    """A corrective pass. All fields neutral by default (neutral == exact identity). Unlike Grade,
    this fixes defects rather than adding a look — every field maps to a diagnosable problem."""
    denoise: float = 0.0          # 0..1; cv2.fastNlMeansDenoisingColored on uint8, h=hColor=round(12*denoise),
                                   # templateWindowSize=7, searchWindowSize=21; fallback without cv2: PIL
                                   # MedianFilter(3) result blended with original by factor=denoise.
    auto_wb: float = 0.0          # 0..1 gray-world strength: per-channel means m_c, gains g_c = mean(m)/m_c
                                   # clipped to [0.7, 1.4]; effective gain 1 + auto_wb*(g_c-1); after applying,
                                   # rescale all channels by (mean luminance before / mean luminance after).
    wb_temp: float = 0.0          # -1..1 corrective warm/cool: R *= 1+0.10*t, B *= 1-0.10*t, then same
    wb_tint: float = 0.0          # -1..1 green/magenta: G *= 1+0.10*tint     # luminance rescale as auto_wb
    auto_levels: float = 0.0      # percentile clip per side in PERCENT (0=off, typical 0.1..1.0). Compute
                                   # lo/hi percentiles; rescale x=(x-lo)/(hi-lo) with guard hi-lo>=1/255.
                                   # Skipped (per channel, or on luminance) when hi-lo is negligible next
                                   # to the image's actual min-max range: flat/near-flat inputs pass
                                   # through unchanged instead of collapsing to black — see _auto_levels.
    per_channel: bool = False     # auto_levels per channel (also corrects casts) vs on luminance (shared lo/hi)
    black: float = 0.0            # manual levels: x=(x-black)/(white-black) clipped, then x**(1/gamma)
    white: float = 1.0
    gamma: float = 1.0            # >1 brightens midtones (Photoshop convention)
    curve: str | None = None      # "in:out,in:out,..." points in 0..255, e.g. "0:0,128:145,255:255"; divide
                                   # by 255, linearly interpolate with np.interp, and apply identically to all
                                   # three channels. Validated: >=2
                                   # points, inputs strictly increasing, all values in 0..255 -> else ValueError.
    scurve: float = 0.0           # -1..1: sm = x*x*(3-2x); out = x + scurve*(sm - x)
    shadows: float = 0.0          # 0..1 shadow lift: Lb = GaussianBlur(luminance, radius=0.02*max(H,W));
                                   # mask=(1-Lb)**2; lifted = x**(1/(1+1.5*shadows)); out = x + mask*(lifted-x)
    highlights: float = 0.0       # 0..1 recovery: mask=Lb**2; rec = x**(1+1.5*highlights); out=x+mask*(rec-x)
    vibrance: float = 0.0         # -1..1: sat = max-min per pixel; w=(1-sat)**2; out = lum + (x-lum)*(1+vibrance*w)
    saturation: float = 1.0       # same math as post_fx
    clarity: float = 0.0          # -1..1 local contrast: L=luminance; Lb=GaussianBlur(L, radius=0.015*max(H,W));
                                   # detail=L-Lb; midw = clip(1-abs(2L-1),0,1); out = x + clarity*(detail*midw)[...,None]
    sharpen: float = 0.0          # USM amount 0..~2: blur = GaussianBlur(x, sharpen_radius); out = x + sharpen*(x-blur)
    sharpen_radius: float = 1.5


PRESETS: dict[str, Retouch] = {
    "none": Retouch(),
    "fix": Retouch(auto_wb=0.5, auto_levels=0.3, clarity=0.08, sharpen=0.25),
    "crisp": Retouch(clarity=0.20, sharpen=0.6, scurve=0.12),
    "soft": Retouch(denoise=0.25, shadows=0.15, clarity=-0.10),
}
DEFAULT_PRESET = "fix"

_NUMERIC_FIELDS = ("denoise", "auto_wb", "wb_temp", "wb_tint", "auto_levels", "black", "white",
                   "gamma", "scurve", "shadows", "highlights", "vibrance", "saturation", "clarity",
                   "sharpen", "sharpen_radius")

_CLOSED_RANGES: dict[str, tuple[float, float]] = {
    "denoise": (0.0, 1.0),
    "auto_wb": (0.0, 1.0),
    "wb_temp": (-1.0, 1.0),
    "wb_tint": (-1.0, 1.0),
    "black": (0.0, 1.0),
    "white": (0.0, 1.0),
    "scurve": (-1.0, 1.0),
    "shadows": (0.0, 1.0),
    "highlights": (0.0, 1.0),
    "vibrance": (-1.0, 1.0),
    "clarity": (-1.0, 1.0),
}


def _validate_retouch(r: Retouch) -> None:
    """Reject values outside the public Retouch contract before any pixel math runs."""
    for name in _NUMERIC_FIELDS:
        value = getattr(r, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")

    for name, (lo, hi) in _CLOSED_RANGES.items():
        value = getattr(r, name)
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")

    if not 0.0 <= r.auto_levels < 50.0:
        raise ValueError(f"auto_levels must be in [0, 50), got {r.auto_levels}")
    if r.gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {r.gamma}")
    if not 0.0 <= r.saturation <= 2.0:
        raise ValueError(f"saturation must be in [0, 2], got {r.saturation}")
    if not 0.0 <= r.sharpen <= 2.0:
        raise ValueError(f"sharpen must be in [0, 2], got {r.sharpen}")
    if r.sharpen_radius <= 0.0:
        raise ValueError(f"sharpen_radius must be > 0, got {r.sharpen_radius}")
    if r.black >= r.white:
        raise ValueError(f"black must be less than white, got black={r.black}, white={r.white}")
    if r.curve:
        _parse_curve(r.curve)


def _gaussian_blur(x: np.ndarray, radius: float) -> np.ndarray:
    """Blur a float32 array in [0, 1] (2D luminance or 3D RGB) via a uint8 PIL round-trip, mirroring
    the blur used by post_fx._halation. The uint8 round-trip loses a little precision but keeps this
    fallback dependency-free; radius is floored so a near-zero request still blurs a hair (avoids a
    zero-radius no-op that would make clarity/sharpen silently inert at tiny strengths)."""
    u8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(u8, mode="L") if x.ndim == 2 else Image.fromarray(u8)
    blurred = img.filter(ImageFilter.GaussianBlur(max(radius, 0.01)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _denoise(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if _HAVE_CV2 and cv2 is not None:
        h = float(round(12 * amount))
        den = cv2.fastNlMeansDenoisingColored(u8, None, h=h, hColor=h,
                                               templateWindowSize=7, searchWindowSize=21)
    else:
        med = np.asarray(Image.fromarray(u8).filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
        den = (u8.astype(np.float32) * (1.0 - amount) + med * amount).astype(np.uint8)
    return den.astype(np.float32) / 255.0


def _auto_wb(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    lum_before = float((arr @ _LUMA).mean())
    means = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), 0.7, 1.4)
    out = arr * (1.0 + strength * (gains - 1.0))
    lum_after = float((out @ _LUMA).mean())
    if lum_after > 1e-6:
        out = out * (lum_before / lum_after)
    return out


def _wb_temp_tint(arr: np.ndarray, temp: float, tint: float) -> np.ndarray:
    """Combined stage (the Apply order lists wb_temp/tint as one step): shift R/B for temp and G for
    tint, then rescale all channels once by the shared before/after luminance ratio, same as auto_wb."""
    if not temp and not tint:
        return arr
    lum_before = float((arr @ _LUMA).mean())
    r = arr[..., 0] * (1.0 + 0.10 * temp)
    g = arr[..., 1] * (1.0 + 0.10 * tint)
    b = arr[..., 2] * (1.0 - 0.10 * temp)
    out = np.stack([r, g, b], axis=-1)
    lum_after = float((out @ _LUMA).mean())
    if lum_after > 1e-6:
        out = out * (lum_before / lum_after)
    return out


_AUTO_LEVELS_MIN_SPAN_RATIO = 0.05  # skip the stretch when the percentile span is this narrow relative
                                     # to the image's real min-max range: a flat or near-flat image (or
                                     # one whose only spread is a handful of outlier pixels the percentile
                                     # clip cuts away) would otherwise divide by a near-zero span and
                                     # collapse every pixel toward one end of 0..1 (confirmed: an all-white
                                     # image, or a 4091-vs-5-pixel near-flat image, both crushed to black)


def _auto_levels(arr: np.ndarray, percent: float, per_channel: bool) -> np.ndarray:
    if percent <= 0:
        return arr
    lo_p, hi_p = percent, 100.0 - percent
    if per_channel:
        lo = np.percentile(arr, lo_p, axis=(0, 1))
        hi = np.percentile(arr, hi_p, axis=(0, 1))
        actual_range = np.maximum(arr.max(axis=(0, 1)) - arr.min(axis=(0, 1)), 1e-6)
        span = hi - lo
        safe = span >= _AUTO_LEVELS_MIN_SPAN_RATIO * actual_range
        stretched = (arr - lo) / np.maximum(span, 1.0 / 255.0)
        return np.where(safe, stretched, arr)
    lum = arr @ _LUMA
    lo = float(np.percentile(lum, lo_p))
    hi = float(np.percentile(lum, hi_p))
    actual_range = max(float(lum.max() - lum.min()), 1e-6)
    span = hi - lo
    if span < _AUTO_LEVELS_MIN_SPAN_RATIO * actual_range:
        return arr
    span = max(span, 1.0 / 255.0)
    return (arr - lo) / span


def _manual_levels(arr: np.ndarray, black: float, white: float, gamma: float) -> np.ndarray:
    if not np.isfinite(gamma) or gamma <= 0.0:
        # 1/gamma below would divide by zero, and a negative gamma sends 0-valued pixels to inf
        raise ValueError(f"gamma must be finite and > 0, got {gamma}")
    if black == 0.0 and white == 1.0 and gamma == 1.0:
        return arr
    span = white - black
    if abs(span) < 1e-6:
        span = 1e-6 if span >= 0 else -1e-6
    out = np.clip((arr - black) / span, 0.0, 1.0)  # clip is part of the spec'd formula: gamma's
    if gamma != 1.0:                                # fractional power below needs a non-negative base
        out = out ** (1.0 / gamma)
    return out


def _parse_curve(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse "in:out,in:out,..." (0..255) into normalized (0..1) x/y control points."""
    points: list[tuple[float, float]] = []
    for token in spec.split(","):
        in_s, sep, out_s = token.strip().partition(":")
        if not sep:
            raise ValueError(f"invalid curve point {token!r}: expected 'in:out'")
        try:
            in_v, out_v = float(in_s), float(out_s)
        except ValueError as exc:
            raise ValueError(f"invalid curve point {token!r}: expected numeric 'in:out'") from exc
        if not (0.0 <= in_v <= 255.0 and 0.0 <= out_v <= 255.0):
            raise ValueError(f"curve point {token!r} out of range 0..255")
        points.append((in_v, out_v))
    if len(points) < 2:
        raise ValueError("curve needs at least 2 points")
    xs = [p[0] for p in points]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("curve input values must be strictly increasing")
    xs_arr = np.asarray(xs, dtype=np.float64) / 255.0
    ys_arr = np.asarray([p[1] for p in points], dtype=np.float64) / 255.0
    return xs_arr, ys_arr


def _apply_curve(arr: np.ndarray, spec: str) -> np.ndarray:
    xs, ys = _parse_curve(spec)
    flat = arr.reshape(-1).astype(np.float64)
    # One interpolation algorithm on every host: the earlier optional PCHIP path diverged from the
    # numpy fallback by >33/255 for valid curves, making the same config environment-dependent.
    out = np.interp(flat, xs, ys)
    return out.reshape(arr.shape).astype(np.float32)


def _scurve(arr: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return arr
    sm = arr * arr * (3.0 - 2.0 * arr)
    return arr + amount * (sm - arr)


def _shadows_highlights(arr: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """Shared Lb (one luminance blur) feeds both masks, matching the spec's shadows doc defining Lb
    and the highlights doc reusing it. highlights operates on the shadows-lifted result, per the
    Apply order listing them as two sequential stages rather than two independent ones."""
    if not shadows and not highlights:
        return arr
    lb = _gaussian_blur(arr @ _LUMA, 0.02 * max(arr.shape[0], arr.shape[1]))
    out = arr
    if shadows:
        mask = ((1.0 - lb) ** 2)[..., None]
        base = np.clip(out, 0.0, 1.0)  # guards the fractional-power base against negative values
        lifted = base ** (1.0 / (1.0 + 1.5 * shadows))
        out = out + mask * (lifted - out)
    if highlights:
        mask = (lb ** 2)[..., None]
        base = np.clip(out, 0.0, 1.0)  # same guard, now against a >1 base for the >=1 exponent
        rec = base ** (1.0 + 1.5 * highlights)
        out = out + mask * (rec - out)
    return out


def _vibrance(arr: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return arr
    lum = (arr @ _LUMA)[..., None]
    sat = (arr.max(axis=-1) - arr.min(axis=-1))[..., None]
    weight = (1.0 - sat) ** 2  # low-saturation pixels get boosted proportionally more
    return lum + (arr - lum) * (1.0 + amount * weight)


def _saturation(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 1.0:
        return arr
    lum = (arr @ _LUMA)[..., None]
    return lum + (arr - lum) * amount


def _clarity(arr: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return arr
    lum = arr @ _LUMA
    lb = _gaussian_blur(lum, 0.015 * max(arr.shape[0], arr.shape[1]))
    detail = lum - lb
    midw = np.clip(1.0 - np.abs(2.0 * lum - 1.0), 0.0, 1.0)  # protects highlight/shadow clipping
    return arr + amount * (detail * midw)[..., None]


def _sharpen(arr: np.ndarray, amount: float, radius: float) -> np.ndarray:
    if not amount:
        return arr
    blur = _gaussian_blur(arr, radius)
    return arr + amount * (arr - blur)


def retouch_image(img: Image.Image, r: Retouch) -> Image.Image:
    """Apply a corrective retouch pass. Deterministic — nothing in this chain uses randomness. Works
    in float32 0..1 RGB; clipped to [0, 1] between stages wherever the preceding math can escape that
    range (white-balance gains, percentile stretch, the s-curve, and the
    vibrance/saturation extrapolation away from luminance all can), so later stages always see a
    valid base to work from."""
    _validate_retouch(r)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    arr = _denoise(arr, r.denoise)
    arr = _auto_wb(arr, r.auto_wb)
    arr = _wb_temp_tint(arr, r.wb_temp, r.wb_tint)
    arr = np.clip(arr, 0.0, 1.0)  # WB gains can push a bright, already-tinted pixel out of range
    arr = _auto_levels(arr, r.auto_levels, r.per_channel)
    arr = np.clip(arr, 0.0, 1.0)  # the percentile stretch can overshoot on outlier pixels
    arr = _manual_levels(arr, r.black, r.white, r.gamma)  # clips internally; see docstring above
    if r.curve:
        arr = _apply_curve(arr, r.curve)
        arr = np.clip(arr, 0.0, 1.0)  # keep an explicit stage boundary after the validated curve
    arr = _scurve(arr, r.scurve)
    arr = np.clip(arr, 0.0, 1.0)  # keeps the shadows/highlights power bases below well-defined
    arr = _shadows_highlights(arr, r.shadows, r.highlights)
    arr = _vibrance(arr, r.vibrance)
    arr = _saturation(arr, r.saturation)
    arr = np.clip(arr, 0.0, 1.0)  # both extrapolate away from the luminance anchor and can escape
    arr = _clarity(arr, r.clarity)
    arr = _sharpen(arr, r.sharpen, r.sharpen_radius)

    out = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out)  # 3-channel uint8 -> RGB inferred


def _retouch_from_args(args: argparse.Namespace) -> Retouch:
    """Start from a preset, then apply any per-field CLI overrides."""
    r = PRESETS[args.preset]
    fields = _NUMERIC_FIELDS + ("curve", "per_channel")
    overrides = {name: getattr(args, name) for name in fields if getattr(args, name) is not None}
    return replace(r, **overrides) if overrides else r


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
    ap = argparse.ArgumentParser(description="Corrective retouch pass (denoise / white balance / "
                                             "levels / curves / shadows-highlights / clarity / sharpen) "
                                             "— fix defects rather than add a look.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--out", default=None, help="output path (default: <stem>_rt.png next to input)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output file (the input image is never replaceable)")
    ap.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET,
                    help=f"base correction (default {DEFAULT_PRESET})")
    for name in _NUMERIC_FIELDS:
        ap.add_argument(f"--{name}", type=float, default=None, help=f"override preset {name}")
    ap.add_argument("--curve", type=str, default=None,
                    help="override preset curve, e.g. '0:0,128:145,255:255'")
    ap.add_argument("--per-channel", dest="per_channel", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="override preset per_channel (auto_levels per channel vs. on luminance); "
                         "--no-per-channel forces it back off")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_rt.png")
    out_exists = out.exists()
    if out.resolve() == src.resolve() or (out_exists and out.samefile(src)):
        sys.exit("output path resolves to the input image; choose a different --out")
    if out.is_symlink() and not out_exists:
        sys.exit(f"output path is a broken symlink: {out}; choose a different --out")
    if out_exists and not args.overwrite:
        sys.exit(f"output already exists: {out} (pass --overwrite to replace it)")
    try:
        r = _retouch_from_args(args)
        with Image.open(src) as source_image:
            rendered = retouch_image(source_image, r)
        _publish_image(rendered, src, out, args.overwrite)
    except ValueError as exc:
        sys.exit(str(exc))
    print(out)


if __name__ == "__main__":
    main()
