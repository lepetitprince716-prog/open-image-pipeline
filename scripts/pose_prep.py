#!/usr/bin/env python3
"""Normalize a pose source (OpenPose skeleton map OR a pose reference photo) to the
generation canvas so ControlNet receives an undistorted, well-framed control image.

Raw pose sources get two things wrong for tall 9:16 renders:
  * aspect mismatch — feeding a ~square skeleton to a 768x1344 gen stretches the pose
    vertically, breaking the proportions ControlNet is supposed to lock.
  * framing — full-body sources place a tiny figure in a big frame, which reproduces the
    "small hands/face" failure the pose work is meant to fix.

This scales the figure to the target aspect WITHOUT distortion (fit by height, then
center-crop or letterbox the width), with an optional vertical crop to reframe
full -> cowboy -> portrait so the subject fills more of the frame, and an optional
horizontal offset for off-center composition.

Feed the output to z3_quality_gen.py via --control-image. For a raw photo, let the
pipeline's OpenposePreprocessor extract the skeleton (keep hands); for an already-made
skeleton map, pass --no-preprocess there.

Usage:
  pose_prep.py skel.png                                 # -> skel_prep.png, 768x1344, full
  pose_prep.py photo.jpg --frame cowboy --out cowboy.png
  pose_prep.py skel.png --frame portrait --offset 0.3   # subject shifted right of centre
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# Fraction of the source height kept, measured from the top (head stays in frame).
FRAME_KEEP = {"full": 1.0, "cowboy": 0.72, "portrait": 0.55}


def prep(img: Image.Image, w: int, h: int, frame: str, offset: float,
         pad: tuple[int, int, int]) -> Image.Image:
    """Return an (w x h) RGB image holding the source figure at the target aspect.

    offset in [-1, 1] shifts the subject horizontally: 0 = centred, +1 = hard right,
    -1 = hard left. Height is always filled; width is cropped (figure wider than canvas)
    or letterboxed (figure narrower) with the pad colour.
    """
    img = img.convert("RGB")
    offset = max(-1.0, min(1.0, offset))

    keep = FRAME_KEEP[frame]
    if keep < 1.0:
        img = img.crop((0, 0, img.width, max(1, round(img.height * keep))))

    scale = h / img.height
    nw = max(1, round(img.width * scale))
    resized = img.resize((nw, h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (w, h), pad)
    if nw >= w:
        # Figure wider than canvas: choose which horizontal window to keep. Sliding the
        # window left (smaller `left`) pushes the centred subject to the right.
        free = nw - w
        left = round((free / 2) * (1 - offset))
        left = min(max(left, 0), free)
        canvas.paste(resized.crop((left, 0, left + w, h)), (0, 0))
    else:
        # Figure narrower than canvas: place it; larger `x` moves the subject right.
        free = w - nw
        x = round((free / 2) * (1 + offset))
        x = min(max(x, 0), free)
        canvas.paste(resized, (x, 0))
    return canvas


def _parse_rgb(spec: str) -> tuple[int, int, int]:
    parts = spec.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad colour must be 'R,G,B'")
    r, g, b = (int(p) for p in parts)
    return (r, g, b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize a pose skeleton / reference photo "
                                             "to the gen canvas for ControlNet.")
    ap.add_argument("image", help="input skeleton map or pose photo")
    ap.add_argument("--out", default=None, help="output path (default: <stem>_prep.png)")
    ap.add_argument("--width", type=int, default=768, help="canvas width (default 768)")
    ap.add_argument("--height", type=int, default=1344, help="canvas height (default 1344)")
    ap.add_argument("--frame", choices=tuple(FRAME_KEEP), default="full",
                    help="vertical reframe: full | cowboy (mid-thigh up) | portrait (waist up)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="horizontal subject shift in [-1,1]; 0 centred, +right, -left")
    ap.add_argument("--pad-color", type=_parse_rgb, default=(0, 0, 0),
                    help="letterbox colour 'R,G,B' (default 0,0,0 — matches skeleton bg)")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_prep.png")
    prep(Image.open(src), args.width, args.height, args.frame, args.offset,
         args.pad_color).save(out)
    print(out)


if __name__ == "__main__":
    main()
