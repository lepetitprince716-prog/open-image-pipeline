#!/usr/bin/env python3
"""Contact sheet: tile N images into one numbered montage PNG so a human (or a multimodal model) can
glance the whole batch and pick by eye. Pure PIL — no GPU.

This is the surfacing half of the artistic line's selection: scalar aesthetic scorers don't separate
"artistic" from "competent render" (verified — both dbaesthetic and aesthetic-predictor-v2.5 rank a
flat baseline above the more artistic candidates), so the artistic intent FILTERS by defects/identity
and then surfaces the survivors here for a by-eye pick rather than auto-crowning by a misleading score.

Usage:
  contact_sheet.py a.png b.png c.png --out sheet.png
  contact_sheet.py outputs/.../*.png --out sheet.png --cols 3 --thumb 512
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_H = 24  # px strip above each thumbnail for the index/name
PAD = 6


def _font() -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheet(paths: list[str], out: str, cols: int | None = None,
                thumb: int = 512, label: bool = True) -> str:
    """Tile images into a numbered grid montage; returns the output path. Missing images are skipped."""
    imgs = [(p, Image.open(p).convert("RGB")) for p in paths if Path(p).is_file()]
    if not imgs:
        raise ValueError("no readable images to tile")
    n = len(imgs)
    cols = cols or max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    cell_w, cell_h = thumb + 2 * PAD, thumb + 2 * PAD + (LABEL_H if label else 0)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    font = _font()

    for i, (path, im) in enumerate(imgs):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        if label:
            draw.text((x0 + PAD, y0 + 4), f"[{i}] {Path(path).stem[:40]}", fill=(235, 235, 235), font=font)
        im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        ox = x0 + PAD + (thumb - im.width) // 2
        oy = y0 + PAD + (LABEL_H if label else 0) + (thumb - im.height) // 2
        sheet.paste(im, (ox, oy))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile images into one numbered contact-sheet PNG for by-eye picking.")
    ap.add_argument("images", nargs="+", help="image paths to tile")
    ap.add_argument("--out", required=True, help="output montage path")
    ap.add_argument("--cols", type=int, default=None, help="columns (default: ~sqrt of count)")
    ap.add_argument("--thumb", type=int, default=512, help="max thumbnail edge px (default 512)")
    ap.add_argument("--no-label", dest="label", action="store_false", help="omit the index/name labels")
    args = ap.parse_args()
    try:
        out = build_sheet(args.images, args.out, args.cols, args.thumb, args.label)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    print(out)


if __name__ == "__main__":
    main()
