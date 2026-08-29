#!/usr/bin/env python3
"""Build numbered review contact sheets for toe-count review.

Big index labels (lesson: small cropped numbers don't match; whole-image view).
Labels are drawn OUTSIDE the image area on a fixed gutter so vision can't read
them as picture content.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(paths: list[Path], out: Path, cols: int = 4,
          cell_w: int = 520, label_h: int = 56) -> None:
    font = load_font(44)
    rows = (len(paths) + cols - 1) // cols
    with Image.open(paths[0]) as im0:
        ratio = im0.height / im0.width
    cell_h = int(cell_w * ratio)
    W, H = cols * cell_w + (cols + 1) * 8, rows * (cell_h + label_h) + (rows + 1) * 8
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x = 8 + c * (cell_w + 8)
        y = 8 + r * (cell_h + label_h + 8)
        with Image.open(p) as im:
            canvas.paste(im.resize((cell_w, cell_h)), (x, y))
        # label in its own gutter band below the image — never over pixels
        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + label_h],
                       fill=(24, 24, 24))
        tag = f"#{i+1}"
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_w - tw) // 2, y + cell_h + 4), tag,
                  fill=(255, 210, 60), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"{out}  ({len(paths)} cells, #{1}-#{len(paths)})")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    *srcs, dst = args
    sheet(srcs, dst)
