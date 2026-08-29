#!/usr/bin/env python3
"""Build A/B contact sheets from a p4_eval_round manifest.

One row per brief: arm A (baseline) left, arm B (detail-pass + composition) right,
labeled with brief index, seed, and prompt. Rows with a failed or missing arm are
skipped and reported. Long rounds split into numbered sheets (--max-rows).

  python3 scripts/p4_contact_sheet.py <stamp>            # sheets into the stamp dir
  python3 scripts/p4_contact_sheet.py <stamp> --thumb-w 512 --max-rows 3

Local-only (PIL): no GPU, no network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT / "outputs" / "_analysis" / "p4_eval"

LABEL_H = 28
HEADER_H = 34
PAD = 8


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def pair_rows(manifest: dict) -> tuple[list[dict], list[str]]:
    """Group manifest results into displayable A/B rows; report unpairable briefs."""
    by_idx: dict[int, dict[str, dict]] = {}
    for r in manifest["results"]:
        by_idx.setdefault(r["idx"], {})[r["arm"]] = r
    rows, skipped = [], []
    for idx in sorted(by_idx):
        arms = by_idx[idx]
        a, b = arms.get("A"), arms.get("B")
        if not (a and b and a["rc"] == 0 and b["rc"] == 0 and a["new_files"] and b["new_files"]):
            skipped.append(f"brief {idx:02d}: incomplete pair "
                           f"(A rc={a['rc'] if a else '-'}, B rc={b['rc'] if b else '-'})")
            continue
        rows.append({"idx": idx, "seed": a["seed"], "prompt": a["prompt"],
                     "a_path": a["new_files"][0], "b_path": b["new_files"][0]})
    return rows, skipped


def render_sheet(rows: list[dict], thumb_w: int, title: str) -> Image.Image:
    thumbs = []
    for row in rows:
        pair = []
        for path in (row["a_path"], row["b_path"]):
            im = Image.open(path)
            h = round(im.height * thumb_w / im.width)
            pair.append(im.resize((thumb_w, h), Image.LANCZOS))
        thumbs.append(pair)

    row_hs = [max(a.height, b.height) + LABEL_H + PAD for a, b in thumbs]
    width = thumb_w * 2 + PAD * 3
    sheet = Image.new("RGB", (width, HEADER_H + sum(row_hs) + PAD), "#181818")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 8), f"{title}    A = baseline | B = +detailers face,eye,hand +composition",
              fill="#e0e0e0", font=_font(16))

    y = HEADER_H
    for row, (ta, tb), row_h in zip(rows, thumbs, row_hs):
        sheet.paste(ta, (PAD, y))
        sheet.paste(tb, (thumb_w + PAD * 2, y))
        label = f"[{row['idx']:02d}] seed {row['seed']} — {row['prompt']}"
        draw.text((PAD, y + row_h - LABEL_H + 4), label[:110], fill="#b0b0b0", font=_font(13))
        y += row_h
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B contact sheets from a p4_eval manifest.")
    ap.add_argument("stamp", help="stamp dir name under outputs/_analysis/p4_eval/")
    ap.add_argument("--thumb-w", type=int, default=512, help="thumbnail width per arm (default 512)")
    ap.add_argument("--max-rows", type=int, default=3, help="briefs per sheet (default 3)")
    args = ap.parse_args()

    base = EVAL_ROOT / args.stamp
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        ap.error(f"no manifest at {manifest_path}")
    rows, skipped = pair_rows(json.loads(manifest_path.read_text()))
    for s in skipped:
        print(f"[skip] {s}")
    if not rows:
        raise SystemExit("[sheet] no complete A/B pairs to render")

    chunks = [rows[i:i + args.max_rows] for i in range(0, len(rows), args.max_rows)]
    for n, chunk in enumerate(chunks, start=1):
        title = f"p4_eval {args.stamp} — sheet {n}/{len(chunks)}"
        out = base / f"contact_A_B_{n}.png"
        render_sheet(chunk, args.thumb_w, title).save(out)
        print(f"[sheet] {out} ({len(chunk)} pairs)")


if __name__ == "__main__":
    main()
