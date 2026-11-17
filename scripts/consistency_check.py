#!/usr/bin/env python3
"""Reference-based identity consistency check for a batch of generated images (via imgutils CCIP).

Given an ANCHOR face image and N generated images (the same OC across varied scenes), score each
candidate's character identity against the anchor with imgutils CCIP and report which images HOLD the
identity and which have DRIFTED. This is the verification half of a "character across a sequence" run:
generation (character LoRA + IPAdapter face-lock + wildcard scenes) holds identity by construction, and
this confirms it numerically and surfaces the drifters. Read-only: it scores existing PNGs, never
generates. Distinct from qc_score.py, which scores no-reference *aesthetic* quality — this scores
reference-based *identity*.

CCIP semantics (imgutils, via imgutils_bridge `ccip` op): `difference` is a distance, LOWER = more
similar; `same_character` = difference < the library threshold. The heavy lifting runs in the external
imgutils venv (the same one qc_score.py / image_recipe.py use); override it with IMGUTILS_VENV_PY.

Usage:
  consistency_check.py --ref anchor.png a.png b.png c.png        # table: per-image difference + HOLD/DRIFT
  consistency_check.py --ref anchor.png outputs/.../*.png --keep # print the held subset, one path per line
  consistency_check.py --ref anchor.png imgs... --best           # print the single most-consistent path
  consistency_check.py --ref anchor.png imgs... --max-diff 0.16  # stricter than the library threshold
  consistency_check.py --ref anchor.png imgs... --min-keep 3     # gate: exit 1 if fewer than 3 images hold
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BRIDGE = ROOT / "imgutils_bridge.py"
# imgutils venv (Codex's comfyui新 tools), consumed read-only — same default as qc_score.py / image_recipe.py.
DEFAULT_VENV_PY = "/opt/comfyui-boxa/tools/visual_features/.venv/bin/python"


def ccip_one(venv: str, ref: str, img: str) -> dict | None:
    """Run the bridge `ccip` op for (ref, img) via the venv interpreter. None on any failure."""
    if not os.path.exists(venv):
        return None
    try:
        out = subprocess.run([venv, str(BRIDGE), "ccip", ref, img],
                             capture_output=True, text=True, timeout=300).stdout
        data = json.loads(out)
    except Exception:
        return None
    return data if data.get("ok") else None


def ccip_many(venv: str, ref: str, images: list[str]) -> dict[str, dict | None]:
    """Compare one anchor with a candidate batch using one bridge process."""
    if len(images) == 1:
        return {images[0]: ccip_one(venv, ref, images[0])}
    if not os.path.exists(venv):
        return {image: None for image in images}
    try:
        out = subprocess.run(
            [venv, str(BRIDGE), "ccip", ref, *images],
            capture_output=True, text=True, timeout=600,
        ).stdout
        data = json.loads(out)
    except Exception:
        return {image: None for image in images}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return {image: None for image in images}
    return {image: (results.get(image) if isinstance(results.get(image), dict)
                    and results[image].get("ok") else None) for image in images}


def held(d: dict, max_diff: float | None) -> bool:
    """True if the candidate keeps the anchor's identity (custom max-diff overrides the library threshold)."""
    return d["difference"] <= max_diff if max_diff is not None else bool(d["same_character"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="reference-based identity consistency check for a batch of images (imgutils CCIP)")
    ap.add_argument("images", nargs="+", help="candidate images to check against --ref")
    ap.add_argument("--ref", required=True, help="anchor / identity reference image (the canonical face)")
    ap.add_argument("--best", action="store_true",
                    help="print only the single most-consistent path (lowest difference that still held)")
    ap.add_argument("--keep", action="store_true",
                    help="print every path that HELD identity (one per line), for scripting")
    ap.add_argument("--max-diff", type=float, default=None,
                    help="override the library same-character threshold (lower = stricter)")
    ap.add_argument("--min-keep", type=int, default=None,
                    help="gate: exit 1 if fewer than N images hold identity")
    ap.add_argument("--venv", default=os.environ.get("IMGUTILS_VENV_PY", DEFAULT_VENV_PY),
                    help="imgutils venv python (default: the comfyui新 tools venv)")
    args = ap.parse_args()

    rows = []  # (difference, held_bool, img, d)
    batch = ccip_many(args.venv, args.ref, args.images)
    for img in args.images:
        d = batch.get(img)
        if d is None:
            print(f"  SKIP (ccip failed / venv missing): {img}", file=sys.stderr)
            continue
        rows.append((d["difference"], held(d, args.max_diff), img, d))
    if not rows:
        sys.exit("no images scored — imgutils venv unavailable? set IMGUTILS_VENV_PY to its python.")
    rows.sort(key=lambda r: r[0])  # ascending difference = most consistent first
    kept = [r for r in rows if r[1]]

    if args.best:
        print((kept[0] if kept else rows[0])[2])  # most-consistent that held; else closest overall
    elif args.keep:
        for r in kept:
            print(r[2])
    else:
        thr = rows[0][3]["threshold"]
        head = f"anchor: {Path(args.ref).name}   threshold: {thr:.4f}"
        print(head + (f"  (override max-diff {args.max_diff})" if args.max_diff is not None else ""))
        for diff, ok, img, _ in rows:
            print(f"{diff:6.4f}  {'HOLD ' if ok else 'DRIFT'}  {Path(img).name}")
        print(f"\n{len(kept)}/{len(rows)} held identity.")

    if args.min_keep is not None and len(kept) < args.min_keep:
        sys.exit(f"[gate] only {len(kept)} image(s) held identity (< --min-keep {args.min_keep})")


if __name__ == "__main__":
    main()
