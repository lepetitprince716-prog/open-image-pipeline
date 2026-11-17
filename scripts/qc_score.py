#!/usr/bin/env python3
"""No-reference QC scorer / best-of-N selector for generated images (via the imgutils venv).

Runs the imgutils venv bridge `qc` op on each image and ranks them, so a `gen --n` batch can be
auto-filtered to the best candidate and obvious defects flagged — without a human eyeballing every
image. Read-only: it scores existing PNGs, never generates. The imgutils heavy lifting runs in the
external venv (same one image_recipe.py uses); override its interpreter with IMGUTILS_VENV_PY.

Ranking = `aesthetic_percentile` (0..1, calibrated against a reference distribution) — the
discriminative metric. The coarse aesthetic *label* (worst..masterpiece) saturates at 'best' for most
decent gens, so it is shown for readability but not ranked on. Defect FLAGS (surfaced, and avoided by
--best; gate-able with --strict):
  - completeness != polished  (sketchy / unfinished render)
  - monochrome >= 0.5         (accidental greyscale when colour was expected)
  - faces != --subjects N     (dropped / duplicated subject — faces is the reliable subject count)
  - hands  > 2*N + 1          (extra-limb tell)

HONEST LIMIT: this ranks aesthetic quality and flags gross structural problems. It does NOT judge
anatomy *correctness* (bad fingers) or character identity — for those use --subjects, CCIP (identity),
or a human. It is a best-of-N pre-filter, not a correctness oracle.

Usage:
  qc_score.py a.png b.png c.png                      # rank by percentile, print a table
  qc_score.py outputs/.../*.png --subjects 1 --best  # print the best UNFLAGGED path (for scripting)
  qc_score.py img.png --min 0.85                     # gate: exit 1 if best percentile < 0.85
  qc_score.py outputs/.../*.png --subjects 2 --strict --min 0.85   # gate also fails if best is flagged
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
# imgutils venv (Codex's comfyui新 tools), consumed read-only — same default as image_recipe.py.
DEFAULT_VENV_PY = "/opt/comfyui-boxa/tools/visual_features/.venv/bin/python"
MONO_THRESHOLD = 0.5  # monochrome score above this = effectively greyscale


def qc_one(venv: str, img: str) -> dict | None:
    """Run the bridge `qc` op on one image via the venv interpreter. None on any failure."""
    if not os.path.exists(venv):
        return None
    try:
        out = subprocess.run([venv, str(BRIDGE), "qc", img],
                             capture_output=True, text=True, timeout=300).stdout
        data = json.loads(out)
    except Exception:
        return None
    return data if data.get("ok") else None


def qc_many(venv: str, images: list[str]) -> dict[str, dict | None]:
    """Score a CLI batch with one bridge process; retain qc_one for direct legacy callers."""
    if len(images) == 1:
        return {images[0]: qc_one(venv, images[0])}
    if not os.path.exists(venv):
        return {image: None for image in images}
    try:
        out = subprocess.run(
            [venv, str(BRIDGE), "qc", *images], capture_output=True, text=True, timeout=600
        ).stdout
        data = json.loads(out)
    except Exception:
        return {image: None for image in images}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return {image: None for image in images}
    return {image: (results.get(image) if isinstance(results.get(image), dict)
                    and results[image].get("ok") else None) for image in images}


def flags_for(d: dict, subjects: int | None) -> list[str]:
    """Defect flags — problems the aesthetic percentile alone won't catch."""
    flags: list[str] = []
    if d["completeness"] != "polished":
        flags.append(f"incomplete:{d['completeness']}")
    if d["monochrome"] >= MONO_THRESHOLD:
        flags.append(f"monochrome={d['monochrome']:.2f}")
    if subjects is not None:
        if d["faces"] != subjects:
            flags.append(f"faces={d['faces']}!={subjects}")
        if d["hands"] > 2 * subjects + 1:
            flags.append(f"hands={d['hands']}")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(
        description="no-reference QC scorer / best-of-N selector for generated images (imgutils venv)")
    ap.add_argument("images", nargs="+", help="image paths to score (a gen --n batch, a glob, ...)")
    ap.add_argument("--subjects", type=int, default=None,
                    help="expected subject count -> anatomy-sanity flags (faces!=N, too many hands)")
    ap.add_argument("--best", action="store_true",
                    help="print only the best path (highest percentile among UNFLAGGED; falls back to "
                         "highest overall with a note if all are flagged)")
    ap.add_argument("--keep", action="store_true",
                    help="print every UNFLAGGED path (one per line) — a defect pre-filter for callers "
                         "that pick the winner themselves (aesthetic percentile is NOT used to rank)")
    ap.add_argument("--min", type=float, default=None,
                    help="gate: exit 1 if the best image's aesthetic percentile is below this (0..1)")
    ap.add_argument("--strict", action="store_true",
                    help="with --min: also fail the gate if the best image carries any defect flag")
    ap.add_argument("--venv", default=os.environ.get("IMGUTILS_VENV_PY", DEFAULT_VENV_PY),
                    help="imgutils venv python (default: the comfyui新 tools venv)")
    args = ap.parse_args()

    rows = []  # (percentile, flags, img, d)
    batch = qc_many(args.venv, args.images)
    for img in args.images:
        d = batch.get(img)
        if d is None:
            print(f"  SKIP (qc failed / venv missing): {img}", file=sys.stderr)
            continue
        rows.append((d["aesthetic_percentile"], flags_for(d, args.subjects), img, d))
    if not rows:
        sys.exit("no images scored — imgutils venv unavailable? set IMGUTILS_VENV_PY to its python.")
    rows.sort(key=lambda r: r[0], reverse=True)

    # best = highest percentile with no flags; fall back to highest overall (noted on stderr)
    unflagged = [r for r in rows if not r[1]]
    best = unflagged[0] if unflagged else rows[0]
    if args.keep:
        if not unflagged:
            print("  note: all candidates flagged; none kept", file=sys.stderr)
        for _, _, img, _ in unflagged:
            print(img)
    elif args.best:
        if not unflagged:
            print("  note: all candidates flagged; returning highest percentile overall", file=sys.stderr)
        print(best[2])
    else:
        for pct, flags, img, d in rows:
            tail = ("  flags: " + ", ".join(flags)) if flags else ""
            print(f"{pct:6.3f}  {d['aesthetic_label']:<11} {d['completeness']:<10} "
                  f"faces={d['faces']} hands={d['hands']}  {Path(img).name}{tail}")

    if args.min is not None:
        if best[0] < args.min:
            sys.exit(f"[gate] best percentile {best[0]:.3f} < --min {args.min}")
        if args.strict and best[1]:
            sys.exit(f"[gate] best image is flagged ({', '.join(best[1])}) and --strict is set")


if __name__ == "__main__":
    main()
