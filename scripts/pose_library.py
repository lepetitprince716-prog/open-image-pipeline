#!/usr/bin/env python3
"""Pose library: turn artistic reference photos into a reusable, catalogued collection of
DWPose skeletons (body + hands + face) normalized to the generation canvas.

Why this exists
---------------
The "特别感" line locks composition with an OpenPose ControlNet, and clean hands need hand
keypoints *in the control map*. A pose preprocessor extracts body + 21-point hands + face
from a reference photo. This tool runs that extraction ONCE, normalizes the skeleton to the
gen canvas (default 768x1344 via pose_prep), and stores it with metadata + a catalog index —
so future gens reuse a curated pose instead of re-extracting every time:

    boxa_quality_gen.py --control openpose --control-image <lib>/<slug>.png --no-preprocess

Extractor
---------
`--extractor openpose` (default) uses OpenposePreprocessor — its weights are cached on BOXB, so
it works there. `--extractor dwpose` (DWPreprocessor) locates finger joints more precisely but
downloads its weights from HuggingFace on first use, which is BLOCKED on BOXB (the download
client is closed mid-request) — use it only on a box that already has the DWPose weights.

Sourcing
--------
`add` accepts either a reference photo (default: DWPose extracts the skeleton) or an
already-made skeleton map (`--from-skeleton`: skip DWPose, just normalize + catalog). Drop
artistic inspiration photos into <library>/_inbox/ and `add` them one at a time.

Host
----
Reads BOXA_HOST. Point it at BOXB, which has DWPreprocessor:
    BOXA_HOST=http://192.0.2.20:8388 python scripts/pose_library.py add photo.jpg --name reach_up

Pairing metadata
----------------
Each pose may carry pose<->lighting<->composition pairing suggestions (`pairing`: lists of prompt
phrases per slot + avoid lists), an `energy` class and a `canvas` hint. `gen.py artistic
--pose-ref SLUG` auto-fills open slots from `pairing` (as {a|b} wildcard groups); `deploy` prints
the suggestions. Backfill/adjust with `annotate` (pure JSON edit, no GPU) — pairing flags are
REPEATABLE, one full phrase per occurrence (phrases contain commas).

Prompt fields vs note fields
----------------------------
`prompt_hint` (and the `pairing` suggestion lists) are the ONLY free text `gen.py --pose-ref`
injects into prompts — tag-style English pose phrases. `desc` and `pairing.notes` are human-facing
(status notes, variant provenance, re-extract TODOs) and are NEVER injected; desc once leaked
'[PARTIAL — ...]' maintenance notes into gen prompts, hence the split.
Canonical hints live in the tracked `pose_prompt_hints.json`; `backfill-hints` restores them into
the local, gitignored catalog after a rebuild without overwriting a conflicting non-empty value.

Commands
--------
  add <image> --name SLUG [--tags a,b] [--desc ...] [--prompt-hint ...]
              [--frame full|cowboy|portrait]
              [--offset F] [--from-skeleton] [--resolution N] [--keep-raw]
              [--pair-lighting P ...] [--pair-camera P ...] [--energy E] [--canvas C] ...
  annotate SLUG [--pair-* P ...] [--avoid-* P ...] [--pair-notes S] [--energy E] [--canvas C]
              [--desc S] [--prompt-hint S] [--tags a,b]   # prints the diff, then writes atomically
  backfill-hints                # atomically fill empty prompt_hint values from the tracked manifest
  list                         # print the catalog ('paired' marks poses with pairing metadata)
  sheet                        # contact sheet of every skeleton for eyeballing
  deploy SLUG [--json]         # upload skeleton + print the ready gen command & pairing suggestions
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS, ROOT  # noqa: E402
from boxa_comfy import (BOXA_HOST, curl, upload_image, looks_like_png, poll_delay,  # sibling stdlib driver
                      POLL_TIMEOUT, free_vram as _shared_free_vram)
from pose_prep import prep  # sibling; canvas normalization (fit-by-height + reframe + offset)

LIB = OUTPUTS / "poses" / "library"
INBOX = LIB / "_inbox"
INDEX = LIB / "index.json"
SHEET = LIB / "_contact_sheet.png"
PROMPT_HINTS = Path(__file__).with_name("pose_prompt_hints.json")
CANVAS = (768, 1344)
EXTRACTORS = {"dwpose": "DWPreprocessor", "openpose": "OpenposePreprocessor"}
# pairing slots consumed by gen.py --pose-ref (suggest lists) + human-guidance avoid lists.
PAIR_KEYS = ("lighting", "camera", "composition", "color_mood", "expression")
AVOID_KEYS = ("avoid_camera", "avoid_lighting")
ENERGIES = ("static", "balanced", "dynamic")
CANVASES = ("9:16", "1:1")


# --------------------------------------------------------------------------- catalog

def _slugify(name: str) -> str:
    """Filesystem-safe slug: lowercase, [a-z0-9_-] only, collapsed separators."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not s:
        raise SystemExit(f"empty slug from name: {name!r}")
    return s


def _load_index() -> dict:
    if INDEX.is_file():
        try:
            return json.loads(INDEX.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"corrupt index {INDEX}: {exc}") from exc
    return {"version": 1, "canvas": list(CANVAS), "poses": []}


def _save_index(idx: dict) -> None:
    tmp = INDEX.with_name(INDEX.name + ".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX)


def _upsert(idx: dict, entry: dict) -> dict:
    """Return a new index with `entry` replacing any existing pose of the same slug."""
    poses = [p for p in idx.get("poses", []) if p["slug"] != entry["slug"]]
    poses.append(entry)
    poses.sort(key=lambda p: p["slug"])
    return {**idx, "poses": poses}


def _apply_annotation(entry: dict, updates: dict) -> dict:
    """Pure merge (no IO, unit-testable): pairing/avoid keys replace PER KEY (an unset key keeps
    the old value); energy/canvas/desc/prompt_hint/tags replace wholesale when given."""
    out = {**entry}
    pairing = {**(entry.get("pairing") or {})}
    for k, v in (updates.get("pairing") or {}).items():
        if v in (None, [], ""):
            continue
        pairing[k] = v
    if pairing:
        out["pairing"] = pairing
    for k in ("energy", "canvas", "desc", "prompt_hint", "tags"):
        v = updates.get(k)
        if v not in (None, "", []):
            out[k] = v
    return out


def _validate_prompt_hint_manifest(raw: object) -> dict[str, str]:
    """Return a validated slug->hint mapping from the tracked manifest document."""
    if not isinstance(raw, dict) or raw.get("schema") != "loramake.pose-prompt-hints/v1":
        raise ValueError("prompt-hint manifest schema must be loramake.pose-prompt-hints/v1")
    hints = raw.get("hints")
    if not isinstance(hints, dict) or not hints:
        raise ValueError("prompt-hint manifest must contain a non-empty hints object")
    out: dict[str, str] = {}
    for slug, hint in hints.items():
        if not isinstance(slug, str) or _slugify(slug) != slug:
            raise ValueError(f"invalid prompt-hint slug: {slug!r}")
        if not isinstance(hint, str) or not hint.strip():
            raise ValueError(f"{slug}: prompt hint must be a non-empty string")
        if hint != hint.strip():
            raise ValueError(f"{slug}: prompt hint must not have surrounding whitespace")
        if not hint.isascii():
            raise ValueError(f"{slug}: prompt hint must be ASCII")
        if "[" in hint or "]" in hint:
            raise ValueError(f"{slug}: bracketed maintenance notes belong in desc")
        out[slug] = hint
    return out


def _load_prompt_hint_manifest(path: Path = PROMPT_HINTS) -> dict[str, str]:
    return _validate_prompt_hint_manifest(json.loads(path.read_text(encoding="utf-8")))


def _apply_prompt_hint_manifest(poses: list[dict], hints: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Pure, all-or-nothing backfill. Existing non-empty values must match the manifest."""
    slugs = [p.get("slug") for p in poses]
    if len(slugs) != len(set(slugs)):
        raise ValueError("pose index contains duplicate slugs")
    pose_slugs = set(slugs)
    hint_slugs = set(hints)
    missing = sorted(pose_slugs - hint_slugs)
    extra = sorted(hint_slugs - pose_slugs)
    if missing or extra:
        raise ValueError(f"prompt-hint manifest/catalog mismatch: missing={missing}, extra={extra}")

    conflicts = sorted(
        p["slug"] for p in poses
        if p.get("prompt_hint", "").strip() and p.get("prompt_hint") != hints[p["slug"]]
    )
    if conflicts:
        raise ValueError(f"existing prompt hints conflict with manifest: {conflicts}")

    changed: list[str] = []
    updated: list[dict] = []
    for entry in poses:
        slug = entry["slug"]
        if entry.get("prompt_hint", "").strip():
            updated.append({**entry})
        else:
            updated.append({**entry, "prompt_hint": hints[slug]})
            changed.append(slug)
    return updated, changed


def _annotation_updates(args: argparse.Namespace, include_desc_tags: bool = False) -> dict:
    """Collect the pairing/energy/canvas (and, for annotate, desc/tags) updates off a namespace."""
    pairing: dict = {}
    for k in PAIR_KEYS:
        v = getattr(args, f"pair_{k}", None)
        if v:
            pairing[k] = v
    for k in AVOID_KEYS:
        v = getattr(args, k, None)
        if v:
            pairing[k] = v
    if getattr(args, "pair_notes", ""):
        pairing["notes"] = args.pair_notes
    upd: dict = {"pairing": pairing, "energy": getattr(args, "energy", None),
                 "canvas": getattr(args, "canvas", None)}
    if include_desc_tags:  # `add` already consumed desc/prompt_hint/tags for the entry itself
        upd["desc"] = getattr(args, "desc", "")
        upd["prompt_hint"] = getattr(args, "prompt_hint", "")
        upd["tags"] = [t.strip() for t in getattr(args, "tags", "").split(",") if t.strip()]
    return upd


# --------------------------------------------------------------------------- DWPose extraction

def _extract_graph(image_name: str, resolution: int, node: str) -> dict:
    """Minimal ComfyUI graph: load photo -> pose preprocessor (hands+body+face) -> save skeleton."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "2": {"class_type": node, "inputs": {
            "image": ["1", 0], "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": resolution}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "poselib"}},
    }


def _run_graph(graph: dict) -> bytes | None:
    """Submit a graph to the box, poll history, return the first output PNG's bytes (None on failure)."""
    payload = json.dumps({"prompt": graph})
    out = curl(["-X", "POST", f"{BOXA_HOST}/prompt", "-H", "Content-Type: application/json",
                "--data-binary", "@-"], data=payload)
    try:
        pid = json.loads(out)["prompt_id"]
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"  submit FAIL: {out[:400]}", file=sys.stderr)
        return None
    print(f"  extract pid={pid} ...", flush=True)
    start = time.monotonic()
    errs = 0
    while time.monotonic() - start < POLL_TIMEOUT:
        try:
            hj = json.loads(curl([f"{BOXA_HOST}/history/{pid}"]))
            errs = 0
        except (json.JSONDecodeError, TypeError) as exc:
            errs += 1
            if errs == 5:
                print(f"    WARN: 5 consecutive /history errors ({exc!r}) — box may be unreachable",
                      file=sys.stderr)
            time.sleep(poll_delay(time.monotonic() - start))
            continue
        if pid in hj and hj[pid].get("outputs"):
            for node in hj[pid]["outputs"].values():
                for im in node.get("images", []):
                    raw = curl(["-G", f"{BOXA_HOST}/view",
                                "--data-urlencode", f"filename={im['filename']}",
                                "--data-urlencode", f"subfolder={im.get('subfolder', '')}",
                                "--data-urlencode", f"type={im.get('type', 'output')}"], binary=True)
                    if looks_like_png(raw):
                        return raw
            print("    extract produced no valid PNG", file=sys.stderr)
            return None
        if hj.get(pid, {}).get("status", {}).get("status_str") == "error":
            print("    extract ERROR (check ComfyUI log)", file=sys.stderr)
            return None
        time.sleep(poll_delay(time.monotonic() - start))
    print(f"    TIMEOUT after {int(time.monotonic() - start)}s", file=sys.stderr)
    return None


def _free() -> None:
    """Best-effort VRAM release after loading the DWPose models (standing op on the shared worker)."""
    _shared_free_vram(BOXA_HOST)


# --------------------------------------------------------------------------- commands

def cmd_add(args: argparse.Namespace) -> None:
    src = Path(args.image)
    if not src.is_file():
        # allow bare filename resolved against the inbox for convenience
        cand = INBOX / args.image
        if cand.is_file():
            src = cand
        else:
            raise SystemExit(f"not a file: {args.image} (also tried {cand})")
    slug = _slugify(args.name)
    LIB.mkdir(parents=True, exist_ok=True)

    if args.from_skeleton:
        skel = Image.open(src)
        source = f"skeleton:{src.name}"
        detect: dict | None = None
        extractor = None
    else:
        server_name = upload_image(str(src))
        if not server_name:
            raise SystemExit(f"failed to upload {src} to {BOXA_HOST}")
        used = args.extractor
        raw = _run_graph(_extract_graph(server_name, args.resolution, EXTRACTORS[used]))
        if not raw and used == "dwpose":
            # DWPose weights are HF-hosted and its download client is flaky; OpenPose is cached.
            print("  DWPose failed — falling back to OpenPose", file=sys.stderr)
            used = "openpose"
            raw = _run_graph(_extract_graph(server_name, args.resolution, EXTRACTORS[used]))
        _free()
        if not raw:
            raise SystemExit(f"pose extraction failed for {src} (extractor={args.extractor}; "
                             f"is BOXA_HOST pointed at a box running the preprocessor?)")
        if args.keep_raw:
            (LIB / f"{slug}.skel_raw.png").write_bytes(raw)
        skel = Image.open(io.BytesIO(raw))
        source = f"photo:{src.name}"
        detect = {"hand": True, "body": True, "face": True}
        extractor = used

    norm = prep(skel, args.width, args.height, args.frame, args.offset, (0, 0, 0))
    out_path = LIB / f"{slug}.png"
    norm.save(out_path)

    # Verify the file decodes at the target size before we commit it to the catalog.
    check = Image.open(out_path)
    if check.size != (args.width, args.height):
        raise SystemExit(f"normalized skeleton is {check.size}, expected {(args.width, args.height)}")

    entry = {
        "slug": slug,
        "file": str(out_path.relative_to(ROOT)),
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else [],
        "desc": args.desc or "",
        "prompt_hint": args.prompt_hint or "",
        "frame": args.frame,
        "offset": args.offset,
        "source": source,
        "source_url": args.source_url or "",
        "license": args.license or "",
        "extractor": extractor,
        "detect": detect,
        "dims": [args.width, args.height],
        "added": date.today().isoformat(),
    }
    entry = _apply_annotation(entry, _annotation_updates(args))  # optional pairing/energy/canvas at add time
    _save_index(_upsert(_load_index(), entry))
    print(f"added '{slug}' -> {entry['file']} ({args.width}x{args.height}, frame={args.frame}, source={source})")


def cmd_annotate(args: argparse.Namespace) -> None:
    """Backfill/adjust a catalogued pose's pairing metadata (pure JSON edit — no GPU, no re-extract)."""
    slug = _slugify(args.slug)
    idx = _load_index()
    entry = next((p for p in idx.get("poses", []) if p["slug"] == slug), None)
    if entry is None:
        raise SystemExit(f"unknown pose: {slug} (see: pose_library.py list)")
    updates = _annotation_updates(args, include_desc_tags=True)
    new_entry = _apply_annotation(entry, updates)
    if new_entry == entry:
        raise SystemExit("nothing to annotate (pass --pair-*/--avoid-*/--pair-notes/--energy/--canvas"
                         "/--desc/--prompt-hint/--tags)")
    for k in sorted(set(entry) | set(new_entry)):  # show the diff before committing it
        if entry.get(k) != new_entry.get(k):
            print(f"  {k}: {entry.get(k)!r} -> {new_entry.get(k)!r}")
    _save_index(_upsert(idx, new_entry))
    print(f"annotated '{slug}' -> {INDEX.relative_to(ROOT)}")


def cmd_backfill_hints(_args: argparse.Namespace) -> None:
    """Fill all empty prompt hints from a tracked source without partial catalog writes."""
    if not INDEX.is_file():
        raise SystemExit(f"pose library index not found: {INDEX}")
    try:
        hints = _load_prompt_hint_manifest()
        idx = _load_index()
        poses, changed = _apply_prompt_hint_manifest(idx.get("poses", []), hints)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"prompt-hint backfill refused: {exc}") from exc
    if not changed:
        print(f"prompt hints already current ({len(hints)} poses)")
        return
    for slug in changed:
        print(f"  {slug}: '' -> {hints[slug]!r}")
    _save_index({**idx, "poses": poses})
    print(f"backfilled {len(changed)} prompt hint(s) -> {INDEX.relative_to(ROOT)}")


def cmd_list(_args: argparse.Namespace) -> None:
    idx = _load_index()
    poses = idx.get("poses", [])
    if not poses:
        print(f"(empty) — no poses in {INDEX}")
        return
    print(f"{len(poses)} pose(s) in {LIB.relative_to(ROOT)}:")
    for p in poses:
        hands = "hands" if (p.get("detect") or {}).get("hand") else "no-hands"
        tags = ",".join(p.get("tags", [])) or "-"
        paired = "paired" if p.get("pairing") else "-"
        print(f"  {p['slug']:<24} {p['frame']:<8} {hands:<9} {paired:<7} [{tags}]  {p.get('desc', '')}")


def cmd_sheet(_args: argparse.Namespace) -> None:
    idx = _load_index()
    poses = idx.get("poses", [])
    if not poses:
        raise SystemExit("no poses to sheet")
    cols = min(6, len(poses))
    rows = (len(poses) + cols - 1) // cols
    tw, th, pad, label_h = 180, 315, 10, 18
    cw, chh = tw + pad * 2, th + pad * 2 + label_h
    sheet = Image.new("RGB", (cols * cw, rows * chh), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for i, p in enumerate(poses):
        r, c = divmod(i, cols)
        x0, y0 = c * cw, r * chh
        fp = ROOT / p["file"]
        if fp.is_file():
            thumb = Image.open(fp).convert("RGB")
            thumb.thumbnail((tw, th))
            sheet.paste(thumb, (x0 + pad + (tw - thumb.width) // 2, y0 + pad))
        draw.text((x0 + pad, y0 + pad + th + 4), p["slug"][:26], fill=(220, 220, 220), font=font)
    sheet.save(SHEET)
    print(f"contact sheet -> {SHEET.relative_to(ROOT)} ({cols}x{rows}, {len(poses)} poses)")


def cmd_deploy(args: argparse.Namespace) -> None:
    slug = _slugify(args.slug)
    fp = LIB / f"{slug}.png"
    if not fp.is_file():
        raise SystemExit(f"unknown pose: {slug} ({fp} missing)")
    entry = next((p for p in _load_index().get("poses", []) if p["slug"] == slug), None)
    if args.json:  # tool-consumable dump (offline; no upload)
        print(json.dumps(entry or {"slug": slug, "file": str(fp.relative_to(ROOT))},
                         ensure_ascii=False, indent=2))
        return
    name = upload_image(str(fp))
    if not name:
        raise SystemExit(f"failed to upload {fp} to {BOXA_HOST}")
    print(f"uploaded '{slug}' to {BOXA_HOST} as {name}\n")
    print("ready gen command (skeleton is pre-made -> --no-preprocess).")
    print("finalized recipe (A/B-verified 2026-07-01): iLustMix locks non-standing poses, so run on")
    print("BOXB (192.0.2.20:8388); --res MUST match the 9:16 skeleton or the control map stretches")
    print("and duplicate faces appear. Add duplicate-face + SFW negatives per LIBRARY.md.")
    res = "1024x1024" if entry and (entry.get("dims") or [768, 1344])[0] == (entry.get("dims") or [768, 1344])[1] \
        else "832x1472"
    print(f"  BOXA_HOST=http://192.0.2.20:8388 python scripts/boxa_quality_gen.py --family illustrious --max \\")
    print(f"    --lora 'character_ilust_v1-000007.safetensors:0.82' \\")
    print(f"    --control openpose --control-image '{fp.relative_to(ROOT)}' --no-preprocess \\")
    print(f"    --control-strength 1.0 --control-end 1.0 --res {res} --detailers face")
    pairing = (entry or {}).get("pairing") or {}
    if pairing:
        print("\npairing suggestions (gen.py --pose-ref injects these automatically; --no-pair disables):")
        for k in (*PAIR_KEYS, *AVOID_KEYS, "notes"):
            v = pairing.get(k)
            if v:
                print(f"  {k:<14} {v if isinstance(v, str) else ' | '.join(v)}")
        print("\nor via the router (recipe + pairing auto-injected):")
        print(f"  python3 scripts/gen.py artistic --box boxb --pose-ref {slug} \\")
        print(f"    --prompt '1girl, solo, <identity tags>' --n 6 --sfw")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="extract + normalize + catalog one pose")
    a.add_argument("image", help="reference photo, or a skeleton map with --from-skeleton "
                                 "(bare name resolves against _inbox/)")
    a.add_argument("--name", required=True, help="catalog slug")
    a.add_argument("--tags", default="", help="comma-separated tags")
    a.add_argument("--desc", default="", help="one-line HUMAN description (status notes / variant "
                                              "provenance welcome; never injected into prompts)")
    a.add_argument("--prompt-hint", dest="prompt_hint", default="",
                   help="pose phrase gen.py --pose-ref INJECTS into the prompt (tag-style English "
                        "only — no maintenance notes, those belong in --desc)")
    a.add_argument("--frame", choices=("full", "cowboy", "portrait"), default="full")
    a.add_argument("--offset", type=float, default=0.0, help="horizontal shift in [-1,1]")
    a.add_argument("--from-skeleton", action="store_true",
                   help="input is already a skeleton map; skip extraction, just normalize")
    a.add_argument("--extractor", choices=tuple(EXTRACTORS), default="dwpose",
                   help="pose preprocessor: dwpose (precise hands, default; auto-falls back to "
                        "openpose on failure) | openpose (cached, robust)")
    a.add_argument("--resolution", type=int, default=1024, help="detect resolution (default 1024)")
    a.add_argument("--keep-raw", action="store_true", help="also save the pre-normalize skeleton")
    a.add_argument("--source-url", default="", help="provenance: URL the reference photo came from")
    a.add_argument("--license", default="", help="provenance: reference photo license (e.g. CC-BY-SA-4.0)")
    a.add_argument("--width", type=int, default=CANVAS[0])
    a.add_argument("--height", type=int, default=CANVAS[1])
    _add_annotation_flags(a)
    a.set_defaults(func=cmd_add)

    an = sub.add_parser("annotate", help="backfill/adjust pairing metadata (pure JSON edit, no GPU)")
    an.add_argument("slug", help="catalog slug to annotate")
    an.add_argument("--desc", default="", help="replace the one-line human description (never injected)")
    an.add_argument("--prompt-hint", dest="prompt_hint", default="",
                    help="replace the injected pose phrase (see add --prompt-hint)")
    an.add_argument("--tags", default="", help="replace the comma-separated tags")
    _add_annotation_flags(an)
    an.set_defaults(func=cmd_annotate)

    sub.add_parser("backfill-hints", help="atomically fill empty prompt hints from the tracked "
                   "pose_prompt_hints.json manifest").set_defaults(func=cmd_backfill_hints)

    sub.add_parser("list", help="print the catalog").set_defaults(func=cmd_list)
    sub.add_parser("sheet", help="build a contact sheet of all skeletons").set_defaults(func=cmd_sheet)

    d = sub.add_parser("deploy", help="upload a skeleton and print the ready gen command + pairing")
    d.add_argument("slug", help="catalog slug to deploy")
    d.add_argument("--json", action="store_true", help="print the catalog entry as JSON (offline; no upload)")
    d.set_defaults(func=cmd_deploy)
    return ap


def _add_annotation_flags(p: argparse.ArgumentParser) -> None:
    """Pairing flags shared by add/annotate. REPEATABLE — one full phrase per occurrence
    (phrases contain commas, so comma-splitting would shred them)."""
    for k in PAIR_KEYS:
        p.add_argument(f"--pair-{k.replace('_', '-')}", dest=f"pair_{k}", action="append", default=None,
                       help=f"suggested {k} phrase(s) for this pose (repeatable)")
    for k in AVOID_KEYS:
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, action="append", default=None,
                       help=f"{k.replace('_', ' ')} phrase(s) — anti-pairing guidance (repeatable)")
    p.add_argument("--pair-notes", default="", help="one-line rationale for the pairing")
    p.add_argument("--energy", choices=ENERGIES, default=None, help="pose energy class")
    p.add_argument("--canvas", choices=CANVASES, default=None,
                   help="skeleton canvas family (9:16 default; 1:1 for floor/lying variants)")


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
