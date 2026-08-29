#!/usr/bin/env python3
"""Base-model quirks cards: hard-won per-checkpoint behavior so gen params aren't re-derived
each session. Reads scripts/model_quirks.json (base-name -> card).

Consumable fields a caller acts on:
  * neg_add   -> pass to z3_quality_gen.py --neg-add (append; does not override the recipe neg)
  * no_freeu  -> pass --no-freeu (this base blows out highlights under FreeU)
Everything else (style / strengths / body_prior / best_shots / ...) is human guidance for
picking the right base per shot.

Lookup is fuzzy: the JSON key is a base filename stem; a query matches if the key is a
case-insensitive substring of the query, so a full ComfyUI path like
'Illustrious\\anime\\miaomiao3DHarem_lh3dEPS20.safetensors' resolves to its card.

Video models live in a separate `video_models` section rather than under `bases`: they
have no negative channel at all, so the neg_add / nsfw fields a base card must carry are
meaningless for them. Reach those with `lookup --video`.

Usage:
  model_quirks.py list
  model_quirks.py lookup "<base name or ckpt path>"      # full card (human)
  model_quirks.py lookup "<base>" --json                 # full card (JSON)
  model_quirks.py lookup minimax_h3 --video              # video-model card
  model_quirks.py neg-add "<base>"                        # just the neg_add string (shell-interpolable)
  model_quirks.py nsfw-neg-add "<base>"                   # just the nsfw.neg_add_nsfw string
  model_quirks.py flags "<base>"                          # extra gen flags, e.g. "--no-freeu"
  model_quirks.py validate                                # schema check (all bases + nsfw fields)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "model_quirks.json"

_LOAD_CACHE: tuple[tuple[int, int], dict] | None = None


def _load() -> dict:
    """Parse model_quirks.json, memoized on (mtime_ns, size) so repeated card lookups in
    one process parse the file once while an edited file is still picked up."""
    global _LOAD_CACHE
    try:
        stat = DATA.stat()
    except FileNotFoundError:
        raise SystemExit(f"missing quirks data: {DATA}")
    key = (stat.st_mtime_ns, stat.st_size)
    if _LOAD_CACHE is not None and _LOAD_CACHE[0] == key:
        return _LOAD_CACHE[1]
    try:
        data = json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing quirks data: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt quirks data {DATA}: {exc}")
    _LOAD_CACHE = (key, data)
    return data


def card(base: str) -> dict | None:
    """Return the quirks card whose key is a substring of `base` (case-insensitive), else None.

    Longest key wins, so a specific variant beats a generic stem when both would match.
    """
    bases = _load().get("bases", {})
    q = base.lower()
    # bidirectional: key-in-query resolves a full ckpt path; query-in-key resolves a short stem.
    # longest key first so a specific variant beats a generic stem.
    for key in sorted(bases, key=len, reverse=True):
        k = key.lower()
        if k in q or q in k:
            return {"base": key, **bases[key]}
    return None


def video_card(name: str) -> dict | None:
    """Same fuzzy match as `card`, over the `video_models` section."""
    models = _load().get("video_models", {})
    q = name.lower()
    for key in sorted(models, key=len, reverse=True):
        k = key.lower()
        if k in q or q in k:
            return {"model": key, **models[key]}
    return None


def _require(base: str, video: bool = False) -> dict:
    c = video_card(base) if video else card(base)
    if not c:
        kind = "video quirks card" if video else "quirks card"
        raise SystemExit(f"no {kind} matches {base!r} (see: model_quirks.py list)")
    return c


BASE_FIELD_ORDER = (
    "arch", "style", "prompt_style", "recipe", "speed", "res_range", "vram",
    "engine", "strengths", "body_prior", "fabric_prior", "exposure", "artifact",
    "pose_control", "lora_strength_ok", "best_shots", "avoid_shots", "no_freeu",
    "neg_add", "nsfw", "ckpt", "note", "verified", "evidence", "portrait_recipe",
)

VIDEO_FIELD_ORDER = (
    "arch", "engine", "weights", "graph", "frame_law", "train_range", "native_canvas",
    "no_cfg", "sampling", "prompt_structure", "memory_model", "sharpness",
    "speedups_not_installed", "quirks", "verified", "evidence",
)


def cmd_list(_args: argparse.Namespace) -> None:
    data = _load()
    for key, c in data.get("bases", {}).items():
        print(f"{key:<40} {c.get('style', '?'):<18} best={', '.join(c.get('best_shots', [])) or '-'}")
    for key, c in data.get("video_models", {}).items():
        print(f"{key:<40} {'video':<18} {c.get('native_canvas', '?')}")


def _render_card(c: dict, title: str, fields: tuple[str, ...]) -> None:
    print(f"# {title}")
    for k in fields:
        v = c.get(k)
        if v in (None, "", [], False):
            continue
        if isinstance(v, list):
            joined = ", ".join(v)
            # Short tag lists read fine inline; quirk lists are prose and need a line each.
            if len(joined) <= 100:
                print(f"  {k:<16} {joined}")
            else:
                print(f"  {k}")
                for item in v:
                    print(f"    - {item}")
            continue
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False, sort_keys=True)
        print(f"  {k:<16} {v}")


def cmd_lookup(args: argparse.Namespace) -> None:
    c = _require(args.base, video=args.video)
    if args.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return
    if args.video:
        _render_card(c, c["model"], VIDEO_FIELD_ORDER)
    else:
        _render_card(c, c["base"], BASE_FIELD_ORDER)


def cmd_neg_add(args: argparse.Namespace) -> None:
    print(_require(args.base).get("neg_add", ""))


def cmd_nsfw_neg_add(args: argparse.Namespace) -> None:
    nsfw = _require(args.base).get("nsfw", {})
    print(nsfw.get("neg_add_nsfw", ""))


def cmd_flags(args: argparse.Namespace) -> None:
    c = _require(args.base)
    flags = ["--no-freeu"] if c.get("no_freeu") else []
    print(" ".join(flags))


def cmd_validate(_args: argparse.Namespace) -> None:
    data = _load()
    bases = data.get("bases", {})
    errors = []
    if not isinstance(data.get("version"), int):
        errors.append("top-level 'version' missing or not an integer")
    for key, c in bases.items():
        if not isinstance(c, dict):
            errors.append(f"{key}: not a dict")
            continue
        # required fields (at minimum a note or ckpt to confirm it's a real card)
        if "style" not in c and "arch" not in c:
            errors.append(f"{key}: missing 'style' or 'arch'")
        # nsfw sub-object
        nsfw = c.get("nsfw")
        if not isinstance(nsfw, dict):
            errors.append(f"{key}: missing or invalid 'nsfw' sub-object")
        else:
            if not nsfw.get("neg_add_nsfw"):
                errors.append(f"{key}.nsfw: missing 'neg_add_nsfw'")
    # Video cards carry no nsfw/neg_add (no negative channel exists), so they get their
    # own required set: enough to reconstruct a run without reopening the wave notes.
    videos = data.get("video_models", {})
    for key, c in videos.items():
        if not isinstance(c, dict):
            errors.append(f"video_models.{key}: not a dict")
            continue
        for field in ("arch", "engine", "frame_law", "sampling", "quirks", "verified"):
            if not c.get(field):
                errors.append(f"video_models.{key}: missing {field!r}")
        if "nsfw" in c or "neg_add" in c:
            errors.append(f"video_models.{key}: base-only field (nsfw/neg_add) on a video card")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise SystemExit(f"{len(errors)} validation error(s)")
    print(f"OK: {len(bases)} base(s) with valid nsfw sub-objects, {len(videos)} video model(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all quirks cards").set_defaults(func=cmd_list)
    lk = sub.add_parser("lookup", help="print a base's full quirks card")
    lk.add_argument("base")
    lk.add_argument("--json", action="store_true")
    lk.add_argument("--video", action="store_true",
                    help="look the name up in video_models instead of bases")
    lk.set_defaults(func=cmd_lookup)
    na = sub.add_parser("neg-add", help="print just the neg_add string for --neg-add")
    na.add_argument("base")
    na.set_defaults(func=cmd_neg_add)
    nna = sub.add_parser("nsfw-neg-add", help="print just the nsfw.neg_add_nsfw string for --nsfw-neg-add")
    nna.add_argument("base")
    nna.set_defaults(func=cmd_nsfw_neg_add)
    fl = sub.add_parser("flags", help="print extra gen flags (e.g. --no-freeu)")
    fl.add_argument("base")
    fl.set_defaults(func=cmd_flags)
    sub.add_parser("validate", help="validate quirks schema").set_defaults(func=cmd_validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
