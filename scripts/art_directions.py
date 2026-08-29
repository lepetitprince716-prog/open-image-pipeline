#!/usr/bin/env python3
"""Art direction cards: coherent pose x lighting x composition x palette x medium bundles, so a
winning concept is a queryable asset instead of a hand-written prompt (mirrors the model_quirks
"JSON cards + query CLI" pattern). Reads scripts/art_directions.json (slug -> card).

A card PINS its listed slots in the artistic line (`gen.py artistic --direction <slug>`); unpinned
slots still roll their __wildcard__ per seed. Slot priority in gen.py: explicit per-slot flag >
pose pairing injection > card slots > wildcard default.

Consumable fields:
  * slots.*     -> the per-slot values art_prompt.build() pins ('none' drops a slot)
  * pose_ref    -> pose-library skeleton slug (gen.py injects the verified ControlNet recipe)
  * base        -> {primary, alt, why}; primary becomes --base on z3 (zzz keeps forcing iLustMix)
  * style_loras -> engine FILE[:STR] specs, used only when the caller passes no --style-lora
  * neg_add     -> appended via z3_quality_gen --neg-add (merged with any user neg_add)
  * flags       -> extra engine flags (e.g. --no-freeu), appended if absent
  * post_fx     -> finishing pass auto-applied on --run as a <stem>_fx.png sibling (preset + overrides;
                   'none' disables it for the card; gen.py --no-fx / LORAMAKE_POST_FX=0 turns it off globally)
  * bg_brief    -> one-line lighting brief for the object/background line (cards never fill a
                   background slot; that dimension belongs to the other conversation)

Usage:
  art_directions.py list
  art_directions.py show <slug> [--json]
  art_directions.py validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "art_directions.json"
POSE_INDEX = ROOT.parent / "outputs/poses/library/index.json"

SLOT_KEYS = ("medium", "camera", "composition", "lighting", "pose", "expression", "color_mood")
POST_FX_PRESETS = {"none", "film", "teal-orange", "muted", "warm"}  # keep in sync with post_fx.py
REQUIRED_KEYS = {"name_cn", "intent", "slots", "aspect", "base", "post_fx", "bg_brief"}
ALLOWED_KEYS = REQUIRED_KEYS | {"pose_ref", "pose_tags_any", "style_loras", "neg_add", "flags",
                                "sfw", "detail", "sfw_note", "note"}


def _load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing direction cards: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt direction cards {DATA}: {exc}")


def cards() -> dict[str, dict]:
    return _load().get("cards", {})


def get(slug: str) -> dict | None:
    """Exact-match card lookup (returns the card dict with its slug included), else None."""
    c = cards().get(slug)
    return {"slug": slug, **c} if c else None


def names() -> list[str]:
    return list(cards())


def pose_index_slugs() -> list[str] | None:
    """Slugs in the pose-library index, or None when the index is not on this machine."""
    if not POSE_INDEX.is_file():
        return None
    try:
        data = json.loads(POSE_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    entries = data.get("poses", []) if isinstance(data, dict) else data
    return [e.get("slug", "") for e in entries]


def cmd_list(_args: argparse.Namespace) -> None:
    for slug, c in cards().items():
        pose = c.get("pose_ref") or ("(text pose)" if c.get("slots", {}).get("pose") else "-")
        fx = c.get("post_fx", {}).get("preset", "-")
        print(f"{slug:<22} {c.get('name_cn', '?'):　<8} base={c.get('base', {}).get('primary', '?'):<38}"
              f" pose={pose:<18} fx={fx:<12} {c.get('intent', '')}")


def cmd_show(args: argparse.Namespace) -> None:
    c = get(args.slug)
    if not c:
        raise SystemExit(f"no direction card {args.slug!r} (see: art_directions.py list)")
    if args.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return
    print(f"# {c['slug']} · {c['name_cn']}")
    print(f"  intent      {c['intent']}")
    for k, v in c.get("slots", {}).items():
        print(f"  slot.{k:<12} {v}")
    for k in ("pose_ref", "pose_tags_any", "aspect", "style_loras", "neg_add", "flags",
              "sfw", "detail", "sfw_note", "note", "bg_brief"):
        v = c.get(k)
        if v in (None, "", [], False):
            continue
        print(f"  {k:<12} {', '.join(v) if isinstance(v, list) else v}")
    b = c.get("base", {})
    print(f"  base        {b.get('primary')} (alt: {b.get('alt') or '-'}) — {b.get('why', '')}")
    fx = c.get("post_fx", {})
    print(f"  post_fx     {fx.get('preset')} {fx.get('overrides', '')}".rstrip())
    print(f"\n  try: python3 scripts/gen.py artistic --prompt '<subject tags>' "
          f"--direction {c['slug']} --n 6 --sfw")


def _validate_card(slug: str, c: dict, quirk_card, aspects: dict, pose_slugs: list[str] | None,
                   problems: list[str]) -> None:
    def bad(msg: str) -> None:
        problems.append(f"{slug}: {msg}")

    unknown = set(c) - ALLOWED_KEYS
    if unknown:
        bad(f"unknown keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(c)
    if missing:
        bad(f"missing required keys {sorted(missing)}")
        return
    for k in ("name_cn", "intent", "bg_brief"):
        if not (isinstance(c[k], str) and c[k].strip()):
            bad(f"{k} must be a non-empty string")
    slots = c["slots"]
    for k, v in slots.items():
        if k not in SLOT_KEYS:
            bad(f"slots.{k} is not a pinnable slot (background belongs to the other line)")
        if not (isinstance(v, str) and v.strip()):
            bad(f"slots.{k} must be a non-empty string")
        elif any(w in v.lower() for w in ("background", "backdrop")):
            bad(f"slots.{k} names a background ({v!r}) — use bg_brief for the other line")
    if c["aspect"] not in aspects:
        bad(f"aspect {c['aspect']!r} not in art_prompt.ASPECTS")
    base = c["base"]
    for role in ("primary", "alt"):
        name = base.get(role)
        if role == "primary" and not name:
            bad("base.primary is required")
        if name and quirk_card(name) is None:
            bad(f"base.{role} {name!r} has no model_quirks card")
    prim = quirk_card(base.get("primary", "")) if base.get("primary") else None
    if prim:
        q_neg = prim.get("neg_add", "")
        if q_neg and q_neg not in c.get("neg_add", ""):
            bad(f"neg_add must contain the primary base's quirks neg_add ({prim['base']})")
        if prim.get("no_freeu") and "--no-freeu" not in c.get("flags", []):
            bad(f"flags must include --no-freeu (primary base {prim['base']} is no_freeu)")
    fx = c["post_fx"]
    if fx.get("preset") not in POST_FX_PRESETS:
        bad(f"post_fx.preset {fx.get('preset')!r} not in {sorted(POST_FX_PRESETS)}")
    for k, typ in (("pose_tags_any", list), ("style_loras", list), ("flags", list),
                   ("sfw", bool), ("detail", bool)):
        if k in c and not isinstance(c[k], typ):
            bad(f"{k} must be {typ.__name__}")
    ref = c.get("pose_ref")
    if ref:
        if pose_slugs is None:
            print(f"  [skip] {slug}: pose_ref {ref!r} unchecked (pose index not on this machine)")
        elif ref not in pose_slugs:
            bad(f"pose_ref {ref!r} not in the pose-library index")


def cmd_validate(_args: argparse.Namespace) -> None:
    import art_prompt
    import model_quirks
    all_cards = cards()
    if not all_cards:
        raise SystemExit("no cards found")
    pose_slugs = pose_index_slugs()
    problems: list[str] = []
    for slug, c in all_cards.items():
        _validate_card(slug, c, model_quirks.card, art_prompt.ASPECTS, pose_slugs, problems)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s) in {DATA.name}")
    print(f"OK — {len(all_cards)} cards valid"
          + ("" if pose_slugs is not None else " (pose_ref checks skipped: no pose index)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="one line per direction card").set_defaults(func=cmd_list)
    sh = sub.add_parser("show", help="print a card in full (+ a ready command)")
    sh.add_argument("slug")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_show)
    sub.add_parser("validate", help="schema + cross-reference checks (offline)").set_defaults(func=cmd_validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
