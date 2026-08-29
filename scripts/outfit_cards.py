#!/usr/bin/env python3
"""Outfit cards: the character-neutral clothing layer of the five-layer decoupling plan.

Reads scripts/outfit_cards.json (id -> card). An outfit card describes a garment /
outfit in character-neutral wording: no hair or eye colors, no body-shape language
(body language belongs to the character pack). The assembler (gen.py artistic, N4
combo engine) samples outfit x action with the motion_class compatibility filter
(card ``motion_allow`` vs the action's ``motion_class``) and the ``sfw`` bit.

Consumable fields:
  * tags           -> positive outfit tokens for the prompt (comma-joinable)
  * neg_add        -> outfit-level negatives (e.g. garment-glitch guards)
  * sfw            -> SFW gate bit (false cards are only reachable on the NSFW line)
  * coverage       -> full / moderate / minimal (body coverage, for gate judgment)
  * motion_allow   -> motion_class whitelist (static/seated/reclining/dance/dynamic),
                      the only compatibility dimension in v1 (plan N4)
  * palette_strong -> true when the outfit carries a strong inherent palette that can
                      fight a direction card's color_mood (v1: advisory only)
  * source         -> provenance (prompt doc / inspiration line / draft)
  * status         -> unvalidated -> verified/rejected (Anima card lifecycle)

Usage:
  outfit_cards.py list
  outfit_cards.py show <id> [--json]
  outfit_cards.py validate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outfit_cards.json"

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MOTION_CLASSES = ("static", "seated", "reclining", "dance", "dynamic")
COVERAGES = ("full", "moderate", "minimal")
STATUSES = ("unvalidated", "verified", "rejected")
REQUIRED_KEYS = {
    "name_cn", "intent", "tags", "neg_add", "sfw", "coverage",
    "motion_allow", "palette_strong", "source", "status",
}
ALLOWED_KEYS = REQUIRED_KEYS | {"note"}


def _load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing outfit cards: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt outfit cards {DATA}: {exc}")


def cards() -> dict[str, dict]:
    return _load().get("cards", {})


def get(outfit_id: str) -> dict | None:
    """Exact-match card lookup (returns the card dict with its id included), else None."""
    c = cards().get(outfit_id)
    return {"id": outfit_id, **c} if c else None


def names() -> list[str]:
    return list(cards())


def cmd_list(_args: argparse.Namespace) -> None:
    for outfit_id, c in cards().items():
        cover = c.get("coverage", "?")
        motion = ",".join(c.get("motion_allow") or [])
        status = c.get("status", "?")
        print(f"{outfit_id:<28} {c.get('name_cn', '?'):　<10} cover={cover:<8}"
              f" motion={motion:<22} status={status:<12} {c.get('intent', '')}")


def cmd_show(args: argparse.Namespace) -> None:
    c = get(args.outfit_id)
    if not c:
        raise SystemExit(f"no outfit card {args.outfit_id!r} (see: outfit_cards.py list)")
    if args.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return
    print(f"# {c['id']} · {c['name_cn']}")
    print(f"  intent      {c['intent']}")
    print(f"  tags        {', '.join(c['tags'])}")
    for k in ("neg_add", "coverage", "motion_allow", "palette_strong", "sfw", "status",
              "source", "note"):
        v = c.get(k)
        if v in (None, "", [], False):
            continue
        print(f"  {k:<12} {', '.join(v) if isinstance(v, list) else v}")


def _validate_card(outfit_id: str, c: dict, problems: list[str]) -> None:
    def bad(msg: str) -> None:
        problems.append(f"{outfit_id}: {msg}")

    if not ID_PATTERN.match(outfit_id):
        bad(f"id {outfit_id!r} must match {ID_PATTERN.pattern}")
    unknown = set(c) - ALLOWED_KEYS
    if unknown:
        bad(f"unknown keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(c)
    if missing:
        bad(f"missing required keys {sorted(missing)}")
        return
    for k in ("name_cn", "intent", "source"):
        if not (isinstance(c[k], str) and c[k].strip()):
            bad(f"{k} must be a non-empty string")
    tags = c["tags"]
    if not isinstance(tags, list) or not tags or any(not (isinstance(t, str) and t.strip()) for t in tags):
        bad("tags must be a non-empty list of non-empty strings")
    if not isinstance(c["neg_add"], str):
        bad("neg_add must be a string")
    if not isinstance(c["sfw"], bool):
        bad("sfw must be a boolean")
    if c["coverage"] not in COVERAGES:
        bad(f"coverage {c['coverage']!r} not in {COVERAGES}")
    motion = c["motion_allow"]
    if not isinstance(motion, list) or not motion:
        bad("motion_allow must be a non-empty list")
    else:
        unknown_motion = set(motion) - set(MOTION_CLASSES)
        if unknown_motion:
            bad(f"motion_allow has unknown motion classes {sorted(unknown_motion)}")
    if not isinstance(c["palette_strong"], bool):
        bad("palette_strong must be a boolean")
    if c["status"] not in STATUSES:
        bad(f"status {c['status']!r} not in {STATUSES}")


def cmd_validate(_args: argparse.Namespace) -> None:
    all_cards = cards()
    if not all_cards:
        raise SystemExit("no cards found")
    problems: list[str] = []
    for outfit_id, c in all_cards.items():
        _validate_card(outfit_id, c, problems)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s) in {DATA.name}")
    print(f"OK — {len(all_cards)} outfit cards valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="one line per outfit card").set_defaults(func=cmd_list)
    sh = sub.add_parser("show", help="print a card in full")
    sh.add_argument("outfit_id")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_show)
    sub.add_parser("validate", help="schema checks (offline)").set_defaults(func=cmd_validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
