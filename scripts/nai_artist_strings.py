#!/usr/bin/env python3
"""NovelAI V5 artist string cards: the global style layer of the NAI line.

Reads scripts/nai_artist_strings.json (id -> card). An artist string card holds
one artist style block for NovelAI V5 in character-neutral wording: no identity
tokens, no subject description (subjects live in the calibration briefs and the
caller's prompt). The block is composed deterministically from ``components``
and injected at the front of the base prompt (style-first ordering per
docs/novelai_line/V45_ARTIST_MIX_GUIDE.md); it never enters char_captions.

Consumable fields:
  * components -> the string's single source of truth; the CLI/`compose_block`
                  renders it into the prompt block (kind=artist gets the
                  ``artist:`` prefix, a weight W renders as ``W::tag ::``)
  * neg_add    -> style-guard negatives appended to the negative prompt
  * pairing    -> advisory only: quality-tag toggle direction plus scene /
                  lighting biases for showcase brief selection (style-level
                  wording only, never subject description)
  * sfw        -> SFW gate bit (v1 deck is SFW-only)
  * status     -> unvalidated -> verified/rejected (calibration lifecycle);
                  verified cards carry model/date/evidence in ``verified``
  * axes       -> five-axis retrieval facets (line/color/shading/medium/mood)
  * source     -> provenance (community source URL / guide / self-designed)

Usage:
  nai_artist_strings.py list [--status S] [--structure T]
  nai_artist_strings.py show <id> [--json]
  nai_artist_strings.py search <keyword> [--axis line|color|shading|medium|mood]
  nai_artist_strings.py mix <id>          # bare prompt block, shell-safe
  nai_artist_strings.py validate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "nai_artist_strings.json"

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
STRUCTURES = ("single", "duo", "trio", "style_only")
ARTIST_COUNT = {"single": 1, "duo": 2, "trio": 3, "style_only": 0}
COMPONENT_KINDS = ("artist", "style")
AXIS_KEYS = ("line", "color", "shading", "medium", "mood")
QUALITY_TAGS = ("on", "off", "unknown")
STATUSES = ("unvalidated", "verified", "rejected")
# Guide sweet spot is 1.2-1.5; anything above 2.0 distorts (V45_ARTIST_MIX_GUIDE).
MAX_WEIGHT = 2.0
# Danbooru spelling uses spaces, but a few artist names genuinely contain "_".
UNDERSCORE_OK_TAGS = frozenset({"y_y (ysk ygc)"})
REQUIRED_KEYS = {
    "name_cn", "intent", "structure", "components", "axes", "pairing",
    "neg_add", "sfw", "source", "status", "verified",
}
ALLOWED_KEYS = REQUIRED_KEYS | {"note"}
PAIRING_REQUIRED = {"quality_tags", "scene_bias", "lighting_bias"}
PAIRING_ALLOWED = PAIRING_REQUIRED | {"note"}
VERIFIED_REQUIRED = {"model", "date", "evidence"}
VERIFIED_ALLOWED = VERIFIED_REQUIRED | {"note"}


def _load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing artist string cards: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt artist string cards {DATA}: {exc}")


def cards() -> dict[str, dict]:
    return _load().get("cards", {})


def get(card_id: str) -> dict | None:
    """Exact-match card lookup (returns the card dict with its id included), else None."""
    c = cards().get(card_id)
    return {"id": card_id, **c} if c else None


def names() -> list[str]:
    return list(cards())


def _component_tag(comp: dict) -> str:
    tag = comp["tag"]
    return f"artist:{tag}" if comp["kind"] == "artist" else tag


def compose_block(card: dict) -> str:
    """Render the card's components into the prompt block, deterministically.

    ``W::artist:x ::`` for weighted components (guide string shape, space before
    the closing ``::``), bare tag otherwise. Components render in card order.
    """
    parts = []
    for comp in card["components"]:
        tag = _component_tag(comp)
        weight = comp.get("weight")
        parts.append(tag if weight is None else f"{weight:g}::{tag} ::")
    return ", ".join(parts)


def _axes_summary(card: dict) -> str:
    picks = []
    for key in ("medium", "mood"):
        values = card.get("axes", {}).get(key) or []
        if values:
            picks.append(values[0])
    return "/".join(picks)


def cmd_list(args: argparse.Namespace) -> None:
    for card_id, c in cards().items():
        if args.status and c.get("status") != args.status:
            continue
        if args.structure and c.get("structure") != args.structure:
            continue
        print(f"{card_id:<28} {c.get('name_cn', '?'):　<8} {c.get('structure', '?'):<7}"
              f" status={c.get('status', '?'):<12} {_axes_summary(c):<26} {c.get('intent', '')}")


def cmd_show(args: argparse.Namespace) -> None:
    c = get(args.card_id)
    if not c:
        raise SystemExit(f"no artist string card {args.card_id!r} (see: nai_artist_strings.py list)")
    if args.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return
    print(f"# {c['id']} · {c['name_cn']}")
    print(f"  intent      {c['intent']}")
    print(f"  block       {compose_block(c)}")
    print(f"  structure   {c['structure']}")
    for key in AXIS_KEYS:
        values = c["axes"].get(key) or []
        if values:
            print(f"  axes.{key:<7} {', '.join(values)}")
    pairing = c.get("pairing") or {}
    print(f"  quality     {pairing.get('quality_tags', '?')}")
    for key in ("scene_bias", "lighting_bias"):
        if pairing.get(key):
            print(f"  {key:<12} {', '.join(pairing[key])}")
    for key in ("neg_add", "sfw", "status", "source", "note"):
        value = c.get(key)
        if value in (None, "", [], False):
            continue
        print(f"  {key:<12} {value}")
    if c.get("verified"):
        v = c["verified"]
        print(f"  verified    {v['model']} @ {v['date']} — {v['evidence']}")


def cmd_mix(args: argparse.Namespace) -> None:
    c = get(args.card_id)
    if not c:
        raise SystemExit(f"no artist string card {args.card_id!r} (see: nai_artist_strings.py list)")
    print(compose_block(c))


def cmd_search(args: argparse.Namespace) -> None:
    needle = args.keyword.lower()
    hits = 0
    for card_id, c in cards().items():
        if args.axis:
            haystack = " ".join(c.get("axes", {}).get(args.axis) or [])
        else:
            axes_text = " ".join(v for values in c.get("axes", {}).values() for v in values)
            haystack = " ".join((card_id, c.get("name_cn", ""), c.get("intent", ""), axes_text))
        if needle in haystack.lower():
            hits += 1
            print(f"{card_id:<28} {c.get('name_cn', '?'):　<8} {c.get('intent', '')}")
    if not hits:
        raise SystemExit(f"no card matches {args.keyword!r}" + (f" on axis {args.axis}" if args.axis else ""))


def _validate_components(card_id: str, c: dict, problems: list[str]) -> None:
    def bad(msg: str) -> None:
        problems.append(f"{card_id}: {msg}")

    comps = c["components"]
    if not isinstance(comps, list) or not comps:
        bad("components must be a non-empty list")
        return
    artist_count = 0
    for i, comp in enumerate(comps):
        if not isinstance(comp, dict):
            bad(f"components[{i}] must be an object")
            continue
        unknown = set(comp) - {"tag", "kind", "weight"}
        if unknown:
            bad(f"components[{i}] unknown keys {sorted(unknown)}")
        tag = comp.get("tag")
        if not (isinstance(tag, str) and tag.strip()):
            bad(f"components[{i}].tag must be a non-empty string")
        elif tag != tag.lower() or ("_" in tag and tag not in UNDERSCORE_OK_TAGS):
            bad(f"components[{i}].tag {tag!r} must be lowercase Danbooru spelling (spaces, no underscores)")
        if comp.get("kind") not in COMPONENT_KINDS:
            bad(f"components[{i}].kind {comp.get('kind')!r} not in {COMPONENT_KINDS}")
        elif comp["kind"] == "artist":
            artist_count += 1
        weight = comp.get("weight")
        if weight is not None:
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                bad(f"components[{i}].weight must be a number or null")
            elif not 0 < float(weight) <= MAX_WEIGHT:
                bad(f"components[{i}].weight {weight} out of (0, {MAX_WEIGHT}]")
    expected = ARTIST_COUNT.get(c.get("structure"))
    if expected is not None and artist_count != expected:
        bad(f"structure {c['structure']!r} needs {expected} artist component(s), found {artist_count}")


def _validate_card(card_id: str, c: dict, problems: list[str]) -> None:
    def bad(msg: str) -> None:
        problems.append(f"{card_id}: {msg}")

    if not ID_PATTERN.match(card_id):
        bad(f"id {card_id!r} must match {ID_PATTERN.pattern}")
    unknown = set(c) - ALLOWED_KEYS
    if unknown:
        bad(f"unknown keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(c)
    if missing:
        bad(f"missing required keys {sorted(missing)}")
        return
    for key in ("name_cn", "intent", "source"):
        if not (isinstance(c[key], str) and c[key].strip()):
            bad(f"{key} must be a non-empty string")
    if c["structure"] not in STRUCTURES:
        bad(f"structure {c['structure']!r} not in {STRUCTURES}")
    _validate_components(card_id, c, problems)
    axes = c["axes"]
    if not isinstance(axes, dict) or set(axes) != set(AXIS_KEYS):
        bad(f"axes must carry exactly the keys {AXIS_KEYS}")
    else:
        for key, values in axes.items():
            if not isinstance(values, list) or not values or any(
                    not (isinstance(v, str) and v.strip()) for v in values):
                bad(f"axes.{key} must be a non-empty list of non-empty strings")
    pairing = c["pairing"]
    if not isinstance(pairing, dict):
        bad("pairing must be an object")
    else:
        unknown = set(pairing) - PAIRING_ALLOWED
        if unknown:
            bad(f"pairing unknown keys {sorted(unknown)}")
        missing_pairing = PAIRING_REQUIRED - set(pairing)
        if missing_pairing:
            bad(f"pairing missing keys {sorted(missing_pairing)}")
        else:
            if pairing["quality_tags"] not in QUALITY_TAGS:
                bad(f"pairing.quality_tags {pairing['quality_tags']!r} not in {QUALITY_TAGS}")
            for key in ("scene_bias", "lighting_bias"):
                values = pairing[key]
                if not isinstance(values, list) or any(
                        not (isinstance(v, str) and v.strip()) for v in values):
                    bad(f"pairing.{key} must be a list of non-empty strings")
    if not isinstance(c["neg_add"], str):
        bad("neg_add must be a string")
    if not isinstance(c["sfw"], bool):
        bad("sfw must be a boolean")
    if c["status"] not in STATUSES:
        bad(f"status {c['status']!r} not in {STATUSES}")
    verified = c["verified"]
    if c["status"] == "verified":
        if not isinstance(verified, dict):
            bad("status=verified requires a verified object")
        else:
            unknown = set(verified) - VERIFIED_ALLOWED
            if unknown:
                bad(f"verified unknown keys {sorted(unknown)}")
            missing_verified = VERIFIED_REQUIRED - set(verified)
            if missing_verified:
                bad(f"verified missing keys {sorted(missing_verified)}")
            else:
                evidence = ROOT.parent / verified["evidence"]
                if not evidence.exists():
                    bad(f"verified.evidence path {verified['evidence']!r} does not exist")
    elif verified is not None:
        bad(f"status {c['status']!r} must keep verified null (rejection reasons go in note)")


def cmd_validate(_args: argparse.Namespace) -> None:
    all_cards = cards()
    if not all_cards:
        raise SystemExit("no cards found")
    problems: list[str] = []
    for card_id, c in all_cards.items():
        _validate_card(card_id, c, problems)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s) in {DATA.name}")
    print(f"OK — {len(all_cards)} artist string cards valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="one line per card")
    ls.add_argument("--status", choices=STATUSES)
    ls.add_argument("--structure", choices=STRUCTURES)
    ls.set_defaults(func=cmd_list)
    sh = sub.add_parser("show", help="print a card in full")
    sh.add_argument("card_id")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_show)
    mx = sub.add_parser("mix", help="bare prompt block for shell interpolation")
    mx.add_argument("card_id")
    mx.set_defaults(func=cmd_mix)
    se = sub.add_parser("search", help="keyword match over id/name/intent/axes")
    se.add_argument("keyword")
    se.add_argument("--axis", choices=AXIS_KEYS)
    se.set_defaults(func=cmd_search)
    sub.add_parser("validate", help="schema checks (offline)").set_defaults(func=cmd_validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
