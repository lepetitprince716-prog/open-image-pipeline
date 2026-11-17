#!/usr/bin/env python3
"""Style cards: list, show, resolve, matrix and lint for the cross-line style registry.

Reads ``catalogs/style/*.json`` and the assets they reference (NovelAI artist library,
LoRA catalog, t2i catalog items). ``resolve`` prints the prompt, LoRA and model fragment one
line needs for a style id; ``matrix`` prints per-line coverage. Offline, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loramake import style_registry as sr  # noqa: E402


def _emit(payload: object, as_json: bool, human: str = "") -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human if human else json.dumps(payload, ensure_ascii=False, indent=2))


def _glyph_row(card: dict) -> str:
    cells = []
    for line in sr.LINES:
        status, _ = sr.line_status(card, line)
        cells.append(sr.STATUS_GLYPH[status])
    return " ".join(cells)


def cmd_list(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    rows = []
    for card_id, card in cards.items():
        if args.family and card.get("family") != args.family:
            continue
        if args.priority and card.get("priority") != args.priority:
            continue
        if args.line:
            status, _ = sr.line_status(card, args.line)
            if status in ("na", "gap"):
                continue
        rows.append({"id": card_id, "family": card.get("family"), "priority": card.get("priority"),
                     "display_name": card.get("display_name"),
                     "lines": {line: sr.line_status(card, line)[0] for line in sr.LINES}})
    human = "\n".join(f"{r['id']:20s} {r['priority']:5s} {r['family']:8s} "
                      f"{_glyph_row(cards[r['id']])}  {r['display_name']}" for r in rows)
    header = f"{'id':20s} {'pri':5s} {'family':8s} {' '.join(ln[:2] for ln in sr.LINES)}\n"
    _emit(rows, args.json, header + human)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    card = cards.get(args.style_id)
    if card is None:
        raise SystemExit(f"unknown style {args.style_id!r}; run `style_cards.py list`")
    if args.json:
        _emit(card, True)
        return 0
    print(f"{card['id']} · {card.get('display_name')} · {card.get('family')} · {card.get('priority')}")
    print(card.get("description", ""))
    for line in sr.LINES:
        reals = (card.get("realizations") or {}).get(line, [])
        if reals:
            print(f"[{line}]")
            for real in reals:
                ref = real.get("ref") or real.get("air") or real.get("caption", "")[:60] or real.get("prompt", "")[:60]
                print(f"  {sr.STATUS_GLYPH[real['tier']]} {real['id']}  {real['kind']}  {ref}")
    for gap in card.get("gaps") or []:
        print(f"[{gap['line']}] ✗ {gap['path']}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    try:
        frag = sr.resolve(args.style_id, args.line, cards=cards, box=args.box, prefer=args.prefer)
    except sr.StyleRegistryError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        _emit(frag.as_dict(), True)
        return 0
    print(f"{frag.style_id} on {frag.line}: {frag.realization_id} ({frag.kind}, {frag.tier})")
    for key in ("pos_prefix", "pos_suffix", "neg_add", "caption", "base_pin", "model", "air",
                "base_model", "trigger", "post_fx", "note"):
        value = getattr(frag, key)
        if value:
            print(f"  {key}: {value}")
    for spec in frag.lora_specs:
        print(f"  lora: {spec}")
    if frag.sampler:
        print(f"  sampler: {json.dumps(frag.sampler, ensure_ascii=False)}")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    if args.by == "aesthetic":
        rows = sr.register_matrix(sr.load_registers(), cards)
        if args.json:
            _emit(rows, True)
        else:
            print(sr.register_matrix_markdown(rows))
            for line in sr.LINES:
                print(f"{line}: verified {', '.join(sr.verified_registers(line, cards=cards)) or '—'}")
        return 0
    rows = sr.matrix(cards)
    summary = sr.matrix_summary(rows)
    if args.json:
        _emit({"summary": summary, "rows": rows}, True)
        return 0
    print(sr.matrix_markdown(rows))
    print()
    per_line = ", ".join(f"{line}: ✅{c['verified']} ◐{c['candidate']} ✗{c['gap']}"
                         for line, c in summary["per_line"].items())
    print(f"{summary['cards']} cards, {summary['any_verified']} with a verified line; {per_line}")
    return 0


def cmd_import_loras(args: argparse.Namespace) -> int:
    import datetime
    import json as _json

    rows = _json.loads(Path(args.gap).read_text(encoding="utf-8"))
    cards = sr.load_cards()
    loras = sr.lora_index()
    date = datetime.date.today().isoformat()
    new, skipped = sr.artist_lora_cards(rows, loras, cards, date, min_posts=args.min_posts)
    lines = {}
    for card in new.values():
        lines[card["lines"][0]] = lines.get(card["lines"][0], 0) + 1
    print(f"{len(new)} new artist cards ({', '.join(f'{k} {v}' for k, v in sorted(lines.items()))}); "
          f"skipped: " + ", ".join(f"{k} {v}" for k, v in skipped.items() if v))
    if not args.apply:
        for card_id in sorted(new)[:args.limit]:
            print(f"  {card_id}: {new[card_id]['display_name']} — {new[card_id]['description']}")
        print("dry run; pass --apply to write the cards, then run lint")
        return 0
    for card_id, card in sorted(new.items()):
        path = sr.CARDS_DIR / f"{card_id}.json"
        path.write_text(_json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    problems = sr.lint(sr.load_cards())
    print(f"wrote {len(new)} cards; lint: {len(problems)} problem(s)")
    for p in problems[:20]:
        print("  " + p)
    return 1 if problems else 0


def cmd_registers(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    rows = sr.register_matrix(sr.load_registers(), cards)
    if args.line:
        rows = [r for r in rows if r["lines"][args.line]["status"] in ("verified", "candidate")]
    if args.json:
        _emit(rows, True)
        return 0
    for row in rows:
        cells = " ".join(sr.STATUS_GLYPH[row["lines"][line]["status"]] for line in sr.LINES)
        print(f"{row['id']:14s} {cells}  {row['name_cn']}  ({', '.join(row['cards'])})")
    print(f"{' ' * 15}{' '.join(ln[:2] for ln in sr.LINES)}")
    return 0


def cmd_resolve_aesthetic(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    try:
        frag = sr.resolve_aesthetic(args.register_id, args.line, density=args.density,
                                   cards=cards, box=args.box)
    except sr.StyleRegistryError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        _emit(frag.as_dict(), True)
        return 0
    print(f"{args.register_id} -> {frag.style_id} on {frag.line}: {frag.realization_id} ({frag.kind}, {frag.tier})")
    for key in ("pos_prefix", "neg_add", "base_pin", "model", "note"):
        if getattr(frag, key):
            print(f"  {key}: {getattr(frag, key)}")
    for spec in frag.lora_specs:
        print(f"  lora: {spec}")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    cards = sr.load_cards(args.cards_dir)
    problems = sr.lint(cards)
    problems += sr.register_problems(sr.load_registers(), cards)
    if args.json:
        _emit({"cards": len(cards), "problems": problems}, True)
    else:
        for problem in problems:
            print(problem)
        print(f"{len(cards)} cards, {len(problems)} problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-line style registry tooling")
    parser.add_argument("--cards-dir", default=None, help="override catalogs/style")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="one row per card with per-line coverage glyphs")
    p.add_argument("--line", choices=sr.LINES)
    p.add_argument("--family", choices=sr.FAMILIES)
    p.add_argument("--priority", choices=sr.PRIORITIES)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="print one card")
    p.add_argument("style_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("resolve", help="render fragment for a style on one line")
    p.add_argument("style_id")
    p.add_argument("--line", required=True, choices=sr.LINES)
    p.add_argument("--box", choices=("boxa", "boxb"))
    p.add_argument("--prefer", help="realization id to force")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("matrix", help="coverage matrix, markdown by default")
    p.add_argument("--by", choices=("card", "aesthetic"), default="card",
                   help="aesthetic: one row per register (catalogs/style/registers.json)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("registers", help="aesthetic registers with per-line best tier")
    p.add_argument("--line", choices=sr.LINES, help="only registers usable on this line")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_registers)

    p = sub.add_parser("resolve-aesthetic", help="register (or card) + density -> fragment for one line")
    p.add_argument("register_id")
    p.add_argument("--line", required=True, choices=sr.LINES)
    p.add_argument("--density", default="medium", help="light|medium|strong or 淡|中|浓")
    p.add_argument("--box", choices=("boxa", "boxb"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resolve_aesthetic)

    p = sub.add_parser("import-loras", help="one candidate card per unreferenced SFW artist LoRA on BOXA")
    p.add_argument("--gap", default=str(sr.ARTIST_GAP_TABLE), help="artist gap table (artist_gap_*.json)")
    p.add_argument("--min-posts", type=int, default=0, help="skip Danbooru-known artists below this post count")
    p.add_argument("--limit", type=int, default=15, help="dry run: how many new cards to list")
    p.add_argument("--apply", action="store_true", help="write the cards into catalogs/style/")
    p.set_defaults(func=cmd_import_loras)

    p = sub.add_parser("lint", help="schema and reference checks; exit 1 on problems")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except sr.StyleRegistryError as exc:
        print(f"style_cards.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
