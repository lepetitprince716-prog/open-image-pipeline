#!/usr/bin/env python3
"""Content explorer: sample composition + content cards by coverage, render them on the
NovelAI free tier with one fixed artist card, and register the review batch.

    python3 scripts/content_deck.py generate --deck content_explore_r1 --count 15 --seed 41
    python3 scripts/content_deck.py render configs/content_decks/content_explore_r1.json [--dry-run]
    python3 scripts/content_deck.py register configs/content_decks/content_explore_r1.json

Cards come from ``configs/composition/vocabulary.json`` (plan section 14); earlier decks in
``configs/content_decks/`` count as history so a new round does not repeat them. Renders go
to ``outputs/nai/<deck>/<card>/`` through ``nai_gen.py`` with ``--anlas-budget 0`` (free tier
only); ``register`` hands the renders to ``review_batch.py`` as the batch named after the deck.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake import composition as comp  # noqa: E402

VOCABULARY = REPO / "configs" / "composition" / "vocabulary.json"
SERIES = REPO / "configs" / "composition" / "series.json"
DECKS_DIR = REPO / "configs" / "content_decks"
OUTPUTS = REPO / "outputs" / "nai"
NAI_GEN = REPO / "scripts" / "nai_gen.py"
REVIEW_BATCH = REPO / "scripts" / "review_batch.py"
DEFAULT_ARTIST = "kieed_solo"
DEFAULT_STEPS = 28
IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}


def parse_styles(spec: str) -> list[str]:
    """``a,b:3,c`` -> [a, b, b, b, c]: a style may carry a repeat count after a colon."""
    styles: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name, _, count = token.partition(":")
        repeat = int(count) if count else 1
        if repeat < 1:
            raise ValueError(f"style repeat must be positive: {token}")
        styles.extend([name.strip()] * repeat)
    return styles


def history_cards(decks_dir: Path, exclude: str | None = None) -> list[dict]:
    cards: list[dict] = []
    for path in sorted(decks_dir.glob("*.json")):
        try:
            deck = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if deck.get("schema") != "loramake.content-deck/v1" or deck.get("deck") == exclude:
            continue
        cards.extend(card for card in deck.get("cards", []) if isinstance(card, dict))
    return cards


def print_coverage(cards: list[dict]) -> None:
    for dim, counts in comp.coverage(cards).items():
        spread = ", ".join(f"{value} {n}" for value, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  {dim:10s} {len(counts)} values: {spread}")


def nai_args(deck: dict, card: dict, out_dir: Path, steps: int, dry_run: bool) -> list[str]:
    args = [sys.executable, str(NAI_GEN), "--prompt", card["prompt"], "--model", deck.get("model", "v5-full"),
            "--width", str(card["width"]), "--height", str(card["height"]), "--steps", str(steps),
            "--seed", str(card["seed"]), "--out", str(out_dir), "--anlas-budget", "0"]
    artist = card.get("artist_card") or deck.get("artist_card")
    if artist:
        args += ["--artist-mix", artist]
    if deck.get("quality_toggle"):
        args.append("--quality-toggle")
    if dry_run:
        args.append("--dry-run")
    return args


def render(deck: dict, *, out_root: Path, steps: int, dry_run: bool) -> list[dict]:
    results = []
    for card in deck["cards"]:
        out_dir = out_root / deck["deck"] / card["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        before = {p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        if before and not dry_run:
            results.append({"id": card["id"], "status": "skipped", "files": [str(p) for p in sorted(before)]})
            print(f"skip     {card['id']} (already rendered)")
            continue
        proc = subprocess.run(nai_args(deck, card, out_dir, steps, dry_run), capture_output=True, text=True)
        after = {p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES} - before
        status = "ok" if proc.returncode == 0 and (after or dry_run) else "failed"
        results.append({"id": card["id"], "status": status, "files": [str(p) for p in sorted(after)],
                        "returncode": proc.returncode, "stderr": proc.stderr[-400:]})
        print(f"{status:8s} {card['id']}" + ("" if status == "ok" else f"  {proc.stderr.strip()[-200:]}"))
    (out_root / deck["deck"] / "run_index.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                                                             encoding="utf-8")
    return results


def review_spec(deck: dict, out_root: Path) -> list[dict]:
    spec = []
    for card in deck["cards"]:
        out_dir = out_root / deck["deck"] / card["id"]
        files = sorted(p for p in out_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if out_dir.is_dir() else []
        if files:
            spec.append({"path": str(files[-1].resolve()), "caption": card["caption"]})
    return spec


REVIEW_ROOT = REPO / "outputs" / "_gallery" / "review_batches"


def deck_verdicts(deck: dict) -> dict:
    """Per-card verdicts of the review batch named after the deck."""
    batch_dir = REVIEW_ROOT / deck["deck"]
    result = json.loads((batch_dir / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {card["id"]: card for card in deck["cards"]}
    label = {}
    for key, name in (("l", "love"), ("y", "yes"), ("n", "no")):
        for index in result.get(key, []):
            label[index] = name
    cards = {}
    for item in manifest.get("items", []):
        caption = item.get("caption", "")
        card_id = caption[1:caption.index("]")] if caption.startswith("[") and "]" in caption else None
        card = by_id.get(card_id)
        if card is None:
            continue
        cards[card_id] = {"style": card["style"], "character": card.get("character"), "motif": card.get("motif"),
                          "verdict": label.get(item["i"], "unjudged")}
    return {"deck": deck["deck"], "batch": deck["deck"], "date": time.strftime("%Y-%m-%d"), "cards": cards}


def verdicts_command(deck: dict, *, dry_run: bool) -> int:
    data = deck_verdicts(deck)
    counts = {v: sum(1 for c in data["cards"].values() if c["verdict"] == v) for v in ("love", "yes", "no", "unjudged")}
    print(f"{deck['deck']}: love {counts['love']} / yes {counts['yes']} / no {counts['no']} / unjudged {counts['unjudged']}")
    if counts["unjudged"]:
        print("error: batch not fully judged", file=sys.stderr)
        return 2
    verdict_path = DECKS_DIR / f"{deck['deck']}.verdicts.json"
    files = []
    for path in sorted(DECKS_DIR.glob("*.verdicts.json")):
        if path != verdict_path:
            files.append(json.loads(path.read_text(encoding="utf-8")))
    files.append(data)
    vocab_path = VOCABULARY
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    groups = vocab.setdefault("style_groups", {})
    already = list(groups.get("excluded_by_user") or [])
    newly = [style for style in comp.styles_to_ban(comp.verdict_totals(files)) if style not in already]
    # The user's rule: a medium whose every card in this round was rejected never appears again.
    round_totals = comp.verdict_totals([data])
    for style in comp.styles_to_ban(round_totals):
        if style not in already and style not in newly:
            newly.append(style)
    for style in sorted({c["style"] for c in data["cards"].values() if c["verdict"] == "no"}):
        print(f"  rejected this round: {style}" + ("  -> banned" if style in newly else ""))
    if dry_run:
        print("(dry run)")
        return 0
    verdict_path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    groups["excluded_by_user"] = already + newly
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"banned media now {len(groups['excluded_by_user'])} (+{len(newly)}: {', '.join(newly) or '-'})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Content explorer: generate, render and register composition decks")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--deck", required=True)
    gen.add_argument("--count", type=int, default=15)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--line", default="nai")
    gen.add_argument("--artist", default=DEFAULT_ARTIST, help="fixed NovelAI artist card so verdicts judge composition")
    gen.add_argument("--max-overlap", type=int, default=comp.DEFAULT_MAX_OVERLAP)
    gen.add_argument("--force", action="store_true")
    gen.add_argument("--quality-toggle", action="store_true", help="render with NovelAI server-side quality tags")
    gen.add_argument("--artists", help="comma-separated NovelAI artist card ids assigned to a share of the wild cards")
    gen.add_argument("--artist-share", type=float, default=0.5)
    # A series is authored and a wild deck is sampled; asking for both is a contradiction,
    # and taking the series branch silently would drop the flag the caller also asked for.
    shape = gen.add_mutually_exclusive_group()
    shape.add_argument("--wild", action="store_true", help="far styles around female characters, no artist card")
    gen.add_argument("--styles", help="comma-separated style ids for --wild, each with an optional :count (default: every style once)")
    gen.add_argument("--per-style", type=int, default=1, help="repeat each style this many times (--wild)")
    shape.add_argument("--series", help="series preset id from configs/composition/series.json: one subject "
                                      "walked along a single axis, --count and --max-overlap ignored "
                                      "(the preset fixes the card list)")
    rnd = sub.add_parser("render")
    rnd.add_argument("deck")
    rnd.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    rnd.add_argument("--dry-run", action="store_true")
    reg = sub.add_parser("register")
    reg.add_argument("deck")
    ver = sub.add_parser("verdicts", help="save a judged deck's verdicts and ban media that only ever got rejected")
    ver.add_argument("deck")
    ver.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vocab = comp.load_vocabulary(VOCABULARY)
    if args.command == "generate":
        path = DECKS_DIR / f"{args.deck}.json"
        if path.exists() and not args.force:
            print(f"error: {path} exists (use --force)", file=sys.stderr)
            return 2
        history = history_cards(DECKS_DIR, exclude=args.deck)
        try:
            if args.series:
                # Resolve the preset first: a name that does not exist is the more basic
                # error, and reporting a knob complaint for it sends the reader after the
                # wrong flag.
                presets = comp.load_series(SERIES)
                if args.series not in presets:
                    print(f"error: unknown series {args.series!r}; have {', '.join(sorted(presets))}",
                          file=sys.stderr)
                    return 2
                # A series authors every card, so the knobs that steer wild sampling have
                # nothing to act on. Accepting and ignoring them is the swallowed-flag bug
                # the --wild exclusion was added to prevent.
                ignored = [name for name, value in (("--styles", args.styles),
                                                    ("--artists", args.artists),
                                                    ("--per-style", args.per_style != 1),
                                                    ("--artist-share", args.artist_share != 0.5))
                           if value]
                if ignored:
                    print(f"error: {', '.join(ignored)} only steer --wild sampling and mean "
                          f"nothing for --series", file=sys.stderr)
                    return 2
                deck = comp.build_series_deck(vocab, presets[args.series], deck_id=args.deck,
                                              line=args.line, seed=args.seed, artist_card=args.artist)
            elif args.wild:
                styles = parse_styles(args.styles) if args.styles else None
                artists = [a.strip() for a in args.artists.split(",") if a.strip()] if args.artists else None
                deck = comp.build_wild_deck(vocab, deck_id=args.deck, line=args.line, count=args.count, seed=args.seed,
                                            history=history, max_overlap=args.max_overlap, styles=styles,
                                            per_style=args.per_style, artists=artists, artist_share=args.artist_share)
            else:
                deck = comp.build_deck(vocab, deck_id=args.deck, line=args.line, count=args.count, seed=args.seed,
                                       artist_card=args.artist, history=history, max_overlap=args.max_overlap)
        except comp.CompositionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.quality_toggle:
            deck["quality_toggle"] = True
        DECKS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(deck, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(deck['cards'])} cards, history {len(history)} cards)")
        if not args.series:
            print_coverage(deck["cards"])
        return 0
    deck = json.loads(Path(args.deck).read_text(encoding="utf-8"))
    if args.command == "render":
        results = render(deck, out_root=OUTPUTS, steps=args.steps, dry_run=args.dry_run)
        failed = sum(1 for r in results if r["status"] == "failed")
        print(f"{deck['deck']}: {len(results)} cards, failed {failed}")
        return 1 if failed else 0
    if args.command == "verdicts":
        return verdicts_command(deck, dry_run=args.dry_run)
    if args.command == "register":
        spec = review_spec(deck, OUTPUTS)
        if not spec:
            print("error: nothing rendered to register", file=sys.stderr)
            return 2
        spec_path = OUTPUTS / deck["deck"] / "review_spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        proc = subprocess.run([sys.executable, str(REVIEW_BATCH), "--batch", deck["deck"], "--spec", str(spec_path), "--json"],
                              capture_output=True, text=True)
        print(proc.stdout.strip() or proc.stderr.strip())
        return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
