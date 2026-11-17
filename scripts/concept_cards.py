#!/usr/bin/env python3
"""Concept-card deck tooling for the creative line.

A concept card is the unit of creative intent: one narrative moment, one light source, one
camera decision, one palette with a single accent, one artist voice from the verified
library, and exactly one bold element. Cards are authored by hand or by a delegated model;
this script never invents cards. It validates a deck, scores each card's novelty against
the taste ledger and against its deck-mates, renders cards into ``nai_gen.py`` job files,
runs them, and emits a review-batch spec whose captions carry the card intent.

Usage:
    python3 scripts/concept_cards.py validate DECK.json
    python3 scripts/concept_cards.py novelty DECK.json [--ledger PATH] [--threshold 0.5]
    python3 scripts/concept_cards.py run DECK.json --out DIR [--seeds 1,2] [--dry-run]
                                     [--nai-args "--steps 28 --scale 5.0"]
    python3 scripts/concept_cards.py spec DIR --out SPEC.json
    python3 scripts/concept_cards.py report BATCH --deck DECK.json [--by FIELD] [--json]

Deck file: ``{"deck": "<id>", "line": "nai", "cards": [ {...}, ... ]}``. Card fields are
listed in ``REQUIRED_FIELDS``; ``base`` and ``character`` are the prompt halves consumed by
``nai_gen.py --job``; ``tags`` is the card's own tag set used only for novelty scoring.
A deck with ``"screening": true`` may use ``unvalidated`` artist cards (candidates under
test); rejected cards are refused in every mode.
Optional ``character_center`` ``{"x": 0..1, "y": 0..1}`` places the character on the
canvas (switches ``use_coords`` on). Any other extra field (``technique``, ``pair``, ...)
is carried through untouched and can be used as a ``report --by`` grouping key.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from config import OUTPUTS, ROOT

SCHEMA = "loramake-concept-deck/v1"
REQUIRED_FIELDS = (
    "id", "title", "moment", "camera", "light_source", "palette", "artist_card",
    "artist_block", "bold_element", "conservative", "plausibility", "base", "character",
    "tags",
)
LINES = ("nai", "anima", "illustrious")
WORD_BAND = (30, 130)
DEFAULT_LEDGER = OUTPUTS / "_gallery" / "taste_ledger.jsonl"
ARTIST_LIBRARY = ROOT / "scripts" / "nai_artist_strings.json"
NAI_GEN = ROOT / "scripts" / "nai_gen.py"
DEFAULT_NEGATIVE = (
    "lowres, worst quality, bad quality, jpeg artifacts, artistic error, watermark, "
    "signature, artist name, text, english text, letters, logo"
)

# Tokens that the target line is known to literalise or ignore. Sources:
# docs/artistic_method_playbook.md (double-meaning and dry-medium tokens on SDXL),
# docs/anima_line/PROMPT_TAG_RULES.md (quality and score tags on Anima).
QUALITY_TAGS = (
    "masterpiece", "best quality", "amazing quality", "very aesthetic", "absurdres",
    "highres", "score_9", "score_8_up",
)
BLACKLIST: dict[str, tuple[str, ...]] = {
    "nai": ("butterfly lighting",),
    "anima": ("butterfly lighting",) + QUALITY_TAGS,
    "illustrious": ("butterfly lighting", "risograph", "charcoal drawing", "pastel drawing"),
}
WEIGHT_RE = re.compile(r"-?\d+(?:\.\d+)?::")
ARTIST_RE = re.compile(r"artist:([a-z0-9 ()._-]+?)\s*::", re.I)


class DeckError(ValueError):
    """A user-facing deck problem."""


def load_deck(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DeckError(f"cannot read deck: {exc}") from None
    except json.JSONDecodeError as exc:
        raise DeckError(f"deck is not valid JSON: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
        raise DeckError("deck must be an object with a 'cards' list")
    if data.get("schema") != SCHEMA:
        raise DeckError(f"deck schema must be {SCHEMA!r}")
    if data.get("line", "nai") not in LINES:
        raise DeckError(f"deck line must be one of {', '.join(LINES)}")
    return data


def load_artist_library(path: Path = ARTIST_LIBRARY) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cards = data.get("cards", data)
    return cards if isinstance(cards, dict) else {}


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"[\s,]+", text) if w])


def _artist_tags(block: str) -> set[str]:
    return {m.strip().lower() for m in ARTIST_RE.findall(block)}


def _library_artist_tags(card: dict) -> set[str]:
    """Artist tags a library card requires: every ``components`` entry of kind ``artist``."""
    tags: set[str] = set()
    for component in card.get("components", []) or []:
        if not isinstance(component, dict) or component.get("kind") != "artist":
            continue
        tag = component.get("tag")
        if isinstance(tag, str) and tag.strip():
            tags.add(tag.strip().lower())
    return tags


def center_problem(center: object) -> str | None:
    """Describe what is wrong with a character center, or None when it is usable."""
    if not isinstance(center, dict):
        return "must be an object like {'x': 0.35, 'y': 0.6}"
    try:
        x, y = float(center["x"]), float(center["y"])
    except (KeyError, TypeError, ValueError):
        return "needs numeric x and y"
    if not (math.isfinite(x) and math.isfinite(y)):
        return "needs finite x and y"
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return "coordinates must lie in 0..1"
    return None


def validate_card(card: dict, line: str, library: dict[str, dict],
                  screening: bool = False) -> list[str]:
    """Return a list of problems for one card; empty means valid.

    ``screening`` decks (``"screening": true`` at deck level) may use ``unvalidated``
    library cards: that is how a candidate artist string earns its verdict. Rejected cards
    stay out in every mode.
    """
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in card:
            problems.append(f"missing field {field}")
    if problems:
        return problems
    if not isinstance(card["conservative"], list) or len(card["conservative"]) < 2:
        problems.append("conservative must list at least two elements kept conventional")
    if not isinstance(card["tags"], list) or len(card["tags"]) < 4:
        problems.append("tags must hold at least four entries")
    palette = card["palette"]
    if not isinstance(palette, dict) or not palette.get("base") or not palette.get("accent"):
        problems.append("palette needs base and accent")
    if card.get("sfw", True) is not True:
        problems.append("card is not marked sfw")
    center = card.get("character_center")
    if center is not None and center_problem(center):
        problems.append(f"character_center {center_problem(center)}")

    lib = library.get(card["artist_card"])
    if lib is None:
        problems.append(f"artist_card {card['artist_card']!r} not in library")
    elif lib.get("status") != "verified" and not (screening and lib.get("status") == "unvalidated"):
        problems.append(f"artist_card {card['artist_card']!r} is not verified")
    else:
        wanted = _library_artist_tags(lib)
        used = _artist_tags(card["artist_block"])
        if not wanted:
            problems.append(f"library card {card['artist_card']!r} lists no artist components")
        elif not used:
            problems.append("artist_block carries no artist: tag")
        elif used != wanted:
            problems.append(f"artist_block tags {sorted(used)} differ from card {sorted(wanted)}")

    prompt_text = f"{card['base']} {card['character']}".lower()
    for token in BLACKLIST.get(line, ()):
        if token in prompt_text:
            problems.append(f"blacklisted token for {line}: {token!r}")
    if line == "anima":
        for match in ARTIST_RE.findall(card["artist_block"]):
            if "_" in match:
                problems.append(f"anima artist tag must not contain '_': {match!r}")
    words = _word_count(card["base"]) + _word_count(card["character"])
    if not WORD_BAND[0] <= words <= WORD_BAND[1]:
        problems.append(f"prompt word count {words} outside band {WORD_BAND}")
    return problems


def validate_deck(deck: dict, library: dict[str, dict]) -> dict[str, list[str]]:
    line = deck.get("line", "nai")
    report: dict[str, list[str]] = {}
    seen: set[str] = set()
    for index, card in enumerate(deck["cards"]):
        key = card.get("id") if isinstance(card, dict) else None
        label = key or f"card#{index}"
        if not isinstance(card, dict):
            report[label] = ["card is not an object"]
            continue
        problems = validate_card(card, line, library, bool(deck.get("screening")))
        if key in seen:
            problems.append("duplicate card id")
        seen.add(key or label)
        if problems:
            report[label] = problems
    return report


def prompt_tag_set(prompt: str) -> set[str]:
    """Split a NAI prompt into a lowercase tag set, dropping numeric weights and artist tags."""
    tags: set[str] = set()
    for raw in prompt.split(","):
        token = WEIGHT_RE.sub("", raw).replace("::", "").strip().lower()
        if not token or token.startswith("artist:"):
            continue
        tags.add(token)
    return tags


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ledger_winners(ledger_path: Path, line: str) -> list[tuple[str, set[str]]]:
    """Return (name, tag set) for love/yes records on ``line`` that carry a recipe prompt."""
    winners: list[tuple[str, set[str]]] = []
    if not ledger_path.exists():
        return winners
    for raw in ledger_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("line") != line or record.get("verdict") not in ("love", "yes"):
            continue
        recipe = record.get("recipe") or {}
        prompt = recipe.get("prompt")
        if not isinstance(prompt, str):
            continue
        parts = [prompt] + [p for p in recipe.get("character_prompts") or [] if isinstance(p, str)]
        winners.append((record.get("name", "?"), prompt_tag_set(", ".join(parts))))
    return winners


def card_tag_set(card: dict) -> set[str]:
    """The card's comparable tag set: base plus character prompt, the same fields a ledger
    winner carries. The editorial ``tags`` field is documentation, not part of the score."""
    return prompt_tag_set(f"{card.get('base', '')}, {card.get('character', '')}")


def novelty_report(deck: dict, ledger_path: Path, threshold: float) -> list[dict]:
    line = deck.get("line", "nai")
    winners = ledger_winners(ledger_path, line)
    cards = [c for c in deck["cards"] if isinstance(c, dict)]
    sets = {c["id"]: card_tag_set(c) for c in cards}
    rows: list[dict] = []
    for card in cards:
        mine = sets[card["id"]]
        nearest_winner = max(((jaccard(mine, s), n) for n, s in winners), default=(0.0, None))
        # Paired A/B arms share everything but one device; the other arm is not a
        # deck-mate for novelty purposes.
        mates = [o for o in cards if o["id"] != card["id"]
                 and not (card.get("pair") and o.get("pair") == card.get("pair"))]
        nearest_mate = max(
            ((jaccard(mine, sets[o["id"]]), o["id"]) for o in mates),
            default=(0.0, None),
        )
        rows.append({
            "id": card["id"],
            "vs_winner": round(nearest_winner[0], 3),
            "winner": nearest_winner[1],
            "vs_mate": round(nearest_mate[0], 3),
            "mate": nearest_mate[1],
            "too_close": nearest_winner[0] >= threshold or nearest_mate[0] >= threshold,
        })
    return rows


def card_job(card: dict, model: str) -> dict:
    base = f"{card['artist_block']}, {card['base']}"
    negative = DEFAULT_NEGATIVE
    if card.get("negative_add"):
        negative = f"{negative}, {card['negative_add']}"
    character: dict = {"prompt": card["character"]}
    center = card.get("character_center")
    if center is not None:
        character["center"] = {"x": float(center["x"]), "y": float(center["y"])}
    return {
        "model": model,
        "base": base,
        "characters": [character],
        "negative": negative,
    }


def write_jobs(deck: dict, out_dir: Path, model: str) -> list[tuple[dict, Path]]:
    jobs_dir = out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[dict, Path]] = []
    for card in deck["cards"]:
        path = jobs_dir / f"{card['id']}.job.json"
        path.write_text(json.dumps(card_job(card, model), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written.append((card, path))
    return written


def run_deck(deck: dict, out_dir: Path, model: str, seeds: Sequence[int],
             extra: Sequence[str], dry_run: bool, force: bool = False) -> list[dict]:
    """Render every card for every seed through nai_gen.py; return the run index.

    A finished run leaves ``run_index.json`` behind; rerunning into the same directory
    would append new images and manifest rows beside the old ones, so it is refused
    unless ``force`` is set.
    """
    if not dry_run and (out_dir / "run_index.json").exists() and not force:
        raise DeckError(f"{out_dir} already holds a run; pick a new --out or pass --force")
    index: list[dict] = []
    for card, job_path in write_jobs(deck, out_dir, model):
        card_dir = out_dir / card["id"]
        card_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            command = [
                sys.executable, str(NAI_GEN), "--job", str(job_path), "--seed", str(seed),
                "--out", str(card_dir), *extra,
            ]
            if dry_run:
                command.append("--dry-run")
            print(f"[concept] {card['id']} seed={seed}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True)
            saved = [ln.strip() for ln in completed.stdout.splitlines()
                     if ln.strip().endswith(".png")]
            entry = {"id": card["id"], "seed": seed, "returncode": completed.returncode,
                     "files": saved, "dry_run": dry_run}
            if completed.returncode != 0:
                entry["stderr"] = completed.stderr.strip()[-800:]
                print(f"[concept] {card['id']} seed={seed} FAILED: {entry['stderr']}",
                      file=sys.stderr, flush=True)
            index.append(entry)
    # A dry run must not leave the marker that makes the next live run refuse the
    # directory; it only proves the job files parse.
    if not dry_run:
        (out_dir / "run_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


CAPTION_LIMIT = 200  # review_batch.py rejects longer captions


def caption_for(card: dict, seed: int) -> str:
    text = (f"[{card['id']}] {card['title']} · {card['artist_card']} · seed {seed} — "
            f"{card['moment']} 反常元素：{card['bold_element']}")
    if len(text) > CAPTION_LIMIT:
        text = text[:CAPTION_LIMIT - 1] + "…"
    return text


def build_spec(deck: dict, out_dir: Path) -> list[dict]:
    index_path = out_dir / "run_index.json"
    if not index_path.exists():
        raise DeckError(f"no run_index.json under {out_dir}; run the deck first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cards = {c["id"]: c for c in deck["cards"]}
    spec: list[dict] = []
    for entry in index:
        card = cards.get(entry["id"])
        if card is None:
            continue
        for file_path in entry.get("files", []):
            spec.append({"path": file_path, "caption": caption_for(card, entry["seed"])})
    return spec


CAPTION_KEY_RE = re.compile(r"^\[(?P<id>[^\]\s]+)\].*?\bseed (?P<seed>\d+)")
VERDICT_ORDER = ("love", "yes", "no")


def parse_caption_key(caption: str) -> tuple[str, int] | None:
    """Recover ``(card_id, seed)`` from a caption written by ``caption_for``."""
    match = CAPTION_KEY_RE.match(caption or "")
    if match is None:
        return None
    return match.group("id"), int(match.group("seed"))


def join_verdicts(deck: dict, result: dict) -> list[dict]:
    """Attach each reviewed render's verdict to its card.

    ``result`` is the report from ``review_result.read_result``. Renders whose caption does
    not name a card in the deck are skipped. Comment-only items get verdict ``comment``.
    """
    cards = {c["id"]: c for c in deck["cards"] if isinstance(c, dict)}
    rows: dict[int, dict] = {}

    def add(item: dict, verdict: str) -> None:
        key = parse_caption_key(str(item.get("caption", "")))
        if key is None or key[0] not in cards:
            return
        card_id, seed = key
        rows[int(item["i"])] = {
            "i": int(item["i"]), "id": card_id, "seed": seed, "verdict": verdict,
            "reasons": list(item.get("reasons") or []), "card": cards[card_id],
        }

    for verdict in VERDICT_ORDER:
        for item in result.get(verdict) or []:
            add(item, verdict)
    for item in (result.get("comments") or {}).values():
        if int(item["i"]) not in rows:
            add(item, "comment")
    return [rows[i] for i in sorted(rows)]


def _field(card: dict, path: str) -> str:
    value: object = card
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return "(none)"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def tally(rows: Iterable[dict], by: str = "id") -> list[dict]:
    """Count verdicts per value of a card field. Comment-only rows stay out of the rates."""
    groups: dict[str, dict] = {}
    for row in rows:
        key = _field(row["card"], by)
        g = groups.setdefault(key, {"key": key, "love": 0, "yes": 0, "no": 0, "comment": 0,
                                    "reasons": {}})
        g[row["verdict"]] += 1
        for reason in row["reasons"]:
            g["reasons"][reason] = g["reasons"].get(reason, 0) + 1
    out = []
    for g in groups.values():
        decided = g["love"] + g["yes"] + g["no"]
        g["decided"] = decided
        g["keep_rate"] = round((g["love"] + g["yes"]) / decided, 3) if decided else 0.0
        g["love_rate"] = round(g["love"] / decided, 3) if decided else 0.0
        out.append(g)
    return sorted(out, key=lambda g: g["key"])


def _cmd_validate(args: argparse.Namespace) -> int:
    deck = load_deck(Path(args.deck))
    report = validate_deck(deck, load_artist_library())
    if not report:
        print(f"deck ok: {len(deck['cards'])} cards")
        return 0
    for card_id, problems in report.items():
        for problem in problems:
            print(f"{card_id}: {problem}")
    return 1


def _cmd_novelty(args: argparse.Namespace) -> int:
    deck = load_deck(Path(args.deck))
    rows = novelty_report(deck, Path(args.ledger), args.threshold)
    print(f"{'id':<28} {'vs_winner':>9} {'winner':<28} {'vs_mate':>7} {'mate':<28} flag")
    flagged = 0
    for r in rows:
        flag = "TOO CLOSE" if r["too_close"] else ""
        flagged += bool(r["too_close"])
        print(f"{r['id']:<28} {r['vs_winner']:>9.3f} {str(r['winner'])[:28]:<28} "
              f"{r['vs_mate']:>7.3f} {str(r['mate'])[:28]:<28} {flag}")
    print(f"flagged={flagged} threshold={args.threshold}")
    return 1 if flagged else 0


def _cmd_run(args: argparse.Namespace) -> int:
    deck = load_deck(Path(args.deck))
    report = validate_deck(deck, load_artist_library())
    if report:
        raise DeckError("deck fails validation; run 'validate' first")
    if deck.get("line", "nai") != "nai":
        raise DeckError(f"run only renders 'nai' decks; this deck is {deck.get('line')!r} "
                        "and has no render adapter yet")
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        raise DeckError(f"--seeds must be comma-separated integers, got {args.seeds!r}") from None
    if not seeds:
        raise DeckError("at least one seed is required")
    out_dir = Path(args.out)
    extra = shlex.split(args.nai_args)
    index = run_deck(deck, out_dir, args.model, seeds, extra, args.dry_run, args.force)
    failed = sum(1 for e in index if e["returncode"] != 0)
    print(f"runs={len(index)} failed={failed} out={out_dir}")
    return 1 if failed else 0


def _cmd_spec(args: argparse.Namespace) -> int:
    deck = load_deck(Path(args.deck))
    spec = build_spec(deck, Path(args.dir))
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(f"spec items={len(spec)} -> {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from review_result import ReviewResultError, read_result

    deck = load_deck(Path(args.deck))
    try:
        result = read_result(args.batch)
    except ReviewResultError as exc:
        raise DeckError(str(exc)) from None
    rows = join_verdicts(deck, result)
    groups = tally(rows, args.by)
    if args.json:
        print(json.dumps({"batch": args.batch, "by": args.by, "rows": [
            {k: v for k, v in r.items() if k != "card"} for r in rows], "groups": groups},
            ensure_ascii=False))
        return 0
    print(f"batch {args.batch}: {len(rows)} renders matched to {deck.get('deck', '?')}")
    for row in rows:
        suffix = f" ({', '.join(row['reasons'])})" if row["reasons"] else ""
        print(f"  [{row['i']:>3}] {row['id']:<28} seed {row['seed']:<10} {row['verdict']}{suffix}")
    print(f"\nby {args.by}:")
    print(f"  {'key':<28} {'love':>4} {'yes':>4} {'no':>4} {'cmt':>4} {'keep':>6} {'love%':>6}  reasons")
    for g in groups:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(g["reasons"].items()))
        print(f"  {g['key'][:28]:<28} {g['love']:>4} {g['yes']:>4} {g['no']:>4} {g['comment']:>4} "
              f"{g['keep_rate']:>6.2f} {g['love_rate']:>6.2f}  {reasons}")
    pending = result.get("pending") or []
    if pending:
        print(f"\npending: {len(pending)} render(s) not yet decided")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concept-card deck tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="schema, artist library and blacklist checks")
    p.add_argument("deck")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("novelty", help="Jaccard distance to ledger winners and deck-mates")
    p.add_argument("deck")
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=_cmd_novelty)

    p = sub.add_parser("run", help="render every card through nai_gen.py")
    p.add_argument("deck")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nai-diffusion-5-full")
    p.add_argument("--seeds", default="1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="render into a directory that already holds a finished run")
    p.add_argument("--nai-args", default="",
                   help="extra flags passed through to nai_gen.py, as one quoted string")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("spec", help="write a review_batch.py --spec file for a finished run")
    p.add_argument("dir")
    p.add_argument("--deck", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_spec)

    p = sub.add_parser("report", help="join a review batch's verdicts back onto the deck")
    p.add_argument("batch", help="batch ID printed by review_batch.py")
    p.add_argument("--deck", required=True)
    p.add_argument("--by", default="id",
                   help="card field to group by; dotted paths like palette.accent work")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DeckError as exc:
        print(f"concept_cards.py: error: {exc}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(f"concept_cards.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
