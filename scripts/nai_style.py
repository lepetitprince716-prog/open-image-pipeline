#!/usr/bin/env python3
"""NovelAI V5 style picker over the artist-string library.

The library (``scripts/nai_artist_strings.json``) records which artist strings the user
has verified; this script turns it into a callable surface:

  nai_style.py pick "<intent>" [--top 5] [--include unvalidated|rejected] [--json]
  nai_style.py block <card_id> [--dark-night] [--json]
  nai_style.py stats [--id <card_id>] [--json]
  nai_style.py render --style <card_id>|auto --character "<block>" [--base "<block>"]
                      [--intent "<text>"] [--seed N] [--n 1] [--landscape|--square]
                      [--paid] [--dry-run] [--json]
  nai_style.py tier <card_id> love|like|none

``pick`` ranks cards by a lexical match of the intent against the card faces (name,
intent, five axes, pairing biases) multiplied by a prior built from the taste ledger
(love / yes / no verdicts on renders that used the card) and the user's ``tier``
bookmark. ``stats`` is that ledger join on its own. ``block`` prints the prompt block a
card composes to; ``render`` builds a ``nai_gen.py --job`` from a style plus the
caller's scene and character blocks, renders on the free tier and registers a gallery
view batch. Every verb prints a JSON envelope as its last stdout line under ``--json``.

Nothing here authors prompts: scene and character blocks come from the caller (see
``docs/runbooks/concept_card_runbook.md`` for the recipe that scores).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nai_artist_strings as nas  # noqa: E402
from concept_cards import DEFAULT_NEGATIVE  # noqa: E402
from loramake.model_resources import field_hit, tokenize_query  # noqa: E402

SCHEMA = "loramake.nai-style/v1"
LEDGER = ROOT / "outputs" / "_gallery" / "taste_ledger.jsonl"
DECKS = ROOT / "configs" / "concept_decks"
SYNONYMS = ROOT / "configs" / "nai_style_synonyms.json"
NAI_GEN = SCRIPTS / "nai_gen.py"
REVIEW_BATCH = SCRIPTS / "review_batch.py"

DARK_NIGHT_BASE = ("deep night, dark background, strong shadows, low key lighting, "
                   "-2::overexposed, glowing ::")
DARK_NIGHT_NEGATIVE = "overexposed, bright, fog, haze, washed out"

FREE_MODEL = "nai-diffusion-5-full"
# V5 Curated tracked artist styles at least as well as Full in the round-13 control
# (six loved artists, same seed: Curated 2 love / Full 1 love / no rejections); SFW only.
CURATED_MODEL = "nai-diffusion-5-curated"
FREE_STEPS = 28
FREE_SCALE = 5.0
FREE_SIZES = {"portrait": (832, 1216), "landscape": (1216, 832), "square": (1024, 1024)}
PAID_SIZE = (1024, 1536)
PAID_STEPS = 40

FIELD_WEIGHTS = {"name_cn": 3.0, "intent": 3.0, "axes": 2.0, "pairing": 1.0, "id": 1.0}
TIER_FACTOR = {"love": 1.3, "like": 1.1}
MIN_DECIDED_FOR_PRIOR = 1

CAPTION_KEY_RE = re.compile(r"^\[(?P<id>[^\]\s]+)\]")
CAPTION_ARTIST_RE = re.compile(r"·\s*(?P<artist>[a-z][a-z0-9_]*)\s*·\s*seed\s+\d+")
ARTIST_TAG_RE = re.compile(r"artist:([a-z0-9 ()._-]+?)\s*::", re.I)

EXIT_OK, EXIT_INVALID, EXIT_RENDER = 0, 2, 3


class StyleError(ValueError):
    """Bad arguments or an unusable card."""


# --------------------------------------------------------------------------- helpers

def load_synonyms(path: Path = SYNONYMS) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data.get("synonyms", {}).items()}


def expand_query(text: str, synonyms: dict[str, list[str]]) -> list[str]:
    """Tokenize the intent and add the English words its Chinese runs map to."""
    tokens = tokenize_query(text)
    extra: list[str] = []
    for tok in tokens:
        if tok.isascii():
            continue
        for key, words in synonyms.items():
            if key in tok:
                extra.extend(w.lower() for w in words)
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens + extra:
        for part in tok.split():
            if part and part not in seen:
                seen.add(part)
                out.append(part)
    return out


def card_fields(card_id: str, card: dict) -> dict[str, str]:
    axes = card.get("axes", {}) or {}
    pairing = card.get("pairing", {}) or {}
    return {
        "name_cn": str(card.get("name_cn", "")),
        "intent": str(card.get("intent", "")),
        "axes": " ".join(" ".join(v) for v in axes.values() if isinstance(v, list)),
        "pairing": " ".join(
            " ".join(pairing.get(k, []) or []) for k in ("scene_bias", "lighting_bias")
        ) + " " + str(pairing.get("note", "") or ""),
        "id": card_id.replace("_", " "),
    }


def lexical_score(tokens: list[str], fields: dict[str, str]) -> tuple[float, list[str]]:
    if not tokens:
        return 0.0, []
    total = 0.0
    why: list[str] = []
    for tok in tokens:
        for name, text in fields.items():
            if text and field_hit(tok, text.lower()):
                total += FIELD_WEIGHTS[name]
                why.append(f"{tok}@{name}")
    return total / (max(FIELD_WEIGHTS.values()) * len(tokens)), why


def prior_score(record: dict | None, tier: str | None) -> float:
    love = record["love"] if record else 0
    yes = record["yes"] if record else 0
    decided = record["decided"] if record else 0
    laplace = (love + 0.5 * yes + 1.0) / (decided + 2.0)
    return laplace * TIER_FACTOR.get(tier or "", 1.0)


# --------------------------------------------------------------------------- ledger join

def load_deck_index(decks_dir: Path = DECKS) -> dict[str, dict]:
    """Concept-card id -> {deck, artist_card, content} across every deck on disk."""
    index: dict[str, dict] = {}
    for path in sorted(decks_dir.glob("*.json")):
        try:
            deck = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for card in deck.get("cards", []) or []:
            cid = card.get("id")
            if isinstance(cid, str) and card.get("artist_card"):
                index[cid] = {"deck": deck.get("deck", path.stem),
                              "artist_card": card["artist_card"],
                              "content": card.get("content")}
    return index


def resolve_artist_card(row: dict, deck_index: dict[str, dict],
                        library: dict[str, dict]) -> str | None:
    """Map one ledger row to a library card id, or None when it did not use one."""
    caption = row.get("caption") or ""
    m = CAPTION_KEY_RE.match(caption)
    if m and m.group("id") in deck_index:
        return deck_index[m.group("id")]["artist_card"]
    m = CAPTION_ARTIST_RE.search(caption)
    if m and m.group("artist") in library:
        return m.group("artist")
    stem = Path(row.get("name") or "").stem
    if stem in library:
        return stem
    recipe = row.get("recipe") or {}
    prompt = recipe.get("prompt") if isinstance(recipe, dict) else None
    if prompt:
        tags = {t.strip().lower() for t in ARTIST_TAG_RE.findall(prompt)}
        if tags:
            for cid, card in library.items():
                if nas_artist_tags(card) == tags:
                    return cid
    return None


def nas_artist_tags(card: dict) -> set[str]:
    return {c["tag"].strip().lower() for c in card.get("components", []) or []
            if isinstance(c, dict) and c.get("kind") == "artist"}


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def artist_stats(rows: list[dict], library: dict[str, dict],
                 deck_index: dict[str, dict]) -> dict[str, dict]:
    """Per-card verdict tallies from the ledger rows that used the card."""
    stats: dict[str, dict] = defaultdict(lambda: {
        "love": 0, "yes": 0, "no": 0, "pending": 0, "decided": 0,
        "reasons": Counter(), "batches": set(), "first": None, "last": None,
    })
    for row in rows:
        if row.get("line") != "nai":
            continue
        cid = resolve_artist_card(row, deck_index, library)
        if cid is None:
            continue
        verdict = row.get("verdict")
        if verdict not in ("love", "yes", "no", "pending"):
            continue
        rec = stats[cid]
        rec[verdict] += 1
        if verdict != "pending":
            rec["decided"] += 1
        if verdict == "no":
            rec["reasons"].update(row.get("reasons") or [])
        rec["batches"].add(row.get("batch"))
        stamp = row.get("batch_time") or ""
        if stamp:
            rec["first"] = min(rec["first"] or stamp, stamp)
            rec["last"] = max(rec["last"] or stamp, stamp)
    out: dict[str, dict] = {}
    for cid, rec in stats.items():
        decided = rec["decided"]
        out[cid] = {
            "love": rec["love"], "yes": rec["yes"], "no": rec["no"], "pending": rec["pending"],
            "decided": decided,
            "love_rate": round(rec["love"] / decided, 3) if decided else None,
            "keep_rate": round((rec["love"] + rec["yes"]) / decided, 3) if decided else None,
            "reasons": dict(rec["reasons"].most_common()),
            "batches": len(rec["batches"]),
            "first": rec["first"], "last": rec["last"],
        }
    return out


# --------------------------------------------------------------------------- verbs

def pick(intent: str, library: dict[str, dict], stats: dict[str, dict],
         include: set[str], top: int, synonyms: dict[str, list[str]]) -> list[dict]:
    tokens = expand_query(intent, synonyms)
    statuses = {"verified"} | include
    ranked: list[dict] = []
    for cid, card in library.items():
        if card.get("status") not in statuses:
            continue
        lex, why = lexical_score(tokens, card_fields(cid, card))
        if tokens and lex == 0.0:
            continue
        record = stats.get(cid)
        prior = prior_score(record, card.get("tier"))
        score = (lex if tokens else 1.0) * prior
        ranked.append({
            "id": cid, "name_cn": card.get("name_cn"), "intent": card.get("intent"),
            "status": card.get("status"), "tier": card.get("tier"),
            "score": round(score, 4), "lexical": round(lex, 4), "prior": round(prior, 4),
            "love": record["love"] if record else 0, "yes": record["yes"] if record else 0,
            "no": record["no"] if record else 0, "decided": record["decided"] if record else 0,
            "love_rate": record["love_rate"] if record else None,
            "block": nas.compose_block(card), "neg_add": card.get("neg_add", ""),
            "why": why,
        })
    ranked.sort(key=lambda r: (-r["score"], -r["prior"], -r["decided"], r["id"]))
    return ranked[:top]


def block_for(card_id: str, library: dict[str, dict], dark_night: bool) -> dict:
    card = library.get(card_id)
    if card is None:
        raise StyleError(f"unknown card {card_id!r}")
    negative_add = card.get("neg_add", "") or ""
    base_tail = ""
    if dark_night:
        base_tail = DARK_NIGHT_BASE
        negative_add = ", ".join(p for p in (negative_add, DARK_NIGHT_NEGATIVE) if p)
    return {"id": card_id, "name_cn": card.get("name_cn"), "status": card.get("status"),
            "block": nas.compose_block(card), "negative_add": negative_add,
            "base_tail": base_tail}


def render_size(landscape: bool, square: bool, paid: bool) -> tuple[int, int]:
    if paid:
        return PAID_SIZE
    if landscape and square:
        raise StyleError("--landscape and --square are mutually exclusive")
    if landscape:
        return FREE_SIZES["landscape"]
    if square:
        return FREE_SIZES["square"]
    return FREE_SIZES["portrait"]


def build_job(card_id: str, library: dict[str, dict], base: str | None, character: str,
              dark_night: bool, model: str = FREE_MODEL) -> dict:
    """The nai_gen.py --job payload: style block first, then the caller's scene block."""
    card = library.get(card_id)
    if card is None:
        raise StyleError(f"unknown card {card_id!r}")
    if not character.strip():
        raise StyleError("--character must not be empty")
    scene = (base or "").strip()
    if not scene:
        scene_bias = (card.get("pairing", {}) or {}).get("scene_bias") or []
        scene = scene_bias[0] if scene_bias else "simple background"
    parts = [nas.compose_block(card), scene]
    if dark_night:
        parts.append(DARK_NIGHT_BASE)
    negative = DEFAULT_NEGATIVE
    for extra in (card.get("neg_add", ""), DARK_NIGHT_NEGATIVE if dark_night else ""):
        if extra:
            negative = f"{negative}, {extra}"
    return {"model": model, "base": ", ".join(parts),
            "characters": [{"prompt": character.strip()}], "negative": negative}


def nai_gen_argv(job_path: Path, out_dir: Path, seed: int | None, size: tuple[int, int],
                 steps: int, dry_run: bool) -> list[str]:
    argv = [sys.executable, str(NAI_GEN), "--job", str(job_path), "--out", str(out_dir),
            "--width", str(size[0]), "--height", str(size[1]), "--steps", str(steps),
            "--scale", str(FREE_SCALE)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if dry_run:
        argv.append("--dry-run")
    return argv


def run_render(job: dict, out_dir: Path, seeds: list[int | None], size: tuple[int, int],
               steps: int, dry_run: bool) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    for seed in seeds:
        argv = nai_gen_argv(job_path, out_dir, seed, size, steps, dry_run)
        completed = subprocess.run(argv, capture_output=True, text=True)
        files = [ln.strip() for ln in completed.stdout.splitlines()
                 if ln.strip().endswith(".png")]
        entry = {"seed": seed, "returncode": completed.returncode, "files": files}
        if completed.returncode != 0:
            entry["stderr"] = completed.stderr.strip()[-800:]
        results.append(entry)
    return results


def register_view_batch(files: list[str], caption: str, out_dir: Path) -> dict | None:
    """Register the rendered files as a :8771 view batch; the spec stays beside the renders."""
    if not files:
        return None
    spec = [{"path": f, "caption": caption[:200]} for f in files]
    spec_path = out_dir / "view_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(REVIEW_BATCH), "--mode", "view", "--spec", str(spec_path), "--json"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()[-400:]}
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"error": "review_batch.py printed no JSON"}


def set_tier(card_id: str, tier: str | None, path: Path = nas.DATA) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    card = data["cards"].get(card_id)
    if card is None:
        raise StyleError(f"unknown card {card_id!r}")
    before = card.get("tier")
    if tier is None:
        card.pop("tier", None)
    else:
        card["tier"] = tier
    problems: list[str] = []
    nas._validate_card(card_id, card, problems)
    if problems:
        raise StyleError("; ".join(problems))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {"id": card_id, "before": before, "after": tier}


# --------------------------------------------------------------------------- CLI

def emit(payload: dict, as_json: bool, human: str = "") -> None:
    if human and not as_json:
        print(human)
    if as_json:
        print(json.dumps({"schema": SCHEMA, **payload}, ensure_ascii=False))


def cmd_pick(args: argparse.Namespace) -> int:
    library = nas.cards()
    stats = artist_stats(read_ledger(), library, load_deck_index())
    include = set(args.include or [])
    ranked = pick(args.intent, library, stats, include, args.top, load_synonyms())
    lines = []
    for i, r in enumerate(ranked, 1):
        rate = "-" if r["love_rate"] is None else f"{r['love_rate']:.2f}"
        badge = {"love": "★", "like": "✓"}.get(r["tier"] or "", " ")
        lines.append(f"{i:>2} {badge} {r['id']:<26} {r['name_cn']:　<6} score={r['score']:.3f}"
                     f" love={rate} ({r['love']}/{r['decided']})  {r['block']}")
    emit({"action": "pick", "intent": args.intent, "results": ranked}, args.json,
         "\n".join(lines) if lines else "no card matches; try fewer words or --include unvalidated")
    return EXIT_OK if ranked else EXIT_INVALID


def cmd_block(args: argparse.Namespace) -> int:
    info = block_for(args.card_id, nas.cards(), args.dark_night)
    human = info["block"]
    if info["negative_add"]:
        human += f"\nnegative_add: {info['negative_add']}"
    if info["base_tail"]:
        human += f"\nbase_tail: {info['base_tail']}"
    emit({"action": "block", **info}, args.json, human)
    return EXIT_OK


def cmd_stats(args: argparse.Namespace) -> int:
    library = nas.cards()
    stats = artist_stats(read_ledger(), library, load_deck_index())
    if args.id:
        if args.id not in library:
            raise StyleError(f"unknown card {args.id!r}")
        stats = {args.id: stats.get(args.id, {"love": 0, "yes": 0, "no": 0, "pending": 0,
                                                "decided": 0, "love_rate": None,
                                                "keep_rate": None, "reasons": {},
                                                "batches": 0, "first": None, "last": None})}
    rows = sorted(stats.items(), key=lambda kv: (-(kv[1]["love_rate"] or 0), -kv[1]["decided"], kv[0]))
    lines = [f"{'card':<26} {'love':>4} {'yes':>4} {'no':>4} {'love%':>6} {'keep%':>6}  reasons"]
    for cid, rec in rows:
        lr = "-" if rec["love_rate"] is None else f"{rec['love_rate']:.2f}"
        kr = "-" if rec["keep_rate"] is None else f"{rec['keep_rate']:.2f}"
        reasons = " ".join(f"{k} {v}" for k, v in rec["reasons"].items())
        lines.append(f"{cid:<26} {rec['love']:>4} {rec['yes']:>4} {rec['no']:>4} {lr:>6} {kr:>6}  {reasons}")
    emit({"action": "stats", "cards": dict(rows)}, args.json, "\n".join(lines))
    return EXIT_OK


def cmd_render(args: argparse.Namespace) -> int:
    library = nas.cards()
    chosen = args.style
    why: list[str] = []
    if chosen == "auto":
        if not args.intent:
            raise StyleError("--style auto needs --intent")
        stats = artist_stats(read_ledger(), library, load_deck_index())
        ranked = pick(args.intent, library, stats, set(), 1, load_synonyms())
        if not ranked:
            raise StyleError("no verified card matches the intent")
        chosen, why = ranked[0]["id"], ranked[0]["why"]
    if chosen not in library:
        raise StyleError(f"unknown card {chosen!r}")
    if library[chosen].get("status") == "rejected":
        raise StyleError(f"{chosen} is rejected; pick a verified card")
    size = render_size(args.landscape, args.square, args.paid)
    steps = PAID_STEPS if args.paid else FREE_STEPS
    job = build_job(chosen, library, args.base, args.character, args.dark_night,
                    CURATED_MODEL if args.curated else FREE_MODEL)
    seeds: list[int | None] = [args.seed + i if args.seed is not None else None
                               for i in range(args.n)]
    out_dir = Path(args.out) if args.out else ROOT / "outputs" / "nai" / "style" / chosen
    results = run_render(job, out_dir, seeds, size, steps, args.dry_run)
    files = [f for r in results for f in r["files"]]
    failed = [r for r in results if r["returncode"] != 0]
    batch = None if args.dry_run else register_view_batch(
        files, f"[{chosen}] {library[chosen].get('name_cn', '')} · {args.intent or args.base or ''}",
        out_dir)
    payload = {"action": "render", "style": chosen, "why": why, "job": job,
               "size": list(size), "steps": steps, "paid": args.paid, "dry_run": args.dry_run,
               "results": results, "files": files, "batch": batch}
    human = f"style {chosen} · {library[chosen].get('name_cn', '')}"
    if why:
        human += f"  (why: {', '.join(why)})"
    human += f"\nbase: {job['base']}\ncharacter: {job['characters'][0]['prompt']}"
    if files:
        human += "\n" + "\n".join(files)
    if batch and batch.get("url"):
        human += f"\nreview: {batch['url']}"
    if failed:
        human += "\nFAILED: " + "; ".join(r.get("stderr", "")[-200:] for r in failed)
    emit(payload, args.json, human)
    return EXIT_RENDER if failed else EXIT_OK


def cmd_tier(args: argparse.Namespace) -> int:
    tier = None if args.tier == "none" else args.tier
    info = set_tier(args.card_id, tier)
    emit({"action": "tier", **info}, args.json, f"{args.card_id}: {info['before']} -> {info['after']}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NovelAI V5 style picker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick", help="rank library cards for an intent")
    p.add_argument("intent", nargs="?", default="")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--include", action="append", choices=["unvalidated", "rejected"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("block", help="print the prompt block a card composes to")
    p.add_argument("card_id")
    p.add_argument("--dark-night", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_block)

    p = sub.add_parser("stats", help="ledger verdicts per card")
    p.add_argument("--id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("render", help="render one style with the caller's scene and character")
    p.add_argument("--style", required=True, help="card id or 'auto'")
    p.add_argument("--intent", help="free text; required with --style auto")
    p.add_argument("--base", help="scene block; defaults to the card's first scene_bias")
    p.add_argument("--character", required=True, help="character block")
    p.add_argument("--dark-night", action="store_true")
    p.add_argument("--curated", action="store_true",
                   help="render on V5 Curated instead of Full (same free-tier size and steps)")
    p.add_argument("--seed", type=int)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--landscape", action="store_true")
    p.add_argument("--square", action="store_true")
    p.add_argument("--paid", action="store_true", help="1024x1536 / 40 steps (costs Anlas)")
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("tier", help="set or clear the taste bookmark on a card")
    p.add_argument("card_id")
    p.add_argument("tier", choices=[*nas.TIERS, "none"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tier)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StyleError as exc:
        print(f"nai_style: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
