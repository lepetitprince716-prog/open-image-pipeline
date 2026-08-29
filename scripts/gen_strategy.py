#!/usr/bin/env python3
"""Goal -> parameters decision layer (SYSTEM_PLAN step 5): turns a named goal into ONE ready
`gen.py artistic` command by combining the three queryable knowledge sources —

  * scripts/model_quirks.json     which base fits (and its neg_add / --no-freeu riders)
  * scripts/art_directions.json   which coherent slot bundle to pin (--direction)
  * outputs/.../poses/index.json  which skeleton to lock (pose_source: library)

so "goal -> base/aspect/LoRA/negatives/pose" stops being re-derived per session. Print-first:
`plan` prints the command + a decision trace; only `plan --run` executes it (gen.py then still
waits for an idle box). Data lives in scripts/gen_strategy.json.

Usage:
  gen_strategy.py list
  gen_strategy.py explain pose-locked-dynamic
  gen_strategy.py plan chiaroscuro-portrait [--subject "..."] [--box z3|zzz] [--n N] [--seed S]
                  [--pose-slug SLUG] [--json] [--run]
  gen_strategy.py validate
"""
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import art_directions
import model_quirks

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "gen_strategy.json"
Z3_MODEL_INDEX = ROOT.parent / "z3_model_index/index.json"
POSE_SOURCES = ("wildcard", "library", "none")
SLOT_FLAGS = {"medium": "--medium", "camera": "--camera", "composition": "--composition",
              "lighting": "--lighting", "pose": "--pose", "color_mood": "--color-mood",
              "expression": "--expression"}


def _load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing strategy data: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt strategy data {DATA}: {exc}")


def goals() -> dict[str, dict]:
    return _load().get("goals", {})


def _require(goal: str) -> dict:
    g = goals().get(goal)
    if not g:
        raise SystemExit(f"unknown goal {goal!r} (see: gen_strategy.py list)")
    return g


def _pick_base(g: dict, trace: list[str]) -> dict | None:
    """First base_pref entry with a quirks card that carries a ckpt path."""
    for name in g.get("base_pref", []):
        c = model_quirks.card(name)
        if c and c.get("ckpt"):
            trace.append(f"base: {c['base']} (first base_pref hit with a ckpt; "
                         f"strengths: {', '.join(c.get('strengths', [])[:2]) or '-'})")
            return c
        trace.append(f"base: skipped {name!r} (no quirks card or no ckpt)")
    return None


def _pick_pose(g: dict, seed: int | None, explicit: str | None, trace: list[str]) -> str | None:
    """A pose-library slug for pose_source=library: explicit --pose-slug wins; otherwise a
    seed-reproducible choice among the poses matching pose_tags_any."""
    if explicit:
        trace.append(f"pose: {explicit} (explicit --pose-slug)")
        return explicit
    slugs = art_directions.pose_index_slugs()
    if slugs is None:
        trace.append("pose: library requested but the pose index is not on this machine — "
                     "falling back to the card/wildcard pose")
        return None
    entries = json.loads(art_directions.POSE_INDEX.read_text(encoding="utf-8")).get("poses", [])
    want = set(g.get("pose_tags_any", []))
    deny = set(g.get("pose_tags_none", []))  # e.g. floor/prone: bad on a 9:16 full-frame (rule R1)
    pool = sorted(e["slug"] for e in entries
                  if (not want or want & set(e.get("tags", []))) and not deny & set(e.get("tags", [])))
    if not pool:
        trace.append(f"pose: no library pose matches tags {sorted(want)} minus {sorted(deny)} — falling back")
        return None
    slug = random.Random(seed).choice(pool)
    trace.append(f"pose: {slug} (from {len(pool)} matching {sorted(want) or 'any'}"
                 + (f" excluding {sorted(deny)}" if deny else "") + "; seed-reproducible)")
    return slug


def build_plan(goal: str, *, subject: str | None = None, box: str | None = None,
               n: int | None = None, seed: int | None = None,
               pose_slug: str | None = None) -> dict:
    """The full decision: returns {goal, argv, trace}. Pure planning — nothing executes."""
    g = _require(goal)
    trace: list[str] = [f"goal: {goal} — {g.get('gist', '')}"]
    subject = subject or _load().get("default_subject", "")
    if not subject:
        raise SystemExit("no subject: pass --subject (and/or set default_subject in gen_strategy.json)")
    box = box or g.get("box", "z3")
    argv: list[str] = [sys.executable, str(ROOT / "gen.py"), "artistic",
                       "--box", box, "--prompt", subject]

    card = None
    if g.get("direction"):
        card = art_directions.get(g["direction"])
        if card is None:
            raise SystemExit(f"goal {goal!r} references unknown direction card {g['direction']!r}")
        argv += ["--direction", g["direction"]]
        trace.append(f"direction: {g['direction']} · {card['name_cn']} (pins {len(card['slots'])} slots)")

    # Five-layer decoupling N1: the strategy line is character-bound through its default
    # subject; pass the pack so gen.py layers the pack's neg_guards (drift + body language)
    # and can inject color_lock for color_hijack cards. Old behavior stays reproducible.
    character = g.get("character") or _load().get("default_character")
    if character:
        argv += ["--character", character]
        trace.append(f"character pack: {character} (neg_guards layered; color_hijack injection)")

    if g.get("pose_source", "wildcard") == "library":
        slug = _pick_pose(g, seed, pose_slug, trace)
        if slug:
            argv += ["--pose-ref", slug]
    elif g.get("pose_source") == "none":
        argv += ["--pose", "none"]
        trace.append("pose: text slot dropped (pose_source=none)")

    for slot, value in (g.get("slot_overrides") or {}).items():
        argv += [SLOT_FLAGS[slot], value]
        trace.append(f"slot override: {slot} = {value}")
    if g.get("aspect"):
        argv += ["--aspect", g["aspect"]]
    argv += ["--n", str(n if n is not None else g.get("n", 6))]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if g.get("detail"):
        argv.append("--detail")
    if g.get("sfw"):
        argv.append("--sfw")
        trace.append("sfw: wd14 content gate ON (negatives alone don't guarantee a SFW image)")

    passthrough: list[str] = []
    if box == "zzz":
        trace.append("base: --box zzz forces iLustMix (the router injects it)")
    else:
        base = _pick_base(g, trace)
        if base:
            passthrough += ["--base", base["ckpt"]]
            if base.get("neg_add"):
                passthrough += ["--neg-add", base["neg_add"]]
                trace.append(f"neg_add rider from quirks: {base['neg_add'][:60]}…"
                             if len(base.get("neg_add", "")) > 60 else
                             f"neg_add rider from quirks: {base.get('neg_add')}")
            if base.get("no_freeu"):
                passthrough.append("--no-freeu")
                trace.append("flags rider from quirks: --no-freeu (base blows out under FreeU)")
    if passthrough:
        argv += ["--", *passthrough]
    return {"goal": goal, "box": box, "argv": argv, "trace": trace}


def run_argv(argv: list[str]) -> list[str]:
    """argv with --run inserted BEFORE the `--` passthrough separator: gen.py only parses router
    flags ahead of the first `--`, so appending --run at the end would silently print-only."""
    out = list(argv)
    sep = out.index("--") if "--" in out else len(out)
    out.insert(sep, "--run")
    return out


def cmd_list(_args: argparse.Namespace) -> None:
    for name, g in goals().items():
        print(f"{name:<22} dir={g.get('direction') or '-':<22} base={'/'.join(g.get('base_pref', [])) or '-':<42}"
              f" pose={g.get('pose_source', 'wildcard'):<9} {g.get('gist', '')}")


def cmd_explain(args: argparse.Namespace) -> None:
    g = _require(args.goal)
    print(json.dumps({args.goal: g}, ensure_ascii=False, indent=2))
    plan = build_plan(args.goal, seed=7)
    print("\ndecision trace (seed 7 example):")
    for t in plan["trace"]:
        print(f"  - {t}")


def cmd_plan(args: argparse.Namespace) -> None:
    plan = build_plan(args.goal, subject=args.subject, box=args.box, n=args.n,
                      seed=args.seed, pose_slug=args.pose_slug)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    for t in plan["trace"]:
        print(f"  - {t}")
    print("\n" + " ".join(shlex.quote(a) for a in plan["argv"]))
    if args.run:
        print(f"\n[run] executing via gen.py --run (waits for an idle box) ...", flush=True)
        rc = subprocess.run(run_argv(plan["argv"])).returncode
        raise SystemExit(rc)
    print("\n(print-first: add --run to execute)")


def cmd_validate(_args: argparse.Namespace) -> None:
    all_goals = goals()
    if not all_goals:
        raise SystemExit("no goals found")
    ckpt_names: set[str] | None = None
    if Z3_MODEL_INDEX.is_file():
        def _strings(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from _strings(v)
            elif isinstance(o, list):
                for v in o:
                    yield from _strings(v)
            elif isinstance(o, str):
                yield o
        ckpt_names = {s for s in _strings(json.loads(Z3_MODEL_INDEX.read_text(encoding="utf-8")))
                      if s.lower().endswith(".safetensors")}
    pose_slugs = art_directions.pose_index_slugs()
    card_names = set(art_directions.names())
    problems: list[str] = []
    for name, g in all_goals.items():
        if g.get("direction") and g["direction"] not in card_names:
            problems.append(f"{name}: direction {g['direction']!r} is not an art_directions card")
        if g.get("pose_source", "wildcard") not in POSE_SOURCES:
            problems.append(f"{name}: pose_source {g.get('pose_source')!r} not in {POSE_SOURCES}")
        for slot in (g.get("slot_overrides") or {}):
            if slot not in SLOT_FLAGS:
                problems.append(f"{name}: slot_overrides.{slot} is not a pinnable slot")
        if not g.get("base_pref"):
            problems.append(f"{name}: base_pref is empty")
        for base in g.get("base_pref", []):
            c = model_quirks.card(base)
            if c is None:
                problems.append(f"{name}: base_pref {base!r} has no model_quirks card")
            elif not c.get("ckpt"):
                problems.append(f"{name}: base_pref {base!r} quirks card has no ckpt path")
            elif ckpt_names is not None and c["ckpt"] not in ckpt_names:
                problems.append(f"{name}: ckpt {c['ckpt']!r} not in z3_model_index")
        if g.get("pose_source") == "library" and pose_slugs is not None:
            entries = json.loads(art_directions.POSE_INDEX.read_text(encoding="utf-8")).get("poses", [])
            want = set(g.get("pose_tags_any", []))
            if want and not any(want & set(e.get("tags", [])) for e in entries):
                problems.append(f"{name}: pose_tags_any {sorted(want)} matches no library pose")
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s) in {DATA.name}")
    notes = []
    if ckpt_names is None:
        notes.append("ckpt cross-check skipped (no z3_model_index)")
    if pose_slugs is None:
        notes.append("pose checks skipped (no pose index)")
    print(f"OK — {len(all_goals)} goals valid" + (f" ({'; '.join(notes)})" if notes else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="one line per goal").set_defaults(func=cmd_list)
    ex = sub.add_parser("explain", help="print a goal's card + an example decision trace")
    ex.add_argument("goal")
    ex.set_defaults(func=cmd_explain)
    pl = sub.add_parser("plan", help="print the ready gen.py command + decision trace (--run executes)")
    pl.add_argument("goal")
    pl.add_argument("--subject", default=None,
                    help="subject/character tags (default: default_subject in gen_strategy.json)")
    pl.add_argument("--box", choices=("z3", "zzz"), default=None, help="override the goal's box")
    pl.add_argument("--n", type=int, default=None)
    pl.add_argument("--seed", type=int, default=None, help="also makes the library pose pick reproducible")
    pl.add_argument("--pose-slug", default=None, help="pose_source=library: force this skeleton")
    pl.add_argument("--json", action="store_true", help="emit {goal, box, argv, trace} as JSON")
    pl.add_argument("--run", action="store_true", help="execute the command (gen.py still waits for idle)")
    pl.set_defaults(func=cmd_plan)
    sub.add_parser("validate", help="offline cross-reference checks").set_defaults(func=cmd_validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
