#!/usr/bin/env python3
"""Standing-art set for one design card: plan main views, bind face differentials, gate, register.

Usage:
    python3 scripts/character_set.py identity CARD --out DIR [--engine anima|illustrious|nai]
                                     [--lora FILE:W ...] [--artist-mix CARD_ID] [--seeds 7,8,9] [--outfit ID] [--expression N]
    python3 scripts/character_set.py plan CARD --out DIR --outfit ID [--anchor IMAGE]
                                     [--artist-mix CARD_ID] [--seed N] [--strength S] [--fidelity F]
    python3 scripts/character_set.py faces CARD --out DIR --bust IMAGE [--lora FILE[:STR] ...] [--seed N]
    python3 scripts/character_set.py bind --out DIR --expression ID --file IMAGE
    python3 scripts/character_set.py gate --out DIR [--anchor IMAGE] [--min-hold 0.85] [--max-diff 0.06] [--json]
    python3 scripts/character_set.py register --out DIR [--json]

``plan`` writes the NovelAI jobs (anchor as Precise Reference) and ``commands_main.sh``;
``faces`` binds the rendered neutral bust to every expression and writes ``commands_faces.sh``
(BOXA face-mask repaints); ``bind`` records a rendered differential; ``gate`` runs the CCIP
identity check against the anchor; ``register`` opens a review batch. Nothing here renders.

Exit codes: 0 ok · 1 gate failed or refused input · 2 usage.
Authority: docs/runbooks/character_design_runbook.md §9 (standing-art set).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import character_design as cd  # noqa: E402
from loramake import character_set as cs  # noqa: E402

CONSISTENCY = HERE / "consistency_check.py"
REVIEW_BATCH = HERE / "review_batch.py"


def _load_card(path: Path) -> dict:
    card = cd.load_card(path)
    findings = cd.lint_card(card)
    errors = [f for f in findings if f.level == "error"]
    if errors:
        for f in errors:
            print(f"error {f.path}: {f.detail}", file=sys.stderr)
        raise SystemExit(1)
    return card


def _rendered(item: dict) -> list[str]:
    if item["kind"] in ("nai", "local"):
        render_dir = Path(item["render_dir"])
        return sorted(str(p) for p in render_dir.glob("*.png")) if render_dir.is_dir() else []
    return [item["file"]] if item.get("file") else []


def cmd_identity(args: argparse.Namespace) -> int:
    card = _load_card(Path(args.card))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.engine == "nai":
        index = cs.plan_identity(card, Path(args.out), seeds, args.artist_mix, args.outfit, args.expression)
        print(f"{index['design']}: identity round, {len(seeds)} free NAI busts"
              f"{' with ' + args.artist_mix if args.artist_mix else ''}")
    else:
        index = cs.plan_identity_local(card, Path(args.out), seeds, args.engine, args.lora or [],
                                       args.outfit, args.expression)
        print(f"{index['design']}: identity round, {len(seeds)} {args.engine} busts on "
              f"{cs.LOCAL_ENGINES[args.engine]['box']} with LoRA {', '.join(args.lora or []) or 'none'}")
    print(f"commands: {Path(args.out) / 'commands.sh'}")
    for c in index["commands"]:
        print("  " + c)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    card = _load_card(Path(args.card))
    anchor = cs.anchor_of(card, args.anchor)
    if args.engine == "nai":
        index = cs.plan_set(card, Path(args.out), args.outfit, anchor, args.seed, args.artist_mix,
                            strength=args.strength, fidelity=args.fidelity)
        main = [i for i in index["items"] if i["kind"] == "nai"]
        note = f"NAI main views (+{5 * len(main)} Anlas for the reference)"
    else:
        index = cs.plan_set_local(card, Path(args.out), args.outfit, anchor, args.seed, args.engine, args.lora or [])
        main = [i for i in index["items"] if i["kind"] == "local"]
        note = f"{args.engine} main views (IPAdapter on the anchor, LoRA {', '.join(args.lora or []) or 'none'})"
    faces = [i for i in index["items"] if i["kind"] == "boxa_face"]
    print(f"{index['design']}: {len(main)} {note}, {len(faces)} face differentials waiting for the bust")
    print(f"commands: {Path(args.out) / 'commands_main.sh'}")
    for c in index["commands"]:
        print("  " + c)
    return 0


def cmd_faces(args: argparse.Namespace) -> int:
    card = _load_card(Path(args.card))
    index = cs.plan_faces(card, Path(args.out), args.bust, args.lora or [], args.seed)
    print(f"{len(index['commands'])} face repaints bound to {args.bust}")
    print(f"commands: {Path(args.out) / 'commands_faces.sh'}")
    for c in index["commands"]:
        print("  " + c)
    return 0


def cmd_bind(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    index = cs.load_index(out_dir)
    if not Path(args.file).is_file():
        raise SystemExit(f"file not found: {args.file}")
    for item in index["items"]:
        if item["kind"] == "boxa_face" and item["expression"] == args.expression:
            item["file"] = str(args.file)
            cs.save_index(out_dir, index)
            print(f"bound {args.expression} -> {args.file}")
            return 0
    raise SystemExit(f"no face differential {args.expression!r} in the index")


def cmd_gate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    index = cs.load_index(out_dir)
    anchor = args.anchor or index["anchor"]
    files: list[str] = []
    for item in index["items"]:
        files.extend(_rendered(item))
    if not files:
        raise SystemExit("nothing rendered yet: run commands_main.sh / commands_faces.sh and bind first")
    argv = [sys.executable, str(CONSISTENCY), "--ref", anchor]
    if args.max_diff is not None:
        argv += ["--max-diff", str(args.max_diff)]
    proc = subprocess.run([*argv, *files], capture_output=True, text=True)
    rows = cs.parse_ccip(proc.stdout)
    summary = cs.gate_summary(rows, args.min_hold)
    report = {"anchor": anchor, "max_diff": args.max_diff, "rows": rows, "summary": summary,
              "stderr": proc.stderr.strip()}
    (out_dir / "gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        for name, row in rows.items():
            print(f"{row['ccip']:6.4f}  {'HOLD ' if row['hold'] else 'DRIFT'}  {name}")
        print(f"{summary['held']}/{summary['total']} held identity "
              f"(need {summary['min_hold_ratio']:.0%}): {'PASS' if summary['pass'] else 'FAIL'}")
    return 0 if summary["pass"] else 1


def cmd_register(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    index = cs.load_index(out_dir)
    spec = []
    for item in index["items"]:
        for path in _rendered(item):
            spec.append({"path": path, "caption": cs.caption_for(index, item, path)})
    if not spec:
        raise SystemExit("nothing rendered yet; register after the renders land")
    spec_path = out_dir / "review_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    argv = [sys.executable, str(REVIEW_BATCH), "--mode", "review", "--spec", str(spec_path), "--json"]
    proc = subprocess.run(argv, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return 1
    try:
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        index["review_batch"] = record.get("batch") or record.get("id")
        cs.save_index(out_dir, index)
    except (ValueError, IndexError):
        print("warning: review_batch.py printed no JSON record; review_batch not recorded in the index",
              file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="standing-art set planner for a design card")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("identity", help="identity round: N free NAI busts of the card, same prompt, many seeds")
    p.add_argument("card")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=("nai", "anima", "illustrious"), default="anima",
                   help="nai = free-tier NAI busts (+ --artist-mix); anima = BOXA fast line; illustrious = BOXB quality line")
    p.add_argument("--lora", action="append", metavar="FILE:W", help="artist LoRA for the local engines (repeatable)")
    p.add_argument("--artist-mix", help="NAI artist card id; leave out for a bare identity probe")
    p.add_argument("--seeds", default="7,8,9,10,11,12,13,14", help="comma list, at most 12")
    p.add_argument("--outfit", help="outfit id to include (default: identity + signature only)")
    p.add_argument("--expression", type=int, default=0, help="index into register.allowed_expressions")
    p.set_defaults(func=cmd_identity)

    p = sub.add_parser("plan", help="write NAI main-view jobs + index")
    p.add_argument("card")
    p.add_argument("--out", required=True)
    p.add_argument("--outfit", required=True, help="outfit id on the card")
    p.add_argument("--anchor", help="anchor image (default card.anchor.image)")
    p.add_argument("--engine", choices=("illustrious", "nai"), default="illustrious",
                   help="illustrious = local main views with IPAdapter + LoRA (default); nai = Precise Reference")
    p.add_argument("--lora", action="append", metavar="FILE:W", help="artist LoRA for the local engine (repeatable)")
    p.add_argument("--artist-mix", help="NAI artist card id (scripts/nai_artist_strings.py list)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strength", type=float, default=cs.DEFAULT_REFERENCE_STRENGTH)
    p.add_argument("--fidelity", type=float, default=cs.DEFAULT_REFERENCE_FIDELITY)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("faces", help="bind the neutral bust and write the BOXA face repaint commands")
    p.add_argument("card")
    p.add_argument("--out", required=True)
    p.add_argument("--bust", required=True, help="rendered neutral bust (NAI)")
    p.add_argument("--lora", action="append", metavar="FILE[:STR]", help="artist LoRA for region_fix")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_faces)

    p = sub.add_parser("bind", help="record a rendered face differential")
    p.add_argument("--out", required=True)
    p.add_argument("--expression", required=True)
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_bind)

    p = sub.add_parser("gate", help="CCIP identity gate against the anchor")
    p.add_argument("--out", required=True)
    p.add_argument("--anchor")
    p.add_argument("--min-hold", type=float, default=0.85)
    p.add_argument("--max-diff", type=float, default=cs.GATE_MAX_DIFF,
                   help=f"CCIP difference that still counts as the same face (default {cs.GATE_MAX_DIFF}; "
                        "the library's 0.21 lets a redrawn face through)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("register", help="open a review batch with every rendered file")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_register)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (cs.SetError, cd.DesignError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
