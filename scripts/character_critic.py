#!/usr/bin/env python3
"""Critic seat for the character atelier: violations against the design card, never a pass.

  character_critic.py review CARD IMAGE... [--kind auto|sheet|book|view|expression] [--expression NAME]
                       [--seat claude|grok] [--min-violations 1] [--no-mechanical] [--dry-run] [--force] [--json]
  character_critic.py stamp CARD            # bind every image the atelier state tracks to the card revision
  character_critic.py stale CARD            # table: artifact · stamped revision · current · stale

CARD is a card path or id under configs/character_designs/. Reports land in
outputs/characters/<id>/qc/<parent>__<stem>.critic.json and the summary in the image sidecar
under gates.critic. Exit codes: review 0 all reviewed · 1 any unverified · 2 usage or the
atelier worker is busy (pass --force); stale 0 none stale · 1 any stale. The seat runs in
its own process group with a hard timeout (loramake.seat_proc). Env: LORAMAKE_CRITIC_SEAT,
LORAMAKE_CRITIC_BIN, LORAMAKE_CRITIC_MODEL (must differ from LORAMAKE_BRAIN_MODEL),
LORAMAKE_CRITIC_RESTRICTED, GROK_BIN. Runbook: docs/runbooks/character_design_runbook.md §11.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake import character_atelier as atelier  # noqa: E402
from loramake import character_critic as critic  # noqa: E402
from loramake import critic_checklist as checklist  # noqa: E402

import character_design as cd  # noqa: E402  (scripts/ is on sys.path via the atelier import)


def card_id_of(arg: str) -> str:
    path = Path(arg)
    if path.is_file():
        card = json.loads(path.read_text(encoding="utf-8"))
        return str(card["id"])
    if cd.ID_RE.match(arg) and (atelier.CARDS / f"{arg}.json").is_file():
        return arg
    raise SystemExit(f"no design card at {arg!r} (path or id under {atelier.CARDS})")


def tracked_rels(state: dict) -> list[str]:
    sheet = state.get("sheet") or {}
    rels = list(sheet.get("versions") or [])
    rels += [sheet.get(k) for k in ("book", "front", "side")]
    rels += list((sheet.get("expressions") or {}).values())
    for items in (state.get("candidates") or {}).values():
        rels += [c.get("path") for c in items or [] if isinstance(c, dict)]
    return list(dict.fromkeys(r for r in rels if isinstance(r, str) and r))


def cmd_review(args: argparse.Namespace) -> int:
    card_id = card_id_of(args.card)
    state = atelier.load_state(card_id)
    if state.get("job") and not args.force and not args.dry_run:
        print(f"the atelier worker is busy on {card_id} ({state['job'].get('kind')}); pass --force to review anyway",
              file=sys.stderr)
        return 2
    kind = None if args.kind == "auto" else args.kind
    worst = 0
    for image in args.images:
        try:
            report = critic.review(card_id, image, kind=kind, expression=args.expression, backend=args.seat,
                                   min_violations=args.min_violations, mechanical=not args.no_mechanical,
                                   dry_run=args.dry_run)
        except critic.CriticError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(report["packet"] if not args.json else json.dumps(report, ensure_ascii=False, indent=2))
            continue
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"{report['status']:<10} violations={report['violations']:<2} uncertain={report['uncertain']:<2} "
                  f"{report['image']}" + (f"  [{report['error']}]" if report.get("error") else ""))
        if report["status"] != "reviewed":
            worst = 1
    return worst


def cmd_stamp(args: argparse.Namespace) -> int:
    card_id = card_id_of(args.card)
    card = atelier.load_card(card_id)
    state = atelier.load_state(card_id)
    sha8 = cd.card_sha8(card)
    known = state.get("artifacts") or {}
    done = 0
    for rel in tracked_rels(state):
        if not (atelier.ROOT / rel).is_file():
            continue
        if rel in known and not args.force:
            continue
        atelier.stamp(card_id, rel, sha8)
        done += 1
    print(f"stamped {done} artifact(s) of {card_id} with card {sha8}")
    return 0


def cmd_stale(args: argparse.Namespace) -> int:
    card_id = card_id_of(args.card)
    view = atelier.annotate(atelier.load_state(card_id), atelier.load_card(card_id))
    rows = sorted((view.get("artifacts") or {}).items())
    if args.json:
        print(json.dumps({"card_sha8": view["card_sha8"], "artifacts": dict(rows)}, ensure_ascii=False, indent=2))
    else:
        print(f"card {card_id} · current revision {view['card_sha8']}")
        for rel, entry in rows:
            critic_summary = entry.get("critic") or {}
            print(f"{'STALE' if entry.get('stale') else 'ok   '}  {entry.get('card_sha8')}  "
                  f"{critic_summary.get('status') or '-':<10} {rel}")
    return 1 if any(e.get("stale") for _, e in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review", help="review images against the card through the critic seat")
    rv.add_argument("card")
    rv.add_argument("images", nargs="+")
    rv.add_argument("--kind", choices=("auto", *checklist.KINDS), default="auto")
    rv.add_argument("--expression", help="expression name for --kind expression")
    rv.add_argument("--seat", choices=("claude", "grok"))
    rv.add_argument("--min-violations", type=int, default=critic.DEFAULT_MIN_VIOLATIONS)
    rv.add_argument("--no-mechanical", action="store_true", help="skip the gate chain and CCIP facts")
    rv.add_argument("--dry-run", action="store_true", help="print the packet; call no seat, write nothing")
    rv.add_argument("--force", action="store_true", help="review even while the atelier worker is busy")
    rv.add_argument("--json", action="store_true")
    rv.set_defaults(fn=cmd_review)
    st = sub.add_parser("stamp", help="bind the images the atelier tracks to the current card revision")
    st.add_argument("card")
    st.add_argument("--force", action="store_true", help="re-stamp artifacts that already carry a revision")
    st.set_defaults(fn=cmd_stamp)
    sl = sub.add_parser("stale", help="which artifacts were rendered from an older card revision")
    sl.add_argument("card")
    sl.add_argument("--json", action="store_true")
    sl.set_defaults(fn=cmd_stale)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
