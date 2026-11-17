#!/usr/bin/env python3
"""Round ledger CLI: every review batch with its verdicts, Buzz spend, spec, gate summary and
write-back citations, derived from the files on disk (see loramake/rounds.py).

    python3 scripts/rounds.py list [--pending] [--json]
    python3 scripts/rounds.py show BATCH

The gallery serves the same data at /rounds and /api/rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake import rounds as rounds_lib  # noqa: E402

REVIEW_ROOT = REPO / "outputs" / "_gallery" / "review_batches"
LEDGER = REPO / "outputs" / "civitai" / "buzz_ledger.jsonl"
CARDS_DIR = REPO / "catalogs" / "style"
LIBRARY = REPO / "scripts" / "nai_artist_strings.json"


def collect() -> list[dict]:
    return rounds_lib.collect(review_root=REVIEW_ROOT, repo=REPO, ledger_path=LEDGER, cards_dir=CARDS_DIR,
                              library_path=LIBRARY)


def _fmt(row: dict) -> str:
    verdicts = row["verdicts"]
    state = "pending" if row["pending"] else ("view" if row["mode"] == "view" else "done")
    spend = f"{row['spend']['buzz']} Buzz" if row.get("spend") else "-"
    gates = f"{row['gates'].get('pass', '?')}/{row['gates'].get('fail', '?')}" if row.get("gates") else "-"
    writeback = f"reg {row['writeback']['registry']} lib {row['writeback']['library']}"
    return (f"{row['created'][:16]:16s} {row['batch']:36s} {state:7s} {row['decided']:>3d}/{row['total']:<3d} "
            f"love {verdicts['love']:<2d} yes {verdicts['yes']:<3d} no {verdicts['no']:<3d} {spend:>9s} gates {gates:7s} "
            f"{writeback}  {row.get('spec') or ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round ledger derived from review batches")
    sub = parser.add_subparsers(dest="command", required=True)
    lst = sub.add_parser("list")
    lst.add_argument("--pending", action="store_true")
    lst.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("batch")
    args = parser.parse_args(argv)
    rows = collect()
    if args.command == "show":
        match = [row for row in rows if row["batch"] == args.batch]
        if not match:
            print(f"error: no round {args.batch}", file=sys.stderr)
            return 2
        print(json.dumps(match[0], ensure_ascii=False, indent=2))
        return 0
    if args.pending:
        rows = [row for row in rows if row["pending"]]
    if args.json:
        print(json.dumps({"v": 1, "rounds": rows}, ensure_ascii=False, indent=1))
    else:
        for row in rows:
            print(_fmt(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
