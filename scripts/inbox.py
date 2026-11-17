#!/usr/bin/env python3
"""Question inbox CLI for decisions that should not block a chat turn.

    python3 scripts/inbox.py ask --id anima-lora-r2 --title "Anima LoRA cloud round 2" \
        --detail "12 renders x 17 Buzz = 204 Buzz" --option approve --option hold
    python3 scripts/inbox.py list [--json] [--pending]
    python3 scripts/inbox.py answer --id anima-lora-r2 --value approve
    python3 scripts/inbox.py reopen --id anima-lora-r2
    python3 scripts/inbox.py drop --id anima-lora-r2

The store is ``outputs/_gallery/inbox.json`` (override with ``--store``); the gallery
``/inbox`` page reads and answers the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake import inbox  # noqa: E402

DEFAULT_STORE = REPO / "outputs" / "_gallery" / "inbox.json"


def _print_table(questions: list[dict]) -> None:
    if not questions:
        print("no questions")
        return
    width = max(len(question["id"]) for question in questions)
    for question in questions:
        status = "pending" if question["answer"] is None else "answered"
        answer = question["answer"] if question["answer"] is not None else "-"
        print(f"{question['id']:<{width}}  {status:<8}  {question['title']}  => {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask, list and answer inbox questions.")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="inbox JSON file")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="add an unanswered question")
    ask.add_argument("--id", required=True, help="stable id, [a-z0-9._-]")
    ask.add_argument("--title", required=True)
    ask.add_argument("--detail", default="")
    ask.add_argument("--option", action="append", required=True, dest="options",
                     help="answer button; repeat for each option")
    ask.add_argument("--replace", action="store_true", help="replace an existing question")

    lst = sub.add_parser("list", help="show questions")
    lst.add_argument("--json", action="store_true", help="print the store as JSON")
    lst.add_argument("--pending", action="store_true", help="unanswered questions only")

    answer = sub.add_parser("answer", help="record an answer")
    answer.add_argument("--id", required=True)
    answer.add_argument("--value", required=True)

    reopen = sub.add_parser("reopen", help="clear an answer")
    reopen.add_argument("--id", required=True)

    drop = sub.add_parser("drop", help="remove a question")
    drop.add_argument("--id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = inbox.load(args.store)
        if args.command == "list":
            questions = inbox.pending(data) if args.pending else data["questions"]
            if args.json:
                print(json.dumps({"v": data["v"], "questions": questions}, ensure_ascii=False, indent=2))
            else:
                _print_table(questions)
            return 0
        if args.command == "ask":
            question = inbox.ask(data, qid=args.id, title=args.title, detail=args.detail,
                                 options=args.options, replace=args.replace)
            inbox.save(args.store, data)
            print(f"asked {question['id']}")
            return 0
        if args.command == "answer":
            question = inbox.answer(data, qid=args.id, value=args.value)
            inbox.save(args.store, data)
            print(f"answered {question['id']}: {question['answer']}")
            return 0
        if args.command == "reopen":
            inbox.reopen(data, qid=args.id)
            inbox.save(args.store, data)
            print(f"reopened {args.id}")
            return 0
        if args.command == "drop":
            dropped = inbox.drop(data, qid=args.id)
            inbox.save(args.store, data)
            print(f"dropped {args.id}" if dropped else f"not found: {args.id}")
            return 0 if dropped else 1
    except inbox.InboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
