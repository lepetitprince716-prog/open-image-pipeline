#!/usr/bin/env python3
"""Append one immutable human verdict record for a direction shot and timecode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ROOT  # inserts the repository root on sys.path
from loramake.video_direction import (
    REVIEW_DIMENSIONS,
    REVIEW_VERDICTS,
    VideoDirectionError,
    atomic_publish_no_clobber,
    build_review_record,
    json_bytes,
    load_json_object,
    validate_direction,
)


def load_direction(path: str | Path) -> tuple[Path, dict]:
    source, data = load_json_object(path, label="direction JSON")
    validate_direction(data)
    return source, data


def _dimension_notes(values: list[str]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for value in values:
        dimension, separator, note = value.partition("=")
        if not separator or dimension not in REVIEW_DIMENSIONS:
            raise VideoDirectionError("--dimension-note must be DIMENSION=TEXT for a known review dimension")
        if dimension in notes:
            raise VideoDirectionError(f"duplicate --dimension-note for {dimension}")
        notes[dimension] = note
    return notes


def record_review(args: argparse.Namespace) -> Path:
    direction_path, direction = load_direction(args.direction)
    verdicts = {dimension: getattr(args, dimension) for dimension in REVIEW_DIMENSIONS}
    record = build_review_record(
        direction,
        direction_path=direction_path,
        shot_id=args.shot_id,
        timecode=args.timecode,
        clip_path=args.clip,
        verdicts=verdicts,
        notes=_dimension_notes(args.dimension_note),
        reviewer=args.reviewer,
        summary=args.summary,
        record_id=args.record_id,
    )
    if args.out:
        target = Path(args.out).expanduser().resolve()
    else:
        project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else direction_path.parent
        target = (
            project_dir
            / "reviews"
            / record["shot_id"]
            / f"{record['timecode']['frame']:06d}-{record['record_id']}.json"
        )
    atomic_publish_no_clobber(target, json_bytes(record))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="write one no-clobber review event")
    record.add_argument("--direction", required=True, help="normalized direction.json")
    record.add_argument("--shot-id", required=True)
    record.add_argument("--timecode", required=True, help="global HH:MM:SS.mmm timecode")
    record.add_argument("--clip", required=True, help="exact local clip being reviewed; path/hash are frozen")
    record.add_argument("--record-id", help="stable event id; generated when omitted")
    record.add_argument("--reviewer", default="human")
    record.add_argument("--summary", default="")
    record.add_argument(
        "--dimension-note",
        action="append",
        default=[],
        metavar="DIMENSION=TEXT",
        help="optional note for one dimension; repeat as needed",
    )
    destination = record.add_mutually_exclusive_group()
    destination.add_argument("--project-dir", help="project root; records append below reviews/<shot>/")
    destination.add_argument("--out", help="exact no-clobber review JSON path")
    for dimension in REVIEW_DIMENSIONS:
        record.add_argument(
            f"--{dimension.replace('_', '-')}",
            dest=dimension,
            required=True,
            choices=REVIEW_VERDICTS,
        )
    args = parser.parse_args(argv)
    try:
        target = record_review(args)
    except VideoDirectionError as exc:
        parser.error(str(exc))
    print(json.dumps({"review": str(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
