#!/usr/bin/env python3
"""Read back what a human decided about one review batch.

The batch page writes ``result.json`` as compact index lists (``y``/``l``/``n``/``c``) that
mean nothing without the manifest beside them. This joins the two and prints the verdicts
with the names and captions attached, so the caller never has to open either file.

Usage:
    python3 scripts/review_result.py BATCH [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config import OUTPUTS

from review_batch import BATCH_RE, REVIEW_ROOT_PARTS


class ReviewResultError(ValueError):
    """A user-facing readback error."""


def _batch_dir(batch: str) -> Path:
    if not BATCH_RE.fullmatch(batch):
        raise ReviewResultError(f"invalid batch ID: {batch}")
    root = (OUTPUTS / REVIEW_ROOT_PARTS[0] / REVIEW_ROOT_PARTS[1]).resolve()
    directory = (root / batch).resolve()
    try:
        directory.relative_to(root)
    except ValueError:
        raise ReviewResultError(f"invalid batch ID: {batch}") from None
    if not directory.is_dir():
        raise ReviewResultError(f"unknown batch: {batch}")
    return directory


def _read_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ReviewResultError(f"missing {label}: {path}") from None
    except json.JSONDecodeError as exc:
        raise ReviewResultError(f"{label} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ReviewResultError(f"{label} is not an object")
    return data


def read_result(batch: str) -> dict[str, object]:
    """Join one batch's manifest and result into a single decided/pending report."""
    directory = _batch_dir(batch)
    manifest = _read_json(directory / "manifest.json", "manifest")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list):
        raise ReviewResultError("manifest has no items")

    items: dict[int, dict[str, object]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict) or not isinstance(raw.get("i"), int):
            raise ReviewResultError("manifest has a malformed item")
        items[raw["i"]] = {
            "i": raw["i"],
            "name": raw.get("name", ""),
            "media": raw.get("media", "image"),
            "caption": raw.get("caption", ""),
        }

    mode = manifest.get("mode", "review")
    if mode == "view":
        # A view batch never collects decisions, so "pending" would be misleading.
        return {
            "batch": batch, "mode": "view", "total": len(items), "reviewed": 0,
            "pending": [], "love": [], "yes": [], "no": [], "comments": {},
            "reason_counts": {},
        }

    result_path = directory / "result.json"
    result = _read_json(result_path, "result") if result_path.exists() else {}

    def decided(key: str) -> list[dict[str, object]]:
        values = result.get(key)
        indexes = [v for v in values if isinstance(v, int) and v in items] \
            if isinstance(values, list) else []
        return [items[index] for index in sorted(indexes)]

    raw_comments = result.get("c") if isinstance(result.get("c"), dict) else {}
    comments = {
        int(key): {**items[int(key)], "comment": text}
        for key, text in raw_comments.items()
        if str(key).lstrip("-").isdigit() and int(key) in items and isinstance(text, str)
    }
    raw_reasons = result.get("r") if isinstance(result.get("r"), dict) else {}
    reasons_map: dict[int, list[str]] = {}
    for key, value in raw_reasons.items():
        if str(key).lstrip("-").isdigit() and int(key) in items and isinstance(value, list):
            reasons_map[int(key)] = [r for r in value if isinstance(r, str)]
    # "l" (love) is the top tier and stays out of "yes": a batch decided before the tier
    # existed simply has no such list, and reads back as an empty one.
    love, yes, no = decided("l"), decided("y"), decided("n")
    reason_counts: dict[str, int] = {}
    for entry in no:
        r_list = reasons_map.get(entry["i"], [])
        entry["reasons"] = r_list
        for r in r_list:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    settled = {item["i"] for item in love} | {item["i"] for item in yes} \
        | {item["i"] for item in no} | set(comments)

    return {
        "batch": batch,
        "mode": "review",
        "total": len(items),
        "reviewed": len(settled),
        "pending": [items[index] for index in sorted(set(items) - settled)],
        "love": love,
        "yes": yes,
        "no": no,
        "comments": {str(index): comments[index] for index in sorted(comments)},
        "reason_counts": reason_counts,
    }


def _format(report: dict[str, object]) -> str:
    lines = [
        f"batch {report['batch']} ({report['mode']}) — "
        f"{report['reviewed']}/{report['total']} decided"
    ]
    if report["mode"] == "view":
        lines.append("view-only batch: no decisions are collected")
        return "\n".join(lines)
    for label, key in (("love", "love"), ("yes", "yes"), ("no", "no")):
        entries = report[key]
        if entries:
            lines.append(f"\n{label} ({len(entries)}):")
            for e in entries:
                reasons = e.get("reasons") or []
                suffix = f" — {', '.join(reasons)}" if reasons else ""
                lines.append(f"  [{e['i']}] {e['name']}{suffix}")
    reason_counts = report.get("reason_counts") or {}
    if reason_counts:
        lines.append("\nreasons: " + ", ".join(f"{k} {v}" for k, v in reason_counts.items()))
    comments = report["comments"]
    if comments:
        lines.append(f"\ncomment ({len(comments)}):")
        lines.extend(f"  [{e['i']}] {e['name']} — {e['comment']}" for e in comments.values())
    pending = report["pending"]
    if pending:
        lines.append(f"\npending ({len(pending)}):")
        lines.extend(f"  [{e['i']}] {e['name']}" for e in pending)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read back one review batch's decisions")
    parser.add_argument("batch", help="batch ID printed by review_batch.py")
    parser.add_argument("--json", action="store_true", help="print the report as one JSON object")
    args = parser.parse_args(argv)

    try:
        report = read_result(args.batch)
    except ReviewResultError as exc:
        parser.exit(2, f"review_result.py: error: {exc}\n")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    else:
        print(_format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
