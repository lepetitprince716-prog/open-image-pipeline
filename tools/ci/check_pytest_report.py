#!/usr/bin/env python3
"""Fail when pytest silently adds, removes, or changes a conditional skip."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_ALLOWLIST = Path(__file__).with_name("pytest_skip_allowlist.json")


def read_skips(report: Path) -> list[tuple[str, str]]:
    root = ET.parse(report).getroot()
    skips: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        node = f"{case.get('classname', '')}::{case.get('name', '')}"
        skips.append((node, skipped.get("message", "")))
    return sorted(skips)


def check_report(
    report: Path,
    *,
    allowlist: Path | None = DEFAULT_ALLOWLIST,
    expected_skips: int | None = None,
) -> list[str]:
    actual = read_skips(report)
    if expected_skips is not None:
        if len(actual) == expected_skips:
            return []
        return [f"expected {expected_skips} skipped tests, observed {len(actual)}: {actual}"]

    if allowlist is None:
        return ["an allowlist or --expected-skips is required"]
    data = json.loads(allowlist.read_text(encoding="utf-8"))
    expected = data.get("skips")
    if data.get("version") != 1 or not isinstance(expected, list):
        return [f"invalid skip allowlist: {allowlist}"]

    problems: list[str] = []
    actual_by_node = {node: message for node, message in actual}
    expected_nodes: set[str] = set()
    for item in expected:
        if not isinstance(item, dict) or not isinstance(item.get("node"), str):
            problems.append(f"invalid allowlist entry: {item!r}")
            continue
        node = item["node"]
        if node in expected_nodes:
            problems.append(f"duplicate allowlist node: {node}")
        expected_nodes.add(node)
    actual_nodes = set(actual_by_node)

    for node in sorted(actual_nodes - expected_nodes):
        problems.append(f"unexpected skip: {node}: {actual_by_node[node]}")
    for node in sorted(expected_nodes - actual_nodes):
        problems.append(f"expected skip disappeared; review and remove it intentionally: {node}")
    for item in expected:
        if not isinstance(item, dict) or not isinstance(item.get("node"), str):
            continue
        node = item.get("node")
        needle = item.get("message_contains")
        if node in actual_by_node and (not isinstance(needle, str) or needle not in actual_by_node[node]):
            problems.append(
                f"skip reason changed for {node}: expected substring {needle!r}, observed {actual_by_node[node]!r}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--expected-skips", type=int)
    args = parser.parse_args()
    problems = check_report(
        args.report,
        allowlist=None if args.expected_skips is not None else args.allowlist,
        expected_skips=args.expected_skips,
    )
    if problems:
        for problem in problems:
            print(f"PYTEST_SKIP_POLICY_ERROR: {problem}")
        return 1
    print(f"pytest skip policy OK: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
