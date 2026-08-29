#!/usr/bin/env python3
"""Registry and launcher for isolated external image-generation lines.

The central registry owns discovery only. Each consumer project owns its
configuration, prompts, references, outputs, run manifests, and execution CLI.
This keeps lines uniformly visible without merging their assets into loramake.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / "projects" / "image_lines.json"


class RegistryError(ValueError):
    """Raised when the image-line registry is malformed."""


def registry_path() -> Path:
    return Path(os.environ.get("LORAMAKE_IMAGE_LINES_REGISTRY", DEFAULT_REGISTRY)).expanduser()


def load_registry(path: Path | None = None) -> dict[str, Any]:
    target = path or registry_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry {target}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != "loramake-image-lines-v1":
        raise RegistryError("registry schema_version must be loramake-image-lines-v1")
    lines = data.get("lines")
    if not isinstance(lines, list):
        raise RegistryError("registry lines must be a list")
    seen: set[str] = set()
    for item in lines:
        if not isinstance(item, dict):
            raise RegistryError("every line entry must be an object")
        line_id = item.get("id")
        if not isinstance(line_id, str) or not line_id:
            raise RegistryError("every line needs a non-empty id")
        if line_id in seen:
            raise RegistryError(f"duplicate line id: {line_id}")
        seen.add(line_id)
        for key in ("name", "root", "runner", "config", "output_root"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise RegistryError(f"line {line_id} needs string field {key}")
    return data


def get_line(data: dict[str, Any], line_id: str) -> dict[str, Any]:
    for item in data["lines"]:
        if item["id"] == line_id:
            return item
    raise RegistryError(f"unknown line id: {line_id}")


def line_paths(item: dict[str, Any]) -> dict[str, Path]:
    root = Path(item["root"]).expanduser()
    if not root.is_absolute():
        root = REPO / root

    def under_root(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    return {
        "root": root,
        "runner": under_root(item["runner"]),
        "config": under_root(item["config"]),
        "output_root": under_root(item["output_root"]),
    }


def line_status(item: dict[str, Any]) -> dict[str, Any]:
    paths = line_paths(item)
    missing = [key for key in ("root", "runner", "config") if not paths[key].exists()]
    return {
        "id": item["id"],
        "name": item["name"],
        "enabled": bool(item.get("enabled", True)),
        "owner_scope": item.get("owner_scope", "unspecified"),
        "isolation": item.get("isolation", "unspecified"),
        "paths": {key: str(value) for key, value in paths.items()},
        "ready": not missing,
        "missing": missing,
    }


def invoke(item: dict[str, Any], args: list[str]) -> int:
    status = line_status(item)
    if not status["enabled"]:
        print(f"line is disabled: {item['id']}", file=sys.stderr)
        return 2
    if not status["ready"]:
        print(f"line is incomplete: {item['id']} missing={','.join(status['missing'])}", file=sys.stderr)
        return 2
    paths = line_paths(item)
    proc = subprocess.run([sys.executable, str(paths["runner"]), *args], cwd=paths["root"])
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage isolated project image-generation lines")
    sub = parser.add_subparsers(dest="command", required=True)
    list_p = sub.add_parser("list", help="list registered lines")
    list_p.add_argument("--json", action="store_true")
    status_p = sub.add_parser("status", help="show registry status and optional project runtime status")
    status_p.add_argument("line_id")
    status_p.add_argument("--deep", action="store_true", help="also invoke the line's own status command")
    status_p.add_argument("--json", action="store_true")
    validate_p = sub.add_parser("validate", help="run the line's own validator")
    validate_p.add_argument("line_id")
    exec_p = sub.add_parser("exec", help="run the line-owned CLI with arguments after --")
    exec_p.add_argument("line_id")
    exec_p.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = load_registry()
        if args.command == "list":
            rows = [line_status(item) for item in data["lines"]]
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                for row in rows:
                    state = "ready" if row["ready"] and row["enabled"] else "hold"
                    print(f"{row['id']:<28} {state:<6} {row['isolation']:<24} {row['name']}")
            return 0
        item = get_line(data, args.line_id)
        if args.command == "status":
            status = line_status(item)
            if args.json:
                print(json.dumps(status, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(status, ensure_ascii=False, indent=2))
            if args.deep and status["ready"]:
                return invoke(item, ["status"])
            return 0 if status["ready"] else 2
        if args.command == "validate":
            return invoke(item, ["validate"])
        if args.command == "exec":
            tail = args.args[1:] if args.args[:1] == ["--"] else args.args
            return invoke(item, tail or ["list"])
    except RegistryError as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
