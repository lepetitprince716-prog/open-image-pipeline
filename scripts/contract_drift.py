#!/usr/bin/env python3
"""Offline endpoint-contract drift guard for the FastAPI app.

The guard intentionally covers only route shape: path, method, path params, and
query params. Header params, including POST /api/jobs Idempotency-Key, and
response bodies are out of scope because the current OpenAPI schema has no
reliable response models for generated body contracts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "docs" / "arch_optimization" / "contract_endpoints.json"
HTTP_METHODS = {"delete", "get", "head", "options", "patch", "post", "put", "trace"}


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _operation_params(path_item: dict[str, Any], operation: dict[str, Any], where: str) -> list[str]:
    params = [*path_item.get("parameters", []), *operation.get("parameters", [])]
    return sorted({p["name"] for p in params if p.get("in") == where and "name" in p})


def build_contract() -> dict[str, Any]:
    _ensure_repo_on_path()
    from app.server import app

    spec = app.openapi()
    endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            endpoints.append({
                "path": path,
                "method": method.upper(),
                "path_params": _operation_params(path_item, operation, "path"),
                "query_params": _operation_params(path_item, operation, "query"),
            })
    endpoints.sort(key=lambda e: (e["path"], e["method"]))
    return {
        "scope": "path+method+path_params+query_params",
        "endpoints": endpoints,
    }


def load_golden(path: Path = GOLDEN) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_golden(contract: dict[str, Any], path: Path = GOLDEN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _by_endpoint(contract: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(e["path"], e["method"]): e for e in contract.get("endpoints", [])}


def diff_contracts(expected: dict[str, Any], live: dict[str, Any]) -> tuple[bool, str]:
    expected_by = _by_endpoint(expected)
    live_by = _by_endpoint(live)
    expected_keys = set(expected_by)
    live_keys = set(live_by)
    lines: list[str] = []

    for path, method in sorted(live_keys - expected_keys):
        lines.append(f"added endpoint: {method} {path}")
    for path, method in sorted(expected_keys - live_keys):
        lines.append(f"removed endpoint: {method} {path}")
    for key in sorted(expected_keys & live_keys):
        expected_ep = expected_by[key]
        live_ep = live_by[key]
        changes = []
        for field in ("path_params", "query_params"):
            if expected_ep.get(field, []) != live_ep.get(field, []):
                changes.append(f"{field}: {expected_ep.get(field, [])} -> {live_ep.get(field, [])}")
        if changes:
            path, method = key
            lines.append(f"changed endpoint: {method} {path}: " + "; ".join(changes))

    if not lines:
        return True, "endpoint contract matches golden"
    return False, "endpoint contract drift detected:\n" + "\n".join(lines)


def check() -> tuple[bool, str]:
    if not GOLDEN.is_file():
        return False, f"missing golden: {GOLDEN}"
    return diff_contracts(load_golden(), build_contract())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or update the app endpoint contract golden.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check live OpenAPI endpoints against the golden (default)")
    mode.add_argument("--update", action="store_true", help="regenerate the golden from the live app")
    args = parser.parse_args(argv)

    if args.update:
        contract = build_contract()
        write_golden(contract)
        print(f"updated {GOLDEN} ({len(contract['endpoints'])} operations)")
        return 0

    ok, report = check()
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
