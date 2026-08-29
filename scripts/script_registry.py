#!/usr/bin/env python3
"""Validate root script inventory, declarations and risk-count ratchet."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "scripts" / "script_registry.json"
SCHEMA = ROOT / "scripts" / "schema" / "script_registry_v1.schema.json"
INDEX = ROOT / "SCRIPTS_INDEX.md"
SUFFIXES = {".py", ".sh", ".zsh", ".ps1"}

RISK_PATTERNS = {
    "host_literal": re.compile(r"(?<![\d.])(?:192\.168\.|100\.\d{1,3}\.)"),
    "network_write": re.compile(
        r"(?:requests\.(?:post|put|patch|delete)|/(?:prompt|upload/image|free|interrupt)\b|"
        r"(?:urlopen|Request)\([\s\S]{0,240}?\b(?:data\s*=|method\s*=\s*['\"]POST)|"
        r"(?:curl|Invoke-RestMethod|Invoke-WebRequest)[\s\S]{0,240}?"
        r"(?:-X\s*(?:POST|PUT|DELETE)|-Method\s*(?:Post|Put|Delete)))",
        re.IGNORECASE,
    ),
    "gpu_submit": re.compile(r"(?:/prompt\b|submit_and_(?:fetch|wait)|queue_prompt)", re.IGNORECASE),
    "delete": re.compile(r"(?:\.unlink\s*\(|shutil\.rmtree\s*\(|os\.(?:remove|unlink)\s*\(|\brm\s+-)"),
    "direct_output_root": re.compile(r"(?:['\"]outputs/|\bOUTPUTS(?:_ROOT)?\b)"),
    "service_change": re.compile(
        r"(?:launchctl|systemctl|Start-Service|Stop-Service|taskkill|TerminateProcess|"
        r"app_server\.sh\s+(?:start|stop|restart))",
        re.IGNORECASE,
    ),
}


def discover(root: Path = ROOT) -> list[str]:
    scripts = root / "scripts"
    return sorted(
        str(path.relative_to(root)) for path in scripts.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES and "__pycache__" not in path.parts
    )


def scan_risks(path: Path) -> dict[str, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return {name: len(pattern.findall(text)) for name, pattern in RISK_PATTERNS.items()}


def validate(*, root: Path = ROOT, registry_path: Path | None = None,
             schema_path: Path | None = None, index_path: Path | None = None) -> dict:
    registry_path = registry_path or root / "scripts" / "script_registry.json"
    schema_path = schema_path or root / "scripts" / "schema" / "script_registry_v1.schema.json"
    index_path = index_path or root / "SCRIPTS_INDEX.md"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(registry)
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for script registry validation") from exc

    errors: list[str] = []
    entries = registry["entries"]
    paths = [entry["path"] for entry in entries]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        errors.append("duplicate registry paths: " + ", ".join(duplicates))
    discovered = discover(root)
    unregistered = sorted(set(discovered) - set(paths))
    missing = sorted(set(paths) - set(discovered))
    if unregistered:
        errors.append("unregistered executable entries: " + ", ".join(unregistered))
    if missing:
        errors.append("registry paths missing on disk: " + ", ".join(missing))

    index_text = index_path.read_text(encoding="utf-8")
    index_missing = [path for path in paths if f"`{path}`" not in index_text]
    if index_missing:
        errors.append("SCRIPTS_INDEX missing exact paths: " + ", ".join(index_missing))

    baseline = registry.get("risk_baseline", {})
    risk_totals = {name: 0 for name in RISK_PATTERNS}
    for entry in entries:
        path = entry["path"]
        if path not in discovered:
            continue
        observed = scan_risks(root / path)
        declared = set(entry["risk_declarations"])
        expected = baseline.get(path, {})
        for risk, count in observed.items():
            risk_totals[risk] += count
            if count and risk not in declared:
                errors.append(f"{path}: detected {risk}={count} but risk is undeclared")
            allowed = int(expected.get(risk, 0))
            if count > allowed:
                errors.append(f"{path}: {risk} increased {allowed}->{count}; update reviewed debt baseline")
        undeclared_effects = {
            "network_write": "network_write", "gpu_submit": "gpu_submit", "delete": "delete",
            "service_change": "service_change",
        }
        for risk, effect in undeclared_effects.items():
            if observed[risk] and effect not in set(entry["effects"]):
                errors.append(f"{path}: risk {risk} requires effect {effect}")
        for test_path in entry["tests"]:
            if not (root / test_path).exists():
                errors.append(f"{path}: declared test path missing: {test_path}")
        if not (root / entry["authority"]).exists():
            errors.append(f"{path}: authority missing: {entry['authority']}")

    result = {"ok": not errors, "entries": len(entries), "discovered": len(discovered),
              "status_counts": {status: sum(entry["status"] == status for entry in entries)
                                for status in ("active", "probe", "legacy", "test", "reference")},
              "risk_totals": risk_totals, "errors": errors}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",), nargs="?", default="validate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate()
    if args.json or not result["ok"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"script registry OK: {result['entries']}/{result['discovered']} entries")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
