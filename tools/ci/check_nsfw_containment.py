#!/usr/bin/env python3
"""Adult content must stay inside nsfw/. Enforced with only the stdlib.

The repository already states this rule in prose, and it has been broken
anyway -- most instructively by an image that reached the tree inside a tracked
office document, where a path-based ignore rule could never see it. Prose binds
only the readers who happen to read it; this check binds everything.

Containers are opened here, unlike in the public gate, because the private tree
legitimately tracks office documents and archives, and the leak that motivated
this check was inside one.

Existing violations are recorded in a baseline rather than fixed in the same
change, so the check can be adopted without a day-one red build. The baseline
is bidirectional, following tools/ci/check_pytest_report.py: an entry that
stops matching must be removed, so the debt list can only shrink.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "nsfw_containment_baseline.json"
SCAN_PATH = REPO_ROOT / "mirror" / "scan.py"

# Paths the rule does not apply to: nsfw/ is where the content belongs, mirror/
# holds the rule list itself (which necessarily names what it hunts), and the
# ignore and agent-instruction files name nsfw paths as their whole job.
EXEMPT_DIRS = (
    "nsfw",
)
EXEMPT_FILES = (
    "mirror/publish_manifest.json",
    "mirror/scan_rules.json",
    "tools/ci/nsfw_containment_baseline.json",
    "tools/ci/check_nsfw_containment.py",
    # Carries sample vocabulary in order to prove the check fires; exempting it
    # is the same self-reference the rules file needs.
    "tests/test_nsfw_containment_policy.py",
    "tests/test_mirror_scan_policy.py",
    ".gitignore",
    ".gitattributes",
    "AGENTS.md",
    "GEMINI.md",
    "CLAUDE.md",
)


def _load_scan():
    spec = importlib.util.spec_from_file_location("mirror_scan", SCAN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True, text=True, check=True,
    )
    return [item for item in result.stdout.split("\0") if item]


def load_baseline(path: Path) -> dict[str, dict[str, int | str]]:
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    violations = document.get("known_violations", {})
    if not isinstance(violations, dict):
        raise ValueError(f"{path}: known_violations must be an object")
    return violations


def _exempt(path: str) -> bool:
    return path in EXEMPT_FILES or any(
        path == directory or path.startswith(directory + "/")
        for directory in EXEMPT_DIRS
    )


def _fingerprint(path: str, problem: str) -> tuple[str, str]:
    prefix, _, detail = problem.partition(" -- ")
    marker = "[fingerprint="
    rule = prefix.rsplit(": ", 1)[-1]
    if marker in detail:
        context = detail.rsplit(marker, 1)[1].rstrip("]")
    else:
        context = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    return f"{path}: {rule}", context


def scan_repository(repo_root: Path) -> dict[str, dict[str, int | str]]:
    """Return stable violation groups for adult content outside nsfw/."""
    scan = _load_scan()
    ruleset = scan.compile_ruleset(scan.load_rules(), "nsfw_containment")
    groups: dict[str, list[str]] = {}
    for path in tracked_files(repo_root):
        if _exempt(path):
            continue
        target = repo_root / path
        if not target.is_file():
            continue
        problems = scan.scan_file(target, ruleset, path, include_fingerprint=True)
        for problem in problems:
            group, context = _fingerprint(path, problem)
            groups.setdefault(group, []).append(context)
    return {
        group: {
            "count": len(contexts),
            "digest": hashlib.sha256("\0".join(sorted(contexts)).encode("ascii")).hexdigest(),
        }
        for group, contexts in sorted(groups.items())
    }


def check(repo_root: Path, baseline_path: Path) -> list[str]:
    """Collect new, changed, and removed violation groups."""
    baseline = load_baseline(baseline_path)
    offenders = scan_repository(repo_root)
    problems: list[str] = []
    for group in sorted(set(baseline) | set(offenders)):
        expected = baseline.get(group)
        actual = offenders.get(group)
        if expected is None:
            problems.append(f"{group}: adult content outside nsfw/ (new)")
        elif actual is None:
            problems.append(f"{group}: known violation group is clean now; update the baseline")
        elif expected != actual:
            problems.append(
                f"{group}: violation fingerprint or count changed; review and update the baseline"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--write-baseline", action="store_true",
                        help="record the current violations as the accepted debt")
    args = parser.parse_args()

    if args.write_baseline:
        offenders = scan_repository(args.repo_root)
        args.baseline.write_text(
            json.dumps(
                {
                    "version": 2,
                    "note": "Violation groups outside nsfw/ that predate this check. "
                            "Each group stores a count and digest of normalized-context hashes "
                            "without matched content. Adding or changing a group needs a reason.",
                    "known_violations": offenders,
                },
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        total = sum(int(item["count"]) for item in offenders.values())
        print(f"Wrote {total} known violation(s) in {len(offenders)} group(s) to {args.baseline}")
        return 0

    problems = check(args.repo_root, args.baseline)
    if problems:
        for problem in problems:
            print(f"NSFW_CONTAINMENT_ERROR: {problem}")
        return 1

    baseline = load_baseline(args.baseline)
    total = sum(int(item["count"]) for item in baseline.values())
    print(f"NSFW containment OK ({total} known violation(s) in {len(baseline)} group(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
