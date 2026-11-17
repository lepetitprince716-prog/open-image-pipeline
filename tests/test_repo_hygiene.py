"""Repository-level guards for generated and independently owned files."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        cwd=ROOT,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def test_no_tracked_repository_junk() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        parsed = PurePosixPath(path)
        if "__pycache__" in parsed.parts or parsed.suffix in {".pyc", ".pyo"}:
            offenders.append(path)
        elif path.startswith("docs/handoffs/") and parsed.suffix == ".zip":
            offenders.append(path)
        elif path.startswith("mind/"):
            offenders.append(path)

    assert offenders == [], f"tracked repository junk: {offenders}"
