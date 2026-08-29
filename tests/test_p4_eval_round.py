"""p4_eval_round plan mode: offset/resume windowing (no GPU, no network)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "p4_eval_round.py"


def _plan(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=60)


def test_default_plan_covers_all_briefs_twice() -> None:
    proc = _plan()
    assert proc.returncode == 0
    assert "24 runs planned (12 briefs x 2 arms)" in proc.stdout


def test_offset_window_keeps_global_indices() -> None:
    proc = _plan("--offset", "9", "--limit", "3")
    assert proc.returncode == 0
    assert "[A09]" in proc.stdout and "[B11]" in proc.stdout
    assert "[A00]" not in proc.stdout
    assert "6 runs planned (3 briefs x 2 arms)" in proc.stdout


def test_offset_past_end_is_an_error() -> None:
    proc = _plan("--offset", "99")
    assert proc.returncode == 2
    assert "past the end" in proc.stderr


def test_into_requires_existing_stamp_dir() -> None:
    proc = _plan("--into", "no_such_stamp", "--run")
    assert proc.returncode == 2
    assert "no such stamp dir" in proc.stderr
