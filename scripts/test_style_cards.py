"""CLI tests for style_cards.py: lint, matrix, resolve and show run offline and agree with the registry."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "style_cards.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT)


def test_lint_is_clean():
    proc = _run("lint", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["problems"] == []
    assert payload["cards"] >= 30


def test_matrix_json_has_one_row_per_card():
    proc = _run("matrix", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload["rows"]) == payload["summary"]["cards"]
    assert {"illustrious", "anima", "nai", "krea2", "civitai"} <= set(payload["rows"][0]["lines"])


def test_resolve_json_carries_prompt_block():
    proc = _run("resolve", "muted_grey", "--line", "nai", "--json")
    assert proc.returncode == 0, proc.stderr
    frag = json.loads(proc.stdout)
    assert "artist:kieed" in frag["pos_prefix"]
    assert frag["tier"] == "verified"


def test_resolve_unknown_line_fails_closed():
    proc = _run("resolve", "house_dreamland", "--line", "nai")
    assert proc.returncode != 0
    assert "no realization" in (proc.stderr + proc.stdout) or "not a target" in (proc.stderr + proc.stdout)


def test_show_and_list_run():
    assert _run("show", "retro_90s").returncode == 0
    proc = _run("list", "--line", "civitai", "--json")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows and all(r["lines"]["civitai"] in ("verified", "candidate") for r in rows)
