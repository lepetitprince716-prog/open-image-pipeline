"""Compile a design card into the checklist the critic seat answers, one line per element.

Positive items come from ``character_design.element_tokens`` (the same granularity the
render ledger keys on); negative items are the card's guards. Nothing here looks at an
image; it only says what the image is supposed to show and what it must not.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

from . import character_recipes as recipes
from .config import OUTPUTS

ROOT = OUTPUTS.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import character_design as cd  # noqa: E402

# artifact kind -> the frame its prompt was compiled for
KINDS = {"sheet": "full", "book": "bust", "view": "full", "expression": "bust"}
# what a bust can show: the identity subset; silhouette, outfit and signature are out of frame
BUST_ELEMENTS = ("hair", "eyes", "face", "skin_marks", "expression")
SHEET_RE = re.compile(r"^sheet_v\d+\.png$")
VIEW_RE = re.compile(r"^(front|side)_\d+\.png$")
EXPR_RE = re.compile(r"^expr_([a-z0-9_]+)\.png$")


class Check(NamedTuple):
    element: str
    must: bool
    text: str


class ChecklistError(ValueError):
    pass


def kind_of(rel: str | Path) -> tuple[str, str | None]:
    """(kind, expression name) from an atelier path; raises for anything the atelier did not make."""
    path = Path(rel)
    parent, name = path.parent.name, path.name
    if parent == "design" and SHEET_RE.match(name):
        return "sheet", None
    if parent == "book" and name.endswith(".png"):
        return "book", None
    if parent == "set":
        match = EXPR_RE.match(name)
        if match:
            return "expression", match.group(1)
        if VIEW_RE.match(name):
            return "view", None
    raise ChecklistError(f"not an atelier artifact: {rel}")


def _expression_tokens(card: dict, kind: str, expression: str | None, default: list[str]) -> list[str]:
    if kind == "expression":
        for name, _label, tags in recipes.EXPRESSIONS:
            if name == expression:
                return [tags]
        raise ChecklistError(f"unknown expression {expression!r}; atelier expressions are "
                             + ", ".join(n for n, _, _ in recipes.EXPRESSIONS))
    if kind == "sheet":
        allowed = [a for a in (card.get("register") or {}).get("allowed_expressions") or [] if isinstance(a, str)]
        return ["six small restrained expression portraits: " + "; ".join(allowed[:4])]
    return default


def _split(value: object) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def checklist_from_card(card: dict, kind: str, expression: str | None = None) -> list[Check]:
    """Every assertion the critic must answer for one artifact kind, in checklist order."""
    if kind not in KINDS:
        raise ChecklistError(f"kind must be one of {tuple(KINDS)}, got {kind!r}")
    frame = KINDS[kind]
    try:
        outfit = cd.outfit_for(card)
        elements = cd.element_tokens(card, frame, f"outfit:{outfit['id']}")
    except (cd.DesignError, KeyError, TypeError) as exc:
        raise ChecklistError(f"card cannot be compiled: {exc}") from exc
    checks: list[Check] = []
    for key, tokens in elements:
        if key == "expression":
            tokens = _expression_tokens(card, kind, expression, list(tokens))
        if frame == "bust" and key not in BUST_ELEMENTS:
            continue
        checks.append(Check(key, True, ", ".join(tokens)))
    marks = [m for m in card.get("skin_marks") or [] if isinstance(m, str) and m.strip()]
    if kind == "sheet" and marks and not any(c.element == "skin_marks" for c in checks):
        index = next((i + 1 for i, c in enumerate(checks) if c.element == "face"), len(checks))
        checks.insert(index, Check("skin_marks", True, ", ".join(marks)))
    palette = card.get("palette") or {}
    base = [b for b in palette.get("base") or [] if isinstance(b, str)]
    if base and palette.get("accent"):
        checks.append(Check("palette", True, f"base colours {', '.join(base)}; the only accent colour is {palette['accent']}"))
    if kind == "sheet":
        for i, acc in enumerate(card.get("accessories") or []):
            if isinstance(acc, dict) and acc.get("phrase"):
                text = str(acc["phrase"]) + (f" ({acc['placement']})" if acc.get("placement") else "")
                checks.append(Check(f"accessory:{i}", True, text))
    for prefix, values in (("guard", card.get("neg_guards")), ("forbidden", card.get("forbidden")),
                           ("excluded", (card.get("register") or {}).get("excluded")),
                           ("neg_add", _split(outfit.get("neg_add")))):
        for i, value in enumerate(v for v in values or [] if isinstance(v, str) and v.strip()):
            checks.append(Check(f"{prefix}:{i}", False, value.strip()))
    return checks
