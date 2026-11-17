#!/usr/bin/env python3
"""Card purity lint - N0 contract of the five-layer decoupling plan, green as of N1.

Direction / outfit card faces must not carry character identity tokens. The
token table is built from ``characters/*.pack.json`` ``prompt_identity``
(``identity_tags`` + ``color_lock``); scanned texts are the card-face fields
of art_directions.json / anima_directions.json / outfit_cards.json and the
positive/negative direction texts of sdxl_structured_pilot.json.

Provenance fields (``note``, ``validation``, ...) are excluded from the scan:
they record how the tokens were validated; they do not inject them.

Acceptance contract (docs/direction_decoupling_plan.md N0/N1):
- N0 locked the exact pre-cleanup violation baseline (25 hits / 9 entries:
  the 8 color-lock ``sfw_note`` cards + the registry komorebi bake).
- N1 cleared the card faces (color locks moved to pack-driven ``color_hijack``
  injection) and emptied BASELINE; the lint is green.
- New cards must land with zero violations: any identity token on a card face
  fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Card-face fields scanned for identity tokens. Provenance / validation notes
# are deliberately excluded (they are history, not binding).
CARD_FACE_FIELDS = frozenset(
    {
        "intent",
        "slots",
        "neg_add",
        "sfw_note",
        "bg_brief",
        "positive_guards",
        "qc_cull",
        "tags",
        "components",
        "axes",
        "pairing",
    }
)
DIRECTION_TEXT_FIELDS = frozenset({"positive_text", "negative_text_add"})
CARD_FILES = (
    "scripts/art_directions.json",
    "scripts/anima_directions.json",
    "scripts/outfit_cards.json",
    "scripts/nai_artist_strings.json",
)

# N1 cleaned the card faces and emptied the baseline: the lint is green now.
# Character packs are private inputs and are intentionally absent from the public tree.
BASELINE: list[tuple[str, str, str, str]] = []


def _pack_paths() -> list[Path]:
    paths = sorted((ROOT / "characters").glob("*.pack.json"))
    if not paths:
        pytest.skip("character packs are not present")
    return paths


def token_table() -> list[str]:
    """Identity tokens owned by the packs (identity_tags + color_lock)."""
    tokens: set[str] = set()
    for pack_path in _pack_paths():
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        identity = pack.get("prompt_identity") or {}
        for key in ("identity_tags", "color_lock"):
            for token in identity.get(key) or []:
                token = str(token).strip()
                if token:
                    tokens.add(token)
    return sorted(tokens)


def _matched_tokens(text: str, tokens: list[str]) -> set[str]:
    hits: set[str] = set()
    for token in tokens:
        if re.search(r"(?<![a-z])" + re.escape(token) + r"(?![a-z])", text, re.IGNORECASE):
            hits.add(token)
    return hits


def _card_face_violations(tokens: list[str]) -> list[tuple[str, str, str, str]]:
    found: list[tuple[str, str, str, str]] = []
    for relative in CARD_FILES:
        path = ROOT / relative
        if not path.exists():
            continue  # all CARD_FILES are present as of N2
        data = json.loads(path.read_text(encoding="utf-8"))
        for slug, card in (data.get("cards") or {}).items():
            for field, value in card.items():
                if field not in CARD_FACE_FIELDS:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for token in sorted(_matched_tokens(text, tokens)):
                    found.append((relative, slug, field, token))
    return found


def _registry_direction_violations(tokens: list[str]) -> list[tuple[str, str, str, str]]:
    found: list[tuple[str, str, str, str]] = []
    registry = json.loads((ROOT / "scripts/sdxl_structured_pilot.json").read_text(encoding="utf-8"))
    for slug, direction in (registry.get("directions") or {}).items():
        for field, value in direction.items():
            if field not in DIRECTION_TEXT_FIELDS:
                continue
            for token in sorted(_matched_tokens(value, tokens)):
                found.append(("scripts/sdxl_structured_pilot.json", f"directions.{slug}", field, token))
    return found


def violations() -> list[tuple[str, str, str, str]]:
    tokens = token_table()
    found = _card_face_violations(tokens) + _registry_direction_violations(tokens)
    return sorted(set(found))


def test_purity_baseline_is_exact() -> None:
    """N0: the lint must mechanically find exactly the locked baseline.

    N1 clears the card faces and empties BASELINE; the test then proves the
    deck is clean. Provenance notes stay untouched (history, not binding).
    """
    assert violations() == BASELINE


def test_token_table_is_populated() -> None:
    assert len(token_table()) >= 3
    assert "white hair" in token_table()
    assert "silver hair" in token_table()


def test_packs_validate_against_schema_v1_1() -> None:
    """Every pack must satisfy the v1.1 contract (jsonschema)."""
    import jsonschema

    schema = json.loads((ROOT / "scripts/character_pack.schema.json").read_text(encoding="utf-8"))
    assert schema["$id"] == "loramake-character-pack-v1.1"
    assert "prompt_identity" in schema["properties"]
    for pack_path in _pack_paths():
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        jsonschema.validate(pack, schema)


def test_character_pack_carries_the_decoupled_identity_contract() -> None:
    pack = json.loads(_pack_paths()[0].read_text(encoding="utf-8"))
    identity = pack["prompt_identity"]
    assert identity["color_lock"]
    assert identity["neg_guards"]
    assert identity["register"]["exclude"]
    assert pack["trigger"]
