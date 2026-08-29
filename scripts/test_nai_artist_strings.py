#!/usr/bin/env python3
"""Unit tests for the NAI V5 artist string cards (offline, no network)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nai_artist_strings as nas
import test_card_purity as purity

HERE = Path(__file__).resolve().parent

# Artist string cards are the global style layer: card faces must never describe
# the subject (universality contract). Single style idioms that merely contain a
# banned word as a substring (girlish, mangaka) do not trip the word-boundary scan.
SUBJECT_LANGUAGE = ("girl", "boy", "woman", "man", "hair", "eyes", "skin")

# Card-face fields scanned by the identity + subject-language checks (note and
# source are provenance, verified is history: all excluded).
SCANNED_FIELDS = ("name_cn", "intent", "components", "axes", "pairing", "neg_add")


def test_validate_cli_passes() -> None:
    proc = subprocess.run([sys.executable, str(HERE / "nai_artist_strings.py"), "validate"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_get_exact_match_and_miss() -> None:
    first = next(iter(nas.cards()))
    c = nas.get(first)
    assert c is not None and c["id"] == first
    assert nas.get("definitely_not_a_card") is None


def test_compose_block_golden_weighted() -> None:
    """Lock the exact block shape so the composition rule cannot drift."""
    card = {"components": [
        {"tag": "ciloranko", "kind": "artist", "weight": 1.3},
        {"tag": "watercolor (medium)", "kind": "style", "weight": 0.8},
    ]}
    assert nas.compose_block(card) == "1.3::artist:ciloranko ::, 0.8::watercolor (medium) ::"


def test_compose_block_golden_unweighted_and_integer() -> None:
    card = {"components": [
        {"tag": "mika pikazo", "kind": "artist", "weight": None},
        {"tag": "cel shading", "kind": "style", "weight": 2},
    ]}
    assert nas.compose_block(card) == "artist:mika pikazo, 2::cel shading ::"


def test_every_card_core_invariants() -> None:
    all_cards = nas.cards()
    assert len(all_cards) >= 12  # first batch; deck grows via calibration
    for card_id, c in all_cards.items():
        assert c["structure"] in nas.STRUCTURES, card_id
        assert c["status"] in nas.STATUSES, card_id
        assert c["sfw"] is True, card_id  # v1 deck is SFW-only
        assert c["source"].strip(), card_id
        block = nas.compose_block(c)
        assert block, card_id
        has_artist = any(comp["kind"] == "artist" for comp in c["components"])
        assert ("artist:" in block) == has_artist, card_id
        assert (c["structure"] == "style_only") == (not has_artist), card_id
        assert "{" not in block and "}" not in block, card_id  # V3 syntax is dead


def test_verified_status_couples_with_evidence() -> None:
    for card_id, c in nas.cards().items():
        if c["status"] == "verified":
            assert c["verified"], card_id
            assert (HERE.parent / c["verified"]["evidence"]).exists(), card_id
        else:
            assert c["verified"] is None, card_id


def _face_texts(c: dict) -> list[tuple[str, str]]:
    texts = []
    for field in SCANNED_FIELDS:
        value = c.get(field)
        if value is None:
            continue
        texts.append((field, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)))
    return texts


def test_faces_carry_no_identity_tokens() -> None:
    """Character identity/color tokens are pack-owned; style faces stay clean."""
    tokens = purity.token_table()
    for card_id, c in nas.cards().items():
        for field, text in _face_texts(c):
            hits = purity._matched_tokens(text, tokens)
            assert not hits, f"{card_id}.{field}: {sorted(hits)}"


def test_faces_carry_no_subject_language() -> None:
    """Universality contract: the style layer never describes the subject."""
    for card_id, c in nas.cards().items():
        for field, text in _face_texts(c):
            for term in SUBJECT_LANGUAGE:
                assert not re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])",
                                     text, re.IGNORECASE), f"{card_id}.{field}: {term}"


def test_purity_lint_scans_the_artist_deck() -> None:
    assert "scripts/nai_artist_strings.json" in purity.CARD_FILES
    for field in ("components", "axes", "pairing"):
        assert field in purity.CARD_FACE_FIELDS
    assert purity.violations() == purity.BASELINE
