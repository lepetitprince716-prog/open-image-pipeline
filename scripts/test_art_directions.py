#!/usr/bin/env python3
"""Unit tests for the art direction cards (data + query CLI; offline, no GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import art_directions as ad
import art_prompt
import model_quirks

HERE = Path(__file__).resolve().parent


def test_validate_cli_passes() -> None:
    proc = subprocess.run([sys.executable, str(HERE / "art_directions.py"), "validate"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_get_exact_match_and_miss() -> None:
    c = ad.get("moonlit_woodblock")
    assert c is not None and c["slug"] == "moonlit_woodblock"
    assert ad.get("moonlit") is None  # exact match only — no fuzzy surprises in a decision layer


def test_every_card_core_invariants() -> None:
    all_cards = ad.cards()
    assert len(all_cards) >= 10  # deck grows via candidate promotion; never hardcode the count
    for slug, c in all_cards.items():
        assert set(c["slots"]) <= set(ad.SLOT_KEYS), slug
        assert c["aspect"] in art_prompt.ASPECTS, slug
        assert c["bg_brief"].strip(), slug  # the background-line interface is mandatory
        prim = model_quirks.card(c["base"]["primary"])
        assert prim is not None, slug
        q_neg = prim.get("neg_add", "")
        if q_neg:  # quirks sync: the card must carry its primary base's negatives
            assert q_neg in c.get("neg_add", ""), slug
        if prim.get("no_freeu"):
            assert "--no-freeu" in c.get("flags", []), slug


def test_validated_winner_concepts_are_frozen() -> None:
    """C2 (EXPERIMENT_LOG winner) must stay queryable as a card. C4-505 rembrandt_workshop
    was retired to the candidates file by user verdict 2026-08-04 and must stay OUT."""
    moon = ad.get("moonlit_woodblock")
    assert "pose_ref" not in moon            # N3: cards no longer pin a skeleton
    assert moon["pose_tags_any"] == ["ballet", "dance", "dynamic"]  # affinity hint stays
    assert "contre-jour" in moon["slots"]["lighting"]
    assert ad.get("rembrandt_workshop") is None


def test_skeleton_cards_sit_on_ilustmix() -> None:
    """N3: no card pins a skeleton anymore — pose selection is explicit (--pose-ref) or
    sampled by the N4 combo engine; the old iLustMix-only skeleton rule is vacuous."""
    assert all("pose_ref" not in c for c in ad.cards().values())


def test_expression_none_is_a_legal_pin() -> None:
    c = ad.get("blue_hour_silhouette")
    assert c["slots"]["expression"] == "none"  # silhouette: no face, slot deliberately dropped
