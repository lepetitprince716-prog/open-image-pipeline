#!/usr/bin/env python3
"""Unit tests for art_prompt.py (pure prompt-assembly logic; no GPU)."""
from __future__ import annotations

import pytest

import art_prompt as ap


def test_default_keeps_segments_as_wildcards() -> None:
    r = ap.build("1girl, solo, silver hair")
    for token in ("__medium__", "__camera__", "__lighting__", "__pose__", "__color_mood__"):
        assert token in r.positive
    assert r.positive.endswith("1girl, solo, silver hair")  # subject is last
    assert (r.width, r.height) == (832, 1216)               # default portrait
    assert r.sampler is None                                # recipe default


def test_negative_has_anatomy_guards_and_signature_block() -> None:
    neg = ap.build("1girl").negative
    for guard in (
        "asymmetrical eyes",
        "deformed",
        "mutated",
        "artist name",
        "chromatic aberration",
        "extra fingers",
        "extra toes",
        "fused fingers",
        "fused toes",
    ):
        assert guard in neg
    for guard in ap.LINE_GUARDS:  # N1 line-strategy guards ride the artistic baseline
        assert guard in neg


def test_artist_mode_requires_artist() -> None:
    with pytest.raises(ValueError):
        ap.build("1girl", style_mode="artist")


def test_artist_tags_formatting_and_position() -> None:
    r = ap.build("1girl, solo", style_mode="artist", artist="by ke-ta:1.0, wlop:0.5")
    assert "(by ke-ta:1.0)" in r.positive
    assert "(by wlop:0.5)" in r.positive
    # artist tags lead the body, ahead of the subject
    assert r.positive.index("(by ke-ta:1.0)") < r.positive.index("1girl, solo")


def test_artist_bare_name_unweighted() -> None:
    assert ap._artist_tags("ciloranko") == "(by ciloranko)"


def test_fixed_segment_overrides_wildcard() -> None:
    r = ap.build("1girl", medium="oil painting", camera="cowboy shot")
    assert "oil painting" in r.positive and "__medium__" not in r.positive
    assert "cowboy shot" in r.positive and "__camera__" not in r.positive
    assert "__lighting__" in r.positive  # untouched segments still vary


def test_segment_none_drops_it() -> None:
    r = ap.build("1girl", pose="none")
    assert "__pose__" not in r.positive


def test_aspect_and_detail() -> None:
    r = ap.build("1girl", aspect="tall", detail=True)
    assert (r.width, r.height) == (768, 1344)
    assert r.sampler == ap.DETAIL_SAMPLER


def test_unknown_aspect_and_mode_raise() -> None:
    with pytest.raises(ValueError):
        ap.build("1girl", aspect="banner")
    with pytest.raises(ValueError):
        ap.build("1girl", style_mode="painterly")


def test_empty_subject_raises() -> None:
    with pytest.raises(ValueError):
        ap.build("   ")


def test_to_command_shape() -> None:
    r = ap.build("1girl", detail=True)
    cmd = ap.to_command(r, ["char:0.8", "Illustrious\\style\\foo.safetensors:0.6"])
    assert "--prompt" in cmd and "--neg" in cmd
    assert "--res" in cmd and "832x1216" in cmd
    assert cmd[cmd.index("--sampler") + 1] == ap.DETAIL_SAMPLER
    assert cmd.count("--lora") == 2


def test_to_command_no_sampler_when_not_detail() -> None:
    cmd = ap.to_command(ap.build("1girl"))
    assert "--sampler" not in cmd


# --- v2: composition slot (default-on) + expression slot (opt-in) --------------------------------

def test_composition_default_wildcard_sits_after_camera() -> None:
    r = ap.build("1girl")
    assert "__composition__" in r.positive
    assert r.positive.index("__camera__") < r.positive.index("__composition__") \
        < r.positive.index("__lighting__")


def test_composition_pin_and_drop() -> None:
    pinned = ap.build("1girl", composition="negative space, small figure in a wide empty field")
    assert "negative space" in pinned.positive and "__composition__" not in pinned.positive
    dropped = ap.build("1girl", composition="none")
    assert "__composition__" not in dropped.positive


def test_expression_absent_by_default() -> None:
    assert "__expression__" not in ap.build("1girl").positive  # opt-in: no slot unless asked


def test_expression_vary_literal_and_none() -> None:
    vary = ap.build("1girl, solo", expression="vary")
    assert "__expression__" in vary.positive
    # character-adjacent: after color_mood, right before the subject
    assert vary.positive.index("__color_mood__") < vary.positive.index("__expression__") \
        < vary.positive.index("1girl, solo")
    pinned = ap.build("1girl", expression="cool aloof expression")
    assert "cool aloof expression" in pinned.positive and "__expression__" not in pinned.positive
    assert "__expression__" not in ap.build("1girl", expression="none").positive
