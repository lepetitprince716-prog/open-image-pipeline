#!/usr/bin/env python3
"""Unit tests for pose_library metadata edits (pure functions — no GPU, no index writes)."""
from __future__ import annotations

import argparse

import pytest

from pose_library import (
    _annotation_updates,
    _apply_annotation,
    _apply_prompt_hint_manifest,
    _load_prompt_hint_manifest,
)

ENTRY = {
    "slug": "cabaret_seated", "file": "outputs/poses/library/cabaret_seated.png",
    "tags": ["seated", "elegant"], "desc": "performer seated on steps", "frame": "full",
    "dims": [768, 1344], "added": "2026-07-01",
}


def test_apply_annotation_merges_pairing_per_key() -> None:
    first = _apply_annotation(ENTRY, {"pairing": {"lighting": ["warm indoor lamplight, cozy enveloping glow"],
                                                  "camera": ["upper body shot"]}})
    assert first["pairing"]["lighting"] and first["pairing"]["camera"]
    assert ENTRY.get("pairing") is None  # pure: the input entry is untouched
    # a later annotate that only sets camera must keep the earlier lighting
    second = _apply_annotation(first, {"pairing": {"camera": ["cowboy shot", "three-quarter view"]}})
    assert second["pairing"]["lighting"] == first["pairing"]["lighting"]
    assert second["pairing"]["camera"] == ["cowboy shot", "three-quarter view"]


def test_apply_annotation_scalar_fields() -> None:
    out = _apply_annotation(ENTRY, {"energy": "static", "canvas": "9:16",
                                    "desc": "new desc", "prompt_hint": "seated on steps, hand to hat",
                                    "tags": ["a", "b"]})
    assert (out["energy"], out["canvas"], out["desc"], out["tags"]) == ("static", "9:16", "new desc", ["a", "b"])
    assert out["prompt_hint"] == "seated on steps, hand to hat"
    unchanged = _apply_annotation(ENTRY, {"pairing": {}, "energy": None, "desc": "", "prompt_hint": "",
                                          "tags": []})
    assert unchanged == ENTRY  # empty updates are no-ops


def test_annotation_updates_reads_repeatable_flags() -> None:
    ns = argparse.Namespace(
        pair_lighting=["rim light, bright edge outline along hair and shoulders, face mostly in shadow"],
        pair_camera=["upper body shot"], pair_composition=None, pair_color_mood=None,
        pair_expression=None, avoid_camera=["full body shot"], avoid_lighting=None,
        pair_notes="hands are the story", energy="static", canvas=None, desc="d",
        prompt_hint="seated, cane across shoulders", tags="x, y")
    upd = _annotation_updates(ns, include_desc_tags=True)
    assert upd["pairing"]["lighting"][0].startswith("rim light")
    assert upd["pairing"]["avoid_camera"] == ["full body shot"]
    assert upd["pairing"]["notes"] == "hands are the story"
    assert upd["energy"] == "static" and upd["desc"] == "d" and upd["tags"] == ["x", "y"]
    assert upd["prompt_hint"] == "seated, cane across shoulders"
    # add-time collection must not touch desc/prompt_hint/tags (the entry already carries them)
    add_time = _annotation_updates(ns)
    assert "desc" not in add_time and "prompt_hint" not in add_time


def test_tracked_prompt_hint_manifest_is_clean_and_complete() -> None:
    hints = _load_prompt_hint_manifest()
    assert len(hints) == 24
    assert hints["arabesque_butterfly"] == (
        "ballet arabesque en pointe, torso leaning forward, both arms reaching forward, "
        "back leg extended straight behind"
    )
    assert hints["warrior_yoga_v2"] == "warrior yoga pose, wide lunge, arms extended overhead"


def test_prompt_hint_backfill_is_pure_idempotent_and_conflict_aware() -> None:
    poses = [{"slug": "a", "prompt_hint": ""}, {"slug": "b", "prompt_hint": "pose b"}]
    hints = {"a": "pose a", "b": "pose b"}
    updated, changed = _apply_prompt_hint_manifest(poses, hints)
    assert changed == ["a"]
    assert [p["prompt_hint"] for p in updated] == ["pose a", "pose b"]
    assert poses[0]["prompt_hint"] == ""  # input remains untouched
    assert _apply_prompt_hint_manifest(updated, hints)[1] == []

    with pytest.raises(ValueError, match="conflict"):
        _apply_prompt_hint_manifest([{"slug": "a", "prompt_hint": "different"}], {"a": "pose a"})
    with pytest.raises(ValueError, match="mismatch"):
        _apply_prompt_hint_manifest([{"slug": "a", "prompt_hint": ""}], {"b": "pose b"})
