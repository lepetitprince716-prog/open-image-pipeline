#!/usr/bin/env python3
"""Unit tests for the N4 combo engine (offline, no GPU)."""
from __future__ import annotations

import pytest

import combo
import outfit_cards as oc


def test_same_seed_replays_the_same_combo() -> None:
    a = combo.sample(1234, scene=True)
    b = combo.sample(1234, scene=True)
    assert a["outfit_id"] == b["outfit_id"]
    assert a["action_id"] == b["action_id"]
    assert a["scene"] == b["scene"]
    assert a["action_text"] == b["action_text"]


def test_different_seed_changes_something() -> None:
    draws = [combo.sample(seed) for seed in range(20)]
    signatures = {(d["outfit_id"], d["action_id"]) for d in draws}
    assert len(signatures) >= 5  # the deck is small; several distinct combos must appear


def test_motion_class_is_compatible_with_outfit_motion_allow() -> None:
    for seed in range(30):
        d = combo.sample(seed)
        card = oc.get(d["outfit_id"])
        assert d["motion_class"] in card["motion_allow"], (seed, d)


def test_pinned_outfit_and_action_are_respected() -> None:
    d = combo.sample(7, outfit_id="gothic_corset_skater", action_id="twirling_in_place_skirt_flaring")
    assert d["outfit_id"] == "gothic_corset_skater"
    assert d["action_id"] == "twirling_in_place_skirt_flaring"


def test_incompatible_pinned_action_is_rejected() -> None:
    # gothic_corset_skater allows static/seated/dance — reclining actions must be rejected
    with pytest.raises(ValueError):
        combo.sample(7, outfit_id="gothic_corset_skater", action_id="lying_down_viewed_from_above")


def test_sfw_filter_keeps_sfw_cards() -> None:
    for seed in range(10):
        assert oc.get(combo.sample(seed)["outfit_id"])["sfw"] is True


def test_register_exclude_drops_marked_actions() -> None:
    for seed in range(10):
        d = combo.sample(seed, exclude_register={"bubbly-cute", "cutesy", "exaggerated"})
        assert "bubbly" not in d["action_text"] and "twirl" not in d["action_text"]


def test_scene_draw_is_deterministic_and_recorded() -> None:
    a = combo.sample(99, scene=True)
    assert a["scene"]["background"]
    assert a["scene"]["time_of_day"]
    assert combo.sample(99, scene=True)["scene"] == a["scene"]
