"""Offline tests for the segm extraction layer. No box, no GPU, no enum cache.

``build_graph`` is pure, so its shape is asserted exactly: a wrong output slot on
UltralyticsDetectorProvider (BBOX_DETECTOR is slot 0, SEGM_DETECTOR is slot 1) fails on the
box with a type error, and a non-zero dilation silently fattens every mask this line exists
to keep tight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loramake import region_segm  # noqa: E402
from loramake.mask_spec import SegmSelect  # noqa: E402
from loramake.region_mask import MaskError  # noqa: E402

ENCODED = "segm/Anzhc%20Face%20seg%20640%20v4%20y11n.pt"
KNOWN = {ENCODED, "segm/Anzhc%20Eyes%20-seg-hd.pt", "segm/foot_yolov8x.pt"}


@pytest.fixture(autouse=True)
def _no_enum_disk_cache(monkeypatch):
    """Never read or write the shared object_info cache from a test."""
    monkeypatch.setenv("LORAMAKE_ENUM_CACHE", "0")


def test_graph_takes_the_segm_detector_slot():
    g = region_segm.build_graph(src_name="s.png", model_name=ENCODED)
    assert g["11"]["inputs"]["model_name"] == ENCODED
    assert g["12"]["inputs"]["segm_detector"] == ["11", 1]
    assert g["12"]["inputs"]["image"] == ["10", 0]


def test_graph_keeps_the_mask_tight():
    g = region_segm.build_graph(src_name="s.png", model_name=ENCODED)
    assert g["12"]["inputs"]["dilation"] == 0


def test_graph_without_index_has_no_filter_node():
    g = region_segm.build_graph(src_name="s.png", model_name=ENCODED, index="all")
    assert "13" not in g
    assert g["14"]["inputs"]["segs"] == ["12", 0]


def test_graph_index_picks_one_segment_by_confidence():
    g = region_segm.build_graph(src_name="s.png", model_name=ENCODED, index=2)
    assert g["13"]["inputs"] == {
        "segs": ["12", 0], "target": "confidence", "order": True,
        "take_start": 2, "take_count": 1}
    assert g["14"]["inputs"]["segs"] == ["13", 0]


def test_graph_threshold_is_passed_through():
    g = region_segm.build_graph(src_name="s.png", model_name=ENCODED, threshold=0.25)
    assert g["12"]["inputs"]["threshold"] == 0.25


def test_resolve_models_maps_friendly_keys(monkeypatch):
    monkeypatch.setattr(region_segm, "known_models", lambda *a, **k: set(KNOWN))
    sel = SegmSelect(model="anzhc_face")
    assert region_segm.resolve_models([sel]) == {sel: ENCODED}


def test_resolve_models_reprobes_live_before_rejecting(monkeypatch):
    """A stale cached enum may delay a rejection by one round-trip; it may never cause one."""
    calls: list[bool] = []

    def fake_known(host=None, *, refresh=False):
        calls.append(refresh)
        return {"segm/old.pt"} if not refresh else set(KNOWN)

    monkeypatch.setattr(region_segm, "known_models", fake_known)
    sel = SegmSelect(model="anzhc_face")
    assert region_segm.resolve_models([sel]) == {sel: ENCODED}
    assert calls == [False, True]


def test_resolve_models_reports_a_genuine_miss(monkeypatch):
    monkeypatch.setattr(region_segm, "known_models", lambda *a, **k: set(KNOWN))
    with pytest.raises(MaskError, match="not installed on the box"):
        region_segm.resolve_models([SegmSelect(model="segm/nope.pt")])


def test_fetch_masks_refuses_an_unresolved_selection(monkeypatch):
    monkeypatch.setattr(region_segm, "upload_image", lambda path: "s.png")
    with pytest.raises(MaskError, match="unresolved segm selections"):
        region_segm.fetch_masks("/tmp/s.png", [SegmSelect(model="anzhc_face")], resolved={})


def test_fetch_masks_refuses_a_failed_upload(monkeypatch):
    sel = SegmSelect(model="anzhc_face")
    monkeypatch.setattr(region_segm, "upload_image", lambda path: None)
    with pytest.raises(MaskError, match="failed to upload"):
        region_segm.fetch_masks("/tmp/s.png", [sel], resolved={sel: ENCODED})


def test_fetch_masks_refuses_a_failed_extraction(monkeypatch):
    sel = SegmSelect(model="anzhc_face")
    monkeypatch.setattr(region_segm, "upload_image", lambda path: "s.png")
    monkeypatch.setattr(region_segm, "submit_and_fetch", lambda *a, **k: None)
    with pytest.raises(MaskError, match="segm extraction failed"):
        region_segm.fetch_masks("/tmp/s.png", [sel], resolved={sel: ENCODED})


def test_fetch_masks_returns_a_greyscale_mask(monkeypatch):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 48), (255, 255, 255)).save(buf, format="PNG")
    sel = SegmSelect(model="anzhc_face")
    monkeypatch.setattr(region_segm, "upload_image", lambda path: "s.png")
    monkeypatch.setattr(region_segm, "submit_and_fetch",
                        lambda *a, **k: [(buf.getvalue(), {"filename": "m.png"})])
    masks = region_segm.fetch_masks("/tmp/s.png", [sel], resolved={sel: ENCODED})
    assert masks[sel].mode == "L"
    assert masks[sel].size == (64, 48)


def test_fetch_masks_refuses_an_unreadable_payload(monkeypatch):
    sel = SegmSelect(model="anzhc_face")
    monkeypatch.setattr(region_segm, "upload_image", lambda path: "s.png")
    monkeypatch.setattr(region_segm, "submit_and_fetch",
                        lambda *a, **k: [(b"not a png", {"filename": "m.png"})])
    with pytest.raises(MaskError, match="unreadable image"):
        region_segm.fetch_masks("/tmp/s.png", [sel], resolved={sel: ENCODED})
