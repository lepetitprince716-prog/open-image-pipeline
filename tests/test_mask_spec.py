"""Golden tests for the mask-spec compiler.

The compiler's output is a hard binary mask, so these assert exact pixel values: a wrong
union or a double subtraction produces a plausible image downstream, not an error, and this
is the only layer where the mistake is still cheap to catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loramake import mask_spec  # noqa: E402
from loramake.region_mask import Box, MaskError  # noqa: E402

SIZE = (512, 384)


def _spec(**overrides) -> mask_spec.Spec:
    payload = {
        "schema": mask_spec.SCHEMA,
        "source": "/tmp/src.png",
        "boxes": [[100, 100, 50, 40]],
    }
    payload.update(overrides)
    return mask_spec.parse_spec(payload)


def test_parse_round_trip_canonical_json():
    spec = _spec(
        detect=[{"class": "feet", "index": 0}],
        keep_outs=[[10, 10, 5, 5]],
        mode="replace",
        grow=8,
    )
    again = mask_spec.parse_spec(mask_spec.canonical_json(spec))
    assert again == spec


def test_parse_rejects_unknown_keys():
    with pytest.raises(MaskError, match="unknown keys.*polygon"):
        mask_spec.parse_spec({"source": "/tmp/s.png", "boxes": [[0, 0, 1, 1]], "polygon": []})


def test_parse_rejects_unknown_detect_class():
    with pytest.raises(MaskError, match="unknown detector class"):
        _spec(detect=[{"class": "ribbons"}])


def test_parse_rejects_empty_selection():
    with pytest.raises(MaskError, match="selects nothing"):
        mask_spec.parse_spec({"source": "/tmp/s.png"})


def test_parse_rejects_bad_mode():
    with pytest.raises(MaskError, match="unknown mode"):
        _spec(mode="detail")


def test_parse_rejects_bad_box_shape():
    with pytest.raises(MaskError, match=r"boxes\[0\]"):
        _spec(boxes=[[0, 0, 10]])
    with pytest.raises(MaskError, match="non-positive"):
        _spec(boxes=[[0, 0, 0, 10]])


def test_compile_union_of_boxes():
    spec = _spec(boxes=[[10, 10, 20, 20], [100, 50, 30, 10]])
    mask = mask_spec.compile_mask(spec, SIZE)
    assert (mask.width, mask.height) == SIZE
    assert mask.getpixel((15, 15)) == 255
    assert mask.getpixel((129, 55)) == 255
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((60, 60)) == 0
    # Hard binary: nothing between 0 and 255.
    hist = mask.histogram()
    assert sum(hist[1:255]) == 0


def test_compile_keep_out_subtracts():
    spec = _spec(boxes=[[0, 0, 100, 100]], keep_outs=[[40, 40, 20, 20]])
    mask = mask_spec.compile_mask(spec, SIZE)
    assert mask.getpixel((10, 10)) == 255
    assert mask.getpixel((50, 50)) == 0
    assert mask.getpixel((39, 39)) == 255
    assert mask.getpixel((59, 59)) == 0
    assert mask.getpixel((60, 60)) == 255


def test_compile_clips_out_of_frame_boxes():
    spec = _spec(boxes=[[500, 370, 100, 100]])
    mask = mask_spec.compile_mask(spec, SIZE)
    assert mask.getpixel((511, 383)) == 255
    assert mask.getpixel((499, 369)) == 0


def test_compile_refuses_fully_outside_box():
    spec = _spec(boxes=[[600, 600, 10, 10]])
    with pytest.raises(MaskError, match="outside the image"):
        mask_spec.compile_mask(spec, SIZE)


def test_compile_refuses_keep_out_erasing_everything():
    spec = _spec(boxes=[[10, 10, 20, 20]], keep_outs=[[0, 0, 100, 100]])
    with pytest.raises(MaskError, match="erased the whole selection"):
        mask_spec.compile_mask(spec, SIZE)


def test_compile_refuses_whole_image_coverage():
    spec = _spec(boxes=[[0, 0, 512, 384]])
    with pytest.raises(MaskError, match="whole-image edit"):
        mask_spec.compile_mask(spec, SIZE)


def test_select_detections_by_index_and_all():
    detections = [
        {"bbox": [10, 10, 30, 30], "label": "feet", "confidence": 0.5},
        {"bbox": [200, 200, 250, 240], "label": "feet", "confidence": 0.9},
        {"bbox": [0, 0, 5, 5], "label": "hands", "confidence": 0.8},
    ]
    spec = _spec(boxes=[], detect=[{"class": "feet", "index": 0}])
    boxes = mask_spec.select_detections(spec, detections)
    # index 0 = strongest confidence, which is the second entry.
    assert boxes == [Box(200, 200, 50, 40)]

    spec_all = _spec(boxes=[], detect=[{"class": "feet", "index": "all"}])
    assert len(mask_spec.select_detections(spec_all, detections)) == 2


def test_select_detections_miss_is_hard_error():
    spec = _spec(boxes=[], detect=[{"class": "eyes"}])
    with pytest.raises(MaskError, match="no 'eyes'"):
        mask_spec.select_detections(spec, [])


def test_select_detections_index_out_of_range():
    detections = [{"bbox": [1, 1, 4, 4], "label": "feet", "confidence": 0.9}]
    spec = _spec(boxes=[], detect=[{"class": "feet", "index": 3}])
    with pytest.raises(MaskError, match="only 1 hit"):
        mask_spec.select_detections(spec, detections)


def test_compile_with_detections_unions_into_mask():
    detections = [{"bbox": [300, 300, 340, 330], "label": "hands", "confidence": 0.7}]
    spec = _spec(boxes=[[10, 10, 20, 20]], detect=[{"class": "hands"}])
    mask = mask_spec.compile_mask(spec, SIZE, detections)
    assert mask.getpixel((15, 15)) == 255
    assert mask.getpixel((320, 315)) == 255


def test_derive_bbox_is_aligned_and_padded():
    spec = _spec(boxes=[[100, 100, 50, 40]])
    mask = mask_spec.compile_mask(spec, SIZE)
    box = mask_spec.derive_bbox(mask)
    assert box.x % 8 == 0 and box.y % 8 == 0
    assert box.x <= 100 - 32 + 7 and box.y <= 100 - 32 + 7
    assert box.width >= 64 and box.height >= 64


# --- segm source -------------------------------------------------------------------------
#
# The compiler never talks to the box: the segmentation raster is injected, so every one of
# these stays offline and byte-deterministic.


def _segm_mask(box: tuple[int, int, int, int], *, size=SIZE, fill=255):
    img = Image.new("L", size, 0)
    x, y, w, h = box
    img.paste(fill, (x, y, x + w, y + h))
    return img


def test_parse_segm_round_trip_canonical_json():
    spec = _spec(segm=[{"model": "anzhc_face", "index": 0, "threshold": 0.4}])
    again = mask_spec.parse_spec(mask_spec.canonical_json(spec))
    assert again == spec


def test_parse_segm_accepts_bare_model_name():
    spec = _spec(segm=["anzhc_eyes"])
    assert spec.segm[0] == mask_spec.SegmSelect(model="anzhc_eyes", index="all", threshold=None)


def test_parse_segm_rejects_unknown_keys():
    with pytest.raises(MaskError, match="segm entry has unknown keys.*dilation"):
        _spec(segm=[{"model": "anzhc_face", "dilation": 4}])


def test_parse_segm_rejects_out_of_range_threshold():
    with pytest.raises(MaskError, match="threshold must be in 0..1"):
        _spec(segm=[{"model": "anzhc_face", "threshold": 1.5}])


def test_parse_segm_alone_is_a_valid_selection():
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": ["anzhc_face"]})
    assert spec.wants_segm() and not spec.boxes


def test_compile_unions_segm_with_boxes():
    spec = _spec(segm=["anzhc_face"])
    fetched = {spec.segm[0]: _segm_mask((300, 200, 20, 20))}
    mask = mask_spec.compile_mask(spec, SIZE, None, fetched)
    assert mask.getpixel((110, 110)) == 255   # the box
    assert mask.getpixel((310, 210)) == 255   # the segmentation
    assert mask.getpixel((400, 300)) == 0


def test_compile_binarises_a_soft_segm_mask():
    """MaskToImage hands back soft edges; the compiler's contract is hard 0/255."""
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": ["anzhc_face"]})
    soft = Image.new("L", SIZE, 0)
    soft.paste(200, (10, 10, 30, 30))   # above WHITE_THRESHOLD
    soft.paste(60, (40, 40, 60, 60))    # below it
    mask = mask_spec.compile_mask(spec, SIZE, None, {spec.segm[0]: soft})
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((50, 50)) == 0


def test_compile_applies_keep_outs_to_segm_area():
    spec = mask_spec.parse_spec({
        "source": "/tmp/s.png", "segm": ["anzhc_face"], "keep_outs": [[100, 100, 40, 40]]})
    mask = mask_spec.compile_mask(spec, SIZE, None, {spec.segm[0]: _segm_mask((90, 90, 60, 60))})
    assert mask.getpixel((95, 95)) == 255
    assert mask.getpixel((110, 110)) == 0


def test_compile_refuses_an_empty_segm_result():
    """The incident-#49 shape: an all-black mask repaints nothing while reporting success."""
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": [{"model": "anzhc_face", "index": 3}]})
    blank = Image.new("L", SIZE, 0)
    with pytest.raises(MaskError, match="selected no pixels"):
        mask_spec.compile_mask(spec, SIZE, None, {spec.segm[0]: blank})


def test_compile_refuses_a_missing_segm_fetch():
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": ["anzhc_face"]})
    with pytest.raises(MaskError, match="has no fetched mask"):
        mask_spec.compile_mask(spec, SIZE, None, {})


def test_compile_refuses_a_segm_mask_of_the_wrong_size():
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": ["anzhc_face"]})
    wrong = _segm_mask((0, 0, 10, 10), size=(256, 256))
    with pytest.raises(MaskError, match="the canvas must rasterise"):
        mask_spec.compile_mask(spec, SIZE, None, {spec.segm[0]: wrong})


def test_compile_refuses_a_whole_frame_segm_selection():
    spec = mask_spec.parse_spec({"source": "/tmp/s.png", "segm": ["anzhc_face"]})
    whole = Image.new("L", SIZE, 255)
    with pytest.raises(MaskError, match="whole-image edit"):
        mask_spec.compile_mask(spec, SIZE, None, {spec.segm[0]: whole})


# --- segm model-name resolution ----------------------------------------------------------
#
# The server enumerates these files percent-encoded. Every note, runbook and human types
# them with spaces, and the node rejects the space form outright.

KNOWN = {
    "bbox/face_yolov9c.pt",
    "segm/Anzhc%20Face%20seg%20640%20v4%20y11n.pt",
    "segm/Anzhc%20Eyes%20-seg-hd.pt",
    "segm/foot_yolov8x.pt",
    "segm/yolo11m-seg.pt",
}


def test_resolve_friendly_key_to_encoded_enum_name():
    assert mask_spec.resolve_segm_model("anzhc_face", KNOWN) == (
        "segm/Anzhc%20Face%20seg%20640%20v4%20y11n.pt")


def test_resolve_accepts_the_space_spelling():
    assert mask_spec.resolve_segm_model("segm/Anzhc Face seg 640 v4 y11n.pt", KNOWN) == (
        "segm/Anzhc%20Face%20seg%20640%20v4%20y11n.pt")


def test_resolve_passes_through_an_unvouched_raw_name():
    assert mask_spec.resolve_segm_model("segm/yolo11m-seg.pt", KNOWN) == "segm/yolo11m-seg.pt"


def test_resolve_refuses_a_bbox_only_model_shelved_under_segm():
    with pytest.raises(MaskError, match="bbox-only model shelved under segm/"):
        mask_spec.resolve_segm_model("segm/foot_yolov8x.pt", KNOWN)


def test_resolve_refuses_a_model_the_box_does_not_have():
    with pytest.raises(MaskError, match="not installed on the box"):
        mask_spec.resolve_segm_model("segm/does_not_exist.pt", KNOWN)


def test_resolve_with_a_failed_probe_does_not_fabricate_a_rejection():
    """An empty enum means the probe failed; comfy.missing_assets skips the guard, so do we."""
    assert mask_spec.resolve_segm_model("segm/whatever.pt", set()) == "segm/whatever.pt"
