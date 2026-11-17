"""Overlay renderer tests: sizes, red-fill fidelity, grid geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loramake import region_overlay  # noqa: E402
from loramake.region_mask import Box  # noqa: E402

SIZE = (400, 300)


def _img() -> Image.Image:
    return Image.new("RGB", SIZE, (40, 90, 40))


def test_grid_overlay_keeps_size_and_draws_lines():
    out = region_overlay.grid_overlay(_img())
    assert out.size == SIZE
    step = region_overlay._round_step(SIZE[0])
    # A grid line pixel differs from the flat background.
    assert out.getpixel((step, 150)) != (40, 90, 40)


def test_round_step_is_round():
    assert region_overlay._round_step(2048) % 50 == 0
    assert region_overlay._round_step(400) % 50 == 0
    assert region_overlay._round_step(64) >= 1


def test_hits_overlay_draws_labeled_boxes():
    detections = [{"bbox": [50, 50, 120, 110], "label": "feet", "confidence": 0.88}]
    out = region_overlay.hits_overlay(_img(), detections)
    assert out.size == SIZE
    assert out.getpixel((50, 80)) != (40, 90, 40)
    # Malformed detections are skipped, not fatal.
    region_overlay.hits_overlay(_img(), [{"bbox": None, "label": "feet"}])


def test_mask_preview_red_pixels_match_mask():
    mask = Image.new("L", SIZE, 0)
    for x in range(100, 150):
        for y in range(100, 130):
            mask.putpixel((x, y), 255)
    out = region_overlay.mask_preview(_img(), mask)
    inside = out.getpixel((120, 110))
    outside = out.getpixel((10, 10))
    assert inside[0] > outside[0]
    assert outside == (40, 90, 40)


def test_mask_preview_rejects_size_mismatch():
    mask = Image.new("L", (10, 10), 0)
    with pytest.raises(ValueError, match="mask is 10x10"):
        region_overlay.mask_preview(_img(), mask)


def test_mask_preview_outlines_boxes_and_work_box():
    mask = Image.new("L", SIZE, 0)
    for x in range(100, 150):
        for y in range(100, 130):
            mask.putpixel((x, y), 255)
    out = region_overlay.mask_preview(
        _img(),
        mask,
        boxes=(Box(100, 100, 50, 30),),
        keep_outs=(Box(10, 10, 20, 20),),
        work_box=Box(64, 64, 128, 96),
    )
    assert out.size == SIZE


def test_grid_steps_match_overlay_step():
    assert region_overlay.grid_steps(SIZE) == (
        region_overlay._round_step(SIZE[0]), region_overlay._round_step(SIZE[1]))


def test_cell_box_maps_and_clips():
    step_x, step_y = region_overlay.grid_steps(SIZE)
    box = region_overlay.cell_box(SIZE, "A1")
    assert box.as_tuple() == (0, 0, step_x, step_y)
    box = region_overlay.cell_box(SIZE, "b2")  # case-insensitive
    assert box.as_tuple() == (step_x, step_y, step_x, step_y)
    # last column clips to the frame edge
    import math
    last_col = chr(ord("A") + math.ceil(SIZE[0] / step_x) - 1)
    box = region_overlay.cell_box(SIZE, f"{last_col}1")
    assert box.x + box.width == SIZE[0]


def test_cell_box_rejects_bad_ids():
    for bad in ("", "9E", "E0", "ZZ", "E999"):
        with pytest.raises(ValueError):
            region_overlay.cell_box(SIZE, bad)


def test_cells_to_boxes_order_preserved():
    boxes = region_overlay.cells_to_boxes(SIZE, ["A1", "B1"])
    assert boxes[0].x < boxes[1].x


def test_merge_boxes_unions_touching_transitively():
    merged = region_overlay.merge_boxes(
        [Box(0, 0, 10, 10), Box(12, 0, 10, 10), Box(24, 0, 10, 10), Box(200, 200, 5, 5)],
        gap=4,
    )
    assert len(merged) == 2
    assert merged[0].as_tuple() == (0, 0, 34, 10)
    assert merged[1].as_tuple() == (200, 200, 5, 5)


def test_merge_boxes_zero_gap_keeps_separated():
    merged = region_overlay.merge_boxes([Box(0, 0, 10, 10), Box(20, 20, 10, 10)], gap=0)
    assert len(merged) == 2


def test_merge_boxes_zero_gap_unions_edge_adjacent():
    merged = region_overlay.merge_boxes([Box(0, 0, 10, 10), Box(10, 0, 10, 10)], gap=0)
    assert len(merged) == 1
    assert merged[0].as_tuple() == (0, 0, 20, 10)
