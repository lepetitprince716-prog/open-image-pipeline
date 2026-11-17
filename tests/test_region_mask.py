"""Geometry tests for loramake.region_mask.

A mask that is off by a scale factor repaints the wrong part of the picture and still produces
a plausible-looking image, so these assert the refusals as hard as the happy path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loramake import region_mask  # noqa: E402


def _mask(size=(512, 512), paint=None) -> Image.Image:
    """Black mask of ``size`` with an optional white rectangle ``(x, y, w, h)``."""
    img = Image.new("L", size, 0)
    if paint:
        x, y, w, h = paint
        img.paste(255, (x, y, x + w, y + h))
    return img


# --- size discipline -------------------------------------------------------------------

def test_a_mask_smaller_than_its_source_is_refused() -> None:
    # The CSS-pixels-instead-of-image-pixels bug: a 512-wide canvas export for a 2048 image.
    with pytest.raises(region_mask.MaskError) as err:
        region_mask.validate_size((512, 768), (2048, 3072))
    assert "true pixel size" in str(err.value)


def test_a_mask_matching_its_source_passes() -> None:
    region_mask.validate_size((2048, 3072), (2048, 3072))


def test_prepare_refuses_a_mismatched_mask_before_doing_any_geometry() -> None:
    with pytest.raises(region_mask.MaskError):
        region_mask.prepare(_mask((512, 512), (10, 10, 50, 50)), (1024, 1024))


# --- emptiness and coverage ------------------------------------------------------------

def test_an_empty_mask_is_refused() -> None:
    with pytest.raises(region_mask.MaskError) as err:
        region_mask.work_box(_mask())
    assert "empty" in str(err.value)


def test_a_mask_painted_below_the_threshold_counts_as_empty() -> None:
    # A canvas that exported a faint overlay instead of a real selection.
    faint = Image.new("L", (256, 256), region_mask.WHITE_THRESHOLD - 1)
    assert region_mask.white_bbox(faint) is None


def test_coverage_reports_the_painted_share() -> None:
    img = _mask((100, 100), (0, 0, 50, 100))
    assert region_mask.coverage(img) == pytest.approx(0.5, abs=0.01)


def test_a_full_frame_mask_reports_coverage_above_the_whole_image_cutoff() -> None:
    img = _mask((64, 64), (0, 0, 64, 64))
    assert region_mask.coverage(img) > region_mask.WHOLE_IMAGE_COVERAGE


# --- the work box ----------------------------------------------------------------------

def test_the_work_box_surrounds_the_painted_area_with_a_margin() -> None:
    box = region_mask.work_box(_mask((512, 512), (200, 200, 64, 64)), padding=32)
    assert box.x <= 200 - 32 + region_mask.ALIGN
    assert box.y <= 200 - 32 + region_mask.ALIGN
    assert box.x + box.width >= 264 + 32 - region_mask.ALIGN
    assert box.y + box.height >= 264 + 32 - region_mask.ALIGN


def test_the_work_box_never_leaves_the_frame() -> None:
    # Painted hard into the corner: the margin has nowhere to go on two sides.
    box = region_mask.work_box(_mask((256, 256), (0, 0, 32, 32)), padding=64)
    assert box.x == 0 and box.y == 0
    assert box.x + box.width <= 256
    assert box.y + box.height <= 256


def test_the_work_box_is_aligned() -> None:
    box = region_mask.work_box(_mask((512, 512), (101, 203, 37, 41)), padding=13)
    assert box.x % region_mask.ALIGN == 0
    assert box.y % region_mask.ALIGN == 0


def test_a_tiny_selection_is_grown_to_the_minimum_side() -> None:
    # A 4px dot must not be upscaled 256x into an invented object.
    box = region_mask.work_box(_mask((512, 512), (250, 250, 4, 4)), padding=0)
    assert box.width >= region_mask.MIN_SIDE
    assert box.height >= region_mask.MIN_SIDE


def test_a_frame_smaller_than_the_minimum_side_yields_the_whole_frame() -> None:
    box = region_mask.work_box(_mask((32, 32), (8, 8, 8, 8)), padding=0)
    assert (box.x, box.y, box.width, box.height) == (0, 0, 32, 32)


# --- scaling ---------------------------------------------------------------------------

def test_a_small_box_is_upscaled_towards_the_guide_size() -> None:
    scale = region_mask.upscale_factor(region_mask.Box(0, 0, 256, 256), 1024)
    assert scale == pytest.approx(4.0)


def test_an_already_large_box_is_never_downscaled() -> None:
    # Shrinking would throw away pixels we are about to composite back at full size.
    scale = region_mask.upscale_factor(region_mask.Box(0, 0, 2048, 2048), 1024)
    assert scale == 1.0


def test_scaled_size_stays_on_the_alignment_grid() -> None:
    width, height = region_mask.scaled_size(region_mask.Box(0, 0, 101, 203), 2.0)
    assert width % region_mask.ALIGN == 0
    assert height % region_mask.ALIGN == 0


# --- end to end ------------------------------------------------------------------------

def test_prepare_returns_a_crop_sized_exactly_for_the_graph() -> None:
    source = (1024, 1024)
    box, scale, target, crop = region_mask.prepare(
        _mask(source, (400, 400, 128, 128)), source, guide_size=1024
    )
    assert crop.size == target, "the uploaded mask must match the upscaled crop exactly"
    assert box.x + box.width <= source[0]
    assert box.y + box.height <= source[1]
    assert scale >= 1.0


def test_prepare_keeps_the_selection_inside_the_returned_crop() -> None:
    # The painted area must survive the crop; a box that missed it would repaint blank canvas.
    source = (512, 512)
    _box, _scale, _target, crop = region_mask.prepare(
        _mask(source, (100, 100, 60, 60)), source, guide_size=512
    )
    assert crop.getbbox() is not None, "the crop lost the painted area"


# --- softening the edge ----------------------------------------------------------------

def test_soften_grows_the_selection_before_blurring_it() -> None:
    # Blur alone eats the solid core inward and leaves the original hard contour visible just
    # outside the softened band. Growing first is what prevents that, so assert the painted
    # area actually got bigger rather than just fuzzier.
    img = _mask((256, 256), (100, 100, 40, 40))
    before = region_mask.white_bbox(img)
    after = region_mask.white_bbox(region_mask.soften(img, grow=8, blur=0))
    assert after.width > before.width
    assert after.height > before.height
    assert after.x < before.x


def test_soften_with_zero_radii_leaves_a_hard_edge() -> None:
    img = _mask((128, 128), (32, 32, 32, 32))
    assert region_mask.soften(img, grow=0, blur=0).tobytes() == img.tobytes()


def test_softening_keeps_the_interior_fully_opaque() -> None:
    # ComfyUI treats a value below 1.0 as "do not change", so a blur that dragged the middle
    # of the selection down to 254 would quietly stop repainting it.
    img = _mask((256, 256), (64, 64, 128, 128))
    softened = region_mask.soften(img)
    assert softened.getpixel((128, 128)) == 255


def test_prepare_softens_the_uploaded_crop() -> None:
    source = (512, 512)
    _b, _s, _t, hard = region_mask.prepare(
        _mask(source, (200, 200, 80, 80)), source, guide_size=512, grow=0, blur=0)
    _b2, _s2, _t2, soft = region_mask.prepare(
        _mask(source, (200, 200, 80, 80)), source, guide_size=512)
    # The softened crop has intermediate values at the edge; the hard one does not.
    assert set(hard.getextrema()) == {0, 255}, "a hard edge must stay purely binary"
    assert any(soft.histogram()[1:255]), "a softened edge must carry intermediate values"


# --- polarity --------------------------------------------------------------------------

def test_invert_selects_the_complement() -> None:
    # An RGBA export where the hole is transparent carries the opposite convention. Guessing
    # which one arrived is how the wrong half of a picture gets repainted, so the caller says.
    source = (256, 256)
    painted = _mask(source, (0, 0, 64, 256))
    normal, _s, _t, _c = region_mask.prepare(painted, source, guide_size=256, grow=0, blur=0)
    flipped, _s2, _t2, _c2 = region_mask.prepare(
        painted, source, guide_size=256, invert=True, grow=0, blur=0)
    assert normal.x == 0
    assert flipped.x > normal.x, "inverted selection should sit on the other side"


def test_an_inverted_full_mask_becomes_empty_and_is_refused() -> None:
    source = (128, 128)
    everything = _mask(source, (0, 0, 128, 128))
    with pytest.raises(region_mask.MaskError):
        region_mask.prepare(everything, source, invert=True)


def test_prepare_binarises_the_uploaded_mask() -> None:
    source = (256, 256)
    soft = Image.new("L", source, 0)
    soft.paste(200, (50, 50, 150, 150))
    _box, _scale, _target, crop = region_mask.prepare(soft, source, guide_size=256)
    # Interior stays fully white after the threshold; only resampled edges may sit between.
    assert crop.getextrema()[1] == 255
