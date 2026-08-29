"""Mask geometry for region-scoped edits: validate, find the work area, cut it out.

The canvas hands the server one flat PNG the size of the source image, white where the user
wants a change and black everywhere else.  Everything geometric happens here, in Python, for
one reason: a pan/zoom canvas is the classic place to confuse CSS pixels, image pixels and
devicePixelRatio, and a mask that is silently off by a scale factor repaints the wrong part of
the picture.  Doing the arithmetic here makes it testable; doing it with ComfyUI nodes would
not.

Polarity is fixed and not negotiable: white means "repaint this", black means "leave it
alone".  ``LoadImageMask`` is read on the red channel, so a plain greyscale or RGB PNG works.

The repo already has a precedent that only WARNs when a mask and its source disagree on size
and then proceeds anyway (scripts/region_prompt.py).  This module refuses instead — a
mis-scaled mask is not a warning, it is the wrong edit.
"""
from __future__ import annotations

from dataclasses import dataclass

# A pixel counts as "change me" at or above this value.  The canvas is expected to export a
# near-binary mask; the threshold exists so antialiased brush edges do not creep outward.
WHITE_THRESHOLD = 128

# Latent geometry is happier on multiples of 8, and the crop gets encoded by the VAE.
ALIGN = 8

# Below this the crop is too small to carry detail once it is upscaled; a stray 3px dot would
# otherwise be blown up to the full guide size and hallucinate something new.
MIN_SIDE = 64

# Above this share of the frame a "region" edit is really a whole-image edit, and the crop
# path costs more than it earns.
WHOLE_IMAGE_COVERAGE = 0.90

# Softening the selection edge is grow-then-blur, done here, and both halves matter.
#
# Grow first: blurring alone eats the solid core inward, which leaves the original hard
# contour visible just outside the softened band. ComfyUI's own VAEEncodeForInpaint defaults
# grow_mask_by=6 with the comment "grow mask by a few pixels to keep things seamless in latent
# space" -- the latent is an 8x downsample, so a few pixels of slack is not optional.
GROW_PX = 8

# Then blur, with a real gaussian. ComfyUI's FeatherMask is NOT a substitute: it multiplies the
# outermost N rows and columns of the mask TENSOR and never looks at the painted contour, so on
# a shape drawn in the middle of the frame it does essentially nothing. Feathering an irregular
# selection has to happen either here or via a contour-aware node; doing it here keeps the graph
# free of a custom-node dependency.
BLUR_PX = 6


class MaskError(ValueError):
    """The mask cannot be used as given."""


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def validate_size(mask_size: tuple[int, int], source_size: tuple[int, int]) -> None:
    """Refuse a mask that is not exactly the source image's pixel size."""
    if tuple(mask_size) != tuple(source_size):
        raise MaskError(
            f"mask is {mask_size[0]}x{mask_size[1]} but the source image is "
            f"{source_size[0]}x{source_size[1]}; the canvas must rasterise at the image's "
            f"true pixel size, not its on-screen size"
        )


def coverage(mask_img) -> float:
    """Share of the frame the user painted, in 0..1."""
    grey = mask_img.convert("L")
    total = grey.width * grey.height
    if not total:
        return 0.0
    hist = grey.histogram()
    painted = sum(hist[WHITE_THRESHOLD:])
    return painted / total


def white_bbox(mask_img) -> Box | None:
    """Tight bounding box of the painted area, or None when nothing is painted."""
    grey = mask_img.convert("L")
    # point() to a hard 0/255 first so getbbox() cannot be fooled by a faint antialias halo.
    binary = grey.point(lambda v: 255 if v >= WHITE_THRESHOLD else 0)
    bounds = binary.getbbox()
    if bounds is None:
        return None
    left, upper, right, lower = bounds
    return Box(left, upper, right - left, lower - upper)


def work_box(
    mask_img,
    *,
    padding: int = 32,
    align: int = ALIGN,
    min_side: int = MIN_SIDE,
) -> Box:
    """The crop the sampler should actually work on.

    The painted area plus a margin, grown to the alignment grid and clamped to the frame.  The
    margin matters: a crop cut exactly at the brush edge gives the sampler no surrounding
    context to blend into, which is where visible seams come from.
    """
    tight = white_bbox(mask_img)
    if tight is None:
        raise MaskError("mask is empty; nothing was selected")

    frame_w, frame_h = mask_img.width, mask_img.height

    left = tight.x - padding
    top = tight.y - padding
    right = tight.x + tight.width + padding
    bottom = tight.y + tight.height + padding

    # Grow to the alignment grid before clamping, so alignment never eats into the margin.
    left -= left % align
    top -= top % align
    right += (-right) % align
    bottom += (-bottom) % align

    left = max(0, left)
    top = max(0, top)
    right = min(frame_w, right)
    bottom = min(frame_h, bottom)

    width = right - left
    height = bottom - top

    # Enforce the floor by growing outward, then re-clamp; a frame smaller than min_side just
    # yields the whole frame.
    if width < min_side:
        deficit = min_side - width
        left = max(0, left - deficit // 2)
        right = min(frame_w, left + min_side)
        left = max(0, right - min_side)
        width = right - left
    if height < min_side:
        deficit = min_side - height
        top = max(0, top - deficit // 2)
        bottom = min(frame_h, top + min_side)
        top = max(0, bottom - min_side)
        height = bottom - top

    return Box(left, top, width, height)


def upscale_factor(box: Box, guide_size: int) -> float:
    """How much to enlarge the crop so the edited area gets real pixels to work with.

    Mirrors scripts/foot_inpaint.py: scale the longer side up to ``guide_size``.  Never
    downscale — an already-large selection is left alone rather than shrunk.
    """
    longest = max(box.width, box.height)
    if longest <= 0:
        raise MaskError("work box has no area")
    return max(1.0, round(guide_size / longest, 3))


def scaled_size(box: Box, scale: float) -> tuple[int, int]:
    """Crop size after upscaling, aligned so the VAE gets clean latent dimensions."""
    width = int(round(box.width * scale))
    height = int(round(box.height * scale))
    width -= width % ALIGN
    height -= height % ALIGN
    return max(ALIGN, width), max(ALIGN, height)


def soften(mask_img, *, grow: int = GROW_PX, blur: int = BLUR_PX):
    """Grow the selection, then blur its edge.  Order matters; see :data:`GROW_PX`.

    Works on the painted contour, unlike ``FeatherMask``.  A zero radius for either step skips
    that step, which is how a caller asks for a hard edge.
    """
    from PIL import ImageFilter

    out = mask_img
    if grow > 0:
        # MaxFilter is a square dilation; the kernel must be odd.
        out = out.filter(ImageFilter.MaxFilter(2 * grow + 1))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def normalize(mask_img, *, invert: bool = False):
    """The canonical form of a selection: greyscale, optional polarity flip, hard 0/255.

    Split out of :func:`prepare` so a caller that stores a mask can store it already
    canonical.  Storing the bytes a client sent instead means the polarity flip has to be
    repeated by every later reader, and any reader that forgets repaints the complement of
    what the user selected.

    The threshold is not cosmetic: ComfyUI rounds in several places and treats a value below
    1.0 as "do not change", so a selection exported at 254 instead of 255 would silently let
    the original show through.
    """
    grey = mask_img.convert("L")
    if invert:
        grey = grey.point(lambda v: 255 - v)
    return grey.point(lambda v: 255 if v >= WHITE_THRESHOLD else 0)


def prepare(
    mask_img,
    source_size: tuple[int, int],
    *,
    guide_size: int = 1024,
    padding: int = 32,
    invert: bool = False,
    grow: int = GROW_PX,
    blur: int = BLUR_PX,
):
    """Everything the graph needs, or a raised MaskError explaining why not.

    Returns ``(box, scale, target_size, crop)`` where ``crop`` is the mask cut to ``box``,
    resized to ``target_size`` and softened — the exact image to upload for ``LoadImageMask``.
    Callers should check :func:`coverage` first and route a near-full mask to the whole-image
    path instead of calling this.

    ``invert`` is the explicit polarity escape hatch.  The canonical form here is white =
    repaint, but a client that exports a hole-is-transparent RGBA has the opposite convention,
    and guessing which one arrived is how the wrong half of a picture gets repainted.  The
    caller decides and says so.
    """
    validate_size((mask_img.width, mask_img.height), source_size)

    binary = normalize(mask_img, invert=invert)

    box = work_box(binary, padding=padding)
    scale = upscale_factor(box, guide_size)
    target = scaled_size(box, scale)

    crop = binary.crop((box.x, box.y, box.x + box.width, box.y + box.height))
    crop = crop.resize(target, resample=_bilinear())
    crop = soften(crop, grow=grow, blur=blur)
    return box, scale, target, crop


def _bilinear():
    from PIL import Image

    return getattr(Image, "Resampling", Image).BILINEAR
