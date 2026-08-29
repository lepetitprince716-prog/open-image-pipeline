"""Overlay renderers for the LLM region-selection loop.

Three pictures, all pure PIL, no ComfyUI:

* :func:`grid_overlay` — the source with a labelled coordinate grid, so a model reading the
  image can answer in real pixel coordinates instead of guessing scale.
* :func:`hits_overlay` — detector bounding boxes drawn on top, one colour per class, so the
  model can pick a hit by class+index instead of re-deriving its box.
* :func:`mask_preview` — the compiled mask as a red translucent fill plus the derived work
  box, which is what the human confirms before any GPU run.  The preview must show exactly
  the pixels the graph will repaint; anything prettier that diverges from the real mask
  defeats its purpose as the second evidence gate.

Grid lines snap to round pixel values so the labels read as usable numbers.  The grid also
carries A1-style cell ids: coarse anchors for a model that reasons better in cells than in
raw offsets.  Both refer to the same geometry, so either answer form compiles to the same
boxes.
"""
from __future__ import annotations

from loramake.region_mask import WHITE_THRESHOLD, Box

# One colour per detector class, chosen to stay readable on both light and dark art.
CLASS_COLORS = {
    "hands": (66, 135, 245),
    "eyes": (255, 170, 0),
    "faces": (0, 200, 120),
    "heads": (170, 110, 255),
    "persons": (120, 120, 120),
    "text": (255, 255, 0),
    "feet": (255, 80, 80),
}
DEFAULT_COLOR = (0, 220, 220)

GRID_COLOR = (255, 255, 255, 110)
GRID_TEXT = (255, 255, 255, 220)
GRID_TEXT_BG = (0, 0, 0, 140)
MASK_FILL = (255, 0, 0, 90)
WORK_BOX_COLOR = (255, 220, 0)
KEEP_OUT_COLOR = (255, 0, 0)
BOX_COLOR = (0, 255, 0)

_CELLS = 8


def _round_step(dim: int, cells: int = _CELLS) -> int:
    """A grid step near dim/cells snapped to a round number so labels stay legible."""
    raw = max(1, dim // cells)
    for unit in (500, 250, 200, 100, 50, 25, 20, 10, 5):
        if raw >= unit:
            return (raw // unit) * unit
    return raw


def grid_steps(size: tuple[int, int]) -> tuple[int, int]:
    """The (x, y) grid step used by :func:`grid_overlay` for an image of this size.

    Public because cell ids are an answer format: a model reads "E9" off the overlay and a
    caller converts it back to pixels.  Both directions must use the same step, so the
    conversion lives here with the grid.
    """
    return _round_step(size[0]), _round_step(size[1])


def cell_box(size: tuple[int, int], cell: str) -> Box:
    """One grid cell id ("A1".."Z99", as labelled by :func:`grid_overlay`) -> its pixel Box.

    Cells are a deliberately coarse answer format: a model that cannot report reliable pixel
    coordinates can still read a printed cell label.  The returned box is the full cell,
    clipped to the frame; select several cells to cover a region and let keep-outs trim.
    """
    cell = cell.strip().upper()
    if len(cell) < 2 or not cell[0].isalpha() or not cell[1:].isdigit():
        raise ValueError(f"cell id must look like E9, got {cell!r}")
    col = ord(cell[0]) - ord("A")
    row = int(cell[1:]) - 1
    if row < 0:
        raise ValueError(f"cell row starts at 1, got {cell!r}")
    step_x, step_y = grid_steps(size)
    x = col * step_x
    y = row * step_y
    if x >= size[0] or y >= size[1]:
        raise ValueError(
            f"cell {cell} is outside the {size[0]}x{size[1]} image "
            f"(grid step {step_x}x{step_y})"
        )
    return Box(x, y, min(step_x, size[0] - x), min(step_y, size[1] - y))


def cells_to_boxes(size: tuple[int, int], cells: list[str]) -> list[Box]:
    return [cell_box(size, c) for c in cells]


def merge_boxes(boxes: list[Box], *, gap: int = 0) -> list[Box]:
    """Union boxes that touch (within ``gap`` px) into one bounding box per group.

    Cells are coarse, so an elongated target (a toe row) spans several of them; cropping
    each cell separately cuts the subject at every boundary and makes a count question
    unanswerable. Grouping is transitive: A touching B touching C yields one box.
    """
    remaining = list(boxes)
    merged: list[Box] = []
    while remaining:
        group = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if (
                    group.x - gap <= other.x + other.width
                    and other.x - gap <= group.x + group.width
                    and group.y - gap <= other.y + other.height
                    and other.y - gap <= group.y + group.height
                ):
                    x0 = min(group.x, other.x)
                    y0 = min(group.y, other.y)
                    x1 = max(group.x + group.width, other.x + other.width)
                    y1 = max(group.y + group.height, other.y + other.height)
                    group = Box(x0, y0, x1 - x0, y1 - y0)
                    remaining.remove(other)
                    changed = True
        merged.append(group)
    return merged


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _label(draw, xy: tuple[int, int], text: str, font, *, fill=GRID_TEXT) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=GRID_TEXT_BG)
    draw.text((x, y), text, font=font, fill=fill)


def grid_overlay(img):
    """The source image with pixel-labelled grid lines and A1-style cell ids."""
    from PIL import Image, ImageDraw

    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(max(12, base.width // 60))
    cell_font = _font(max(14, base.width // 45))

    step_x = _round_step(base.width)
    step_y = _round_step(base.height)

    for x in range(step_x, base.width, step_x):
        draw.line((x, 0, x, base.height), fill=GRID_COLOR, width=1)
        _label(draw, (x + 3, 2), str(x), font)
    for y in range(step_y, base.height, step_y):
        draw.line((0, y, base.width, y), fill=GRID_COLOR, width=1)
        _label(draw, (2, y + 2), str(y), font)

    # Cell ids: columns A.. left-to-right, rows 1.. top-to-bottom, centred in each cell.
    cols = (base.width + step_x - 1) // step_x
    rows = (base.height + step_y - 1) // step_y
    for row in range(rows):
        for col in range(cols):
            cx = col * step_x + step_x // 2
            cy = row * step_y + step_y // 2
            if cx >= base.width or cy >= base.height:
                continue
            _label(draw, (cx, cy), f"{chr(ord('A') + col)}{row + 1}", cell_font)

    return Image.alpha_composite(base, layer).convert("RGB")


def hits_overlay(img, detections: list[dict]):
    """Detector hits drawn over the (already gridded or plain) image.

    ``detections`` uses the imgutils bridge shape.  Hits are numbered per class in confidence
    order, matching how ``mask_spec.select_detections`` resolves an index.
    """
    from PIL import ImageDraw

    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(max(13, out.width // 55))
    width = max(2, out.width // 400)

    by_class: dict[str, list[dict]] = {}
    for det in detections:
        by_class.setdefault(str(det.get("label")), []).append(det)
    for label, hits in by_class.items():
        hits.sort(key=lambda d: -float(d.get("confidence", 0.0)))
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        for index, hit in enumerate(hits):
            bbox = hit.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
            conf = float(hit.get("confidence", 0.0))
            _label(draw, (x0 + 2, max(0, y0 - 16)), f"{label}#{index} ({conf:.2f})", font)
    return out


def mask_preview(img, mask, *, boxes: tuple[Box, ...] = (), keep_outs: tuple[Box, ...] = (), work_box: Box | None = None):
    """The confirmation picture: red fill where the mask repaints, outlines for the inputs.

    The red fill is derived from the actual compiled mask, thresholded the same way the
    graph will read it, so what the human approves is what the sampler gets.
    """
    from PIL import Image, ImageDraw

    base = img.convert("RGBA")
    if (mask.width, mask.height) != base.size:
        raise ValueError(
            f"mask is {mask.width}x{mask.height} but the image is {base.width}x{base.height}"
        )

    binary = mask.convert("L").point(lambda v: 255 if v >= WHITE_THRESHOLD else 0)
    fill = Image.new("RGBA", base.size, MASK_FILL)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(fill, (0, 0), binary)
    out = Image.alpha_composite(base, layer)

    draw = ImageDraw.Draw(out)
    width = max(2, base.width // 400)
    for box in boxes:
        draw.rectangle(
            (box.x, box.y, box.x + box.width - 1, box.y + box.height - 1),
            outline=BOX_COLOR,
            width=width,
        )
    for box in keep_outs:
        draw.rectangle(
            (box.x, box.y, box.x + box.width - 1, box.y + box.height - 1),
            outline=KEEP_OUT_COLOR,
            width=width,
        )
    if work_box is not None:
        draw.rectangle(
            (
                work_box.x,
                work_box.y,
                work_box.x + work_box.width - 1,
                work_box.y + work_box.height - 1,
            ),
            outline=WORK_BOX_COLOR,
            width=width,
        )
    return out.convert("RGB")
