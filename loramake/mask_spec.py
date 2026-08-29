"""The mask-spec contract: a compositional selection an LLM can emit and a compiler for it.

The LLM never paints pixels.  It reads a coordinate-grid overlay plus detector hits and emits
a small JSON program — boxes to include, detector hits to include, boxes to subtract — and the
compiler here rasterises that program into the binary mask the inpaint graph consumes.  This
is a deliberate policy, not a shortcut: docs/runbooks/foot_fix_runbook.md settled that the
LLM-callable selection form is boxes on a grid, not free-hand polygons, because box
coordinates survive a glance-check and thirty polygon vertices do not.

A third selection source is pixel-level segmentation (``segm``): the ComfyUI box runs an
Ultralytics segmentation model and hands back a raster mask, which the caller injects here.
That one costs a round-trip to the box; boxes and detector hits stay purely local.

The compiled mask is a hard 0/255 binary at the source image's exact pixel size.  Softening
(grow-then-blur) stays in the graph layer, where scripts/_repair_graph.py already does it per
crop; keeping the compiler binary keeps its golden tests byte-deterministic.

Keep-outs are burned into the compiled mask here.  Callers must NOT also pass them to the
graph builders, or the subtraction happens twice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote, unquote

from loramake.region_mask import (
    WHITE_THRESHOLD,
    WHOLE_IMAGE_COVERAGE,
    Box,
    MaskError,
    coverage,
    validate_size,
    work_box,
)

SCHEMA = "loramake.mask-spec/v1"

MODES = ("refine", "replace", "polish")

# Detector classes the local imgutils bridge can produce (scripts/imgutils_bridge.py detect).
DETECT_CLASSES = ("hands", "eyes", "faces", "heads", "persons", "text", "feet")

# Friendly keys for the segm models this repo has live evidence for (Z3 survey, 2026-08-27:
# the SEGS chain was driven end to end against each one and returned a pixel mask).  Anything
# else may still be passed as a raw server enum name -- it is simply not vouched for here.
SEGM_MODELS = {
    "anzhc_face": "segm/Anzhc%20Face%20seg%20640%20v4%20y11n.pt",
    "anzhc_eyes": "segm/Anzhc%20Eyes%20-seg-hd.pt",
}

# Shelved under segm/ but trained bbox-only: UltralyticsDetectorProvider leaves SEGM_DETECTOR
# empty for them, so SegmDetectorSEGS dies on `NoneType.data` mid-graph.  Refuse them here with
# the reason instead of letting a caller burn a Z3 round-trip on a node crash.
BBOX_ONLY_SEGM = {
    "segm/foot_yolov8x.pt",
    "segm/foot_yolov8x_v2.pt",
}

_SPEC_KEYS = frozenset(
    {"schema", "source", "boxes", "detect", "segm", "keep_outs", "mode", "grow", "feather"}
)


@dataclass(frozen=True)
class DetectSelect:
    """One detector selection: a class plus which hit(s) to take.

    ``index`` is an int for one hit (ordered by detector confidence, 0 = strongest) or the
    string ``"all"`` for every hit of that class.
    """

    cls: str
    index: int | str = "all"


@dataclass(frozen=True)
class SegmSelect:
    """One pixel-level segmentation selection, resolved on the ComfyUI box, not locally.

    ``model`` is a :data:`SEGM_MODELS` key or a raw ``UltralyticsDetectorProvider`` enum name.
    ``index`` matches :class:`DetectSelect` exactly -- an int picks one segment ordered by
    confidence (0 = strongest), ``"all"`` takes every segment above ``threshold``.
    """

    model: str
    index: int | str = "all"
    threshold: float | None = None

    def label(self) -> str:
        return f"{self.model}[{self.index}]"


@dataclass(frozen=True)
class Spec:
    source: str
    boxes: tuple[Box, ...] = ()
    detect: tuple[DetectSelect, ...] = ()
    segm: tuple[SegmSelect, ...] = ()
    keep_outs: tuple[Box, ...] = ()
    mode: str = "refine"
    grow: int | None = None
    feather: int | None = None

    def wants_detection(self) -> bool:
        return bool(self.detect)

    def wants_segm(self) -> bool:
        return bool(self.segm)


def _as_box(value: object, *, label: str) -> Box:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise MaskError(f"{label} must be [x, y, w, h] with integer pixels, got {value!r}")
    x, y, w, h = value
    if w <= 0 or h <= 0:
        raise MaskError(f"{label} has non-positive size: {value!r}")
    if x < 0 or y < 0:
        raise MaskError(f"{label} has a negative origin: {value!r}")
    return Box(x, y, w, h)


def _as_detect(value: object) -> DetectSelect:
    if isinstance(value, str):
        value = {"class": value}
    if not isinstance(value, dict):
        raise MaskError(f"detect entries must be a class name or an object, got {value!r}")
    unknown = set(value) - {"class", "index"}
    if unknown:
        raise MaskError(f"detect entry has unknown keys: {sorted(unknown)}")
    cls = value.get("class")
    if cls not in DETECT_CLASSES:
        raise MaskError(
            f"unknown detector class {cls!r}; choose from {', '.join(DETECT_CLASSES)}"
        )
    return DetectSelect(cls=cls, index=_as_index(value.get("index", "all"), label="detect"))


def _as_index(index: object, *, label: str) -> int | str:
    is_int = isinstance(index, int) and not isinstance(index, bool)
    if not (index == "all" or (is_int and index >= 0)):
        raise MaskError(f"{label} index must be a non-negative int or \"all\", got {index!r}")
    return index


def _as_segm(value: object) -> SegmSelect:
    if isinstance(value, str):
        value = {"model": value}
    if not isinstance(value, dict):
        raise MaskError(f"segm entries must be a model name or an object, got {value!r}")
    unknown = set(value) - {"model", "index", "threshold"}
    if unknown:
        raise MaskError(f"segm entry has unknown keys: {sorted(unknown)}")
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise MaskError(
            f"segm entry needs a non-empty \"model\"; known keys are "
            f"{', '.join(sorted(SEGM_MODELS))}, or pass a raw server enum name"
        )
    threshold = value.get("threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise MaskError(f"segm threshold must be a number in 0..1, got {threshold!r}")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise MaskError(f"segm threshold must be in 0..1, got {threshold}")
    return SegmSelect(
        model=model.strip(),
        index=_as_index(value.get("index", "all"), label="segm"),
        threshold=threshold,
    )


def _enum_variants(name: str) -> tuple[str, ...]:
    """The spellings of one model name a ComfyUI enum might use.

    The server enumerates these files percent-encoded (``segm/Anzhc%20Face...``) while every
    human-written note spells them with spaces, and passing the space form is rejected by the
    node's combo validation.  Try both so a caller can write either.
    """
    seen: list[str] = []
    for candidate in (name, quote(name, safe="/-_.%"), unquote(name)):
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def resolve_segm_model(model: str, known: set[str] | None = None) -> str:
    """Friendly key or raw name -> the exact enum value the server accepts.

    ``known`` is the live ``UltralyticsDetectorProvider.model_name`` enum; pass ``None`` to
    skip membership checking (the caller then owns the miss).  An empty set means the probe
    failed, and is treated the same as ``None`` -- the repo-wide convention in
    ``comfy.missing_assets``: a failed probe must not become a fabricated rejection.
    """
    resolved = SEGM_MODELS.get(model.strip().lower(), model.strip())
    if unquote(resolved) in {unquote(n) for n in BBOX_ONLY_SEGM}:
        raise MaskError(
            f"{resolved} is a bbox-only model shelved under segm/: it exposes no "
            "SEGM_DETECTOR and SegmDetectorSEGS crashes on it. Use a box selection or "
            "\"detect\" for this target."
        )
    if not known:
        return resolved
    for variant in _enum_variants(resolved):
        if variant in known:
            return variant
    segm_names = sorted(n for n in known if n.startswith("segm/"))
    raise MaskError(
        f"segm model {model!r} is not installed on the box (tried {', '.join(_enum_variants(resolved))}); "
        f"available segm entries: {', '.join(segm_names) or 'none'}"
    )


def parse_spec(raw: str | dict) -> Spec:
    """Parse and validate a mask spec.  Unknown keys are refused, not ignored.

    A typo'd key silently ignored means the LLM believes it excluded a region it did not, so
    the refusal message names the bad key for self-correction.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MaskError(f"mask spec is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MaskError(f"mask spec must be a JSON object, got {type(raw).__name__}")

    unknown = set(raw) - _SPEC_KEYS
    if unknown:
        raise MaskError(
            f"mask spec has unknown keys: {sorted(unknown)}; known keys are {sorted(_SPEC_KEYS)}"
        )
    schema = raw.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise MaskError(f"unsupported mask spec schema {schema!r}; expected {SCHEMA}")

    source = raw.get("source", "")
    if not isinstance(source, str) or not source.strip():
        raise MaskError("mask spec needs a non-empty \"source\" image path")

    mode = raw.get("mode", "refine")
    if mode not in MODES:
        raise MaskError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")

    boxes = tuple(
        _as_box(b, label=f"boxes[{i}]") for i, b in enumerate(raw.get("boxes") or [])
    )
    keep_outs = tuple(
        _as_box(b, label=f"keep_outs[{i}]") for i, b in enumerate(raw.get("keep_outs") or [])
    )
    detect = tuple(_as_detect(d) for d in raw.get("detect") or [])
    segm = tuple(_as_segm(v) for v in raw.get("segm") or [])

    if not boxes and not detect and not segm:
        raise MaskError(
            "mask spec selects nothing: give at least one of \"boxes\", \"detect\" or \"segm\""
        )

    def _radius(key: str) -> int | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MaskError(f"{key} must be a non-negative integer or null, got {value!r}")
        return value

    return Spec(
        source=source.strip(),
        boxes=boxes,
        detect=detect,
        segm=segm,
        keep_outs=keep_outs,
        mode=mode,
        grow=_radius("grow"),
        feather=_radius("feather"),
    )


def select_detections(spec: Spec, detections: list[dict]) -> list[Box]:
    """Resolve the spec's detector selections against actual detector output.

    ``detections`` uses the imgutils bridge shape: ``{"bbox": [x0, y0, x1, y1], "label": ...,
    "confidence": ...}``.  A selection that matches nothing is a hard error — repainting a
    region the detector never saw is the incident-#49 failure mode, so the miss is surfaced
    instead of silently shrinking the mask.
    """
    boxes: list[Box] = []
    for sel in spec.detect:
        hits = [d for d in detections if d.get("label") == sel.cls]
        hits.sort(key=lambda d: -float(d.get("confidence", 0.0)))
        if not hits:
            raise MaskError(
                f"detector found no {sel.cls!r} in the image; adjust the spec or use boxes"
            )
        if sel.index == "all":
            chosen = hits
        else:
            if sel.index >= len(hits):
                raise MaskError(
                    f"detect asks for {sel.cls}[{sel.index}] but only "
                    f"{len(hits)} hit(s) exist"
                )
            chosen = [hits[sel.index]]
        for hit in chosen:
            bbox = hit.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise MaskError(f"malformed detection bbox: {bbox!r}")
            x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
            if x1 <= x0 or y1 <= y0:
                raise MaskError(f"degenerate detection bbox: {bbox!r}")
            boxes.append(Box(x0, y0, x1 - x0, y1 - y0))
    return boxes


def _clip(box: Box, size: tuple[int, int]) -> Box | None:
    width, height = size
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(width, box.x + box.width)
    y1 = min(height, box.y + box.height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1 - x0, y1 - y0)


def merge_segm(mask, spec: Spec, source_size: tuple[int, int], segm_masks: dict | None):
    """Union the fetched segmentation masks into ``mask`` and return the result.

    ``segm_masks`` maps each :class:`SegmSelect` to the greyscale mask the ComfyUI SEGS chain
    produced for it (see :mod:`loramake.region_segm`).  Injecting the fetched raster instead
    of fetching here is what keeps this compiler offline and its golden tests deterministic.

    Every failure is hard.  An all-black mask means the detector matched nothing, or the
    ``index`` ran past the last segment; either way the run would repaint nothing while
    reporting success, which is the incident-#49 shape this line is built to refuse.
    """
    from PIL import ImageChops

    for sel in spec.segm:
        fetched = (segm_masks or {}).get(sel)
        if fetched is None:
            raise MaskError(
                f"segm selection {sel.label()} has no fetched mask; the caller must resolve "
                "segm selections on the box before compiling"
            )
        validate_size(fetched.size, source_size)
        binary = fetched.convert("L").point(lambda v: 255 if v >= WHITE_THRESHOLD else 0)
        if binary.getbbox() is None:
            raise MaskError(
                f"segm model {sel.label()} selected no pixels: either it found no segment "
                f"above its threshold, or index {sel.index!r} is past the last segment. "
                "Check the overlay, lower the threshold, or use \"all\"."
            )
        mask = ImageChops.lighter(mask, binary)
    return mask


def compile_mask(spec: Spec, source_size: tuple[int, int], detections: list[dict] | None = None,
                 segm_masks: dict | None = None):
    """Rasterise the spec into a hard-binary mask PNG image (white = repaint).

    Union of the spec's boxes, its resolved detector boxes and its fetched segmentation
    masks, minus its keep-outs, at the source image's exact pixel size.  Refuses an empty
    result and a near-whole-frame result; the latter means the caller wanted a whole-image
    edit and the crop path would only add seams and cost.

    Segmentation masks arrive soft-edged from ``MaskToImage`` and are binarised at
    ``WHITE_THRESHOLD`` on the way in: this compiler's output stays hard 0/255, and softening
    stays in the graph layer that already owns grow-then-blur.
    """
    from PIL import Image, ImageDraw

    include = list(spec.boxes)
    if spec.detect:
        include.extend(select_detections(spec, detections or []))

    mask = Image.new("L", source_size, 0)
    draw = ImageDraw.Draw(mask)
    kept_any = False
    for box in include:
        clipped = _clip(box, source_size)
        if clipped is None:
            continue
        kept_any = True
        draw.rectangle(
            (clipped.x, clipped.y, clipped.x + clipped.width - 1, clipped.y + clipped.height - 1),
            fill=255,
        )
    if include and not kept_any:
        raise MaskError("every selected box falls outside the image; check the coordinates")

    # Segmentation joins the union before the keep-out subtraction, so a keep-out clips a
    # segmented region exactly as it clips a box, and before the coverage guards below.
    mask = merge_segm(mask, spec, source_size, segm_masks)
    draw = ImageDraw.Draw(mask)

    for box in spec.keep_outs:
        clipped = _clip(box, source_size)
        if clipped is None:
            continue
        draw.rectangle(
            (clipped.x, clipped.y, clipped.x + clipped.width - 1, clipped.y + clipped.height - 1),
            fill=0,
        )

    validate_size((mask.width, mask.height), source_size)
    share = coverage(mask)
    if share <= 0.0:
        raise MaskError("keep-outs erased the whole selection; nothing left to repaint")
    if share >= WHOLE_IMAGE_COVERAGE:
        raise MaskError(
            f"selection covers {share:.0%} of the frame; that is a whole-image edit — "
            "use the promote/img2img path instead of a region repaint"
        )
    return mask


def derive_bbox(mask, *, padding: int = 32) -> Box:
    """The aligned work box the sampler should crop to, from a compiled mask."""
    return work_box(mask, padding=padding)


def canonical_json(spec: Spec) -> str:
    """The spec as stable JSON, for sidecars and idempotent re-submission."""
    payload = {
        "schema": SCHEMA,
        "source": spec.source,
        "boxes": [list(b.as_tuple()) for b in spec.boxes],
        "detect": [
            {"class": d.cls, "index": d.index} for d in spec.detect
        ],
        "segm": [
            {"model": v.model, "index": v.index, "threshold": v.threshold} for v in spec.segm
        ],
        "keep_outs": [list(b.as_tuple()) for b in spec.keep_outs],
        "mode": spec.mode,
        "grow": spec.grow,
        "feather": spec.feather,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
