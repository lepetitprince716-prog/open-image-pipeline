"""Pixel-level segmentation masks for the mask-spec, extracted on the ComfyUI box.

Boxes and the local imgutils detector both produce rectangles.  A rectangle around a face
repaints the hair, background and collar that share its bounding box; a segmentation mask
does not.  The Impact Pack chain that yields one already ships on Z3, so this module is
wiring, not new machinery::

    LoadImage -> UltralyticsDetectorProvider.SEGM_DETECTOR -> SegmDetectorSEGS
              [-> ImpactSEGSOrderedFilter]  -> SegsToCombinedMask -> MaskToImage -> SaveImage

Two deliberate choices:

``dilation=0``.  The node defaults to 10, and region_fix then adds its own GrowMask plus the
graph's grow-then-blur.  Stacking three growth passes on a mask whose entire value is being
pixel-tight defeats the point, so the tight mask is taken here and softening is left to the
one layer that already owns it.

``index`` means what it means for the local detector: confidence-descending, 0 = strongest,
``"all"`` for every segment.  One vocabulary across both mask sources, because two would be
mixed up by exactly the callers this contract exists to keep honest.

The extraction runs inference on the box's GPU.  It is small next to a sampler pass, but it
is not free, and it is why a segm spec makes ``region_fix --phase preview`` reach the network
where a box-only spec does not.
"""
from __future__ import annotations

from loramake.comfy import object_info_enum, submit_and_fetch, upload_image
from loramake.mask_spec import SegmSelect, resolve_segm_model
from loramake.region_mask import MaskError

DETECTOR_NODE = "UltralyticsDetectorProvider"
DETECTOR_FIELD = "model_name"

DEFAULT_THRESHOLD = 0.5
# See the module docstring: the tight mask is the product; growth belongs downstream.
DILATION = 0
CROP_FACTOR = 3.0
DROP_SIZE = 10


def known_models(host: str | None = None, *, refresh: bool = False) -> set[str]:
    return object_info_enum(DETECTOR_NODE, DETECTOR_FIELD, host=host, refresh=refresh)


def resolve_models(
    selections: tuple[SegmSelect, ...] | list[SegmSelect], host: str | None = None
) -> dict[SegmSelect, str]:
    """Map each selection to the exact enum name the box accepts.

    A miss is re-probed live before it is reported, mirroring ``comfy.preflight_assets``: the
    enum has a disk cache, and a cached answer must be able to delay a rejection by one
    round-trip, never to manufacture one.
    """
    known = known_models(host)
    resolved: dict[SegmSelect, str] = {}
    for sel in selections:
        try:
            resolved[sel] = resolve_segm_model(sel.model, known)
        except MaskError:
            if not known:
                raise
            resolved[sel] = resolve_segm_model(sel.model, known_models(host, refresh=True))
    return resolved


def build_graph(*, src_name: str, model_name: str, index: int | str = "all",
                threshold: float = DEFAULT_THRESHOLD,
                filename_prefix: str = "z3_segm_mask") -> dict:
    """The extraction graph for one selection.  Pure -- no network, no filesystem."""
    g: dict = {
        "10": {"class_type": "LoadImage", "inputs": {"image": src_name, "upload": "image"}},
        "11": {"class_type": DETECTOR_NODE, "inputs": {DETECTOR_FIELD: model_name}},
        "12": {"class_type": "SegmDetectorSEGS", "inputs": {
            "segm_detector": ["11", 1], "image": ["10", 0], "threshold": float(threshold),
            "dilation": DILATION, "crop_factor": CROP_FACTOR, "drop_size": DROP_SIZE,
            "labels": "all"}},
    }
    segs_ref = ["12", 0]
    if index != "all":
        g["13"] = {"class_type": "ImpactSEGSOrderedFilter", "inputs": {
            "segs": segs_ref, "target": "confidence", "order": True,
            "take_start": int(index), "take_count": 1}}
        segs_ref = ["13", 0]
    g["14"] = {"class_type": "SegsToCombinedMask", "inputs": {"segs": segs_ref}}
    g["15"] = {"class_type": "MaskToImage", "inputs": {"mask": ["14", 0]}}
    g["16"] = {"class_type": "SaveImage", "inputs": {
        "images": ["15", 0], "filename_prefix": filename_prefix}}
    return g


def fetch_masks(
    image: str,
    selections: tuple[SegmSelect, ...] | list[SegmSelect],
    *,
    host: str | None = None,
    timeout: float | None = None,
    resolved: dict[SegmSelect, str] | None = None,
):
    """Run each selection on the box and return ``{SegmSelect: PIL greyscale mask}``.

    Every failure is a :class:`MaskError`.  A segm selection that quietly resolves to nothing
    would compile to a mask that repaints nothing while the run reports success, so no path
    here degrades to an empty result.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    selections = list(selections)
    if not selections:
        return {}

    if resolved is None:
        resolved = resolve_models(selections, host)
    missing = [sel for sel in selections if sel not in resolved]
    if missing:
        raise MaskError(
            "unresolved segm selections: " + ", ".join(sel.label() for sel in missing))

    src_name = upload_image(image)
    if not src_name:
        raise MaskError(f"failed to upload {image} to the box for segm extraction")

    masks: dict[SegmSelect, Image.Image] = {}
    for sel in selections:
        graph = build_graph(
            src_name=src_name,
            model_name=resolved[sel],
            index=sel.index,
            threshold=DEFAULT_THRESHOLD if sel.threshold is None else sel.threshold,
        )
        result = submit_and_fetch(
            graph, label=f"segm:{sel.label()}", host=host, timeout=timeout)
        if not result:
            raise MaskError(
                f"segm extraction failed on the box for {sel.label()} "
                f"(model {resolved[sel]}); see the node error above"
            )
        try:
            with Image.open(BytesIO(result[0][0])) as im:
                masks[sel] = im.convert("L")
        except (UnidentifiedImageError, OSError) as exc:
            raise MaskError(
                f"segm extraction returned an unreadable image for {sel.label()}: {exc}"
            ) from exc
    return masks
