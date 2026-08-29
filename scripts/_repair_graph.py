#!/usr/bin/env python3
"""Shared ComfyUI graph builders for the defect-repair toolkit (scripts/defect_fix.py).

Two mutate-and-return-a-ref pass builders, generalized from the proven detect/inpaint patterns already
in the repo:

  detailer_pass    the digit_fix._limb_detailer / z3_quality_gen._facedetailer FaceDetailer pattern
                    (UltralyticsDetectorProvider + optional SAMLoader -> FaceDetailer), parameterized.
  bbox_inpaint_pass the foot_inpaint.foot_chain pattern (crop -> upscale -> optional self-canny ->
                    ControlNet -> VAEEncode -> fractional bottom mask + feather -> SetLatentNoiseMask ->
                    KSampler -> VAEDecode -> downscale -> feathered composite back), generalized to any
                    bbox target. lock="canny" polishes geometry; lock="none" is the extra-toe recount path.

Both functions mutate the passed graph dict `g` in place and return the ["node_id", slot] ref for the
resulting image, so a caller can chain several passes onto the same base image (mirroring how digit_fix
chains hands+feet, and foot_inpaint chains multiple foot bboxes). Node ids are namespaced by the graph's
current size at call time (`len(g)`), which is always monotonically increasing within one graph dict —
safe for repeated calls in the same graph without an explicit tag argument.

Deliberately dependency-free: stdlib only, no imports from defect_fix.py or any driver module, so this
file is unit-testable by building a plain graph dict and asserting node structure / class_types.
"""
from __future__ import annotations

import math

# SAM model used for detailer_pass's optional segmentation refinement. Confirmed present on Z3
# (SAMLoader.model_name enum); the smaller ViT-B checkpoint is fast enough for a per-region mask.
SAM_MODEL = "sam_vit_b_01ec64.pth"


def detailer_pass(g: dict, image_ref, model_ref, clip_ref, vae_ref, pos_ref, neg_ref, *,
                  detector: str, guide_size: float, denoise: float, steps: int, cfg: float,
                  sampler: str, scheduler: str, seed: int, bbox_dilation: int = 20,
                  bbox_crop_factor: float = 3.0, feather: int = 8, bbox_threshold: float = 0.50,
                  sam: bool = True) -> list:
    """UltralyticsDetectorProvider(detector) [+ SAMLoader] -> FaceDetailer. Returns the output image ref.

    Mirrors digit_fix._limb_detailer's dilation/crop-factor defaults (tuned so toe/finger tips and small
    bbox targets like eyes stay inside the detected+dilated mask) and z3_quality_gen._facedetailer's
    sam_detection_hint/threshold defaults. `detector` accepts either a `bbox/*.pt` or `segm/*.pt`
    UltralyticsDetectorProvider model — both expose a BBOX_DETECTOR output on slot 0, which is what
    FaceDetailer.bbox_detector consumes (segm files still degrade correctly on this input, as in
    digit_fix). `sam=True` wires an optional SAMLoader for a tighter per-pixel mask than the bbox
    detector's rectangle alone would give; segm detectors already yield the tighter mask on their own, so
    disabling SAM for segm-backed axes is a reasonable caller choice, not a hard requirement here."""
    p = f"dp{len(g)}"
    g[f"{p}_det"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": detector}}
    det_ref = [f"{p}_det", 0]
    sam_ref = None
    if sam:
        g[f"{p}_sam"] = {"class_type": "SAMLoader", "inputs": {"model_name": SAM_MODEL, "device_mode": "AUTO"}}
        sam_ref = [f"{p}_sam", 0]
    g[f"{p}_fd"] = {"class_type": "FaceDetailer", "inputs": {
        "image": image_ref, "model": model_ref, "clip": clip_ref, "vae": vae_ref,
        "positive": pos_ref, "negative": neg_ref, "guide_size": guide_size, "guide_size_for": True,
        "max_size": 1024.0, "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise, "feather": feather,
        "noise_mask": True, "force_inpaint": True, "bbox_threshold": bbox_threshold,
        "bbox_dilation": bbox_dilation, "bbox_crop_factor": bbox_crop_factor,
        "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
        "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False",
        "drop_size": 10, "bbox_detector": det_ref, "sam_model_opt": sam_ref, "wildcard": "", "cycle": 1}}
    return [f"{p}_fd", 0]


def bbox_inpaint_pass(g: dict, image_ref, model_ref, clip_ref, vae_ref, pos_ref, neg_ref, *,
                      bbox: tuple[int, int, int, int], guide_size: float, denoise: float, steps: int,
                      cfg: float, sampler: str, scheduler: str, seed: int, cn_strength: float = 0.55,
                      cn_end: float = 0.8, controlnet_name: str | None = None, mask_frac: float = 1.0,
                      feather_px: int = 28, feather_even: bool = False, lock: str = "canny") -> list:
    """Crop bbox -> upscale -> optional self-canny lock -> masked inpaint -> feathered composite back.

    Generalizes foot_inpaint.foot_chain to an arbitrary rectangular target (mouth, a single foot, etc.)
    instead of being foot-specific; the node-by-node structure is otherwise identical. `mask_frac` is the
    fraction of the crop's height, measured from the bottom, that gets denoised (1.0 = the whole crop is
    denoised — used for mouth, where the defect is not anchored to one edge; <1.0 restricts denoise to a
    bottom band — used for a foot's toe-opening). Everything outside `bbox` is untouched by construction:
    the final composite is masked and feathered only over that region.

    `lock="canny"` (default) is the polish path: self-canny ControlNet holds existing geometry, which
    also holds a wrong toe *count*. `lock="none"` skips Canny/ControlNet so a high-denoise pass can
    change topology (extra/missing toes). `controlnet_name` is required only when lock is canny.
    """
    if lock not in ("canny", "none"):
        raise ValueError(f"lock must be 'canny' or 'none', got {lock!r}")
    if lock == "canny" and not controlnet_name:
        raise ValueError("controlnet_name is required when lock='canny'")
    x, y, w, h, scale, uw, uh = _crop_scale(bbox, guide_size)
    p = f"bi{len(g)}"
    g[f"{p}_crop"] = {"class_type": "ImageCrop", "inputs": {"image": image_ref, "width": w, "height": h, "x": x, "y": y}}
    g[f"{p}_up"] = {"class_type": "ImageScaleBy", "inputs": {"image": [f"{p}_crop", 0], "upscale_method": "lanczos", "scale_by": scale}}
    up = [f"{p}_up", 0]
    pos_use, neg_use = pos_ref, neg_ref
    if lock == "canny":
        g[f"{p}_canny"] = {"class_type": "Canny", "inputs": {"image": up, "low_threshold": 0.15, "high_threshold": 0.45}}
        g[f"{p}_cnl"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet_name}}
        g[f"{p}_cn"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos_ref, "negative": neg_ref, "control_net": [f"{p}_cnl", 0], "image": [f"{p}_canny", 0],
            "strength": cn_strength, "start_percent": 0.0, "end_percent": cn_end}}
        pos_use, neg_use = [f"{p}_cn", 0], [f"{p}_cn", 1]
    g[f"{p}_enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": up, "vae": vae_ref}}
    lat = [f"{p}_enc", 0]
    if mask_frac < 1.0:
        mh = int(round(uh * mask_frac))
        g[f"{p}_mb"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": uw, "height": uh}}
        g[f"{p}_mw"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": uw, "height": mh}}
        g[f"{p}_mc"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [f"{p}_mb", 0], "source": [f"{p}_mw", 0], "x": 0, "y": uh - mh, "operation": "add"}}
        g[f"{p}_mf"] = {"class_type": "FeatherMask", "inputs": {
            "mask": [f"{p}_mc", 0], "left": 48, "top": 48, "right": 48, "bottom": 16}}
        g[f"{p}_lm"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": lat, "mask": [f"{p}_mf", 0]}}
        lat = [f"{p}_lm", 0]
    g[f"{p}_ks"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
        "denoise": denoise, "model": model_ref, "positive": pos_use, "negative": neg_use,
        "latent_image": lat}}
    g[f"{p}_dec"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{p}_ks", 0], "vae": vae_ref}}
    g[f"{p}_dn"] = {"class_type": "ImageScale", "inputs": {
        "image": [f"{p}_dec", 0], "upscale_method": "area", "width": w, "height": h, "crop": "disabled"}}
    g[f"{p}_bm"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": w, "height": h}}
    g[f"{p}_bf"] = {"class_type": "FeatherMask", "inputs": {
        "mask": [f"{p}_bm", 0], "left": feather_px, "top": feather_px,
        "right": feather_px, "bottom": feather_px if feather_even else feather_px // 2}}
    g[f"{p}_comp"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": image_ref, "source": [f"{p}_dn", 0], "x": x, "y": y, "resize_source": False, "mask": [f"{p}_bf", 0]}}
    return [f"{p}_comp", 0]


def _crop_scale(bbox: tuple[int, int, int, int], guide_size: float) -> tuple[int, int, int, int, float, int, int]:
    x, y, w, h = bbox
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"bbox requires x,y>=0 and w,h>0, got {bbox!r}")
    raw_scale = max(guide_size / max(w, h), 8.0 / min(w, h), 0.01)
    scale = min(8.0, math.ceil(raw_scale * 1000.0) / 1000.0)
    uw, uh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    return x, y, w, h, scale, uw, uh


def clip_rect(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Intersection of two x,y,w,h rects, or None if they do not overlap."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x1, y1 = max(ix, ox), max(iy, oy)
    x2, y2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _subtract_keep_outs(
    g: dict, p: str, mask_ref: list, bbox: tuple[int, int, int, int],
    keep_outs: list[tuple[int, int, int, int]] | None,
) -> list:
    """Punch keep-out rects (source pixels) out of a crop-resolution MASK."""
    x, y, w, h = bbox
    out = mask_ref
    for i, ko in enumerate(keep_outs or ()):
        clipped = clip_rect(ko, (x, y, w, h))
        if clipped is None:
            continue
        cx, cy, cw, ch = clipped
        g[f"{p}_ko{i}"] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": cw, "height": ch}}
        g[f"{p}_ks{i}"] = {"class_type": "MaskComposite", "inputs": {
            "destination": out, "source": [f"{p}_ko{i}", 0],
            "x": cx - x, "y": cy - y, "operation": "subtract"}}
        out = [f"{p}_ks{i}", 0]
    return out


def dd_inpaint_pass(
    g: dict, image_ref, model_ref, vae_ref, pos_ref, neg_ref, *,
    bbox: tuple[int, int, int, int], guide_size: float, denoise: float, steps: int,
    cfg: float, sampler: str, scheduler: str, seed: int, mask_frac: float = 1.0,
    feather_px: int = 28, feather_even: bool = False, grow: int = 12, blur: int = 20,
    mask_full_ref: list | None = None,
    keep_outs: list[tuple[int, int, int, int]] | None = None,
) -> list:
    """Crop bbox -> upscale -> InpaintModelConditioning + DifferentialDiffusion -> composite.

    Replaces the VAEEncode + SetLatentNoiseMask + optional Canny chain. The mask is Grow+Blur so
    DifferentialDiffusion sees an edge gradient (hard 0/1 masks collapse DD to a binary noise mask).
    `model_ref` must already be wrapped in DifferentialDiffusion by the caller.

    `mask_full_ref` is an optional MASK at full source resolution (pre-made lasso). It is cropped
    to `bbox` and scaled with the pixels.

    `keep_outs` are axis-aligned rects in source pixels punched out of the crop mask (thigh
    keep-out). Pixels inside the bbox but outside the remaining mask stay the destination original.

    Irregular masks composite with MaskBlur+ on the crop-resolution mask, not FeatherMask on the
    bbox rectangle — FeatherMask only fades the tensor's four sides.
    """
    x, y, w, h, scale, uw, uh = _crop_scale(bbox, guide_size)
    p = f"dd{len(g)}"
    g[f"{p}_crop"] = {"class_type": "ImageCrop", "inputs": {
        "image": image_ref, "width": w, "height": h, "x": x, "y": y}}
    g[f"{p}_up"] = {"class_type": "ImageScaleBy", "inputs": {
        "image": [f"{p}_crop", 0], "upscale_method": "lanczos", "scale_by": scale}}
    up = [f"{p}_up", 0]
    irregular = mask_full_ref is not None or bool(keep_outs)
    crop_mask_ref = None
    if irregular:
        if mask_full_ref is not None:
            g[f"{p}_mcrop"] = {"class_type": "CropMask", "inputs": {
                "mask": mask_full_ref, "x": x, "y": y, "width": w, "height": h}}
            crop_mask_ref = [f"{p}_mcrop", 0]
        else:
            g[f"{p}_mw"] = {"class_type": "SolidMask", "inputs": {
                "value": 1.0, "width": w, "height": h}}
            crop_mask_ref = [f"{p}_mw", 0]
        crop_mask_ref = _subtract_keep_outs(g, p, crop_mask_ref, (x, y, w, h), keep_outs)
        g[f"{p}_m2i"] = {"class_type": "MaskToImage", "inputs": {"mask": crop_mask_ref}}
        g[f"{p}_mup"] = {"class_type": "ImageScaleBy", "inputs": {
            "image": [f"{p}_m2i", 0], "upscale_method": "lanczos", "scale_by": scale}}
        g[f"{p}_i2m"] = {"class_type": "ImageToMask", "inputs": {
            "image": [f"{p}_mup", 0], "channel": "red"}}
        mask_ref = [f"{p}_i2m", 0]
    elif mask_frac < 1.0:
        mh = max(8, int(round(uh * mask_frac)))
        g[f"{p}_mb"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": uw, "height": uh}}
        g[f"{p}_mw"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": uw, "height": mh}}
        g[f"{p}_mc"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [f"{p}_mb", 0], "source": [f"{p}_mw", 0], "x": 0, "y": uh - mh, "operation": "add"}}
        mask_ref = [f"{p}_mc", 0]
    else:
        g[f"{p}_mw"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": uw, "height": uh}}
        mask_ref = [f"{p}_mw", 0]
    g[f"{p}_gr"] = {"class_type": "GrowMask", "inputs": {
        "mask": mask_ref, "expand": grow, "tapered_corners": True}}
    g[f"{p}_bl"] = {"class_type": "MaskBlur+", "inputs": {
        "mask": [f"{p}_gr", 0], "amount": blur, "device": "auto"}}
    g[f"{p}_imc"] = {"class_type": "InpaintModelConditioning", "inputs": {
        "positive": pos_ref, "negative": neg_ref, "vae": vae_ref,
        "pixels": up, "mask": [f"{p}_bl", 0], "noise_mask": True}}
    g[f"{p}_ks"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
        "denoise": denoise, "model": model_ref,
        "positive": [f"{p}_imc", 0], "negative": [f"{p}_imc", 1],
        "latent_image": [f"{p}_imc", 2]}}
    g[f"{p}_dec"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{p}_ks", 0], "vae": vae_ref}}
    g[f"{p}_dn"] = {"class_type": "ImageScale", "inputs": {
        "image": [f"{p}_dec", 0], "upscale_method": "area", "width": w, "height": h, "crop": "disabled"}}
    if crop_mask_ref is not None:
        g[f"{p}_cbl"] = {"class_type": "MaskBlur+", "inputs": {
            "mask": crop_mask_ref, "amount": max(4, feather_px // 2), "device": "auto"}}
        comp_mask = [f"{p}_cbl", 0]
        g[f"{p}_comp"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": image_ref, "source": [f"{p}_dn", 0], "x": x, "y": y,
            "resize_source": False, "mask": comp_mask}}
        return [f"{p}_comp", 0]
    g[f"{p}_bm"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": w, "height": h}}
    feat_bottom = feather_px if feather_even else feather_px // 2
    g[f"{p}_bf"] = {"class_type": "FeatherMask", "inputs": {
        "mask": [f"{p}_bm", 0], "left": feather_px, "top": feather_px,
        "right": feather_px, "bottom": feat_bottom}}
    g[f"{p}_comp"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": image_ref, "source": [f"{p}_dn", 0], "x": x, "y": y,
        "resize_source": False, "mask": [f"{p}_bf", 0]}}
    return [f"{p}_comp", 0]
