#!/usr/bin/env python3
"""Dedicated foot/toe correction pipeline.

Protocol: docs/t2i_prompt_validate_anatomy.md

Extra/missing toes are a topology problem. digit_fix / defect_fix / foot_inpaint default to
denoise 0.35–0.50 plus a self-canny lock, which paints a wrong count more clearly. This line
crops each foot large, skips canny on recount/closeup, uses 5toes tags, optional foot LoRA,
and reseeds the region only.

Modes:
  recount  extra/missing toes. No canny. denoise 0.65. n=4. bottom-band mask 0.45
  polish   count already correct, just soft. Canny lock. denoise 0.35
  closeup  foot is the subject. Toe-row mask 0.28. No canny. denoise 0.72

--bbox is the protocol path (one rectangle per foot). --keep-out punches thigh (etc.)
out of that rectangle so those pixels are not denoised and not pasted back. --mask is a
same-size PNG if a lasso already exists. --detect falls back to foot YOLO FaceDetailer
when bboxes are unknown. Do not pass a hand detector.

Output -> outputs/<character>/by_model/<DRIVING_MODEL>/foot_fix/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """The checkout that owns ``loramake/``.

    That is this script's parent in the SFW tree and one level further up in the NSFW
    mirror (``nsfw/scripts`` -> ``nsfw`` -> repo root), so walk instead of assuming.
    """
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate the repo root containing loramake/__init__.py")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

from loramake.comfy_endpoints import DIRECT_COMFY_HOSTS, resolve_comfy_host  # noqa: E402
from loramake.model_resources import anima_family  # noqa: E402

HOSTS = dict(DIRECT_COMFY_HOSTS)
FOOT_DETECTOR = "segm/foot_yolov8x_v2.pt"
ANATOMY_POS = "5toes, toes, toenails"
ANATOMY_NEG = "extra toes, 6toes, 4toes, fused toes, extra feet, extra legs, mutated"
CLOSEUP_POS = "foot focus"
# Civitai 1496218 Toes Adjuster: Pony weights are on disk. Illustrious weights are not.
FOOT_LORA = {
    "pony": "Pony\\concept\\toes_adjuster_pony_v1.0.safetensors",
}
FAMILY_ALIASES = {
    "illustrious": "illustrious",
    "sdxl": "illustrious",
    "noobai": "illustrious",
    "pony": "illustrious",
    "anima": "anima",
    "flow": "anima",
}
MODE_PRESETS = {
    "recount": {
        "denoise": 0.65, "n": 4, "guide_size": 1024.0, "mask_frac": 0.45,
        "lock": "none", "lora_strength": 0.85, "steps": 28, "pad": True,
    },
    "polish": {
        "denoise": 0.35, "n": 2, "guide_size": 1024.0, "mask_frac": 0.55,
        "lock": "canny", "lora_strength": 0.70, "steps": 24, "pad": False,
    },
    "closeup": {
        "denoise": 0.72, "n": 4, "guide_size": 1024.0, "mask_frac": 0.28,
        "lock": "none", "lora_strength": 1.00, "steps": 28, "pad": True,
    },
}


def _pre_parse_box() -> tuple[str, bool]:
    for i, tok in enumerate(sys.argv[1:], 1):
        if tok == "--box" and i < len(sys.argv) - 1:
            return sys.argv[i + 1], True
        if tok.startswith("--box="):
            return tok.split("=", 1)[1], True
    return "boxa", False


def _selected_endpoint(box: str) -> str:
    return resolve_comfy_host(box, explicit=os.environ.get("BOXA_HOST"))


if __name__ == "__main__" or "foot_fix" in Path(sys.argv[0]).stem:
    _BOX, _BOX_EXPLICIT = _pre_parse_box()
    if _BOX in HOSTS:
        os.environ["BOXA_HOST"] = _selected_endpoint(_BOX)

sys.path.insert(0, str(SCRIPT_DIR))
from config import OUTPUT_CHARACTER, by_model_dir  # noqa: E402
from boxa_comfy import (  # noqa: E402
    BOXA_HOST, png_size, upload_image,
    looks_like_png as _looks_like_png, submit_and_fetch as _shared_submit_and_fetch,
    atomic_write_text as _atomic_write_text, preflight_assets,
    object_info_enum as _object_info_enum, missing_assets as _missing_assets,
    free_vram as _free_vram,
)
import anima_models  # noqa: E402
from boxa_quality_gen import RECIPES, CONTROLNET, resolve_lora_file  # noqa: E402
from _repair_graph import bbox_inpaint_pass, dd_inpaint_pass, detailer_pass  # noqa: E402

MODEL = os.environ.get("DRIVING_MODEL", "claude").strip().lower() or "claude"
# The public tree follows the configured, character-neutral output root.
OUT = by_model_dir(OUTPUT_CHARACTER, "foot_fix", model=MODEL)
CN_CANNY = CONTROLNET["canny"]["model"]


def parse_bbox(spec: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in spec.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--bbox must be 'x,y,w,h' (ints); got {spec!r}") from None
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--bbox requires x,y >= 0 and w,h > 0")
    return x, y, w, h


def expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    up: float = 0.18,
    side: float = 0.08,
    down: float = 0.04,
) -> tuple[int, int, int, int]:
    """Grow a tight foot box upward so the crop keeps some ankle."""
    x, y, w, h = bbox
    width, height = size
    nx = max(0, x - int(w * side))
    ny = max(0, y - int(h * up))
    nr = min(width, x + w + int(w * side))
    nb = min(height, y + h + int(h * down))
    return nx, ny, max(1, nr - nx), max(1, nb - ny)


def bbox_from_mask(mask_path: str, *, pad: int = 24,
                   size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Axis-aligned bounds of non-zero mask pixels, padded, clipped to the image."""
    from PIL import Image
    with Image.open(mask_path) as im:
        mask = im.convert("L")
        if size is not None and mask.size != size:
            raise ValueError(f"mask size {mask.size} != image size {size}")
        box = mask.getbbox()
        mw, mh = mask.size
    if box is None:
        raise ValueError(f"empty mask: {mask_path}")
    l, t, r, b = box
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(mw, r + pad)
    b = min(mh, b + pad)
    return (l, t, max(1, r - l), max(1, b - t))


def read_meta(image_path: str) -> dict:
    mp = Path(image_path + ".meta.json")
    if not mp.is_file():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def source_loras(meta: dict) -> list[tuple[str, float]]:
    loras: list[tuple[str, float]] = []
    for cl in meta.get("content_loras") or []:
        if isinstance(cl, dict) and cl.get("file"):
            loras.append((cl["file"], float(cl.get("strength", 0.8))))
    role_loras = (meta.get("three_step") or {}).get("role_loras") or {}
    for lst in role_loras.values():
        for item in lst:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                loras.append((item[0], float(item[1])))
    return loras


def source_family(explicit: str | None, meta: dict) -> str:
    """Raw family name for a repaint: --family, then the sidecar's family, then anima when the
    sidecar names an Anima UNET (boxa_anima_fast drafts record ``unet`` but no ``family``; Krea,
    Qwen and Kontext sidecars also record a ``unet`` and must stay on the default)."""
    name = (explicit or meta.get("family")
            or ("anima" if anima_family(meta.get("unet")) else "illustrious"))
    return str(name).strip().lower()


def resolve_family(name: str | None) -> str:
    key = (name or "illustrious").strip().lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"unsupported family {name!r}; want {sorted(FAMILY_ALIASES)}")
    return FAMILY_ALIASES[key]


def compose_prompts(mode: str, meta: dict, prompt: str | None, neg: str | None,
                    keep_source: bool, recipe_neg: str) -> tuple[str, str]:
    pos_parts: list[str] = []
    if keep_source and meta.get("prompt"):
        pos_parts.append(str(meta["prompt"]))
    if prompt:
        pos_parts.append(prompt)
    pos_parts.append(ANATOMY_POS)
    if mode == "closeup":
        pos_parts.append(CLOSEUP_POS)
    neg_parts: list[str] = []
    if neg:
        neg_parts.append(neg)
    elif keep_source:
        neg_parts.append(str(meta.get("negative") or meta.get("neg") or recipe_neg))
    else:
        neg_parts.append("bad anatomy, deformed feet")
    neg_parts.append(ANATOMY_NEG)
    return ", ".join(pos_parts), ", ".join(neg_parts)


def resolve_foot_lora(lora_family: str, strength: float, explicit: list[tuple[str, float]] | None,
                      no_foot_lora: bool) -> tuple[list[tuple[str, float]], str | None]:
    if explicit:
        return list(explicit), None
    if no_foot_lora:
        return [], None
    path = FOOT_LORA.get(lora_family)
    if path:
        return [(path, strength)], None
    return [], (
        f"no installed Toes Adjuster for family={lora_family}; "
        "Illustrious Civitai 1496218 weights are not on disk. Continue tag-only, or pass --lora."
    )


def _loaders(g: dict, family: str, ckpt: str, loras: list[tuple[str, float]],
             pos: str, neg: str) -> tuple[list, list, list]:
    r = RECIPES[family]
    if r["arch"] == "sdxl":
        g["4"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
        model_ref, clip_src = ["4", 0], ["4", 1]
        for i, (lname, lstr) in enumerate(loras):
            nid = f"15_{i}"
            g[nid] = {"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_src, "lora_name": lname,
                "strength_model": lstr, "strength_clip": lstr}}
            model_ref, clip_src = [nid, 0], [nid, 1]
        g["10"] = {"class_type": "CLIPSetLastLayer",
                   "inputs": {"clip": clip_src, "stop_at_clip_layer": r["clip_skip"]}}
        clip_ref = ["10", 0]
        g["40"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["40", 0]
    else:
        g["44"] = {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}}
        model_ref = ["44", 0]
        for i, (lname, lstr) in enumerate(loras):
            nid = f"46_{i}"
            g[nid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model_ref, "lora_name": lname, "strength_model": lstr}}
            model_ref = [nid, 0]
        g["45"] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": r["clip"], "type": r["clip_type"], "device": "default"}}
        clip_ref = ["45", 0]
        g["15"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["15", 0]
        ms = r.get("model_sampling") or {}
        if ms.get("mode") == "auraflow" and ms.get("shift") is not None:
            g["47"] = {"class_type": "ModelSamplingAuraFlow",
                       "inputs": {"model": model_ref, "shift": ms["shift"]}}
            model_ref = ["47", 0]
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": pos}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": neg}}
    return model_ref, clip_ref, vae_ref


def build_graph(
    *,
    family: str,
    ckpt: str,
    loras: list[tuple[str, float]],
    pos: str,
    neg: str,
    seed: int,
    src_name: str,
    bboxes: list[tuple[int, int, int, int]] | None,
    detect: bool,
    guide_size: float,
    denoise: float,
    mask_frac: float,
    lock: str,
    steps: int,
    sampler: str,
    scheduler: str,
    cfg: float,
    cn_strength: float = 0.55,
    cn_end: float = 0.8,
    backend: str = "crop",
    feather_px: int = 28,
    feather_even: bool = False,
    mask_name: str | None = None,
    keep_outs: list[tuple[int, int, int, int]] | None = None,
    grow: int = 12,
) -> dict:
    g: dict = {}
    model_ref, clip_ref, vae_ref = _loaders(g, family, ckpt, loras, pos, neg)
    g["80"] = {"class_type": "LoadImage", "inputs": {"image": src_name, "upload": "image"}}
    out = ["80", 0]
    mask_full_ref = None
    if mask_name:
        g["81"] = {"class_type": "LoadImage", "inputs": {"image": mask_name, "upload": "image"}}
        g["82"] = {"class_type": "ImageToMask", "inputs": {"image": ["81", 0], "channel": "red"}}
        mask_full_ref = ["82", 0]
    r = RECIPES[family]
    if backend not in ("dd", "crop"):
        raise ValueError(f"backend must be 'dd' or 'crop', got {backend!r}")
    if backend == "dd" and lock == "canny":
        raise ValueError("polish/canny lock needs --backend crop")
    if (mask_name or keep_outs) and backend != "dd":
        raise ValueError("--mask / --keep-out need --backend dd")
    if bboxes:
        if lock == "canny" and r["arch"] != "sdxl":
            raise ValueError("polish/canny lock is SDXL-only; Anima use --mode recount|closeup")
        if backend == "dd":
            g["dd"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model_ref, "strength": 1.0}}
            dd_model = ["dd", 0]
            for bbox in bboxes:
                out = dd_inpaint_pass(
                    g, out, dd_model, vae_ref, ["6", 0], ["7", 0],
                    bbox=bbox, guide_size=guide_size, denoise=denoise, steps=steps,
                    cfg=cfg, sampler=sampler, scheduler=scheduler, seed=seed,
                    mask_frac=mask_frac, feather_px=feather_px, feather_even=feather_even,
                    mask_full_ref=mask_full_ref, keep_outs=keep_outs, grow=grow,
                )
        else:
            for bbox in bboxes:
                out = bbox_inpaint_pass(
                    g, out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0],
                    bbox=bbox, guide_size=guide_size, denoise=denoise, steps=steps,
                    cfg=cfg, sampler=sampler, scheduler=scheduler, seed=seed,
                    cn_strength=cn_strength, cn_end=cn_end,
                    controlnet_name=CN_CANNY if lock == "canny" else None,
                    mask_frac=mask_frac, lock=lock,
                    feather_px=feather_px, feather_even=feather_even,
                )
    elif detect:
        out = detailer_pass(
            g, out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0],
            detector=FOOT_DETECTOR, guide_size=guide_size, denoise=denoise,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
            seed=seed, sam=False,
        )
    else:
        raise ValueError("build_graph requires bboxes or detect=True")
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": "boxa_footfix"}}
    return g


def submit_and_fetch(graph: dict, dst: Path) -> bool:
    res = _shared_submit_and_fetch(graph, label=dst.stem, validate=lambda _raw: True, first_only=True)
    if not res:
        return False
    raw = res[0][0]
    if not _looks_like_png(raw):
        print(f"    SKIP {dst.name}: /view returned {len(raw)} bytes (not a PNG)")
        return False
    dst.write_bytes(raw)
    return True


def color_match_masked(dst_path: Path, src_path: Path,
                       bbox: tuple[int, int, int, int] | None = None, radius_px: int = 48) -> None:
    """Deterministic post-pass: align the inpaint region's low-frequency colour field to the source.

    Root cause this treats (2026-08-21 final4/final5 post-mortem): a pasted inpaint crop carries its
    own low-frequency tone field; at full-image zoom-out it reads as a visible "seam band" even when
    1:1 pixel diffs look clean. High-frequency detail (skin texture, line art) is preserved — only
    the low-pass field is transferred from the source.

    Pure numpy/PIL, no GPU. In-place rewrite of `dst_path` when a shift is actually applied;
    left byte-identical when the low-frequency delta is already sub-perceptual.
    """
    import numpy as np
    from PIL import Image as _PILImage, ImageFilter

    with _PILImage.open(src_path) as im:
        src_full = np.asarray(im.convert("RGB"), dtype=np.float32)
    with _PILImage.open(dst_path) as im:
        out_arr = np.asarray(im.convert("RGB"), dtype=np.float32).copy()
    if src_full.shape[:2] != out_arr.shape[:2]:
        print("    color-match skipped: size mismatch vs source")
        return

    mask = np.ones(out_arr.shape[:2], dtype=np.float32)
    x0 = y0 = x1 = y1 = 0
    if bbox is not None:
        x, y, w, h = bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(out_arr.shape[1], x + w), min(out_arr.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        mask[:] = 0.0
        mask[y0:y1, x0:x1] = 1.0

    def _lowpass_masked(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Normalized-convolution low pass: blur(a*m)/blur(m).

        Uses only pixels inside the mask, so the kernel's mixing with the untouched
        outside cancels between numerator and denominator — the estimate is not
        diluted even when the kernel is wider than the region.
        """
        r = float(radius_px)
        am = _PILImage.fromarray(np.clip(a * m, 0, 255).astype(np.uint8))
        mm = _PILImage.fromarray(np.clip(m * 255.0, 0, 255).astype(np.uint8))
        num = np.asarray(am.filter(ImageFilter.GaussianBlur(radius=r)), dtype=np.float32)
        den = np.asarray(mm.filter(ImageFilter.GaussianBlur(radius=r)), dtype=np.float32) / 255.0
        return num / np.maximum(den, 1e-3)

    # Per-channel masked low-frequency alignment: map output stats onto source stats
    # (mean+std match, gain clipped to +-30%). Stats come from the masked low pass,
    # so the untouched outside never biases the estimate.
    gains, lifts = [], []
    msum = max(float(mask.sum()), 1.0)
    for c in range(3):
        lo_src = _lowpass_masked(src_full[..., c], mask)
        lo_out = _lowpass_masked(out_arr[..., c], mask)
        mu_s = float((lo_src * mask).sum() / msum)
        mu_o = float((lo_out * mask).sum() / msum)
        sd_s = float((((lo_src - mu_s) * mask) ** 2).sum() / msum) ** 0.5
        sd_o = float((((lo_out - mu_o) * mask) ** 2).sum() / msum) ** 0.5
        g_c = float(np.clip(sd_s / sd_o, 0.7, 1.3)) if sd_o > 1e-6 else 1.0
        lifts.append(mu_s - g_c * mu_o)
        gains.append(g_c)

    shift = np.stack([out_arr[..., c] * gains[c] + lifts[c] for c in range(3)], axis=-1) - out_arr
    m3 = mask[..., None]
    blended = out_arr + shift * m3
    delta = float(np.abs(shift * m3).sum() / (msum * 3.0))

    if delta < 1.5:  # sub-perceptual (uint8 quantisation alone reads ~1/255): keep bytes
        print(f"    color-match: delta {delta:.2f} below threshold, unchanged")
        return
    result = _PILImage.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    result.save(dst_path)
    print(f"    color-match applied on masked region: mean|shift| {delta:.1f}/255 "
          f"(gains {['%.2f' % g for g in gains]}, lifts {['%.0f' % l for l in lifts]})")


def _preview_crops(image_path: str, bboxes: list[tuple[int, int, int, int]], stem: str,
                   keep_outs: list[tuple[int, int, int, int]] | None = None) -> list[Path]:
    from PIL import Image, ImageDraw
    OUT.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        for i, (x, y, w, h) in enumerate(bboxes):
            crop = rgb.crop((x, y, x + w, y + h))
            dest = OUT / f"footcrop_{stem}_{i:02d}_{x}_{y}_{w}_{h}.png"
            crop.save(dest)
            saved.append(dest)
        if keep_outs:
            vis = rgb.convert("RGBA")
            overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(overlay)
            for (x, y, w, h) in bboxes:
                dr.rectangle([x, y, x + w - 1, y + h - 1], outline=(0, 220, 0, 255), width=4)
            for (x, y, w, h) in keep_outs:
                dr.rectangle([x, y, x + w - 1, y + h - 1], fill=(220, 0, 0, 90),
                             outline=(220, 0, 0, 255), width=4)
            dest = OUT / f"footmask_{stem}_overlay.png"
            Image.alpha_composite(vis, overlay).convert("RGB").save(dest)
            saved.append(dest)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Dedicated foot/toe correction (protocol docs/t2i_prompt_validate_anatomy.md)")
    ap.add_argument("--image", required=True, help="source PNG")
    ap.add_argument("--mode", choices=tuple(MODE_PRESETS), default="recount",
                    help="recount=change toe count (default); polish=count ok, soften; closeup=toe-row mask")
    ap.add_argument("--bbox", action="append", type=parse_bbox, metavar="X,Y,W,H",
                    help="explicit foot bbox in source pixels (repeatable, one per foot)")
    ap.add_argument("--keep-out", action="append", type=parse_bbox, metavar="X,Y,W,H",
                    dest="keep_out",
                    help="rect in source pixels that stays original (repeatable; thigh keep-out)")
    ap.add_argument("--mask", default=None,
                    help="same-size PNG lasso (white=redraw); bbox defaults to its bounds")
    ap.add_argument("--grow", type=int, default=None,
                    help="GrowMask expand on the noise mask (default 12; 6 with --mask/--keep-out)")
    ap.add_argument("--detect", action="store_true",
                    help="foot YOLO FaceDetailer fallback when --bbox is omitted")
    ap.add_argument("--preview-crops", action="store_true",
                    help="save each (possibly padded) bbox crop locally and exit; no GPU")
    ap.add_argument("--box", choices=tuple(HOSTS), default="boxa")
    ap.add_argument("--n", type=int, default=None, help="best-of-N (mode default 4/2/4)")
    ap.add_argument("--denoise", type=float, default=None)
    ap.add_argument("--guide-size", type=float, default=None)
    ap.add_argument("--mask-frac", type=float, default=None,
                    help="bottom fraction of the crop to denoise (closeup default 0.28)")
    ap.add_argument("--family", default=None, help="illustrious|pony|noobai|anima (default: meta / illustrious)")
    ap.add_argument("--base", default=None, help="checkpoint / UNET override")
    ap.add_argument("--lora", action="append", default=None, metavar="FILE[:STR]",
                    help="foot LoRA override (repeatable); replaces the catalog default")
    ap.add_argument("--no-foot-lora", action="store_true", help="do not auto-attach Toes Adjuster")
    ap.add_argument("--keep-source-loras", action="store_true",
                    help="also reload the source image's character/style LoRAs (off by protocol)")
    ap.add_argument("--backend", choices=("dd", "crop"), default="dd",
                    help="dd=InpaintModelConditioning+DifferentialDiffusion (default); "
                         "crop=legacy VAEEncode+SetLatentNoiseMask chain")
    ap.add_argument("--keep-source-prompt", action="store_true",
                    help="prepend the source meta prompt (default is anatomy tags only)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--neg", default=None)
    ap.add_argument("--no-pad", action="store_true", help="do not expand bboxes for ankle")
    ap.add_argument("--feather-px", type=int, default=28,
                    help="composite FeatherMask on each side of the bbox (default 28; bottom is half unless --feather-even)")
    ap.add_argument("--feather-even", action="store_true",
                    help="feather the bbox bottom as much as the other sides (needed when the cut is mid-foot, not at the floor)")
    ap.add_argument("--no-color-match", action="store_true",
                    help="skip the deterministic low-frequency colour-match post-pass (default on: aligns the inpaint region's tone field to the source to kill zoom-out seam bands)")
    ap.add_argument("--color-match-radius", type=int, default=48,
                    help="low-pass radius in source px for the colour-match post-pass (default 48)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="tile original + candidates when n>1")
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()
    preset = MODE_PRESETS[args.mode]
    denoise = preset["denoise"] if args.denoise is None else args.denoise
    n = preset["n"] if args.n is None else args.n
    guide_size = preset["guide_size"] if args.guide_size is None else args.guide_size
    mask_frac = preset["mask_frac"] if args.mask_frac is None else args.mask_frac
    lock = preset["lock"]
    steps = preset["steps"]
    if n < 1:
        ap.error("--n must be >= 1")
    if not 0.05 <= denoise <= 0.95:
        ap.error("--denoise must be in 0.05..0.95")
    if not 0.05 <= mask_frac <= 1.0:
        ap.error("--mask-frac must be in 0.05..1.0")
    if args.feather_px < 0:
        ap.error("--feather-px must be >= 0")

    src = Path(args.image)
    if not src.is_file():
        ap.error(f"image not found: {args.image}")
    size = png_size(args.image)
    if size is None:
        ap.error(f"--image must be a readable PNG: {args.image}")
    width, height = size

    if args.mask and args.detect:
        ap.error("use --mask or --detect, not both")
    if args.bbox and args.detect:
        ap.error("use --bbox or --detect, not both")
    if not args.bbox and not args.detect and not args.mask:
        ap.error("need --bbox (protocol path), --mask, or --detect")

    backend = args.backend
    if args.mode == "polish" and backend == "dd":
        backend = "crop"
        print("note: polish/canny uses --backend crop")
    if (args.mask or args.keep_out) and backend != "dd":
        ap.error("--mask / --keep-out need --backend dd")
    if args.keep_out and not args.bbox and not args.mask:
        ap.error("--keep-out needs --bbox or --mask")

    mask_name_local = None
    if args.mask:
        mp = Path(args.mask)
        if not mp.is_file():
            ap.error(f"mask not found: {args.mask}")
        try:
            derived = bbox_from_mask(args.mask, pad=24, size=(width, height))
        except ValueError as exc:
            ap.error(str(exc))
        mask_name_local = str(mp)
        if args.bbox is None:
            print(f"note: --bbox from mask bounds+pad -> {derived[0]},{derived[1]},{derived[2]},{derived[3]}")

    bboxes = list(args.bbox or ([derived] if args.mask else []))
    keep_outs = list(args.keep_out or [])
    if bboxes:
        for (x, y, w, h) in bboxes:
            if x + w > width or y + h > height:
                ap.error(f"--bbox {x},{y},{w},{h} is out of bounds for {width}x{height}")
        # dd assumes the bbox is already the defect (toe row). Expanding to the whole
        # limb is what made the previous crop chain redraw a healthy foot.
        if preset["pad"] and not args.no_pad and backend != "dd":
            bboxes = [expand_bbox(b, (width, height)) for b in bboxes]
    for i, ko in enumerate(keep_outs):
        x, y, w, h = ko
        if x + w > width or y + h > height or x < 0 or y < 0:
            ap.error(f"--keep-out {x},{y},{w},{h} is out of bounds for {width}x{height}")

    if args.mask or keep_outs:
        if args.mask_frac is None:
            mask_frac = 1.0
        grow = 6 if args.grow is None else args.grow
    else:
        grow = 12 if args.grow is None else args.grow
    if grow < 0:
        ap.error("--grow must be >= 0")

    stem = src.stem
    if args.preview_crops:
        if not bboxes:
            ap.error("--preview-crops requires --bbox or --mask")
        saved = _preview_crops(args.image, bboxes, stem, keep_outs=keep_outs or None)
        print("preview crops ->", OUT)
        for p in saved:
            print(f"  {p.name} {p.stat().st_size // 1024}KB")
        return

    meta = {} if args.no_meta else read_meta(args.image)
    lora_family = source_family(args.family, meta)
    try:
        family = resolve_family(lora_family)
    except ValueError as exc:
        ap.error(str(exc))
    r = RECIPES[family]
    if lock == "canny" and r["arch"] != "sdxl":
        ap.error("polish/canny lock is SDXL-only; Anima use --mode recount or --mode closeup")
    if args.base:
        ckpt = args.base
    elif family == "anima":
        ckpt = meta.get("unet") or meta.get("base_ckpt") or r["ckpt"]
    elif args.box == "boxb":
        ckpt = "iLustMix.safetensors"
    else:
        ckpt = meta.get("base_ckpt") or r["ckpt"]
    if family == "anima":
        # Loader names differ per box (anima_models.json unet_by_box); preflight_assets below
        # checks the result against the box's UNETLoader enum.
        ckpt = anima_models.unet_for_box(ckpt, args.box)

    explicit_lora = None
    if args.lora:
        explicit_lora = []
        for spec in args.lora:
            f, _, s = spec.partition(":")
            if not f:
                ap.error("empty --lora")
            explicit_lora.append((f, float(s) if s else preset["lora_strength"]))
    foot_loras, lora_note = resolve_foot_lora(
        lora_family if lora_family in FOOT_LORA else family,
        preset["lora_strength"], explicit_lora, args.no_foot_lora,
    )
    loras = list(foot_loras)
    if args.keep_source_loras:
        loras = source_loras(meta) + loras

    pos, neg = compose_prompts(
        args.mode, meta, args.prompt, args.neg, args.keep_source_prompt, r["neg"])

    if loras:
        known = _object_info_enum("LoraLoader", "lora_name")
        loras = [(resolve_lora_file(f, known), s) for f, s in loras]
    preflight_assets(r["arch"], ckpt, [f for f, _ in loras],
                     controlnets=([CN_CANNY] if lock == "canny" else ()))
    if args.detect:
        missing_det = _missing_assets(
            _object_info_enum("UltralyticsDetectorProvider", "model_name"), [FOOT_DETECTOR])
        if missing_det:
            ap.error(f"foot detector not found: {missing_det}")

    src_name = upload_image(args.image)
    if not src_name:
        ap.error(f"failed to upload image to {args.box.upper()}: {args.image}")
    mask_upload = None
    if mask_name_local:
        mask_upload = upload_image(mask_name_local)
        if not mask_upload:
            ap.error(f"failed to upload mask to {args.box.upper()}: {mask_name_local}")

    OUT.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else (int(time.time() * 1000) % 2_000_000_000)
    print(f"foot-fix box={args.box} endpoint={BOXA_HOST} mode={args.mode} backend={backend} family={family} "
          f"ckpt={ckpt} n={n} denoise={denoise} lock={lock} guide={guide_size} "
          f"mask_frac={mask_frac} grow={grow} feather={args.feather_px}"
          f"{' even' if args.feather_even else ''} model={MODEL}")
    if lora_note:
        print(f"  note: {lora_note}")
    if loras:
        print(f"  LoRAs: {', '.join(f'{Path(f).name}@{s}' for f, s in loras)}")
    if bboxes:
        print(f"  feet: {bboxes}")
    else:
        print(f"  detector: {FOOT_DETECTOR}")
    if keep_outs:
        print(f"  keep-out: {keep_outs}")
    if mask_upload:
        print(f"  mask: {mask_name_local} -> {mask_upload}")
    print(f"  + {pos}")
    print(f"  - {neg}")

    ok = 0
    written: list[Path] = []
    try:
        for i in range(n):
            seed = base_seed + i
            graph = build_graph(
                family=family, ckpt=ckpt, loras=loras, pos=pos, neg=neg, seed=seed,
                src_name=src_name, bboxes=bboxes or None, detect=args.detect,
                guide_size=guide_size, denoise=denoise, mask_frac=mask_frac, lock=lock,
                steps=steps, sampler=r["sampler"], scheduler=r["scheduler"], cfg=r["cfg"],
                backend=backend, feather_px=args.feather_px, feather_even=args.feather_even,
                mask_name=mask_upload, keep_outs=keep_outs or None, grow=grow,
            )
            tag = args.mode if backend == "crop" else f"{args.mode}_{backend}"
            dst = OUT / f"ff_{stem}_{tag}_d{denoise}_{i:02d}_{seed}.png"
            if submit_and_fetch(graph, dst):
                if bboxes and not args.no_color_match:
                    for bb in bboxes:
                        color_match_masked(dst, src, bbox=bb, radius_px=args.color_match_radius)
                meta_out = {
                    "driving_model": MODEL, "gen_engine": f"comfyui:{args.box}",
                    "pipeline": "foot_fix", "protocol": "t2i_prompt_validate_anatomy",
                    "source": str(src), "mode": args.mode, "backend": backend, "family": family,
                    "lora_family": lora_family, "base_ckpt": ckpt, "bboxes": bboxes,
                    "detect": bool(args.detect), "detector": FOOT_DETECTOR if args.detect else None,
                    "denoise": denoise, "lock": lock, "mask_frac": mask_frac,
                    "feather_px": args.feather_px, "feather_even": args.feather_even,
                    "color_match": bool(bboxes) and not args.no_color_match,
                    "color_match_radius": args.color_match_radius,
                    "keep_outs": keep_outs, "mask": mask_name_local, "grow": grow,
                    "guide_size": guide_size, "seed": seed, "n": n, "index": i,
                    "loras": [{"file": f, "strength": s} for f, s in loras],
                    "prompt": pos, "negative": neg,
                }
                _atomic_write_text(Path(str(dst) + ".meta.json"),
                                   json.dumps(meta_out, ensure_ascii=False, indent=2))
                print(f"  [{i:02d} seed={seed}] -> {dst.name} {dst.stat().st_size // 1024}KB")
                written.append(dst)
                ok += 1
    finally:
        _free_vram()

    if args.contact_sheet and written:
        try:
            from contact_sheet import build_sheet
            sheet_tag = args.mode if backend == "crop" else f"{args.mode}_{backend}"
            sheet = OUT / f"ff_{stem}_{sheet_tag}_contact_{base_seed}.png"
            build_sheet([str(src)] + [str(p) for p in written], str(sheet))
            print(f"  contact sheet -> {sheet.name}")
        except Exception as exc:
            print(f"  contact sheet skipped: {exc}")

    print(f"done: {ok}/{n} -> {OUT}")
    print("  pick by eye (toe count is not an auto-gate). lock=none + high denoise is the recount path.")


if __name__ == "__main__":
    main()
