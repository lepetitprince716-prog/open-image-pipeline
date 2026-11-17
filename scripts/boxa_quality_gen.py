#!/usr/bin/env python3
"""Highest-quality image-gen on BOXA ComfyUI for two model families.

Two researched, source-grounded "max quality" pipelines (full recipe + sources in
scripts/BOXA_QUALITY_RECIPES.md):

  illustrious  SDXL-architecture. CheckpointLoaderSimple + CLIP-skip + KSampler.
               default ckpt waiIllustriousSDXL_v170, euler_ancestral / karras, 28 steps,
               cfg 6.0, clip skip -2, 832x1216, external VAE sdxl_vae_fp16_fix.

  anima        Official flow-matching Anima (NOT SDXL). UNETLoader + CLIPLoader(qwen_3_06b_base,
               type=stable_diffusion) + qwen_image_vae + plain KSampler. er_sde / simple,
               35 steps, cfg 4.5, no clip skip, 896x1152. Graph wiring is taken VERBATIM from the
               official ComfyUI workflow embedded in circlestone-labs/Anima example.png.

ADVANCED (--max): a strategic pre/post stack, applied per architecture (every node verified present
on BOXA):
  pre  (model patches, SDXL only by default)  FreeU_V2 -> Perturbed-Attention-Guidance -> RescaleCFG
  post (both families)                        model-upscale (Remacri 1.5x) + low-denoise 2nd-pass
                                              refine -> FaceDetailer(face) -> FaceDetailer(eyes)
  Anima is flow-matching, so the SDXL-specific model patches (FreeU/PAG/RescaleCFG) are OFF by
  default for it; the post stack (tiled upscale + detailers) is architecture-safe and stays on.
  Override per-patch with --freeu/--no-freeu, --pag, --rescale, --detailers.

Without --max the simpler BOXA_HIRES tiers apply: off | face (FaceDetailer) | full (model-upscale +
2nd-pass refine + FaceDetailer).

GPU policy (BOXA is a shared gaming box, RTX 5080 16GB): the --max stack and the BOXA_HIRES tiers are
heavy (extra forward passes + tiled diffusion) — run them only when the GPU is free.

Output -> outputs/<character>/by_model/<DRIVING_MODEL>/boxa_quality/ + a per-image .meta.json sidecar
(outputs/MODEL_FOLDER_RULE.md). DRIVING_MODEL identifies the driver (default claude).

Usage:
  python3 scripts/boxa_quality_gen.py --family anima --prompt "1girl, rooftop at dusk, city lights"
  python3 scripts/boxa_quality_gen.py --family illustrious --prompt "..." --max          # full pre/post stack
  python3 scripts/boxa_quality_gen.py --family anima --prompt "..." --max --pag 2.0       # try PAG on Anima
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_CHARACTER, by_model_dir  # noqa: E402
from boxa_comfy import (BOXA_HOST, upload_image, png_size, image_size,  # sibling; thin stdlib driver layer
                      save_image_with_meta as _save_image_with_meta, preflight_assets,
                      object_info_enum, node_present, submit_and_fetch as _submit_and_fetch,
                      free_vram)
from lora_catalog import base_model_of, parse_spec, pick, record_of, triggers_for  # sibling; stdlib-only at import time
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from loramake import model_resources as _mr
from loramake.tags import WRAP_TEMPLATES, wrap_prompt
from wildcard import WildcardEngine, WildcardError  # sibling; {a|b}/__file__ prompt expansion
import anima_models  # sibling; single source of truth for the Anima text-encoder/VAE contract
MODEL = os.environ.get("DRIVING_MODEL", "claude").strip().lower() or "claude"
OUT = by_model_dir(OUTPUT_CHARACTER, "boxa_quality", model=MODEL)
# Default "full" = the BOXB iLustMix base pipeline (image-space hires + face detailer); set BOXA_HIRES=off
# for a quick bare-latent preview. GPU headroom is fine (RTX 5080, not gaming during generation).
HIRES = os.environ.get("BOXA_HIRES", "full").strip().lower()
FACE_DETECTOR = os.environ.get("BOXA_FACE_DETECTOR", "bbox/face_yolov9c.pt")  # known present on BOXA
EYE_DETECTOR = os.environ.get("BOXA_EYE_DETECTOR", "bbox/Eyeful_v2-Individual.pt")
HAND_DETECTOR = os.environ.get("BOXA_HAND_DETECTOR", "segm/hand_yolov8s.pt")  # optional SEGM; not a guaranteed anatomy fix
FOOT_DETECTOR = os.environ.get("BOXA_FOOT_DETECTOR", "segm/foot_yolov8x_v2.pt")  # optional SEGM; not a guaranteed anatomy fix
UPSCALER = os.environ.get("BOXA_UPSCALER", "4x_foolhardy_Remacri.pth")
FREEU_DEFAULT = (1.3, 1.4, 0.9, 0.2)  # FreeU_V2 b1,b2,s1,s2 (SDXL community values)
# LORAMAKE_NEG_PROFILE: only the exact value "nsfw" (case/space-tolerant) selects the NSFW
# profile; unset, "sfw", or any other value keeps the rating token in the illustrious
# negative — fail closed (env supplied by the mirror).
_NSFW_PROFILE = os.environ.get("LORAMAKE_NEG_PROFILE", "").strip().lower() == "nsfw"
_RATING_NSFW_TOKEN = "" if _NSFW_PROFILE else "nsfw, "


def apply_quality_scaffold(body: str, recipe: dict, *, skip: bool = False) -> str:
    """One wrap site: loramake.tags.wrap_prompt. Anima keeps its own non-quality prefix."""
    if skip:
        return body
    wrap = recipe.get("wrap")
    if wrap:
        return wrap_prompt(body, wrap)
    return (recipe.get("pos_prefix") or "") + body

# ControlNet (SDXL / Illustrious only; the BOXA control models are all SDXL).
# type -> {model on BOXA, pre: auto-preprocessor node}. The controlnet_aux preprocessors print progress and
# used to OSError[Errno 22] on this box; after the 2026-06-25 stdout-redirect launcher fix they run, so
# openpose and lineart now auto-preprocess from a raw image (canny uses the core `Canny` node). Pass
# --no-preprocess to feed an already-made control map instead.
CONTROLNET = {
    "openpose": {"model": "IllustriousXL_openpose.safetensors", "pre": "OpenposePreprocessor"},
    "canny": {"model": "diffusers_xl_canny_mid.safetensors", "pre": "canny"},
    "lineart": {"model": "mistoLine_rank256.safetensors", "pre": "LineArtPreprocessor"},
    # Depth maps follow the MiDaS convention (near = white). Rendered maps from the
    # 3D stage (3d/chargen/render_control_maps.py) are fed with --no-preprocess.
    "depth": {"model": "diffusers_xl_depth_full.safetensors", "pre": "DepthAnythingV2Preprocessor"},
}

# Anima (flow-matching) control. Every public weight is trained on the 28-block Anima 2B DiT. The
# core / kohya nodes patch by block index with no remap, so with them the CLI refuses any other block
# count (anima_models.json `blocks`; --no-arch-guard bypasses); the repo's own LoramakeAnimaLLLiteApply
# (tools/comfy_nodes/loramake_anima) remaps 28->40 on the fly and also takes a 40-block base.
#   kind "lllite"      -> kohya ControlNet-LLLite v2: a MODEL patch. LoramakeAnimaLLLiteApply and BOXB's
#                         AnimaLLLiteApply_sdscripts take the file name as a widget (model_patches/ or
#                         controlnet/); BOXA core needs ModelPatchLoader (models/model_patches/) +
#                         AnimaLLLiteApply.
#   kind "control_lora" -> Claquasse Anima-Control-Pose: LoraLoaderModelOnly (lora.* keys) +
#                         AnimaControlApply (control_embedder.* keys, VAE-encoded skeleton latent).
# "invert": the preprocessor emits white-on-black while the weight was trained black-on-white.
# Both kinds patch the MODEL, not the conditioning: build_graph applies them to the base KSampler
# only and keeps the refine / upscale / detailer passes on the unpatched model unless --control-refine.
ANIMA_CONTROL = {
    "lineart": {"kind": "lllite", "file": "anima-lllite-lineart-1.safetensors", "pre": "LineArtPreprocessor",
                "invert": True, "convention": "black lines on white (Preview3-era weight)"},
    "scribble": {"kind": "lllite", "file": "anima-lllite-scribble-1.safetensors", "pre": "HEDPreprocessor",
                 "invert": False, "convention": "white strokes on black (HED output as-is)"},
    "depth": {"kind": "lllite", "file": "anima-lllite-depth-1.safetensors", "pre": "DepthAnythingV2Preprocessor",
              "invert": False, "convention": "near = white (Depth Anything V2)"},
    "any": {"kind": "lllite", "file": "anima-lllite-any-test-like-v2.safetensors", "pre": "LineArtPreprocessor",
            "invert": False, "convention": "lineart / scribble / grayscale, either polarity (Base v1.0 weight, "
                                           "trained with random inversion)"},
    "pose": {"kind": "lllite", "file": "anima-lllite-pose-1.safetensors", "pre": "DWPreprocessor",
             "invert": False, "soft_prior": True,
             "convention": "DWPose colour skeleton on black; author-rated weak -- a soft prior, prefer pose_lora"},
    "pose_lora": {"kind": "control_lora", "file": "anima_pose_preview2.safetensors", "pre": "DWPreprocessor",
                  "invert": False, "strength_range": (0.0, 2.0),
                  "convention": "thin DWPose skeleton on black; strength 1.0 follows the skeleton"},
    # Tile & Repair v2: fed the degraded image itself (no preprocessor); without --control-image it
    # takes --init-image. Measured on BOXA 2026-10-01 (Dreamland, 0.9): img2img at denoise 0.5 / 0.75
    # stays far closer to the source (mean abs diff 7.5 vs 16.9, 9.6 vs 34.3). 28-block weight;
    # not tested on a 40-block base.
    "tile": {"kind": "lllite", "file": "anima_tile_taskplus_v2.safetensors", "pre": None,
             "invert": False, "strength": 0.9,
             "convention": "the degraded image itself; Tile & Repair v2 (Civitai 3068951)"},
}
# node candidates per kind, in preference order (repo remap node, core, kohya); main() picks by /object_info presence
REMAP_LLLITE_NODE = "LoramakeAnimaLLLiteApply"
# Preview3-era kohya weights: verified 2026-09-16 not to transfer to a 40-block base even when
# remapped (ultrabake1 / anima29B_v10, lineart-1 at 0.8 and 0.4); any-test-like-v2 follows the sketch.
PREVIEW3_CONTROL_TYPES = frozenset({"lineart", "scribble", "depth", "pose"})
ANIMA_CONTROL_NODES = {
    "lllite": (REMAP_LLLITE_NODE, "AnimaLLLiteApply", "AnimaLLLiteApply_sdscripts"),
    "control_lora": ("AnimaControlApply",),
}
ANIMA_CONTROL_INSTALL_DOC = "docs/anima_line/ANIMA_CONTROL_LINE.md"
CONTROL_STRENGTH_DEFAULT = 0.8


def _preprocess_node(pre: str, image_ref: list) -> dict:
    """The auto-preprocessor node for a control type (controlnet_aux names, core Canny)."""
    if pre == "canny":
        return {"class_type": "Canny", "inputs": {"image": image_ref, "low_threshold": 0.4, "high_threshold": 0.8}}
    if pre == "OpenposePreprocessor":
        return {"class_type": "OpenposePreprocessor", "inputs": {
            "image": image_ref, "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": 1024}}
    if pre == "DWPreprocessor":
        return {"class_type": "DWPreprocessor", "inputs": {
            "image": image_ref, "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": 1024}}
    if pre == "LineArtPreprocessor":
        return {"class_type": "LineArtPreprocessor",
                "inputs": {"image": image_ref, "coarse": "disable", "resolution": 1024}}
    if pre == "HEDPreprocessor":
        return {"class_type": "HEDPreprocessor", "inputs": {"image": image_ref, "safe": "enable", "resolution": 1024}}
    if pre == "DepthAnythingV2Preprocessor":
        return {"class_type": "DepthAnythingV2Preprocessor", "inputs": {
            "image": image_ref, "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 1024}}
    raise ValueError(f"unknown preprocessor {pre!r}")


def control_block_guard(ckpt: str, allow: bool, node: str | None = None) -> None:
    """Refuse Anima control on a base whose block count ``node`` cannot serve (SystemExit).

    Every public weight is 28-block. The core / kohya nodes patch by block index with no remap and
    only take a registered 28-block base; LoramakeAnimaLLLiteApply remaps 28->40 on the fly and
    also takes a 40-block base. ``allow`` (--no-arch-guard) bypasses.
    """
    card = anima_models.card(ckpt)
    blocks = (card or {}).get("blocks")
    accepted = (28, 40) if node == REMAP_LLLITE_NODE else (28,)
    if allow or blocks in accepted:
        return
    sys.exit(f"ERROR: Anima control via {node or 'the core node'} needs a "
             f"{'/'.join(str(b) for b in accepted)}-block base: {ckpt!r} is "
             f"{'unregistered in anima_models.json' if card is None else f'{blocks}-block'}. "
             f"Every public LLLite / control-LoRA weight is 28-block and the core patcher has no "
             f"28->40 remap, so a 40-block base would misalign silently ({REMAP_LLLITE_NODE} remaps; "
             f"see {ANIMA_CONTROL_INSTALL_DOC}). --no-arch-guard bypasses.")


def lora_block_guard(ckpt: str, allow: bool, loras: list) -> None:
    """Refuse an Anima LoRA whose block layout differs from the registered base's (SystemExit).

    This line loads Anima LoRAs with LoraLoaderModelOnly, which only fits a LoRA whose key layout
    matches the base: 28-block (2B) content LoRAs on 28-block bases, the native 40-block turbo
    extract (``anima_turbo_29b_extract``) on the 40-block 2.9B family.  A LoRA the registry does
    not know, or one without a ``blocks`` entry, is assumed to be an ordinary 28-block content
    LoRA.  A 40-block base with 28-block LoRAs needs the AnimaLoRARemapTagLoader chain that only
    the fast line builds; a 28-block base with the 40-block extract has no valid loader at all.
    An unregistered base is left alone (nothing known to compare against); ``--no-arch-guard``
    bypasses.
    """
    if not loras or allow:
        return
    card = anima_models.card(ckpt)
    base_blocks = (card or {}).get("blocks")
    if card is None or base_blocks is None:
        return
    for name, _ in loras:
        entry = anima_models.lora_card(str(name)) or {}
        lora_blocks = entry.get("blocks") or 28
        if lora_blocks == base_blocks:
            continue
        if base_blocks == 28:
            sys.exit(f"ERROR: {name!r} is a {lora_blocks}-block LoRA ({entry.get('name', 'unregistered')}); "
                     f"{ckpt!r} is a 28-block base and has no loader for it. Its card names the base it "
                     f"fits (compat); --no-arch-guard bypasses.")
        sys.exit(f"ERROR: {ckpt!r} is a {base_blocks}-block base; the quality line loads Anima LoRAs "
                 f"with LoraLoaderModelOnly, and {name!r} is a {lora_blocks}-block LoRA that only fits "
                 f"{lora_blocks}-block bases. Use the fast line (gen.py draft / boxa_anima_fast.py, "
                 f"AnimaLoRARemapTagLoader) for content LoRAs on 40-block bases, drop the LoRAs "
                 f"(promote inherits the draft's; pass --lora to replace them), or --no-arch-guard to "
                 f"bypass. The native 40-block turbo extract (anima_turbo_29b_extract) is accepted here.")


def card_sampling_defaults(ckpt: str) -> dict:
    """Sampling defaults a registered Anima card states (anima_models.json ``recipe``): steps / cfg /
    sampler / scheduler where present.  RECIPES["anima"] describes the quality_default card only; a
    --base onto another card (turbo bake, 2.9B, Turbo v1.1) must not inherit Dreamland's 30 steps /
    cfg 4.5.  Explicit CLI flags still win (see apply_sampling_defaults)."""
    card = anima_models.card(ckpt)
    recipe = (card or {}).get("recipe")
    if not isinstance(recipe, dict):
        return {}
    return {key: recipe[key] for key in ("steps", "cfg", "sampler", "scheduler")
            if recipe.get(key) not in (None, "")}


def apply_sampling_defaults(opts: dict, defaults: dict) -> dict:
    """opts with every unset sampling key (None) filled from ``defaults`` — CLI flag > card > RECIPES."""
    merged = dict(opts)
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    return merged


def control_end_guard(kind: str, control_end: float) -> str | None:
    """The error text for a --control-end the selected control route cannot honour, else None.

    The LLLite nodes and SDXL ControlNetApplyAdvanced take an end_percent; AnimaControlApply
    (``control_lora`` / pose_lora) has no timestep range, so anything but the default 1.0 would be
    accepted and silently ignored there.
    """
    if kind == "control_lora" and control_end != 1.0:
        return (f"--control-end {control_end} cannot be honoured on the pose_lora route: "
                f"AnimaControlApply has no end_percent input (the control stays active for the whole "
                f"sample). Leave it at 1.0 or use an LLLite control type.")
    return None


def effective_sampling(family: str, opts: dict, *, three_step: bool = False) -> dict:
    """The sampling values the graphs actually render with (CLI flag > card recipe > RECIPES), for
    banners and sidecars: sampler, scheduler, steps, cfg and, on the flow line, the
    ModelSamplingAuraFlow settings (None when the node is skipped or under --three-step)."""
    r = RECIPES[family]
    out = {"sampler": opts.get("sampler") or r["sampler"],
           "scheduler": opts.get("scheduler") or r["scheduler"],
           "steps": opts["steps"] if opts.get("steps") is not None else r["steps"],
           "cfg": opts["cfg"] if opts.get("cfg") is not None else r["cfg"],
           "model_sampling": None}
    if r["arch"] == "flow" and not three_step:
        ms_cfg = r.get("model_sampling") or {}
        ms_mode = opts.get("model_sampling") or ms_cfg.get("mode")
        ms_shift = opts.get("shift") if opts.get("shift") is not None else ms_cfg.get("shift")
        if ms_mode == "auraflow" and ms_shift is not None:
            out["model_sampling"] = {"mode": ms_mode, "shift": ms_shift}
    return out


def preview3_on_40_block_warning(ckpt: str, control_type: str) -> str | None:
    """A warning line when a Preview3-era LLLite type is used on a 40-block base, else None."""
    if control_type not in PREVIEW3_CONTROL_TYPES:
        return None
    if (anima_models.card(ckpt) or {}).get("blocks") != 40:
        return None
    return (f"warn: --control {control_type} is a Preview3-era weight; on a 40-block base it does not "
            f"follow the sketch even remapped (measured 2026-09-16). Use --control any.")


def preflight_remap_lllite(file: str, enum=object_info_enum) -> None:
    """SystemExit unless LoramakeAnimaLLLiteApply.lllite_name lists ``file`` (live re-probe on a miss)."""
    known = enum(REMAP_LLLITE_NODE, "lllite_name")
    if file not in known:
        known = enum(REMAP_LLLITE_NODE, "lllite_name", refresh=True)
    if file not in known:
        sys.exit(f"ERROR: {file} is not listed by {REMAP_LLLITE_NODE}.lllite_name on this box "
                 f"(models/model_patches/ or models/controlnet/). Install steps: {ANIMA_CONTROL_INSTALL_DOC}")


def pick_control_node(candidates: tuple[str, ...], present=node_present) -> str:
    """First candidate the box registers; the first candidate with a warning when the probe is
    unreachable; SystemExit naming the install steps when the box has none of them."""
    verdicts = [(c, present(c)) for c in candidates]
    for name, ok in verdicts:
        if ok:
            return name
    if any(ok is None for _, ok in verdicts):
        print(f"  warn: /object_info unreachable; assuming {candidates[0]} is installed")
        return candidates[0]
    sys.exit(f"ERROR: none of {', '.join(candidates)} is registered on this box. "
             f"Install steps: {ANIMA_CONTROL_INSTALL_DOC}")

# Anima IP-Adapter (custom nodes ComfyUI-Anima_IP-Adapter, SigLIP2). A MODEL patch like the SDXL
# one, but it mutates the cached DiT in place: after a run the same base renders differently until
# ComfyUI's node cache is reset, so run() always ends an Anima IPA run with a full /free.
# Strength 0.6: 1.0 over-transfers the reference's composition (BOXA, 2026-10-01).
ANIMA_IPADAPTER = {"file": "anima_ip_adapter-Character_Reference-10.safetensors", "weight": 0.6,
                   "ref_image_size": 512, "siglip_layer": -1, "ip_cfg_scale": 4.0,
                   "ip_cfg_separate": False, "gray_null": False, "use_lora": True}
ANIMA_IPADAPTER_NODES = ("AnimaIPAdapterLoader", "AnimaIPAdapterApply")


def preflight_anima_ipadapter(present=node_present, enum=object_info_enum) -> None:
    """SystemExit when the box lacks the Anima IP-Adapter nodes or weight; warn when unreachable."""
    verdicts = {n: present(n) for n in ANIMA_IPADAPTER_NODES}
    missing = [n for n, ok in verdicts.items() if ok is False]
    if missing:
        sys.exit(f"ERROR: Anima --ip-image needs {', '.join(missing)} (custom node pack "
                 f"ComfyUI-Anima_IP-Adapter), not registered on this box ({BOXA_HOST}).")
    if None in verdicts.values():
        print("  warn: /object_info unreachable; assuming the Anima IP-Adapter nodes are installed")
        return
    known = enum("AnimaIPAdapterLoader", "ip_adapter_name")
    if known and ANIMA_IPADAPTER["file"] not in known:
        sys.exit(f"ERROR: {ANIMA_IPADAPTER['file']} is not listed by AnimaIPAdapterLoader on this box "
                 f"({BOXA_HOST}); install it under models/ipadapter/.")


# IPAdapter (SDXL / Illustrious only; the BOXA IPAdapter + CLIP-Vision models are SDXL/ViT-H).
# mode -> {ipadapter model, clip_vision, weight_type, default weight}. Applied as a MODEL patch.
IPADAPTER = {
    # Per-mode defaults for weight / end_at / embeds_scaling. Face uses end_at=0.5
    # (apply IPAdapter only on early composition steps so the late steps + face/eye
    # detailer render clean eyes and avoid colour bleed into the hair) and K+V
    # embeds_scaling (stronger identity fidelity). Tuned on the character face line;
    # all are overridable via --ip-weight / --ip-end / --ip-scaling.
    "style": {"model": "plus.sdxl.vit-h.safetensors",
              "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
              "weight_type": "style transfer", "weight": 0.8,
              "end": 1.0, "scaling": "V only"},
    "face": {"model": "plus.face.sdxl.vit-h.safetensors",
             "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
             "weight_type": "linear", "weight": 0.9,
             "end": 0.5, "scaling": "K+V"},
    # Chained dual lock: the face patch (early steps, K+V) plus a full-range style
    # transfer stage on top — identity AND aesthetic from one reference. Primary
    # fields mirror 'face' (they drive the first stage and the --ip-weight/--ip-end
    # defaults); the style stage reads style_model/style_weight and pins
    # weight_type="style transfer" / V-only scaling (not exposed per call —
    # only the weight is a knob). style_weight 0.4
    # measured 2026-08-27 (same source/seeds sweep vs 0.7 and face-only): 0.7
    # overcooks into hatched/striped texture (LPIPS worsens 0.37->0.47), 0.4 keeps
    # the reference's palette and eye colour with clean shading and beat both
    # alternatives on ccip/graycorr/tag-F1.
    "both": {"model": "plus.face.sdxl.vit-h.safetensors",
             "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
             "weight_type": "linear", "weight": 0.9,
             "end": 0.5, "scaling": "K+V",
             "style_model": "plus.sdxl.vit-h.safetensors", "style_weight": 0.4},
}

# Mirror of IPAdapterAdvanced's weight_type enum on BOXA (verified 2026-08-28 via
# /object_info). Kept literal so --ip-weight-type fails fast on typos offline.
IP_WEIGHT_TYPES = (
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
    "weak input", "weak output", "weak middle", "strong middle",
    "style transfer", "composition", "strong style transfer",
    "style and composition", "style transfer precise", "composition precise",
)

# Source-grounded recipes (see BOXA_QUALITY_RECIPES.md). Settings within the researched ranges.
_ANIMA_CONTRACT = anima_models.shared_contract()
RECIPES = {
    "illustrious": {
        "arch": "sdxl",
        "ckpt": "Illustrious\\anime\\waiIllustriousSDXL_v170.safetensors",
        "vae": "sdxl_vae_fp16_fix.safetensors",
        # base spec aligned to the BOXB iLustMix workflow (boxb_showcase.py): dpmpp_2m/karras, 32 steps,
        # cfg 5.0, native 1024x1536 (no latent upscale — image-space hires only).
        "sampler": "dpmpp_2m", "scheduler": "karras",
        "steps": 32, "cfg": 5.0, "clip_skip": -2, "res": "1024x1536",
        # Single wrap site: loramake.tags.WRAP_TEMPLATES["illustrious"]
        "wrap": "illustrious",
        "pos_prefix": WRAP_TEMPLATES["illustrious"].split("{prompt}")[0],
        "neg": ("bad quality, worst quality, worst detail, sketch, censor, " + _RATING_NSFW_TOKEN +
                "lowres, jpeg "
                "artifacts, signature, watermark, username, artist name, bad hands, extra digits, "
                "missing limbs, chromatic aberration, blurry, out of focus"),
        "hires_scale": 2.0, "hires_steps": 20, "hires_denoise": 0.35,  # 1024x1536 -> 2048x3072 (BOXB final)
        # advanced (--max) model patches: SDXL UNet techniques.
        "adv": {"freeu": FREEU_DEFAULT, "pag": 3.0, "rescale_cfg": 0.7},
    },
    "anima": {
        "arch": "flow",
        # The quality-line base is the registry's single role=quality_default card (anima_models.json;
        # Dreamland ANM v0.1 since 2026-09-02), not a literal here -- the fast line resolves its own
        # distilled default the same way. --base still overrides for any other Anima checkpoint.
        "ckpt": anima_models.quality_default()["unet"],   # loaded via UNETLoader
        # clip/clip_type/vae come from anima_models.json's shared_contract (verified constant
        # across every known Anima checkpoint) rather than a second hardcoded copy here.
        "clip": _ANIMA_CONTRACT["clip_name"], "clip_type": _ANIMA_CONTRACT["clip_type"],
        "vae": _ANIMA_CONTRACT["vae_name"],
        "sampler": "er_sde", "scheduler": "simple",
        # 30 steps: docs/anima_line/DREAMLAND_ANM_CALIBRATION.md (2026-09-02) saw no difference
        # 30 vs 40 on Dreamland; cfg 4.5 sits between the two tested clean values.
        "steps": 30, "cfg": 4.5, "clip_skip": None, "res": "896x1152",
        # official ComfyUI wiring includes ModelSamplingAuraFlow(shift=3) (anima_comparison.json); must
        # be explicit per model (REPORT.md 2026-07-15: shift=3 vs no-ModelSampling both run, composition
        # stable, fine detail differs) -- override with --model-sampling/--shift.
        "model_sampling": {"mode": "auraflow", "shift": 3.0},
        # masterpiece-family tags cut 2026-07-17 (stage 1 EXP-Q: no quality gain,
        # composition perturbation only); "safe" kept for SFW steering
        "pos_prefix": "safe, ",
        # photorealistic/3d render appended 2026-07-17 (stage 2 EXP-N2: the only negative
        # category with a demonstrated effect — empty-negative arms drift to a 3D-render
        # look on some seeds; cheap seed-dependent insurance)
        "neg": ("worst quality, low quality, early, old, ugly, disfigured, bad anatomy, bad hands, "
                "missing fingers, extra digits, fewer digits, long body, cropped, very displeasing, "
                "artist name, blurry, jpeg artifacts, lowres, censor, photorealistic, 3d render"),
        "hires_scale": 1.5, "hires_steps": 18, "hires_denoise": 0.28,
        # flow-matching: SDXL-specific model patches OFF by default (FreeU/PAG target the SDXL UNet).
        "adv": {"freeu": None, "pag": 0.0, "rescale_cfg": 0.0},
    },
}


# Three-step sampling line. Each role's LoRAs are applied (model-only) at base_strength * ratio in each
# step, where ratio comes from this schedule [step1 character, step2 style+hires, step3 detailer]. A 0
# ratio means the role is absent from that step. Tunable; these are researched defaults, not law.
ROLE_ORDER = ("character", "style", "concept", "pose", "detail")
ROLE_SCHEDULE = {
    "character": (1.0, 0.55, 0.45),  # establish the subject -> keep identity under style -> face identity
    "style":     (0.25, 1.0, 0.35),  # light while the character forms -> dominant during hires reskin
    "concept":   (0.9, 0.7, 0.2),    # manifest the concept at compose -> hold through the hires reskin -> fade at the face detailer
    "pose":      (1.0, 0.5, 0.0),    # compose -> hold -> irrelevant to the face detailer
    "detail":    (0.0, 0.4, 1.0),    # nothing to detail yet -> some -> shines in the detailer's re-sample
}


# Driver primitives (upload_image, png_size), save helpers, asset preflight, and the submit/poll/fetch
# transport loop now live in the shared boxa_comfy module and are imported at the top of this file.


def _facedetailer(image_ref, model_ref, clip_ref, vae_ref, pos_ref, neg_ref, bbox_ref, r, seed,
                  guide_size=512.0, denoise=0.40, sampler=None, scheduler=None, cfg=None):
    """Proven FaceDetailer config (from boxa_gen), retargeted per recipe / per detector."""
    return {"class_type": "FaceDetailer", "inputs": {
        "image": image_ref, "model": model_ref, "clip": clip_ref, "vae": vae_ref,
        "positive": pos_ref, "negative": neg_ref, "guide_size": guide_size, "guide_size_for": True,
        "max_size": 1024.0, "seed": seed + 7, "steps": 20, "cfg": r["cfg"] if cfg is None else cfg,
        "sampler_name": sampler or r["sampler"], "scheduler": scheduler or r["scheduler"],
        "denoise": denoise, "feather": 5,
        "noise_mask": True, "force_inpaint": True, "bbox_threshold": 0.5, "bbox_dilation": 10,
        "bbox_crop_factor": 3.0, "sam_detection_hint": "center-1", "sam_dilation": 0,
        "sam_threshold": 0.93, "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
        "sam_mask_hint_use_negative": "False", "drop_size": 10, "bbox_detector": bbox_ref,
        "wildcard": "", "cycle": 1}}


def adapt_detailer_assets(detailers: list[str] | None = None) -> list[str]:
    """Reconcile the face/eye/hand detector defaults with what the target box actually serves.

    The module defaults name BOXA's detector files; other boxes (BOXB) carry a different subset, and one
    missing UltralyticsDetectorProvider model fails ComfyUI validation for the whole submit. Known-
    equivalent downgrades are applied here BEFORE any graph is built — drop the eye pass, or swap the
    face detector for a bbox/face_* model the box does have — each reported as an `[adapt]` line and
    recorded in the images' meta. An unreachable /object_info probe changes nothing (same fail-open
    stance as preflight_assets). Blanking a detector constant disables that detailer everywhere the
    graph builders consult it. The optional hand pass is adapted only when it is explicitly requested
    in ``detailers`` so the default face/eye path remains unchanged."""
    global FACE_DETECTOR, EYE_DETECTOR, HAND_DETECTOR, FOOT_DETECTOR
    known = object_info_enum("UltralyticsDetectorProvider", "model_name")
    if not known:
        return []
    adapted: list[str] = []
    if EYE_DETECTOR and EYE_DETECTOR not in known:
        adapted.append(f"eye detailer disabled: {EYE_DETECTOR} not on this box")
        EYE_DETECTOR = ""
    if FACE_DETECTOR and FACE_DETECTOR not in known:
        alts = sorted(k for k in known if k.startswith("bbox/face_"))
        if alts:
            adapted.append(f"face detector swapped: {FACE_DETECTOR} -> {alts[0]}")
            FACE_DETECTOR = alts[0]
        else:
            adapted.append(f"face detailer disabled: {FACE_DETECTOR} not on this box, no bbox/face_* alternative")
            FACE_DETECTOR = ""
    if detailers is not None and "hand" in detailers and HAND_DETECTOR and HAND_DETECTOR not in known:
        alts = sorted(k for k in known if k.startswith(("segm/hand_", "bbox/hand_")))
        if alts:
            adapted.append(f"hand detector swapped: {HAND_DETECTOR} -> {alts[0]}")
            HAND_DETECTOR = alts[0]
        else:
            adapted.append(f"hand detailer disabled: {HAND_DETECTOR} not on this box, no segm/hand_* or bbox/hand_* alternative")
            HAND_DETECTOR = ""
    if detailers is not None and "foot" in detailers and FOOT_DETECTOR and FOOT_DETECTOR not in known:
        alts = sorted(k for k in known if k.startswith(("segm/foot_", "bbox/foot_")))
        if alts:
            adapted.append(f"foot detector swapped: {FOOT_DETECTOR} -> {alts[0]}")
            FOOT_DETECTOR = alts[0]
        else:
            adapted.append(f"foot detailer disabled: {FOOT_DETECTOR} not on this box, no segm/foot_* or bbox/foot_* alternative")
            FOOT_DETECTOR = ""
    for line in adapted:
        print(f"  [adapt] {line}")
    return adapted


def resolve_ipadapter_file(mode: str) -> str:
    """Resolve the IPAdapter model name against what the target box actually serves.

    The registered filename differs per box for the same weights (BOXA: 'plus.face.sdxl.vit-h.safetensors',
    BOXB: 'ip-adapter-plus-face_sdxl_vit-h.safetensors'), and a wrong name fails ComfyUI validation
    mid-submit rather than at preflight. Exact configured name wins; otherwise fall back to the box's
    single candidate of the right kind (face = name contains 'face'; 'faceid' variants are excluded —
    they need an InsightFace input this graph does not wire). No usable candidate is a hard stop with
    the fix spelled out; an unreachable probe keeps the configured name (same fail-open stance as
    preflight_assets)."""
    configured = IPADAPTER[mode]["model"]
    known = object_info_enum("IPAdapterModelLoader", "ipadapter_file")
    if not known or configured in known:
        return configured
    want_face = mode in ("face", "both")  # 'both' resolves its face stage here; style via a second call
    candidates = sorted(n for n in known
                        if "faceid" not in n.lower() and ("face" in n.lower()) == want_face)
    if len(candidates) == 1:
        return candidates[0]
    detail = (f"ambiguous fallback {candidates}" if candidates
              else f"no '{mode}' candidate among {sorted(known)}")
    sys.exit(f"ERROR: IPAdapter {mode} model not found on this box ({BOXA_HOST}): configured "
             f"{configured!r} absent, {detail} — install one or use --box boxa")


def resolve_lora_file(name: str, known: set[str], kind: str = "LoRA") -> str:
    """Resolve a LoRA name against the target box's LoraLoader enum (same stance as
    resolve_ipadapter_file): the same weights may register under a different path per box —
    BOXA keeps subfolders, BOXB is partly flat. Exact name wins; otherwise a unique basename
    match swaps in this box's registered name; an ambiguous basename is a hard stop (never
    guess between two files); no match returns the name unchanged so preflight_assets
    reports it with its existing message. An empty enum (probe failed) changes nothing.
    ``kind`` only labels the ambiguity error (run() reuses this rule for the checkpoint)."""
    if not known or name in known:
        return name
    base = name.replace("\\", "/").split("/")[-1].lower()
    candidates = sorted(n for n in known if n.replace("\\", "/").split("/")[-1].lower() == base)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        sys.exit(f"ERROR: {kind} {name!r} not on this box ({BOXA_HOST}) and its filename matches "
                 f"{len(candidates)} entries there: {candidates} — pass the exact registered name")
    return name


def _resolve_adv(family: str, opts: dict) -> dict:
    """Merge per-family advanced defaults with CLI overrides."""
    adv = RECIPES[family]["adv"]
    freeu_on = bool(adv["freeu"]) if opts.get("freeu") is None else opts["freeu"]
    return {
        "freeu": (adv["freeu"] or FREEU_DEFAULT) if freeu_on else None,
        "pag": adv["pag"] if opts.get("pag") is None else opts["pag"],
        "rescale_cfg": adv["rescale_cfg"] if opts.get("rescale") is None else opts["rescale"],
        "detailers": opts.get("detailers", ["face", "eye"]),
        "upscale_by": opts.get("upscale_by") or RECIPES[family]["hires_scale"],
    }


def build_graph(family: str, ckpt: str, pos: str, neg: str, seed: int, w: int, h: int,
                content_loras: list[tuple[str, float]], advanced: bool, opts: dict,
                control: dict | None = None, ip: dict | None = None,
                init_name: str | None = None, init_denoise: float = 1.0) -> dict:
    r = RECIPES[family]
    g: dict = {}
    samp = opts.get("sampler") or r["sampler"]
    sched = opts.get("scheduler") or r["scheduler"]
    # `or` is wrong for numeric overrides (a real --cfg 0 would be silently discarded); steps/cfg
    # both use `is not None` unlike the string sampler/scheduler above.
    steps = opts["steps"] if opts.get("steps") is not None else r["steps"]
    cfg = opts["cfg"] if opts.get("cfg") is not None else r["cfg"]

    # --- load model / clip / vae (+ optional LoRA) ---
    if r["arch"] == "sdxl":
        g["4"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
        model_ref, clip_src = ["4", 0], ["4", 1]
        for i, (lname, lstr) in enumerate(content_loras):
            nid = f"15_{i}"
            g[nid] = {"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_src, "lora_name": lname,
                "strength_model": lstr, "strength_clip": lstr}}
            model_ref, clip_src = [nid, 0], [nid, 1]
        g["10"] = {"class_type": "CLIPSetLastLayer",
                   "inputs": {"clip": clip_src, "stop_at_clip_layer": r["clip_skip"]}}
        clip_ref = ["10", 0]
        # External VAE for BOTH base and --max. The BOXB workflow decodes with the checkpoint's baked VAE,
        # but BOXA's ilustmix_v80 baked VAE decodes severely washed-out (verified 2026-06-25) — which is
        # exactly why this recipe pins the external sdxl_vae_fp16_fix. The baked-VAE detail of the BOXB
        # spec does not port to this box.
        g["40"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["40", 0]
    else:  # flow-matching Anima (official ComfyUI wiring)
        g["44"] = {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}}
        model_ref = ["44", 0]
        for i, (lname, lstr) in enumerate(content_loras):
            nid = f"46_{i}"
            g[nid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model_ref, "lora_name": lname, "strength_model": lstr}}
            model_ref = [nid, 0]
        # ModelSamplingAuraFlow is optional and per-model (REPORT.md 2026-07-15: some Anima checkpoints'
        # author recipe omits it entirely). opts override wins; falls back to the recipe's own card.
        # Structurally confined to this flow-matching branch -- the sdxl branch above never reaches here,
        # so RECIPES["illustrious"] does not need (and must not gain) a "model_sampling" key.
        ms_cfg = r.get("model_sampling") or {}
        ms_mode = opts.get("model_sampling") or ms_cfg.get("mode")
        ms_shift = opts.get("shift") if opts.get("shift") is not None else ms_cfg.get("shift")
        if ms_mode == "auraflow" and ms_shift is not None:
            g["47"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_ref, "shift": ms_shift}}
            model_ref = ["47", 0]
        g["45"] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": r["clip"], "type": r["clip_type"], "device": "default"}}
        clip_ref = ["45", 0]
        g["15"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["15", 0]

    # --- IPAdapter (SDXL only): image-prompt the model from a reference image ---
    if ip and r["arch"] == "sdxl":
        spec = IPADAPTER[ip["mode"]]
        g["90"] = {"class_type": "LoadImage", "inputs": {"image": ip["image_name"], "upload": "image"}}
        # Center-crop the reference to a square. IPAdapterPlus prints a non-square warning, and on this
        # BOXA (ComfyUI Desktop) any stdout print crashes with OSError[22] in the logger flush — squaring
        # the input avoids the print entirely.
        g["94"] = {"class_type": "ImageScale", "inputs": {
            "image": ["90", 0], "upscale_method": "lanczos", "width": 768, "height": 768, "crop": "center"}}
        g["91"] = {"class_type": "IPAdapterModelLoader",
                   "inputs": {"ipadapter_file": ip.get("model", spec["model"])}}
        g["92"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": spec["clip_vision"]}}
        g["93"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": model_ref, "ipadapter": ["91", 0], "image": ["94", 0], "weight": ip["weight"],
            "weight_type": ip.get("weight_type", spec["weight_type"]), "combine_embeds": "concat", "start_at": 0.0,
            "end_at": ip.get("end", 1.0), "embeds_scaling": ip.get("scaling", "V only"),
            "clip_vision": ["92", 0]}}
        model_ref = ["93", 0]
        if ip["mode"] == "both":
            # Second stage: full-range style transfer chained on the face lock. Shares
            # the squared reference (94) and CLIP-Vision (92); only the adapter file,
            # weight_type, and range differ.
            g["95"] = {"class_type": "IPAdapterModelLoader",
                       "inputs": {"ipadapter_file": ip.get("style_model",
                                                           IPADAPTER["both"]["style_model"])}}
            g["96"] = {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": model_ref, "ipadapter": ["95", 0], "image": ["94", 0],
                "weight": ip.get("style_weight", IPADAPTER["both"]["style_weight"]),
                "weight_type": "style transfer", "combine_embeds": "concat", "start_at": 0.0,
                "end_at": 1.0, "embeds_scaling": "V only", "clip_vision": ["92", 0]}}
            model_ref = ["96", 0]
    elif ip and r["arch"] == "flow":
        # ids 97-99: 90 is the pose_lora VAEEncode
        g["97"] = {"class_type": "LoadImage", "inputs": {"image": ip["image_name"], "upload": "image"}}
        g["98"] = {"class_type": "AnimaIPAdapterLoader",
                   "inputs": {"ip_adapter_name": ANIMA_IPADAPTER["file"], "auto_download": False}}
        g["99"] = {"class_type": "AnimaIPAdapterApply", "inputs": {
            "model": model_ref, "ip_adapter": ["98", 0], "ref_image": ["97", 0], "strength": ip["weight"],
            **{k: ANIMA_IPADAPTER[k] for k in ("ref_image_size", "siglip_layer", "ip_cfg_scale",
                                               "ip_cfg_separate", "gray_null", "use_lora")}}}
        model_ref = ["99", 0]

    adv = _resolve_adv(family, opts) if advanced else None

    # --- advanced model patches (chain on the model; SDXL UNet techniques) — --max only ---
    if adv:
        nid = 50
        if adv["freeu"]:
            b1, b2, s1, s2 = adv["freeu"]
            g[str(nid)] = {"class_type": "FreeU_V2", "inputs": {
                "model": model_ref, "b1": b1, "b2": b2, "s1": s1, "s2": s2}}
            model_ref = [str(nid), 0]; nid += 1
        if adv["pag"] > 0:
            g[str(nid)] = {"class_type": "PerturbedAttentionGuidance",
                           "inputs": {"model": model_ref, "scale": adv["pag"]}}
            model_ref = [str(nid), 0]; nid += 1
        if adv["rescale_cfg"] > 0:
            g[str(nid)] = {"class_type": "RescaleCFG",
                           "inputs": {"model": model_ref, "multiplier": adv["rescale_cfg"]}}
            model_ref = [str(nid), 0]; nid += 1

    # --- conditioning + base sample ---
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": pos}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": neg}}
    # base latent: img2img from an init image (promote a draft) or a fresh empty latent
    if init_name:
        g["43"] = {"class_type": "LoadImage", "inputs": {"image": init_name, "upload": "image"}}
        g["42"] = {"class_type": "ImageScale", "inputs": {"image": ["43", 0], "upscale_method": "lanczos",
                                                          "width": w, "height": h, "crop": "center"}}
        g["41"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["42", 0], "vae": vae_ref}}
        latent_ref = ["41", 0]
    else:
        g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        latent_ref = ["5", 0]

    # --- ControlNet (SDXL): preprocess the control image, apply to the base conditioning.
    # --- Anima control (flow): a MODEL patch on model_ctrl_ref, wired to the base KSampler only.
    pos_s, neg_s = ["6", 0], ["7", 0]
    model_ctrl_ref = model_ref
    if control and r["arch"] == "sdxl":
        spec = CONTROLNET[control["type"]]
        g["80"] = {"class_type": "LoadImage", "inputs": {"image": control["image_name"], "upload": "image"}}
        cn_img = ["80", 0]
        if control["preprocess"] and spec["pre"]:
            g["81"] = _preprocess_node(spec["pre"], ["80", 0])
            cn_img = ["81", 0]
        g["82"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": spec["model"]}}
        g["83"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "control_net": ["82", 0], "image": cn_img,
            "strength": control["strength"], "start_percent": 0.0, "end_percent": control["end"]}}
        pos_s, neg_s = ["83", 0], ["83", 1]
    elif control and r["arch"] == "flow":
        spec = ANIMA_CONTROL[control["type"]]
        node = control.get("node") or ANIMA_CONTROL_NODES[spec["kind"]][0]
        g["80"] = {"class_type": "LoadImage", "inputs": {"image": control["image_name"], "upload": "image"}}
        cn_img = ["80", 0]
        if control["preprocess"] and spec["pre"]:
            g["81"] = _preprocess_node(spec["pre"], cn_img)
            cn_img = ["81", 0]
            if spec.get("invert"):
                g["88"] = {"class_type": "ImageInvert", "inputs": {"image": cn_img}}
                cn_img = ["88", 0]
        # the patch conditioning / control latent must match the render canvas exactly
        g["89"] = {"class_type": "ImageScale", "inputs": {
            "image": cn_img, "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}}
        cn_img = ["89", 0]
        if spec["kind"] == "lllite":
            if node == REMAP_LLLITE_NODE:
                g["85"] = {"class_type": REMAP_LLLITE_NODE, "inputs": {
                    "model": model_ref, "lllite_name": spec["file"], "image": cn_img,
                    "strength": control["strength"], "start_percent": 0.0, "end_percent": control["end"]}}
            elif node == "AnimaLLLiteApply":
                g["84"] = {"class_type": "ModelPatchLoader", "inputs": {"name": spec["file"]}}
                g["85"] = {"class_type": "AnimaLLLiteApply", "inputs": {
                    "model": model_ref, "model_patch": ["84", 0], "image": cn_img,
                    "strength": control["strength"], "start_percent": 0.0, "end_percent": control["end"]}}
            elif node == "AnimaLLLiteApply_sdscripts":
                g["85"] = {"class_type": "AnimaLLLiteApply_sdscripts", "inputs": {
                    "model": model_ref, "lllite_name": spec["file"], "image": cn_img,
                    "strength": control["strength"], "start_percent": 0.0, "end_percent": control["end"],
                    "preserve_wrapper": True}}
            else:
                raise ValueError(f"unknown Anima LLLite node {node!r}")
            model_ctrl_ref = ["85", 0]
        else:  # control_lora: LoRA deltas + zero-init ControlEmbedder fed a VAE-encoded skeleton
            g["86"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model_ref, "lora_name": spec["file"], "strength_model": control["strength"]}}
            g["90"] = {"class_type": "VAEEncode", "inputs": {"pixels": cn_img, "vae": vae_ref}}
            g["87"] = {"class_type": "AnimaControlApply", "inputs": {
                "model": ["86", 0], "control_latent": ["90", 0], "control_embedder_path": spec["file"],
                "strength": control["strength"]}}
            model_ctrl_ref = ["87", 0]

    # Keep the historical max/refine behaviour by default: the base pass uses ControlNet, while
    # later diffusion sees the plain text conditioning.  Complex skeletons can drift back toward
    # a checkpoint's upright prior during that second pass, so callers may opt in to carrying the
    # same ControlNet conditioning through refine.  This is deliberately per-run, not a global
    # default change, because some finishing workflows want late denoise freedom.
    refine_pos, refine_neg = ([pos_s, neg_s] if control and control.get("refine")
                              else [["6", 0], ["7", 0]])
    # Anima control is a MODEL patch, so the same opt-in decides which model every later
    # diffusion stage sees: the upscale resample and the face / eye / hand / foot detailers alike.
    refine_model = model_ctrl_ref if control and control.get("refine") else model_ref

    g["3"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": samp,
        "scheduler": sched, "denoise": init_denoise, "model": model_ctrl_ref,
        "positive": pos_s, "negative": neg_s, "latent_image": latent_ref}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_ref}}
    out = ["8", 0]

    # --- post: image-space upscale. base mode = BOXB-style pure ESRGAN upscale -> downscale (no 2nd
    # pass); --max adds a comprehensive 2nd-pass diffusion refine (or opt-in UltimateSDUpscale). ---
    if adv or HIRES == "full":
        scale = adv["upscale_by"] if adv else r["hires_scale"]
        g["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
        if adv and opts.get("ultimate_upscale"):
            # --max opt-in: tiled upscale + resample in one node (memory-friendly vs the full-frame pass).
            g["31"] = {"class_type": "UltimateSDUpscale", "inputs": {
                "image": out, "model": refine_model, "positive": refine_pos, "negative": refine_neg,
                "vae": vae_ref, "upscale_model": ["30", 0], "upscale_by": scale, "seed": seed + 1,
                "steps": r["hires_steps"], "cfg": cfg, "sampler_name": samp, "scheduler": sched,
                "denoise": r["hires_denoise"], "mode_type": "Linear", "tile_width": 1024,
                "tile_height": 1024, "mask_blur": 8, "tile_padding": 32, "seam_fix_mode": "None",
                "seam_fix_denoise": 1.0, "seam_fix_width": 64, "seam_fix_mask_blur": 8,
                "seam_fix_padding": 16, "force_uniform_tiles": True, "tiled_decode": False, "batch_size": 1}}
            out = ["31", 0]
        else:
            sw, sh = int(w * scale), int(h * scale)
            g["31"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["30", 0], "image": out}}
            g["32"] = {"class_type": "ImageScale", "inputs": {
                "image": ["31", 0], "upscale_method": "lanczos", "width": sw, "height": sh, "crop": "disabled"}}
            out = ["32", 0]
            if adv:
                # --max only: 2nd-pass diffusion refine over the upscaled image (base/BOXB skips this).
                g["33"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["32", 0], "vae": vae_ref}}
                g["34"] = {"class_type": "KSampler", "inputs": {
                    "seed": seed + 1, "steps": r["hires_steps"], "cfg": cfg, "sampler_name": samp,
                    "scheduler": sched, "denoise": r["hires_denoise"], "model": refine_model,
                    "positive": refine_pos, "negative": refine_neg, "latent_image": ["33", 0]}}
                g["35"] = {"class_type": "VAEDecode", "inputs": {"samples": ["34", 0], "vae": vae_ref}}
                out = ["35", 0]

    # --- post: detailer chain (face -> eyes). Always run when --max, AND auto-run when ControlNet or
    # IPAdapter is active (both degrade small faces — a face/eye detailer pass is mandatory there to
    # avoid broken faces), or under a BOXA_HIRES tier. ---
    detailers = (adv["detailers"] if adv else
                 (["face", "eye"] if (control or ip) else
                  (["face"] if HIRES in ("face", "full") else [])))
    if "face" in detailers and FACE_DETECTOR:
        g["70"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DETECTOR}}
        g["71"] = _facedetailer(out, refine_model, clip_ref, vae_ref, ["6", 0], ["7", 0], ["70", 0],
                                r, seed, guide_size=512.0, denoise=0.40, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["71", 0]
    if "eye" in detailers and EYE_DETECTOR:
        g["72"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": EYE_DETECTOR}}
        g["73"] = _facedetailer(out, refine_model, clip_ref, vae_ref, ["6", 0], ["7", 0], ["72", 0],
                                r, seed, guide_size=256.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["73", 0]
    if "hand" in detailers and HAND_DETECTOR:
        g["74"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DETECTOR}}
        g["75"] = _facedetailer(out, refine_model, clip_ref, vae_ref, ["6", 0], ["7", 0], ["74", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["75", 0]
    # optional foot SEGM detailer — opt-in only; do not treat as guaranteed anatomy fix
    if "foot" in detailers and FOOT_DETECTOR:
        g["76"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FOOT_DETECTOR}}
        g["77"] = _facedetailer(out, refine_model, clip_ref, vae_ref, ["6", 0], ["7", 0], ["76", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["77", 0]

    g["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": "boxa_quality"}}
    return g


def build_refine_graph(family: str, ckpt: str, pos: str, neg: str, seed: int,
                       content_loras: list[tuple[str, float]], opts: dict, init_name: str,
                       target_w: int, target_h: int, refine_steps: int, refine_denoise: float,
                       detailers: list[str]) -> dict:
    """Standalone refine pass: load an image -> model-upscale -> resize to target -> low-denoise
    2nd-pass KSampler -> optional face/eye detailer. No first txt2img pass, no FreeU/PAG/IPAdapter/
    ControlNet — only nodes already proven on BOXA in the --max path. The loader block mirrors
    build_graph's (kept inline so the BOXA-verified build_graph stays untouched)."""
    r = RECIPES[family]
    g: dict = {}
    samp = opts.get("sampler") or r["sampler"]
    sched = opts.get("scheduler") or r["scheduler"]

    # --- load model / clip / vae (+ optional content LoRAs); mirror of build_graph's loader ---
    if r["arch"] == "sdxl":
        g["4"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
        model_ref, clip_src = ["4", 0], ["4", 1]
        for i, (lname, lstr) in enumerate(content_loras):
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
    else:  # flow-matching Anima
        g["44"] = {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}}
        model_ref = ["44", 0]
        for i, (lname, lstr) in enumerate(content_loras):
            nid = f"46_{i}"
            g[nid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model_ref, "lora_name": lname, "strength_model": lstr}}
            model_ref = [nid, 0]
        g["45"] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": r["clip"], "type": r["clip_type"], "device": "default"}}
        clip_ref = ["45", 0]
        g["15"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["15", 0]

    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": pos}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": neg}}

    # --- load image -> GAN model-upscale -> resize to absolute target -> encode ---
    g["80"] = {"class_type": "LoadImage", "inputs": {"image": init_name, "upload": "image"}}
    g["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["31"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["30", 0], "image": ["80", 0]}}
    g["32"] = {"class_type": "ImageScale", "inputs": {
        "image": ["31", 0], "upscale_method": "lanczos", "width": target_w, "height": target_h, "crop": "disabled"}}
    g["33"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["32", 0], "vae": vae_ref}}

    # --- low-denoise 2nd-pass refine ---
    g["34"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": refine_steps, "cfg": r["cfg"], "sampler_name": samp,
        "scheduler": sched, "denoise": refine_denoise, "model": model_ref,
        "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["33", 0]}}
    g["35"] = {"class_type": "VAEDecode", "inputs": {"samples": ["34", 0], "vae": vae_ref}}
    out = ["35", 0]

    # --- optional detailer chain (face -> eyes), same configs as the --max path ---
    if "face" in detailers and FACE_DETECTOR:
        g["70"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DETECTOR}}
        g["71"] = _facedetailer(out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0], ["70", 0],
                                r, seed, guide_size=512.0, denoise=0.40, sampler=samp, scheduler=sched)
        out = ["71", 0]
    if "eye" in detailers and EYE_DETECTOR:
        g["72"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": EYE_DETECTOR}}
        g["73"] = _facedetailer(out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0], ["72", 0],
                                r, seed, guide_size=256.0, denoise=0.35, sampler=samp, scheduler=sched)
        out = ["73", 0]
    if "hand" in detailers and HAND_DETECTOR:
        g["74"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DETECTOR}}
        g["75"] = _facedetailer(out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0], ["74", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched)
        out = ["75", 0]
    if "foot" in detailers and FOOT_DETECTOR:
        g["76"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FOOT_DETECTOR}}
        g["77"] = _facedetailer(out, model_ref, clip_ref, vae_ref, ["6", 0], ["7", 0], ["76", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched)
        out = ["77", 0]

    g["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": "boxa_refine"}}
    return g


def build_3step_graph(family: str, ckpt: str, pos: str, neg: str, seed: int,
                      role_loras: dict, opts: dict, base_w: int, base_h: int,
                      target_w: int, target_h: int, step2_denoise: float,
                      detailers: list[str]) -> dict:
    """Role-aware three-step sampling for stacking heterogeneous LoRAs without a single overcooked pass:
      1. character: full-denoise txt2img (character/pose LoRAs dominant) — establish the subject.
      2. style + hires: model-upscale the step-1 image, then a low-denoise resample with the step-2 chain
         (style LoRA dominant, character reduced to keep identity) — confirm the look at high res.
      3. detailer: face/eye FaceDetailer with the step-3 chain (detail LoRAs dominant) — final refinement.
    Each step gets its own model-only LoRA chain (strength = base * ROLE_SCHEDULE[role][step]); clip and
    vae are shared (one conditioning). LoRAs are applied model-only so the three chains stay independent."""
    r = RECIPES[family]
    g: dict = {}
    samp = opts.get("sampler") or r["sampler"]
    sched = opts.get("scheduler") or r["scheduler"]
    # Same precedence as build_graph: CLI flag > card recipe (apply_sampling_defaults) > RECIPES.
    # steps drives the step-1 base pass only (step 2 keeps hires_steps, the detailers their own 20);
    # cfg is the one shared value across all three steps.
    steps = opts["steps"] if opts.get("steps") is not None else r["steps"]
    cfg = opts["cfg"] if opts.get("cfg") is not None else r["cfg"]

    # base loaders — clip + vae shared across all steps
    if r["arch"] == "sdxl":
        g["4"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
        base_model = ["4", 0]
        g["10"] = {"class_type": "CLIPSetLastLayer",
                   "inputs": {"clip": ["4", 1], "stop_at_clip_layer": r["clip_skip"]}}
        clip_ref = ["10", 0]
        g["40"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["40", 0]
    else:  # flow-matching Anima
        g["44"] = {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}}
        base_model = ["44", 0]
        g["45"] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": r["clip"], "type": r["clip_type"], "device": "default"}}
        clip_ref = ["45", 0]
        g["15"] = {"class_type": "VAELoader", "inputs": {"vae_name": r["vae"]}}
        vae_ref = ["15", 0]

    def step_model(step_idx: int, prefix: str):
        """Build the per-step model chain: each active role's LoRAs at base * schedule ratio."""
        m = base_model
        i = 0
        for role in ROLE_ORDER:
            ratio = ROLE_SCHEDULE[role][step_idx]
            if ratio <= 0:
                continue
            for fname, base_str in role_loras.get(role, []):
                nid = f"{prefix}_{i}"
                i += 1
                g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "model": m, "lora_name": fname, "strength_model": round(float(base_str) * ratio, 3)}}
                m = [nid, 0]
        return m

    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": pos}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": neg}}

    # --- step 1: character / composition (txt2img, full denoise) ---
    model1 = step_model(0, "L1")
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": base_w, "height": base_h, "batch_size": 1}}
    g["3"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": samp, "scheduler": sched,
        "denoise": 1.0, "model": model1, "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_ref}}
    out = ["8", 0]

    # --- step 2: style + hires (model-upscale, low-denoise resample with the step-2 chain) ---
    model2 = step_model(1, "L2")
    g["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
    g["31"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["30", 0], "image": out}}
    g["32"] = {"class_type": "ImageScale", "inputs": {
        "image": ["31", 0], "upscale_method": "lanczos", "width": target_w, "height": target_h, "crop": "disabled"}}
    g["33"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["32", 0], "vae": vae_ref}}
    g["34"] = {"class_type": "KSampler", "inputs": {
        "seed": seed + 1, "steps": r["hires_steps"], "cfg": cfg, "sampler_name": samp,
        "scheduler": sched, "denoise": step2_denoise, "model": model2,
        "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["33", 0]}}
    g["35"] = {"class_type": "VAEDecode", "inputs": {"samples": ["34", 0], "vae": vae_ref}}
    out = ["35", 0]

    # --- step 3: detailer chain (detail LoRAs dominant via the step-3 model) ---
    model3 = step_model(2, "L3")
    if "face" in detailers and FACE_DETECTOR:
        g["70"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DETECTOR}}
        g["71"] = _facedetailer(out, model3, clip_ref, vae_ref, ["6", 0], ["7", 0], ["70", 0],
                                r, seed, guide_size=512.0, denoise=0.40, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["71", 0]
    if "eye" in detailers and EYE_DETECTOR:
        g["72"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": EYE_DETECTOR}}
        g["73"] = _facedetailer(out, model3, clip_ref, vae_ref, ["6", 0], ["7", 0], ["72", 0],
                                r, seed, guide_size=256.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["73", 0]
    if "hand" in detailers and HAND_DETECTOR:
        g["74"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DETECTOR}}
        g["75"] = _facedetailer(out, model3, clip_ref, vae_ref, ["6", 0], ["7", 0], ["74", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["75", 0]
    if "foot" in detailers and FOOT_DETECTOR:
        g["76"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FOOT_DETECTOR}}
        g["77"] = _facedetailer(out, model3, clip_ref, vae_ref, ["6", 0], ["7", 0], ["76", 0],
                                r, seed, guide_size=384.0, denoise=0.35, sampler=samp, scheduler=sched, cfg=cfg)
        out = ["77", 0]

    g["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": "boxa_3step"}}
    return g


def _unique_dest(directory: Path, filename: str, *, taken: set[str] | None = None) -> Path:
    """Return directory/filename, or a _2/_3… sibling that is free on disk and not in `taken`.

    First output of a job keeps the historical stem so existing callers still resolve
    the original name. Extra history images and a pre-existing same-stem
    file get a numeric suffix instead of silently clobbering.
    """
    claimed = taken if taken is not None else set()
    path = Path(directory) / filename
    stem, suffix = path.stem, path.suffix
    n = 2
    while path.name in claimed or path.exists() or Path(str(path) + ".meta.json").exists():
        path = Path(directory) / f"{stem}_{n}{suffix}"
        n += 1
    return path


def run(family: str, ckpt: str, pos: str, neg: str, n: int, base_seed: int, w: int, h: int,
        content_loras: list[tuple[str, float]], advanced: bool, opts: dict,
        control: dict | None = None, ip: dict | None = None,
        init_name: str | None = None, init_denoise: float = 1.0, label: str = "",
        engine: WildcardEngine | None = None, refine: dict | None = None,
        three_step: dict | None = None, extra_meta: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = RECIPES[family]
    known_loras = object_info_enum("LoraLoader", "lora_name")
    lora_swaps: list[str] = []
    # The checkpoint follows the same per-box naming rule as LoRAs: anima_models.json records
    # different UNETLoader paths on BOXA and BOXB (unet_by_box), and a draft's sidecar carries the
    # name of the box that made it.
    loader = ("UNETLoader", "unet_name") if r["arch"] == "flow" else ("CheckpointLoaderSimple", "ckpt_name")
    box_ckpt = resolve_lora_file(ckpt, object_info_enum(*loader), kind="checkpoint")
    if box_ckpt != ckpt:
        lora_swaps.append(f"checkpoint swapped: {ckpt} -> {box_ckpt}")
        print(f"  [adapt] {lora_swaps[-1]}")
        ckpt = box_ckpt

    def _resolve_pairs(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        out = []
        for fname, strength in pairs:
            res = resolve_lora_file(fname, known_loras)
            if res != fname:
                lora_swaps.append(f"lora swapped: {fname} -> {res}")
                print(f"  [adapt] {lora_swaps[-1]}")
            out.append((res, strength))
        return out

    content_loras = _resolve_pairs(content_loras)
    if three_step:
        three_step = {**three_step,
                      "role_loras": {role: _resolve_pairs(lst)
                                     for role, lst in three_step["role_loras"].items()}}
    pre_loras = [f for f, _ in content_loras]
    if three_step:
        for _lst in three_step["role_loras"].values():
            pre_loras += [f for f, _ in _lst]
    cn_models: list[str] = []
    patch_models: list[str] = []
    if control and r["arch"] == "sdxl":
        cn_models = [CONTROLNET[control["type"]]["model"]]
    elif control:
        spec = ANIMA_CONTROL[control["type"]]
        if spec["kind"] == "control_lora":
            pre_loras.append(spec["file"])
        elif control.get("node") == REMAP_LLLITE_NODE:
            preflight_remap_lllite(spec["file"])   # the node enumerates model_patches/ + controlnet/
        elif control.get("node") == "AnimaLLLiteApply_sdscripts":
            cn_models = [spec["file"]]      # the custom node enumerates models/controlnet/
        else:
            patch_models = [spec["file"]]   # core ModelPatchLoader enumerates models/model_patches/
    preflight_assets(r["arch"], ckpt, pre_loras,
                     controlnets=cn_models, model_patches=patch_models,
                     upscalers=[UPSCALER] if (advanced or HIRES != "off" or refine or three_step) else (),
                     lora_enums=known_loras)
    adapted = (adapt_detailer_assets(opts["detailers"]) if "detailers" in opts
               else adapt_detailer_assets())
    adapted += lora_swaps
    if ip and r["arch"] == "flow":
        preflight_anima_ipadapter()
    elif ip:
        resolved = resolve_ipadapter_file(ip["mode"])
        if resolved != IPADAPTER[ip["mode"]]["model"]:
            adapted.append(f"ipadapter {ip['mode']} model swapped: "
                           f"{IPADAPTER[ip['mode']]['model']} -> {resolved}")
            print(f"  [adapt] {adapted[-1]}")
        ip = {**ip, "model": resolved}
        if ip["mode"] == "both":
            style_resolved = resolve_ipadapter_file("style")
            if style_resolved != IPADAPTER["both"]["style_model"]:
                adapted.append(f"ipadapter style model swapped: "
                               f"{IPADAPTER['both']['style_model']} -> {style_resolved}")
                print(f"  [adapt] {adapted[-1]}")
            ip = {**ip, "style_model": style_resolved}
    samp = opts.get("sampler") or r["sampler"]
    base_stem = ckpt.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    if three_step:
        nrole = sum(len(v) for v in three_step["role_loras"].values())
        tag = f"3step_d{three_step['step2_denoise']}_l{nrole}"
    elif refine:
        tag = f"refine_u{refine['upscale_by']}_d{refine['denoise']}"
    else:
        tag = "max" if advanced else (HIRES if HIRES != "off" else "base")
        if control:
            tag += f"_{'cnl' if r['arch'] == 'flow' else 'cn'}-{control['type']}"
        if ip:
            tag += f"_ip{ip['mode']}{ip['weight']}"
        if init_name:
            tag += f"_i2i{init_denoise}"
        if content_loras:
            tag += f"_lora{len(content_loras)}"
    if label:
        tag += f"_{label}"
    ok = 0
    try:
        for i in range(n):
            seed = base_seed + i
            pos_i = engine.expand(pos, seed) if engine else pos
            if three_step:
                graph = build_3step_graph(family, ckpt, pos_i, neg, seed, three_step["role_loras"], opts,
                                          w, h, three_step["target"][0], three_step["target"][1],
                                          three_step["step2_denoise"], three_step["detailers"])
            elif refine:
                graph = build_refine_graph(family, ckpt, pos_i, neg, seed, content_loras, opts,
                                           refine["init_name"], refine["target"][0], refine["target"][1],
                                           refine["steps"], refine["denoise"], refine["detailers"])
            else:
                graph = build_graph(family, ckpt, pos_i, neg, seed, w, h, content_loras, advanced, opts,
                                    control, ip, init_name, init_denoise)
            expected_name = f"boxaq_{family}_{base_stem}_{samp}_{tag}_{i:02d}_{seed}.png"
            fetched = _submit_and_fetch(graph, label=f"{family}/{tag} #{i} seed={seed}", first_only=False,
                                        allow_preview=True)
            if not fetched:
                continue
            taken: set[str] = set()
            for raw, item in fetched:
                dest = _unique_dest(OUT, expected_name, taken=taken)
                taken.add(dest.name)
                adv = _resolve_adv(family, opts) if advanced else None
                eff = effective_sampling(family, opts, three_step=bool(three_step))
                meta = {"driving_model": MODEL, "gen_engine": "comfyui:boxa", "host": BOXA_HOST,
                        "family": family, "arch": r["arch"], "base_ckpt": ckpt,
                        "vae": r["vae"], "content_loras": [{"file": f, "strength": s} for f, s in content_loras],
                        "prompt": pos_i, "negative": neg, "seed": seed,
                        "sampler": f"{eff['sampler']}/{eff['scheduler']}", "scheduler": eff["scheduler"],
                        "steps": eff["steps"], "cfg": eff["cfg"], "model_sampling": eff["model_sampling"],
                        "clip_skip": r["clip_skip"], "resolution": f"{w}x{h}",
                        "advanced": advanced, "hires_mode": (None if advanced else HIRES),
                        "adv": adv, "control": control, "ipadapter": ip, "refine": refine,
                        "three_step": three_step, "adapted": adapted or None,
                        "source_image": dest.name}
                if item.get("fetched_preview"):  # only the preview crossed the link; the PNG stays on the box
                    meta["remote_original"] = {k: item.get(k) for k in ("box", "filename", "subfolder", "type")}
                    meta["fetched_preview"] = item["fetched_preview"]
                # Lineage from a caller that knows this image is a revision of another one
                # (parent_image, revision_id, changed/frozen dimensions, minimal_delta). It is
                # merged here rather than written by a second pass afterwards so the picture and
                # the record of why it exists are produced by the same write -- a later pass that
                # failed would leave an image whose freeze certificate silently went missing.
                meta.update(extra_meta or {})
                _save_image_with_meta(dest, raw, meta, overwrite=False)
                print(f"    -> {dest.name} {len(raw) // 1024}KB"
                      + (f"\n       prompt: {pos_i}" if pos_i != pos else "")); ok += 1
    finally:
        if ip and r["arch"] == "flow":
            print("  free: Anima IP-Adapter patched the cached DiT in place; POST /free with "
                  "free_memory to reset the node cache so later renders on this base are clean")
            if not free_vram(BOXA_HOST, clear_cache=True):
                print("  warn: cache-clearing /free failed; run `freeboxa` (full) before the next "
                      "render on this base", file=sys.stderr)
    print(f"done: {ok} image(s) -> {OUT}")





def _resolve_loras(args: argparse.Namespace, arch: str) -> tuple[list[tuple[str, float]], str]:
    """Resolve content LoRAs (manual --lora + semantic --from-context) with an arch guard;
    return (loras, trigger_text). flow family requires Anima LoRAs; sdxl rejects Anima,
    SD 1.5, and video/non-SDXL-family LoRAs."""
    loras: list[tuple[str, float]] = []
    records: list[dict] = []
    for spec in (args.lora or []):
        fname, strength = parse_spec(spec, args.strength)
        loras.append((fname, float(strength)))
        rec = record_of(fname)
        if rec:
            records.append(rec)
    if args.from_context:
        base = "Anima" if arch == "flow" else None
        picks = pick(args.from_context, base=base, k=args.lora_k, min_score=args.min_score)
        if not picks:
            print(f" no LoRA scored >= {args.min_score} for '{args.from_context}'")
        for rec in picks:
            strength = float(rec.get("strength_default") or args.strength)
            loras.append((rec["file"], strength))
            records.append(rec)
            print(f" picked: {rec['file']} score={rec['score']} str={strength}")
    if not args.no_arch_guard:
        want_anima = arch == "flow"
        for fname, _ in loras:
            bm = base_model_of(fname)
            if bm is None:
                print(f" warn: {fname} not in catalog — cannot verify base-model compatibility")
            elif want_anima and not _mr.family_compatible(bm, "anima"):
                sys.exit(f"ERROR: LoRA {fname} is base_model={bm!r}, not Anima (flow-matching "
                         f"incompatible). Use --no-arch-guard to override.")
            elif not want_anima and not _mr.family_compatible(bm, "sdxl"):
                sys.exit(f"ERROR: LoRA {fname} is base_model={bm!r}, incompatible with the SDXL "
                         f"family. Use --no-arch-guard to override.")
    trig = "" if args.no_trigger_inject else triggers_for(records)
    return loras, trig


def _resolve_role_loras(args: argparse.Namespace, arch: str) -> tuple[dict, str]:
    """Resolve the role->LoRA map for the three-step line (--char/style/pose/detail-lora) with the
    arch guard; return (role_loras, trigger_text). role_loras[role] = [[file, base_strength], ...]."""
    role_specs = {"character": args.char_lora, "style": args.style_lora,
                  "concept": args.concept_lora, "pose": args.pose_lora, "detail": args.detail_lora}
    role_loras: dict[str, list] = {}
    records: list[dict] = []
    for role, specs in role_specs.items():
        lst: list = []
        for spec in (specs or []):
            fname, strength = parse_spec(spec, args.strength)
            lst.append([fname, float(strength)])
            rec = record_of(fname)
            if rec:
                records.append(rec)
        if lst:
            role_loras[role] = lst
    if not args.no_arch_guard:
        want_anima = arch == "flow"
        for role, lst in role_loras.items():
            for fname, _ in lst:
                bm = base_model_of(fname)
                if bm is None:
                    print(f" warn: {fname} ({role}) not in catalog — cannot verify base-model compatibility")
                elif want_anima and not _mr.family_compatible(bm, "anima"):
                    sys.exit(f"ERROR: {role} LoRA {fname} is base_model={bm!r}, not Anima (flow-matching "
                             f"incompatible). Use --no-arch-guard to override.")
                elif not want_anima and not _mr.family_compatible(bm, "sdxl"):
                    sys.exit(f"ERROR: {role} LoRA {fname} is base_model={bm!r}, incompatible with the "
                             f"SDXL family. Use --no-arch-guard to override.")
    trig = "" if args.no_trigger_inject else triggers_for(records)
    return role_loras, trig


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Highest-quality gen on BOXA (illustrious | anima)")
    ap.add_argument("--family", default=None, choices=tuple(RECIPES),
                    help="recipe family (required, except --refine-image defaults it to illustrious)")
    ap.add_argument("--prompt", default=None,
                    help="subject/scene; the quality scaffold is prepended (optional in --refine-image mode)")
    ap.add_argument("--neg", default=None, help="override the recipe's negative prompt")
    ap.add_argument("--neg-add", default=None,
                    help="append to the recipe/--neg negative (does NOT override); use for stackable "
                         "body/exposure/fabric constraints, e.g. from a base-model quirks card")
    ap.add_argument("--base", default=None, help="override the checkpoint (must match the family arch)")
    ap.add_argument("--lora", action="append", default=None,
                    help="content LoRA file (repeatable; sdxl: model+clip, anima: model-only); 'FILE:STRENGTH' sets strength")
    ap.add_argument("--strength", type=float, default=0.8, help="default LoRA strength when not 'FILE:STRENGTH'")
    ap.add_argument("--from-context", default=None,
                    help="semantic auto-pick of content LoRA(s) (Anima-filtered for --family anima)")
    ap.add_argument("--lora-k", type=int, default=1, help="how many LoRAs --from-context picks")
    ap.add_argument("--min-score", type=float, default=0.4, help="similarity floor for --from-context picks")
    ap.add_argument("--no-arch-guard", action="store_true", help="skip the base-model compatibility check")
    ap.add_argument("--no-trigger-inject", action="store_true", help="do not auto-prepend LoRA trigger words")
    ap.add_argument("--no-wildcard", action="store_true",
                    help="disable {a|b}/__file__ wildcard expansion of the prompt")
    ap.add_argument("--wildcard-dir", default=None,
                    help="dir holding __name__.txt wildcard files (default: ./wildcards)")
    ap.add_argument("--composition", nargs="?", const="__composition__", default=None, metavar="TEXT",
                    help="append composition language to the prompt; bare flag rolls the "
                         "composition wildcard per seed, or pass explicit text")
    ap.add_argument("--res", default=None, help="WxH; defaults to the recipe resolution")
    ap.add_argument("--no-scaffold", action="store_true", help="do not prepend the quality scaffold")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    # advanced pre/post stack
    ap.add_argument("--max", dest="advanced", action="store_true",
                    help="enable the advanced pre/post stack (FreeU+PAG+RescaleCFG[SDXL] + upscale-refine "
                         "+ face/eye detailers; add --ultimate-upscale for tiled upscaling)")
    ap.add_argument("--freeu", dest="freeu", action="store_true", default=None, help="force FreeU on")
    ap.add_argument("--no-freeu", dest="freeu", action="store_false", help="force FreeU off")
    ap.add_argument("--pag", type=float, default=None, help="Perturbed-Attention-Guidance scale (0=off)")
    ap.add_argument("--rescale", type=float, default=None, help="RescaleCFG multiplier (0=off)")
    ap.add_argument("--upscale-by", type=float, default=None, help="post-upscale factor (default: recipe hires_scale)")
    ap.add_argument("--ultimate-upscale", action="store_true",
                    help="use tiled UltimateSDUpscale for the post-upscale (memory-friendly) instead of the 2-pass path")
    ap.add_argument("--detailers", default="face,eye",
                    help="comma list: face,eye,hand,foot (or 'none'). hand/foot are OPTIONAL SEGM passes — do not treat as a guaranteed anatomy fix")
    # ControlNet (SDXL) / Anima control (flow)
    ap.add_argument("--control-image", default=None, help="control source image (uploaded to the box)")
    ap.add_argument("--control", default=None, choices=tuple(dict.fromkeys((*CONTROLNET, *ANIMA_CONTROL))),
                    help="control type. sdxl: openpose (default) | canny | lineart | depth. "
                         "anima: lineart | scribble | depth | any | pose | pose_lora | tile (must be explicit; "
                         "tile defaults its control image to --init-image; "
                         "pose_lora is the recommended pose route, pose is a weak soft prior)")
    ap.add_argument("--control-strength", type=float, default=None,
                    help=f"default {CONTROL_STRENGTH_DEFAULT}; anima tile 0.9")
    ap.add_argument("--control-end", type=float, default=1.0,
                    help="control end_percent, 0..1 (lower frees later denoise steps); SDXL ControlNet "
                         "and the Anima LLLite types only -- pose_lora (AnimaControlApply) has no "
                         "timestep range and rejects anything but 1.0")
    ap.add_argument("--control-node", default="auto",
                    choices=("auto", *dict.fromkeys(n for c in ANIMA_CONTROL_NODES.values() for n in c)),
                    help="Anima only: force the control node instead of picking the first one the box "
                         "registers (A/B of the loramake remap node against the core node)")
    ap.add_argument("--control-refine", action="store_true",
                    help="carry ControlNet conditioning into --max/UltimateSDUpscale diffusion refine "
                         "(opt-in; default preserves the historical text-only finishing pass)")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="treat --control-image as an already-made control map (skip the preprocessor)")
    # IPAdapter (Illustrious / SDXL only)
    ap.add_argument("--ip-image", default=None, help="IPAdapter reference image (uploaded to BOXA)")
    ap.add_argument("--ip", default="style", choices=tuple(IPADAPTER),
                    help="IPAdapter mode: style | face | both (face lock + chained style transfer)")
    ap.add_argument("--ip-style-weight", type=float, default=None,
                    help="both mode only: weight of the chained style stage "
                         "(default the recipe's measured 0.4)")
    ap.add_argument("--ip-weight", type=float, default=None, help="IPAdapter weight (default per mode)")
    ap.add_argument("--ip-end", type=float, default=None,
                    help="IPAdapter end_at (lower = IP only on early steps; default per mode)")
    ap.add_argument("--ip-scaling", default=None,
                    help="IPAdapter embeds_scaling (V only | K+V | ...; default per mode)")
    ap.add_argument("--ip-weight-type", default=None, choices=IP_WEIGHT_TYPES,
                    help="IPAdapter weight_type override for single-stage modes "
                         "(e.g. 'strong style transfer', 'style transfer precise'; "
                         "default per mode; 'both' keeps its pinned two-stage types)")
    # sampler / scheduler / steps / cfg / ModelSampling overrides + filename label (for A/B tuning)
    ap.add_argument("--sampler", default=None, help="override sampler_name (e.g. dpmpp_2m, dpmpp_3m_sde)")
    ap.add_argument("--scheduler", default=None, help="override scheduler (e.g. karras, sgm_uniform)")
    ap.add_argument("--steps", type=int, default=None, help="override base-pass sampling steps (default: recipe steps)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="override CFG, applied to the base pass + refine/upscale + detailers "
                         "(default: recipe cfg)")
    ap.add_argument("--model-sampling", dest="model_sampling", default=None, choices=("auraflow", "none"),
                    help="anima (flow) only: force ModelSamplingAuraFlow on/off (default: recipe card)")
    ap.add_argument("--shift", type=float, default=None,
                    help="anima (flow) only: ModelSamplingAuraFlow shift (default: recipe card)")
    ap.add_argument("--label", default="", help="extra token in the output filename (A/B tagging)")
    # img2img / promote: refine an existing image instead of txt2img
    ap.add_argument("--init-image", default=None,
                    help="img2img: refine this image (promote a draft) instead of starting from noise")
    ap.add_argument("--denoise", type=float, default=0.5,
                    help="img2img denoise when --init-image is set (lower = more faithful to the draft)")
    # standalone refine line: upscale + 2nd-pass resample (+detailers) an existing image, no txt2img
    ap.add_argument("--refine-image", default=None,
                    help="standalone refine: upscale + 2nd-sample this PNG (skips txt2img and model patches)")
    ap.add_argument("--refine-steps", type=int, default=None,
                    help="refine 2nd-pass steps (default: recipe hires_steps)")
    ap.add_argument("--refine-denoise", type=float, default=None,
                    help="refine 2nd-pass denoise, lower = more faithful (default: recipe hires_denoise)")
    # three-step sampling line: role-aware multi-LoRA staging (character -> style+hires -> detailer)
    ap.add_argument("--three-step", action="store_true",
                    help="role-aware 3-step sampling for stacking style/character/pose/detail LoRAs without one overcooked pass")
    ap.add_argument("--char-lora", action="append", default=None, help="three-step: character-role LoRA FILE[:STR] (repeatable)")
    ap.add_argument("--style-lora", action="append", default=None, help="three-step: style-role LoRA FILE[:STR] (repeatable)")
    ap.add_argument("--concept-lora", action="append", default=None, help="three-step: concept-role LoRA FILE[:STR] (repeatable; clothing/effect/see-through etc.)")
    ap.add_argument("--pose-lora", action="append", default=None, help="three-step: pose-role LoRA FILE[:STR] (repeatable)")
    ap.add_argument("--detail-lora", action="append", default=None, help="three-step: detail-role LoRA FILE[:STR] (repeatable)")
    ap.add_argument("--step2-denoise", type=float, default=0.5,
                    help="three-step: step-2 (style+hires) denoise; lower keeps more of step 1's composition")
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    if args.composition and args.no_wildcard and "__" in args.composition:
        ap.error("--composition with a wildcard token requires the wildcard engine "
                 "(drop --no-wildcard or pass literal text)")

    if args.family is None:
        if not args.refine_image:
            ap.error("--family is required (only --refine-image defaults it to illustrious)")
        args.family = "illustrious"
    if not args.prompt and not args.refine_image and not args.three_step:
        ap.error("--prompt is required (it is optional only in --refine-image / --three-step mode)")
    if args.refine_image and (args.control_image or args.ip_image or args.init_image):
        ap.error("--refine-image is a standalone line; do not combine it with "
                 "--control-image / --ip-image / --init-image")
    if args.control == "tile" and not args.control_image:
        if not args.init_image:
            ap.error("--control tile needs --control-image or --init-image (the degraded image itself)")
        args.control_image = args.init_image
        print(f" control tile: control image = --init-image {args.init_image}")
    if args.control_refine and not args.control_image:
        ap.error("--control-refine requires --control-image")

    r = RECIPES[args.family]
    ckpt = args.base or r["ckpt"]
    neg = args.neg if args.neg is not None else r["neg"]
    if args.neg_add:
        neg = f"{neg}, {args.neg_add}"
    content_loras, trig = _resolve_loras(args, r["arch"])
    raw_prompt = args.prompt or ""
    body = raw_prompt if not trig else (f"{trig}, {raw_prompt}" if raw_prompt else trig)
    pos = apply_quality_scaffold(body, r, skip=args.no_scaffold)
    if args.composition:  # default __composition__ token re-rolls per seed via the engine
        pos = f"{pos}, {args.composition}"
    engine = None if args.no_wildcard else WildcardEngine(args.wildcard_dir)
    if engine is not None:
        try:
            engine.expand(pos, seed=0)  # fail fast on a missing file / bad syntax before any GPU work
        except WildcardError as exc:
            sys.exit(f"ERROR: wildcard expansion failed: {exc}")
    w, h = (int(x) for x in (args.res or r["res"]).lower().split("x"))
    if args.init_image and not args.res:
        # img2img: fit the source aspect into the recipe's pixel budget (/8), otherwise
        # the ImageScale crop:center against the recipe's fixed aspect beheads the source
        src_size = image_size(args.init_image)
        if src_size:
            budget, aspect = w * h, src_size[0] / src_size[1]
            w = max(512, int((budget * aspect) ** 0.5) // 8 * 8)
            h = max(512, int((budget / aspect) ** 0.5) // 8 * 8)
            print(f" init aspect {src_size[0]}x{src_size[1]} -> render {w}x{h} (--res overrides)")
        else:
            print(f" warn: could not read init image size ({args.init_image}); "
                  f"using recipe res {w}x{h} (crop:center may clip a mismatched aspect)")
    seed = args.seed if args.seed is not None else random.randint(1, 2_000_000_000)
    detailers = [] if args.detailers.strip().lower() == "none" else [
        d.strip() for d in args.detailers.split(",") if d.strip()]
    opts = {"freeu": args.freeu, "pag": args.pag, "rescale": args.rescale,
            "upscale_by": args.upscale_by, "detailers": detailers,
            "sampler": args.sampler, "scheduler": args.scheduler,
            "steps": args.steps, "cfg": args.cfg,
            "model_sampling": args.model_sampling, "shift": args.shift,
            "ultimate_upscale": args.ultimate_upscale}
    if r["arch"] != "flow" and (args.model_sampling is not None or args.shift is not None):
        ap.error(f"--model-sampling/--shift are anima (flow) only; '{args.family}' is {r['arch']!r}")
    if r["arch"] == "flow":
        # --base onto another registered Anima card: its own recipe replaces the quality_default
        # card's sampling (Dreamland 30 / 4.5 / er_sde / simple), unless a flag says otherwise.
        card_defaults = card_sampling_defaults(ckpt)
        if card_defaults and ckpt != r["ckpt"]:
            opts = apply_sampling_defaults(opts, card_defaults)
            print(" card recipe: " + " ".join(f"{k}={v}" for k, v in card_defaults.items())
                  + " (from anima_models.json; explicit flags win)")
        lora_block_guard(ckpt, args.no_arch_guard, content_loras)

    control = None
    if args.control_image:
        if r["arch"] == "sdxl":
            control_type = args.control or "openpose"
            if control_type not in CONTROLNET:
                ap.error(f"--control {control_type} is an Anima control type; sdxl takes {', '.join(CONTROLNET)}")
            node, spec = None, None
            if args.control_strength is None:
                args.control_strength = CONTROL_STRENGTH_DEFAULT
        elif r["arch"] == "flow":
            if not args.control or args.control not in ANIMA_CONTROL:
                ap.error(f"Anima control needs an explicit --control from {', '.join(ANIMA_CONTROL)}")
            control_type = args.control
            spec = ANIMA_CONTROL[control_type]
            candidates = ANIMA_CONTROL_NODES[spec["kind"]]
            if args.control_node != "auto":
                if args.control_node not in candidates:
                    ap.error(f"--control-node {args.control_node} does not serve --control {control_type} "
                             f"({spec['kind']}: {', '.join(candidates)})")
                candidates = (args.control_node,)
            node = pick_control_node(candidates)
            control_block_guard(ckpt, args.no_arch_guard, node)
            warning = preview3_on_40_block_warning(ckpt, control_type)
            if warning:
                print(f"  {warning}")
            end_error = control_end_guard(spec["kind"], args.control_end)
            if end_error:
                ap.error(end_error)
            if args.control_strength is None:
                args.control_strength = spec.get("strength", CONTROL_STRENGTH_DEFAULT)
            lo, hi = spec.get("strength_range", (0.0, 1.0))
            if not lo <= args.control_strength <= hi:
                ap.error(f"--control-strength for {control_type} must be within {lo}..{hi}")
        else:
            ap.error(f"--control-image is not supported for '{args.family}' ({r['arch']})")
        if not 0.0 <= args.control_end <= 1.0:
            ap.error("--control-end is an end_percent and must be within 0..1")
        preprocess = not args.no_preprocess and not (spec and spec["pre"] is None)  # tile: none
        name = upload_image(args.control_image)
        if not name:
            ap.error(f"failed to upload control image: {args.control_image}")
        control = {"image_name": name, "type": control_type, "strength": args.control_strength,
                   "end": args.control_end, "preprocess": preprocess, "refine": args.control_refine}
        if spec:
            control.update({"kind": spec["kind"], "file": spec["file"], "node": node,
                            "convention": spec["convention"]})

    ipa = None
    if args.ip_image and r["arch"] == "flow":
        sdxl_only = [f for f, given in (("--ip", args.ip != "style"),
                                        ("--ip-end", args.ip_end is not None),
                                        ("--ip-scaling", args.ip_scaling is not None),
                                        ("--ip-weight-type", args.ip_weight_type is not None),
                                        ("--ip-style-weight", args.ip_style_weight is not None)) if given]
        if sdxl_only:
            ap.error(f"{', '.join(sdxl_only)} apply to the SDXL IPAdapter only; Anima --ip-image "
                     f"takes --ip-weight (default {ANIMA_IPADAPTER['weight']})")
        ip_name = upload_image(args.ip_image)
        if not ip_name:
            ap.error(f"failed to upload IPAdapter image: {args.ip_image}")
        ipa = {"image_name": ip_name, "mode": "anima", "file": ANIMA_IPADAPTER["file"],
               "weight": args.ip_weight if args.ip_weight is not None else ANIMA_IPADAPTER["weight"]}
    elif args.ip_image:
        if args.ip == "both":
            if args.ip_weight_type is not None:
                ap.error("--ip-weight-type does not apply to --ip both; its two stages "
                         "are pinned (linear face patch + style transfer)")
        elif args.ip_style_weight is not None:
            ap.error("--ip-style-weight only applies to --ip both")
        ip_name = upload_image(args.ip_image)
        if not ip_name:
            ap.error(f"failed to upload IPAdapter image: {args.ip_image}")
        w_ip = args.ip_weight if args.ip_weight is not None else IPADAPTER[args.ip]["weight"]
        end_ip = args.ip_end if args.ip_end is not None else IPADAPTER[args.ip].get("end", 1.0)
        scaling_ip = args.ip_scaling if args.ip_scaling is not None else IPADAPTER[args.ip].get("scaling", "V only")
        wt_ip = (args.ip_weight_type if args.ip_weight_type is not None
                 else IPADAPTER[args.ip]["weight_type"])
        ipa = {"image_name": ip_name, "mode": args.ip, "weight": w_ip,
               "end": end_ip, "scaling": scaling_ip, "weight_type": wt_ip}
        if args.ip == "both":
            ipa["style_weight"] = (args.ip_style_weight if args.ip_style_weight is not None
                                   else IPADAPTER["both"]["style_weight"])

    init_name = None
    if args.init_image:
        init_name = upload_image(args.init_image)
        if not init_name:
            ap.error(f"failed to upload init image: {args.init_image}")

    refine = None
    if args.refine_image:
        if args.model_sampling is not None or args.shift is not None:
            print(" warn: --model-sampling/--shift are ignored by --refine-image "
                  "(build_refine_graph does not build a ModelSamplingAuraFlow node)")
        src = png_size(args.refine_image)
        if src is None:
            ap.error(f"--refine-image must be a readable PNG (gen outputs are PNG): {args.refine_image}")
        ub = args.upscale_by or r["hires_scale"]
        refine_name = upload_image(args.refine_image)
        if not refine_name:
            ap.error(f"failed to upload refine image: {args.refine_image}")
        refine = {"init_name": refine_name, "upscale_by": ub,
                  "target": [int(src[0] * ub), int(src[1] * ub)],
                  "steps": args.refine_steps or r["hires_steps"],
                  "denoise": args.refine_denoise if args.refine_denoise is not None else r["hires_denoise"],
                  "detailers": detailers, "source": args.refine_image, "source_size": list(src)}

    three_step = None
    if args.three_step:
        if args.refine_image or args.control_image or args.ip_image or args.init_image:
            ap.error("--three-step is a standalone line; do not combine with "
                     "--refine-image / --control-image / --ip-image / --init-image")
        if args.model_sampling is not None or args.shift is not None:
            print(" warn: --model-sampling/--shift are ignored by --three-step "
                  "(build_3step_graph does not build a ModelSamplingAuraFlow node)")
        role_loras, trig3 = _resolve_role_loras(args, r["arch"])
        if not role_loras:
            ap.error("--three-step needs at least one role LoRA "
                     "(--char-lora / --style-lora / --pose-lora / --detail-lora)")
        raw3 = args.prompt or ""
        body3 = raw3 if not trig3 else (f"{trig3}, {raw3}" if raw3 else trig3)
        pos = apply_quality_scaffold(body3, r, skip=args.no_scaffold)
        if args.composition:
            pos = f"{pos}, {args.composition}"
        ub3 = args.upscale_by or r["hires_scale"]
        three_step = {"role_loras": role_loras, "step2_denoise": args.step2_denoise,
                      "target": [int(w * ub3), int(h * ub3)], "upscale_by": ub3,
                      "detailers": detailers, "triggers": trig3}

    print(f"family={args.family} arch={r['arch']} base={ckpt} vae={r['vae']}")
    eff_steps = opts["steps"] if opts.get("steps") is not None else r["steps"]
    eff_cfg = opts["cfg"] if opts.get("cfg") is not None else r["cfg"]
    print(f" {opts.get('sampler') or r['sampler']}/{opts.get('scheduler') or r['scheduler']} "
          f"steps={eff_steps} cfg={eff_cfg} clip_skip={r['clip_skip']} res={w}x{h} model={MODEL}")
    if args.advanced:
        print(f" ADVANCED: {_resolve_adv(args.family, opts)}")
    else:
        print(f" hires={HIRES}")
    if content_loras:
        print(f" loras={', '.join(f'{f}@{s}' for f, s in content_loras)}"
              + (f" triggers={trig}" if trig else ""))
    if control:
        print(f" control={control['type']} src={control['image_name']} "
              f"strength={control['strength']} preprocess={control['preprocess']} "
              f"refine={control['refine']}")
    if ipa:
        print(f" ipadapter={ipa['mode']} src={ipa['image_name']} weight={ipa['weight']}")
    if refine:
        print(f" refine: src={refine['source']} {refine['source_size']} -> {refine['target']} "
              f"steps={refine['steps']} denoise={refine['denoise']} detailers={refine['detailers'] or 'none'}")
    if three_step:
        print(f" three-step: {w}x{h} -> {three_step['target']} step2_denoise={three_step['step2_denoise']} "
              f"detailers={three_step['detailers'] or 'none'}")
        for role in ROLE_ORDER:
            for fname, base_str in three_step["role_loras"].get(role, []):
                s1, s2, s3 = (round(base_str * x, 3) for x in ROLE_SCHEDULE[role])
                print(f"   {role:9} {fname}  base={base_str} -> step[{s1}, {s2}, {s3}]")
        if three_step["triggers"]:
            print(f"   triggers injected: {three_step['triggers']}")
    # txt2img must denoise fully from the empty latent (1.0). --denoise (default 0.5) is the img2img
    # refine strength and only applies when --init-image is given; passing 0.5 to a txt2img empty latent
    # half-forms the image into a washed-out fog (the base-mode washout bug).
    base_denoise = args.denoise if init_name else 1.0
    run(args.family, ckpt, pos, neg, args.n, seed, w, h, content_loras,
        args.advanced, opts, control, ipa, init_name, base_denoise, args.label, engine, refine, three_step)


if __name__ == "__main__":
    main()
