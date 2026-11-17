#!/usr/bin/env python3
"""Anima fast-gen line for BOXA — few-step flow-matching drafts (no upscale, no face/eye detailer).

Defaults come from the sibling anima_models.json (via anima_models.py), not hardcoded literals:
  --unet omitted            -> anima_models.primary_output() (current production checkpoint, Harem)
  --turbo (opt-in)          -> anima_models.turbo_profile() (anima29B_v10_turboBaked_v1, 8/cfg1/euler/simple)
  --legacy-turbo            -> LEGACY_TURBO_UNET + that card's turbo pairing (regression only)
  --steps/--cfg/--shift/--sampler/--scheduler omitted -> the resolved checkpoint's own recipe card

--turbo is opt-in on this CLI: the primary_output checkpoint (miaomiaoHarem_anima8Step10) is itself
a pre-distilled few-step card. The turbo path is the 40-block baked 2.9B UNET
(anima29B_v10_turboBaked_v1). Turbo is already baked in, so no extra turbo LoRA node is attached.
28-block content LoRAs (character/style/concept/accent) go through AnimaLoRARemapTagLoader, not
vanilla LoraLoaderModelOnly. --legacy-turbo keeps the LEGACY_TURBO_UNET pairing chain.

Graph: UNETLoader -> [LoraLoaderModelOnly(turbo LoRA) only on live/legacy pairing] ->
[content LoRAs: remap on 40-block, vanilla on 28-block] -> [ModelSamplingAuraFlow if shift] ->
KSampler. CLIPLoader(qwen_3_06b_base, stable_diffusion) + VAELoader(qwen_image_vae).
At cfg 1.0 the negative prompt is inert.

This is the speed line: no upscale, no face/eye detailer. For a final hero image, promote a chosen
draft through `gen.py promote` (Anima --max img2img in boxa_quality_gen.py).

Output -> outputs/anima/by_model/<DRIVING_MODEL>/anima_fast/ + a .meta.json sidecar
(per outputs/MODEL_FOLDER_RULE.md). Env: BOXA_HOST, DRIVING_MODEL (default claude).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_models  # sibling; stdlib-only at import time
from lora_catalog import base_model_of, pick, record_of, triggers_for  # sibling; stdlib-only at import time
from wildcard import WildcardEngine, WildcardError  # sibling; {a|b}/__file__ prompt expansion
from boxa_comfy import (POLL_TIMEOUT, POST_FX_DEFAULT, BOXA_HOST, apply_post_fx, atomic_write_text,
                      emit_event, looks_like_png, preflight_assets,
                      report_generated_file)  # shared driver helpers

# BOXA_HOST is always a LAN (192.168.x) or Tailscale (100.x) address reached directly, never
# through an HTTP proxy. An ambient HTTP_PROXY/HTTPS_PROXY (e.g. a TUN proxy's env vars) would
# otherwise make urllib route these calls through the proxy and return 403. Install an
# empty-proxy opener so every urlopen below bypasses proxies — the in-code equivalent of the
# `curl --noproxy` already used by boxa_quality_gen.py.
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
MODEL = os.environ.get("DRIVING_MODEL", "claude").strip().lower() or "claude"
OUTDIR = os.environ.get("LORAMAKE_ANIMA_FAST_OUTDIR", f"outputs/anima/by_model/{MODEL}/anima_fast")

# Last-resort fallback only: used when the resolved checkpoint has no card at all (neither a
# primary_output/turbo_profile match nor an explicit --unet hit in anima_models.json) so a recipe
# field is still missing after resolution. Not a default in the normal path.
FALLBACK = {"steps": 10, "cfg": 1.0, "sampler": "er_sde", "scheduler": "simple", "shift": 1.73}
LEGACY_TURBO_UNET = os.environ.get("LORAMAKE_ANIMA_LEGACY_TURBO_UNET", "anima-preview3-base")
TURBO_EXTRACT_28 = (
    "Anima\\turbo\\AnimaTurboBase11_v1.safetensors",
    "Anima\\turbo\\anima_turbo_1_1_extracted_BlueberryTrain.safetensors",
)
REMAP_MANIFEST = "expand_manifest_preview_v1.json"
# Quality-tag boilerplate removed 2026-07-17: same-seed A/B showed the masterpiece
# family adds no quality and only perturbs composition on Anima (stage 1 EXP-Q,
# outputs/_analysis/anima_prompt_mechanics_stage1_2026-07-17/). "safe" kept as the
# SFW steering token pending a dedicated ablation.
SCAFFOLD = "safe, "
NEG_DEFAULT = "lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, signature, watermark"
RES_PRESETS = {"portrait": (896, 1152), "square": (1024, 1024), "landscape": (1152, 896), "tall": (832, 1216)}
# How many recent /history entries a fully cached re-run searches for its graph's earlier render.
CACHED_LOOKBACK = 200


def _req(path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(BOXA_HOST + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def _json(path: str, data: bytes | None = None, headers: dict | None = None) -> dict:
    return json.loads(_req(path, data, headers))


def is_anima29b_card(profile: dict | None, unet: str | None = None) -> bool:
    """True for 40-block Anima-2.9B cards (live v10 or turbo-baked)."""
    blob = " ".join(
        str(x or "") for x in (
            (profile or {}).get("name"),
            (profile or {}).get("unet"),
            unet,
        )
    ).lower()
    if (profile or {}).get("blocks") == 40 or (profile or {}).get("baked_turbo"):
        return True
    return "29b" in blob.replace("-", "").replace("_", "")


def is_28block_turbo_extract(lora_name: str) -> bool:
    name = (lora_name or "").replace("/", "\\")
    return name in TURBO_EXTRACT_28 or name.endswith("AnimaTurboBase11_v1.safetensors")


def build_graph(pos: str, neg: str, seed: int, w: int, h: int, steps: int, cfg: float,
                shift: float | None, sampler: str, scheduler: str, unet: str,
                turbo: dict | None = None,
                content_loras: list[tuple[str, float]] | None = None,
                remap_content: bool = False,
                latent_class: str = "EmptyLatentImage") -> dict:
    """turbo, when given, is {"lora": str, "strength": float} and inserts a conditional
    LoraLoaderModelOnly node; when None that node is omitted entirely (not strength=0 — a
    zero-strength node would still force preflight_assets to require the turbo LoRA file exist,
    and would misleadingly read as "turbo was used, just weakly" in the sidecar).

    remap_content=True loads 28-block content LoRAs through AnimaLoRARemapTagLoader (40-block
    UNET). The turbo extract, when present, stays on LoraLoaderModelOnly (native 40-block).

    shift None likewise omits the ModelSamplingAuraFlow node entirely (a card whose author
    recipe runs without ModelSampling, e.g. anima_aestheticV11's mode "none") — ComfyUI
    rejects a null shift on that node, so None must never reach it."""
    contract = anima_models.card(unet) or anima_models.shared_contract()
    g: dict = {
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "7": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": contract["clip_name"], "type": contract["clip_type"], "device": "default"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": contract["vae_name"]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": pos}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": neg}},
        "11": {"class_type": latent_class, "inputs": {"width": w, "height": h, "batch_size": 1}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["8", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "boxa_anima_fast"}},
    }
    model_ref: list = ["4", 0]
    if turbo:
        g["5"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model_ref, "lora_name": turbo["lora"], "strength_model": turbo["strength"]}}
        model_ref = ["5", 0]
    # chain content LoRAs after the (optional) turbo LoRA, before the AuraFlow shift node
    for i, (lname, lstr) in enumerate(content_loras or []):
        nid = f"5_{i}"
        if remap_content:
            g[nid] = {"class_type": "AnimaLoRARemapTagLoader",
                      "inputs": {"model": model_ref, "text": f"<lora:{lname}:{lstr}>",
                                 "default_weight": lstr, "weight_multiplier": 1.0,
                                 "auto_remap": True, "save_remapped": False,
                                 "extend_to_new_layers": False, "extend_strength": 0.5,
                                 "manifest": REMAP_MANIFEST}}
        else:
            g[nid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model_ref, "lora_name": lname, "strength_model": lstr}}
        model_ref = [nid, 0]
    if shift is not None:
        g["6"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_ref, "shift": shift}}
        model_ref = ["6", 0]
    g["3"] = {"class_type": "KSampler",
              "inputs": {"model": model_ref, "positive": ["9", 0], "negative": ["10", 0],
                         "latent_image": ["11", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}}
    return g


def _poll_delay(elapsed: float) -> float:
    """Poll backoff tuned for the fast line: tight early to catch the typical ~5-7s job with
    minimal latency, then relax so a draft queued behind a long quality run does not hammer
    /history. (boxa_comfy.poll_delay is tuned for minute-long quality jobs and would add ~1-2s
    of latency to every fast draft, so it is intentionally not reused here.)"""
    if elapsed < 12:
        return 0.5
    if elapsed < 90:
        return 2.0
    return 5.0


def earlier_output(pid: str, entry: dict, history: dict) -> dict | None:
    """First image of the newest other /history entry that ran the same graph as ``entry``.

    Re-submitting an identical graph is served entirely from ComfyUI's execution cache, so
    SaveImage writes no new file. ComfyUI 0.33.1 still lists the cached image (the file the
    earlier run saved); 0.27.1 drops cached UI outputs from /history for prompts submitted
    without a client_id and lists none. This recovers that file. ``history`` is a /history
    listing, oldest first."""
    graph = entry.get("prompt", [None] * 3)[2]
    if graph is None:
        return None
    for other_pid, other in reversed(list(history.items())):
        if other_pid == pid or other.get("prompt", [None] * 3)[2] != graph:
            continue
        for out in other.get("outputs", {}).values():
            for img in out.get("images", []):
                return img
    return None


def submit_and_wait(graph: dict) -> tuple[dict, float]:
    t0 = time.time()
    pid = _json("/prompt", json.dumps({"prompt": graph}).encode(), {"Content-Type": "application/json"})["prompt_id"]
    emit_event("prompt_submitted", host=BOXA_HOST, label="anima-fast", prompt_id=pid)
    consecutive_errs = 0
    while time.time() - t0 < POLL_TIMEOUT:
        try:
            hist = _json(f"/history/{pid}")
            consecutive_errs = 0
        except Exception as exc:  # transient network blip — keep polling within the budget
            consecutive_errs += 1
            if consecutive_errs >= 5 and consecutive_errs % 5 == 0:
                print(f"  warn: {consecutive_errs} consecutive poll errors "
                      f"(last: {type(exc).__name__}: {exc})", file=sys.stderr)
            time.sleep(_poll_delay(time.time() - t0))
            continue
        entry = hist.get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                for out in entry.get("outputs", {}).values():
                    for img in out.get("images", []):
                        emit_event("prompt_completed", host=BOXA_HOST, label="anima-fast", prompt_id=pid,
                                   elapsed_s=round(time.time() - t0, 3), outputs=1)
                        return img, time.time() - t0
                # Completed without an image: a fully cached re-run (see earlier_output). The
                # entry is final, so polling it again would only wait out POLL_TIMEOUT.
                img = earlier_output(pid, entry, _json(f"/history?max_items={CACHED_LOOKBACK}"))
                if img is None:
                    msg = (f"prompt {pid} completed from ComfyUI's execution cache without an image, and no "
                           f"earlier run of this graph is in the last {CACHED_LOOKBACK} history entries; "
                           "re-run with another --seed")
                    emit_event("prompt_error", host=BOXA_HOST, label="anima-fast", prompt_id=pid,
                               elapsed_s=round(time.time() - t0, 3), cached=True,
                               error_class="cached_without_output", exception_message=msg)
                    raise RuntimeError(msg)
                emit_event("prompt_completed", host=BOXA_HOST, label="anima-fast", prompt_id=pid,
                           elapsed_s=round(time.time() - t0, 3), outputs=1, cached=True)
                print(f"  cache hit: identical graph already rendered, reusing {img['filename']}",
                      file=sys.stderr)
                return img, time.time() - t0
            if status.get("status_str") == "error":
                emit_event("prompt_error", host=BOXA_HOST, label="anima-fast", prompt_id=pid,
                           elapsed_s=round(time.time() - t0, 3))
                raise RuntimeError(json.dumps(status)[:800])
        time.sleep(_poll_delay(time.time() - t0))
    emit_event("prompt_timeout", host=BOXA_HOST, label="anima-fast", prompt_id=pid,
               elapsed_s=round(time.time() - t0, 3))
    raise TimeoutError(f"render timed out after {POLL_TIMEOUT:.0f}s")


def fetch_png(img: dict) -> bytes:
    """Fetch the rendered image bytes from /view; the caller validates the PNG and writes it."""
    query = urllib.parse.urlencode(
        {"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
    return _req("/view?" + query)


def resolve_res(res: str, profile: dict | None = None) -> tuple[int, int]:
    if res == "portrait" and is_anima29b_card(profile):
        return 1152, 1536
    if res in RES_PRESETS:
        return RES_PRESETS[res]
    w, h = (int(x) for x in res.lower().split("x"))
    return w, h


def resolve_loras(args: argparse.Namespace) -> tuple[list[tuple[str, float]], list[dict]]:
    """Resolve content LoRAs from --lora (manual FILE[:STR]) and --from-context (semantic, Anima-filtered)."""
    loras: list[tuple[str, float]] = []
    records: list[dict] = []
    for spec in (args.lora or []):
        if ":" in spec:
            fname, raw = spec.rsplit(":", 1)
            strength = float(raw)
        else:
            fname, strength = spec, args.content_strength
        loras.append((fname, float(strength)))
        rec = record_of(fname)
        if rec:
            records.append(rec)
    if args.from_context:
        picks = pick(args.from_context, base="Anima", k=args.lora_k, min_score=args.min_score)
        if not picks:
            print(f"  no Anima LoRA scored >= {args.min_score} for '{args.from_context}'")
        for rec in picks:
            strength = float(rec.get("strength_default") or args.content_strength)
            loras.append((rec["file"], strength))
            records.append(rec)
            print(f"  picked: {rec['file']}  score={rec['score']}  str={strength}")
    return loras, records


def resolve_injected_triggers(lora_records: list, *, no_trigger_inject: bool) -> str:
    """boxa_index trigger dump to prepend, or '' when the caller already injected triggers.

    Default is inject-on (the historical fast-line behaviour). Batch callers whose prompts
    already carry activation triggers pass --no-trigger-inject to avoid doubling them.
    """
    if no_trigger_inject:
        return ""
    return triggers_for(lora_records)


def arch_guard(loras: list[tuple[str, float]], allow: bool) -> None:
    """Refuse non-Anima LoRAs on the flow line unless explicitly allowed."""
    for fname, _ in loras:
        base = base_model_of(fname)
        if base is None:
            print(f"  warn: {fname} not in catalog — cannot verify it is an Anima LoRA")
        elif base.lower() != "anima":
            msg = f"LoRA {fname} is base_model={base!r}, not Anima (flow-matching incompatible)"
            if allow:
                print(f"  warn (--no-arch-guard): {msg}")
            else:
                sys.exit(f"ERROR: {msg}. Use --no-arch-guard to override.")


def resolve_defaults(args: argparse.Namespace) -> dict | None:
    """Resolve --unet/--steps/--cfg/--sampler/--scheduler/--shift against anima_models.json and
    mutate args in place (argparse defaults are None sentinels; an explicit flag always wins over
    the resolved card). Returns the turbo pairing dict ({"lora":..., "strength":...}) or None.

    --turbo selects anima_models.turbo_profile() (baked 2.9B UNET). --legacy-turbo forces the
    regression pairing (LEGACY_TURBO_UNET + that card's turbo LoRA). Otherwise primary_output().
    A baked_turbo card uses its recipe and returns None (no extra turbo LoRA).
    An explicit --unet that matches a known card overrides which card supplies the recipe
    defaults; an --unet that matches no card falls back to FALLBACK for any still-missing field.
    """
    legacy = bool(getattr(args, "legacy_turbo", False))
    if legacy:
        args.turbo = True
        if args.unet is None:
            args.unet = LEGACY_TURBO_UNET

    if args.unet is None:
        profile = anima_models.turbo_profile() if args.turbo else anima_models.primary_output()
        if profile is None:
            sys.exit("ERROR: --turbo given but no anima model card is flagged turbo_default=true "
                     "(see: anima_models.py list)")
        args.unet = profile["unet"]
    else:
        profile = anima_models.card(args.unet)

    baked = bool(profile and profile.get("baked_turbo"))
    # NOTE: the card schema is NOT uniform here -- turbo dicts hold shift alongside
    # steps/cfg/sampler/scheduler in one flat dict, but a normal card's "recipe" (steps/cfg/
    # sampler/scheduler) and "model_sampling" (mode/shift) are two separate dicts. Reading
    # shift from "recipe" in the non-turbo path would silently always miss and fall back to
    # FALLBACK["shift"] (1.73, the turbo-specific value) instead of the card's real shift (e.g.
    # miaomiaoHarem_anima8Step10's 3.0) -- so shift is resolved separately per path below.
    use_turbo_recipe = bool(args.turbo and profile and not baked)
    recipe = (profile.get("turbo") if use_turbo_recipe else (profile.get("recipe") if profile else None)) or {}
    args.steps = args.steps if args.steps is not None else recipe.get("steps", FALLBACK["steps"])
    args.cfg = args.cfg if args.cfg is not None else recipe.get("cfg", FALLBACK["cfg"])
    args.sampler = args.sampler or recipe.get("sampler", FALLBACK["sampler"])
    args.scheduler = args.scheduler or recipe.get("scheduler", FALLBACK["scheduler"])
    if args.shift is None:
        if use_turbo_recipe:
            args.shift = recipe.get("shift", FALLBACK["shift"])
        elif profile is None:
            args.shift = FALLBACK["shift"]
        else:
            # Mirror boxa_quality_gen's guard: ModelSamplingAuraFlow is per-model and only built
            # for a real numeric shift. A card whose author recipe runs WITHOUT the node
            # (mode "none" / shift null, e.g. anima_aestheticV11) must resolve to None so
            # build_graph omits it -- .get's default only covers a MISSING key, so an explicit
            # null would otherwise reach ComfyUI as a null FLOAT input and be rejected, and
            # falling back to FALLBACK["shift"] would silently bolt the turbo-tuned 1.73 onto
            # a checkpoint that never asked for it. An explicit --shift still forces the node
            # on (it is the fast line's only ModelSampling knob).
            model_sampling = profile.get("model_sampling") or {}
            if model_sampling.get("mode", "auraflow") == "auraflow":
                args.shift = model_sampling.get("shift")

    if not args.turbo:
        if args.lora_strength is not None:
            print("  warn: --lora-strength has no effect without --turbo")
        return None

    if baked:
        if getattr(args, "turbo_lora", None):
            sys.exit("ERROR: --turbo-lora on a baked_turbo card would double-apply distillation")
        return None

    turbo_card = (profile.get("turbo") if profile else None) or {}
    lora_name = turbo_card.get("lora")
    if getattr(args, "turbo_lora", None):
        lora_name = args.turbo_lora
    if not lora_name:
        sys.exit(f"ERROR: --turbo given but no turbo pairing is known for --unet {args.unet!r} "
                 f"(see: anima_models.py list)")
    if is_anima29b_card(profile, args.unet) and is_28block_turbo_extract(lora_name):
        sys.exit("ERROR: 28-block turbo extract must not stack on the 40-block Anima-2.9B UNET")
    strength = args.lora_strength if args.lora_strength is not None else turbo_card.get("lora_strength", 1.0)
    return {"lora": lora_name, "strength": strength}


def main() -> None:
    ap = argparse.ArgumentParser(description="Anima fast-gen (few-step drafts) on BOXA")
    ap.add_argument("--prompt", required=True, help="subject/scene; the quality scaffold is prepended")
    ap.add_argument("--neg", default=NEG_DEFAULT, help="negative prompt (inert at cfg 1.0)")
    ap.add_argument("--n", type=int, default=1, help="batch: number of images (incrementing seeds)")
    ap.add_argument("--seed", type=int, default=None, help="base seed (random if omitted)")
    ap.add_argument("--res", default="portrait", help="WxH or preset: " + " / ".join(RES_PRESETS))
    ap.add_argument("--steps", type=int, default=None, help="sampling steps (default: resolved checkpoint's recipe)")
    ap.add_argument("--cfg", type=float, default=None, help="CFG (default: resolved checkpoint's recipe)")
    ap.add_argument("--sampler", default=None, help="sampler_name (default: resolved checkpoint's recipe)")
    ap.add_argument("--scheduler", default=None, help="scheduler (default: resolved checkpoint's recipe)")
    ap.add_argument("--shift", type=float, default=None, help="ModelSamplingAuraFlow shift (default: resolved checkpoint's recipe)")
    ap.add_argument("--unet", default=None,
                    help="UNET override (default: anima_models.primary_output(), or turbo_profile() with --turbo)")
    ap.add_argument("--turbo", action="store_true",
                    help="fast lane: anima29B_v10_turboBaked_v1 (8/cfg1/euler/simple). No extra turbo LoRA "
                         "(already baked). Content LoRAs remap onto the 40-block UNET.")
    ap.add_argument("--legacy-turbo", action="store_true",
                    help=f"regression: {LEGACY_TURBO_UNET} + that card's turbo LoRA (implies --turbo)")
    ap.add_argument("--lora-strength", type=float, default=None,
                    help="turbo LoRA strength override (only meaningful with a live/legacy turbo pairing)")
    ap.add_argument("--turbo-lora", default=None,
                    help="override the turbo LoRA filename (refused on baked_turbo; for --legacy-turbo)")
    ap.add_argument("--no-scaffold", action="store_true", help="do not prepend the quality scaffold")
    ap.add_argument("--label", default="", help="extra token in the output filename")
    # content LoRA invocation (Anima-only; chained after the optional turbo LoRA)
    ap.add_argument("--lora", action="append", default=None,
                    help="content LoRA file (repeatable); use 'FILE:STRENGTH' to set strength")
    ap.add_argument("--from-context", default=None,
                    help="semantic auto-pick of Anima content LoRA(s) matching this description")
    ap.add_argument("--lora-k", type=int, default=1, help="how many LoRAs --from-context picks")
    ap.add_argument("--min-score", type=float, default=0.4, help="similarity floor for --from-context picks")
    ap.add_argument("--content-strength", type=float, default=0.8,
                    help="default content-LoRA strength when not given as FILE:STRENGTH")
    ap.add_argument("--no-arch-guard", action="store_true", help="skip the Anima base-model compatibility check")
    ap.add_argument("--no-trigger-inject", action="store_true", help="do not auto-prepend LoRA trigger words")
    ap.add_argument("--no-wildcard", action="store_true",
                    help="disable {a|b}/__file__ wildcard expansion of the prompt")
    ap.add_argument("--fx", default=None,
                    help="post_fx preset for an automatic <stem>_fx.png sibling "
                         "(default: $LORAMAKE_POST_FX_PRESET or film; 'none'/'off' disables)")
    ap.add_argument("--no-fx", action="store_true",
                    help="skip the automatic post_fx finishing pass (raw outputs only)")
    ap.add_argument("--wildcard-dir", default=None,
                    help="dir holding __name__.txt wildcard files (default: ./wildcards)")
    args = ap.parse_args()
    fx_preset = args.fx or os.environ.get("LORAMAKE_POST_FX_PRESET") or POST_FX_DEFAULT
    fx_off = (args.no_fx or os.environ.get("LORAMAKE_POST_FX", "1") == "0"
              or str(fx_preset).lower() in ("none", "off"))

    turbo = resolve_defaults(args)
    profile = anima_models.card(args.unet)
    remap = is_anima29b_card(profile, args.unet)
    w, h = resolve_res(args.res, profile)
    content_loras, lora_records = resolve_loras(args)
    arch_guard(content_loras, args.no_arch_guard)
    if remap and turbo and is_28block_turbo_extract(turbo["lora"]):
        sys.exit("ERROR: 28-block turbo extract must not stack on the 40-block Anima-2.9B UNET")
    # reject a typo'd UNET / content LoRA before any GPU work (best-effort; no-op if BOXA unreachable)
    turbo_loras = [turbo["lora"]] if turbo else []
    preflight_assets("anima", args.unet, turbo_loras + [f for f, _ in content_loras])
    trig = resolve_injected_triggers(lora_records, no_trigger_inject=args.no_trigger_inject)
    body = args.prompt if not trig else f"{trig}, {args.prompt}"
    pos = body if args.no_scaffold else SCAFFOLD + body
    engine = None if args.no_wildcard else WildcardEngine(args.wildcard_dir)
    if engine is not None:
        try:
            engine.expand(pos, seed=0)  # fail fast on a missing file / bad syntax before any GPU work
        except WildcardError as exc:
            sys.exit(f"ERROR: wildcard expansion failed: {exc}")
    base_seed = args.seed if args.seed is not None else (int(time.time() * 1000) % 2_000_000_000)
    os.makedirs(OUTDIR, exist_ok=True)
    unet_stem = args.unet.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    lora_tag = f"_lora{len(content_loras)}" if content_loras else ""

    if turbo:
        turbo_desc = f"turbo={turbo['lora']}@{turbo['strength']}"
    elif args.turbo and profile and profile.get("baked_turbo"):
        turbo_desc = "turbo=baked"
    else:
        turbo_desc = "turbo=off"
    print(f"anima-fast unet={args.unet} {turbo_desc} shift={args.shift} "
          f"steps={args.steps} cfg={args.cfg} {args.sampler}/{args.scheduler} {w}x{h} model={MODEL} n={args.n}")
    if content_loras:
        print(f"  content LoRAs: {', '.join(f'{f}@{s}' for f, s in content_loras)}")
        if trig:
            print(f"  triggers injected: {trig}")
    total = 0.0
    for i in range(args.n):
        seed = base_seed + i
        pos_i = engine.expand(pos, seed) if engine else pos
        graph = build_graph(pos_i, args.neg, seed, w, h, args.steps, args.cfg,
                            args.shift, args.sampler, args.scheduler, args.unet, turbo, content_loras,
                            remap_content=remap,
                            latent_class="EmptySD3LatentImage" if remap else "EmptyLatentImage")
        img, elapsed = submit_and_wait(graph)
        total += elapsed
        raw = fetch_png(img)
        if not looks_like_png(raw):
            print(f"  [{i:02d} seed={seed}] SKIP: /view returned {len(raw)} bytes, not a PNG", file=sys.stderr)
            continue
        tag = f"_{args.label}" if args.label else ""
        stem = f"boxaaf{tag}{lora_tag}_{unet_stem}_{args.steps}st_{i:02d}_{seed}"
        dst = os.path.join(OUTDIR, stem + ".png")
        with open(dst, "wb") as fh:
            fh.write(raw)
        meta = {"seed": seed, "steps": args.steps, "cfg": args.cfg, "unet": args.unet,
                "lora": turbo["lora"] if turbo else None,
                "lora_strength": turbo["strength"] if turbo else None,
                "turbo": turbo, "shift": args.shift,
                "model_sampling": "auraflow" if args.shift is not None else "none",
                "content_loras": [{"file": f, "strength": s} for f, s in content_loras],
                "triggers_injected": trig,
                "res": [w, h], "sampler": args.sampler, "scheduler": args.scheduler,
                "prompt": pos_i, "neg": args.neg, "elapsed_s": round(elapsed, 1), "driving_model": MODEL}
        atomic_write_text(Path(dst + ".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
        report_generated_file(dst)
        if not fx_off:
            fx_path = apply_post_fx(dst, fx_preset)
            if fx_path:
                report_generated_file(fx_path)
        line = f"  [{i:02d} seed={seed}] {elapsed:.1f}s -> {dst}"
        if pos_i != pos:
            line += f"\n        prompt: {pos_i}"
        print(line)
    if args.n > 1:
        print(f"done: {args.n} image(s), {total:.1f}s total, {total / args.n:.1f}s avg")


if __name__ == "__main__":
    main()
