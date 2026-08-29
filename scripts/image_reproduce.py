#!/usr/bin/env python3
"""Faithful visual-reproduction gen loop: source image -> regenerated image.

The last stage of the image-reproduction pipeline. Given a source image (and,
optionally, a pre-built ``reproduction_spec_v1``), this drives the existing Z3
ComfyUI engine to regenerate an approximation of the source and iterates toward
it, scoring each candidate by CCIP character-similarity.

Why this is the only realistic path (settled): Pixiv strips embedded generation
metadata on upload and AI artists almost never disclose their recipe, so an
arbitrary AI image's true recipe is unrecoverable (see ``pixiv_ingest.py`` /
``image_recipe.py``). The achievable goal is *visual* reproduction.

Strategy — faithful (same character + same composition):
  - ControlNet (openpose / canny / lineart) preprocessed from the source locks
    pose + composition.
  - IPAdapter with the source as reference transfers colour / aesthetic (style
    mode) or identity (face mode).
  - A semantically-selected character LoRA reinforces identity.
  - Each round generates N candidates; every candidate is scored with
    ``ccip_difference(source, candidate)`` (lower = closer). The loop tightens the
    ControlNet / IPAdapter lock and advances seeds until the best candidate falls
    below the same-character threshold, the score plateaus, or the round budget is
    spent.

Scoring is multi-metric: CCIP ranks identity (primary), graycorr scores
composition, WD14 general-tag F1 scores content, and colorcorr (palette) +
LPIPS (texture) are recorded as the advisory style signal (never a gate). The
stop condition needs the identity threshold AND the composition/content pass
bands, or a plateau. Each round also feeds a tag-diff of source vs round-best
back into the prompt (missing source tags appended positive, spurious candidate
tags appended negative, capped per round, LoRA triggers and person-count tags
protected) — the inversion lever. Portrait sources run the chained dual
IPAdapter (face lock + full-range style transfer) so the source's rendering
aesthetic transfers alongside identity.

Reuses (does not reimplement): ``z3_quality_gen.run`` (engine: ControlNet +
IPAdapter + LoRA + hires), ``lora_catalog.pick`` (semantic LoRA select),
``z3_comfy`` (Z3 driver + upload), the imgutils venv via ``imgutils_bridge.py``
(CCIP), and ``image_repro_spec.build_spec`` (spec assembly).

SFW only. Z3 is a shared gaming box — every round probes it and only fires when
idle (never onto a busy/gaming GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUTS  # noqa: E402
import image_recipe as ir
import image_repro_spec as rs
import recipe_guess as rg
import z3_comfy
import z3_quality_gen as zqg
# pick / triggers_for resolve at runtime (z3_quality_gen imports them the same way);
# Pyright cannot see lora_catalog's module-level defs — a confirmed static false positive.
from lora_catalog import pick, record_of, triggers_for  # pyright: ignore[reportAttributeAccessIssue]

# Faithful reproduction needs ControlNet + IPAdapter, which are SDXL/Illustrious
# only on Z3 — so the family is fixed to illustrious regardless of the spec's
# candidate_family (Anima / Flux cannot run CN+IPA here).
FAMILY = "illustrious"
REPRO_OUTPUT_BASE = OUTPUTS / "image_repro"

# Tags that signal a human figure (-> lineart locks the composition; see
# choose_cn_type for the measured openpose-vs-lineart verdict) vs. a scene.
_PERSON_TAGS = frozenset({
    "1girl", "2girls", "3girls", "1boy", "2boys", "multiple_girls", "multiple_boys",
    "solo", "solo_focus", "couple",
})
# Tags that signal a face-dominant crop (-> IPAdapter face mode is safe; the ref
# is auto center-cropped to 768px, so face mode needs the face near centre).
_PORTRAIT_TAGS = frozenset({"portrait", "close-up", "face_focus", "headshot"})
# Base-model families that run on the SDXL pipeline (others are dropped from the
# LoRA selection to protect the ControlNet/IPAdapter graph).
_SDXL_BASE_HINTS = ("illustrious", "noobai", "sdxl", "pony", "xl")

PLATEAU_DELTA = 0.01  # CCIP improvement below this counts as "no progress"
# Composition/content pass bands (B2/B3 of the acceptance vocabulary in
# docs/GEN_ROUTES.md): stopping on identity alone lets a same-character candidate
# with the wrong scene win, so both bands must hold alongside the CCIP threshold.
GRAYCORR_PASS = 0.60
TAG_F1_PASS = 0.60


# --------------------------------------------------------------------------- #
# Spec + asset selection
# --------------------------------------------------------------------------- #
def load_or_build_spec(source: str, spec_path: str | None) -> dict[str, Any]:
    """Load an existing reproduction_spec_v1 or build the mechanical half fresh."""
    if spec_path:
        with open(spec_path, encoding="utf-8") as fh:
            return json.load(fh)
    return rs.build_spec(source, source_type="unknown",
                         reference_strategy="ipadapter", candidate_family="unknown")


def general_tags(spec: dict[str, Any]) -> dict[str, float]:
    tags = spec.get("tag_candidates") or {}
    return tags.get("general", {}) if isinstance(tags, dict) else {}


def choose_cn_type(spec: dict[str, Any], override: str) -> str:
    """ControlNet preprocessor: explicit strategy, else inferred from tags. Z3 has
    no depth control model, so 'depth' maps to canny (the structural fallback).

    Person images default to LINEART, not openpose: measured 2026-08-27 on the
    same source/seeds (waiIllustrious, square, IPA face 0.9), lineart beat openpose
    on composition (graycorr 0.34-0.46 vs 0.17-0.33) and slightly on identity —
    consistent with Z3 Illustrious responding weakly to openpose (pose-lock is
    only verified on ZZZ/iLustMix). Openpose stays available via --control."""
    if override != "auto":
        return override
    strat = spec.get("reference_strategy")
    if strat in ("openpose", "canny", "lineart"):
        return strat
    if strat == "depth":
        return "canny"
    tags = general_tags(spec)
    return "lineart" if any(t in tags for t in _PERSON_TAGS) else "canny"


def choose_ip_mode(spec: dict[str, Any], override: str) -> str:
    """Portraits get the chained dual lock (face + style transfer): face alone
    reproduces the character but not the source's rendering aesthetic — style
    fidelity is half the reproduction goal. Non-portraits use pure style."""
    if override != "auto":
        return override
    tags = general_tags(spec)
    return "both" if any(t in tags for t in _PORTRAIT_TAGS) else "style"


def build_context(spec: dict[str, Any]) -> str:
    """Free-text context for semantic LoRA selection."""
    parts: list[str] = []
    for tag in spec.get("content_tags") or []:
        if isinstance(tag, str):
            parts.append(tag)
        elif isinstance(tag, (list, tuple)) and tag:
            parts.append(str(tag[0]))
    if spec.get("prompt_candidate"):
        parts.append(spec["prompt_candidate"])
    return " ".join(p for p in parts if p)[:500]


def _is_sdxl_base(record: dict[str, Any]) -> bool:
    base = str(record.get("base_model") or "").lower()
    return any(hint in base for hint in _SDXL_BASE_HINTS)


def select_loras(spec: dict[str, Any], manual: list[str], k: int,
                 min_score: float) -> tuple[list[tuple[str, float]], str, list[dict[str, Any]]]:
    """Resolve content LoRAs. Manual --lora FILE:STR wins; otherwise semantic pick
    from the Z3 catalog, filtered to SDXL-arch bases. Degrades to no LoRA (with a
    warning) when the catalog index is missing or the embedder is unavailable."""
    if manual:
        loras: list[tuple[str, float]] = []
        for spec_str in manual:
            if ":" in spec_str:
                name, raw = spec_str.rsplit(":", 1)
                strength = float(raw)
            else:
                name, strength = spec_str, 0.8
            loras.append((name, strength))
            # Soft arch guard: this pipeline is fixed to illustrious SDXL (ControlNet+IPAdapter),
            # so a catalogued non-SDXL LoRA will be rejected by ComfyUI. Warn but honor the operator's
            # explicit choice — unknown (uncatalogued) LoRAs pass through silently.
            rec = record_of(name)
            if rec and not _is_sdxl_base(rec):
                print(f"  WARN: manual LoRA {rec.get('file', name)!r} base_model="
                      f"{rec.get('base_model')!r} is not SDXL-arch — the {FAMILY} "
                      f"ControlNet+IPAdapter pipeline will likely reject it")
        return loras, "", []

    # Character-LoRA hints from the wd14 character bucket (recipe_guess) outrank the
    # generic semantic pick: a recognized character names its LoRA directly.
    hinted: list[dict[str, Any]] = []
    for hint in (spec.get("inferred") or {}).get("lora_hints") or []:
        if not hint.get("file") or str(hint.get("content_rating")) == "nsfw":
            continue
        rec = record_of(hint["file"])
        if rec and _is_sdxl_base(rec):
            hinted.append(rec)
    if hinted:
        top = hinted[:k]
        print(f"  character LoRA via wd14 inference: {[r['file'] for r in top]}")
        return ([(r["file"], r.get("strength_default", 0.8)) for r in top],
                triggers_for(top), top)

    context = build_context(spec)
    if not context:
        print("  WARN: empty context — skipping semantic LoRA selection")
        return [], "", []
    try:
        records = pick(context, base=None, k=k, min_score=min_score)
    except SystemExit:
        print("  WARN: LoRA catalog index missing (run lora_catalog.py build && embed) "
              "— proceeding without a character LoRA")
        return [], "", []
    except Exception as exc:  # embedder unavailable etc. — degrade, do not crash
        print(f"  WARN: LoRA selection failed ({type(exc).__name__}) — proceeding without a LoRA")
        return [], "", []

    usable = [r for r in records if _is_sdxl_base(r)]
    dropped = len(records) - len(usable)
    if dropped:
        print(f"  note: dropped {dropped} non-SDXL LoRA candidate(s) for pipeline compatibility")
    loras = [(r["file"], r.get("strength_default", 0.8)) for r in usable]
    trigger = triggers_for(usable) if usable else ""
    return loras, trigger, usable


def build_prompt(spec: dict[str, Any], trigger_text: str) -> tuple[str, str]:
    r = zqg.RECIPES[FAMILY]
    parts = [r["pos_prefix"].strip().rstrip(",")]
    if trigger_text:
        parts.append(trigger_text)
    if spec.get("prompt_candidate"):
        parts.append(spec["prompt_candidate"])
    pos = ", ".join(p for p in parts if p)
    neg_extra = (spec.get("negative_candidate") or "").strip()
    neg = r["neg"] + (", " + neg_extra if neg_extra else "")
    return pos, neg


# --------------------------------------------------------------------------- #
# Z3 idle probe (probe-then-run; never fire onto a busy/gaming GPU)
# --------------------------------------------------------------------------- #
def probe_z3_idle(vram_threshold: float) -> tuple[bool, str]:
    """True only when Z3 ComfyUI is reachable, has VRAM headroom, AND its queue is
    empty. Catches the gaming-while-idle-ComfyUI case (queue empty but VRAM busy)
    via the /system_stats VRAM ratio."""
    host = z3_comfy.Z3_HOST
    try:
        stats = json.loads(z3_comfy.curl([f"{host}/system_stats"]) or "{}")
    except Exception:
        return False, "system_stats unreachable"
    devices = stats.get("devices") or []
    if not devices:
        return False, "no GPU devices reported (Z3 unreachable?)"
    dev = devices[0]
    total = dev.get("vram_total") or 0
    free = dev.get("vram_free") or 0
    ratio = (free / total) if total else 0.0
    if ratio < vram_threshold:
        return False, f"VRAM only {ratio:.0%} free ({free // (1024**2)}MB/{total // (1024**2)}MB)"
    try:
        queue = json.loads(z3_comfy.curl([f"{host}/queue"]) or "{}")
    except Exception:
        return False, "queue unreachable"
    running = len(queue.get("queue_running") or [])
    pending = len(queue.get("queue_pending") or [])
    if running or pending:
        return False, f"queue busy (running={running}, pending={pending})"
    return True, f"idle (VRAM {ratio:.0%} free)"


def wait_for_z3(vram_threshold: float, max_wait: float, interval: float) -> tuple[bool, str]:
    """Poll the idle probe until Z3 is free or max_wait elapses. max_wait <= 0
    probes exactly once."""
    deadline = time.monotonic() + max_wait
    while True:
        ok, reason = probe_z3_idle(vram_threshold)
        if ok or time.monotonic() >= deadline:
            return ok, reason
        print(f"  Z3 not idle: {reason} — waiting {int(interval)}s ...", flush=True)
        time.sleep(interval)


def free_z3_memory() -> None:
    """Release ComfyUI's loaded models + cache on Z3 after a run, so the shared
    gaming box does not hold VRAM idle. Best-effort: never raises."""
    if z3_comfy.free_vram(z3_comfy.Z3_HOST):
        print("  freed Z3 ComfyUI VRAM/cache (POST /free)")
    else:
        print("  note: Z3 /free cleanup skipped")


# --------------------------------------------------------------------------- #
# Multi-metric scoring (CCIP identity / graycorr composition / wd14 F1 content /
# LPIPS advisory)
# --------------------------------------------------------------------------- #
def score_ccip(source: str, candidate: str) -> dict[str, Any] | None:
    """Run the imgutils venv bridge to score CCIP difference between two images."""
    return ir.run_bridge("ccip", source, candidate)


def wd14_general_batch(paths: list[str]) -> dict[str, dict[str, float]]:
    """One bridge process, N images -> {path: general_tags}. Batch keeps the WD14
    model load to one per round; missing entries just mean the tagger failed for
    that image (callers treat the F1 as unavailable, never as zero)."""
    venv = os.environ.get("IMGUTILS_VENV_PY", ir.DEFAULT_VENV_PY)
    if not paths or not os.path.exists(venv):
        return {}
    argv = [venv, ir.BRIDGE, "wd14", *paths]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        payload = json.loads(proc.stdout)
    except Exception:
        return {}
    if len(paths) == 1:  # single-image call returns the flat op_wd14 shape
        return {paths[0]: payload.get("general") or {}} if payload.get("ok") else {}
    out: dict[str, dict[str, float]] = {}
    for path, entry in (payload.get("results") or {}).items():
        if entry.get("ok"):
            out[path] = entry.get("general") or {}
    return out


def tag_f1(source_general: dict[str, float] | None,
           candidate_general: dict[str, float] | None,
           threshold: float = rg.GENERAL_THRESHOLD) -> tuple[float | None, int]:
    """WD14 general-tag F1 between source and candidate at the presence threshold.
    None = one side has no tagging result (unavailable, not zero overlap)."""
    if source_general is None or candidate_general is None:
        return None, 0
    src = {t for t, s in source_general.items() if s >= threshold}
    cand = {t for t, s in candidate_general.items() if s >= threshold}
    if not src and not cand:
        return None, 0
    shared = len(src & cand)
    return round(2 * shared / (len(src) + len(cand)), 4), shared


def score_candidate(source: str, candidate: str,
                    source_general: dict[str, float] | None,
                    candidate_general: dict[str, float] | None) -> dict[str, Any]:
    """All four metrics for one candidate. CCIP is the only ranking primary; a
    missing secondary metric stays None and only weakens the tie-break."""
    metrics: dict[str, Any] = {
        "ccip_difference": None, "ccip_threshold": None, "same_character": None,
        "graycorr": None, "colorcorr": None, "lpips": None, "tag_f1": None, "tags_shared": 0,
    }
    ccip = score_ccip(source, candidate)
    if ccip and ccip.get("ok"):
        metrics.update(ccip_difference=ccip["difference"], ccip_threshold=ccip["threshold"],
                       same_character=ccip["same_character"])
    gray = ir.run_bridge("graycorr", source, candidate)
    if gray and gray.get("ok"):
        metrics["graycorr"] = gray["correlation"]
    color = ir.run_bridge("colorcorr", source, candidate)
    if color and color.get("ok"):
        metrics["colorcorr"] = color["correlation"]
    lpips = ir.run_bridge("lpips", source, candidate)
    if lpips and lpips.get("ok"):
        metrics["lpips"] = lpips["difference"]
    f1, shared = tag_f1(source_general, candidate_general)
    metrics["tag_f1"] = f1
    metrics["tags_shared"] = shared
    return metrics


def rank_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """CCIP-primary sort key with (graycorr desc, tag F1 desc, colorcorr desc,
    LPIPS asc) breaks. colorcorr is the palette half of the style signal; LPIPS
    the texture half. Unavailable secondaries sort as worst so a fully-scored
    candidate wins ties."""
    return (metrics.get("ccip_difference") if metrics.get("ccip_difference") is not None else 9.9,
            -(metrics.get("graycorr") if metrics.get("graycorr") is not None else -9.9),
            -(metrics.get("tag_f1") if metrics.get("tag_f1") is not None else -9.9),
            -(metrics.get("colorcorr") if metrics.get("colorcorr") is not None else -9.9),
            metrics.get("lpips") if metrics.get("lpips") is not None else 9.9)


def bands_pass(metrics: dict[str, Any]) -> bool:
    """B1+B2+B3: identity threshold AND composition AND content pass bands. A band
    whose metric is unavailable fails closed (cannot claim a pass unmeasured)."""
    return bool(metrics.get("same_character")
                and (metrics.get("graycorr") or 0.0) >= GRAYCORR_PASS
                and (metrics.get("tag_f1") or 0.0) >= TAG_F1_PASS)


# --------------------------------------------------------------------------- #
# Generation round
# --------------------------------------------------------------------------- #
def _snapshot_outputs() -> dict[Path, float]:
    if not zqg.OUT.exists():
        return {}
    return {p: p.stat().st_mtime for p in zqg.OUT.glob("*.png")}


def _new_outputs(before: dict[Path, float], label: str) -> list[Path]:
    """Outputs written since the snapshot. Compared by (path, mtime), not path set:
    a rerun with the same source/seed/label OVERWRITES the previous round's files
    in place, and a pure path diff silently loses the whole round."""
    fresh = sorted(p for p, mtime in _snapshot_outputs().items()
                   if p not in before or mtime > before[p])
    labelled = [p for p in fresh if label in p.name]
    return labelled or fresh


def generate_round(ckpt: str, pos: str, neg: str, n: int, base_seed: int, w: int, h: int,
                   loras: list[tuple[str, float]], control: dict[str, Any], ip: dict[str, Any],
                   advanced: bool, opts: dict[str, Any], label: str) -> list[Path]:
    before = _snapshot_outputs()
    r = zqg.RECIPES[FAMILY]
    zqg.run(family=FAMILY, ckpt=ckpt, pos=pos, neg=neg or r["neg"], n=n, base_seed=base_seed,
            w=w, h=h, content_loras=loras, advanced=advanced, opts=opts,
            control=control, ip=ip, label=label)
    return _new_outputs(before, label)


def make_record(spec: dict[str, Any], label: str, pos: str, neg: str, candidate: Path,
                metrics: dict[str, Any], control: dict[str, Any],
                ip: dict[str, Any]) -> dict[str, Any]:
    record = rs.new_experiment_record(spec)
    record["workflow_id"] = f"z3_quality_gen:{FAMILY}:cn+ipa"
    record["asset_query"] = {
        "control_type": control["type"], "control_strength": control["strength"],
        "control_end": control["end"], "ip_mode": ip["mode"], "ip_weight": ip["weight"],
        "label": label,
    }
    record["prompt_summary"] = pos[:300]
    record["negative_summary"] = neg[:200]
    record["history_status"] = "completed"
    record["output_path"] = str(candidate)
    record["metrics"] = metrics
    diff = metrics.get("ccip_difference")
    if diff is not None:
        record["verdict"] = (f"ccip={diff:.4f} (threshold {metrics['ccip_threshold']:.4f}, "
                             f"same_character={metrics['same_character']}) "
                             f"graycorr={metrics.get('graycorr')} tag_f1={metrics.get('tag_f1')}")
    else:
        record["verdict"] = "ccip scoring unavailable"
    return record


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_resolution(value: str | None, source: str | None = None) -> tuple[int, int]:
    """Explicit --res wins; otherwise fit the SOURCE aspect into the recipe's pixel
    budget (rounded to /8). Rendering at the recipe's fixed portrait aspect against
    e.g. a square source depressed graycorr by construction — the composition band
    was unreachable before the aspects matched."""
    if value:
        w, h = value.lower().split("x")
        return int(w), int(h)
    rw_s, rh_s = zqg.RECIPES[FAMILY]["res"].lower().split("x")
    rw, rh = int(rw_s), int(rh_s)
    if not source:
        return rw, rh
    try:
        with Image.open(source) as im:
            sw, sh = im.size
    except Exception:
        return rw, rh
    if not sw or not sh:
        return rw, rh
    budget = rw * rh
    aspect = sw / sh
    w = max(512, int((budget * aspect) ** 0.5) // 8 * 8)
    h = max(512, int((budget / aspect) ** 0.5) // 8 * 8)
    return w, h


def reproduce(source: str, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sha12 = spec["sha256"][:12]
    out_dir = Path(args.out_dir) if args.out_dir else REPRO_OUTPUT_BASE / sha12
    out_dir.mkdir(parents=True, exist_ok=True)
    ir._dump(str(out_dir / "repro_spec.json"), spec)

    # Redirect the raw gen candidates out of the shared z3_quality_gen bucket when
    # asked (run() writes to the module-level OUT, which image_reproduce also reads
    # back). This is how an NSFW run keeps every output confined under nsfw/.
    if args.candidate_dir:
        zqg.OUT = Path(args.candidate_dir)

    cn_type = choose_cn_type(spec, args.control)
    ip_mode = choose_ip_mode(spec, args.ip_mode)
    ip_defaults = zqg.IPADAPTER[ip_mode]
    ip_weight = args.ip_weight if args.ip_weight is not None else ip_defaults["weight"]
    style_weight = None
    if ip_mode == "both":
        style_weight = (args.ip_style_weight if args.ip_style_weight is not None
                        else ip_defaults["style_weight"])
    elif args.ip_style_weight is not None:
        # Same hard stance as z3_quality_gen's own CLI: silently ignoring a
        # mis-scoped flag hides an operator mistake.
        sys.exit("--ip-style-weight only applies to --ip-mode both")
    ckpt = args.ckpt or zqg.RECIPES[FAMILY]["ckpt"]
    w, h = parse_resolution(args.res, source)
    loras, trigger, _ = select_loras(spec, args.lora, args.lora_k, args.min_score)
    pos, neg = build_prompt(spec, trigger)
    opts = {"sampler": None, "scheduler": None, "freeu": None, "pag": None,
            "rescale": None, "upscale_by": None, "detailers": ["face", "eye"],
            "ultimate_upscale": False}

    if spec.get("candidate_family") in ("anima", "flux"):
        print(f"  WARN: spec candidate_family={spec['candidate_family']} cannot run "
              f"ControlNet+IPAdapter on Z3 — using the {FAMILY} (SDXL) pipeline instead")

    plan = {
        "source": source, "sha256": spec["sha256"], "out_dir": str(out_dir),
        "candidate_dir": str(zqg.OUT),
        "family": FAMILY, "ckpt": ckpt, "resolution": f"{w}x{h}",
        "control": {"type": cn_type, "strength": args.control_strength, "end": args.control_end},
        "ipadapter": {"mode": ip_mode, "weight": ip_weight,
                      "end": ip_defaults["end"], "scaling": ip_defaults["scaling"],
                      **({"style_weight": style_weight} if style_weight is not None else {})},
        "loras": [{"file": f, "strength": s} for f, s in loras],
        "trigger_text": trigger, "positive": pos, "negative": neg,
        "candidates_per_round": args.candidates, "max_rounds": args.max_rounds,
        "advanced": not args.no_advanced,
    }
    print("Reproduction plan:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.dry_run:
        ir._dump(str(out_dir / "reproduction_plan.json"), plan)
        print(f"\n[dry-run] no Z3 contact; plan written to {out_dir}")
        return {"dry_run": True, "plan": plan}

    # Fail-fast CCIP self-probe BEFORE any Z3 contact: without CCIP the loop cannot
    # rank or stop, and used to burn the full candidates*rounds GPU budget blind.
    probe = score_ccip(source, source)
    if not probe or not probe.get("ok"):
        reason = (probe or {}).get("reason") or "imgutils bridge unavailable"
        stop_reason = f"ccip unavailable — {reason}"
        print(f"ERROR: {stop_reason}")
        print("  provision once from a terminal with network access, then re-run:")
        print("    HF_HUB_OFFLINE=0 $IMGUTILS_VENV_PY scripts/imgutils_bridge.py ccip <a.png> <b.png>")
        result = {"schema": "image_reproduction_result_v1", "source_path": source,
                  "sha256": spec["sha256"], "stop_reason": stop_reason,
                  "plan": plan, "best": None}
        ir._dump(str(out_dir / "reproduction_result.json"), result)
        return result

    src_general = general_tags(spec)
    protected = rg.protected_tags(trigger, zqg.RECIPES[FAMILY]["pos_prefix"], src_general)

    cn_name = z3_comfy.upload_image(source)
    ip_name = z3_comfy.upload_image(source)
    if not cn_name or not ip_name:
        print("ERROR: failed to upload the source image to Z3 — aborting")
        return {"error": "upload_failed"}

    log_path = out_dir / "iteration_log.jsonl"
    best: dict[str, Any] | None = None
    strength = args.control_strength
    weight = ip_weight
    base_seed = args.seed
    plateau = 0
    stop_reason = "max_rounds reached"

    for rnd in range(args.max_rounds):
        if not args.no_probe:
            ok, reason = wait_for_z3(args.probe_vram, args.probe_max_wait, args.probe_interval)
            if not ok:
                stop_reason = f"Z3 not idle at round {rnd}: {reason}"
                print(f"  {stop_reason} — stopping")
                break

        control = {"image_name": cn_name, "type": cn_type, "strength": round(strength, 3),
                   "end": args.control_end, "preprocess": True}
        ip = {"image_name": ip_name, "mode": ip_mode, "weight": round(weight, 3),
              "end": ip_defaults["end"], "scaling": ip_defaults["scaling"]}
        if style_weight is not None:
            ip["style_weight"] = style_weight
        label = f"repro_{sha12}_r{rnd}"
        print(f"\n=== round {rnd + 1}/{args.max_rounds} "
              f"(cn {control['type']}@{control['strength']}, ip {ip['mode']}@{ip['weight']}) ===")

        candidates = generate_round(ckpt, pos, neg, args.candidates, base_seed, w, h,
                                    loras, control, ip, not args.no_advanced, opts, label)
        base_seed += args.candidates
        if not candidates:
            print("  no images produced this round (submit/timeout) — stopping")
            stop_reason = f"no output at round {rnd}"
            break

        prev_best = best["metrics"]["ccip_difference"] if best else None
        cand_generals = wd14_general_batch([str(c) for c in candidates])
        round_scored: list[tuple[dict[str, Any], Path]] = []
        with open(log_path, "a", encoding="utf-8") as fh:
            for cand in candidates:
                metrics = score_candidate(source, str(cand), src_general,
                                          cand_generals.get(str(cand)))
                fh.write(json.dumps(make_record(spec, label, pos, neg, cand, metrics,
                                                control, ip), ensure_ascii=False) + "\n")
                if metrics["ccip_difference"] is not None:
                    round_scored.append((metrics, cand))
                    print(f"  {cand.name}: ccip={metrics['ccip_difference']:.4f} "
                          f"graycorr={metrics.get('graycorr')} tag_f1={metrics.get('tag_f1')} "
                          f"same_character={metrics['same_character']}")
                else:
                    print(f"  {cand.name}: ccip unavailable")

        if not round_scored:
            # The self-probe passed, so this is a mid-run degradation — stop rather
            # than burn the remaining round budget with no ranking or stop condition.
            stop_reason = f"ccip scoring unavailable at round {rnd}"
            print(f"  {stop_reason} — stopping")
            break
        round_scored.sort(key=lambda mc: rank_key(mc[0]))
        rb_metrics, rb_path = round_scored[0]
        rb_diff = rb_metrics["ccip_difference"]
        if best is None or rb_diff < best["metrics"]["ccip_difference"]:
            best = {"path": str(rb_path), "metrics": rb_metrics, "label": label}
        print(f"  round best ccip={rb_diff:.4f} | global best "
              f"ccip={best['metrics']['ccip_difference']:.4f}")

        if bands_pass(rb_metrics):
            stop_reason = "reached identity + composition/content bands"
            print(f"  {stop_reason} — stopping")
            break

        # Prompt refinement — the tag-diff feedback lever. Runs every scored round;
        # an actual change resets the plateau so the new prompt gets a fresh round
        # before the CN/IP locks tighten.
        refined = False
        rb_general = cand_generals.get(str(rb_path))
        if src_general and rb_general is not None:
            diff = rg.tag_diff(src_general, rb_general, protected)
            pos2, neg2, notes = rg.refine_prompt(pos, neg, diff)
            if notes:
                pos, neg = pos2, neg2
                refined = True
                print(f"  prompt refined: {'; '.join(notes)}")
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"schema": "prompt_revision_v1", "round": rnd,
                                         "notes": notes, "positive": pos[:400],
                                         "negative": neg[:300]}, ensure_ascii=False) + "\n")

        improvement = (prev_best - best["metrics"]["ccip_difference"]) \
            if prev_best is not None else float("inf")
        if improvement < PLATEAU_DELTA:
            if refined:
                plateau = 0
            else:
                plateau += 1
                strength = min(1.0, strength + 0.05)
                weight = min(1.0, weight + 0.05)
                if plateau >= 2:
                    stop_reason = "CCIP plateaued over 2 rounds"
                    print(f"  {stop_reason} — stopping")
                    break
        else:
            plateau = 0

    return finalize(source, spec, out_dir, best, stop_reason, plan)


def finalize(source: str, spec: dict[str, Any], out_dir: Path, best: dict[str, Any] | None,
             stop_reason: str, plan: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "image_reproduction_result_v1",
        "source_path": source, "sha256": spec["sha256"],
        "stop_reason": stop_reason, "plan": plan, "best": None,
    }
    if best:
        best_src = Path(best["path"])
        metrics = best["metrics"]
        diff_tag = f"{metrics['ccip_difference']:.4f}".replace("0.", "0_")
        best_dst = out_dir / f"best_ccip{diff_tag}{best_src.suffix}"
        try:
            shutil.copy2(best_src, best_dst)
            meta_src = best_src.with_name(best_src.name + ".meta.json")
            if meta_src.exists():
                shutil.copy2(meta_src, best_dst.with_name(best_dst.name + ".meta.json"))
        except Exception as exc:
            print(f"  WARN: could not copy best candidate: {exc!r}")
            best_dst = best_src
        result["best"] = dict(metrics, candidate_path=str(best_src), saved_path=str(best_dst),
                              bands_pass=bands_pass(metrics))
        print(f"\nBest reproduction: ccip={metrics['ccip_difference']:.4f} "
              f"graycorr={metrics.get('graycorr')} tag_f1={metrics.get('tag_f1')} "
              f"(same_character={metrics.get('same_character')}, "
              f"bands_pass={bands_pass(metrics)})")
        print(f"  saved: {best_dst}")
    else:
        print("\nNo scored reproduction produced.")
    ir._dump(str(out_dir / "reproduction_result.json"), result)
    print(f"  stop reason: {stop_reason}")
    print(f"  artifacts: {out_dir}")
    free_z3_memory()
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Faithful visual reproduction of a source image on Z3 "
                    "(ControlNet + IPAdapter + LoRA, CCIP-scored iteration).")
    p.add_argument("source", help="path to the source image to reproduce")
    p.add_argument("--spec", help="reuse an existing reproduction_spec_v1 JSON (else built fresh)")
    p.add_argument("--candidates", type=int, default=4, help="candidates per round (default 4)")
    p.add_argument("--max-rounds", type=int, default=4, help="max iteration rounds (default 4)")
    p.add_argument("--seed", type=int, default=42, help="base seed (default 42)")
    p.add_argument("--ckpt", help="override the SDXL checkpoint (default the illustrious recipe)")
    p.add_argument("--res", help="output resolution WxH (default the illustrious recipe)")
    p.add_argument("--control", choices=("auto", "openpose", "canny", "lineart"), default="auto",
                   help="ControlNet preprocessor (default auto: from spec/tags)")
    p.add_argument("--control-strength", type=float, default=0.85,
                   help="initial ControlNet strength (default 0.85)")
    p.add_argument("--control-end", type=float, default=0.7,
                   help="ControlNet end_percent (default 0.7 — frees late detail)")
    p.add_argument("--ip-mode", choices=("auto", "style", "face", "both"), default="auto",
                   help="IPAdapter mode (default auto: both [face+style] for portraits, else style)")
    p.add_argument("--ip-style-weight", type=float, default=None,
                   help="both mode: weight of the chained style stage (default engine recipe)")
    p.add_argument("--ip-weight", type=float, default=None,
                   help="IPAdapter weight (default the mode's recipe value)")
    p.add_argument("--lora", action="append", default=[],
                   help="manual LoRA FILE:STRENGTH (repeatable; bypasses semantic select)")
    p.add_argument("--lora-k", type=int, default=2, help="semantic LoRA candidates to keep (default 2)")
    p.add_argument("--min-score", type=float, default=0.2,
                   help="semantic LoRA cosine floor (default 0.2)")
    p.add_argument("--no-advanced", action="store_true",
                   help="disable the --max hires/detailer stack (faster, lower fidelity)")
    p.add_argument("--probe-vram", type=float, default=0.40,
                   help="min free-VRAM ratio to consider Z3 idle (default 0.40)")
    p.add_argument("--probe-max-wait", type=float, default=600.0,
                   help="seconds to wait for an idle Z3 each round (default 600; 0 = probe once)")
    p.add_argument("--probe-interval", type=float, default=60.0,
                   help="seconds between idle probes (default 60)")
    p.add_argument("--no-probe", action="store_true",
                   help="skip the Z3 idle probe (fire regardless — use with care)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the plan and select assets, but do not contact Z3")
    p.add_argument("--out-dir", help="artifacts directory (spec/log/best; default outputs/image_repro/<sha12>/)")
    p.add_argument("--candidate-dir",
                   help="directory for the raw gen candidates (default the z3_quality_gen bucket); "
                        "set this to confine outputs elsewhere, e.g. an NSFW run into nsfw/")
    return p


def metadata_guard(source: str) -> None:
    """Cheapest deterministic layer first: an image that carries its recipe is
    never inverted — the metadata-replay engines reproduce it exactly. Advise and
    exit (never silently switch kernels: forge_repro is Z3-only and plan-first)."""
    try:
        parsed = ir.parse_image(source)
    except Exception:
        return  # unreadable metadata is not a reason to block inversion
    if parsed.get("comfy_prompt"):
        sys.exit("source carries an embedded ComfyUI prompt graph — replay it exactly instead:\n"
                 f"  python3 scripts/gen.py comfy_repro --source {source} --run")
    if parsed.get("params") or parsed.get("prompt"):
        sys.exit("source carries embedded A1111 parameters — replay them exactly instead:\n"
                 f"  python3 scripts/gen.py forge_repro --source {source} --run\n"
                 "(Anima-family A1111 images: see docs/anima_line/A1111_CROSS_KERNEL_REPLAY.md)")
    if parsed.get("workflow"):
        sys.exit("source carries a ComfyUI UI workflow chunk (no API prompt graph) — convert it\n"
                 "to an API graph and replay via comfy_repro instead of inverting:\n"
                 "  docs/anima_line/A1111_CROSS_KERNEL_REPLAY.md describes the manual mapping")


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args()
    if not os.path.isfile(args.source):
        build_parser().error(f"not a file: {args.source}")

    metadata_guard(args.source)
    spec = load_or_build_spec(args.source, args.spec)
    reproduce(args.source, spec, args)


if __name__ == "__main__":
    main()
