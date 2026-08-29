"""Brief -> gen.py command mapping for the app backend (M1).

A job brief is the app's unit of work — the same JSON shape a workbench UI submits or a
dispatcher hands to a gen-runner agent (docs/app_blueprint.md §二: job = brief JSON).
This module owns boundary validation and the translation into a gen.py invocation; it
never executes anything itself. gen.py keeps every guardrail (preflight, per-box adapt,
gates, idle probe, auto /free) — the app layer must not re-implement any of it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from loramake import anima_structured_pilot as anima_pilot

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "gen.py"
QUIRKS = ROOT / "scripts" / "model_quirks.json"
# NSFW router (nsfw/APP_SWITCH_DESIGN.md): same CLI surface as gen.py but output-rooted at
# nsfw/outputs/ with the wd14 SFW gate refused. The nsfw/ tree is gitignored, so its absence
# on a fresh clone is a validation error, not a crash.
GEN_NSFW = ROOT / "nsfw" / "gen_nsfw.py"

# Tiers are gen.py intents surfaced 1:1 (blueprint R2). Finer sub-tiers (e.g. a separate
# "max" vs "standard") wait until the engine grows a distinct recipe for them.
PILOT_TIER = "anima_structured_pilot"
TIERS = ("draft", "best", "artistic", "sequence", "forge", PILOT_TIER)
# "auto" resolves at launch (gen.resolve_box): quick-scan Z3 first; if Z3 looks
# gaming/busy route to ZZZ immediately, otherwise use Z3 (2026-08-16 policy).
BOXES = ("auto", "z3", "zzz")
PROVIDERS = ("local", "civitai", "gpt_image")
EXTERNAL_PROVIDERS = ("civitai", "gpt_image")
EXTERNAL_FIELDS = {
    "tier", "prompt", "box", "provider", "confirm_paid_generation", "sfw",
    "n", "seed", "steps", "cfg", "res",
}
MAX_N = 16
MAX_EXTRA_LORAS = 4  # concept/pose/clothing stacks beyond this melt identity anyway

# Late-bound network kill-switch hook (app/server.py wires it to
# app.network.enforcement.brief_problems). None — the default for CLI and
# pytest imports of this module — means no network gating.
NETWORK_PROBLEMS = None
ENGINE_SAMPLERS = (
    "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
    "dpmpp_sde", "ddim", "uni_pc", "lcm",
)
ENGINE_SCHEDULERS = (
    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta",
)
ARTISTIC_ASPECTS = ("portrait", "tall", "square", "landscape", "wide", "hi-portrait")
ARTISTIC_STYLE_MODES = ("lora", "artist", "none")
ARTISTIC_TEXT_FIELDS = (
    "artist", "medium", "camera", "composition", "lighting", "pose", "color_mood", "expression",
)
ARTISTIC_ONLY_FIELDS = (
    "aspect", "detail", "style_mode", "pose_ref", "character", "outfit", "action", "scene",
    *ARTISTIC_TEXT_FIELDS,
)
# Quality-line knobs (z3_quality_gen only): the Anima turbo line behind `draft` has no
# --neg-add/--no-freeu/--pag, so a draft brief carrying them would die in argparse.
QUALITY_ONLY_FIELDS = ("neg_add", "no_freeu", "pag")
PAG_MAX = 10.0
# Explicit WxH. Artistic sizing goes through `aspect` instead: gen.py derives --res from the
# art_prompt recipe there, and a second --res would silently win over it.
RES_TIERS = ("best", "sequence", "draft", "forge")
# Forge (A1111 kernel) has its own sampler/scheduler display names and no quirks-registered
# checkpoints; these comfy-vocabulary fields are refused rather than silently re-interpreted.
FORGE_REJECTED_FIELDS = {
    "base": "base is not selectable for forge tier (Forge checkpoints are aki-junctioned, "
            "not quirks-registered; pass ckpt via the CLI)",
    "sampler": "sampler is not selectable for forge tier (A1111 sampler names differ; CLI only)",
    "scheduler": "scheduler is not selectable for forge tier (A1111 scheduler names differ; CLI only)",
    "neg_add": "neg_add is a quality-line knob; forge tier takes the prompt as-is",
    "no_freeu": "no_freeu does not apply to the Forge engine",
    "pag": "pag does not apply to the Forge engine",
    "ref": "ref does not apply to forge tier (no IPAdapter on the Forge line)",
}
DRAFT_RES_PRESETS = ("portrait", "square", "landscape", "tall")  # z3_anima_fast RES_PRESETS
RES_MIN, RES_MAX = 256, 4096
# --pick-best crowns the aesthetic best of N. gen.py refuses it on sequence; artistic already
# runs its own survivor filter + contact sheet, and draft is a throwaway turbo batch.
PICK_BEST_TIERS = ("best",)
PILOT_FIELDS = {"tier", "prompt", "box", "provider", "task_spec", "sfw"}


def _res_problem(res: object, tier: object) -> str | None:
    """None when `res` is a usable size for this tier, else the message to report."""
    if not isinstance(res, str) or not res.strip():
        return "res must be a WxH string (e.g. 896x1152)"
    value = res.strip()
    if tier == "draft" and value in DRAFT_RES_PRESETS:
        return None
    width, _, height = value.partition("x")
    if not (width.isdigit() and height.isdigit()):
        preset_hint = f" or one of {', '.join(DRAFT_RES_PRESETS)}" if tier == "draft" else ""
        return f"res must be WxH (e.g. 896x1152){preset_hint}"
    if not all(RES_MIN <= int(side) <= RES_MAX for side in (width, height)):
        return f"res sides must be between {RES_MIN} and {RES_MAX}"
    return None


def _base_cards() -> dict:
    """model_quirks base cards (stem -> card); the selectable-base registry lives there."""
    try:
        return json.loads(QUIRKS.read_text(encoding="utf-8")).get("bases") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def validate(brief: object) -> list[str]:
    """Boundary validation of an untrusted brief. Returns problems; empty list = valid."""
    problems: list[str] = []
    if not isinstance(brief, dict):
        return ["brief must be a JSON object"]
    if NETWORK_PROBLEMS is not None:
        problems.extend(NETWORK_PROBLEMS(brief))
    tier = brief.get("tier")
    provider = brief.get("provider", "local")
    if provider not in PROVIDERS:
        problems.append(f"provider must be one of {', '.join(PROVIDERS)}")
    if tier not in TIERS:
        problems.append(f"tier must be one of {', '.join(TIERS)}")
    if not str(brief.get("prompt") or "").strip():
        problems.append("prompt is required")
    if brief.get("box", "auto") not in BOXES:
        problems.append(f"box must be one of {', '.join(BOXES)}")
    if tier == "draft" and brief.get("box", "auto") not in ("z3", "auto"):
        problems.append("draft tier runs on z3 only (Anima turbo line); auto resolves to z3")
    if tier == "forge" and brief.get("box", "auto") not in ("z3", "auto"):
        problems.append("forge tier runs on z3 only (the Forge deployment lives on Z3); auto resolves to z3")
    if tier == "sequence" and not str(brief.get("ref") or "").strip():
        problems.append("sequence tier requires ref (identity-anchor image path)")
    ref = brief.get("ref")
    if ref and not Path(ref).is_file():
        problems.append(f"ref image not found: {ref}")
    n = brief.get("n")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N):
        problems.append(f"n must be an integer in 1..{MAX_N}")
    seed = brief.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        problems.append("seed must be a non-negative integer")
    confirmation = brief.get("confirm_paid_generation")
    if confirmation is not None and not isinstance(confirmation, bool):
        problems.append("confirm_paid_generation must be a boolean")
    nsfw = brief.get("nsfw")
    if nsfw is not None and not isinstance(nsfw, bool):
        problems.append("nsfw must be a boolean")
    sfw = brief.get("sfw")
    if sfw is not None and not isinstance(sfw, bool):
        problems.append("sfw must be a boolean")
    keep_resident = brief.get("keep_resident")
    if keep_resident is not None and not isinstance(keep_resident, bool):
        problems.append("keep_resident must be a boolean")

    # Cloud channels are intentionally narrow: one SFW best-tier request, no local LoRA/
    # direction/checkpoint semantics, and an explicit confirmation on every submission.
    # This returns before the local-engine validators below so unsupported fields are never
    # silently reinterpreted as ComfyUI knobs.
    if provider in EXTERNAL_PROVIDERS:
        for key in sorted(set(brief) - EXTERNAL_FIELDS):
            problems.append(f"{key} is not accepted in external provider briefs")
        if tier != "best":
            problems.append("external providers currently support the best tier only")
        if confirmation is not True:
            problems.append("external provider generation requires confirm_paid_generation=true")
        if nsfw is True:
            problems.append("external provider channels are SFW-only")
        if sfw is False:
            problems.append("external provider channels require sfw=true or omission")
        unsupported = (
            "base", "lora", "style_lora", "loras", "direction", "ref", "sampler", "scheduler",
            "neg_add", "no_freeu", "pag", "pick_best", "aspect", "detail", "style_mode",
            "keep_resident", *ARTISTIC_TEXT_FIELDS,
        )
        for key in unsupported:
            if brief.get(key) is not None:
                problems.append(f"{key} is not supported by external providers")
        if n is not None:
            maximum = 12 if provider == "civitai" else 10
            if isinstance(n, int) and not isinstance(n, bool) and n > maximum:
                problems.append(f"{provider} n must be in 1..{maximum}")
        if provider == "gpt_image" and seed is not None:
            problems.append("seed is not supported by the GPT Image adapter")
        steps = brief.get("steps")
        cfg = brief.get("cfg")
        if provider == "gpt_image":
            if steps is not None:
                problems.append("steps is not supported by the GPT Image adapter")
            if cfg is not None:
                problems.append("cfg is not supported by the GPT Image adapter")
        else:
            if steps is not None and (
                    not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 150):
                problems.append("Civitai steps must be an integer in 1..150")
            if cfg is not None and (
                    not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or not 0.0 <= cfg <= 30.0):
                problems.append("Civitai cfg must be a number in 0.0..30.0")
        res = brief.get("res")
        if res is not None:
            res_problem = _res_problem(res, "best")
            if res_problem:
                problems.append(res_problem)
            elif provider == "civitai":
                width, height = (int(side) for side in str(res).split("x", 1))
                if width > 2048 or height > 2048 or width % 16 or height % 16:
                    problems.append("Civitai res must be <=2048 per side and divisible by 16")
        return problems

    if tier == PILOT_TIER:
        for key in sorted(set(brief) - PILOT_FIELDS):
            problems.append(f"{key} is not accepted in {PILOT_TIER} briefs")
        if provider != "local":
            problems.append(f"{PILOT_TIER} requires provider=local")
        if brief.get("box", "zzz") != "zzz":
            problems.append(f"{PILOT_TIER} runs on zzz only")
        if brief.get("sfw") is not True:
            problems.append(f"{PILOT_TIER} requires sfw=true")
        task_spec = brief.get("task_spec")
        if task_spec is None:
            problems.append("task_spec is required")
        else:
            problems.extend(f"task_spec: {problem}" for problem in anima_pilot.validate_task(task_spec))
            if isinstance(task_spec, dict):
                source = task_spec.get("source")
                exact = source.get("text") if isinstance(source, dict) else None
                if brief.get("prompt") != exact:
                    problems.append("prompt must byte-for-byte equal task_spec.source.text")
        if not anima_pilot.pilot_enabled():
            problems.append(
                f"{PILOT_TIER} is disabled; set {anima_pilot.FEATURE_ENV}=1 only for an approved Pilot run"
            )
        return problems

    if confirmation is True:
        problems.append("confirm_paid_generation only applies to external providers")
    if nsfw:
        if not GEN_NSFW.is_file():
            problems.append("NSFW pipeline not installed (nsfw/gen_nsfw.py missing)")
        if brief.get("sfw") is True:
            problems.append("sfw gate is mutually exclusive with nsfw mode")
        if tier == "forge":
            problems.append("forge tier is not wired into the nsfw router yet")
    if tier == "forge":
        for key, why in FORGE_REJECTED_FIELDS.items():
            if brief.get(key) is not None:
                problems.append(why)
    base = brief.get("base")
    if base is not None:
        card = _base_cards().get(base) if isinstance(base, str) else None
        if card is None or not card.get("ckpt") or not card.get("boxes"):
            problems.append("base must be a registered base-model stem (see /api/meta bases)")
        else:
            if tier == "draft":
                problems.append("base is not selectable for draft tier (Anima turbo line is fixed)")
            if (card.get("family") or "illustrious") != "illustrious":
                problems.append(f"base {base} is not an illustrious-family checkpoint")
            boxes = card.get("boxes") or []
            declared = brief.get("box", "auto")
            # "auto" resolves at launch; gen.py re-validates the card against the
            # concrete box then (_resolve_base), so only pin mismatches fail here.
            if declared != "auto" and declared not in boxes:
                problems.append(f"base {base} is not registered on box {declared}")
    steps = brief.get("steps")
    if steps is not None and (
            not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 60):
        problems.append("steps must be an integer in 1..60")
    cfg = brief.get("cfg")
    if cfg is not None and (
            not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or not 1.0 <= cfg <= 15.0):
        problems.append("cfg must be a number in 1.0..15.0")
    sampler = brief.get("sampler")
    if sampler is not None and sampler not in ENGINE_SAMPLERS:
        problems.append(f"sampler must be one of {', '.join(ENGINE_SAMPLERS)}")
    scheduler = brief.get("scheduler")
    if scheduler is not None and scheduler not in ENGINE_SCHEDULERS:
        problems.append(f"scheduler must be one of {', '.join(ENGINE_SCHEDULERS)}")
    neg_add = brief.get("neg_add")
    if neg_add is not None and (not isinstance(neg_add, str) or not neg_add.strip()):
        problems.append("neg_add must be a non-empty string")
    no_freeu = brief.get("no_freeu")
    if no_freeu is not None and not isinstance(no_freeu, bool):
        problems.append("no_freeu must be a boolean")
    pag = brief.get("pag")
    if pag is not None and (
            not isinstance(pag, (int, float)) or isinstance(pag, bool) or not 0.0 <= pag <= PAG_MAX):
        problems.append(f"pag must be a number in 0.0..{PAG_MAX}")
    if tier == "draft":
        for key in QUALITY_ONLY_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is not available on draft tier (Anima turbo line)")
    res = brief.get("res")
    if res is not None:
        res_problem = _res_problem(res, tier)
        if res_problem:
            problems.append(res_problem)
        if tier in TIERS and tier not in RES_TIERS:
            problems.append(f"res is only valid for {', '.join(RES_TIERS)} tiers (artistic sizes via aspect)")
    pick_best = brief.get("pick_best")
    if pick_best is not None:
        if not isinstance(pick_best, bool):
            problems.append("pick_best must be a boolean")
        elif pick_best and tier in TIERS and tier not in PICK_BEST_TIERS:
            problems.append(f"pick_best is only valid for {', '.join(PICK_BEST_TIERS)} tier")
    aspect = brief.get("aspect")
    if aspect is not None and aspect not in ARTISTIC_ASPECTS:
        problems.append(f"aspect must be one of {', '.join(ARTISTIC_ASPECTS)}")
    detail = brief.get("detail")
    if detail is not None and not isinstance(detail, bool):
        problems.append("detail must be a boolean")
    style_mode = brief.get("style_mode")
    if style_mode is not None and style_mode not in ARTISTIC_STYLE_MODES:
        problems.append(f"style_mode must be one of {', '.join(ARTISTIC_STYLE_MODES)}")
    for key in ARTISTIC_TEXT_FIELDS:
        val = brief.get(key)
        if val is not None and (not isinstance(val, str) or not val.strip()):
            problems.append(f"{key} must be a non-empty string")
    if tier != "artistic":
        for key in ARTISTIC_ONLY_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is only valid for artistic tier")
    for key in ("lora", "style_lora", "direction", "pose_ref", "character", "outfit", "action"):
        val = brief.get(key)
        if val is not None and not isinstance(val, str):
            problems.append(f"{key} must be a string")
    if brief.get("scene") is not None and not isinstance(brief["scene"], bool):
        problems.append("scene must be a boolean")
    pose_ref = brief.get("pose_ref")
    if pose_ref and brief.get("box", "auto") == "z3":
        # pose-library line is ZZZ-only (Z3's Illustrious ignores openpose); "auto" is fine —
        # the box resolver prefers zzz, so it only fails loudly on an EXPLICIT z3 pin.
        problems.append("pose_ref requires box zzz (pose-library line is ZZZ-only)")
    extra = brief.get("loras")
    if extra is not None:
        if (not isinstance(extra, list)
                or not all(isinstance(x, str) and x.strip() for x in extra)):
            problems.append("loras must be a list of LoRA spec strings (FILE[:WEIGHT])")
        elif len(extra) > MAX_EXTRA_LORAS:
            problems.append(f"loras: at most {MAX_EXTRA_LORAS} extra LoRAs per job")
    group = brief.get("group")
    if group is not None and (
            not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", group)):
        problems.append("group must match [A-Za-z0-9_-]{1,40}")
    return problems


def to_argv(brief: dict, json_out: str | Path) -> list[str]:
    """The gen.py invocation for a validated brief. Always --run --json-out (the app reads
    results from the JSON file, never from stdout). nsfw briefs route to the gen_nsfw
    wrapper — same CLI, output rooted at nsfw/outputs/, rating=nsfw injected."""
    provider = brief.get("provider", "local")
    if brief.get("tier") == PILOT_TIER:
        return [
            sys.executable,
            str(GEN),
            PILOT_TIER,
            "--box",
            "zzz",
            "--task-json",
            anima_pilot.canonical_json(brief["task_spec"]),
            "--run",
            "--json-out",
            str(json_out),
        ]
    if provider in EXTERNAL_PROVIDERS:
        argv = [sys.executable, str(GEN), brief["tier"],
                "--provider", provider,
                "--prompt", str(brief["prompt"]).strip(),
                "--box", brief.get("box", "zzz")]
        if brief.get("n") is not None:
            argv += ["--n", str(brief["n"])]
        if brief.get("seed") is not None:
            argv += ["--seed", str(brief["seed"])]
        if brief.get("confirm_paid_generation") is True:
            argv += ["--confirm-paid-generation"]
        argv += ["--run", "--json-out", str(json_out)]
        provider_params: list[str] = []
        for key in ("steps", "cfg", "res"):
            if brief.get(key) is not None:
                provider_params += [f"--{key}", str(brief[key])]
        if provider_params:
            argv += ["--", *provider_params]
        return argv

    nsfw = bool(brief.get("nsfw"))
    argv = [sys.executable, str(GEN_NSFW if nsfw else GEN), brief["tier"],
            "--prompt", str(brief["prompt"]).strip(),
            "--box", brief.get("box", "auto")]
    if brief.get("base"):
        argv += ["--base", brief["base"]]
    if brief.get("lora"):
        argv += ["--lora", brief["lora"]]
    for spec in brief.get("loras") or []:  # extra LoRAs (concept/pose/clothing/...) — --lora repeats
        argv += ["--lora", spec]
    if brief.get("n") is not None:
        argv += ["--n", str(brief["n"])]
    if brief.get("seed") is not None:
        argv += ["--seed", str(brief["seed"])]
    if brief.get("ref"):
        argv += ["--ref", brief["ref"]]
    tier = brief["tier"]
    if tier == "artistic":
        if brief.get("direction"):
            argv += ["--direction", brief["direction"]]
        if brief.get("character"):
            argv += ["--character", brief["character"]]
        if brief.get("outfit"):
            argv += ["--outfit", brief["outfit"]]
        if brief.get("action"):
            argv += ["--action", brief["action"]]
        if brief.get("scene") is True:
            argv += ["--scene"]
        if brief.get("pose_ref"):
            argv += ["--pose-ref", brief["pose_ref"]]
        if brief.get("style_lora"):
            argv += ["--style-lora", brief["style_lora"]]
        if brief.get("aspect") is not None:
            argv += ["--aspect", brief["aspect"]]
        if brief.get("detail") is True:
            argv += ["--detail"]
        if brief.get("style_mode") is not None:
            argv += ["--style-mode", brief["style_mode"]]
        if brief.get("artist") is not None:
            argv += ["--artist", brief["artist"]]
        for key in ("medium", "camera", "composition", "lighting", "pose", "color_mood", "expression"):
            if brief.get(key) is not None:
                argv += [f"--{key.replace('_', '-')}", brief[key]]
        if not nsfw and brief.get("sfw", True):  # artistic defaults to the SFW gate ON
            argv += ["--sfw"]                    # (blueprint R8); gen_nsfw refuses --sfw
    if tier == "sequence":
        argv += ["--ccip-keep"]  # delegated runs want the machine-readable filtered shape
    if brief.get("pick_best") is True:  # router-level crown; needs --run, which is always on here
        argv += ["--pick-best"]
    if brief.get("keep_resident") is True:
        argv += ["--no-free"]
    argv += ["--run", "--json-out", str(json_out)]
    engine_params: list[str] = []
    for key in ("steps", "cfg", "sampler", "scheduler", "res"):
        if brief.get(key) is not None:
            engine_params += [f"--{key}", str(brief[key])]
    if brief.get("neg_add") is not None:
        # single-valued on the engine; gen.py merges this with an artistic card's own appendix
        engine_params += ["--neg-add", str(brief["neg_add"]).strip()]
    if brief.get("no_freeu") is True:
        engine_params += ["--no-freeu"]
    if brief.get("pag") is not None:
        engine_params += ["--pag", str(brief["pag"])]
    if engine_params:
        argv += ["--", *engine_params]
    return argv
