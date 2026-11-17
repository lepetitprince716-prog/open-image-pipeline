"""Brief -> gen.py command mapping for the app backend (M1).

A job brief is the app's unit of work — the same JSON shape a workbench UI submits or a
dispatcher hands to a gen-runner agent (docs/app_blueprint.md §二: job = brief JSON).
This module owns boundary validation and the translation into a gen.py invocation; it
never executes anything itself. gen.py keeps every guardrail (preflight, per-box adapt,
gates, idle probe, auto /free) — the app layer must not re-implement any of it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from loramake import anima_structured_pilot as anima_pilot
from loramake import civitai_cloud
from loramake.qwen21_canvas import CANVAS as QWEN21_CANVAS
from loramake import revision

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "gen.py"
QUIRKS = ROOT / "scripts" / "model_quirks.json"
# NSFW router (nsfw/APP_SWITCH_DESIGN.md): same CLI surface as gen.py but output-rooted at
# nsfw/outputs/ with the wd14 SFW gate refused. The nsfw/ tree is gitignored, so its absence
# on a fresh clone is a validation error, not a crash.
GEN_NSFW = ROOT / "nsfw" / "gen_nsfw.py"
# Independent remake engine (not gen.py / gen_nsfw). Often local-only; absence is a
# validation error, same shape as a missing gen_nsfw.py on a fresh clone.
GEN_GACHA_NSFW = ROOT / "nsfw" / "t2i_gacha_nsfw.py"
GEN_QWEN21 = ROOT / "scripts" / "qwen21_gen.py"

# Tiers are gen.py intents surfaced 1:1 (blueprint R2). Finer sub-tiers (e.g. a separate
# "max" vs "standard") wait until the engine grows a distinct recipe for them.
# t2i_gacha_nsfw is the exception: Studio brief tier, not a gen.py intent.
PILOT_TIER = "anima_structured_pilot"
GACHA_TIER = "gacha2.0"
GACHA_NSFW_TIER = "t2i_gacha_nsfw"
# Qwen-Image-2.1 line: Studio brief tier backed by scripts/qwen21_gen.py (a standalone adapter, not a
# gen.py intent). Natural-language prompt, BOXA sidecar only, no LoRA / base / negative (cfg 1).
QWEN21_TIER = "qwen21"
# MOSS-Image line: Studio brief tier backed by scripts/moss_gen.py (a standalone adapter like qwen21).
# app/moss.py owns its fields, capability table and argv; previews only, BOXA only, SFW only. It is
# imported where it is used: the public mirror publishes this module but not app/moss.py.
MOSS_TIER = "moss"
# A revise brief does not describe a picture; it describes a bounded change to one that
# already exists. Everything it does not name is inherited from the parent's sidecar, which
# is what makes the freeze declaration checkable rather than decorative.
REVISE_TIER = "revise"
TIERS = (
    "draft", "best", "artistic", "sequence", "forge", "rescue", "promote",
    GACHA_TIER, GACHA_NSFW_TIER, PILOT_TIER, QWEN21_TIER, MOSS_TIER, REVISE_TIER,
)
# Checkpoint family each tier's engine loads, in model_quirks `family` terms (gen.ENGINES
# resource_family "sdxl" == quirks "illustrious"). tests/test_brief_free_layer.py pins this table
# to gen.ENGINES so it cannot drift silently. draft / gacha2.0 reject any base (fixed Anima line).
TIER_FAMILY = {
    "draft": "anima",
    "best": "illustrious",
    "artistic": "illustrious",
    "sequence": "illustrious",
    "forge": "illustrious",
    "rescue": "illustrious",
    "promote": "anima",
    GACHA_TIER: "anima",
    GACHA_NSFW_TIER: "anima",
    PILOT_TIER: "illustrious",
    QWEN21_TIER: "qwen21",
    MOSS_TIER: "moss",
    # revise inherits the parent's checkpoint and takes no base; its engine is the
    # boxa_quality_gen (Illustrious) chain, so gen.ENGINES reports the sdxl default.
    REVISE_TIER: "illustrious",
}
# Source intents surfaced on the Studio queue. Frontend sends opaque ids only; this module
# resolves them to a real path before to_argv. region is a gen.py intent but NOT a brief tier —
# the form hands off to /design/?source=<media_id>.
SOURCE_TIERS = ("rescue", "promote")
# Catalog-random Anima line. Prompt comes from catalogs/t2i; Studio exposes it as a third
# task (扭蛋), not a fifth generate-tier chip.
PROMPT_OPTIONAL_TIERS = (*SOURCE_TIERS, GACHA_TIER, GACHA_NSFW_TIER)
GACHA_KNOB_FIELDS = ("chaos", "blend_p", "style_lora_p", "gate_min", "overscale", "gacha_promote")
GACHA_SHARED_KNOB_FIELDS = ("style_lora_p", "gate_min", "overscale")
GACHA_SFW_ONLY_FIELDS = ("chaos", "blend_p", "gacha_promote")
GACHA_NSFW_ONLY_FIELDS = (
    "concept_p", "accent_p", "pin_char", "pin_exposure", "pin_behavior",
    "gacha_quality", "gacha_legacy_turbo",
)
GACHA_CHAR_SOURCES = ("lora_char", "oc_existing", "oc_llm")
GACHA_EXPOSURE_BANDS = ("covered_tease", "partial", "exposed", "nude")
GACHA_BEHAVIOR_BANDS = ("display", "solo", "explicit")
GACHA_REJECTED_FIELDS = {
    "prompt": "gacha2.0 does not take a prompt; catalogs/t2i supplies it",
    "base": "base is not selectable for gacha2.0 (Anima fast line is fixed)",
    "lora": "lora is not selectable for gacha2.0 (identity comes from the catalog)",
    "style_lora": "style_lora is not selectable for gacha2.0 (use style_lora_p)",
    "loras": "loras are not selectable for gacha2.0",
    "source_media_id": "source_media_id does not apply to gacha2.0",
    "ref_media_id": "ref_media_id does not apply to gacha2.0",
    "ref": "ref does not apply to gacha2.0",
    "turbo": "turbo is a draft-tier Anima flag; gacha2.0 does not take it",
    "pick_best": "gacha2.0 already runs a fail-closed wd14+qc+aesthetic gate",
}
GACHA_NSFW_REJECTED_FIELDS = {
    "prompt": "t2i_gacha_nsfw does not take a prompt; the remake sampler supplies it",
    "base": "base is not selectable for t2i_gacha_nsfw (Anima remake line is fixed)",
    "lora": "lora is not selectable for t2i_gacha_nsfw (identity comes from char_source)",
    "style_lora": "style_lora is not selectable for t2i_gacha_nsfw (use style_lora_p)",
    "loras": "loras are not selectable for t2i_gacha_nsfw",
    "source_media_id": "source_media_id does not apply to t2i_gacha_nsfw",
    "ref_media_id": "ref_media_id does not apply to t2i_gacha_nsfw",
    "ref": "ref does not apply to t2i_gacha_nsfw",
    "turbo": "turbo is a draft-tier Anima flag; t2i_gacha_nsfw uses gacha_quality / gacha_legacy_turbo",
    "pick_best": "t2i_gacha_nsfw already runs overscale + aesthetic floor (no wd14)",
    "chaos": "chaos is a gacha2.0 catalog knob; t2i_gacha_nsfw does not take it",
    "blend_p": "blend_p is a gacha2.0 catalog knob; t2i_gacha_nsfw does not take it",
    "gacha_promote": "gacha_promote is a gacha2.0 catalog knob; t2i_gacha_nsfw does not take it",
}
REF_MEDIA_TIERS = ("sequence", "artistic", QWEN21_TIER)
MEDIA_ID_RE = re.compile(r"^med_[0-9a-f]{24}$")
# A job id names the job's first image; ``.<k>`` names image k in result_image_candidates order.
JOB_ID_RE = re.compile(r"^(?:job_)?([0-9a-f]{12})(?:\.(\d{1,2}))?$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# Late-bound media resolver (app/server.py may replace it). None = default gallery+job lookup.
# Signature: (media_id, zone) -> Path. Raise MediaResolveError or return None to fail closed.
RESOLVE_MEDIA = None
# "auto" resolves at launch (gen.resolve_box): quick-scan BOXA first; if BOXA looks
# gaming/busy route to BOXB immediately, otherwise use BOXA (2026-08-16 policy).
BOXES = ("auto", "boxa", "boxb")
PROVIDERS = ("local", "civitai", "gpt_image")
EXTERNAL_PROVIDERS = ("civitai", "gpt_image")
EXTERNAL_FIELDS = {
    "tier", "prompt", "box", "provider", "confirm_paid_generation", "sfw",
    "n", "seed", "steps", "cfg", "res", "reference", "group", "channel",
    "base", "loras",  # Civitai only: checkpoint alias/AIR and LoRA AIR[:strength] specs
}
MAX_N = 16
MAX_EXTRA_LORAS = 4  # concept/pose/clothing stacks beyond this melt identity anyway

# Late-bound network kill-switch hook (app/server.py wires it to
# app.network.enforcement.brief_problems). None — the default for CLI and
# pytest imports of this module — means no network gating.
NETWORK_PROBLEMS = None

# Late-bound resource constraint hook (app/server.py wires it to
# gen.resource_problems_for_brief). None means resource-aware validation is off
# (CLI / pytest default).
RESOURCE_PROBLEMS = None
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
# Quality-line knobs (boxa_quality_gen only): the Anima turbo line behind `draft` has no
# --neg-add/--no-freeu/--pag, so a draft brief carrying them would die in argparse.
QUALITY_ONLY_FIELDS = ("neg_add", "no_freeu", "pag")
# MOSS knobs (app/moss.py checks them against the capability table); every other tier refuses them.
MOSS_ONLY_FIELDS = ("negative", "recipe", "shift", "vary_strength", "upscale")
# rescue_gen.py has a fixed recipe: it does not parse sampling overrides. A form
# that still holds steps/cfg/sampler/scheduler from a quality tier would otherwise
# pass validate and then die in argparse at launch.
RESCUE_FIXED_RECIPE_FIELDS = ("steps", "cfg", "sampler", "scheduler")
PAG_MAX = 10.0
# Explicit WxH. Artistic sizing goes through `aspect` instead: gen.py derives --res from the
# art_prompt recipe there, and a second --res would silently win over it.
RES_TIERS = ("best", "sequence", "draft", "forge", "promote")
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
    "source_media_id": "source_media_id does not apply to forge tier",
    "ref_media_id": "ref_media_id does not apply to forge tier (no IPAdapter on the Forge line)",
    "turbo": "turbo is a draft-tier Anima flag; forge tier does not take it",
}
# qwen21_gen.py takes prompt / ratio / size / seed / n / steps, or a reference image (ref / ref_media_id)
# for an edit, and nothing else; every other knob the form could carry is refused here so it can never
# be dropped on the floor at launch.
QWEN21_REJECTED_FIELDS = {
    key: f"{key} does not apply to the qwen21 tier (natural-language sentences on the BOXA sidecar; "
         "no LoRA, base, negative or SDXL knobs)"
    for key in (
        "base", "lora", "loras", "style_lora", "direction", "source_media_id",
        "turbo", "sfw", "ir", "cfg", "sampler", "scheduler", "res", "pick_best",
        *QUALITY_ONLY_FIELDS, *ARTISTIC_ONLY_FIELDS,
    )
}
QWEN21_DEFAULT_RATIO, QWEN21_DEFAULT_SIZE = "2:3", "std"
DRAFT_RES_PRESETS = ("portrait", "square", "landscape", "tall")  # boxa_anima_fast RES_PRESETS
RES_MIN, RES_MAX = 256, 4096
# --pick-best crowns the aesthetic best of N. gen.py refuses it on sequence; artistic already
# runs its own survivor filter + contact sheet, and draft is a throwaway turbo batch.
PICK_BEST_TIERS = ("best",)
PILOT_FIELDS = {"tier", "prompt", "box", "provider", "task_spec", "sfw"}
# No "prompt": the child's prompt is the parent's with minimal_delta written over it
# (revision.apply_delta). Accepting a resolved prompt alongside the delta would give two
# representations of the same value with nothing forcing them to agree, and the server would
# have to either trust the caller's copy or recompute it — at which point why accept it.
# No "sfw"/"nsfw" either: content rating is inherited from the parent. revision.py refuses
# them in the delta for the same reason, so accepting them here would just move the hole.
# "parent_path" is server-owned. The client names its parent by catalog identity
# (parent_media) because that is what the UI has and what /api/jobs/by-media speaks; the
# design router resolves that identity through gallery_catalog -- which already confines the
# result to the zone's source root -- and overwrites this field unconditionally before the
# job is stored. A value sent by a client is therefore replaced, never honoured. It lives in
# the brief rather than in a to_argv argument because app/jobs.py has one generic call site
# for every tier, and threading a per-tier keyword through it would put tier knowledge in the
# job runner.
REVISE_FIELDS = {
    "tier", "provider", "box", "keep_resident", "parent_media", "parent_path",
    "region", "allow_topology",
    "changed_dimensions", "frozen_dimensions", "minimal_delta", "rung", "scope",
    "needs_mask", "confidence", "confidence_note", "revision_id", "user_feedback",
    "risk_notes", "channel",
}
# Rung 7 (workflow topology) is off unless the server says otherwise. Same shape as the Pilot
# tier's FEATURE_ENV: a switch outside the request, because the caller asking for the
# escalation is the one that would otherwise be granting it.
REVISE_TOPOLOGY_ENV = "LORAMAKE_REVISE_ALLOW_TOPOLOGY"


def topology_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return (values.get(REVISE_TOPOLOGY_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


GALLERY_ZONES = ("sfw", "nsfw")
# Where POST /api/design/masks stores validated selections. A region brief may only name a
# file in here, so the field cannot be used to point the engine at an arbitrary image.
MASKS_DIR = (ROOT / "app" / "data" / "masks").resolve()
# What a delta may write: exactly the fields scripts/revise.py reads back out.
#
# This was derived from revision.FIELD_DIMENSION minus the routing and content fields, which
# admitted about fifteen names the engine never looks at -- neg_add, detailer, pose_ref, ref,
# style_lora, direction, scene, character, outfit, action, aspect among them. A proposal
# writing one of those passed every check, rendered with the identical seed and prompt, and
# returned a byte-identical picture under a sidecar asserting the dimension had changed. A
# silent no-op is worse than a refusal, so the list is now the engine's actual reach and
# anything outside it is rejected by name.
#
# The cost is that ladder rung 2 (revision.RUNG_FIELDS[2] == {"neg_add"}) has no reachable
# field until revise.py learns to compose a negative prompt. Being unusable and saying so
# beats appearing to work.
REVISE_ROUTING_FIELDS = frozenset({"tier", "box", "provider"})
REVISE_DELTA_FIELDS = frozenset({
    # subject and composition text, folded into the prompt through art_prompt.build
    "prompt", "artist", "medium", "camera", "composition", "lighting", "pose",
    "color_mood", "expression", "style_mode", "detail",
    # the recipe the child inherits and may override
    "base", "loras", "seed", "steps", "cfg", "sampler", "scheduler", "res",
    # region-only: how hard the masked area is repainted
    "denoise",
})
# Kept honest at import: every name above has to be classified, or revision.validate would
# reject it with "not a classified brief field" after this layer had already allowed it.
assert not (REVISE_DELTA_FIELDS - set(revision.FIELD_DIMENSION)), (
    "REVISE_DELTA_FIELDS names a field revision.py does not classify: "
    f"{sorted(REVISE_DELTA_FIELDS - set(revision.FIELD_DIMENSION))}"
)


class MediaResolveError(ValueError):
    """Opaque media/job id could not be resolved in the requested zone."""


def brief_zone(brief: dict) -> str:
    """SFW/NSFW partition for media resolution. Cross-zone ids fail closed."""
    return "nsfw" if brief.get("nsfw") is True else "sfw"


def _looks_like_path(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    return len(value) >= 2 and value[1] == ":" and value[0].isalpha()


def _require_image(path: Path, field: str) -> Path:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise MediaResolveError(f"{field} must resolve to an image file")
    if not path.is_file():
        raise MediaResolveError(f"{field} image is not a readable file")
    return path


def _brief_channel(brief: dict) -> str | None:
    """The channel stamp submit() wrote into the brief (None = unscoped/local caller)."""
    value = brief.get("channel")
    return value if isinstance(value, str) and value else None


def _require_channel_claim(path: Path, field: str, channel: str | None) -> None:
    """A channel-scoped brief may only touch media its own channel produced: the resolved
    file must be claimed by a same-channel job. Unscoped (local) briefs are unaffected."""
    if channel is None:
        return
    from app import jobs
    if jobs.find_job_by_file(str(path), channel=channel) is None:
        raise MediaResolveError(
            f"{field} resolves to a file no {channel!r} channel job produced; "
            "channel-scoped briefs may only reference their own channel's results"
        )


def _resolve_gallery_media(media_id: str, zone: str, field: str,
                           channel: str | None = None) -> Path:
    from app import gallery_catalog
    other = "nsfw" if zone == "sfw" else "sfw"
    try:
        path = gallery_catalog.asset_path(zone, media_id)
    except gallery_catalog.GalleryError as exc:
        if exc.status == 404:
            try:
                gallery_catalog.asset_path(other, media_id)
            except gallery_catalog.GalleryError:
                raise MediaResolveError(
                    f"{field} {media_id} was not found in the {zone} gallery"
                ) from exc
            raise MediaResolveError(
                f"{field} {media_id} belongs to the {other} zone; {zone} briefs cannot use it"
            ) from exc
        raise MediaResolveError(f"{field} {media_id}: {exc.detail}") from exc
    _require_channel_claim(path, field, channel)
    return _require_image(path, field)


def _resolve_job_media(job_id: str, zone: str, field: str,
                       channel: str | None = None, slot: int | None = None) -> Path:
    from app import jobs
    job = jobs.get_job(job_id)
    if job is None:
        raise MediaResolveError(f"{field} job {job_id} was not found")
    if channel is not None and jobs.job_channel(job) != channel:
        raise MediaResolveError(
            f"{field} job {job_id} belongs to another channel; "
            f"{channel!r} channel briefs may only use their own channel's jobs"
        )
    job_zone = "nsfw" if (job.get("brief") or {}).get("nsfw") is True else "sfw"
    if job_zone != zone:
        raise MediaResolveError(
            f"{field} job {job_id} is {job_zone}; {zone} briefs cannot use it"
        )
    if slot is None:
        path = jobs.primary_result_path(job)
    else:
        candidates = jobs.result_image_candidates(job.get("result") or {})
        path = Path(candidates[slot]) if slot < len(candidates) else None
        if path is not None and not path.is_file():
            path = None
    if path is None:
        where = "" if slot is None else f" at image {slot}"
        raise MediaResolveError(f"{field} job {job_id} has no usable image result{where}")
    if not jobs.path_in_zone(path, zone):
        raise MediaResolveError(
            f"{field} job {job_id} result is outside the {zone} outputs tree"
        )
    return _require_image(path, field)


def _with_original(path: Path) -> Path:
    """A source/anchor that arrived as an off-LAN preview is swapped for its original first;
    when the box is unreachable the preview is used and the gap is logged."""
    from loramake import remote_original  # noqa: PLC0415

    if remote_original.pending(path) and not remote_original.fetch(path):
        print(f"[media] using the preview of {path.name}: its original is still on the box",
              file=sys.stderr)
    return path


def resolve_media_id(media_id: object, *, zone: str, field: str = "media_id",
                     channel: str | None = None) -> Path:
    """Resolve an opaque gallery (`med_…`) or job id to a real image path. Fail closed.

    ``channel`` scopes the resolution to media that channel produced (see
    ``_require_channel_claim``); unscoped callers keep the historical behaviour."""
    if not isinstance(media_id, str) or not media_id.strip():
        raise MediaResolveError(f"{field} must be a non-empty opaque gallery or job id")
    mid = media_id.strip()
    if _looks_like_path(mid):
        raise MediaResolveError(
            f"{field} must be an opaque gallery or job id, not a filesystem path"
        )
    if RESOLVE_MEDIA is not None:
        path = RESOLVE_MEDIA(mid, zone)
        if path is None:
            raise MediaResolveError(f"{field} {mid} was not found in the {zone} zone")
        path = _require_image(Path(path), field)
        # channel claim first: another channel's preview must not trigger an original fetch
        _require_channel_claim(path, field, channel)
        return _with_original(path)
    if MEDIA_ID_RE.fullmatch(mid):
        return _with_original(_resolve_gallery_media(mid, zone, field, channel))
    job_match = JOB_ID_RE.fullmatch(mid)
    if job_match:
        slot = None if job_match.group(2) is None else int(job_match.group(2))
        return _with_original(_resolve_job_media(job_match.group(1), zone, field, channel, slot))
    raise MediaResolveError(
        f"{field} must be a gallery id (med_[0-9a-f]{{24}}) or job id, not {mid!r}"
    )


def qwen21_canvas() -> dict[tuple[str, str], tuple[int, int]]:
    """The engine's validated (ratio, size) -> (w, h) table (loramake.qwen21_canvas, shared with qwen21_gen)."""
    return dict(QWEN21_CANVAS)


def _qwen21_problems(brief: dict) -> list[str]:
    problems: list[str] = []
    if brief.get("provider", "local") != "local":
        problems.append("qwen21 tier requires provider=local")
    if brief.get("box", "auto") != "boxa":
        problems.append("qwen21 tier runs on boxa only (the Qwen-Image-2.1 sidecar); send box=boxa")
    if brief.get("nsfw"):
        problems.append("qwen21 tier has no nsfw router yet")
    for key, why in QWEN21_REJECTED_FIELDS.items():
        if brief.get(key) is not None:
            problems.append(why)
    if brief.get("preview") is not None and not isinstance(brief.get("preview"), bool):
        problems.append("preview must be a boolean (fetch the webp preview instead of the PNG)")
    ratio, size = brief.get("ratio"), brief.get("size")
    has_ref = bool(str(brief.get("ref") or "").strip() or str(brief.get("ref_media_id") or "").strip())
    if has_ref and (ratio is not None or size is not None):
        problems.append("ratio and size do not apply to a qwen21 edit (the canvas comes from the reference image)")
    elif (ratio is None) != (size is None):
        problems.append("ratio and size must be sent together")
    elif ratio is not None:
        canvas = qwen21_canvas()
        if not isinstance(ratio, str) or not isinstance(size, str) or (ratio, size) not in canvas:
            supported = ", ".join(f"{r}/{z}" for r, z in canvas)
            problems.append(f"ratio/size {ratio}/{size} is not a validated Qwen canvas; supported: {supported}")
    return problems


def _resolved_source_path(brief: dict) -> str | None:
    mid = brief.get("source_media_id")
    if not mid:
        return None
    return str(resolve_media_id(mid, zone=brief_zone(brief), field="source_media_id",
                                channel=_brief_channel(brief)))


def _resolved_ref_path(brief: dict) -> str | None:
    channel = _brief_channel(brief)
    mid = brief.get("ref_media_id")
    if mid:
        return str(resolve_media_id(mid, zone=brief_zone(brief), field="ref_media_id",
                                    channel=channel))
    ref = brief.get("ref")
    if not ref:
        return None
    if not isinstance(ref, str):
        raise MediaResolveError("ref must be a string")
    ref_path = Path(ref)
    if not ref_path.is_file():
        raise MediaResolveError(f"ref image not found: {ref}")
    zone = brief_zone(brief)
    zone_problem = _legacy_ref_zone_problem(ref_path, zone)
    if zone_problem:
        raise MediaResolveError(zone_problem)
    _require_channel_claim(ref_path.resolve(), "ref", channel)
    return str(ref_path)


def _legacy_ref_zone_problem(ref_path: Path, zone: str) -> str | None:
    """Cross-zone guard for legacy ref paths under product output trees only."""
    from app import jobs
    if jobs.path_in_zone(ref_path, "nsfw"):
        if zone != "nsfw":
            return f"ref belongs to the nsfw zone; {zone} briefs cannot use it"
    elif jobs.path_in_zone(ref_path, "sfw"):
        if zone != "sfw":
            return f"ref belongs to the sfw zone; {zone} briefs cannot use it"
    return None


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


def _unit_interval_problem(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
        return f"{name} must be a number in 0.0..1.0"
    return None


def _base_cards() -> dict:
    """model_quirks base cards (stem -> card); the selectable-base registry lives there."""
    try:
        return json.loads(QUIRKS.read_text(encoding="utf-8")).get("bases") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _revise_proposal(brief: dict) -> revision.RevisionProposal:
    """Build a proposal from an untrusted brief without asserting anything about it.

    Every field keeps whatever the caller sent, wrong type included, because revision.validate
    is the thing that reports type problems and it produces better messages than a TypeError
    raised here would.
    """
    return revision.RevisionProposal(
        changed_dimensions=brief.get("changed_dimensions") or [],
        frozen_dimensions=brief.get("frozen_dimensions") or [],
        minimal_delta=brief.get("minimal_delta"),
        rung=brief.get("rung"),
        scope=brief.get("scope", "whole"),
        needs_mask=bool(brief.get("needs_mask")),
        confidence=brief.get("confidence", "heuristic"),
        confidence_note=str(brief.get("confidence_note") or ""),
        revision_id=str(brief.get("revision_id") or ""),
        user_feedback=str(brief.get("user_feedback") or ""),
        risk_notes=brief.get("risk_notes") or [],
    )


def _revise_parent_problems(parent: object) -> list[str]:
    """The parent image must resolve, or the freeze declaration has nothing to be true about."""
    if not isinstance(parent, dict):
        return ["parent_media is required and must be an object with zone and media_id"]
    problems = []
    for key in ("zone", "media_id"):
        if not str(parent.get(key) or "").strip():
            problems.append(f"parent_media.{key} is required")
    zone = parent.get("zone")
    # gallery_catalog refuses anything else at resolution time; checking it here puts the
    # failure where every other brief problem lands instead of surfacing as a catalog error.
    if zone is not None and zone not in GALLERY_ZONES:
        problems.append(f"parent_media.zone must be one of {', '.join(GALLERY_ZONES)}")
    for key in sorted(set(parent) - {"zone", "media_id"}):
        problems.append(f"parent_media.{key} is not accepted")
    return problems


def _gallery_roots(zone: object) -> list[Path]:
    """The scanner roots of a gallery zone: where every picture the catalog serves for it lives."""
    from app import gallery_catalog  # imports gallery_scan with scripts/ on the path
    return [Path(source["root"]).resolve() for source in gallery_catalog.gallery_scan.SOURCES
            if source["rating"] == zone]


# Late-bound like RESOLVE_MEDIA so tests can point it at a fixture tree. Signature: zone -> roots.
REVISE_PARENT_ROOTS = _gallery_roots


def _revise_parent_path_problems(path: Path, parent: object) -> list[str]:
    """parent_path must be a gallery picture in parent_media's zone.

    The design router derives it from the catalog, but other in-process callers merge client
    fields into a brief before jobs.submit (recipe overrides do), so the field can still arrive
    from a caller. Holding it to the zone's scanner roots keeps it to pictures the catalog serves
    anyway, and on the parent's side of the sfw/nsfw line.
    """
    zone = parent.get("zone") if isinstance(parent, dict) else None
    resolved = path.resolve()
    if any(resolved.is_relative_to(root) for root in REVISE_PARENT_ROOTS(zone)):
        return []
    return [f"parent_path must be a {zone} gallery image"]


def _revise_region_problems(brief: dict, proposal: revision.RevisionProposal) -> list[str]:
    """Structural agreement between scope, needs_mask and region.

    Only the parts that can be decided from the brief alone. Whether the mask actually lines
    up with the parent image — matching pixel size, a non-empty selection, coverage below the
    whole-image threshold — needs both files open and is enforced in loramake.region_mask at
    apply time, where a mismatch is raised rather than warned about.
    """
    problems: list[str] = []
    region = brief.get("region")
    if region is None:
        if proposal.needs_mask:
            problems.append("needs_mask requires a region with a mask_path")
        return problems
    if proposal.scope != "region":
        problems.append('region is only accepted with scope="region"')
    if not isinstance(region, dict):
        return [*problems, "region must be an object with a mask_path"]
    for key in sorted(set(region) - {"mask_path"}):
        problems.append(f"region.{key} is not accepted")
    mask_path = region.get("mask_path")
    if not str(mask_path or "").strip():
        problems.append("region.mask_path is required")
        return problems
    # Confined to the directory POST /api/design/masks writes to. Without this the field names
    # any readable file and the engine uploads it to the box, so a brief could exfiltrate an
    # arbitrary image by asking for it to be used as a selection.
    try:
        resolved = Path(str(mask_path)).resolve()
        inside = resolved.is_relative_to(MASKS_DIR)
    except (OSError, ValueError):
        resolved, inside = None, False
    if not inside:
        problems.append(f"region.mask_path must be a mask stored by /api/design/masks: {mask_path}")
    elif not resolved.is_file():
        problems.append(f"region mask not found: {mask_path}")
    return problems


def validate(brief: object) -> list[str]:
    """Boundary validation of an untrusted brief. Returns problems; empty list = valid."""
    problems: list[str] = []
    if not isinstance(brief, dict):
        return ["brief must be a JSON object"]
    if NETWORK_PROBLEMS is not None:
        problems.extend(NETWORK_PROBLEMS(brief))
    if RESOURCE_PROBLEMS is not None:
        problems.extend(RESOURCE_PROBLEMS(brief))
    tier = brief.get("tier")
    provider = brief.get("provider", "local")
    if provider not in PROVIDERS:
        problems.append(f"provider must be one of {', '.join(PROVIDERS)}")
    if tier not in TIERS:
        problems.append(f"tier must be one of {', '.join(TIERS)}")
    prompt = brief.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        problems.append("prompt must be a string")
    # Every other tier states the picture it wants; a revise brief inherits it from the parent
    # and says only what changes, so demanding a prompt here would force the caller to send a
    # resolved copy that nothing checks against the delta.
    if tier not in PROMPT_OPTIONAL_TIERS and tier != REVISE_TIER and not str(prompt or "").strip():
        problems.append("prompt is required")
    if brief.get("box", "auto") not in BOXES:
        problems.append(f"box must be one of {', '.join(BOXES)}")
    if tier == "draft" and brief.get("turbo") is True and brief.get("box", "auto") not in ("boxa", "auto"):
        problems.append("draft turbo runs on boxa only (2.9B baked UNET); auto resolves to boxa")
    if tier == GACHA_TIER and brief.get("box", "auto") not in ("boxa", "auto"):
        problems.append("gacha2.0 runs on boxa only (catalog Anima line); auto resolves to boxa")
    if tier == GACHA_NSFW_TIER and brief.get("box", "auto") not in ("boxa", "auto"):
        problems.append("t2i_gacha_nsfw runs on boxa only (Anima remake line); auto resolves to boxa")
    if tier == "forge" and brief.get("box", "auto") not in ("boxa", "auto"):
        problems.append("forge tier runs on boxa only (the Forge deployment lives on BOXA); auto resolves to boxa")
    if tier == "rescue" and brief.get("box", "auto") not in ("boxb", "auto"):
        problems.append("rescue tier runs on boxb only (pose-lock line); auto resolves to boxb")
    if tier == QWEN21_TIER:
        problems.extend(_qwen21_problems(brief))
    elif tier != MOSS_TIER and (brief.get("ratio") is not None or brief.get("size") is not None):
        problems.append("ratio and size are only valid for the qwen21 tier and the moss tier")
    if brief.get("source") is not None:
        problems.append("source is not accepted; send source_media_id (opaque id), never a path")
    source_media_id = brief.get("source_media_id")
    if source_media_id is not None:
        if tier not in (*SOURCE_TIERS, MOSS_TIER):
            problems.append("source_media_id is only valid for rescue, promote and moss")
        else:
            try:
                resolve_media_id(source_media_id, zone=brief_zone(brief), field="source_media_id",
                                 channel=_brief_channel(brief))
            except MediaResolveError as exc:
                problems.append(str(exc))
    elif tier in SOURCE_TIERS:
        problems.append(f"{tier} tier requires source_media_id (opaque gallery or job id)")
    ref_media_id = brief.get("ref_media_id")
    if ref_media_id is not None:
        if tier not in REF_MEDIA_TIERS:
            problems.append(f"ref_media_id is only valid for {', '.join(REF_MEDIA_TIERS)}")
        else:
            try:
                resolve_media_id(ref_media_id, zone=brief_zone(brief), field="ref_media_id",
                                 channel=_brief_channel(brief))
            except MediaResolveError as exc:
                problems.append(str(exc))
    ref = brief.get("ref")
    if tier == "sequence" and not str(ref_media_id or "").strip() and not str(ref or "").strip():
        problems.append(
            "sequence tier requires ref_media_id (opaque identity-anchor id) or legacy ref path"
        )
    if ref is not None:
        if not isinstance(ref, str):
            problems.append("ref must be a string")
        elif ref.strip():
            if ref_media_id:
                problems.append("send ref_media_id only; do not also send a ref path")
            elif tier not in REF_MEDIA_TIERS:
                problems.append(f"ref is only valid for {', '.join(REF_MEDIA_TIERS)}")
            else:
                ref_path = Path(ref)
                if not ref_path.is_file():
                    problems.append(f"ref image not found: {ref}")
                else:
                    zone_problem = _legacy_ref_zone_problem(ref_path, brief_zone(brief))
                    if zone_problem:
                        problems.append(zone_problem)
                    else:
                        try:
                            _require_channel_claim(ref_path.resolve(), "ref", _brief_channel(brief))
                        except MediaResolveError as exc:
                            problems.append(str(exc))
    turbo = brief.get("turbo")
    if turbo is not None:
        if not isinstance(turbo, bool):
            problems.append("turbo must be a boolean")
        elif turbo and tier != "draft":
            problems.append("turbo is only valid for draft tier (2.9B baked)")
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
            *(() if provider == "civitai" else ("base", "loras")),
            "lora", "style_lora", "direction", "ref", "sampler", "scheduler",
            "neg_add", "no_freeu", "pag", "pick_best", "aspect", "detail", "style_mode",
            "keep_resident", "source_media_id", "ref_media_id", "turbo", *ARTISTIC_TEXT_FIELDS,
        )
        for key in unsupported:
            if brief.get(key) is not None:
                problems.append(f"{key} is not supported by external providers")
        if provider == "civitai":
            problems.extend(_civitai_resource_problems(brief))
        problems.extend(_group_channel_problems(brief))
        if n is not None:
            maximum = 12 if provider == "civitai" else 10
            if isinstance(n, int) and not isinstance(n, bool) and n > maximum:
                problems.append(f"{provider} n must be in 1..{maximum}")
        if provider == "gpt_image" and seed is not None:
            problems.append("seed is not supported by the GPT Image adapter")
        reference = brief.get("reference")
        if reference is not None:
            if provider != "gpt_image":
                problems.append("reference is only supported by the GPT Image adapter")
            elif not isinstance(reference, str) or not reference.strip():
                problems.append("reference must be an image path")
            elif not Path(reference).is_file():
                problems.append(f"reference image not found: {reference}")
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

    if tier == REVISE_TIER:
        for key in sorted(set(brief) - REVISE_FIELDS):
            problems.append(f"{key} is not accepted in {REVISE_TIER} briefs")
        if provider != "local":
            problems.append(f"{REVISE_TIER} runs on the local engine only")
        allow_topology = brief.get("allow_topology")
        if allow_topology is not None and not isinstance(allow_topology, bool):
            problems.append("allow_topology must be a boolean")
        # The flag rides in the request body, and the body is written by the chat agent, so on
        # its own it is the proposal authorising its own escalation -- the same defect as a
        # client-supplied parent_path. Rung 7 additionally needs a switch the request cannot
        # reach, which is where a human decision can actually live; the Pilot tier gates itself
        # the same way a few lines below.
        if allow_topology is True and not topology_enabled():
            problems.append(
                f"allow_topology needs {REVISE_TOPOLOGY_ENV}=1 on the server; rung "
                f"{revision.TOPOLOGY_RUNG} is not a decision the request can make for itself"
            )
        delta_res = (brief.get("minimal_delta") or {}).get("res") \
            if isinstance(brief.get("minimal_delta"), dict) else None
        if delta_res is not None:
            # res is the one rung-7 field the delta whitelist admits, and the revise branch
            # returns before the shared res check further down runs. Unbounded, it reaches the
            # sampler as the latent size: "20000x20000" is a plausible-looking string that
            # allocates until the box dies.
            res_problem = _res_problem(delta_res, "best")
            if res_problem:
                problems.append(res_problem)
        problems.extend(_revise_parent_problems(brief.get("parent_media")))
        # Required, not optional: to_argv has to put a real path behind --parent-image, and a
        # brief that validated without one would build a command with an empty flag value that
        # only fails later, inside the engine. The design router resolves parent_media through
        # the catalog and writes this field before it validates, so a 404 on an unknown media
        # id surfaces as a catalog error rather than as a brief problem.
        if (brief.get("parent_media") or {}).get("zone") == "nsfw" and not GEN_NSFW.is_file():
            problems.append("NSFW pipeline not installed (nsfw/gen_nsfw.py missing)")
        parent_path = brief.get("parent_path")
        if not str(parent_path or "").strip():
            problems.append("parent_path is required (the router resolves it from parent_media)")
        elif not Path(str(parent_path)).is_file():
            problems.append(f"parent image not found: {parent_path}")
        else:
            problems.extend(_revise_parent_path_problems(Path(str(parent_path)), brief.get("parent_media")))
        proposal = _revise_proposal(brief)
        # revision.py owns the dimension algebra: unknown dimensions, changed/frozen overlap,
        # the two-dimension ceiling, delta fields writing a frozen dimension, and the ladder
        # rung a field actually needs. Restating any of it here would give two answers.
        problems.extend(revision.validate(
            proposal,
            known_fields=REVISE_DELTA_FIELDS,
            allow_topology=allow_topology is True,
        ))
        problems.extend(_revise_region_problems(brief, proposal))
        return problems

    if tier == PILOT_TIER:
        for key in sorted(set(brief) - PILOT_FIELDS):
            problems.append(f"{key} is not accepted in {PILOT_TIER} briefs")
        if provider != "local":
            problems.append(f"{PILOT_TIER} requires provider=local")
        if brief.get("box", "boxb") != "boxb":
            problems.append(f"{PILOT_TIER} runs on boxb only")
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

    if tier == MOSS_TIER:
        from app import moss
        # app/moss.py checks every field against the capability table; the SDXL free layer below
        # (steps 1..60, ENGINE_SAMPLERS, ...) does not describe this engine.
        problems.extend(moss.problems(brief))
        problems.extend(_group_channel_problems(brief))
        return problems
    for key in MOSS_ONLY_FIELDS:
        if brief.get(key) is not None:
            problems.append(f"{key} is only valid for the {MOSS_TIER} tier")

    if confirmation is True:
        problems.append("confirm_paid_generation only applies to external providers")
    if nsfw:
        if not GEN_NSFW.is_file():
            problems.append("NSFW pipeline not installed (nsfw/gen_nsfw.py missing)")
        if brief.get("sfw") is True:
            problems.append("sfw gate is mutually exclusive with nsfw mode")
        if tier == "forge":
            problems.append("forge tier is not wired into the nsfw router yet")
        if tier == GACHA_TIER:
            problems.append("gacha2.0 is the SFW catalog line; NSFW remake is nsfw/t2i_gacha_nsfw.py")
    if tier == GACHA_NSFW_TIER and nsfw is not True:
        problems.append("t2i_gacha_nsfw requires nsfw=true (independent remake; not gacha2.0)")
    if tier == GACHA_TIER:
        if provider != "local":
            problems.append("gacha2.0 requires provider=local")
        if str(prompt or "").strip():
            problems.append(GACHA_REJECTED_FIELDS["prompt"])
        for key, why in GACHA_REJECTED_FIELDS.items():
            if key == "prompt":
                continue
            if brief.get(key) is not None:
                problems.append(why)
        for key in ("chaos", "blend_p", "style_lora_p", "gate_min"):
            unit_problem = _unit_interval_problem(brief.get(key), key)
            if unit_problem:
                problems.append(unit_problem)
        overscale = brief.get("overscale")
        if overscale is not None and (
                not isinstance(overscale, (int, float)) or isinstance(overscale, bool) or overscale < 1.0):
            problems.append("overscale must be a number >= 1")
        gacha_promote = brief.get("gacha_promote")
        if gacha_promote is not None and not isinstance(gacha_promote, bool):
            problems.append("gacha_promote must be a boolean")
        for key in (*QUALITY_ONLY_FIELDS, *RESCUE_FIXED_RECIPE_FIELDS, "res"):
            if brief.get(key) is not None:
                problems.append(f"{key} is not available on gacha2.0 (catalog Anima line)")
    if tier == GACHA_NSFW_TIER:
        if not GEN_GACHA_NSFW.is_file():
            problems.append("NSFW remake not installed (nsfw/t2i_gacha_nsfw.py missing)")
        if provider != "local":
            problems.append("t2i_gacha_nsfw requires provider=local")
        if str(prompt or "").strip():
            problems.append(GACHA_NSFW_REJECTED_FIELDS["prompt"])
        for key, why in GACHA_NSFW_REJECTED_FIELDS.items():
            if key == "prompt":
                continue
            if brief.get(key) is not None:
                problems.append(why)
        for key in ("style_lora_p", "concept_p", "accent_p", "gate_min"):
            unit_problem = _unit_interval_problem(brief.get(key), key)
            if unit_problem:
                problems.append(unit_problem)
        overscale = brief.get("overscale")
        if overscale is not None and (
                not isinstance(overscale, (int, float)) or isinstance(overscale, bool) or overscale < 1.0):
            problems.append("overscale must be a number >= 1")
        pin_char = brief.get("pin_char")
        if pin_char is not None:
            if not isinstance(pin_char, str) or not pin_char.strip():
                problems.append("pin_char must be a non-empty string (char_source or card id)")
        pin_exposure = brief.get("pin_exposure")
        if pin_exposure is not None and pin_exposure not in GACHA_EXPOSURE_BANDS:
            problems.append(
                f"pin_exposure must be one of {', '.join(GACHA_EXPOSURE_BANDS)}"
            )
        pin_behavior = brief.get("pin_behavior")
        if pin_behavior is not None and pin_behavior not in GACHA_BEHAVIOR_BANDS:
            problems.append(
                f"pin_behavior must be one of {', '.join(GACHA_BEHAVIOR_BANDS)}"
            )
        gacha_quality = brief.get("gacha_quality")
        if gacha_quality is not None and not isinstance(gacha_quality, bool):
            problems.append("gacha_quality must be a boolean")
        gacha_legacy_turbo = brief.get("gacha_legacy_turbo")
        if gacha_legacy_turbo is not None and not isinstance(gacha_legacy_turbo, bool):
            problems.append("gacha_legacy_turbo must be a boolean")
        if gacha_quality is True and gacha_legacy_turbo is True:
            problems.append("gacha_quality and gacha_legacy_turbo are mutually exclusive")
        for key in (*QUALITY_ONLY_FIELDS, *RESCUE_FIXED_RECIPE_FIELDS, "res"):
            if brief.get(key) is not None:
                problems.append(f"{key} is not available on t2i_gacha_nsfw (Anima remake line)")
    if tier != GACHA_TIER:
        for key in GACHA_SFW_ONLY_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is only valid for gacha2.0")
    if tier != GACHA_NSFW_TIER:
        for key in GACHA_NSFW_ONLY_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is only valid for {GACHA_NSFW_TIER}")
    if tier not in (GACHA_TIER, GACHA_NSFW_TIER):
        for key in GACHA_SHARED_KNOB_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is only valid for gacha2.0 or {GACHA_NSFW_TIER}")
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
            engine_family = TIER_FAMILY.get(tier, "illustrious")
            if (card.get("family") or "illustrious") != engine_family:
                problems.append(f"base {base} is not an {engine_family}-family checkpoint")
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
    if tier == "rescue":
        for key in QUALITY_ONLY_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is not available on rescue tier (fixed recipe)")
        for key in RESCUE_FIXED_RECIPE_FIELDS:
            if brief.get(key) is not None:
                problems.append(f"{key} is not available on rescue tier (fixed recipe)")
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
    if pose_ref and brief.get("box", "auto") == "boxa":
        # pose-library line is BOXB-only (BOXA's Illustrious ignores openpose); "auto" is fine —
        # the box resolver prefers boxb, so it only fails loudly on an EXPLICIT boxa pin.
        problems.append("pose_ref requires box boxb (pose-library line is BOXB-only)")
    extra = brief.get("loras")
    if extra is not None:
        if (not isinstance(extra, list)
                or not all(isinstance(x, str) and x.strip() for x in extra)):
            problems.append("loras must be a list of LoRA spec strings (FILE[:WEIGHT])")
        elif len(extra) > MAX_EXTRA_LORAS:
            problems.append(f"loras: at most {MAX_EXTRA_LORAS} extra LoRAs per job")
    problems.extend(_group_channel_problems(brief))
    return problems


def _group_channel_problems(brief: dict) -> list[str]:
    problems = []
    for key in ("group", "channel"):
        value = brief.get(key)
        if value is not None and (
                not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value)):
            problems.append(f"{key} must match [A-Za-z0-9_-]{{1,40}}")
    return problems


def _civitai_resource_problems(brief: dict) -> list[str]:
    """Checkpoint alias/AIR and LoRA specs, resolved here so a bad one never reaches the queue."""
    base = brief.get("base")
    if base is not None and (not isinstance(base, str) or not base.strip()):
        return ["base must be a Civitai model alias or checkpoint AIR"]
    loras = brief.get("loras")
    if loras is not None and (
            not isinstance(loras, list) or not all(isinstance(x, str) and x.strip() for x in loras)):
        return ["loras must be a list of Civitai LoRA AIR[:strength] strings"]
    try:
        resolved = civitai_cloud.resolve_model(base)
        civitai_cloud.parse_loras(loras, resolved["ecosystem"], preset=resolved["loras"])
    except (OSError, ValueError) as exc:
        return [str(exc)]
    return []


def to_argv(brief: dict, json_out: str | Path) -> list[str]:
    """The gen.py invocation for a validated brief. Always --run --json-out (the app reads
    results from the JSON file, never from stdout). nsfw briefs route to the gen_nsfw
    wrapper — same CLI, output rooted at nsfw/outputs/, rating=nsfw injected."""
    provider = brief.get("provider", "local")
    if brief.get("tier") == REVISE_TIER:
        # One JSON argument rather than a flag per field: the revision payload is a nested
        # object (parent_media, region, the delta itself) that no flat argv can carry without
        # inventing an encoding, and the pilot tier already set this precedent next door.
        payload = {key: value for key, value in brief.items()
                   if key not in {"tier", "provider", "box", "keep_resident", "parent_path", "channel"}}
        # Content rating is inherited from the parent, so the zone it came from -- not an
        # nsfw flag the brief is not allowed to carry -- decides the router. Without this an
        # nsfw parent's child ran through the shared engine and landed in the SFW output tree,
        # where gallery_scan would index it as rating=sfw.
        parent_zone = (brief.get("parent_media") or {}).get("zone")
        engine = GEN_NSFW if parent_zone == "nsfw" else GEN
        return [
            sys.executable,
            str(engine),
            REVISE_TIER,
            "--box",
            brief.get("box", "auto"),
            "--source",
            str(brief.get("parent_path") or ""),
            "--revision-json",
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "--run",
            "--json-out",
            str(json_out),
        ]
    if brief.get("tier") == PILOT_TIER:
        return [
            sys.executable,
            str(GEN),
            PILOT_TIER,
            "--box",
            "boxb",
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
                "--box", brief.get("box") or "boxb"]
        if brief.get("n") is not None:
            argv += ["--n", str(brief["n"])]
        if brief.get("seed") is not None:
            argv += ["--seed", str(brief["seed"])]
        if brief.get("confirm_paid_generation") is True:
            argv += ["--confirm-paid-generation"]
        argv += ["--run", "--json-out", str(json_out)]
        provider_params: list[str] = []
        for key in ("steps", "cfg", "res", "reference"):
            if brief.get(key) is not None:
                provider_params += [f"--{key}", str(brief[key])]
        if brief.get("base"):
            provider_params += ["--model", str(brief["base"]).strip()]
        for spec in brief.get("loras") or []:
            provider_params += ["--lora", spec.strip()]
        if provider_params:
            argv += ["--", *provider_params]
        return argv

    nsfw = bool(brief.get("nsfw"))
    tier = brief["tier"]
    if tier == GACHA_NSFW_TIER:
        from app.gacha_nsfw import to_argv as gacha_nsfw_to_argv
        return gacha_nsfw_to_argv(brief, json_out)
    if tier == MOSS_TIER:
        from app import moss
        return moss.to_argv(brief, json_out, _resolved_source_path(brief))
    if tier == QWEN21_TIER:
        argv = [sys.executable, str(GEN_QWEN21), "--prompt", str(brief["prompt"]).strip(), "--box", "boxa"]
        ref_path = _resolved_ref_path(brief)
        if ref_path:  # edit: the engine sizes the canvas from the reference, so no ratio/size travels
            argv += ["--ref", ref_path]
        else:
            argv += ["--ratio", str(brief.get("ratio") or QWEN21_DEFAULT_RATIO),
                     "--size", str(brief.get("size") or QWEN21_DEFAULT_SIZE)]
        for key in ("seed", "n", "steps"):
            if brief.get(key) is not None:
                argv += [f"--{key}", str(brief[key])]
        if brief.get("preview"):  # remote link: fetch the sidecar's webp preview, the PNG stays on BOXA
            argv += ["--fetch", "webp"]
        return argv + ["--run", "--json-out", str(json_out)]
    if tier == GACHA_TIER:
        argv = [sys.executable, str(GEN), GACHA_TIER, "--box", brief.get("box") or "boxa"]
        if brief.get("n") is not None:
            argv += ["--n", str(brief["n"])]
        if brief.get("seed") is not None:
            argv += ["--seed", str(brief["seed"])]
        argv += ["--run", "--json-out", str(json_out)]
        engine_params: list[str] = []
        for key, flag in (
            ("chaos", "--chaos"),
            ("blend_p", "--blend-p"),
            ("style_lora_p", "--style-lora-p"),
            ("gate_min", "--gate-min"),
            ("overscale", "--overscale"),
        ):
            if brief.get(key) is not None:
                engine_params += [flag, str(brief[key])]
        if brief.get("gacha_promote") is True:
            engine_params.append("--promote")
        if engine_params:
            argv += ["--", *engine_params]
        return argv
    argv = [sys.executable, str(GEN_NSFW if nsfw else GEN), tier]
    prompt = str(brief.get("prompt") or "").strip()
    if prompt:
        argv += ["--prompt", prompt]
    ir = brief.get("ir")
    if isinstance(ir, str) and ir.strip():
        argv += ["--ir", ir.strip()]
    argv += ["--box", brief.get("box", "auto")]
    source_path = _resolved_source_path(brief)
    if source_path:
        argv += ["--source", source_path]
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
    ref_path = _resolved_ref_path(brief)
    if ref_path:
        argv += ["--ref", ref_path]
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
    if tier != "rescue":
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
        if brief.get("turbo") is True:
            engine_params += ["--turbo"]
    if engine_params:
        argv += ["--", *engine_params]
    return argv
