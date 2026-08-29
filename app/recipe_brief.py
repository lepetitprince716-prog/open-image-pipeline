"""Recipe DTO normalization and recipe -> gen brief conversion.

Pure functions shared by the Studio HTTP surface (app/routers/civitai_bridge.py) and
the CLI entry (scripts/recipe_job.py) — the same role app/brief.py plays for gen_job.
Input is the LoRA Manager recipe shape (GET /api/lm/recipes on Z3); output is either a
stable camelCase DTO or a gen brief that MUST still pass app.brief.validate at the call
site. Nothing here executes or performs I/O.
"""
from __future__ import annotations

import re
from typing import Any

from app import brief as brief_mod

RECIPE_TIER_DEFAULT = "best"

# A1111-ecosystem inline syntax that ComfyUI engines would render as literal text.
_INLINE_LORA = re.compile(r"<lora:[^>]+>", re.IGNORECASE)
_INLINE_EXTENSION = re.compile(r"<(?:segment|hypernet|lyco)\b[^>]*>", re.IGNORECASE)

# Civitai/A1111 sampler labels -> gen.py engine sampler names (app.brief.ENGINE_SAMPLERS).
SAMPLER_MAP = {
    "euler": "euler",
    "euler a": "euler_ancestral",
    "euler_a": "euler_ancestral",
    "euler ancestral": "euler_ancestral",
    "euler_ancestral": "euler_ancestral",
    "dpm++ 2m": "dpmpp_2m",
    "dpmpp_2m": "dpmpp_2m",
    "dpm++ 2m karras": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpm++ 2m sde karras": "dpmpp_2m_sde",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpmpp_3m_sde": "dpmpp_3m_sde",
    "dpm++ sde": "dpmpp_sde",
    "dpmpp_sde": "dpmpp_sde",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
    "unipc": "uni_pc",
    "lcm": "lcm",
}

# LM sometimes stores a scheduler name in the sampler slot (observed live: "karras").
_SCHEDULER_NAMES = set(brief_mod.ENGINE_SCHEDULERS)

_NSFW_PROMPT_RE = re.compile(
    r"\b(?:nsfw|adult)\b",
    re.IGNORECASE,
)


def _as_int(value: Any) -> int | None:
    if value is None or value is False or value is True:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_stem(file_name: Any) -> str:
    text = str(file_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    if text.lower().endswith(".safetensors"):
        text = text[: -len(".safetensors")]
    return text


def recipe_from_lm_item(item: dict | None) -> dict[str, Any]:
    """One LM /api/lm/recipes item -> stable camelCase DTO."""
    if not isinstance(item, dict):
        return {}
    loras: list[dict[str, Any]] = []
    for raw in item.get("loras") or []:
        if not isinstance(raw, dict):
            continue
        version_id = _as_int(raw.get("modelVersionId"))
        if version_id == 0:
            version_id = None
        strength = _as_float(raw.get("strength"))
        loras.append({
            "fileName": str(raw.get("file_name") or ""),
            "stem": _file_stem(raw.get("file_name")),
            "hash": str(raw.get("hash") or "").lower() or None,
            # 0.0 is a real recorded weight, not a missing value — `or` would promote it to 1.0
            "strength": 1.0 if strength is None else strength,
            "modelVersionId": version_id,
            "modelName": str(raw.get("modelName") or ""),
            "inLibrary": bool(raw.get("inLibrary")),
            "isDeleted": bool(raw.get("isDeleted")),
            "excluded": bool(raw.get("exclude")),
            "localPath": raw.get("localPath"),
        })
    gen = item.get("gen_params") if isinstance(item.get("gen_params"), dict) else {}
    size = str(gen.get("size") or "").strip()
    width = _as_int(item.get("width"))
    height = _as_int(item.get("height"))
    if size and "x" in size:
        left, _, right = size.partition("x")
        width = _as_int(left) or width
        height = _as_int(right) or height
    dto = {
        "id": str(item.get("id") or ""),
        "title": str(item.get("title") or ""),
        "sourceUrl": str(item.get("source_path") or ""),
        "baseModel": str(item.get("base_model") or ""),
        "fingerprint": str(item.get("fingerprint") or ""),
        "created": item.get("created_date"),
        "modified": item.get("modified"),
        "previewUrl": item.get("file_url"),
        "previewNsfwLevel": _as_int(item.get("preview_nsfw_level")) or 0,
        "width": width,
        "height": height,
        "loras": loras,
        "genParams": {
            "prompt": str(gen.get("prompt") or ""),
            "negativePrompt": str(gen.get("negative_prompt") or ""),
            "sampler": str(gen.get("sampler") or ""),
            "steps": _as_int(gen.get("steps")),
            "cfgScale": _as_float(gen.get("cfg_scale")),
            "seed": _as_int(gen.get("seed")),
            "size": size or None,
        },
    }
    dto["missingCount"] = len(missing_loras(dto))
    return dto


def missing_loras(recipe_dto: dict) -> list[dict[str, Any]]:
    """LoRAs the recipe needs but Z3 does not have (excluded ones don't count)."""
    out: list[dict[str, Any]] = []
    for lora in recipe_dto.get("loras") or []:
        if lora.get("excluded"):
            continue
        if lora.get("inLibrary") and not lora.get("isDeleted"):
            continue
        entry = dict(lora)
        if entry.get("modelVersionId") is None:
            entry["skipped"] = "no version id"
        out.append(entry)
    return out


def looks_nsfw(recipe_dto: dict, site: str | None = None) -> bool:
    if site == "red":
        return True
    if str(recipe_dto.get("sourceUrl") or "").startswith("https://civitai.red"):
        return True
    if (recipe_dto.get("previewNsfwLevel") or 0) >= 4:  # civitai bitmask: 4=R and above
        return True
    prompt = str((recipe_dto.get("genParams") or {}).get("prompt") or "")
    return bool(_NSFW_PROMPT_RE.search(prompt))


def _clean_prompt(prompt: str, warnings: list[str]) -> str:
    cleaned = _INLINE_LORA.sub("", prompt)
    cleaned = _INLINE_EXTENSION.sub("", cleaned)
    if cleaned != prompt:
        warnings.append("stripped A1111 inline syntax (<lora:...>/<segment:...>) from the prompt")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,\n")


def _map_sampler(raw: str, warnings: list[str]) -> tuple[str | None, str | None]:
    """-> (sampler, scheduler); LM sometimes stores a scheduler name in the sampler slot."""
    key = raw.strip().lower()
    if not key:
        return None, None
    if key in _SCHEDULER_NAMES:
        warnings.append(f"recipe sampler '{raw}' is a scheduler name; applied as scheduler")
        return None, key
    mapped = SAMPLER_MAP.get(key)
    if mapped is None and key.endswith(" karras"):
        # Composite labels ("euler a karras") without an explicit table entry.
        mapped = SAMPLER_MAP.get(key[: -len(" karras")].strip())
        if mapped:
            return mapped, "karras"
    if mapped is None:
        warnings.append(f"sampler '{raw}' has no engine equivalent; dropped")
        return None, None
    scheduler = "karras" if "karras" in key else None
    return mapped, scheduler


def brief_from_recipe(recipe_dto: dict, overrides: dict | None = None) -> tuple[dict[str, Any], list[str]]:
    """Recipe DTO -> gen brief + warnings. Caller MUST run app.brief.validate on the result."""
    warnings: list[str] = []
    gen = recipe_dto.get("genParams") or {}
    brief: dict[str, Any] = {"tier": RECIPE_TIER_DEFAULT}

    prompt = _clean_prompt(str(gen.get("prompt") or ""), warnings)
    if prompt:
        brief["prompt"] = prompt

    negative = str(gen.get("negativePrompt") or "").strip()
    if negative:
        brief["neg_add"] = negative
        warnings.append("recipe negative prompt applied additively (neg_add)")

    usable = [
        lora for lora in recipe_dto.get("loras") or []
        if not lora.get("excluded") and lora.get("inLibrary") and not lora.get("isDeleted")
    ]
    missing = missing_loras(recipe_dto)
    if missing:
        warnings.append(
            f"{len(missing)} LoRA(s) not in the Z3 library were left out: "
            + ", ".join(sorted(str(m.get("modelName") or m.get("stem") or "?") for m in missing))
        )
    if len(usable) > brief_mod.MAX_EXTRA_LORAS:
        dropped = usable[brief_mod.MAX_EXTRA_LORAS:]
        usable = usable[: brief_mod.MAX_EXTRA_LORAS]
        warnings.append(
            f"recipe has more than {brief_mod.MAX_EXTRA_LORAS} in-library LoRAs; dropped: "
            + ", ".join(lora["stem"] for lora in dropped)
        )
    specs = [f"{lora['stem']}:{lora['strength']:g}" for lora in usable if lora.get("stem")]
    if specs:
        brief["loras"] = specs

    sampler, scheduler = _map_sampler(str(gen.get("sampler") or ""), warnings)
    if sampler:
        brief["sampler"] = sampler
    if scheduler:
        brief["scheduler"] = scheduler

    steps = _as_int(gen.get("steps"))
    if steps is not None:
        if steps > 60:
            warnings.append(f"steps {steps} clamped to 60 (engine ceiling)")
            steps = 60
        if steps >= 1:
            brief["steps"] = steps

    cfg = _as_float(gen.get("cfgScale"))
    if cfg is not None:
        clamped = min(max(cfg, 1.0), 15.0)
        if clamped != cfg:
            warnings.append(f"cfg {cfg:g} clamped to {clamped:g}")
        brief["cfg"] = clamped

    seed = _as_int(gen.get("seed"))
    if seed is not None and seed >= 0:
        brief["seed"] = seed
    elif seed is not None:
        warnings.append("recipe seed is negative (random); left unset")

    width, height = recipe_dto.get("width"), recipe_dto.get("height")
    if width and height:
        clamped_w = min(max(int(width), brief_mod.RES_MIN), brief_mod.RES_MAX)
        clamped_h = min(max(int(height), brief_mod.RES_MIN), brief_mod.RES_MAX)
        if (clamped_w, clamped_h) != (int(width), int(height)):
            warnings.append(f"res {width}x{height} clamped to {clamped_w}x{clamped_h}")
        brief["res"] = f"{clamped_w}x{clamped_h}"

    if looks_nsfw(recipe_dto):
        brief["nsfw"] = True
        warnings.append("recipe looks NSFW; routed to the nsfw pipeline (override with \"nsfw\": false)")

    for key, value in (overrides or {}).items():
        if value is None:
            brief.pop(key, None)
        else:
            brief[key] = value

    if brief.get("tier") == "draft":
        for key in brief_mod.QUALITY_ONLY_FIELDS:
            if brief.pop(key, None) is not None:
                warnings.append(f"{key} dropped (not available on draft tier)")
    return brief, warnings
