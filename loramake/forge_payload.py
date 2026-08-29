"""Pure A1111-metadata -> Forge txt2img payload mapping (no network, no subprocess).

The substance of the repro line: every material key of an A1111/Forge
``parameters`` block lands in a documented payload slot (top-level field,
``override_settings``, or ``alwayson_scripts``), asset references become hash
stubs for live resolution, and anything unrecognized is surfaced instead of
silently dropped.  Field-mapping table: docs/forge_line.md.

Kept import-light on purpose — this module is the offline unit-test surface
(tests/test_forge_payload.py) and must never require a reachable box.
"""
from __future__ import annotations

import dataclasses
import json
import re

LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([0-9.]+))?>")

# Legacy pre-"Schedule type" sampler names carry the scheduler as a suffix.
_LEGACY_SCHEDULER_SUFFIXES = (
    (" SGM Uniform", "sgm_uniform"),
    (" Polyexponential", "polyexponential"),
    (" Exponential", "exponential"),
    (" Karras", "karras"),
    (" Uniform", "uniform"),
)

# Display label -> internal scheduler name for the labels that do not reduce to
# a plain lowercase/underscore transform.
_SCHEDULER_LABELS = {
    "automatic": "automatic",
    "align your steps gits": "align_your_steps_GITS",
    "align your steps 11": "align_your_steps_11",
    "align your steps 32": "align_your_steps_32",
}

# Metadata keys that legitimately differ between a source image and a faithful
# reproduction (tool build stamps, not recipe fields). "Model" is additionally
# treated as benign at diff time when both sides carry an equal "Model hash" —
# the hash proves file identity; the display title is whatever the uploader's
# UI called the checkpoint.
BENIGN_DIFF_KEYS = frozenset({
    "Version", "ADetailer version", "ADetailer version 2nd",
    "Hires Module 1", "Hires Module 2",
})

# ADetailer unit defaults for fields the infotext omits (= the extension's own
# defaults; the args order is the flat-array shape debugged against ADetailer
# 26.2.0 — enable, model, model2, confidence, mask_blur, dilate_erode, denoise,
# inpaint_only_masked, inpaint_padding. NEVER submit via script_name: HTTP 422.)
_AD_DEFAULTS = {
    "confidence": 0.3,
    "mask_blur": 4,
    "dilate_erode": 4,
    "denoise": 0.4,
    "inpaint_only_masked": True,
    "inpaint_padding": 32,
}


@dataclasses.dataclass
class MappedPayload:
    payload: dict
    assets: dict            # {"checkpoint": {...}|None, "vae": {...}|None, "loras": [...], "embeddings": [...]}
    warnings: list[str]
    unmapped: dict          # material keys we could not place (key -> raw value)
    report_only: dict       # benign provenance fields (Version, ADetailer version, ...)


def _int(value: str, default: int | None = None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: str, default: float | None = None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _unquote(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def split_legacy_sampler(name: str) -> tuple[str, str | None]:
    """"DPM++ 2M Karras" -> ("DPM++ 2M", "karras"); modern names pass through."""
    text = str(name).strip()
    for suffix, scheduler in _LEGACY_SCHEDULER_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)].strip(), scheduler
    return text, None


def normalize_scheduler(label: str) -> str:
    """A "Schedule type" display label -> the API scheduler name."""
    text = str(label).strip()
    return _SCHEDULER_LABELS.get(text.lower(), text.lower().replace(" ", "_"))


def parse_lora_tags(prompt: str) -> list[dict]:
    """Inline ``<lora:name:weight>`` specs, in order of appearance."""
    tags = []
    for match in LORA_TAG.finditer(prompt or ""):
        tags.append({"name": match.group(1).strip(),
                     "weight": _float(match.group(2), 1.0) if match.group(2) else 1.0})
    return tags


def _parse_pair_map(value: str) -> dict[str, str]:
    """``"a: h1, b: h2"`` (optionally quoted) -> {a: h1, b: h2}."""
    out: dict[str, str] = {}
    for pair in _unquote(value).split(","):
        if ":" in pair:
            name, digest = pair.split(":", 1)
            if name.strip() and digest.strip():
                out[name.strip()] = digest.strip()
    return out


def _parse_hashes_json(value: str) -> dict[str, str]:
    try:
        parsed = json.loads(_unquote(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if isinstance(v, str) and v}


def adetailer_args(ad_fields: dict[str, str], warnings: list[str]) -> list:
    """The debugged flat args array for one ADetailer unit (see _AD_DEFAULTS note)."""
    model = str(ad_fields.get("ADetailer model", "")).strip()
    model2 = str(ad_fields.get("ADetailer model 2nd", "") or "None").strip() or "None"
    if model2 != "None":
        warnings.append(
            "source used a 2nd ADetailer model; v1 passes it in the model2 slot with "
            "first-unit settings — per-unit 2nd settings are not mapped")
    confidence = _float(ad_fields.get("ADetailer confidence"), _AD_DEFAULTS["confidence"])
    mask_blur = _int(ad_fields.get("ADetailer mask blur"), _AD_DEFAULTS["mask_blur"])
    dilate = _int(ad_fields.get("ADetailer dilate erode"), _AD_DEFAULTS["dilate_erode"])
    denoise = _float(ad_fields.get("ADetailer denoising strength"), _AD_DEFAULTS["denoise"])
    only_masked = (_bool(ad_fields["ADetailer inpaint only masked"])
                   if "ADetailer inpaint only masked" in ad_fields
                   else _AD_DEFAULTS["inpaint_only_masked"])
    padding = _int(ad_fields.get("ADetailer inpaint padding"), _AD_DEFAULTS["inpaint_padding"])
    return [True, model, model2, confidence, mask_blur, dilate, denoise, only_masked, padding]


def payload_from_params(parsed: dict) -> MappedPayload:
    """Map ``image_recipe.parse_a1111_parameters`` output to a txt2img payload.

    Consumes recognized keys; whatever remains lands in ``unmapped`` so a
    "100%" claim can be checked (empty unmapped) instead of assumed.
    """
    fields = dict(parsed.get("parameters") or {})
    prompt = str(parsed.get("prompt") or "")
    payload: dict = {
        "prompt": prompt,
        "negative_prompt": str(parsed.get("negative_prompt") or ""),
        "send_images": True,
        "save_images": False,
    }
    override: dict = {}
    assets: dict = {"checkpoint": None, "vae": None, "loras": [], "embeddings": []}
    warnings: list[str] = []
    report_only: dict = {}

    def take(key: str) -> str | None:
        return fields.pop(key, None)

    steps = take("Steps")
    if steps is not None:
        payload["steps"] = _int(steps)
    sampler = take("Sampler")
    schedule_type = take("Schedule type")
    if sampler is not None:
        if schedule_type is None:
            sampler, legacy_scheduler = split_legacy_sampler(sampler)
            if legacy_scheduler:
                payload["scheduler"] = legacy_scheduler
        payload["sampler_name"] = str(sampler).strip()
    if schedule_type is not None and str(schedule_type).strip().lower() != "automatic":
        payload["scheduler"] = normalize_scheduler(schedule_type)
    cfg = take("CFG scale")
    if cfg is not None:
        payload["cfg_scale"] = _float(cfg)
    seed = take("Seed")
    if seed is not None:
        payload["seed"] = _int(seed)
    size = take("Size")
    if size is not None and "x" in str(size):
        width, _, height = str(size).partition("x")
        if width.strip().isdigit() and height.strip().isdigit():
            payload["width"], payload["height"] = int(width), int(height)
    subseed = take("Variation seed")
    if subseed is not None:
        payload["subseed"] = _int(subseed)
    subseed_strength = take("Variation seed strength")
    if subseed_strength is not None:
        payload["subseed_strength"] = _float(subseed_strength)
    denoise = take("Denoising strength")
    if denoise is not None:
        payload["denoising_strength"] = _float(denoise)

    clip_skip = take("Clip skip")
    if clip_skip is not None:
        override["CLIP_stop_at_last_layers"] = _int(clip_skip)
    ensd = take("ENSD")
    if ensd is not None:
        override["eta_noise_seed_delta"] = _int(ensd)
    eta = take("Eta")
    if eta is not None:
        payload["eta"] = _float(eta)
    rng = take("RNG")
    if rng is not None:
        override["randn_source"] = str(rng).strip()
        warnings.append(f"source used RNG={rng!r}; reproduction assumes the server honors "
                        "override_settings.randn_source")
    tomerge = take("Token merging ratio")
    if tomerge is not None:
        override["token_merging_ratio"] = _float(tomerge)
        warnings.append("source used token merging; minor detail drift is expected")
    emphasis = take("Emphasis")
    if emphasis is not None:
        override["emphasis"] = str(emphasis).strip()
    discard_sigma = take("Discard penultimate sigma")
    if discard_sigma is not None:
        override["always_discard_next_to_last_sigma"] = _bool(discard_sigma)

    # -- assets -----------------------------------------------------------------
    model_name = take("Model")
    model_hash = take("Model hash")
    hashes = _parse_hashes_json(take("Hashes") or "")
    if model_name or model_hash or hashes.get("model"):
        assets["checkpoint"] = {
            "name": str(model_name).strip() if model_name else None,
            "short_hash": (str(model_hash).strip() if model_hash else None) or hashes.get("model"),
        }
    vae_name = take("VAE")
    vae_hash = take("VAE hash")
    if vae_name:
        assets["vae"] = {"name": _unquote(vae_name), "short_hash": (vae_hash or "").strip() or None}

    lora_hashes = _parse_pair_map(take("Lora hashes") or "")
    for key, digest in hashes.items():
        if key.startswith("lora:"):
            lora_hashes.setdefault(key[len("lora:"):], digest)
        elif key.startswith("embed:"):
            assets["embeddings"].append({"name": key[len("embed:"):], "short_hash": digest})
    tag_weights = {t["name"]: t["weight"] for t in parse_lora_tags(prompt)}
    seen_loras: set[str] = set()
    for name in list(tag_weights) + [n for n in lora_hashes if n not in tag_weights]:
        if name in seen_loras:
            continue
        seen_loras.add(name)
        assets["loras"].append({"name": name, "weight": tag_weights.get(name),
                                "short_hash": lora_hashes.get(name),
                                "in_prompt": name in tag_weights})
    for entry in assets["loras"]:
        if not entry["in_prompt"]:
            warnings.append(
                f"LoRA {entry['name']!r} is hashed in metadata but has no <lora:> tag in the "
                "prompt — the source text may be the pre-injection version; add the tag and "
                "trigger words manually if the reproduction misses the style")
    if assets["loras"]:
        warnings.append(
            "LoRA trigger words are not auto-injected (the A1111 Civitai extension adds them "
            "AFTER metadata capture); pass --trigger to append any the source relied on")

    ti_hashes = _parse_pair_map(take("TI hashes") or "")
    known_embeds = {e["name"] for e in assets["embeddings"]}
    for name, digest in ti_hashes.items():
        if name not in known_embeds:
            assets["embeddings"].append({"name": name, "short_hash": digest})

    # -- hires fix --------------------------------------------------------------
    hr_scale = take("Hires upscale")
    hr_resize = take("Hires resize")
    hr_upscaler = take("Hires upscaler")
    hr_steps = take("Hires steps")
    hr_cfg = take("Hires CFG Scale")
    hr_prompt = take("Hires prompt")
    hr_negative = take("Hires negative prompt")
    enable_hr = any(v is not None for v in (hr_scale, hr_resize, hr_upscaler))
    if enable_hr:
        payload["enable_hr"] = True
        # Omitting this crashes Forge's hires path with a NoneType error.
        payload["hr_additional_modules"] = ["Use same choices"]
        if hr_scale is not None:
            payload["hr_scale"] = _float(hr_scale)
        if hr_resize is not None and "x" in str(hr_resize):
            rx, _, ry = str(hr_resize).partition("x")
            if rx.strip().isdigit() and ry.strip().isdigit():
                payload["hr_resize_x"], payload["hr_resize_y"] = int(rx), int(ry)
        if hr_upscaler is not None:
            payload["hr_upscaler"] = _unquote(hr_upscaler)
        if hr_steps is not None:
            payload["hr_second_pass_steps"] = _int(hr_steps)
        if hr_cfg is not None:
            payload["hr_cfg"] = _float(hr_cfg)
        if hr_prompt is not None:
            payload["hr_prompt"] = _unquote(hr_prompt)
        if hr_negative is not None:
            payload["hr_negative_prompt"] = _unquote(hr_negative)
        if "denoising_strength" not in payload:
            warnings.append("hires fix without Denoising strength in metadata; server default applies")

    # -- ADetailer --------------------------------------------------------------
    ad_fields = {k: fields.pop(k) for k in list(fields) if k.startswith("ADetailer ")}
    for benign in ("ADetailer version", "ADetailer version 2nd"):
        if benign in ad_fields:
            report_only[benign] = ad_fields.pop(benign)
    if ad_fields.get("ADetailer model"):
        payload["alwayson_scripts"] = {"adetailer": {"args": adetailer_args(ad_fields, warnings)}}
        recognized = {"ADetailer model", "ADetailer model 2nd", "ADetailer confidence",
                      "ADetailer mask blur", "ADetailer dilate erode",
                      "ADetailer denoising strength", "ADetailer inpaint only masked",
                      "ADetailer inpaint padding"}
        for key in sorted(set(ad_fields) - recognized):
            warnings.append(f"ADetailer field not mapped: {key}={ad_fields[key]!r}")
    elif ad_fields:
        warnings.append("ADetailer fields present without 'ADetailer model'; unit not enabled")

    version = take("Version")
    if version is not None:
        report_only["Version"] = version

    if override:
        payload["override_settings"] = override
        payload["override_settings_restore_afterwards"] = False

    return MappedPayload(payload=payload, assets=assets, warnings=warnings,
                         unmapped=dict(fields), report_only=report_only)


# --- live-enum matching (pure functions over enum snapshots) ------------------------------

def _stem(name: str) -> str:
    base = str(name).lower().replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def match_checkpoint(name: str | None, short_hash: str | None,
                     models: list[dict]) -> tuple[dict | None, str]:
    """Match against /sd-models entries: hash first, title/stem fallback.

    Returns (entry, basis) with basis in {"sha256", "short_hash", "name", ""}.
    ``short_hash`` is the AutoV2 form (sha256 prefix); pre-AutoV2 8-hex hashes
    never match and fall through to the name.
    """
    short = (short_hash or "").strip().lower()
    if short and len(short) >= 10:
        for entry in models:
            sha = str(entry.get("sha256") or "").lower()
            if sha and (sha == short or sha.startswith(short)):
                return entry, "sha256" if len(short) == 64 else "short_hash"
            if str(entry.get("hash") or "").lower() == short:
                return entry, "short_hash"
    if name:
        want = _stem(name)
        for entry in models:
            candidates = {_stem(entry.get(field) or "")
                          for field in ("title", "model_name", "filename")}
            # titles carry a trailing " [hash]" — compare the pre-bracket stem too
            title = str(entry.get("title") or "")
            candidates.add(_stem(title.rsplit(" [", 1)[0]))
            if want in candidates:
                return entry, "name"
    return None, ""


def match_lora(name: str, entries: list[dict]) -> tuple[dict | None, str]:
    """Match one ``<lora:name:...>`` reference against /loras entries."""
    want = str(name).strip().lower()
    want_stem = _stem(name)
    for entry in entries:
        if want in {str(entry.get("name") or "").lower(), str(entry.get("alias") or "").lower()}:
            return entry, "name"
    for entry in entries:
        if want_stem and want_stem in {_stem(entry.get("name") or ""), _stem(entry.get("alias") or ""),
                                       _stem(entry.get("path") or "")}:
            return entry, "stem"
    return None, ""


# --- metadata round-trip diff -------------------------------------------------------------

def _normalize_value(key: str, value: object) -> object:
    text = _unquote(str(value))
    if key in {"Hashes"}:
        parsed = _parse_hashes_json(text)
        return parsed or text
    if key in {"Lora hashes", "TI hashes"}:
        parsed_map = _parse_pair_map(text)
        return parsed_map or text
    number = _float(text)
    if number is not None and re.fullmatch(r"-?\d+(\.\d+)?", text):
        return number
    return text


def diff_parameters(source: dict, produced: dict) -> list[dict]:
    """Field-level diff of two parse_a1111_parameters results.

    Rows: {field, source, produced, status, benign} with status one of
    MATCH | DIFF | SOURCE_ONLY | PRODUCED_ONLY. Acceptance for a faithful
    reproduction = zero non-benign DIFF/SOURCE_ONLY rows.
    """
    rows: list[dict] = []
    src_fields_peek = source.get("parameters") or {}
    out_fields_peek = produced.get("parameters") or {}
    same_model_hash = (
        src_fields_peek.get("Model hash") is not None
        and _normalize_value("Model hash", src_fields_peek.get("Model hash"))
        == _normalize_value("Model hash", out_fields_peek.get("Model hash"))
    )

    def add(field: str, src: object, out: object) -> None:
        benign = field in BENIGN_DIFF_KEYS or (field == "Model" and same_model_hash)
        if src is None and out is None:
            return
        if src is None:
            status = "PRODUCED_ONLY"
        elif out is None:
            status = "SOURCE_ONLY"
        else:
            status = "MATCH" if _normalize_value(field, src) == _normalize_value(field, out) else "DIFF"
        rows.append({"field": field, "source": src, "produced": out,
                     "status": status, "benign": benign})

    add("Prompt", source.get("prompt"), produced.get("prompt"))
    add("Negative prompt", source.get("negative_prompt"), produced.get("negative_prompt"))
    src_fields = source.get("parameters") or {}
    out_fields = produced.get("parameters") or {}
    for key in sorted(set(src_fields) | set(out_fields)):
        add(key, src_fields.get(key), out_fields.get(key))
    return rows
