#!/usr/bin/env python3
"""Extract the generation "recipe" of a Civitai image via the public API.

Replicates ComfyUI-LoRA-Manager's "import recipe from a Civitai image" feature, but
using only the public Civitai REST API (and only the SFW civitai.com host — never the
.red mirror). Given a civitai.com/images/<id> URL (or a bare image id), it:

  1. fetches the image's embedded generation metadata     (/api/v1/images?imageId=<id>)
  2. parses the checkpoint + every LoRA (with weights)     (meta.resources / hashes /
                                                            civitaiResources / additionalResources
                                                            + inline <lora:name:weight> in the prompt)
  3. resolves each resource to a concrete Civitai model    (by-hash → /model-versions/by-hash/<h>
                                                            by id   → /model-versions/<id>)
  4. emits a structured recipe (checkpoint, loras[], gen params, base model).

Transport, token handling, SFW enforcement and proxy come from scripts/civitai.py
(unchanged). Resources whose hash/id Civitai cannot resolve (private or unpublished
models, hash-type mismatches) degrade gracefully: they are reported with name + hash +
weight and resolved=false rather than aborting the whole recipe.

With --local, each resolved resource is checked against the local LoRA catalog
(lora_catalog/z3_index.json): a match on resolved SHA256[:12] == catalog record id, or on
modelVersionId == the catalog file stem, marks it present locally. Unresolved resources
(private / unpublished) cannot be checked against the public catalog and are reported as such.

CLI:
  python3 scripts/civitai_recipe.py https://civitai.com/images/63949551
  python3 scripts/civitai_recipe.py 63949551 --json --local
  python3 scripts/civitai_recipe.py 63949551 --save out/recipe.json --raw-meta
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from civitai import by_hash, images, model_version  # verified Civitai transport (SFW, curl+proxy, token via fd3)

# Scalar generation params we surface, mapped from the Civitai meta keys to stable snake_case.
PARAM_MAP: dict[str, str] = {
    "prompt": "prompt",
    "negativePrompt": "negative_prompt",
    "sampler": "sampler",
    "cfgScale": "cfg_scale",
    "steps": "steps",
    "seed": "seed",
    "Size": "size",
    "clipSkip": "clip_skip",
    "Denoising strength": "denoise",
    "Schedule type": "schedule",
}
# A1111 inline LoRA/LyCORIS syntax: <lora:NAME:WEIGHT> (weight optional, may be e.g. ".2", "0.8", "1").
_INLINE_LORA = re.compile(r"<(?:lora|lyco):([^:>]+?)(?::([0-9]*\.?[0-9]+))?(?::[^>]*)?>", re.IGNORECASE)
_IMAGE_ID = re.compile(r"/images/(\d+)")
# Local LoRA catalog written by scripts/lora_catalog.py (its record `id` is the SHA256[:12] prefix).
_CATALOG_INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "lora_catalog", "z3_index.json")


def parse_image_id(arg: str) -> int | None:
    """Accept a civitai image URL, a bare numeric id, or an imageId=<n> token."""
    arg = arg.strip()
    if arg.isdigit():
        return int(arg)
    m = _IMAGE_ID.search(arg) or re.search(r"imageId=(\d+)", arg)
    return int(m.group(1)) if m else None


def _same_id(a: Any, b: Any) -> bool:
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return False


def fetch_image(image_id: int) -> tuple[int, dict[str, Any] | None]:
    """Return (http_status, image_object). item is None when the call failed (non-200) or the image is
    absent from a 200 response (deleted, or filtered out by the SFW nsfw=false guard)."""
    code, data = images(imageId=image_id, withMeta="true")
    if code != 200 or not isinstance(data, dict):
        return code, None
    for item in (data.get("items") or []):  # `or []` guards a null/missing items field
        if isinstance(item, dict) and _same_id(item.get("id"), image_id):
            return code, item
    return code, None


def inline_lora_weights(prompt: Any) -> dict[str, float]:
    """Map lowercased LoRA name -> weight parsed from inline <lora:name:weight> tags in the prompt.

    Civitai A1111 images often carry the real per-LoRA weight ONLY here; meta.resources frequently
    omits the weight field. Names are lowercased for case-insensitive matching against resources."""
    out: dict[str, float] = {}
    if not isinstance(prompt, str):
        return out
    for name, weight in _INLINE_LORA.findall(prompt):
        if weight:
            out[name.strip().lower()] = float(weight)
    return out


def _hash_from_map(meta: dict[str, Any], name: str) -> str | None:
    """Look a LoRA's hash up in meta.hashes by its 'lora:<name>' key (case-insensitive)."""
    hashes = meta.get("hashes")
    if not isinstance(hashes, dict):
        return None
    want = f"lora:{name}".lower()
    for key, value in hashes.items():
        if str(key).lower() == want:
            return value if isinstance(value, str) and value else None
    return None


def _version_id_from_urn(name: str) -> int | None:
    """additionalResources names are AIR URNs: urn:air:...:civitai:<modelId>@<versionId>."""
    if "civitai:" in name and "@" in name:
        tail = name.split("@", 1)[1]
        digits = re.match(r"(\d+)", tail)
        if digits:
            return int(digits.group(1))
    return None


def collect_resources(meta: dict[str, Any]) -> tuple[dict | None, list[dict]]:
    """Merge LoRA/checkpoint stubs from every meta source, deduped. Returns (checkpoint_stub, lora_stubs).

    A stub = {name, hash, version_id, weight}. Faithful to LoRA-Manager's multi-source merge:
    resources / hashes / civitaiResources / additionalResources / Model+Model hash, plus the
    inline-prompt weights this tool additionally recovers."""
    weights = inline_lora_weights(meta.get("prompt", ""))
    checkpoint: dict | None = None
    loras: list[dict] = []
    seen: set[str] = set()

    def key_of(stub: dict) -> str:
        if stub.get("hash"):
            return f"h:{str(stub['hash']).lower()}"
        if stub.get("version_id"):
            return f"v:{stub['version_id']}"
        return f"n:{str(stub.get('name', '')).lower()}"

    def add_lora(name: str, hash_: str | None, version_id: int | None, weight: Any) -> None:
        if not (hash_ or version_id or name):  # no identity at all → nothing to report
            return
        # Structured weight wins; else the inline <lora:name:w> weight; else 1.0. Non-numeric → fallback.
        weight = _coerce_weight(weight, weights.get(str(name).lower(), 1.0))
        stub = {"name": name, "hash": hash_, "version_id": version_id, "weight": weight}
        k = key_of(stub)
        name_key = f"n:{str(name).lower()}" if name else None
        if k in seen or (name_key and name_key in seen):  # dedup across sources by id OR name
            return
        seen.add(k)
        if name_key:
            seen.add(name_key)  # so a later name-only entry can't duplicate a hash/version one
        loras.append(stub)

    def set_checkpoint(name: str, hash_: str | None, version_id: int | None) -> None:
        nonlocal checkpoint
        if checkpoint is None:
            checkpoint = {"name": name, "hash": hash_, "version_id": version_id}

    # Source A: meta.resources — [{hash, name, type, weight?}]
    for r in meta.get("resources", []) if isinstance(meta.get("resources"), list) else []:
        if not isinstance(r, dict):
            continue
        rtype = str(r.get("type", "")).lower()
        name = r.get("name", "")
        hash_ = r.get("hash") or _hash_from_map(meta, name)
        vid = r.get("modelVersionId")
        if rtype in ("model", "checkpoint"):
            set_checkpoint(name, hash_, vid)
        elif rtype in ("lora", "lycoris", "locon"):
            add_lora(name, hash_, vid, r.get("weight"))

    # Source B: meta.civitaiResources — [{type, weight, modelVersionId, modelName, modelVersionName}]
    for r in meta.get("civitaiResources", []) if isinstance(meta.get("civitaiResources"), list) else []:
        if not isinstance(r, dict):
            continue
        rtype = str(r.get("type", "")).lower()
        vid = r.get("modelVersionId")
        name = r.get("modelName") or r.get("modelVersionName") or ""
        if rtype == "checkpoint":
            set_checkpoint(name, None, vid)
        else:
            add_lora(name, None, vid, r.get("weight"))

    # Source C: meta.additionalResources — [{name(AIR urn), type, strength}]
    for r in meta.get("additionalResources", []) if isinstance(meta.get("additionalResources"), list) else []:
        if not isinstance(r, dict):
            continue
        if str(r.get("type", "")).lower() not in ("lora", "lycoris", "locon"):
            continue
        name = r.get("name", "")
        add_lora(name, None, _version_id_from_urn(name), r.get("strength"))

    # Source D: meta.hashes 'lora:<name>' entries not yet captured (weight from inline prompt / 1.0)
    if isinstance(meta.get("hashes"), dict):
        for key, value in meta["hashes"].items():
            ks = str(key)
            if ks.lower().startswith("lora:"):
                hv = value if isinstance(value, str) and value else None
                add_lora(ks.split(":", 1)[1], hv, None, None)

    # Source E: checkpoint fallback from the A1111 Model / Model hash fields
    if checkpoint is None and (meta.get("Model") or meta.get("Model hash")):
        set_checkpoint(meta.get("Model", ""), meta.get("Model hash"), None)

    return checkpoint, loras


def _unwrap_meta(meta: Any) -> dict[str, Any] | None:
    """Return the flat generation meta. The /images?imageId=<id> endpoint nests it one level
    deep under meta['meta'] (the modelVersionId endpoint returns it flat). Descend through pure
    wrapper levels (keys ⊆ {id, meta}) and stop as soon as real gen fields appear — handles any
    nesting depth by content, not a fixed count. Returns None for missing/empty meta."""
    for _ in range(8):  # hard cap backstops the content check against pathological nesting
        if (isinstance(meta, dict) and isinstance(meta.get("meta"), dict)
                and set(meta.keys()) <= {"id", "meta"}):
            meta = meta["meta"]
        else:
            break
    return meta if isinstance(meta, dict) and meta else None


def _coerce_weight(value: Any, default: float) -> float:
    """LoRA weight/strength → float, rounded. Civitai sometimes records it as a numeric string;
    a non-numeric value falls back to default rather than crashing the whole recipe."""
    if value is not None:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            pass
    return round(float(default), 4)


def _primary_file(version: dict[str, Any]) -> dict[str, Any]:
    raw = version.get("files")
    files = [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []
    return next((f for f in files if f.get("primary")), files[0] if files else {})


def _civitai_block(version: dict[str, Any]) -> dict[str, Any]:
    prim = _primary_file(version)
    file_hashes = prim.get("hashes")
    sha = file_hashes.get("SHA256", "") if isinstance(file_hashes, dict) else ""
    model = version.get("model")
    model = model if isinstance(model, dict) else {}
    version_id = version.get("id")
    model_id = version.get("modelId")
    url = (f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
           if model_id and version_id else None)
    return {
        "version_id": version_id,
        "model_id": model_id,
        "model_name": model.get("name"),
        "version_name": version.get("name"),
        "type": model.get("type"),
        "base_model": version.get("baseModel"),
        "sha256": sha.lower() or None,
        "file": prim.get("name"),
        "size_kb": prim.get("sizeKB"),
        "trained_words": version.get("trainedWords") or [],
        "url": url,
    }


def resolve_stub(stub: dict[str, Any]) -> dict[str, Any]:
    """Resolve one stub to a Civitai model-version (by id first, else by hash). Degrades gracefully."""
    result = dict(stub)
    version_id, hash_ = stub.get("version_id"), stub.get("hash")
    try:
        if version_id:
            code, data = model_version(version_id)
        elif hash_:
            code, data = by_hash(hash_)
        else:
            result.update(resolved=False, reason="no hash or version id", civitai=None)
            return result
    except Exception as exc:  # transport blew up — isolate this resource, never abort the whole recipe
        result.update(resolved=False, civitai=None, transient=True, reason=f"lookup error: {type(exc).__name__}")
        return result
    if code == 200 and isinstance(data, dict) and data.get("id"):
        result.update(resolved=True, civitai=_civitai_block(data))
    elif code == 404:
        result.update(resolved=False, civitai=None,
                      reason="model not found on civitai (private/unpublished or hash-type mismatch)")
    else:  # 5xx / 429 / curl error (HTTP 0) — transient, not an authoritative "absent"
        result.update(resolved=False, civitai=None, transient=True,
                      reason=f"lookup failed (HTTP {code}) — transient, retry")
    return result


def pick_base_model(meta: dict[str, Any], checkpoint: dict | None, loras: list[dict]) -> str | None:
    """meta.baseModel → resolved checkpoint baseModel → most common resolved-LoRA baseModel."""
    if meta.get("baseModel"):
        return meta["baseModel"]
    if checkpoint and checkpoint.get("civitai") and checkpoint["civitai"].get("base_model"):
        return checkpoint["civitai"]["base_model"]
    bms = [lora["civitai"]["base_model"] for lora in loras
           if lora.get("civitai") and lora["civitai"].get("base_model")]
    return Counter(bms).most_common(1)[0][0] if bms else None


def gen_params(meta: dict[str, Any]) -> dict[str, Any]:
    return {dst: meta[src] for src, dst in PARAM_MAP.items() if src in meta}


def load_local_catalog(path: str = _CATALOG_INDEX) -> dict[str, dict] | None:
    """Index the local LoRA catalog for presence lookups, or None if the catalog file is absent.

    Returns {'by_sha12': {sha256[:12] -> rec}, 'by_vid': {version_id_str -> rec}} — the two verified
    join keys: the catalog record `id` is the SHA256[:12] prefix, and LoRA-Manager downloads are
    named '<modelVersionId>.safetensors'."""
    try:
        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(records, list):  # unexpected shape → treat as unavailable so the rebuild note fires
        return None
    by_sha12: dict[str, dict] = {}
    by_vid: dict[str, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("id", "")).lower()
        if len(rid) >= 12:
            by_sha12.setdefault(rid[:12], rec)
        stem = str(rec.get("file", "")).rsplit(".", 1)[0]
        if stem.isdigit():
            by_vid.setdefault(stem, rec)
    return {"by_sha12": by_sha12, "by_vid": by_vid}


def local_match(lora: dict[str, Any], catalog: dict[str, dict]) -> dict[str, Any]:
    """Locate a resolved LoRA in the local catalog by SHA256[:12], else by modelVersionId.

    Unresolved LoRAs (no Civitai resolution) carry neither key, so they report present=False with a
    note rather than a false negative — the public catalog simply cannot confirm a private model."""
    civ = lora.get("civitai") or {}
    sha = (civ.get("sha256") or "").lower()
    if sha and sha[:12] in catalog["by_sha12"]:
        rec = catalog["by_sha12"][sha[:12]]
        return {"present": True, "file": rec.get("file"), "name": rec.get("name"), "matched_by": "sha256"}
    vid = civ.get("version_id")
    if vid is not None and str(vid) in catalog["by_vid"]:
        rec = catalog["by_vid"][str(vid)]
        return {"present": True, "file": rec.get("file"), "name": rec.get("name"), "matched_by": "version_id"}
    if not lora.get("resolved"):
        return {"present": False, "checked": False, "reason": "unresolved — not checkable against the public catalog"}
    return {"present": False, "checked": True}


def recipe_string(checkpoint: dict | None, loras: list[dict]) -> str:
    ckpt = "?"
    if checkpoint:
        civ = checkpoint.get("civitai")
        ckpt = (civ.get("model_name") if civ else None) or checkpoint.get("name") or "?"
    parts = [f"Checkpoint: {ckpt}"]
    for lora in loras:
        civ = lora.get("civitai")
        nm = (civ.get("model_name") if civ else None) or lora.get("name") or "?"
        parts.append(f"<lora:{nm}:{lora['weight']:g}>")
    return "  ".join(parts)


def build_recipe(item: dict[str, Any], include_raw: bool = False,
                 catalog: dict[str, dict] | None = None) -> dict[str, Any]:
    """Top-level: turn a fetched Civitai image object into the structured recipe."""
    meta = _unwrap_meta(item.get("meta"))
    image_id = item.get("id")
    recipe: dict[str, Any] = {
        "image_id": image_id,
        "image_page": f"https://civitai.com/images/{image_id}" if image_id else None,
        "image_url": item.get("url"),
        "nsfw_level": item.get("nsfwLevel"),
    }
    if not meta:
        recipe["error"] = "image has no embedded generation metadata on the API (no meta block)"
        recipe["checkpoint"], recipe["loras"], recipe["params"], recipe["base_model"] = None, [], {}, None
        return recipe

    checkpoint_stub, lora_stubs = collect_resources(meta)
    # Resolve every resource concurrently — each is an independent, side-effect-free API round-trip
    # (curl subprocess); pool.map preserves order, so the recipe content is identical to serial.
    all_stubs = ([checkpoint_stub] if checkpoint_stub else []) + lora_stubs
    if all_stubs:
        with ThreadPoolExecutor(max_workers=min(8, len(all_stubs))) as pool:
            resolved = list(pool.map(resolve_stub, all_stubs))
    else:
        resolved = []
    checkpoint = resolved[0] if checkpoint_stub else None
    loras = resolved[1:] if checkpoint_stub else resolved
    if catalog is not None:
        for lora in loras:
            lora["local"] = local_match(lora, catalog)
    recipe["checkpoint"] = checkpoint
    recipe["loras"] = loras
    recipe["base_model"] = pick_base_model(meta, checkpoint, loras)
    recipe["params"] = gen_params(meta)
    recipe["recipe_string"] = recipe_string(checkpoint, loras)
    recipe["unresolved"] = [lora.get("name") or lora.get("hash") or "(unknown)"
                            for lora in loras if not lora.get("resolved")]
    net_errs = sum(1 for r in ([checkpoint] + loras) if r and r.get("transient"))
    if net_errs:
        recipe["network_errors"] = net_errs
    if catalog is not None:
        recipe["local_present"] = sum(1 for lora in loras if (lora.get("local") or {}).get("present"))
    if include_raw:
        recipe["raw_meta"] = meta
    return recipe


def get_recipe(arg: str, include_raw: bool = False, local: bool = False) -> dict[str, Any]:
    """Public entry point: image URL/id string → recipe dict (or {'error': ...})."""
    image_id = parse_image_id(arg)
    if image_id is None:
        return {"error": f"could not parse a Civitai image id from {arg!r}"}
    code, item = fetch_image(image_id)
    if item is None:
        if code != 200:
            return {"error": f"Civitai API returned HTTP {code} for image {image_id} — likely transient; retry",
                    "image_id": image_id, "http_status": code}
        return {"error": f"image {image_id} not found via API (deleted, or rated above SFW so it is filtered)",
                "image_id": image_id}
    catalog = load_local_catalog() if local else None
    recipe = build_recipe(item, include_raw=include_raw, catalog=catalog)
    if local and catalog is None:
        recipe["local_note"] = ("local catalog lora_catalog/z3_index.json not found — "
                                "run `python3 scripts/lora_catalog.py build` on Z3 first")
    return recipe


def _print_summary(recipe: dict[str, Any]) -> None:
    if recipe.get("error"):
        print(f"! {recipe['error']}")
        return
    print(f"Image  {recipe.get('image_page')}")
    ck = recipe.get("checkpoint")
    if ck:
        civ = ck.get("civitai") or {}
        tag = "" if ck.get("resolved") else f"  [unresolved: {ck.get('reason')}]"
        print(f"Base   {civ.get('model_name') or ck.get('name')}  "
              f"({recipe.get('base_model') or '?'}){tag}")
    print(f"LoRAs  {len(recipe.get('loras', []))}"
          + (f"  ({len(recipe['unresolved'])} unresolved)" if recipe.get("unresolved") else ""))
    for lora in recipe.get("loras", []):
        civ = lora.get("civitai") or {}
        nm = civ.get("model_name") or lora.get("name")
        mark = "" if lora.get("resolved") else "  [unresolved]"
        ver = f"  v{civ.get('version_id')}" if civ.get("version_id") else ""
        loc = lora.get("local")
        if loc is not None:
            mark += "  ✓ local" if loc.get("present") else ("  · not checked" if loc.get("checked") is False else "  ✗ missing")
        print(f"  · {lora['weight']:<5g} {nm}{ver}{mark}")
    if recipe.get("local_note"):
        print(f"  ({recipe['local_note']})")
    p = recipe.get("params", {})
    if p:
        line = "  ".join(f"{k}={p[k]}" for k in ("sampler", "cfg_scale", "steps", "seed", "size", "clip_skip")
                         if k in p)
        if line:
            print(f"Params {line}")
    print(f"\n{recipe.get('recipe_string', '')}")
    if recipe.get("network_errors"):
        print(f"! {recipe['network_errors']} resource(s) failed to resolve from a transient API error "
              "— recipe may be incomplete; retry")


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:  # TextIOWrapper.reconfigure — never crash on ✓/✗ glyphs under a non-UTF-8 stdout
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    ap = argparse.ArgumentParser(description="Extract a Civitai image's generation recipe via the public API.")
    ap.add_argument("image", help="civitai image URL (…/images/<id>) or a bare image id")
    ap.add_argument("--json", action="store_true", help="print the full recipe as JSON (default: human summary)")
    ap.add_argument("--save", metavar="PATH", help="write the full recipe JSON to PATH")
    ap.add_argument("--raw-meta", action="store_true", help="include the original Civitai meta blob in the recipe")
    ap.add_argument("--local", action="store_true",
                    help="flag which resolved LoRAs are already installed locally (lora_catalog/z3_index.json)")
    args = ap.parse_args()

    recipe = get_recipe(args.image, include_raw=args.raw_meta, local=args.local)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh, ensure_ascii=False, indent=2)
        print(f"saved → {args.save}")
    if args.json or args.save:
        if args.json:
            print(json.dumps(recipe, ensure_ascii=False, indent=2))
    else:
        _print_summary(recipe)
    if recipe.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
