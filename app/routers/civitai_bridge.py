"""HTTP surface for the Civitai Bridge Chrome extension."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app import civitai_bridge as bridge_mod


_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


def _bool_field(body: dict, field: str) -> bool | None:
    """JSON booleans, plus the string spellings a form post or query string produces.
    None passes through so callers can keep "unset" distinct from an explicit false."""
    value = body.get(field)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    elif isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise HTTPException(status_code=400, detail=f"{field} must be a boolean")


def _overrides_field(body: dict | None) -> dict | None:
    overrides = (body or {}).get("overrides")
    if overrides is None or isinstance(overrides, dict):
        return overrides
    raise HTTPException(status_code=400, detail="overrides must be an object")


def _ids_from_body(body: dict | None) -> list[dict]:
    body = body or {}
    raw = body.get("ids")
    if raw is None and (body.get("modelId") is not None or body.get("versionId") is not None):
        raw = [body]
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(item)
    return out


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    def _bridge() -> bridge_mod.CivitaiBridge:
        getter = getattr(state, "civitai_bridge", None)
        if callable(getter):
            return getter()
        existing = getattr(state, "CIVITAI_BRIDGE", None)
        if existing is not None:
            return existing
        return bridge_mod.get_bridge()

    def _network_enabled(service_id: str) -> bool:
        network_state = getattr(state, "network_state", None)
        return network_state is None or network_state.enabled(service_id)

    def _require(service_id: str, label: str) -> None:
        if not _network_enabled(service_id):
            raise HTTPException(status_code=503, detail=f"{label} 已在网络页停用")

    @router.get("/api/civitai-bridge/health")
    def health():
        return _bridge().health()

    @router.post("/api/civitai-bridge/owned")
    def owned(body: dict | None = None):
        ids = _ids_from_body(body)
        if not ids:
            raise HTTPException(status_code=400, detail="ids required")
        enrich = True if not body or "enrich" not in body else bool(body.get("enrich"))
        return {"results": _bridge().owned(ids, enrich=enrich)}

    @router.post("/api/civitai-bridge/inspect")
    def inspect(body: dict | None = None):
        _require("civitai", "Civitai")
        body = body or {}
        return _bridge().inspect(
            url=str(body.get("url") or ""),
            image_id=body.get("imageId"),
            model_id=body.get("modelId"),
            version_id=body.get("versionId") or body.get("modelVersionId"),
            site=body.get("site"),
        )

    @router.post("/api/civitai-bridge/download")
    def download(body: dict | None = None):
        _require("civitai", "Civitai")
        _require("lora_manager", "LoRA Manager")
        body = body or {}
        result = _bridge().download(
            model_id=body.get("modelId"),
            version_id=body.get("versionId") or body.get("modelVersionId"),
            site=str(body.get("site") or "com"),
            allow_nsfw=_bool_field(body, "allowNsfw"),
        )
        if not result.get("ok") and not result.get("already"):
            raise HTTPException(status_code=400, detail=result.get("error") or "download failed")
        return result

    @router.post("/api/civitai-bridge/recipe")
    def recipe(body: dict | None = None):
        _require("civitai", "Civitai")
        body = body or {}
        image_id = body.get("imageId")
        if image_id is None:
            parsed = bridge_mod.parse_page_url(str(body.get("url") or ""))
            image_id = parsed.get("imageId")
        if image_id is None:
            raise HTTPException(status_code=400, detail="imageId or image url required")
        result = _bridge().save_recipe(
            image_id=int(image_id),
            url=str(body.get("url") or ""),
            site=str(body.get("site") or "com"),
            name=body.get("name"),
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "recipe failed")
        return result

    @router.get("/api/civitai-bridge/recipes")
    def recipes(page: int = 1, pageSize: int = 20, search: str = ""):
        result = _bridge().recipes(page=page, page_size=pageSize, search=search)
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error") or "LoRA Manager unreachable")
        return result

    @router.get("/api/civitai-bridge/recipes/{rid}")
    def recipe_detail(rid: str):
        result = _bridge().recipe_detail(rid)
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error") or "recipe unavailable")
        return result

    @router.post("/api/civitai-bridge/recipes/sync")
    def recipes_sync():
        result = _bridge().sync_recipes()
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error") or "LoRA Manager unreachable")
        return result

    @router.post("/api/civitai-bridge/recipes/{rid}/brief")
    def recipe_brief_route(rid: str, body: dict | None = None):
        overrides = _overrides_field(body)
        bridge = _bridge()
        detail = bridge.recipe_detail(rid)
        if not detail.get("ok"):
            raise HTTPException(status_code=502, detail=detail.get("error") or "recipe unavailable")
        job_brief, warnings = bridge_mod.recipe_brief.brief_from_recipe(
            detail["recipe"], overrides
        )
        from app import brief as brief_validation

        return {
            "brief": job_brief,
            "warnings": warnings,
            "problems": brief_validation.validate(job_brief),
            "missing": bridge_mod.recipe_brief.missing_loras(detail["recipe"]),
        }

    @router.post("/api/civitai-bridge/recipes/{rid}/generate")
    def recipe_generate(rid: str, body: dict | None = None):
        result = _bridge().recipe_generate(rid, _overrides_field(body))
        if not result.get("ok"):
            if result.get("problems"):
                raise HTTPException(status_code=400, detail="; ".join(result["problems"]))
            raise HTTPException(status_code=502, detail=result.get("error") or "recipe unavailable")
        return result

    @router.post("/api/civitai-bridge/download-missing")
    def download_missing(body: dict | None = None):
        _require("civitai", "Civitai")
        _require("lora_manager", "LoRA Manager")
        body = body or {}
        recipe_id = body.get("recipeId")
        resources = body.get("resources")
        if not recipe_id and not isinstance(resources, list):
            raise HTTPException(status_code=400, detail="recipeId or resources required")
        result = _bridge().download_missing(
            recipe_id=str(recipe_id) if recipe_id else None,
            resources=resources if isinstance(resources, list) else None,
            allow_nsfw=_bool_field(body, "allowNsfw") or False,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error") or "LoRA Manager unreachable")
        return result

    @router.get("/api/civitai-bridge/progress")
    def progress(downloadId: str | None = None):
        # Soft-degrade rather than 503: the panel polls this at 1.5s and an error
        # status would spam toasts; "disabled" lets it pause politely.
        if not _network_enabled("lora_manager"):
            return {"jobs": [], "active": [], "stats": {}, "disabled": True}
        return _bridge().progress(downloadId)

    @router.post("/api/civitai-bridge/refresh-index")
    def refresh_index():
        _require("lora_manager", "LoRA Manager")
        relink = _bridge().relink_unlinked()
        idx = _bridge().index(force=True)
        return {"ok": True, **idx.counts(), "source": idx.source, "relink": relink}

    return router
