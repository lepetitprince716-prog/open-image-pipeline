"""Thumbnail and LoRA-preview proxy endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/lora-preview")
    def lora_preview(p: str):
        """Proxy and downscale LoRA-Manager previews for tunneled/mobile clients."""
        if not p.startswith(state.LM_PREVIEW_PREFIX):
            raise HTTPException(status_code=400, detail="not a LoRA-Manager preview path")
        try:
            cached = state._ensure_preview(p)
        except Exception:
            raise HTTPException(status_code=404, detail="preview unavailable (z3 offline?)")
        return FileResponse(cached, media_type="image/webp", headers=state.IMMUTABLE_CACHE)

    @router.get("/api/thumb")
    def thumb(p: str, w: int = 480):
        return state._thumb_response(state._sfw_outputs_path(p), w, state.THUMB_CACHE)

    @router.get("/api/nsfw/thumb")
    def nsfw_thumb(p: str, w: int = 480):
        return state._thumb_response(state._nsfw_outputs_path(p), w, state.NSFW_THUMB_CACHE)

    return router
