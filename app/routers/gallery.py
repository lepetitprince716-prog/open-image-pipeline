"""Gallery listing endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/gallery")
    def gallery(limit: int = 60, offset: int = 0):
        """Recent SFW generated images, newest first."""
        return state._scan_gallery(state.BY_MODEL_DIR, state._gallery_cache, limit, offset, nsfw=False)

    @router.get("/api/nsfw/gallery")
    def nsfw_gallery(limit: int = 60, offset: int = 0):
        """NSFW mirror of /api/gallery over the separate nsfw/outputs tree."""
        return state._scan_gallery(state.NSFW_OUTPUTS, state._nsfw_gallery_cache, limit, offset, nsfw=True)

    return router
