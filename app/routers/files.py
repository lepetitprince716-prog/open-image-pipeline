"""Rendered-file serving endpoints and frontend entry route."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    # GET+HEAD: mobile browsers / CF sometimes probe HEAD before painting; GET-only
    # fell through to the SPA StaticFiles mount and returned HTML 404. HEAD twins are
    # schema-hidden so each OpenAPI operation id stays unique.
    @router.get("/files")
    @router.head("/files", include_in_schema=False)
    def serve_file(p: str):
        """Serve a generated image by absolute path (SFW outputs tree only)."""
        return FileResponse(state._sfw_outputs_path(p))

    @router.get("/nsfw/files")
    @router.head("/nsfw/files", include_in_schema=False)
    def serve_nsfw_file(p: str):
        """Serve an NSFW result by absolute path (nsfw/outputs tree only)."""
        return FileResponse(state._nsfw_outputs_path(p))

    @router.get("/")
    def index():
        if (state.WEB_OUT / "index.html").is_file():
            return FileResponse(state.WEB_OUT / "index.html", headers={"Cache-Control": "no-cache"})
        raise HTTPException(status_code=503,
                            detail="frontend not built — run: cd app-web && pnpm build")

    return router
