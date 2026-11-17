"""Rendered-file serving endpoints and frontend entry route."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse

from loramake import remote_original


def _full_size(path, original: bool):
    """``original=1`` (the download link) swaps an off-LAN preview for its original first; an
    original that stays on its box is refused, never fetched."""
    if original and remote_original.stays_remote(path):
        raise HTTPException(status_code=409, detail=remote_original.STAYS_REMOTE_DETAIL)
    if original and remote_original.pending(path) and not remote_original.fetch(path):
        raise HTTPException(status_code=502, detail="the original is still on the render box, "
                                                    "which is not reachable right now")
    return FileResponse(path)


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    def _channel(header: str | None) -> str | None:
        try:
            return state.jobs.parse_channel(header)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _channel_claim(path, channel: str | None) -> None:
        """A channel-scoped caller may only fetch files its own channel's jobs produced."""
        if channel is None:
            return
        if state.jobs.find_job_by_file(str(path), channel=channel) is None:
            raise HTTPException(status_code=404, detail="no such file")

    # GET+HEAD: mobile browsers / CF sometimes probe HEAD before painting; GET-only
    # fell through to the SPA StaticFiles mount and returned HTML 404. HEAD twins are
    # schema-hidden so each OpenAPI operation id stays unique.
    @router.get("/files")
    @router.head("/files", include_in_schema=False)
    def serve_file(p: str, original: bool = False,
                   x_loramake_channel: str | None = Header(default=None)):
        """Serve a generated image by absolute path (SFW outputs tree only)."""
        path = state._sfw_outputs_path(p)
        # channel claim first: a scoped caller must not trigger an original fetch for another channel
        _channel_claim(path, _channel(x_loramake_channel))
        return _full_size(path, original)

    @router.get("/nsfw/files")
    @router.head("/nsfw/files", include_in_schema=False)
    def serve_nsfw_file(p: str, original: bool = False,
                        x_loramake_channel: str | None = Header(default=None)):
        """Serve an NSFW result by absolute path (nsfw/outputs tree only)."""
        path = state._nsfw_outputs_path(p)
        _channel_claim(path, _channel(x_loramake_channel))
        return _full_size(path, original)

    @router.get("/")
    def index():
        if (state.WEB_OUT / "index.html").is_file():
            return FileResponse(state.WEB_OUT / "index.html", headers={"Cache-Control": "no-cache"})
        raise HTTPException(status_code=503,
                            detail="frontend not built — run: cd app-web && pnpm build")

    return router
