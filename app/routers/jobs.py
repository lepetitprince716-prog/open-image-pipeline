"""Job queue and SSE endpoints for the loramake app."""
from __future__ import annotations

import asyncio
import io
import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from loramake import remote_original

# Longest-side clamp for the job-result webp preview; the tunnel uplink is ~460 KB/s.
_PREVIEW_MAX_EDGE = 2048
_PREVIEW_MIN_EDGE = 64


def _result_paths(result: dict):
    for key in ("survivors", "images", "new_files", "best"):
        value = result.get(key)
        for path in [value] if isinstance(value, str) else value or []:
            if isinstance(path, str):
                yield Path(path)


def _keep_originals(job: dict) -> None:
    """A kept job must not stay a preview in the archive: pull its originals in the background,
    except those that stay on their box."""
    todo = [p for p in set(_result_paths(job.get("result") or {}))
            if remote_original.pending(p) and not remote_original.stays_remote(p)]
    if todo:
        threading.Thread(target=lambda: [remote_original.fetch(p) for p in todo],
                         daemon=True, name=f"originals-{job.get('id')}").start()


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    def _channel(header: str | None) -> str | None:
        """Validated request channel; a malformed one is a 400, never a silent widening."""
        try:
            return state.jobs.parse_channel(header)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _visible(job: dict | None, channel: str | None) -> dict | None:
        """Channel-scoped visibility: a scoped caller sees only its own channel's jobs."""
        if job is None:
            return None
        if channel is not None and state.jobs.job_channel(job) != channel:
            return None
        return job

    @router.post("/api/jobs", status_code=201)
    def submit_job(body: dict, idempotency_key: str | None = Header(default=None),
                   x_loramake_channel: str | None = Header(default=None)):
        # A revise brief carries parent_path, which names a file to read and is meant to be
        # derived by the server from a catalog id. Only /api/design/revisions does that
        # derivation, so accepting one here would let a caller supply the path itself and
        # read any image on disk with a sidecar -- including across the sfw/nsfw boundary the
        # catalog exists to keep.
        if isinstance(body, dict) and body.get("tier") == "revise":
            raise HTTPException(
                status_code=400,
                detail="revise briefs go to POST /api/design/revisions, which resolves the "
                       "parent image from parent_media")
        try:
            return state.jobs.submit(body, client_id=idempotency_key,
                                     channel=_channel(x_loramake_channel))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/jobs")
    def get_jobs(limit: int = 50, x_loramake_channel: str | None = Header(default=None)):
        return state.jobs.list_jobs(limit=limit, channel=_channel(x_loramake_channel))

    @router.get("/api/jobs/search")
    def search_jobs(q: str = "", status: str = "", limit: int = 50, offset: int = 0,
                    tier: str = "", x_loramake_channel: str | None = Header(default=None)):
        """Paged job history; q substring-matches the brief, status and tier filter exactly."""
        return state.jobs.search_jobs(q=q, status=status, limit=limit, offset=offset,
                                      channel=_channel(x_loramake_channel), tier=tier)

    @router.get("/api/jobs/by-media")
    def job_by_media(zone: str, media_id: str,
                     x_loramake_channel: str | None = Header(default=None)):
        """Reverse-map a gallery catalog asset to the job that produced it."""
        channel = _channel(x_loramake_channel)
        try:
            path = state.gallery_catalog.asset_path(zone, media_id)
        except state.gallery_catalog.GalleryError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        job = _visible(state.jobs.find_job_by_file(str(path)), channel)
        if job is None:
            raise HTTPException(status_code=404, detail="no job recorded for this asset")
        return job

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str, x_loramake_channel: str | None = Header(default=None)):
        job = _visible(state.jobs.get_job(job_id), _channel(x_loramake_channel))
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @router.get("/api/jobs/{job_id}/preview")
    def job_preview(job_id: str, w: int = 1024,
                    x_loramake_channel: str | None = Header(default=None)):
        """Webp preview of a done job's primary result image.

        The bounded media surface for off-LAN channels: it serves only images a job
        result claims, never an arbitrary path, and a scoped channel still sees only
        its own jobs.
        """
        job = _visible(state.jobs.get_job(job_id), _channel(x_loramake_channel))
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail=f"job is {job.get('status')}")
        path = state.jobs.primary_result_path(job)
        if path is None:
            raise HTTPException(status_code=404, detail="job result has no readable image")
        edge = max(_PREVIEW_MIN_EDGE, min(int(w), _PREVIEW_MAX_EDGE))
        from PIL import Image

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((edge, edge), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "WEBP", quality=80)
        except OSError as exc:
            raise HTTPException(status_code=404,
                                detail="result image is not decodable") from exc
        return Response(content=buf.getvalue(), media_type="image/webp",
                        headers={"Cache-Control": "no-cache"})

    @router.put("/api/jobs/{job_id}/review")
    def set_review(job_id: str, body: dict,
                   x_loramake_channel: str | None = Header(default=None)):
        """Upsert the per-job verdict (up/down [+ note]); verdict null clears it."""
        channel = _channel(x_loramake_channel)
        if _visible(state.jobs.get_job(job_id), channel) is None:
            raise HTTPException(status_code=404, detail="no such job")
        try:
            job = state.jobs.set_review(
                job_id, body.get("verdict"), note=str(body.get("note") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if body.get("verdict") == "up":
            _keep_originals(job)
        return job

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, x_loramake_channel: str | None = Header(default=None)):
        channel = _channel(x_loramake_channel)
        if _visible(state.jobs.get_job(job_id), channel) is None:
            raise HTTPException(status_code=404, detail="no such job")
        try:
            job = state.jobs.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @router.get("/api/events")
    async def events(request: Request):
        """SSE job-state stream. The DB stays the source of truth after drops.

        A channel-scoped stream drops events for jobs outside the channel, including
        bare progress events (they carry no brief to scope by)."""
        channel = _channel(request.headers.get("x-loramake-channel"))
        q = state.jobs.subscribe()

        async def stream():
            loop = asyncio.get_running_loop()
            try:
                # GZipMiddleware withholds the response headers until the first body
                # chunk; open the stream with a comment frame so EventSource.onopen fires
                # at once instead of after the first event or the 15s keepalive.
                yield ": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        # Dedicated SSE pool: a 15s blocking get per open stream would
                        # exhaust the small asyncio default executor tabs alone.
                        event = await loop.run_in_executor(state.SSE_EXECUTOR, q.get, True, 15)
                    except queue.Empty:
                        yield ": keepalive\n\n"  # comment frame keeps proxies/tunnels open
                        continue
                    if channel is not None:
                        job = event.get("job") if isinstance(event, dict) else None
                        if not isinstance(job, dict) or state.jobs.job_channel(job) != channel:
                            continue
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                state.jobs.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return router
