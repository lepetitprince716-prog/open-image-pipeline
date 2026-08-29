"""Job queue and SSE endpoints for the loramake app."""
from __future__ import annotations

import asyncio
import json
import queue
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/api/jobs", status_code=201)
    def submit_job(body: dict, idempotency_key: str | None = Header(default=None)):
        try:
            return state.jobs.submit(body, client_id=idempotency_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/jobs")
    def get_jobs(limit: int = 50):
        return state.jobs.list_jobs(limit=limit)

    @router.get("/api/jobs/search")
    def search_jobs(q: str = "", status: str = "", limit: int = 50, offset: int = 0):
        """Paged job history; q substring-matches the brief, status filters exactly."""
        return state.jobs.search_jobs(q=q, status=status, limit=limit, offset=offset)

    @router.get("/api/jobs/by-media")
    def job_by_media(zone: str, media_id: str):
        """Reverse-map a gallery catalog asset to the job that produced it."""
        try:
            path = state.gallery_catalog.asset_path(zone, media_id)
        except state.gallery_catalog.GalleryError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail)
        job = state.jobs.find_job_by_file(str(path))
        if job is None:
            raise HTTPException(status_code=404, detail="no job recorded for this asset")
        return job

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = state.jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @router.put("/api/jobs/{job_id}/review")
    def set_review(job_id: str, body: dict):
        """Upsert the per-job verdict (up/down [+ note]); verdict null clears it."""
        try:
            job = state.jobs.set_review(
                job_id, body.get("verdict"), note=str(body.get("note") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            job = state.jobs.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @router.get("/api/events")
    async def events(request: Request):
        """SSE job-state stream. The DB stays the source of truth after drops."""
        q = state.jobs.subscribe()

        async def stream():
            loop = asyncio.get_running_loop()
            try:
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
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                state.jobs.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return router
