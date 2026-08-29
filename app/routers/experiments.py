"""Versioned, frontend-neutral experiment result and review endpoints."""
from __future__ import annotations

import asyncio
import json
import queue
from functools import partial
from typing import Any

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse


def problem_response(exc: Exception, request_id: str) -> JSONResponse:
    """Map the stable domain error to RFC 9457-style problem JSON."""
    return JSONResponse(
        status_code=exc.status,
        media_type="application/problem+json",
        content={
            "type": f"urn:loramake:problem:{exc.code}",
            "title": exc.code,
            "status": exc.status,
            "code": exc.code,
            "detail": exc.detail,
            "request_id": request_id,
        },
    )


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/capabilities")
    def capabilities():
        return state.experiments.capabilities()

    def list_for_zone(
        *,
        zone: str,
        cursor: str | None,
        limit: int,
        status: str | None,
        model: str | None,
    ):
        return state.experiments.list_experiments(
            zone=zone,
            cursor=cursor,
            limit=limit,
            status=status,
            model=model,
        )

    @router.get("/api/v1/experiments")
    def list_sfw_experiments(
        cursor: str | None = None,
        limit: int = Query(default=30, ge=1, le=100),
        status: str | None = None,
        model: str | None = None,
    ):
        return list_for_zone(zone="sfw", cursor=cursor, limit=limit, status=status, model=model)

    @router.get("/api/v1/nsfw/experiments")
    def list_nsfw_experiments(
        cursor: str | None = None,
        limit: int = Query(default=30, ge=1, le=100),
        status: str | None = None,
        model: str | None = None,
    ):
        return list_for_zone(zone="nsfw", cursor=cursor, limit=limit, status=status, model=model)

    def event_stream(request: Request, *, zone: str, since_revision: int):
        subscriber = state.experiments.subscribe()

        async def stream():
            loop = asyncio.get_running_loop()
            last_revision = since_revision
            try:
                while True:
                    # Dedicated SSE pool (see server.SSE_EXECUTOR): keeps long stream
                    # waits off the small asyncio default executor.
                    durable = await loop.run_in_executor(
                        state.SSE_EXECUTOR,
                        partial(
                            state.experiments.catalog_events_since,
                            zone=zone,
                            since_revision=last_revision,
                        ),
                    )
                    for event in durable:
                        last_revision = max(last_revision, int(event["revision"]))
                        yield (
                            f"id: {event['revision']}\n"
                            "event: catalog\n"
                            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        )
                    if len(durable) < 1000:
                        break
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await loop.run_in_executor(state.SSE_EXECUTOR, subscriber.get, True, 15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    if event["zone"] != zone or int(event["revision"]) <= last_revision:
                        continue
                    last_revision = int(event["revision"])
                    yield (
                        f"id: {event['revision']}\n"
                        "event: catalog\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
            finally:
                state.experiments.unsubscribe(subscriber)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def revision_from_headers(since_revision: int, last_event_id: str | None) -> int:
        if last_event_id is None:
            return since_revision
        try:
            return max(since_revision, int(last_event_id))
        except ValueError as exc:
            raise state.experiments.ExperimentError(
                400, "invalid_revision", "Last-Event-ID must be an integer catalog revision"
            ) from exc

    @router.get("/api/v1/experiments/events")
    async def sfw_events(
        request: Request,
        since_revision: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        return event_stream(
            request,
            zone="sfw",
            since_revision=revision_from_headers(since_revision, last_event_id),
        )

    @router.get("/api/v1/nsfw/experiments/events")
    async def nsfw_events(
        request: Request,
        since_revision: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        return event_stream(
            request,
            zone="nsfw",
            since_revision=revision_from_headers(since_revision, last_event_id),
        )

    def detail_for_zone(experiment_id: str, *, zone: str):
        payload, etag = state.experiments.get_experiment(experiment_id, zone=zone)
        return JSONResponse(content=payload, headers={"ETag": etag})

    @router.get("/api/v1/experiments/{experiment_id}")
    def get_sfw_experiment(experiment_id: str):
        return detail_for_zone(experiment_id, zone="sfw")

    @router.get("/api/v1/nsfw/experiments/{experiment_id}")
    def get_nsfw_experiment(experiment_id: str):
        return detail_for_zone(experiment_id, zone="nsfw")

    def asset_for_zone(
        experiment_id: str,
        asset_id: str,
        *,
        zone: str,
        variant: str,
    ):
        path, asset = state.experiments.get_asset(experiment_id, asset_id, zone=zone)
        if variant == "original":
            target = path
            media_type = asset["media_type"]
            etag = f'"sha256-{asset["sha256"]}"'
        elif variant in {"thumb_256", "thumb_480", "thumb_768"}:
            width = int(variant.removeprefix("thumb_"))
            target = state.experiments.thumbnail(path, asset, width, zone=zone)
            media_type = "image/webp"
            etag = f'"sha256-{asset["sha256"]}-{width}"'
        else:
            raise state.experiments.ExperimentError(
                400, "invalid_asset_variant", "asset variant is unsupported"
            )
        return FileResponse(
            target,
            media_type=media_type,
            headers={**state.experiments.IMMUTABLE_CACHE, "ETag": etag},
        )

    # HEAD twins (schema-hidden): client preflight probes must not fall through to the
    # SPA mount; hiding them keeps OpenAPI operation ids unique.
    @router.get("/api/v1/experiments/{experiment_id}/assets/{asset_id}")
    @router.head("/api/v1/experiments/{experiment_id}/assets/{asset_id}", include_in_schema=False)
    def get_sfw_asset(
        experiment_id: str,
        asset_id: str,
        variant: str = "original",
    ):
        return asset_for_zone(experiment_id, asset_id, zone="sfw", variant=variant)

    @router.get("/api/v1/nsfw/experiments/{experiment_id}/assets/{asset_id}")
    @router.head("/api/v1/nsfw/experiments/{experiment_id}/assets/{asset_id}", include_in_schema=False)
    def get_nsfw_asset(
        experiment_id: str,
        asset_id: str,
        variant: str = "original",
    ):
        return asset_for_zone(experiment_id, asset_id, zone="nsfw", variant=variant)

    def reviews_for_zone(
        experiment_id: str,
        *,
        zone: str,
        cursor: str | None,
        limit: int,
    ):
        return state.experiments.list_reviews(
            experiment_id,
            zone=zone,
            cursor=cursor,
            limit=limit,
        )

    @router.get("/api/v1/experiments/{experiment_id}/reviews")
    def list_sfw_reviews(
        experiment_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=100),
    ):
        return reviews_for_zone(
            experiment_id, zone="sfw", cursor=cursor, limit=limit
        )

    @router.get("/api/v1/nsfw/experiments/{experiment_id}/reviews")
    def list_nsfw_reviews(
        experiment_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=100),
    ):
        return reviews_for_zone(
            experiment_id, zone="nsfw", cursor=cursor, limit=limit
        )

    def create_for_zone(
        experiment_id: str,
        *,
        zone: str,
        body: dict,
        idempotency_key: str | None,
        client_id: str | None,
        if_match: str | None,
    ):
        missing = [
            name
            for name, value in (
                ("Idempotency-Key", idempotency_key),
                ("X-Loramake-Client", client_id),
                ("If-Match", if_match),
            )
            if not value
        ]
        if missing:
            raise state.experiments.ExperimentError(
                400, "missing_precondition", f"missing required header: {', '.join(missing)}"
            )
        event, created = state.experiments.create_review(
            experiment_id,
            zone=zone,
            body=body,
            client_id=client_id,
            idempotency_key=idempotency_key,
            if_match=if_match,
        )
        return JSONResponse(
            status_code=201,
            content=event,
            headers={"Idempotent-Replayed": "false" if created else "true"},
        )

    @router.post("/api/v1/experiments/{experiment_id}/reviews", status_code=201)
    def create_sfw_review(
        experiment_id: str,
        body: dict,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        client_id: str | None = Header(default=None, alias="X-Loramake-Client"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        return create_for_zone(
            experiment_id,
            zone="sfw",
            body=body,
            idempotency_key=idempotency_key,
            client_id=client_id,
            if_match=if_match,
        )

    @router.post("/api/v1/nsfw/experiments/{experiment_id}/reviews", status_code=201)
    def create_nsfw_review(
        experiment_id: str,
        body: dict,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        client_id: str | None = Header(default=None, alias="X-Loramake-Client"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        return create_for_zone(
            experiment_id,
            zone="nsfw",
            body=body,
            idempotency_key=idempotency_key,
            client_id=client_id,
            if_match=if_match,
        )

    return router
