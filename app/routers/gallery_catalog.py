"""Versioned gallery catalog endpoints: read + recoverable trash-delete."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field


class GalleryLoraSpecResponse(BaseModel):
    """Public LoRA metadata carried by one gallery catalog record."""

    model_config = ConfigDict(extra="forbid")

    name: str
    strength: float | None


class GalleryItemResponse(BaseModel):
    """Path-free, zone-bound projection of one gallery media record."""

    model_config = ConfigDict(extra="forbid")

    media_id: str = Field(pattern=r"^med_[0-9a-f]{24}$")
    zone: Literal["sfw", "nsfw"]
    name: str
    bytes: int = Field(ge=0)
    asset_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    variants: list[Literal["thumb_256", "thumb_480", "thumb_768", "original"]]
    collection: str
    driving_model: str
    line: str
    curated: bool
    intent: str | None
    engine: str | None
    run_date: str | None
    checkpoint: str | None
    loras: list[str]
    lora_specs: list[GalleryLoraSpecResponse]
    char_loras: list[str]
    style_loras: list[str]
    engine_line: str
    outfit: str | None
    gacha_date: str | None
    section: str | None
    qc_score: float | None
    concept: str | None
    prompt: str | None
    negative: str | None
    clip_skip: int | None
    seed: int | None
    sampler: str | None
    steps: int | None
    cfg: float | None
    resolution: str | None
    width: int | None
    height: int | None
    media: Literal["image", "video"]
    duration_s: float | None
    fps: int | None
    meta_source: Literal["sidecar", "png", "filename"]
    mtime: float
    # Recipe/plan facets (populated for both zones when derivable — sidecar or
    # checkpoint/path hints; tier/l1 remain NSFW-recipe taxonomy).
    character: str | None = None
    l1_flags: list[str] = Field(default_factory=list)
    leaf_ids: list[str] = Field(default_factory=list)
    tier: str | None = None
    prompt_domain: str | None = None
    base: str | None = None
    recipe_id: str | None = None
    recipe_request: str | None = None


class GalleryPageResponse(BaseModel):
    """Opaque-cursor gallery page returned from either zone."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.api.gallery-page.v1"] = Field(alias="schema")
    zone: Literal["sfw", "nsfw"]
    catalog_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    built_at: float
    index_stale: bool
    items: list[GalleryItemResponse]
    total: int = Field(ge=0)
    next_cursor: str | None


class GalleryProblemResponse(BaseModel):
    """RFC 7807-compatible machine problem returned by gallery endpoints."""

    model_config = ConfigDict(extra="forbid")

    type: str
    title: str
    status: int = Field(ge=400, le=599)
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    detail: str
    request_id: str


class GalleryDeleteRequest(BaseModel):
    """Batch trash-delete request; a single delete is a one-element list."""

    model_config = ConfigDict(extra="forbid")

    media_ids: list[str] = Field(description="Opaque media IDs to trash-delete (1-200).")


class GalleryDeleteFailure(BaseModel):
    """Per-item outcome for an ID that could not be trash-deleted."""

    model_config = ConfigDict(extra="forbid")

    media_id: str
    reason: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")


class GalleryDeleteResponse(BaseModel):
    """Per-item trash-delete outcomes plus the post-delete catalog revision."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.api.gallery-delete.v1"] = Field(alias="schema")
    zone: Literal["sfw", "nsfw"]
    deleted: list[str]
    failed: list[GalleryDeleteFailure]
    catalog_revision: str = Field(pattern=r"^[0-9a-f]{64}$")


_PROBLEM_RESPONSES = {
    status: {
        "description": "Versioned problem detail. See `code` for a stable failure category.",
        "content": {
            "application/problem+json": {
                "schema": GalleryProblemResponse.model_json_schema(),
            }
        },
    }
    for status in (400, 404, 409, 415, 422, 503)
}

_ASSET_SUCCESS_RESPONSE = {
    "description": "Opaque zone-bound gallery original or fixed thumbnail variant.",
    "content": {
        "image/png": {"schema": {"type": "string", "format": "binary"}},
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
        "image/webp": {"schema": {"type": "string", "format": "binary"}},
        "video/mp4": {"schema": {"type": "string", "format": "binary"}},
        "video/webm": {"schema": {"type": "string", "format": "binary"}},
    },
}


def _problem(exc: Exception, request: Request) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:20]}"
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

    def page_for_zone(
        request: Request,
        *,
        zone: str,
        cursor: str | None,
        limit: int,
        query: str,
        curated_only: bool,
        sort: str | None,
        character: str | None,
        tier: str | None,
        l1: str | None,
        base: str | None,
        recipe_id: str | None,
    ):
        try:
            return state.gallery_catalog.list_gallery(
                zone=zone,
                cursor=cursor,
                limit=limit,
                query=query,
                curated_only=curated_only,
                sort=sort,
                character=character,
                tier=tier,
                l1=l1,
                base=base,
                recipe_id=recipe_id,
            )
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)

    @router.get(
        "/api/v1/gallery",
        response_model=GalleryPageResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="List SFW gallery media",
    )
    def list_sfw_gallery(
        request: Request,
        cursor: str | None = None,
        limit: int = Query(default=30, ge=1, le=100),
        q: str = Query(default="", max_length=160),
        curated_only: bool = True,
        sort: str | None = Query(default=None, max_length=16),
        character: str | None = Query(default=None, max_length=64),
        tier: str | None = Query(default=None, max_length=8),
        l1: str | None = Query(default=None, max_length=64),
        base: str | None = Query(default=None, max_length=64),
        recipe_id: str | None = Query(default=None, max_length=96),
    ):
        return page_for_zone(
            request,
            zone="sfw",
            cursor=cursor,
            limit=limit,
            query=q,
            curated_only=curated_only,
            sort=sort,
            character=character,
            tier=tier,
            l1=l1,
            base=base,
            recipe_id=recipe_id,
        )

    @router.get(
        "/api/v1/nsfw/gallery",
        response_model=GalleryPageResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="List NSFW gallery media",
    )
    def list_nsfw_gallery(
        request: Request,
        cursor: str | None = None,
        limit: int = Query(default=30, ge=1, le=100),
        q: str = Query(default="", max_length=160),
        curated_only: bool = True,
        sort: str | None = Query(default=None, max_length=16),
        character: str | None = Query(default=None, max_length=64),
        tier: str | None = Query(default=None, max_length=8),
        l1: str | None = Query(default=None, max_length=64),
        base: str | None = Query(default=None, max_length=64),
        recipe_id: str | None = Query(default=None, max_length=96),
    ):
        return page_for_zone(
            request,
            zone="nsfw",
            cursor=cursor,
            limit=limit,
            query=q,
            curated_only=curated_only,
            sort=sort,
            character=character,
            tier=tier,
            l1=l1,
            base=base,
            recipe_id=recipe_id,
        )

    def asset_for_zone(
        request: Request,
        media_id: str,
        *,
        zone: str,
        variant: str,
        revision: str | None,
    ):
        try:
            path, media_type, etag = state.gallery_catalog.get_asset(
                zone,
                media_id,
                variant,
                expected_revision=revision,
            )
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                **(
                    state.gallery_catalog.IMMUTABLE_CACHE
                    if revision is not None
                    else {"Cache-Control": "private, no-cache"}
                ),
                "ETag": etag,
            },
        )

    # HEAD twins (schema-hidden): client preflight probes must not fall through to the
    # SPA mount; hiding them keeps OpenAPI operation ids unique.
    @router.get(
        "/api/v1/gallery/{media_id}/asset",
        response_class=FileResponse,
        responses={200: _ASSET_SUCCESS_RESPONSE, **_PROBLEM_RESPONSES},
        tags=["gallery"],
        summary="Read an opaque SFW gallery asset",
    )
    @router.head("/api/v1/gallery/{media_id}/asset", include_in_schema=False)
    def get_sfw_gallery_asset(
        request: Request,
        media_id: str,
        variant: str = "original",
        rev: str | None = Query(default=None, min_length=64, max_length=64),
    ):
        return asset_for_zone(request, media_id, zone="sfw", variant=variant, revision=rev)

    @router.get(
        "/api/v1/nsfw/gallery/{media_id}/asset",
        response_class=FileResponse,
        responses={200: _ASSET_SUCCESS_RESPONSE, **_PROBLEM_RESPONSES},
        tags=["gallery"],
        summary="Read an opaque NSFW gallery asset",
    )
    @router.head("/api/v1/nsfw/gallery/{media_id}/asset", include_in_schema=False)
    def get_nsfw_gallery_asset(
        request: Request,
        media_id: str,
        variant: str = "original",
        rev: str | None = Query(default=None, min_length=64, max_length=64),
    ):
        return asset_for_zone(request, media_id, zone="nsfw", variant=variant, revision=rev)

    def delete_for_zone(request: Request, payload: GalleryDeleteRequest, *, zone: str):
        try:
            return state.gallery_catalog.delete_media(zone=zone, media_ids=payload.media_ids)
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)

    @router.post(
        "/api/v1/gallery/delete",
        response_model=GalleryDeleteResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Trash-delete SFW gallery media (recoverable via legacy trash)",
    )
    def delete_sfw_gallery(request: Request, payload: GalleryDeleteRequest):
        return delete_for_zone(request, payload, zone="sfw")

    @router.post(
        "/api/v1/nsfw/gallery/delete",
        response_model=GalleryDeleteResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Trash-delete NSFW gallery media (recoverable via legacy trash)",
    )
    def delete_nsfw_gallery(request: Request, payload: GalleryDeleteRequest):
        return delete_for_zone(request, payload, zone="nsfw")

    return router
