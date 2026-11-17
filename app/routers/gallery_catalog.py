"""Versioned gallery catalog endpoints: read + recoverable trash-delete."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from loramake import remote_original


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
    variants: list[Literal["thumb_64", "thumb_256", "thumb_480", "thumb_768", "original"]]
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
    # Render placement + sampling extras (2026-09-03): the box name is public, the
    # machine address behind it never is. Gates are the persisted wd14 / QC / CCIP
    # outcomes from the sidecar, None when the engine recorded nothing.
    box: str | None = None
    scheduler: str | None = None
    shift: float | None = None
    elapsed_s: float | None = None
    script: str | None = None
    gate_wd14: Literal["pass", "fail"] | None = None
    gate_qc: Literal["pass", "fail"] | None = None
    gate_ccip: Literal["pass", "fail"] | None = None
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


class GalleryMetaResponse(BaseModel):
    """On-demand normalised sidecar (``loramake.image-meta.v1``) for one asset."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.api.image-meta.v1"] = Field(alias="schema")
    zone: Literal["sfw", "nsfw"]
    media_id: str = Field(pattern=r"^med_[0-9a-f]{24}$")
    asset_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["sidecar", "png", "none"]
    meta: dict[str, Any] | None


class ProvenanceAttemptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    attempt_no: int = Field(ge=0)
    status: str | None
    stage: str | None
    box: str | None
    remote_prompt_id: str | None
    timings: dict[str, Any] | None
    error_class: str | None


class ProvenanceJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str | None
    tier: str | None
    intent: str | None
    box: str | None
    provider: str | None
    direction: str | None
    group: str | None
    created_at: float | None
    finished_at: float | None
    review: dict[str, Any] | None
    attempt: ProvenanceAttemptResponse | None
    url: str


class ProvenanceRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_dir: str
    bin: Literal["raw", "selected"]
    run_date: str
    intent: str
    engine: str
    box: str
    runid: str
    record: dict[str, Any] | None


class ProvenanceVerdictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str | None
    decision: Literal["yes", "no", "comment"] | None
    comment: str | None
    actor: dict[str, Any] | None
    created_at: str | None


class ProvenanceExperimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^exp_[a-z0-9][a-z0-9_-]{2,63}$")
    zone: Literal["sfw", "nsfw"]
    title: str
    status: str | None
    origin: dict[str, Any] | None
    sample_id: str | None
    arm_id: str | None
    dimension_id: str | None
    seed: int | None
    asset_id: str = Field(pattern=r"^ast_[0-9a-f]{20}$")
    verdict: ProvenanceVerdictResponse | None
    url: str


class ProvenanceReviewBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch: str
    mode: Literal["view", "review"]
    index: int = Field(ge=0)
    name: str | None
    verdict: Literal["love", "yes", "no", "comment", "pending"]
    reasons: list[str]
    comment: str | None
    url: str


class ProvenanceResponse(BaseModel):
    """Read-only join: job → attempt → experiments/batches → run folder → :8771 verdicts."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.api.provenance.v1"] = Field(alias="schema")
    zone: Literal["sfw", "nsfw"]
    media_id: str = Field(pattern=r"^med_[0-9a-f]{24}$")
    repo_path: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    job: ProvenanceJobResponse | None
    run: ProvenanceRunResponse | None
    experiments: list[ProvenanceExperimentResponse]
    review_batches: list[ProvenanceReviewBatchResponse]
    urls: dict[str, str]


class GalleryRefreshResponse(BaseModel):
    """Result of a forced catalog rescan (registration hooks call this instead of polling)."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.api.gallery-refresh.v1"] = Field(alias="schema")
    zone: Literal["sfw", "nsfw"]
    catalog_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    built_at: float
    total: int = Field(ge=0)
    experiments_catalog_revision: int | None


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
        full: bool = False,
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
        if full and variant == "original" and remote_original.stays_remote(path):
            raise HTTPException(status_code=409, detail=remote_original.STAYS_REMOTE_DETAIL)
        if full and variant == "original" and remote_original.pending(path):
            # The download link asks for the full-size file: swap the off-LAN preview for the
            # original first. The bytes change, so this response must not be cached by revision.
            if not remote_original.fetch(path):
                raise HTTPException(status_code=502, detail="the original is still on the render box, "
                                                            "which is not reachable right now")
            return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, no-cache"})
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
        full: bool = False,
    ):
        return asset_for_zone(request, media_id, zone="sfw", variant=variant, revision=rev, full=full)

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
        full: bool = False,
    ):
        return asset_for_zone(request, media_id, zone="nsfw", variant=variant, revision=rev, full=full)

    def meta_for_zone(request: Request, response: Response, media_id: str, *, zone: str):
        try:
            payload, etag = state.gallery_catalog.get_meta(zone, media_id)
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "private, no-cache"
        return payload

    @router.get(
        "/api/v1/gallery/{media_id}/meta",
        response_model=GalleryMetaResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Read the normalised sidecar of an SFW gallery asset",
    )
    def get_sfw_gallery_meta(request: Request, response: Response, media_id: str):
        return meta_for_zone(request, response, media_id, zone="sfw")

    @router.get(
        "/api/v1/nsfw/gallery/{media_id}/meta",
        response_model=GalleryMetaResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Read the normalised sidecar of an NSFW gallery asset",
    )
    def get_nsfw_gallery_meta(request: Request, response: Response, media_id: str):
        return meta_for_zone(request, response, media_id, zone="nsfw")

    def provenance_for_zone(request: Request, media_id: str, *, zone: str):
        try:
            return state.provenance.resolve(zone, media_id)
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)

    @router.get(
        "/api/v1/gallery/{media_id}/provenance",
        response_model=ProvenanceResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Trace an SFW gallery asset to its job, run, batches and verdicts",
    )
    def get_sfw_gallery_provenance(request: Request, media_id: str):
        return provenance_for_zone(request, media_id, zone="sfw")

    @router.get(
        "/api/v1/nsfw/gallery/{media_id}/provenance",
        response_model=ProvenanceResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Trace an NSFW gallery asset to its job, run, batches and verdicts",
    )
    def get_nsfw_gallery_provenance(request: Request, media_id: str):
        return provenance_for_zone(request, media_id, zone="nsfw")

    def refresh_for_zone(request: Request, *, zone: str):
        try:
            snapshot = state.gallery_catalog.refresh(zone)
        except state.gallery_catalog.GalleryError as exc:
            return _problem(exc, request)
        experiments_revision: int | None = None
        experiments_mod = getattr(state, "experiments", None)
        if experiments_mod is not None:
            try:
                experiments_revision = int(experiments_mod.scan(force=True)["catalog_revision"])
            except Exception:
                experiments_revision = None
        return {
            "schema": "loramake.api.gallery-refresh.v1",
            "zone": zone,
            "catalog_revision": snapshot.revision,
            "built_at": snapshot.built_at,
            "total": len(snapshot.records),
            "experiments_catalog_revision": experiments_revision,
        }

    @router.post(
        "/api/v1/gallery/refresh",
        response_model=GalleryRefreshResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Force an SFW catalog rescan (batch registration hook)",
    )
    def refresh_sfw_gallery(request: Request):
        return refresh_for_zone(request, zone="sfw")

    @router.post(
        "/api/v1/nsfw/gallery/refresh",
        response_model=GalleryRefreshResponse,
        responses=_PROBLEM_RESPONSES,
        tags=["gallery"],
        summary="Force an NSFW catalog rescan (batch registration hook)",
    )
    def refresh_nsfw_gallery(request: Request):
        return refresh_for_zone(request, zone="nsfw")

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
