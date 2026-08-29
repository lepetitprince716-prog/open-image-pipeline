"""Read-only compile surface for the isolated Anima structured Pilot."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from loramake import anima_structured_pilot as pilot


def create_router(_state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/anima-pilot/meta")
    def meta():
        return pilot.public_metadata()

    @router.post("/api/anima-pilot/compile")
    def compile_task(body: dict):
        allowed = {"source_text", "selection", "seed"}
        extra = sorted(set(body) - allowed)
        missing = sorted(allowed - set(body))
        problems = [f"{key} is required" for key in missing]
        problems.extend(f"{key} is not allowed" for key in extra)
        if problems:
            raise HTTPException(status_code=422, detail={"problems": problems})
        try:
            task = pilot.compile_task(body["source_text"], body["selection"], body["seed"])
        except pilot.ContractError as exc:
            raise HTTPException(status_code=422, detail={"problems": exc.problems}) from exc
        enabled = pilot.pilot_enabled()
        conditioning = pilot.effective_conditioning(task)
        return {
            "schema": "loramake.anima-structured-compile-result.v1",
            "enabled": enabled,
            "task_spec": task,
            "task_sha256": pilot.task_digest(task),
            "effective_conditioning": conditioning,
            "brief": {
                "tier": "anima_structured_pilot",
                "prompt": task["source"]["text"],
                "box": "zzz",
                "provider": "local",
                "sfw": True,
                "task_spec": task,
            },
        }

    return router
