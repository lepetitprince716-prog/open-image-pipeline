"""Network dashboard endpoints: status snapshot, enable toggle, forced probe."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..network.registry import BY_ID


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/network/status")
    def network_status():
        return state.network_status_snapshot()

    @router.post("/api/network/toggle")
    def network_toggle(body: dict):
        service_id = str(body.get("service") or "")
        if BY_ID.get(service_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown service: {service_id or '(missing)'}")
        enabled_flag = body.get("enabled")
        if not isinstance(enabled_flag, bool):
            raise HTTPException(status_code=400, detail="enabled must be a boolean")
        state.network_state.set_enabled(service_id, enabled_flag)
        snapshot = state.network_status_snapshot()
        card = next(item for item in snapshot["services"] if item["id"] == service_id)
        return {"service": card}

    @router.post("/api/network/probe")
    def network_probe(body: dict):
        service_id = str(body.get("service") or "")
        spec = BY_ID.get(service_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown service: {service_id or '(missing)'}")
        if spec.kind == "box":
            started = state.BOX_STATUS.refresh_async()
        else:
            started = state.NETWORK_PROBES.refresh_async(service_id)
        return {"service": service_id, "started": bool(started)}

    return router
