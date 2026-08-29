"""System, metadata, catalog-refresh, and maintenance endpoints."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        # root: which checkout this process serves. A worktree ROOT only sees that
        # tree's outputs, which reads as "the gallery is empty" (2026-08-08 incident).
        return {"ok": True, "liveness": "ok", "root": str(state.ROOT),
                "tiers": list(state.brief_mod.TIERS),
                "boxes": state.BOX_STATUS.snapshot()}

    @router.get("/api/meta")
    def get_meta(request: Request):
        cached = state.meta_mod.snapshot()
        payload = {"tiers": list(state.brief_mod.TIERS), "boxes": list(state.brief_mod.BOXES),
                   **cached, "nsfw_available": state.brief_mod.GEN_NSFW.is_file()}
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        etag = '"' + hashlib.sha256(raw).hexdigest() + '"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(raw, media_type="application/json", headers=headers)

    @router.post("/api/loras/refresh")
    def loras_refresh(body: dict | None = None):
        """Manual catalog refresh. GET /api/loras/refresh polls the latest outcome."""
        return state.lora_sync.refresh(force=bool((body or {}).get("force")))

    @router.get("/api/loras/refresh")
    def loras_refresh_status():
        return state.lora_sync.status()

    @router.put("/api/loras/archive")
    def loras_archive(body: dict):
        """Reversible soft-archive mark for one catalog LoRA (never touches weights)."""
        file = str(body.get("file") or "").strip()
        if not file:
            raise HTTPException(status_code=400, detail="file is required")
        archived = state.lora_archive.set_archived(file, bool(body.get("archived", True)))
        return {"file": file, "archived": archived}

    @router.get("/api/lora-manager")
    def lora_manager_status():
        """Reachability of the standalone LoRA Manager on Z3 (:9000)."""
        return state.lora_manager_status()

    @router.post("/api/lora-manager")
    def lora_manager_action(body: dict):
        """Start or stop the standalone manager via scripts/z3_lora_manager.py."""
        action = body.get("action")
        if action not in ("start", "stop"):
            raise HTTPException(status_code=400, detail="action must be start or stop")
        return state.lora_manager_action(action)

    @router.get("/api/forge")
    def forge_status():
        """Reachability + loaded checkpoint of the Z3 Forge API (17862 loopback)."""
        return state.forge_status()

    @router.post("/api/forge")
    def forge_action(body: dict):
        """Manage the Forge service via scripts/forge_service.py (start/stop/refresh/unload)."""
        action = body.get("action")
        if action not in state.FORGE_ACTIONS:
            raise HTTPException(status_code=400,
                                detail=f"action must be one of {', '.join(state.FORGE_ACTIONS)}")
        return state.forge_action(action)

    @router.get("/api/nsfw/meta")
    def get_nsfw_meta():
        """NSFW presets are fetched only after the frontend's confirmed NSFW switch."""
        return {"presets": state.nsfw_meta.presets()}

    @router.get("/api/gacha/today")
    def gacha_today():
        date = state.time.strftime("%Y-%m-%d")
        day = state.GACHA_DIR / date
        status = None
        manifest = day / "manifest.json"
        if manifest.is_file():
            try:
                status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                status = None
        images = sorted(str(p) for p in day.glob("*.png")) if day.is_dir() else []
        return {"date": date, "status": status, "images": images}

    @router.post("/api/free")
    def free_vram(box: str):
        """Manual VRAM release on one ComfyUI box."""
        if box not in state.gen.HOSTS:
            raise HTTPException(status_code=400, detail=f"box must be one of {', '.join(state.gen.HOSTS)}")
        network_state = getattr(state, "network_state", None)
        if network_state is not None and not network_state.enabled(box):
            raise HTTPException(status_code=409, detail=f"box '{box}' 已在网络页停用")
        return {"box": box, "ok": bool(state.gen._free_vram(state.gen.HOSTS[box]))}

    return router
