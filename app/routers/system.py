"""System, metadata, catalog-refresh, and maintenance endpoints."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from .. import spend as spend_mod
from .. import stats as stats_mod


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    def health():
        # root: which checkout this process serves. A worktree ROOT only sees that
        # tree's outputs, which reads as "the gallery is empty" (2026-08-08 incident).
        return {"ok": True, "liveness": "ok", "root": str(state.ROOT),
                "tiers": list(state.brief_mod.TIERS),
                "boxes": state.BOX_STATUS.snapshot()}

    @router.get("/api/spend/today")
    def spend_today():
        """Dashboard strip (F11): today's Anlas + Buzz from the ledgers, BOXA state from cache."""
        return spend_mod.spend_today(
            state.ROOT, state.BOX_STATUS.snapshot(trigger_refresh=False))

    @router.get("/api/stats")
    def machine_stats(days: int = 7):
        """Dashboard tuning strip: box utilisation, job timings and spend over `days` (1-30)."""
        if not 1 <= days <= stats_mod.MAX_DAYS:
            raise HTTPException(status_code=400, detail=f"days must be 1..{stats_mod.MAX_DAYS}")
        return stats_mod.compute(state.ROOT, state.jobs.DB_PATH, days=days)

    @router.get("/api/meta")
    def get_meta(request: Request):
        cached = state.meta_mod.snapshot()
        payload = {"tiers": list(state.brief_mod.TIERS), "boxes": list(state.brief_mod.BOXES),
                   **cached, "nsfw_available": state.brief_mod.GEN_NSFW.is_file(),
                   "gacha_nsfw_available": state.brief_mod.GEN_GACHA_NSFW.is_file()}
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
        """Reachability of the standalone LoRA Manager on BOXA (:9000)."""
        return state.lora_manager_status()

    @router.post("/api/lora-manager")
    def lora_manager_action(body: dict):
        """Start or stop the standalone manager via scripts/boxa_lora_manager.py."""
        action = body.get("action")
        if action not in ("start", "stop"):
            raise HTTPException(status_code=400, detail="action must be start or stop")
        return state.lora_manager_action(action)

    @router.get("/api/forge")
    def forge_status():
        """Reachability + loaded checkpoint of the BOXA Forge API (17862 loopback)."""
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

    @router.post("/api/gacha/preview")
    def gacha_preview(body: dict | None = None):
        """Sample catalogs/t2i only — no GPU. Same knobs as a gacha2.0 brief."""
        payload = body or {}
        if payload.get("nsfw") is True:
            raise HTTPException(
                status_code=400,
                detail="gacha2.0 is the SFW catalog line; NSFW remake is nsfw/t2i_gacha_nsfw.py",
            )
        problems = state.brief_mod.validate({
            "tier": state.brief_mod.GACHA_TIER,
            "box": payload.get("box") or "boxa",
            **{key: payload[key] for key in ("n", "seed", "chaos", "blend_p",
                                             "style_lora_p", "gate_min", "overscale")
               if key in payload},
        })
        if problems:
            raise HTTPException(status_code=400, detail="; ".join(problems))
        argv = [sys.executable, str(state.ROOT / "scripts" / "t2i_gacha.py"),
                "--dry-run", "--json"]
        count = payload.get("n")
        argv += ["--count", str(count if count is not None else 6)]
        if payload.get("seed") is not None:
            argv += ["--seed", str(payload["seed"])]
        for key, flag in (
            ("chaos", "--chaos"),
            ("blend_p", "--blend-p"),
            ("style_lora_p", "--style-lora-p"),
            ("gate_min", "--gate-min"),
            ("overscale", "--overscale"),
        ):
            if payload.get(key) is not None:
                argv += [flag, str(payload[key])]
        try:
            proc = subprocess.run(
                argv, cwd=str(state.ROOT), capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="gacha preview timed out") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "preview failed").strip().splitlines()
            raise HTTPException(status_code=400, detail=tail[-1] if tail else "preview failed")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="gacha preview did not return JSON") from exc

    @router.post("/api/gacha/nsfw/preview")
    def gacha_nsfw_preview(body: dict | None = None):
        """Sample the NSFW remake PLAN only — no GPU. Distinct from /api/gacha/preview."""
        from app import gacha_nsfw as gacha_nsfw_mod

        payload = body or {}
        if payload.get("nsfw") is not True:
            raise HTTPException(
                status_code=400,
                detail="t2i_gacha_nsfw requires nsfw=true (independent remake; not gacha2.0)",
            )
        knobs = (
            "n", "seed", "style_lora_p", "concept_p", "accent_p", "gate_min", "overscale",
            "pin_char", "pin_exposure", "pin_behavior", "gacha_quality", "gacha_legacy_turbo",
        )
        problems = state.brief_mod.validate({
            **payload,
            "tier": state.brief_mod.GACHA_NSFW_TIER,
            "box": payload.get("box") or "boxa",
            "nsfw": True,
        })
        if problems:
            raise HTTPException(status_code=400, detail="; ".join(problems))
        try:
            return gacha_nsfw_mod.preview({
                key: payload[key] for key in knobs if key in payload
            })
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc) or "preview failed") from exc

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
