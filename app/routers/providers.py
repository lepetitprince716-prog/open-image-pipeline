"""External-provider helper endpoints (cost preview; never submits paid work)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from loramake import image_providers


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/api/providers/whatif")
    def whatif(body: dict):
        """Free whatif cost preview for an external-provider brief subset.

        Accepts the same fields an external brief carries (provider, prompt, n,
        seed, steps, cfg, res, and for Civitai base + loras); performs no paid
        submission. The submit path in gen.py re-runs its own whatif and budget
        check, so this endpoint is advisory only.
        """
        provider = str((body or {}).get("provider") or "")
        prompt = str((body or {}).get("prompt") or "").strip()
        options: dict[str, Any] = {}
        if body.get("res") is not None:
            options["size"] = str(body["res"])
        for key in ("steps", "cfg"):
            if body.get(key) is not None:
                options[key] = body[key]
        if body.get("base") is not None:
            options["model"] = str(body["base"])
        loras = body.get("loras")
        if loras is not None:
            if not isinstance(loras, list) or not all(isinstance(x, str) for x in loras):
                raise HTTPException(status_code=400, detail="loras must be a list of strings")
            options["loras"] = loras
        try:
            return image_providers.preview_cost(
                provider, prompt,
                n=body.get("n"), seed=body.get("seed"),
                options=options or None,
            )
        except image_providers.ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
