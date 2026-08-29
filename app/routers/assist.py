"""Natural-language prompt assist endpoint."""
from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/api/assist")
    def assist(body: dict):
        """Natural language -> booru tag prompt via a single-turn CLI helper."""
        text = str((body or {}).get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > state.MAX_ASSIST_INPUT:
            raise HTTPException(status_code=400, detail=f"text too long ({state.MAX_ASSIST_INPUT} chars max)")
        template = state.ASSIST_TEMPLATE
        if body.get("nsfw") and state.NSFW_ASSIST_PROMPT.is_file():
            template = state.NSFW_ASSIST_PROMPT.read_text(encoding="utf-8").strip() or state.ASSIST_TEMPLATE
        if not state.GROK_BIN.is_file():
            raise HTTPException(status_code=503, detail="assist backend not installed (grok CLI missing)")
        try:
            out = state._assist_llm(f"{template}\n\nUser description:\n{text}")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=503, detail="assist timed out")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"assist failed: {exc}")
        if not out:
            raise HTTPException(status_code=503, detail="assist returned nothing")
        return {"prompt": out}

    return router
