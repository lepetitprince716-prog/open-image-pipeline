"""Natural-language prompt assist: fill + compose + validate."""
from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException

from loramake.prompt_compose import (
    PromptComposeError,
    assist_compose,
    load_assist_fill_instruction,
)
from loramake.tags import TagDBError


def _nsfw_template(state: Any) -> str:
    path = state.NSFW_ASSIST_PROMPT
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="NSFW assist template missing; refuse SFW fallback",
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise HTTPException(
            status_code=503,
            detail="NSFW assist template empty; refuse SFW fallback",
        )
    return text


def _optional_slot_proposal(state: Any, instruction: str, text: str) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if not state.GROK_BIN.is_file():
        warnings.append("assist LLM missing; filled from text only (lint/warn)")
        return None, warnings
    try:
        out = state._assist_llm(f"{instruction}\n\nUser description:\n{text}")
    except subprocess.TimeoutExpired:
        warnings.append("assist LLM timed out; filled from text only (lint/warn)")
        return None, warnings
    except Exception as exc:
        warnings.append(f"assist LLM failed ({exc}); filled from text only (lint/warn)")
        return None, warnings
    if not out:
        warnings.append("assist LLM returned nothing; filled from text only (lint/warn)")
        return None, warnings
    return out, warnings


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/api/assist")
    def assist(body: dict):
        """NL → fill + compose + validate. Never returns a raw remembered tag wall."""
        text = str((body or {}).get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > state.MAX_ASSIST_INPUT:
            raise HTTPException(status_code=400, detail=f"text too long ({state.MAX_ASSIST_INPUT} chars max)")
        nsfw = bool((body or {}).get("nsfw"))
        if nsfw:
            instruction = _nsfw_template(state)
        else:
            try:
                instruction = load_assist_fill_instruction()
            except PromptComposeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        llm_text, llm_warnings = _optional_slot_proposal(state, instruction, text)
        slots = (body or {}).get("slots")
        if slots is not None and not isinstance(slots, dict):
            raise HTTPException(status_code=400, detail="slots must be a JSON object")
        try:
            payload = assist_compose(
                text,
                nsfw=nsfw,
                domain=(body or {}).get("domain") or (body or {}).get("prompt_domain"),
                profile=(body or {}).get("profile"),
                pack_id=(body or {}).get("pack") or (body or {}).get("pack_id"),
                slots=slots,
                llm_text=llm_text,
            )
        except TagDBError as exc:
            raise HTTPException(status_code=503, detail=f"TagDB not ready: {exc}") from exc
        except PromptComposeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload["warnings"] = list(payload.get("warnings") or []) + llm_warnings
        return {
            "ir": payload["ir"],
            "prompt": payload["prompt"],
            "prompt_domain": payload["prompt_domain"],
            "dropped_tags": payload["dropped_tags"],
            "rewrites": payload["rewrites"],
            "loras_suggested": payload["loras_suggested"],
            "tagdb_ready": payload.get("tagdb_ready"),
            "lint": payload.get("lint"),
            "warnings": payload["warnings"],
        }

    return router
