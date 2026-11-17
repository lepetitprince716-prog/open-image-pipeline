#!/usr/bin/env python3
"""Pure classification of ComfyUI failure payloads into what an operator does next.

Extracted from ``loramake.comfy`` (C3). Every function here is a total function of its
arguments — no transport, no configuration, no module state — so the transport layer can call
it and a caller that only needs to read an error payload need not pull the driver in.
``loramake.comfy`` re-exports all three names, and its own call sites still resolve them
through its globals so a ``comfy.<name>`` monkeypatch keeps reaching them.
"""
from __future__ import annotations

import json


def classify_gen_error(exception_type: str, message: str) -> str:
    """Coarse triage bucket for a ComfyUI failure: oom | model_missing | node_missing | other.

    Buckets match what the operator does next (free VRAM / fix the model name / install the
    node), not ComfyUI's exception taxonomy — keep them few and stable."""
    blob = f"{exception_type} {message}".lower()
    if "out of memory" in blob or "outofmemory" in blob or "cuda error" in blob:
        return "oom"
    if ("not in list" in blob or "value_not_in_list" in blob or "no such file" in blob
            or "safetensors" in blob and "not found" in blob):
        return "model_missing"
    if "does not exist" in blob or "invalid_prompt" in blob or "unknown node" in blob:
        return "node_missing"
    return "other"


def _execution_error_detail(item: dict) -> dict | None:
    """The execution_error payload from a /history item's status.messages, reduced to the
    fields an operator acts on. None when the item carries no parseable error message."""
    messages = (item.get("status") or {}).get("messages") or []
    for entry in messages:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        kind, payload = entry
        if kind != "execution_error" or not isinstance(payload, dict):
            continue
        exception_type = str(payload.get("exception_type") or "")
        exception_message = str(payload.get("exception_message") or "").strip()
        traceback_lines = payload.get("traceback") or []
        detail = {
            "node_id": str(payload.get("node_id") or ""),
            "node_type": str(payload.get("node_type") or ""),
            "exception_type": exception_type,
            "exception_message": exception_message[:1000],
            "error_class": classify_gen_error(exception_type, exception_message),
        }
        if isinstance(traceback_lines, list) and traceback_lines:
            detail["traceback_tail"] = "".join(str(ln) for ln in traceback_lines[-3:])[-1000:]
        return detail
    return None


def _submit_error_detail(body: str) -> dict | None:
    """Structured detail from a rejected POST /prompt response (ComfyUI returns
    {"error": {...}, "node_errors": {...}} on validation failure)."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
    node_errors = parsed.get("node_errors") if isinstance(parsed.get("node_errors"), dict) else {}
    message = str(error.get("message") or "")
    if not message and not node_errors:
        return None
    parts = []
    for node_id, node_err in list(node_errors.items())[:3]:
        if not isinstance(node_err, dict):
            continue
        node_type = str(node_err.get("class_type") or "")
        first = next(iter(node_err.get("errors") or []), None)
        first_msg = str(first.get("message") or "") if isinstance(first, dict) else ""
        parts.append(f"node {node_id} ({node_type}): {first_msg}".strip())
    summary = "; ".join(p for p in parts if p)
    exception_message = "; ".join(p for p in (message, summary) if p)[:1000]
    return {
        "exception_type": str(error.get("type") or "validation"),
        "exception_message": exception_message,
        "error_class": classify_gen_error(str(error.get("type") or ""), exception_message),
    }
