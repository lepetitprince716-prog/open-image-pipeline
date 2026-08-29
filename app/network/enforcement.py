"""Gate helpers: turn persisted network intent into submit/launch decisions.

Consumed through late-bound hooks (brief.NETWORK_PROBLEMS, jobs.NETWORK_GATE)
that app/server.py wires at import time; the hooks default to None so CLI and
pytest imports of those modules stay gate-free.
"""
from __future__ import annotations

from . import state
from .registry import SERVICES

_PROVIDER_SERVICE = {spec.provider_id: spec.id for spec in SERVICES if spec.provider_id}
_BOX_IDS = tuple(spec.id for spec in SERVICES if spec.kind == "box")


def provider_service_id(provider: str) -> str | None:
    """Service id gating this brief provider, or None when unmanaged (local, gpt_image)."""
    return _PROVIDER_SERVICE.get(provider)


def service_enabled(service_id: str) -> bool:
    return state.enabled(service_id)


def enabled_boxes() -> list[str]:
    return [box for box in _BOX_IDS if state.enabled(box)]


def brief_problems(brief: dict) -> list[str]:
    """Submit-time problems from the network kill switches (empty = pass)."""
    problems: list[str] = []
    provider = str(brief.get("provider") or "local")
    service_id = _PROVIDER_SERVICE.get(provider)
    if service_id is not None and not state.enabled(service_id):
        problems.append(f"provider '{provider}' 已在网络页停用")
        return problems
    if provider == "local":
        box = str(brief.get("box") or "auto")
        if box in _BOX_IDS and not state.enabled(box):
            problems.append(f"box '{box}' 已在网络页停用")
        elif box == "auto" and not enabled_boxes():
            problems.append("全部 box 已在网络页停用")
    return problems


def launch_gate(brief: dict) -> tuple[str | None, str | None, str | None]:
    """Launch-time backstop for jobs queued before a toggle flipped.

    Returns (error_class, error_message, forced_box). forced_box is set when
    box=auto has exactly one enabled box, so the worker uses it directly and
    never probes the disabled one.
    """
    provider = str(brief.get("provider") or "local")
    service_id = _PROVIDER_SERVICE.get(provider)
    if service_id is not None and not state.enabled(service_id):
        return "provider_disabled", f"provider '{provider}' 已在网络页停用", None
    if provider != "local":
        return None, None, None
    box = str(brief.get("box") or "auto")
    if box in _BOX_IDS and not state.enabled(box):
        return "box_disabled", f"box '{box}' 已在网络页停用", None
    if box == "auto":
        boxes = enabled_boxes()
        if not boxes:
            return "box_disabled", "全部 box 已在网络页停用", None
        if len(boxes) == 1:
            return None, None, boxes[0]
    return None, None, None
