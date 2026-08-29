"""Resolve ComfyUI endpoints without weakening Claude's managed network sandbox.

Normal terminal/app callers use the canonical LAN endpoints.  Managed Claude
callers use loopback-only SSH forwards, because Seatbelt blocks raw LAN sockets
and the managed HTTP proxy is not a reliable data plane for multi-megabyte
multipart image uploads.

This module is deliberately pure: endpoint selection never probes the network
or starts a tunnel at import time.  The launchd adapters are documented in
``docs/claude_comfy_loopback.md``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping


DIRECT_COMFY_HOSTS = {
    "z3": "http://192.0.2.10:8188",
    "zzz": "http://192.0.2.20:8388",
}

CLAUDE_LOOPBACK_COMFY_HOSTS = {
    "z3": "http://127.0.0.1:18188",
    "zzz": "http://127.0.0.1:18388",
}


def is_managed_claude(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the caller is running under Claude's managed launcher."""
    values = os.environ if env is None else env
    wrapper = values.get("CLAUDE_CODE_PROCESS_WRAPPER", "").strip()
    if "claude-network-guard" in wrapper:
        return True
    return values.get("CLAUDECODE", "").strip().lower() in {"1", "true", "yes"}


def resolve_comfy_host(
    box: str,
    *,
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve one box to an explicit adapter, managed loopback, or direct LAN.

    ``explicit`` is highest priority and preserves the established ``Z3_HOST``
    adapter contract.  ``box`` still owns model/detector quirks; it selects only
    the transport default here.
    """
    if box not in DIRECT_COMFY_HOSTS:
        # Preserve the router's historical illegal-box failure surface.
        raise KeyError(box)
    chosen = (explicit or "").strip()
    if chosen:
        return chosen
    if is_managed_claude(env):
        return CLAUDE_LOOPBACK_COMFY_HOSTS[box]
    return DIRECT_COMFY_HOSTS[box]
