"""Resolve the Forge (Stable Diffusion WebUI Forge) endpoint on Z3.

Unlike ComfyUI (see ``comfy_endpoints``), Forge has no reachable LAN-direct
host: the Windows firewall on Z3 blocks inbound connections to port 7862, so
every caller — plain terminal and managed Claude alike — goes through the
loopback SSH forward ``127.0.0.1:17862 -> Z3 127.0.0.1:7862``.  The launchd
adapter that keeps that forward alive is
``scripts/com.loramake.forge-z3-loopback.plist`` (install/verify:
``docs/claude_comfy_loopback.md``).

This module is deliberately pure: endpoint selection never probes the network
or starts a tunnel at import time.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

# The only transport that works from the Mac; see module docstring.
LOOPBACK_FORGE_HOSTS = {
    "z3": "http://127.0.0.1:17862",
}

# Documentation constants — where the loopback forward terminates.
FORGE_UPSTREAM = {
    "z3": "127.0.0.1:7862 on Z3 (D:\\Forge\\forge, launch.py --nowebui --xformers)",
}


def resolve_forge_host(
    box: str = "z3",
    *,
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve one box to an explicit adapter or the loopback forward.

    ``explicit`` is highest priority and carries the ``FORGE_HOST`` adapter
    contract (a proven SSH/Cloudflare/local adapter endpoint).  When it is
    unset, the env mapping (default ``os.environ``) is consulted for
    ``FORGE_HOST`` so direct callers behave like router-launched ones.
    """
    if box not in LOOPBACK_FORGE_HOSTS:
        # Preserve the router's historical illegal-box failure surface.
        raise KeyError(box)
    values = os.environ if env is None else env
    chosen = (explicit or "").strip() or (values.get("FORGE_HOST") or "").strip()
    if chosen:
        return chosen
    return LOOPBACK_FORGE_HOSTS[box]
