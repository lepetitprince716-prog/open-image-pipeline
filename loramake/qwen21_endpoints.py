"""Resolve the Qwen-Image-2.1 sidecar ComfyUI endpoint on BOXA.

The sidecar (ComfyUI master, ``--port 8189 --use-ck-attention``) listens on BOXA's
loopback only, so every Mac caller goes through the SSH forward
``127.0.0.1:18189 -> BOXA 127.0.0.1:8189`` kept alive by
``scripts/com.loramake.qwen21-boxa-loopback.plist`` (install/verify:
``docs/claude_comfy_loopback.md``).

It is deliberately *not* a key in ``comfy_endpoints.DIRECT_COMFY_HOSTS``: ``box=boxa``
keeps meaning the production ``:8188`` everywhere else in the repo, and the engine
still probes ``:8188`` for the shared-GPU idle policy (same pattern as Forge, see
``forge_endpoints``). Pure module: no network at import time.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

LOOPBACK_QWEN21_HOSTS = {
    "boxa": "http://127.0.0.1:18189",
}

# Documentation constants — where the loopback forward terminates.
QWEN21_UPSTREAM = {
    "boxa": "127.0.0.1:8189 on BOXA (ComfyUI master sidecar, run_master_ck.ps1)",
}


def resolve_qwen21_host(
    box: str = "boxa",
    *,
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve one box to an explicit adapter or the loopback forward.

    ``explicit`` wins; otherwise ``QWEN21_HOST`` from ``env`` (default ``os.environ``);
    otherwise the loopback forward. Unknown boxes raise ``KeyError`` like the router.
    """
    if box not in LOOPBACK_QWEN21_HOSTS:
        raise KeyError(box)
    values = os.environ if env is None else env
    chosen = (explicit or "").strip() or (values.get("QWEN21_HOST") or "").strip()
    if chosen:
        return chosen
    return LOOPBACK_QWEN21_HOSTS[box]
