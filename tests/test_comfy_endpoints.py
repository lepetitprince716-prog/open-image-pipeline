"""Pure endpoint-routing tests for normal and managed Claude callers."""
from __future__ import annotations

import pytest

from loramake.comfy_endpoints import (
    CLAUDE_LOOPBACK_COMFY_HOSTS,
    DIRECT_COMFY_HOSTS,
    is_managed_claude,
    resolve_comfy_host,
)


def test_normal_callers_keep_direct_lan_endpoints() -> None:
    assert resolve_comfy_host("z3", env={}) == DIRECT_COMFY_HOSTS["z3"]
    assert resolve_comfy_host("zzz", env={}) == DIRECT_COMFY_HOSTS["zzz"]


def test_managed_claude_uses_loopback_adapters() -> None:
    env = {
        "CLAUDE_CODE_PROCESS_WRAPPER":
            "/usr/local/libexec/claude-network-guard/cc-process-launcher",
    }
    assert is_managed_claude(env) is True
    assert resolve_comfy_host("z3", env=env) == CLAUDE_LOOPBACK_COMFY_HOSTS["z3"]
    assert resolve_comfy_host("zzz", env=env) == CLAUDE_LOOPBACK_COMFY_HOSTS["zzz"]


def test_claudecode_marker_also_selects_loopback() -> None:
    assert resolve_comfy_host("zzz", env={"CLAUDECODE": "1"}) == \
        CLAUDE_LOOPBACK_COMFY_HOSTS["zzz"]


def test_explicit_adapter_wins_for_selected_box() -> None:
    adapter = "http://127.0.0.1:29999"
    env = {"CLAUDECODE": "1"}
    assert resolve_comfy_host("zzz", explicit=f"  {adapter}  ", env=env) == adapter


def test_unknown_box_fails_closed() -> None:
    with pytest.raises(KeyError):
        resolve_comfy_host("other", env={})
