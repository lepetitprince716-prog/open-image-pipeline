"""Resolution matrix for the Forge endpoint seam."""
from __future__ import annotations

import pytest

from loramake.forge_endpoints import LOOPBACK_FORGE_HOSTS, resolve_forge_host


def test_default_is_the_loopback_forward():
    assert resolve_forge_host("z3", env={}) == LOOPBACK_FORGE_HOSTS["z3"] == "http://127.0.0.1:17862"


def test_explicit_adapter_wins():
    assert resolve_forge_host("z3", explicit="http://127.0.0.1:9999", env={}) == "http://127.0.0.1:9999"


def test_env_forge_host_is_honored():
    assert resolve_forge_host("z3", env={"FORGE_HOST": "http://127.0.0.1:8888"}) == "http://127.0.0.1:8888"


def test_blank_explicit_falls_through_to_env_then_loopback():
    assert resolve_forge_host("z3", explicit="  ", env={"FORGE_HOST": "http://127.0.0.1:7777"}) \
        == "http://127.0.0.1:7777"
    assert resolve_forge_host("z3", explicit="", env={"FORGE_HOST": " "}) == LOOPBACK_FORGE_HOSTS["z3"]


def test_unknown_box_preserves_keyerror_surface():
    with pytest.raises(KeyError):
        resolve_forge_host("zzz", env={})
