"""Fail fast when CI tests try to reach a non-loopback network endpoint.

This module is loaded automatically by Python when its directory is placed on
``PYTHONPATH``.  It is intentionally inert unless
``LORAMAKE_CI_BLOCK_NETWORK=1`` is set, so normal production and developer
runs are unchanged.
"""

from __future__ import annotations

import errno
import ipaddress
import os
import socket
from typing import Any


_ENABLED = os.environ.get("LORAMAKE_CI_BLOCK_NETWORK") == "1"


def _is_loopback(host: Any) -> bool:
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False

    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    # IPv6 scope IDs are irrelevant to the address classification.
    normalized = normalized.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _guard_address(family: int, address: Any) -> None:
    if family == socket.AF_UNIX:
        return
    if family not in {socket.AF_INET, socket.AF_INET6}:
        raise OSError(
            errno.ENETUNREACH,
            f"CI network guard blocked unsupported socket family {family}",
        )
    if not isinstance(address, tuple) or not address:
        raise OSError(errno.ENETUNREACH, f"CI network guard blocked malformed address {address!r}")
    host = address[0]
    if _is_loopback(host):
        return
    port = address[1] if len(address) > 1 else "?"
    raise OSError(
        errno.ENETUNREACH,
        f"CI network guard blocked non-loopback connection to {host!r}:{port}",
    )


if _ENABLED and not getattr(socket, "_loramake_ci_network_guard", False):
    _original_connect = socket.socket.connect
    _original_connect_ex = socket.socket.connect_ex
    _original_create_connection = socket.create_connection

    def _guarded_connect(self: socket.socket, address: Any) -> Any:
        _guard_address(self.family, address)
        return _original_connect(self, address)

    def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
        _guard_address(self.family, address)
        return _original_connect_ex(self, address)

    def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        family = socket.AF_INET6 if isinstance(address, tuple) and len(address) > 2 else socket.AF_INET
        _guard_address(family, address)
        return _original_create_connection(address, *args, **kwargs)

    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.create_connection = _guarded_create_connection
    socket._loramake_ci_network_guard = True  # type: ignore[attr-defined]
