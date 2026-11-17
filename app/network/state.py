"""Persisted per-service network enable state.

Atomic JSON under app/data/ (same tmp+replace idiom as app/lora_archive.py).
JSON over sqlite on purpose: the meta cache invalidates off this file's mtime,
which changes exactly when the state changes — a sqlite file's mtime is noisy
with WAL checkpoints.

An absent file or absent service entry means ENABLED: the feature is a kill
switch, so missing state must reproduce today's behavior.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .registry import SERVICES

PATH = Path(__file__).resolve().parent.parent / "data" / "network_state.json"
_LOCK = threading.Lock()


def _load() -> dict:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    services = data.get("services")
    return services if isinstance(services, dict) else {}


def enabled(service_id: str) -> bool:
    entry = _load().get(service_id)
    return not (isinstance(entry, dict) and entry.get("enabled") is False)


def all_enabled() -> dict[str, bool]:
    services = _load()
    out: dict[str, bool] = {}
    for spec in SERVICES:
        entry = services.get(spec.id)
        out[spec.id] = not (isinstance(entry, dict) and entry.get("enabled") is False)
    return out


def set_enabled(service_id: str, on: bool) -> bool:
    with _LOCK:
        try:
            doc = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        if not isinstance(doc, dict):
            doc = {}
        services = doc.get("services")
        if not isinstance(services, dict):
            services = {}
        services[service_id] = {"enabled": bool(on), "changed_at": time.time()}
        doc.update(version=1, services=services)
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PATH)
    return bool(on)
