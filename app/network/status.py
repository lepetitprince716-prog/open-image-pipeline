"""Non-blocking status cache for non-box services plus the dashboard composition.

ProbeCache generalizes app/box_status.py's BoxStatusCache with two upgrades the
provider probes need: per-service refresh/stale intervals (third-party probes
must stay rare) and per-service single-flight (a slow NovelAI probe must not
block the LoRA Manager one). BoxStatusCache itself is untouched — /api/health
depends on its exact shape.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from ..box_status import _iso  # same package family; one ISO format everywhere
from .registry import SERVICES

Probe = Callable[[], tuple[str, str, dict]]

# (refresh_after_s, stale_after_s) per service; providers hit third parties and
# must stay rare. Anything unlisted gets DEFAULT_INTERVAL.
INTERVALS: dict[str, tuple[float, float]] = {
    "lora_manager": (15.0, 60.0),
    "civitai": (120.0, 600.0),
    "novelai": (300.0, 900.0),
}
DEFAULT_INTERVAL: tuple[float, float] = (120.0, 600.0)


class ProbeCache:
    def __init__(self, probes: dict[str, Probe], *,
                 intervals: dict[str, tuple[float, float]] | None = None):
        self.probes = probes
        self.intervals = dict(INTERVALS if intervals is None else intervals)
        self._lock = threading.Lock()
        self._refreshing: set[str] = set()
        self._values: dict[str, dict] = {}

    def _interval(self, service_id: str) -> tuple[float, float]:
        return self.intervals.get(service_id, DEFAULT_INTERVAL)

    def refresh_async(self, service_id: str) -> bool:
        """Start at most one background refresh per service. True = started here."""
        if service_id not in self.probes:
            return False
        with self._lock:
            if service_id in self._refreshing:
                return False
            self._refreshing.add(service_id)
        threading.Thread(target=self._refresh_one, args=(service_id,),
                         daemon=True, name=f"net-probe-{service_id}").start()
        return True

    def _refresh_one(self, service_id: str) -> None:
        started = time.monotonic()
        try:
            state, detail, extra = self.probes[service_id]()
            value = {"state": str(state), "detail": str(detail),
                     "extra": dict(extra or {}),
                     "observed_epoch": time.time(),
                     "latency_ms": round((time.monotonic() - started) * 1000, 3),
                     "refresh_error": None}
        except Exception as exc:
            with self._lock:
                previous = dict(self._values.get(service_id) or {})
            if previous:
                value = previous
                value["refresh_error"] = f"{type(exc).__name__}: {exc}"[:240]
            else:
                value = {"state": "unknown", "detail": "unknown", "extra": {},
                         "observed_epoch": time.time(),
                         "latency_ms": round((time.monotonic() - started) * 1000, 3),
                         "refresh_error": f"{type(exc).__name__}: {exc}"[:240]}
        with self._lock:
            self._values[service_id] = value
            self._refreshing.discard(service_id)

    def snapshot(self, *, trigger_refresh: bool = True) -> dict[str, dict]:
        now = time.time()
        with self._lock:
            values = {sid: dict(self._values.get(sid) or {}) for sid in self.probes}
            refreshing = set(self._refreshing)
        out: dict[str, dict] = {}
        for sid, value in values.items():
            refresh_after, stale_after = self._interval(sid)
            observed = value.get("observed_epoch")
            age = max(0.0, now - observed) if isinstance(observed, (int, float)) else None
            out[sid] = {
                "state": value.get("state") or "unknown",
                "detail": value.get("detail") or "unknown",
                "observed_at": _iso(observed) if isinstance(observed, (int, float)) else None,
                "age": round(age, 3) if age is not None else None,
                "stale": age is None or age > stale_after,
                "refreshing": sid in refreshing,
                "latency_ms": value.get("latency_ms"),
                "refresh_error": value.get("refresh_error"),
                "extra": dict(value.get("extra") or {}),
            }
            if trigger_refresh and (age is None or age >= refresh_after):
                self.refresh_async(sid)
        return out


_BOX_STATE_MAP = {"idle": "ok", "busy": "busy"}


def compose_status(*, box_snapshot: dict[str, dict], probe_snapshot: dict[str, dict],
                   enabled_map: dict[str, bool], lm_url: str, lm_op: dict) -> dict:
    """Merge box/probe caches, persisted intent, and the LM op readback into the
    dashboard contract. `enabled` is intent (drives the switch); `state` is
    observation (drives the dot); a disabled service is overlaid here so the
    underlying caches and /api/health keep their exact shapes."""
    services: list[dict] = []
    for spec in SERVICES:
        if spec.kind == "box":
            src = box_snapshot.get(spec.id) or {}
            card = {
                "state": _BOX_STATE_MAP.get(str(src.get("state")), "unknown"),
                "detail": src.get("detail") or "unknown",
                "observed_at": src.get("observed_at"),
                "age": src.get("age"),
                "stale": bool(src.get("stale", True)),
                "latency_ms": src.get("latency_ms"),
                "refresh_error": src.get("refresh_error"),
                "extra": {},
            }
        else:
            src = probe_snapshot.get(spec.id) or {}
            extra = dict(src.get("extra") or {})
            if spec.id == "lora_manager":
                extra.update(url=lm_url, op=dict(lm_op))
            card = {
                "state": src.get("state") or "unknown",
                "detail": src.get("detail") or "unknown",
                "observed_at": src.get("observed_at"),
                "age": src.get("age"),
                "stale": bool(src.get("stale", True)),
                "latency_ms": src.get("latency_ms"),
                "refresh_error": src.get("refresh_error"),
                "extra": extra,
            }
        enabled = enabled_map.get(spec.id, True)
        if not enabled:
            card.update(state="disabled", detail="已停用")
        card.update(id=spec.id, kind=spec.kind, label=spec.label,
                    enabled=enabled, actions=list(spec.actions))
        services.append(card)
    return {"generated_at": _iso(time.time()), "services": services}
