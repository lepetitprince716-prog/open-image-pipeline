"""Non-blocking, single-flight cache for remote ComfyUI readiness observations."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from collections.abc import Callable, Mapping


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# A "not idle" verdict whose reason is one of these is an observation failure, not a busy box.
UNKNOWN_MARKERS = ("unreachable", "timeout", "timed out", "failed", "invalid", "unknown")


def is_unknown_detail(detail: str) -> bool:
    lowered = str(detail).lower()
    return any(marker in lowered for marker in UNKNOWN_MARKERS)


class BoxStatusCache:
    def __init__(self, hosts: Mapping[str, str] | Callable[[], Mapping[str, str]],
                 probe: Callable[[str], tuple[bool, str]], *,
                 refresh_after_s: float = 5.0, stale_after_s: float = 30.0,
                 should_probe: Callable[[str], bool] | None = None):
        self._hosts = hosts
        self.probe = probe
        self.refresh_after_s = refresh_after_s
        self.stale_after_s = stale_after_s
        # Optional per-box kill switch (network dashboard); None = probe everything.
        # A skipped box keeps its last observation, which goes honestly stale.
        self.should_probe = should_probe
        self._lock = threading.Lock()
        self._refreshing = False
        self._values: dict[str, dict] = {}
        self._boxes: tuple[str, ...] | None = None

    def _host_map(self) -> dict[str, str]:
        mapping = dict(self._hosts() if callable(self._hosts) else self._hosts)
        self._boxes = tuple(mapping)  # URLs move between routes; box names do not
        return mapping

    def _probe_one(self, box: str, host: str) -> None:
        started = time.monotonic()
        try:
            idle, detail = self.probe(host)
            detail = str(detail)
            unknown = not idle and is_unknown_detail(detail)
            value = {"idle": None if unknown else bool(idle), "detail": detail,
                     "state": "unknown" if unknown else ("idle" if idle else "busy"),
                     "observed_epoch": time.time(),
                     "latency_ms": round((time.monotonic() - started) * 1000, 3),
                     "refresh_error": None, "host": host}
        except Exception as exc:
            with self._lock:
                previous = dict(self._values.get(box) or {})
            if previous:
                value = previous
                value["refresh_error"] = f"{type(exc).__name__}: {exc}"[:240]
            else:
                value = {"idle": None, "detail": "unknown", "state": "unknown",
                         "observed_epoch": time.time(),
                         "latency_ms": round((time.monotonic() - started) * 1000, 3),
                         "refresh_error": f"{type(exc).__name__}: {exc}"[:240],
                         "host": host}
        with self._lock:
            self._values[box] = value

    def _refresh(self) -> None:
        # Boxes are probed in parallel: sequentially, two slow probes (up to ~16s
        # each) overshoot the 30s stale window this cache promises.
        try:
            mapping = self._host_map()
            workers = [
                threading.Thread(target=self._probe_one, args=(box, host),
                                 daemon=True, name=f"box-probe-{box}")
                for box, host in mapping.items()
                if self.should_probe is None or self.should_probe(box)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        finally:
            with self._lock:
                self._refreshing = False

    def refresh_async(self) -> bool:
        """Start at most one background refresh. True means this call started it."""
        with self._lock:
            if self._refreshing:
                return False
            self._refreshing = True
        threading.Thread(target=self._refresh, daemon=True, name="box-status-refresh").start()
        return True

    def refresh_now(self) -> None:
        """Synchronous test/maintenance entry point; normal HTTP requests never call this."""
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        self._refresh()

    def snapshot(self, *, trigger_refresh: bool = True) -> dict[str, dict]:
        now = time.time()
        # Host resolution is a route probe (seconds when off-LAN). snapshot runs on
        # the request thread and only needs the box names, so never resolve here
        # once a refresh has: it would both stall the request and hold the lock.
        boxes = self._boxes or tuple(self._host_map())
        with self._lock:
            values = {box: dict(self._values.get(box) or {}) for box in boxes}
            refreshing = self._refreshing
        oldest_age = 0.0
        out = {}
        for box, value in values.items():
            observed = value.get("observed_epoch")
            age = max(0.0, now - observed) if isinstance(observed, (int, float)) else None
            oldest_age = max(oldest_age, age or self.refresh_after_s + 1)
            out[box] = {
                "idle": value.get("idle"),
                "detail": value.get("detail") or "unknown",
                "state": value.get("state") or "unknown",
                "observed_at": _iso(observed) if isinstance(observed, (int, float)) else None,
                "age": round(age, 3) if age is not None else None,
                "stale": age is None or age > self.stale_after_s,
                "refreshing": refreshing,
                "latency_ms": value.get("latency_ms"),
                "refresh_error": value.get("refresh_error"),
                "host": value.get("host"),
            }
        if trigger_refresh and oldest_age >= self.refresh_after_s:
            self.refresh_async()
        return out
