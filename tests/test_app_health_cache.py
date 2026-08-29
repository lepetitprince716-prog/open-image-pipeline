"""The liveness endpoint never waits for remote GPU boxes."""
from __future__ import annotations

import statistics
import time

from app.box_status import BoxStatusCache


def test_snapshot_p95_stays_below_50ms_while_both_probes_timeout():
    calls = []

    def slow(host: str):
        calls.append(host)
        time.sleep(0.15)
        raise TimeoutError(host)

    cache = BoxStatusCache({"z3": "http://z3", "zzz": "http://zzz"}, slow,
                           refresh_after_s=0, stale_after_s=0.1)
    durations = []
    for _ in range(40):
        started = time.perf_counter()
        body = cache.snapshot()
        durations.append((time.perf_counter() - started) * 1000)
        assert set(body) == {"z3", "zzz"}
        assert all(item["state"] == "unknown" for item in body.values())
    p95 = statistics.quantiles(durations, n=20)[18]
    assert p95 < 50.0
    # Repeated HTTP-style reads coalesce behind a single refresh thread.
    assert len(calls) <= 2


def test_cached_observation_reports_age_staleness_and_single_flight():
    calls = []
    cache = BoxStatusCache(
        {"zzz": "http://zzz"},
        lambda host: calls.append(host) or (True, "idle (VRAM 90% free)"),
        refresh_after_s=60,
        stale_after_s=60,
    )
    cache.refresh_now()
    body = cache.snapshot(trigger_refresh=False)["zzz"]
    assert body["idle"] is True and body["state"] == "idle"
    assert body["observed_at"].endswith("Z")
    assert body["age"] >= 0 and body["stale"] is False
    assert body["refresh_error"] is None
    assert calls == ["http://zzz"]


def test_refresh_probes_boxes_in_parallel():
    def slow(host: str):
        time.sleep(0.2)
        return True, "idle (VRAM 90% free)"

    cache = BoxStatusCache({"z3": "http://z3", "zzz": "http://zzz"}, slow,
                           refresh_after_s=60, stale_after_s=60)
    started = time.perf_counter()
    cache.refresh_now()
    elapsed = time.perf_counter() - started
    # Sequential probing would take ~0.4s; parallel lands near one probe's 0.2s.
    assert elapsed < 0.35
    body = cache.snapshot(trigger_refresh=False)
    assert all(item["state"] == "idle" for item in body.values())
