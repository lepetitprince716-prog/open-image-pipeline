#!/usr/bin/env python3
"""One call to see every ComfyUI channel Claude depends on.

Reports, per box, whether the resolved endpoint answers ``/system_stats``, its round-trip
time, and a guess whether the loopback forward is currently riding the LAN or the
Cloudflare tunnel (a plain RTT threshold: the tunnel costs ~1 s per request, the LAN
forward tens of ms). Also reports the selection-bridge file the browser extension keeps
in userdata, and with ``--ping`` round-trips the page itself.

Replaces the "open five shells and probe each hop by hand" loop. Pure read-only; the
endpoint always comes from ``loramake.comfy_endpoints.resolve_comfy_host``.

    python3 scripts/comfy_net_status.py [--json] [--ping] [--box boxa|boxb]
    python3 scripts/comfy_net_status.py --bench [--box boxa] [--json] [--save]

``--bench`` is the transport measurement port: RTT distribution, /view throughput for a
real output, the same file through ComfyUI's webp preview transcode, and the /ws
handshake time. Read-only — it never queues a prompt or touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake.comfy_endpoints import resolve_comfy_host  # noqa: E402
# The two websocket helpers are private to loramake.comfy because no other caller needed
# them; the bench measures the very handshake they perform, so it reuses them rather than
# rebuilding the uri and the Cloudflare Access headers a second way.
from loramake.comfy import _ws_connect, _ws_target_uri_and_headers  # noqa: E402

BOXES = ("boxa", "boxb")
PROBE_TIMEOUT_S = 8.0
LAN_RTT_MAX_MS = 200.0   # a forward riding the LAN answers well under this
TUNNEL_RTT_MIN_MS = 500.0  # cloudflared adds roughly a second per hop
BRIDGE_FILE = "/api/userdata/loramake%2Fselection.json"
BENCH_SCHEMA = "loramake.comfy-bench/v1"
BENCH_SAMPLES = 10
BENCH_FETCH_TIMEOUT_S = 120.0  # a 6 MB PNG over the Cloudflare tunnel takes ~30 s
BENCH_FETCH_LIMIT = 64 << 20   # _get caps at 1 MiB; a truncated read would flatter the link
BENCH_PREVIEW = "webp;85"
BENCH_WS_TIMEOUT_S = 10.0

_urlopen = urllib.request.urlopen  # patchable seam


def _get(url: str) -> tuple[int | None, bytes, float, str | None]:
    started = time.perf_counter()
    try:
        with _urlopen(url, timeout=PROBE_TIMEOUT_S) as resp:
            raw = resp.read(1 << 20)
            return resp.status, raw, (time.perf_counter() - started) * 1000, None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), (time.perf_counter() - started) * 1000, None
    except (urllib.error.URLError, OSError) as exc:
        return None, b"", (time.perf_counter() - started) * 1000, str(exc)


def classify_path(rtt_ms: float | None) -> str:
    if rtt_ms is None:
        return "down"
    if rtt_ms <= LAN_RTT_MAX_MS:
        return "lan"
    if rtt_ms >= TUNNEL_RTT_MIN_MS:
        return "tunnel"
    return "unclear"


def host_for(box: str) -> str:
    explicit = os.environ.get("BOXA_HOST") if box == "boxa" else None
    return resolve_comfy_host(box, explicit=explicit)


def probe_box(box: str) -> dict:
    host = host_for(box)
    code, raw, rtt, err = _get(f"{host}/system_stats")
    ok = code == 200
    version = None
    if ok:
        try:
            version = json.loads(raw).get("system", {}).get("comfyui_version")
        except (ValueError, AttributeError):
            version = None
    return {
        "box": box,
        "host": host,
        "ok": ok,
        "http": code,
        "rtt_ms": round(rtt, 1),
        "path": classify_path(rtt if ok else None),
        "comfyui_version": version,
        "error": err,
    }


def probe_bridge(box: str, ping: bool) -> dict:
    host = host_for(box)
    code, raw, rtt, err = _get(f"{host}{BRIDGE_FILE}")
    out: dict = {"box": box, "selection_file": code == 200, "http": code, "rtt_ms": round(rtt, 1), "error": err}
    if code == 200:
        try:
            doc = json.loads(raw)
            out["age_s"] = round(time.time() - float(doc.get("ts", 0)), 1)
            out["node_count"] = doc.get("node_count")
            out["tab"] = doc.get("tab")
        except (ValueError, AttributeError, TypeError):
            out["error"] = "selection.json is not the expected JSON"
    if ping:
        from scripts.comfy_bridge_mcp import BridgeClient, BridgeError

        started = time.perf_counter()
        try:
            out["ping"] = BridgeClient(host).send("ping")
            out["ping_ok"] = True
        except BridgeError as exc:
            out["ping_ok"] = False
            out["ping_error"] = str(exc)
        out["ping_rtt_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _unique_fetch_path(directory: Path, filename: str, *, preview: bool = False,
                       now_ns: int | None = None) -> Path:
    """A dest that two images sharing a stem, or two bench runs, cannot clobber.

    Reserves the path with O_EXCL so a concurrent process that picks the same stamp
    lands on the next counter suffix instead of truncating this one.
    """
    src = Path(Path(str(filename)).name)
    suffix = ".webp" if preview else (src.suffix or ".bin")
    stamp = str(time.time_ns() if now_ns is None else now_ns)
    directory.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        extra = "" if n == 0 else f"-{n + 1}"
        dest = directory / f"{src.stem}-{stamp}{extra}{suffix}"
        try:
            fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return dest


def _timed_fetch(url: str, dest: Path, limit: int = BENCH_FETCH_LIMIT) -> tuple[int | None, int, float, str | None]:
    """Fetch a URL whole into dest and time it. Unlike _get this does not cap at 1 MiB — a
    truncated read would understate the transfer time that the bench exists to measure.

    dest is filled on a successful read. A failed fetch unlinks that dest so cleanup
    does not leave an empty transcode file behind.
    """
    started = time.perf_counter()
    try:
        with _urlopen(url, timeout=BENCH_FETCH_TIMEOUT_S) as resp:
            raw = resp.read(limit)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            return resp.status, len(raw), (time.perf_counter() - started) * 1000, None
    except urllib.error.HTTPError as exc:
        _unlink_quiet(dest)
        return exc.code, 0, (time.perf_counter() - started) * 1000, f"HTTP {exc.code}: {exc.reason}"
    except (urllib.error.URLError, OSError) as exc:
        _unlink_quiet(dest)
        return None, 0, (time.perf_counter() - started) * 1000, str(exc)


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. Manual rather than statistics.quantiles because the bench
    runs with n as small as 1, where quantiles() raises."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return round(ordered[rank - 1], 1)


def _mbps(nbytes: int, ms: float) -> float | None:
    if nbytes <= 0 or ms <= 0:
        return None
    return round(nbytes / (1 << 20) / (ms / 1000), 2)


def classify_transport(rtt_ms: float | None) -> str:
    """Name the link the measured RTT implies. Tailscale is a direct hop, so it lands in the
    same band as the LAN; the two are distinguished by the host, not by latency."""
    if rtt_ms is None:
        return "down"
    if rtt_ms <= LAN_RTT_MAX_MS:
        return "direct"
    if rtt_ms >= TUNNEL_RTT_MIN_MS:
        return "tunnel"
    return "unclear"


def discover_output(host: str) -> dict | None:
    """Newest /history entry that carries a fetchable image, as {filename, subfolder, type}.

    Read-only: /history never mutates the queue. Returns None when the box has no history
    (a freshly restarted service), which the caller reports rather than treating as failure.
    """
    code, raw, _, _ = _get(f"{host}/history?max_items=20")
    if code != 200:
        return None
    try:
        history = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(history, dict):
        return None
    for entry in reversed(list(history.values())):
        outputs = (entry or {}).get("outputs")
        if not isinstance(outputs, dict):
            continue
        for node in outputs.values():
            for image in (node or {}).get("images") or []:
                if image.get("filename"):
                    return {
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    }
    return None


def view_url(host: str, ref: dict, preview: str | None = None) -> str:
    query = {"filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
             "type": ref.get("type", "output")}
    if preview:
        query["preview"] = preview
    return f"{host}/view?{urlencode(query)}"


def bench_ws(host: str) -> dict:
    """Time the /ws handshake. A3 rests on this number: the driver's open timeout must
    exceed it, or the websocket fast path silently degrades to pure polling."""
    connect = _ws_connect()
    if connect is None:
        return {"available": False, "reason": "websockets not installed"}
    uri, headers = _ws_target_uri_and_headers(host)
    if uri is None:
        return {"available": False, "reason": f"cannot derive a ws uri from {host}"}
    started = time.perf_counter()
    try:
        websocket = connect(uri, open_timeout=BENCH_WS_TIMEOUT_S, close_timeout=0.2,
                            additional_headers=headers or None)
    except Exception as exc:  # any failure is a datum, not a crash
        return {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "handshake_ms": round((time.perf_counter() - started) * 1000, 1)}
    handshake_ms = round((time.perf_counter() - started) * 1000, 1)
    try:
        websocket.close()
    except Exception:
        pass
    return {"available": True, "ok": True, "handshake_ms": handshake_ms}


def bench_box(box: str, samples: int = BENCH_SAMPLES, image: dict | None = None,
              scratch: Path | None = None, keep: bool = False) -> dict:
    """Five measurements on one box: RTT distribution, PNG throughput, webp-preview
    throughput, websocket handshake, and the transport those numbers imply.

    Read-only throughout: /system_stats, /history and /view never touch the queue or the GPU.
    Downloaded bytes land under ``scratch`` with unique names and are removed after timing
    unless ``keep`` is set (``--save``).
    """
    host = host_for(box)
    rtts: list[float] = []
    errors: list[str] = []
    status: int | None = None
    version = None
    for _ in range(max(1, samples)):
        code, raw, rtt, err = _get(f"{host}/system_stats")
        status = code
        if code == 200:
            rtts.append(rtt)
            if version is None:
                try:
                    version = json.loads(raw).get("system", {}).get("comfyui_version")
                except (ValueError, AttributeError):
                    version = None
        elif err:
            errors.append(err)

    p50 = _percentile(rtts, 0.50)
    out: dict = {
        "box": box,
        "host": host,
        "ok": bool(rtts),
        "http": status,
        "comfyui_version": version,
        "samples": len(rtts),
        "rtt_p50_ms": p50,
        "rtt_p95_ms": _percentile(rtts, 0.95),
        "transport": classify_transport(p50),
        "errors": sorted(set(errors))[:3],
    }
    if not rtts:
        return out

    ref = image or discover_output(host)
    if ref is None:
        out["view"] = {"skipped": "no /history entry carries an image"}
        out["ws"] = bench_ws(host)
        return out

    out["image"] = ref
    owned_scratch = scratch is None
    if scratch is None:
        scratch = Path(tempfile.mkdtemp(prefix="comfy_bench_"))
    written: list[Path] = []
    try:
        png_dest = _unique_fetch_path(scratch, ref["filename"])
        written.append(png_dest)
        png_code, png_bytes, png_ms, png_err = _timed_fetch(view_url(host, ref), png_dest)
        out["view"] = {"http": png_code, "bytes": png_bytes, "ms": round(png_ms, 1),
                       "mbps": _mbps(png_bytes, png_ms), "error": png_err}
        pv_dest = _unique_fetch_path(scratch, ref["filename"], preview=True)
        written.append(pv_dest)
        pv_code, pv_bytes, pv_ms, pv_err = _timed_fetch(view_url(host, ref, BENCH_PREVIEW), pv_dest)
        out["view_preview"] = {"preview": BENCH_PREVIEW, "http": pv_code, "bytes": pv_bytes,
                               "ms": round(pv_ms, 1), "mbps": _mbps(pv_bytes, pv_ms), "error": pv_err}
        if png_bytes > 0 and pv_bytes > 0:
            out["view_preview"]["shrink_ratio"] = round(png_bytes / pv_bytes, 1)
        out["ws"] = bench_ws(host)
        if keep and owned_scratch:
            # Caller kept the bytes but gave no dir: surface where they landed.
            out["artifacts_dir"] = str(scratch)
        return out
    finally:
        if not keep:
            for path in written:
                _unlink_quiet(path)
            if owned_scratch:
                shutil.rmtree(scratch, ignore_errors=True)


def bench(boxes: tuple[str, ...] = BOXES, samples: int = BENCH_SAMPLES,
          image: dict | None = None, scratch: Path | None = None, keep: bool = False) -> dict:
    return {
        "schema": BENCH_SCHEMA,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "boxes": [bench_box(box, samples=samples, image=image, scratch=scratch, keep=keep)
                  for box in boxes],
    }


def _render_transfer(kind: str, info: dict) -> str:
    """One view/preview line. A failed fetch must not print ``None MB/s``."""
    label = f"     {kind:<9}"
    mbps = info.get("mbps")
    err = info.get("error")
    if err or mbps is None:
        reason = err or (
            f"HTTP {info['http']}" if info.get("http") not in (None, 200) else "no bytes transferred"
        )
        return f"{label}FAILED: {reason}"
    extra = f"   {info['shrink_ratio']}x smaller" if info.get("shrink_ratio") else ""
    return f"{label}{info['bytes']} B in {info['ms']} ms   {info['mbps']} MB/s{extra}"


def render_bench(report: dict) -> str:
    lines = [f"{report['schema']}  {report['measured_at']}"]
    for row in report["boxes"]:
        if not row["ok"]:
            detail = row["errors"][0] if row["errors"] else f"HTTP {row['http']}"
            lines.append(f"{row['box']:<4} DOWN  {row['host']}  {detail}")
            continue
        lines.append(f"{row['box']:<4} {row['host']}  comfyui {row['comfyui_version'] or '?'}")
        lines.append(f"     rtt      p50 {row['rtt_p50_ms']} ms   p95 {row['rtt_p95_ms']} ms"
                     f"   n={row['samples']}   transport={row['transport']}")
        view = row.get("view") or {}
        if "skipped" in view:
            lines.append(f"     view     skipped: {view['skipped']}")
        else:
            lines.append(_render_transfer("view", view))
            lines.append(_render_transfer("preview", row.get("view_preview") or {}))
        ws = row.get("ws") or {}
        if not ws.get("available"):
            lines.append(f"     ws       unavailable: {ws.get('reason')}")
        elif ws.get("ok"):
            lines.append(f"     ws       handshake {ws['handshake_ms']} ms")
        else:
            lines.append(f"     ws       FAILED after {ws.get('handshake_ms')} ms: {ws.get('error')}")
    return "\n".join(lines)


def bench_save_path(report: dict, root: Path) -> Path:
    stamp = report["measured_at"].replace(":", "").replace("-", "")
    day = stamp.split("T")[0]
    return root / "outputs" / "_analysis" / f"comfy_bench_{day}" / f"bench-{stamp}.json"


def collect(boxes: tuple[str, ...] = BOXES, ping: bool = False) -> dict:
    report = {"boxes": [probe_box(box) for box in boxes]}
    if "boxa" in boxes:
        report["bridge"] = probe_bridge("boxa", ping)
    return report


def render(report: dict) -> str:
    lines = []
    for row in report["boxes"]:
        status = "OK " if row["ok"] else "DOWN"
        detail = row["error"] or f"HTTP {row['http']}"
        lines.append(f"{row['box']:<4} {status} {row['rtt_ms']:>8.1f} ms  path={row['path']:<8} {row['host']}  {detail}")
    bridge = report.get("bridge")
    if bridge:
        if bridge["selection_file"]:
            lines.append(f"bridge selection.json: age {bridge['age_s']} s, nodes {bridge['node_count']}, tab {bridge['tab']}")
        else:
            lines.append(f"bridge selection.json: absent (HTTP {bridge['http']}; {bridge['error'] or 'page never published'})")
        if "ping_ok" in bridge:
            verdict = "page online" if bridge["ping_ok"] else f"page offline: {bridge.get('ping_error')}"
            lines.append(f"bridge ping: {verdict} ({bridge['ping_rtt_ms']} ms)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--ping", action="store_true", help="also round-trip the bridge page")
    parser.add_argument("--box", choices=BOXES, action="append", help="limit to a box (repeatable)")
    parser.add_argument("--bench", action="store_true",
                        help="measure the transport instead of reporting status (read-only)")
    parser.add_argument("--bench-samples", type=int, default=BENCH_SAMPLES,
                        help=f"--bench: /system_stats samples per box (default {BENCH_SAMPLES})")
    parser.add_argument("--bench-image", metavar="FILENAME",
                        help="--bench: fetch this output instead of the newest one in /history")
    parser.add_argument("--save", action="store_true",
                        help="--bench: also write the JSON under outputs/_analysis/comfy_bench_<date>/")
    args = parser.parse_args(argv)
    boxes = tuple(args.box) if args.box else BOXES
    if args.bench:
        if args.ping:
            print("note: --ping is skipped in --bench mode", file=sys.stderr)
        image = {"filename": args.bench_image, "subfolder": "", "type": "output"} if args.bench_image else None
        scratch = Path(tempfile.mkdtemp(prefix="comfy_bench_")) if args.save else None
        try:
            report = bench(boxes, samples=args.bench_samples, image=image,
                           scratch=scratch, keep=args.save)
            if args.save:
                path = bench_save_path(report, REPO)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                if scratch is not None:
                    for child in list(scratch.iterdir()):
                        shutil.move(str(child), str(path.parent / child.name))
            print(json.dumps(report, ensure_ascii=False, indent=1) if args.json else render_bench(report))
            if args.save and not args.json:
                print(f"[saved] {path}")
            return 0 if all(row["ok"] for row in report["boxes"]) else 1
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
    report = collect(boxes, ping=args.ping)
    print(json.dumps(report, ensure_ascii=False, indent=1) if args.json else render(report))
    return 0 if all(row["ok"] for row in report["boxes"]) else 1


if __name__ == "__main__":
    sys.exit(main())
