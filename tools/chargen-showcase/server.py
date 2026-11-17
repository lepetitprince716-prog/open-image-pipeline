#!/usr/bin/env python3
"""Standalone offline server for the chargen 3D showcase page.

Stdlib only. Serves:
  /                 -> web/index.html (and other static files under web/)
  /vendor/*         -> ../chargen-viewer/web/vendor/* (vendored three.js build)
  /api/runs         -> JSON list of runs under 3d/chargen/runs with GLB/VRM files
  /api/file?run=&f= -> one model file from a run directory

Port: reads $PORT (portless-compatible), else --port, else 8123.
Run from anywhere: paths are resolved relative to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
WEB_ROOT = HERE / "web"
VENDOR_ROOT = HERE.parent / "chargen-viewer" / "web" / "vendor"
RUNS_ROOT = REPO_ROOT / "3d" / "chargen" / "runs"

MODEL_EXTS = {".glb", ".vrm"}
GATE_REPORT = Path("vrm_gate") / "base_gate_report.json"
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".glb": "model/gltf-binary",
    ".vrm": "application/octet-stream",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _safe_under(root: Path, rel: str) -> Path | None:
    path = (root / unquote(rel).lstrip("/")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path

def _size(p: Path) -> int | None:
    try:
        return p.stat().st_size
    except OSError:
        return None


def _load_json(path: Path) -> dict | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _run_reports(run_dir: Path) -> dict:
    """Summarize the JSON reports a run directory carries (read-only)."""
    out: dict = {}
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest:
        params = manifest.get("parameters") or {}
        out["generation"] = {k: params.get(k) for k in
                             ("ckpt", "seed", "octree_resolution")
                             if params.get(k) is not None}
    cleanup = _load_json(run_dir / "cleanup_report.json")
    if cleanup:
        info: dict = {}
        for side in ("input", "output"):
            block = cleanup.get(side) or {}
            if block.get("faces") is not None:
                info[f"{side}_faces"] = block["faces"]
            if block.get("vertices") is not None:
                info[f"{side}_verts"] = block["vertices"]
        if cleanup.get("retopo_method"):
            info["method"] = cleanup["retopo_method"]
        if info:
            out["cleanup"] = info
    rig = _load_json(run_dir / "rig_report.json")
    if rig:
        out["rig"] = {
            "bones": rig.get("bones"),
            "weighted_ratio": (rig.get("weights") or {}).get("weighted_ratio"),
            "bind_method": rig.get("bind_method"),
        }
    gate = _load_json(run_dir / GATE_REPORT)
    if gate and isinstance(gate.get("checks"), dict):
        checks = gate["checks"]
        out["vrm_gate"] = {
            "all_pass": gate.get("all_pass"),
            "passed": sum(1 for ok in checks.values() if ok),
            "total": len(checks),
            "blockers": gate.get("blockers") or [],
        }
    return out


def _list_runs() -> dict:
    runs = []
    if RUNS_ROOT.is_dir():
        def _mtime(p: Path) -> float:
            try:  # 断链 symlink / 扫描中被删：跳过而不是炸掉整个响应
                return p.stat().st_mtime
            except OSError:
                return 0.0

        for run_dir in sorted(RUNS_ROOT.iterdir(), key=_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            models = sorted(
                ({"name": p.name, "size": s}
                 for p in run_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in MODEL_EXTS
                 and (s := _size(p)) is not None),
                key=lambda m: m["name"],
            )
            if models:
                runs.append({"run": run_dir.name, "models": models,
                             "reports": _run_reports(run_dir)})
    return {"runs": runs}


class Handler(BaseHTTPRequestHandler):
    server_version = "chargen-showcase/1.0"

    def log_message(self, format: str, *args) -> None:  # quiet 404 noise for favicon
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _send_file(self, path: Path) -> None:
        self._send(200, path.read_bytes(), MIME.get(path.suffix.lower(),
                                                    "application/octet-stream"))

    def _handle(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/runs":
            self._send_json(_list_runs())
            return

        if route == "/api/file":
            query = parse_qs(parsed.query)
            run = query.get("run", [""])[0]
            name = query.get("f", [""])[0]
            if not run or not name:
                self._send(400, b"missing run/f parameter")
                return
            path = _safe_under(RUNS_ROOT, f"{run}/{name}")
            if path is None:
                self._send(403, b"path escapes runs tree")
                return
            if path.suffix.lower() not in MODEL_EXTS or not path.is_file():
                self._send(404, b"no such model file")
                return
            self._send_file(path)
            return

        if route.startswith("/vendor/"):
            path = _safe_under(VENDOR_ROOT, route[len("/vendor/"):])
            if path is None or not path.is_file():
                self._send(404, b"not found")
                return
            self._send_file(path)
            return

        rel = "index.html" if route in ("/", "") else route.lstrip("/")
        path = _safe_under(WEB_ROOT, rel)
        if path is None or not path.is_file():
            self._send(404, b"not found")
            return
        self._send_file(path)

    do_GET = _handle
    do_HEAD = _handle


def main() -> None:
    parser = argparse.ArgumentParser(description="chargen showcase server")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8123")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not VENDOR_ROOT.is_dir():
        sys.exit(f"missing vendored three.js at {VENDOR_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"chargen showcase: http://{args.host}:{args.port}/  (runs: {RUNS_ROOT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
