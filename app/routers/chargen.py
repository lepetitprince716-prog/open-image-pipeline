"""Chargen 3D run listing and model-file serving for the viewer sub-site."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

MODEL_EXTS = {".glb", ".vrm"}
SERVABLE_EXTS = MODEL_EXTS | {".json", ".png", ".jpg", ".jpeg"}
MEDIA_TYPES = {
    ".glb": "model/gltf-binary",
    ".vrm": "application/octet-stream",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
REPORT_FILES = ("run_manifest.json", "sheet_manifest.json", "cleanup_report.json",
                "rig_report.json")
GATE_REPORT = Path("vrm_gate") / "base_gate_report.json"


def _load_json(path: Path) -> dict | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _gate_summary(run_dir: Path) -> dict | None:
    report = _load_json(run_dir / GATE_REPORT)
    if report is None:
        return None
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return None
    return {
        "all_pass": report.get("all_pass"),
        "checks_passed": sum(1 for ok in checks.values() if ok),
        "checks_total": len(checks),
        "blockers": report.get("blockers", []),
    }


def create_router(state: Any) -> APIRouter:
    router = APIRouter()
    runs_root: Path = (Path(state.ROOT) / "3d" / "chargen" / "runs").resolve()

    def _run_file(run: str, name: str) -> Path:
        path = (runs_root / run / name).resolve()
        if not path.is_relative_to(runs_root):
            raise HTTPException(status_code=403, detail="path escapes the chargen runs tree")
        if path.suffix.lower() not in SERVABLE_EXTS:
            raise HTTPException(status_code=400, detail=f"unsupported extension: {path.suffix}")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no such file")
        return path

    @router.get("/api/chargen/runs")
    def list_runs() -> dict:
        runs = []
        if runs_root.is_dir():
            for run_dir in sorted(runs_root.iterdir(),
                                  key=lambda p: p.stat().st_mtime, reverse=True):
                if not run_dir.is_dir():
                    continue
                models = sorted(
                    ({"name": p.name, "size": p.stat().st_size}
                     for p in run_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in MODEL_EXTS),
                    key=lambda m: m["name"],
                )
                if not models:
                    continue
                reports = {name.removesuffix(".json"): report for name in REPORT_FILES
                           if (report := _load_json(run_dir / name)) is not None}
                runs.append({
                    "run": run_dir.name,
                    "models": models,
                    "reports": reports,
                    "gate": _gate_summary(run_dir),
                })
        return {"runs": runs}

    @router.get("/api/chargen/file")
    @router.head("/api/chargen/file", include_in_schema=False)
    def serve_run_file(run: str, f: str):
        path = _run_file(run, f)
        return FileResponse(path, media_type=MEDIA_TYPES[path.suffix.lower()])

    return router
