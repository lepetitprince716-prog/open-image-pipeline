"""Uniform quality gate chain for generated images: SFW, defect floor, structure counts.

Every gate answers ``pass``, ``fail`` or ``unverified``. On an automated line an
unverified answer is a rejection; on an interactive line the image is kept and the gap is
recorded. All detectors run through the imgutils venv bridge (``scripts/imgutils_bridge.py``)
on this machine, so cloud renders get the same chain as local ones.

The thresholds below are the single source for the chain; ``scripts/sfw_gate.py`` and the
gacha floor mirror them until they import from here (``tests/test_gates.py`` guards drift).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loramake.config import ROOT

BRIDGE = ROOT / "scripts" / "imgutils_bridge.py"
DEFAULT_VENV = ROOT / "comfyui-boxa" / "tools" / "visual_features" / ".venv" / "bin" / "python"
BRIDGE_TIMEOUT_S = 900

EXPLICIT_MAX = 0.15
QUESTIONABLE_MAX = 0.25
AESTHETIC_FLOOR = 0.55
MONO_THRESHOLD = 0.5

MODES = ("automated", "interactive")
GATES = ("sfw", "defect", "structure")
STATUSES = ("pass", "fail", "unverified")
REPORT_SCHEMA = "loramake.gate-report/v1"
# Styles that are grey, ink or monochrome on purpose: the monochrome rule is advisory there.
MONOCHROME_STYLES = frozenset({"muted_grey", "manga_bw", "sketch_line", "guofeng_ink", "handmade_imperfect"})


def monochrome_expected(style_id: str | None) -> bool:
    return bool(style_id) and style_id in MONOCHROME_STYLES


def venv_python() -> str:
    return os.environ.get("IMGUTILS_VENV_PY") or str(DEFAULT_VENV)


# --------------------------------------------------------------------------- bridge calls


def _bridge(venv: str, op: str, paths: Sequence[str]) -> dict | None:
    if not paths or not os.path.exists(venv):
        return None
    try:
        proc = subprocess.run([venv, str(BRIDGE), op, *paths], capture_output=True, text=True,
                              timeout=BRIDGE_TIMEOUT_S)
        data = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def wd14_many(venv: str, paths: Sequence[str]) -> dict[str, dict | None]:
    """Rating dict per path from one batched ``wd14`` call; None where unverifiable."""
    data = _bridge(venv, "wd14", paths)
    out: dict[str, dict | None] = {p: None for p in paths}
    if data is None:
        return out

    def rating_of(entry: Any) -> dict | None:
        if isinstance(entry, dict) and entry.get("ok") and isinstance(entry.get("rating"), dict):
            return entry["rating"]
        return None

    results = data.get("results")
    if isinstance(results, dict):
        for p in paths:
            out[p] = rating_of(results.get(p))
    elif len(paths) == 1:
        out[paths[0]] = rating_of(data)
    return out


def qc_many(venv: str, paths: Sequence[str]) -> dict[str, dict | None]:
    """QC dict per path from one batched ``qc`` call (one model load for the whole batch)."""
    data = _bridge(venv, "qc", paths)
    out: dict[str, dict | None] = {p: None for p in paths}
    if data is None:
        return out
    results = data.get("results")
    if isinstance(results, dict):
        for p in paths:
            entry = results.get(p)
            out[p] = entry if isinstance(entry, dict) and entry.get("ok") else None
    elif len(paths) == 1 and data.get("ok"):
        out[paths[0]] = data
    return out


def detect_many(venv: str, paths: Sequence[str]) -> dict[str, dict | None]:
    """Detector counts per path from one batched ``detect`` call."""
    data = _bridge(venv, "detect", paths)
    out: dict[str, dict | None] = {p: None for p in paths}
    if not data or not data.get("ok") or not isinstance(data.get("results"), dict):
        return out
    for p in paths:
        entry = data["results"].get(p)
        if not isinstance(entry, dict):
            continue
        counts = {}
        for key in ("hands", "eyes", "faces", "heads", "persons", "text", "feet"):
            boxes = entry.get(key)
            counts[key] = len(boxes) if isinstance(boxes, list) else None
        completeness = entry.get("completeness")
        counts["completeness"] = completeness.get("label") if isinstance(completeness, dict) else None
        out[p] = counts
    return out


# --------------------------------------------------------------------------- gate rules


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sfw_verdict(rating: dict | None, explicit_max: float = EXPLICIT_MAX,
                questionable_max: float = QUESTIONABLE_MAX) -> tuple[str, list[str]]:
    if not rating or not _finite(rating.get("explicit")) or not _finite(rating.get("questionable")):
        return "unverified", ["no_rating"]
    reasons = []
    if rating["explicit"] >= explicit_max:
        reasons.append(f"explicit={rating['explicit']:.3f}>={explicit_max}")
    if rating["questionable"] >= questionable_max:
        reasons.append(f"questionable={rating['questionable']:.3f}>={questionable_max}")
    return ("fail" if reasons else "pass"), reasons


def defect_verdict(qc: dict | None, subjects: int | None = 1,
                   floor: float = AESTHETIC_FLOOR, allow_monochrome: bool = False) -> tuple[str, list[str]]:
    """Mirror of the gacha triple gate: completeness, monochrome, counts, then the floor.

    ``allow_monochrome`` drops the monochrome rule for styles that are grey, ink or
    monochrome on purpose (muted palettes, manga, sketch); the score is still recorded.
    """
    if not qc:
        return "unverified", ["qc_unavailable"]
    reasons = []
    if qc.get("completeness") != "polished":
        reasons.append(f"incomplete:{qc.get('completeness')}")
    mono = qc.get("monochrome")
    if not allow_monochrome and _finite(mono) and mono >= MONO_THRESHOLD:
        reasons.append(f"monochrome={mono:.2f}")
    if subjects is not None:
        faces, hands = qc.get("faces"), qc.get("hands")
        if _finite(faces) and faces != subjects:
            reasons.append(f"faces={faces}!={subjects}")
        # Deliberately looser than structure_verdict: this mirrors the gacha qc rule
        # (one spurious hand box is tolerated); the structure gate is the strict count.
        if _finite(hands) and hands > 2 * subjects + 1:
            reasons.append(f"hands={hands}")
    pct = qc.get("aesthetic_percentile")
    if not _finite(pct):
        return "unverified", reasons + ["qc_score_missing"]
    if pct < floor:
        reasons.append(f"below_floor:{pct:.3f}<{floor}")
    return ("fail" if reasons else "pass"), reasons


def structure_verdict(counts: dict | None, subjects: int = 1) -> tuple[str, list[str]]:
    """Detector-count consistency: the subject is there once, with at most its own limbs."""
    if not counts:
        return "unverified", ["detect_unavailable"]
    persons, heads, hands = counts.get("persons"), counts.get("heads"), counts.get("hands")
    if any(v is None for v in (persons, heads, hands)):
        return "unverified", ["detect_incomplete"]
    reasons = []
    if persons == 0:
        reasons.append("no_person")
    elif persons > subjects:
        reasons.append(f"extra_persons={persons}")
    if heads == 0:
        reasons.append("no_head")
    elif heads > subjects:
        reasons.append(f"extra_heads={heads}")
    if hands > 2 * subjects:
        reasons.append(f"extra_hands={hands}")
    if counts.get("completeness") not in (None, "polished"):
        reasons.append(f"completeness:{counts['completeness']}")
    return ("fail" if reasons else "pass"), reasons


# --------------------------------------------------------------------------- chain


@dataclass
class GateResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class GateReport:
    path: str
    mode: str
    subjects: int
    gates: dict[str, GateResult]
    overall: str
    schema: str = REPORT_SCHEMA

    def as_dict(self) -> dict:
        return {"schema": self.schema, "path": self.path, "mode": self.mode, "subjects": self.subjects,
                "overall": self.overall,
                "gates": {name: asdict(result) for name, result in self.gates.items()}}


def overall_status(gates: dict[str, GateResult], mode: str) -> str:
    """fail wins; otherwise unverified counts as fail on automated lines."""
    statuses = [g.status for g in gates.values()]
    if "fail" in statuses:
        return "fail"
    if "unverified" in statuses:
        return "fail" if mode == "automated" else "unverified"
    return "pass"


def run_chain(paths: Sequence[str], *, mode: str = "automated", subjects: int = 1,
              venv: str | None = None, floor: float = AESTHETIC_FLOOR, allow_monochrome: bool = False,
              wd14_fn: Callable[[str, Sequence[str]], dict] = wd14_many,
              qc_fn: Callable[[str, Sequence[str]], dict] = qc_many,
              detect_fn: Callable[[str, Sequence[str]], dict] = detect_many) -> list[GateReport]:
    """Run every gate on every image and return one report per image.

    SFW runs first on the whole batch; images that fail it skip the heavier gates so a
    rejected image never has its content scored. Unverified SFW still runs the rest, so an
    interactive caller sees every signal that was available.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    paths = [str(p) for p in paths]
    venv = venv or venv_python()
    ratings = wd14_fn(venv, paths)
    sfw: dict[str, GateResult] = {}
    to_score: list[str] = []
    for p in paths:
        status, reasons = sfw_verdict(ratings.get(p))
        sfw[p] = GateResult(status, reasons, {"rating": ratings.get(p)})
        if status != "fail":
            to_score.append(p)
    qc = qc_fn(venv, to_score) if to_score else {}
    detect = detect_fn(venv, to_score) if to_score else {}
    reports = []
    for p in paths:
        gates = {"sfw": sfw[p]}
        if p in to_score:
            d_status, d_reasons = defect_verdict(qc.get(p), subjects, floor, allow_monochrome)
            gates["defect"] = GateResult(d_status, d_reasons, {"qc": qc.get(p)})
            s_status, s_reasons = structure_verdict(detect.get(p), subjects)
            gates["structure"] = GateResult(s_status, s_reasons, {"counts": detect.get(p)})
        else:
            gates["defect"] = GateResult("fail", ["skipped:sfw_fail"])
            gates["structure"] = GateResult("fail", ["skipped:sfw_fail"])
        reports.append(GateReport(p, mode, subjects, gates, overall_status(gates, mode)))
    return reports


def sidecar_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    return path.with_name(path.name + ".gates.json")


def write_sidecars(reports: Sequence[GateReport]) -> list[Path]:
    written = []
    for report in reports:
        target = sidecar_path(report.path)
        target.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        written.append(target)
    return written


def summarize(reports: Sequence[GateReport]) -> dict:
    summary = {"images": len(reports), "overall": {s: 0 for s in STATUSES},
               "gates": {g: {s: 0 for s in STATUSES} for g in GATES}, "reasons": {}}
    for report in reports:
        summary["overall"][report.overall] += 1
        for name, result in report.gates.items():
            summary["gates"][name][result.status] += 1
            for reason in result.reasons:
                key = reason.split("=")[0].split(":")[0]
                summary["reasons"][key] = summary["reasons"].get(key, 0) + 1
    return summary
