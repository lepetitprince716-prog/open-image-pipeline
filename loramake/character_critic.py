"""The critic (作画監督) for the character atelier: one image, one design card, a list of violations.

The seat answers every checklist element with ok / violation / uncertain and evidence; the
report is written next to the character's other output and summarised into the image's
sidecar under ``gates.critic``. There is no ``pass``: a report is ``reviewed`` or
``unverified``, and nothing downstream is allowed to act on it — the human still picks.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import character_atelier as atelier
from . import character_recipes as recipes
from . import critic_checklist as checklist
from . import critic_seat as seat
from .character_set import GATE_MAX_DIFF
from .config import OUTPUTS

ROOT = OUTPUTS.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import character_design as cd  # noqa: E402

SCHEMA = "loramake.critic-report/v1"
CROP_PAD_FRAC = 0.25      # same numbers as scripts/defect_scan.py: digit-level acuity needs the zoom
CROP_MIN_SIDE = 448
CROP_MAX_SCALE = 4
CROP_MAX = 8
DEFAULT_MIN_VIOLATIONS = 1
STATUSES = ("reviewed", "unverified")


class CriticError(RuntimeError):
    pass


# ---------------------------------------------------------------- paths

def _slug(rel: str | Path) -> str:
    path = Path(rel)
    return f"{path.parent.name}__{path.stem}"


def qc_dir(card_id: str) -> Path:
    return atelier.home(card_id) / "qc"


def report_path(card_id: str, rel: str | Path) -> Path:
    return qc_dir(card_id) / f"{_slug(rel)}.critic.json"


def packet_path(card_id: str, rel: str | Path) -> Path:
    return qc_dir(card_id) / f"{_slug(rel)}.packet.md"


def crops_dir(card_id: str, rel: str | Path) -> Path:
    return qc_dir(card_id) / "crops" / _slug(rel)


# ---------------------------------------------------------------- packet ingredients

def make_crops(image: Path, out_dir: Path, found: dict | None = None) -> list[dict]:
    """Padded, upscaled crops of heads, hands and (on a multi-figure sheet) each figure."""
    try:
        from PIL import Image
    except ImportError:
        return []
    if found is None:
        found = recipes._detect(image)
    heads = recipes._boxes(found, "heads")
    hands = recipes._boxes(found, "hands")
    persons = sorted(recipes._boxes(found, "persons"))
    wanted: list[tuple[str, tuple[int, int, int, int], str]] = []
    wanted += [("head", box, f"head {i + 1} of {len(heads)}") for i, box in enumerate(heads)]
    wanted += [("hand", box, f"hand {i + 1} of {len(hands)}") for i, box in enumerate(hands)]
    if len(persons) > 1:
        wanted += [("figure", box, f"figure {i + 1} of {len(persons)}, left to right") for i, box in enumerate(persons)]
    if not wanted:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    crops: list[dict] = []
    with Image.open(image) as im:
        im = im.convert("RGB")
        for index, (label, (x0, y0, x1, y1), note) in enumerate(wanted[:CROP_MAX]):
            pw, ph = int((x1 - x0) * CROP_PAD_FRAC), int((y1 - y0) * CROP_PAD_FRAC)
            box = (max(0, x0 - pw), max(0, y0 - ph), min(im.width, x1 + pw), min(im.height, y1 + ph))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            crop = im.crop(box)
            scale = min(CROP_MAX_SCALE, max(1, -(-CROP_MIN_SIDE // min(crop.width, crop.height))))
            if scale > 1:
                crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
            path = out_dir / f"crop_{index:02d}_{label}.png"
            crop.save(path)
            crops.append({"label": label, "path": str(path), "note": note})
    return crops


def mechanical_facts(image: Path, card_id: str, card: dict, kind: str, venv: str | None = None) -> dict:
    """Settled numbers for the packet: the gate chain and CCIP against the sheet bust."""
    if kind == "sheet":
        return {"gate_chain": None, "ccip": None,
                "note": "multi-figure reference sheet: per-subject structure and defect counts do not apply"}
    from . import gates
    venv = venv or gates.venv_python()
    if not Path(venv).is_file():
        return {"gate_chain": None, "ccip": None, "note": f"imgutils venv missing ({venv}); no mechanical facts"}
    report = gates.run_chain([str(image)], mode="interactive", subjects=1, venv=venv)[0]
    facts: dict[str, Any] = {"gate_chain": {name: {"status": r.status, "reasons": list(r.reasons)}
                                            for name, r in report.gates.items()}, "ccip": None, "note": None}
    ref = atelier.home(card_id) / "design" / "bust.png"
    if not ref.is_file():
        anchor = (card.get("anchor") or {}).get("image")
        ref = ROOT / anchor if anchor else ref
    if ref.is_file():
        import consistency_check as cc
        found = cc.ccip_many(venv, str(ref), [str(image)]).get(str(image))
        if found and "difference" in found:
            diff = float(found["difference"])
            facts["ccip"] = {"difference": round(diff, 4), "max_diff": GATE_MAX_DIFF,
                             "hold": diff <= GATE_MAX_DIFF, "ref": atelier._rel(ref)}
    return facts


def build_packet(card: dict, sha8: str, kind: str, checks: list[checklist.Check], image: Path,
                 crops: list[dict], facts: dict | None, min_violations: int) -> str:
    frame = checklist.KINDS[kind]
    lines = [f"# Critic packet — {card.get('name_cn', '')} ({card.get('id', '')}) · card {sha8} · {kind} ({frame} frame)",
             "", f"Read the full image first: {image}", ""]
    if crops:
        lines.append("Zoomed crops (read each one):")
        lines += [f"- {c['path']} — {c['note']}" for c in crops]
    else:
        lines.append("No detector crops for this image; judge from the full image only.")
    lines.append("")
    lines.append("Mechanical facts (settled; do not re-judge):")
    if facts and facts.get("gate_chain"):
        for name, gate in facts["gate_chain"].items():
            reasons = ", ".join(gate.get("reasons") or []) or "-"
            lines.append(f"- gate {name}: {gate.get('status')} ({reasons})")
    if facts and facts.get("ccip"):
        c = facts["ccip"]
        lines.append(f"- identity vs the sheet bust: CCIP difference {c['difference']} "
                     f"({'holds' if c['hold'] else 'drifts'} at max {c['max_diff']})")
    if not facts or (not facts.get("gate_chain") and not facts.get("ccip")):
        lines.append(f"- {(facts or {}).get('note') or 'none available'}")
    lines += ["", "Checklist — answer EVERY element exactly once, by its bracketed key:"]
    lines += [f"- {'MUST' if c.must else 'MUST NOT'} [{c.element}] {c.text}" for c in checks]
    lines += ["", f"Expect at least {min_violations} violation(s) on a typical render; if you find fewer, "
              "re-read the crops before answering.",
              'Respond with ONLY: {"checks": [{"element": "<key>", "status": "ok|violation|uncertain", "evidence": "<what you saw, where>"}, ...]}']
    return "\n".join(lines) + "\n"


def summary_of(report: dict) -> dict:
    """What the atelier state and the sidecar keep: counts and pointers, never a pass."""
    return {"status": report.get("status"), "violations": int(report.get("violations") or 0),
            "uncertain": int(report.get("uncertain") or 0), "card_sha8": report.get("card_sha8"),
            "report": report.get("report"), "at": report.get("at")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- the review

def review(card_id: str, image: str | Path, *, kind: str | None = None, expression: str | None = None,
           backend: str | None = None, min_violations: int = DEFAULT_MIN_VIOLATIONS, mechanical: bool = True,
           dry_run: bool = False, timeout_s: float = seat.SEAT_TIMEOUT_S) -> dict:
    """Checklist -> facts -> crops -> packet -> seat -> validated report -> stamp.

    Usage errors raise CriticError. Everything the seat can get wrong (missing binary, exit
    code, timeout, garbage, a reply that skips an element) yields ``status: unverified``.
    """
    card = atelier.load_card(card_id)
    if card is None:
        raise CriticError(f"no design card {card_id!r}")
    image = Path(image)
    if not image.is_file():
        raise CriticError(f"no such image: {image}")
    image = image.resolve()
    rel = atelier._rel(image) or str(image)
    sha8 = cd.card_sha8(card)
    try:
        if kind is None:
            kind, expression = checklist.kind_of(rel)
        checks = checklist.checklist_from_card(card, kind, expression)
    except checklist.ChecklistError as exc:
        raise CriticError(str(exc)) from exc
    keys = [c.element for c in checks]
    facts = mechanical_facts(image, card_id, card, kind) if mechanical else None
    crops = [] if dry_run else make_crops(image, crops_dir(card_id, rel))
    packet = build_packet(card, sha8, kind, checks, image, crops, facts, min_violations)
    if dry_run:
        return {"kind": kind, "expression": expression, "card_sha8": sha8, "image": rel,
                "checklist": [c._asdict() for c in checks], "crops": crops, "facts": facts, "packet": packet}

    report: dict[str, Any] = {
        "schema": SCHEMA, "card_id": card_id, "card_sha8": sha8, "image": rel, "image_sha256": _sha256(image),
        "kind": kind, "frame": checklist.KINDS[kind], "expression": expression,
        "seat": {"backend": None, "model": None, "model_used": None, "elapsed_s": 0.0},
        "facts": facts, "checklist": [c._asdict() for c in checks], "checks": [],
        "violations": 0, "uncertain": 0, "violations_min": min_violations, "violations_min_met": False,
        "status": "unverified", "error": None, "raw_tail": None,
        "report": atelier._rel(report_path(card_id, rel)), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        cfg = seat.seat_config(backend)
    except seat.SeatError as exc:
        cfg = None
        report["error"] = str(exc)
    if cfg is not None:
        packet_file = packet_path(card_id, rel)
        packet_file.parent.mkdir(parents=True, exist_ok=True)
        packet_file.write_text(packet, encoding="utf-8")
        result = seat.call_seat(packet, cfg, packet_file, timeout_s=timeout_s, crops=len(crops))
        report["seat"] = {"backend": cfg["backend"], "model": cfg["model"], "model_used": result["model_used"],
                          "elapsed_s": result["elapsed_s"]}
        if not result["ok"]:
            report["error"] = result["error"]
            report["raw_tail"] = (result["raw"] or "")[-500:] or None
        else:
            answers, err = seat.parse_checks(result["raw"], keys)
            if err:
                report["error"] = f"seat reply rejected: {err}"
                report["raw_tail"] = (result["raw"] or "")[-500:] or None
            else:
                report["checks"] = answers
                report["violations"] = sum(1 for a in answers if a["status"] == "violation")
                report["uncertain"] = sum(1 for a in answers if a["status"] == "uncertain")
                report["violations_min_met"] = report["violations"] >= min_violations
                report["status"] = "reviewed"
    out = report_path(card_id, rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atelier.stamp(card_id, rel, sha8, critic=summary_of(report))
    return report
