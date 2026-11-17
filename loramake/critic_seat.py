"""The critic seat: one external vision model, single turn, strict JSON back.

Two transports. ``claude`` runs the same headless Claude Code binary the designer brain
uses (a different model is enforced, so the judge is never the author); ``grok`` runs the
grok-proxy CLI the defect scanner already uses. Both read the image by path with their own
file tool; nothing here uploads bytes. The reply is validated before anyone reads it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

from . import designer_brain as brain
from . import seat_proc
from .config import OUTPUTS

ROOT = OUTPUTS.parent
BACKENDS = ("claude", "grok")
STATUSES = ("ok", "violation", "uncertain")
DEFAULT_MODEL = "sonnet"
EVIDENCE_MAX = 200
SEAT_TIMEOUT_S = 180.0
DEFAULT_GROK_BIN = Path.home() / ".local" / "bin" / "grok-proxy"

SEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "evidence": {"type": "string"},
                },
                "required": ["element", "status", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["checks"],
    "additionalProperties": False,
}

SEAT_NOTE = (
    "You are the supervising critic (作画監督) for one anime character. You receive her design card "
    "as a checklist and one rendered image plus zoomed crops. Your only job is to list where the image "
    "violates the checklist. Judge only what is visible. Answer every element exactly once with status "
    "ok | violation | uncertain and evidence (at most 200 characters: what you saw and where). "
    "Do NOT propose fixes, prompts or edits. Do NOT rank, praise, pick or crown. Do NOT judge style or "
    "aesthetics. Mechanical facts given to you are settled; do not re-judge them. Use uncertain when a "
    "part is occluded or too small; never guess. Expect at least N violations on a typical render; if you "
    "find fewer, re-read the crops before answering. Reply with ONLY the JSON object."
)


class SeatError(RuntimeError):
    pass


def seat_config(backend: str | None = None) -> dict:
    """Resolve the seat from env: which binary, which model; refuses to share the brain's model."""
    name = (backend or os.environ.get("LORAMAKE_CRITIC_SEAT", "claude")).strip().lower()
    if name not in BACKENDS:
        raise SeatError(f"critic seat must be one of {BACKENDS}, got {name!r}")
    model = os.environ.get("LORAMAKE_CRITIC_MODEL", "").strip()
    if name == "claude":
        binary = Path(os.environ.get("LORAMAKE_CRITIC_BIN", "").strip() or brain.brain_bin())
        model = model or DEFAULT_MODEL
        brain_model = os.environ.get("LORAMAKE_BRAIN_MODEL", "").strip()
        if brain_model and brain_model == model:
            raise SeatError(f"critic model {model!r} is the designer brain's model; the judge must not be the author")
        restricted = os.environ.get("LORAMAKE_CRITIC_RESTRICTED", "1").strip() not in ("0", "", "no", "false")
        return {"backend": "claude", "bin": binary, "model": model, "restricted": restricted}
    binary = Path(os.path.expanduser(os.environ.get("GROK_BIN", "").strip() or str(DEFAULT_GROK_BIN)))
    return {"backend": "grok", "bin": binary, "model": model or "grok", "restricted": False}


def claude_argv(prompt: str, cfg: dict) -> list[str]:
    argv = [str(cfg["bin"]), "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(SEAT_SCHEMA),
            "--model", cfg["model"], "--system-prompt", SEAT_NOTE, "--tools", "Read", "--allowedTools", "Read",
            "--permission-prompts", "none", "--no-session-persistence"]
    if cfg.get("restricted"):
        argv.append("--restricted")
    return argv


def grok_argv(prompt_path: Path, cfg: dict, crops: int = 0) -> list[str]:
    argv = [str(cfg["bin"]), "--prompt-file", str(prompt_path), "--json-schema", json.dumps(SEAT_SCHEMA),
            "--max-turns", str(min(12, 4 + crops)), "--always-approve", "--no-memory", "--no-subagents",
            "--tools", "read_file"]
    if cfg.get("model") and cfg["model"] != "grok":
        argv += ["-m", cfg["model"]]
    return argv


def _model_used(raw: str) -> str | None:
    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError):
        return None
    usage = envelope.get("modelUsage") if isinstance(envelope, dict) else None
    return ", ".join(sorted(usage)) if isinstance(usage, dict) and usage else None


def call_seat(prompt: str, cfg: dict, packet_path: Path, *, timeout_s: float = SEAT_TIMEOUT_S,
              crops: int = 0) -> dict:
    """One seat call. Returns {ok, raw, error, elapsed_s, model_used}; never raises."""
    binary = Path(cfg["bin"])
    if not binary.is_file():
        return {"ok": False, "raw": "", "error": f"seat binary missing: {binary}", "elapsed_s": 0.0, "model_used": None}
    env = dict(os.environ)
    if cfg["backend"] == "claude":
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        argv = claude_argv(prompt, cfg)
    else:
        env.setdefault("GROK_CLAUDE_MCPS_ENABLED", "false")
        env.setdefault("GROK_CURSOR_MCPS_ENABLED", "false")
        argv = grok_argv(packet_path, cfg, crops)
    started = time.monotonic()
    result = seat_proc.run_seat(argv, timeout_s=timeout_s, env=env, cwd=str(ROOT))
    return {"ok": bool(result["ok"]), "raw": result.get("raw") or "", "error": result.get("error"),
            "elapsed_s": round(time.monotonic() - started, 1), "model_used": _model_used(result.get("raw") or "")}


def validate_checks(checks: object, keys: Sequence[str]) -> tuple[list[dict] | None, str | None]:
    """Exact-shape validation before a model verdict can be written anywhere."""
    if not isinstance(checks, list) or not checks:
        return None, "checks is not a non-empty list"
    wanted = list(keys)
    seen: dict[str, dict] = {}
    for index, item in enumerate(checks):
        if not isinstance(item, dict) or set(item) != {"element", "status", "evidence"}:
            return None, f"check {index} has unexpected fields"
        element, status, evidence = item["element"], item["status"], item["evidence"]
        if element not in wanted:
            return None, f"check {index} names an unknown element {element!r}"
        if element in seen:
            return None, f"element {element!r} answered twice"
        if status not in STATUSES:
            return None, f"check {index} has invalid status {status!r}"
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > EVIDENCE_MAX:
            return None, f"check {index} has invalid evidence"
        seen[element] = {"element": element, "status": status, "evidence": evidence.strip()}
    missing = [k for k in wanted if k not in seen]
    if missing:
        return None, "unanswered elements: " + ", ".join(missing)
    return [seen[k] for k in wanted], None


def parse_checks(raw: str, keys: Sequence[str]) -> tuple[list[dict] | None, str | None]:
    """Find the checks array in a seat reply (structured_output, result text, or bare JSON)."""
    candidates: list[dict] = []
    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError):
        envelope = None
    if isinstance(envelope, dict):
        if isinstance(envelope.get("structured_output"), dict):
            candidates.append(envelope["structured_output"])
        if isinstance(envelope.get("result"), str):
            body = brain._extract_json(envelope["result"])
            if body:
                candidates.append(body)
        candidates.append(envelope)
    body = brain._extract_json(raw or "")
    if body:
        candidates.append(body)
    for candidate in candidates:
        if isinstance(candidate.get("checks"), list):
            return validate_checks(candidate["checks"], keys)
    return None, "no checks array in the seat reply"
