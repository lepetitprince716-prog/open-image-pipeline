"""Civitai cloud helpers for the provider channel.

Resolves checkpoint aliases (``configs/civitai_models.json``) and LoRA specs into the
``imageGen`` fields Civitai expects, and keeps the standing Buzz budget: a per-call cap and
a daily cap checked against the free whatif quote before any paid submission, with every
charged image appended to the shared ledger ``outputs/civitai/buzz_ledger.jsonl``.
The budget is client-side only; Civitai has no per-API-key spend cap.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import OUTPUTS

CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs" / "civitai_models.json"
LEDGER_PATH = OUTPUTS / "civitai" / "buzz_ledger.jsonl"
LEDGER_BATCH = "provider"  # ledger rows written by this channel; probe batches use their own names
DEFAULT_MAX_CALL_BUZZ = 30
DEFAULT_DAILY_BUZZ = 150
DEFAULT_LORA_STRENGTH = 0.8
MAX_LORAS = 5
MIN_LORA_STRENGTH, MAX_LORA_STRENGTH = -2.0, 2.0
_CHECKPOINT_AIR = re.compile(r"^urn:air:([a-z0-9]+):checkpoint:civitai:\d+@\d+$")
_LORA_AIR = re.compile(r"^urn:air:([a-z0-9]+):(?:lora|lycoris|locon|dora):civitai:\d+@\d+$")


class BudgetExceeded(ValueError):
    """The quoted cost would break the standing Buzz budget; nothing was submitted."""


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("default") not in catalog.get("models", {}):
        raise ValueError(f"{path.name}: default alias is not defined")
    return catalog


def aliases(path: Path = CATALOG_PATH) -> dict[str, str]:
    """alias -> human label, for tool descriptions and error messages."""
    return {name: entry.get("label", name) for name, entry in load_catalog(path)["models"].items()}


def resolve_model(spec: str | None, path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Alias or checkpoint AIR -> engine/ecosystem/model field and default knobs.

    A raw AIR borrows the engine and defaults of the first alias in the same ecosystem,
    so only ecosystems with a whatif-proven alias are accepted.
    """
    catalog = load_catalog(path)
    models = catalog["models"]
    name = (spec or "").strip() or catalog["default"]
    if name in models:
        return {"loras": [], "triggers": [], **models[name], "alias": name}
    match = _CHECKPOINT_AIR.match(name)
    if not match:
        raise ValueError(
            f"unknown Civitai model {name!r}; use an alias ({', '.join(models)}) or a checkpoint AIR"
        )
    ecosystem = match.group(1)
    family = next((entry for entry in models.values() if entry["ecosystem"] == ecosystem), None)
    if family is None:
        supported = sorted({entry["ecosystem"] for entry in models.values()})
        raise ValueError(f"Civitai ecosystem {ecosystem!r} is not supported here ({', '.join(supported)})")
    # A raw AIR borrows engine and sampling defaults only, never an alias's LoRA stack.
    return {**family, "alias": None, "air": name, "label": name, "source": "caller AIR",
            "loras": [], "triggers": []}


def parse_loras(specs: list[str] | None, ecosystem: str,
                preset: list[str] | None = None) -> dict[str, float]:
    """``AIR[:strength]`` strings -> the ``loras`` map, checked against the checkpoint ecosystem.

    ``preset`` is an alias's LoRA stack; a caller spec with the same AIR overrides its strength.
    """
    stack = _parse_specs(preset, ecosystem)
    extra = _parse_specs(specs, ecosystem)
    out = {**stack, **extra}
    if len(out) > MAX_LORAS:
        raise ValueError(f"at most {MAX_LORAS} LoRAs per Civitai request")
    return out


def _parse_specs(specs: list[str] | None, ecosystem: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in specs or []:
        spec = str(raw).strip()
        air, strength = spec, DEFAULT_LORA_STRENGTH
        head, sep, tail = spec.rpartition(":")
        if sep:
            try:
                strength = float(tail)
                air = head
            except ValueError:
                pass  # the last colon belongs to the AIR itself
        match = _LORA_AIR.match(air)
        if not match:
            raise ValueError(f"LoRA must be a Civitai LoRA AIR[:strength], got {spec!r}")
        if match.group(1) != ecosystem:
            raise ValueError(f"LoRA {air} is {match.group(1)}, but the checkpoint is {ecosystem}")
        if not MIN_LORA_STRENGTH <= strength <= MAX_LORA_STRENGTH:
            raise ValueError(f"LoRA strength must be in {MIN_LORA_STRENGTH}..{MAX_LORA_STRENGTH}")
        if air in out:
            raise ValueError(f"LoRA {air} is listed twice")
        out[air] = strength
    return out


def apply_triggers(prompt: str, triggers: list[str] | None) -> str:
    """Add an alias's trigger tokens that the prompt lacks, after a leading ``safe`` rating tag."""
    missing = [token for token in triggers or [] if token not in prompt]
    if not missing:
        return prompt
    head, sep, rest = prompt.partition(",")
    if sep and head.strip() == "safe":
        return f"{head}, {', '.join(missing)},{rest}"
    return f"{', '.join(missing)}, {prompt}"


def buzz_total(cost_entries: list[dict[str, Any]]) -> int | None:
    """Sum of a normalized ``transactions`` list; None when the provider reported nothing."""
    amounts = [entry["amount"] for entry in cost_entries if isinstance(entry.get("amount"), (int, float))]
    return int(round(sum(amounts))) if amounts else None


def _env_budget(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def spent_today(path: Path | None = None, date: str | None = None) -> int:
    """Buzz this channel charged on one local calendar day (the dashboard's day boundary)."""
    path = path or LEDGER_PATH
    day = date or time.strftime("%Y-%m-%d")
    if not path.is_file():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(row, dict) and row.get("batch") == LEDGER_BATCH
                and str(row.get("time") or "").startswith(day)
                and isinstance(row.get("cost_buzz"), (int, float))):
            total += int(row["cost_buzz"])
    return total


def check_budget(cost: int | None, path: Path | None = None) -> dict[str, int]:
    """Fail closed unless the quoted cost fits both the per-call and the daily cap."""
    max_call = _env_budget("LORAMAKE_CIVITAI_MAX_CALL_BUZZ", DEFAULT_MAX_CALL_BUZZ)
    daily = _env_budget("LORAMAKE_CIVITAI_DAILY_BUZZ", DEFAULT_DAILY_BUZZ)
    spent = spent_today(path)
    status = {"cost": cost, "max_call": max_call, "daily": daily, "spent_today": spent}
    if cost is None:
        raise BudgetExceeded("Civitai whatif returned no cost; refusing to submit an unpriced request")
    if cost > max_call:
        raise BudgetExceeded(f"quoted {cost} Buzz exceeds the per-call cap of {max_call}")
    if spent + cost > daily:
        raise BudgetExceeded(
            f"quoted {cost} Buzz would bring today's spend to {spent + cost}, over the daily cap of {daily}"
        )
    # ponytail: check-then-submit is not atomic; the Studio worker serializes jobs, add a file
    # lock here if direct gen.py runs ever go parallel.
    return status


def append_ledger(files: list[str], *, request_id: str | None, model: str, charged: int,
                  path: Path | None = None) -> None:
    """One row per stored image, in the probe's row shape so spend views and rounds read it.
    No files (paid but not stored) still writes one row with ``file: null`` for the charge."""
    path = path or LEDGER_PATH
    slots: list[str | None] = list(files) or [None]
    share, remainder = divmod(charged, len(slots))
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, file in enumerate(slots):
            row = {
                "time": stamp,
                "batch": LEDGER_BATCH,
                "id": f"{request_id or 'unknown'}#{index + 1}",
                "model": model,
                "cost_buzz": share + (remainder if index == 0 else 0),
                "file": file,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
