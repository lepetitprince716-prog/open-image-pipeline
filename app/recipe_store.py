"""Local recipe library — the management layer on top of the LoRA Manager store.

The LM store on Z3 owns the raw recipes and previews; this library owns the three
things LM never records: provenance (the raw Civitai metadata captured at import
time), lineage (which recipes a fused recipe derives from), and runs (which gen jobs
an applied recipe produced). Records are one-JSON-per-recipe, keyed by the LM recipe
id, split SFW/NSFW by tree:

    recipes/library/<id>.json          SFW
    nsfw/recipes/library/<id>.json     NSFW (site=red / preview level / prompt screen)

Schema loramake.recipe/v1. Writes are atomic (tmp + replace). No network I/O here.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from app import recipe_brief

SCHEMA = "loramake.recipe/v1"
ROOT = Path(__file__).resolve().parent.parent
# Tests (and any sandboxed caller) point this at a scratch dir so library writes never
# land in the repo tree; see tests/conftest.py.
LIBRARY_ROOT_ENV = "LORAMAKE_RECIPE_LIBRARY_ROOT"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Reentrant: upsert/append hold it across their read-modify-write while save_record
# re-acquires it for the atomic replace.
_write_lock = threading.RLock()


def sfw_library() -> Path:
    override = os.environ.get(LIBRARY_ROOT_ENV)
    if override:
        return Path(override) / "recipes" / "library"
    return ROOT / "recipes" / "library"


def nsfw_library() -> Path:
    override = os.environ.get(LIBRARY_ROOT_ENV)
    if override:
        return Path(override) / "nsfw" / "recipes" / "library"
    return ROOT / "nsfw" / "recipes" / "library"


def _library_for(nsfw: bool) -> Path:
    return nsfw_library() if nsfw else sfw_library()


def _record_path(recipe_id: str, nsfw: bool) -> Path:
    if not _ID_RE.match(recipe_id):
        raise ValueError(f"invalid recipe id: {recipe_id!r}")
    return _library_for(nsfw) / f"{recipe_id}.json"


def find_record_path(recipe_id: str) -> Path | None:
    """Existing record path in either tree (SFW checked first)."""
    for nsfw in (False, True):
        try:
            path = _record_path(recipe_id, nsfw)
        except ValueError:
            return None
        if path.is_file():
            return path
    return None


def load_record(recipe_id: str) -> dict[str, Any] | None:
    path = find_record_path(recipe_id)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_record(record: dict[str, Any]) -> Path:
    recipe_id = str(record.get("id") or "")
    nsfw = bool(record.get("nsfw"))
    path = _record_path(recipe_id, nsfw)
    with _write_lock:
        # A record that flips SFW<->NSFW moves trees; drop the stale twin.
        other = find_record_path(recipe_id)
        if other is not None and other != path:
            other.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
    return path


def list_records(*, include_nsfw: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    trees = [sfw_library()] + ([nsfw_library()] if include_nsfw else [])
    for tree in trees:
        if not tree.is_dir():
            continue
        for path in sorted(tree.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                records.append(data)
    records.sort(key=lambda r: r.get("importedAt") or "", reverse=True)
    return records


def new_record(
    recipe_dto: dict[str, Any],
    *,
    origin: str = "civitai",
    site: str | None = None,
    provenance: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    stamp = _iso(now)
    return {
        "schema": SCHEMA,
        "id": str(recipe_dto.get("id") or ""),
        "lmId": str(recipe_dto.get("id") or "") or None,
        "origin": origin,
        "title": recipe_dto.get("title") or "",
        "sourceUrl": recipe_dto.get("sourceUrl") or "",
        "nsfw": recipe_brief.looks_nsfw(recipe_dto, site),
        "importedAt": stamp,
        "updatedAt": stamp,
        "provenance": provenance or {},
        "recipe": recipe_dto,
        "lineage": {"parents": [], "strategy": None, "conflictsResolved": []},
        "runs": [],
    }


def upsert_from_lm(
    recipe_dto: dict[str, Any],
    *,
    site: str | None = None,
    provenance: dict[str, Any] | None = None,
    now: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Mirror one LM recipe into the library. -> (record, created). Existing lineage,
    runs, and captured provenance are preserved; the recipe snapshot is refreshed."""
    recipe_id = str(recipe_dto.get("id") or "")
    if not recipe_id:
        raise ValueError("recipe DTO has no id")
    with _write_lock:
        existing = load_record(recipe_id)
        if existing is None:
            record = new_record(recipe_dto, site=site, provenance=provenance, now=now)
            save_record(record)
            return record, True
        existing["recipe"] = recipe_dto
        existing["title"] = recipe_dto.get("title") or existing.get("title") or ""
        existing["nsfw"] = recipe_brief.looks_nsfw(recipe_dto, site) or bool(existing.get("nsfw"))
        if provenance:
            merged = dict(existing.get("provenance") or {})
            for key, value in provenance.items():
                merged.setdefault(key, value)
            existing["provenance"] = merged
        existing["updatedAt"] = _iso(now)
        save_record(existing)
        return existing, False


def append_run(recipe_id: str, run: dict[str, Any], *, now: float | None = None) -> dict[str, Any] | None:
    """Attach one generation run ({jobId|resultSummary|...}) to a record, if it exists."""
    with _write_lock:
        record = load_record(recipe_id)
        if record is None:
            return None
        runs = record.setdefault("runs", [])
        runs.append({"at": _iso(now), **run})
        record["updatedAt"] = _iso(now)
        save_record(record)
        return record


def enrichment(recipe_id: str) -> dict[str, Any] | None:
    """The local-only fields worth merging into an LM detail response."""
    record = load_record(recipe_id)
    if record is None:
        return None
    return {
        "origin": record.get("origin"),
        "importedAt": record.get("importedAt"),
        "lineage": record.get("lineage"),
        "runs": record.get("runs") or [],
        "provenanceCaptured": bool(record.get("provenance")),
    }


def _iso(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now if now is not None else time.time()))
