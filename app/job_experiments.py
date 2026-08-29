"""Auto-record finished Studio jobs as canonical experiment manifests.

Every completed console job is written to
``outputs/_analysis/studio_jobs/<experiment_id>/manifest.json``
(schema ``loramake.experiment-manifest.v1``) so the experiments catalog keeps
being fed by real console activity (2026-08-10 user decision: every generation
run is recorded). Ungrouped jobs get one experiment each; jobs carrying a
batch-matrix ``group`` tag accumulate into one experiment per group. The
experiments catalog picks manifests up on its next scan; this module only
emits files and never touches the catalog DB. The ``studio_jobs/`` namespace
is owned by this recorder — do not hand-edit manifests inside it.

Recording is best-effort: ``record_finished_job`` never raises, so a recording
failure can never change job state.

Deliberate simplifications (see ``_progress/active/2026-08-10-auto-experiment-manifest.md``):

- ``arm.model`` is the console source line (``studio-<tier>@<box>``), not the
  base checkpoint: the ``loramake.gen-result/v2`` contract does not carry it yet.
- ``arm.recipe`` is omitted: the app layer cannot see the engine's effective
  sampling parameters, and the canonical schema allows recipe-less arms.
- ``asset_availability`` is ``local_only``: gallery trash-delete may remove
  renders after recording; the recorded sha256/bytes stay the evidence of
  record and the experiment stays indexable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_ROOT = ROOT / "outputs" / "_analysis" / "studio_jobs"

_SCHEMA = "loramake.experiment-manifest.v1"
_EXPERIMENT_ID_RE = re.compile(r"exp_[a-z0-9][a-z0-9_-]{2,63}\Z")
_SLUG_STRIP = re.compile(r"[^a-z0-9_-]+")
# Display/prompt caps mirror app/experiments.py _normalize_canonical; keep in sync.
_MAX_TITLE = 200
_MAX_LABEL = 120
_MAX_MODEL = 200
_MAX_DIMENSION_PROMPT = 16000


def _slug(text: object, max_len: int) -> str:
    return _SLUG_STRIP.sub("_", str(text).lower()).strip("_")[:max_len]


def _in_zone_tree(path: Path, zone: str) -> bool:
    # Mirrors app/experiments.py _allowed_asset; keep in sync.
    if zone == "nsfw":
        return path.is_relative_to(ROOT / "nsfw" / "outputs")
    if not path.is_relative_to(ROOT / "outputs"):
        return False
    return not any("nsfw" in part.lower() for part in path.parts)


def _collect_assets(new_files: object, zone: str) -> tuple[list[dict], int]:
    """Hash the run's real renders into asset descriptors; returns (entries, skipped)."""
    root = ROOT.resolve()
    entries: list[dict] = []
    skipped = 0
    if not isinstance(new_files, list):
        return entries, skipped
    for raw in new_files:
        if not isinstance(raw, str) or not raw:
            skipped += 1
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_symlink():
            # Same refusal as experiments._register_asset: evidence must be the
            # real file, not a pointer that can be re-targeted later.
            skipped += 1
            continue
        try:
            resolved = candidate.resolve()
            rel = resolved.relative_to(root)
        except (OSError, ValueError):
            skipped += 1
            continue
        if not _in_zone_tree(resolved, zone) or not resolved.is_file():
            skipped += 1
            continue
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:  # streamed: renders can be tens of MB
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        entries.append({
            "path": rel.as_posix(),
            "sha256": digest.hexdigest(),
            "bytes": size,
        })
    return entries, skipped


def _merge_by_id(existing: object, key: str, new_items: list[dict]) -> list[dict]:
    """Id-keyed merge so re-recording the same job never duplicates entries."""
    merged: dict[str, dict] = {}
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, dict) and item.get(key):
                merged[str(item[key])] = item
    for item in new_items:
        merged[str(item[key])] = item
    return list(merged.values())


def _as_dict(value: object) -> dict | None:
    """Accept both raw DB rows (JSON strings) and _row_to_dict projections (dicts)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _record(job: dict, *, experiments_root: Path) -> Path | None:
    brief = _as_dict(job.get("brief")) or {}
    result = _as_dict(job.get("result"))
    if result is None or result.get("rc") != 0:
        return None

    tier = str(brief.get("tier") or result.get("intent") or "unknown")
    box = str(result.get("box") or brief.get("box") or "auto")
    zone = "nsfw" if brief.get("nsfw") is True else "sfw"
    entries, skipped = _collect_assets(result.get("new_files"), zone)
    if not entries:
        print(f"[job-experiments] job {job.get('id')}: no recordable assets "
              f"({skipped} skipped); manifest not written", file=sys.stderr)
        return None

    group = brief.get("group")
    group_slug = _slug(group, 48) if isinstance(group, str) and group else ""
    prefix = "exp_nsfw_" if zone == "nsfw" else "exp_"
    if group_slug:
        experiment_id = f"{prefix}{group_slug}"
    else:
        experiment_id = f"{prefix}job_{_slug(job.get('id'), 24)}"
    if not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"derived experiment_id is invalid: {experiment_id!r}")

    prompt_text = str(brief.get("prompt") or "").strip()
    seed = brief.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        seed = 0
    timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
    try:
        elapsed_s = max(0.0, float(timings.get("engine_s") or timings.get("total_s") or 0.0))
    except (TypeError, ValueError):
        elapsed_s = 0.0

    style = brief.get("style_lora")
    style_slug = ""
    if isinstance(style, str) and style.strip():
        style_slug = _slug(str(style).split(":", 1)[0].rsplit(".", 1)[0], 24)
    arm_id = _slug(f"{tier}_{style_slug}" if style_slug else tier, 63) or "unknown"
    arm = {
        "arm_id": arm_id,
        "label": (f"{tier} · {style}" if style_slug else tier)[:_MAX_LABEL],
        "model": f"studio-{tier}@{box}"[:_MAX_MODEL],
    }
    dimension = {
        "dimension_id": f"dim_{hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:12]}",
        "label": (prompt_text[:40] or "prompt")[:_MAX_LABEL],
        "prompt": prompt_text[:_MAX_DIMENSION_PROMPT],
        "criteria": [],
    }
    job_slug = _slug(job.get("id"), 24) or "unknown"
    new_samples = [{
        "sample_id": f"smp_{job_slug}_{index:02d}",
        "arm_id": arm_id,
        "dimension_id": dimension["dimension_id"],
        "seed": seed,
        "asset": asset,
        "elapsed_s": elapsed_s,
    } for index, asset in enumerate(entries)]

    exp_dir = experiments_root / experiment_id
    manifest_path = exp_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.is_file():
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = parsed
        except (OSError, json.JSONDecodeError):
            existing = {}

    stamp = datetime.fromtimestamp(time.time(), timezone.utc).strftime("%Y-%m-%d")
    title_core = prompt_text[:60] or tier
    if group_slug:
        title = f"[batch {tier}] {title_core}（{str(group)[:40]}）"
    else:
        title = f"[{tier}] {stamp} {title_core}"
    created = existing.get("created_at")
    if not isinstance(created, str) or not created:
        try:
            created = datetime.fromtimestamp(
                float(job.get("created_at") or time.time()), timezone.utc,
            ).isoformat()
        except (TypeError, ValueError, OSError):
            created = datetime.now(timezone.utc).isoformat()
    warnings = [w for w in (existing.get("warnings") or []) if isinstance(w, str)]
    if skipped:
        note = (f"{skipped} reported file(s) were outside the {zone} zone tree or "
                "missing at record time")
        if note not in warnings:
            warnings.append(note)

    manifest = {
        "schema": _SCHEMA,
        "experiment_id": experiment_id,
        "zone": zone,
        "title": title[:_MAX_TITLE],
        "status": "complete",
        "asset_availability": "local_only",
        "created_at": created,
        "arms": _merge_by_id(existing.get("arms"), "arm_id", [arm]),
        "dimensions": _merge_by_id(existing.get("dimensions"), "dimension_id", [dimension]),
        "samples": _merge_by_id(existing.get("samples"), "sample_id", new_samples),
        "warnings": warnings,
    }

    exp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=exp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, manifest_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return manifest_path


def record_finished_job(job: dict, *, experiments_root: Path | None = None) -> Path | None:
    """Best-effort manifest emission for a finished job; never raises."""
    root = Path(experiments_root) if experiments_root is not None else DEFAULT_EXPERIMENTS_ROOT
    try:
        return _record(job, experiments_root=root)
    except Exception as exc:  # recording must never break the queue
        print(f"[job-experiments] record failed for job {job.get('id')}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return None
