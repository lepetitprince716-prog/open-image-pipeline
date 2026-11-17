#!/usr/bin/env python3
"""Aggregate every gallery review verdict into one taste ledger.

The gallery review system writes a human verdict per batch under
``outputs/_gallery/review_batches/<batch>/`` as ``manifest.json`` + ``result.json``, but
nothing rolls them up. This script builds one JSONL ledger of every reviewed image with
its verdict, comment, hand-assigned issue tags, inferred generation line, and (when
resolvable) the generation recipe, then prints statistics.

Four other surfaces also carry human verdicts that never reached this ledger: the
atelier's pick/release, the gallery's star ratings, the Studio's per-job up/down, and the
per-image yes/no marks the /sessions/ and /moss/ pages append to the review-event ledger.
The ``--include-*`` flags read those sources (read-only, never written back) and map them
onto the same record shape, tagged by ``source``. An ingested item that a review batch
already judged keeps its record but gains ``superseded_by``, so the batch stays canonical
without anything being dropped.

Usage:
    python3 scripts/taste_ledger.py build [--out PATH]
        [--include-atelier] [--include-gallery] [--include-studio] [--include-review-events]
    python3 scripts/taste_ledger.py stats [--ledger PATH] [--by line|batch|tag|reason|month|source]
        [--include-superseded]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import sqlite3
from datetime import datetime, timezone

from config import OUTPUTS, ROOT

from review_batch import BATCH_RE, REVIEW_ROOT_PARTS

SCHEMA = "loramake-taste-ledger/v1"
DEFAULT_LEDGER = OUTPUTS / REVIEW_ROOT_PARTS[0] / "taste_ledger.jsonl"
DEFAULT_NAI_MANIFEST = OUTPUTS / "nai" / "manifest.jsonl"
DEFAULT_TAGS = ROOT / "configs" / "taste_comment_tags.json"
NAI_PARAM_KEYS = ("width", "height", "steps", "scale", "sampler", "seed", "noise_schedule")
META_KEYS = ("driving_model", "family", "base_ckpt", "lora", "lora_strength", "prompt", "seed")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")
REFERENCE_FRAGMENTS = ("references/", "/recommended/", "/reference")
NAI_FRAGMENTS = ("/nai/", "/outputs/nai", "nai_v5", "nai_v4", "novelai")
VERDICTS = ("love", "yes", "no", "pending")
# The verdicts a human actually clicked. "pending" is the rest of VERDICTS, and an
# absent or empty verdict means the same thing: nobody decided.
DECIDED_VERDICTS = frozenset(VERDICTS) - {"pending"}

SOURCE_REVIEW = "review"
DEFAULT_RATINGS = OUTPUTS / "_gallery" / "ratings.json"
DEFAULT_JOBS_DB = ROOT / "app" / "data" / "jobs.sqlite3"
DEFAULT_EXPERIMENTS_DB = ROOT / "app" / "data" / "experiments.sqlite3"
# The gallery rates 1-5; the ledger speaks love/yes/no. Five stars is the only rating the
# gallery treats as a keeper, one or two is an explicit rejection, and the middle is a pass.
# One line to change if the gallery's scale is ever reinterpreted.
STARS_TO_VERDICT = {5: "love", 4: "yes", 3: "yes", 2: "no", 1: "no"}
# The atelier, the Studio and the review-event pages all express a binary keep/reject, so
# they map to the two ends. None of them can express the ledger's "yes", and inventing it
# would read as a judgement the human never made.
ATELIER_VERDICTS = {"picked": "love", "released": "no"}
# Synthetic batch names for sources that have none. The colon is the point: BATCH_RE
# forbids it, so an ingested batch can never collide with a real review batch directory
# -- and `review_batches/studio` already exists, so the bare names are taken.
GALLERY_BATCH = "gallery:stars"
STUDIO_BATCH = "studio:jobs"
STUDIO_VERDICTS = {"up": "love", "down": "no"}
# A page's keep mark is the review-event decision "yes": the keep end, not the middle.
REVIEW_EVENT_VERDICTS = {"yes": "love", "no": "no"}


def _review_root() -> Path:
    return OUTPUTS / REVIEW_ROOT_PARTS[0] / REVIEW_ROOT_PARTS[1]


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    except json.JSONDecodeError as exc:
        print(f"warn: {path} is not valid JSON: {exc}", file=sys.stderr)
        return None


def _read_nai_manifest(nai_manifest: Path) -> dict[str, dict]:
    """Map image basename -> NAI record for one manifest.jsonl (empty when unreadable)."""
    index: dict[str, dict] = {}
    try:
        text = nai_manifest.read_text(encoding="utf-8")
    except OSError:
        return index
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        for file_path in record.get("files", []) or []:
            if isinstance(file_path, str):
                index[Path(file_path).name] = record
    return index


class NaiIndex:
    """Lazy NAI recipe lookup.

    nai_gen.py appends one manifest.jsonl per output directory, so the record for an image
    normally sits beside it. Look there first (cached per directory), then fall back to the
    manifests under the configured root (a file, or a directory searched recursively).
    """

    def __init__(self, nai_manifest: Path) -> None:
        self._dirs: dict[Path, dict[str, dict]] = {}
        self._fallback: dict[str, dict] = {}
        if nai_manifest.is_file():
            self._fallback.update(_read_nai_manifest(nai_manifest))
        elif nai_manifest.is_dir():
            for path in sorted(nai_manifest.rglob("manifest.jsonl")):
                self._fallback.update(_read_nai_manifest(path))

    def lookup(self, image_path: Path) -> dict | None:
        directory = image_path.parent
        if directory not in self._dirs:
            manifest = directory / "manifest.jsonl"
            self._dirs[directory] = _read_nai_manifest(manifest) if manifest.is_file() else {}
        record = self._dirs[directory].get(image_path.name)
        if record is None:
            record = self._fallback.get(image_path.name)
        return record


def _load_tags(tags_path: Path) -> dict[tuple[str, int], list[str]]:
    """Load hand-assigned issue tags keyed by (batch, i)."""
    data = _load_json(tags_path)
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return {}
    out: dict[tuple[str, int], list[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        batch = entry.get("batch")
        i = entry.get("i")
        tags = entry.get("tags")
        if not isinstance(batch, str) or not isinstance(i, int) or not isinstance(tags, list):
            continue
        out[(batch, i)] = [t for t in tags if isinstance(t, str)]
    return out


def _verdict_for(i: int, love: set[int], yes: set[int], no: set[int]) -> str:
    if i in love:
        return "love"
    if i in yes:
        return "yes"
    if i in no:
        return "no"
    return "pending"


def _read_sidecar(image_path: Path) -> dict | None:
    side = image_path.with_name(image_path.name + ".meta.json")
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _nai_character_prompts(params: dict) -> list[str]:
    """Character captions from the V4 prompt block, when the request carried any."""
    caption = (params.get("v4_prompt") or {}).get("caption") if isinstance(params, dict) else None
    if not isinstance(caption, dict):
        return []
    prompts: list[str] = []
    for entry in caption.get("char_captions") or []:
        text = entry.get("char_caption") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            prompts.append(text)
    return prompts


def _nai_recipe(record: dict) -> dict:
    params = record.get("parameters") if isinstance(record.get("parameters"), dict) else {}
    subset = {k: params[k] for k in NAI_PARAM_KEYS if k in params}
    return {
        "source": "nai_manifest",
        "model": record.get("model"),
        "prompt": record.get("prompt"),
        "character_prompts": _nai_character_prompts(params),
        "negative": record.get("negative"),
        "parameters": subset,
    }


def _sidecar_recipe(sidecar: dict) -> dict:
    return {"source": "meta_sidecar", **{k: sidecar.get(k) for k in META_KEYS}}


def _infer_line(
    name: str, path: str, sidecar: dict | None, media: str = "image"
) -> tuple[str, str]:
    """Return (line, line_hint) using the documented precedence."""
    low_name = name.lower()
    low_path = path.lower()
    if media == "video" or low_name.endswith(VIDEO_SUFFIXES):
        return "video", "media/extension"
    if any(k in low_path for k in REFERENCE_FRAGMENTS):
        return "reference", "path reference fragment"
    if low_name.startswith("nai_") or any(k in low_path for k in NAI_FRAGMENTS):
        return "nai", "name/path nai fragment"
    if sidecar:
        # moss_gen records the agent as driving_model, so its engine is what names the line.
        if sidecar.get("engine") == "moss_gen":
            return "moss", "sidecar engine"
        family = str(sidecar.get("family") or "").lower()
        driving = str(sidecar.get("driving_model") or "").lower()
        blob = f"{family} {driving}"
        if "anima" in blob:
            return "anima", "sidecar family/driving_model"
        if any(k in blob for k in ("illustrious", "ilustmix", "noobai", "wai")):
            return "illustrious", "sidecar family/driving_model"
        if "krea" in blob:
            return "krea", "sidecar family/driving_model"
    if "/moss/" in low_path:
        return "moss", "path moss"
    if "anima" in low_path:
        return "anima", "path anima"
    if any(k in low_path for k in ("ilustmix", "illustrious", "noobai")):
        return "illustrious", "path illustrious"
    if "civitai" in low_path:
        return "civitai", "path civitai"
    if "gpt-image" in low_path or "gptimage" in low_path:
        return "gpt-image", "path gpt-image"
    if "krea" in low_path:
        return "krea", "path krea"
    return "other", "fallback"


def _exists(source_path: str | None, rel_path: str) -> bool:
    if source_path and Path(source_path).exists():
        return True
    return (ROOT / rel_path).exists()


def _build_record(
    batch: str, batch_time: str, item: dict, verdict: str, comment: str | None,
    issue_tags: list[str], reasons: list[str], nai_index: NaiIndex,
) -> dict:
    name = item.get("name", "")
    path = item.get("path", "")
    source_path = item.get("source_path")
    media = item.get("media", "image")
    line, line_hint = _infer_line(name, path, None, media)
    image_path = Path(source_path) if source_path else (ROOT / path)
    recipe: dict | None = None
    if line == "nai":
        nai_record = nai_index.lookup(image_path)
        if nai_record is not None:
            recipe = _nai_recipe(nai_record)
    elif line not in ("video", "reference"):
        sidecar = _read_sidecar(image_path)
        if sidecar is not None:
            recipe = _sidecar_recipe(sidecar)
            line, line_hint = _infer_line(name, path, sidecar, media)
    return {
        "schema": SCHEMA,
        "batch": batch,
        "batch_time": batch_time,
        "i": item.get("i"),
        "name": name,
        "sha256": item.get("sha256"),
        "path": path,
        "source_path": source_path,
        "exists": _exists(source_path, path),
        "verdict": verdict,
        "comment": comment,
        "issue_tags": issue_tags,
        "reasons": reasons,
        "line": line,
        "line_hint": line_hint,
        "recipe": recipe,
        "caption": item.get("caption", ""),
        "source": SOURCE_REVIEW,
    }


def _process_batch(
    batch_dir: Path, batch: str, tags: dict[tuple[str, int], list[str]],
    nai_index: NaiIndex, seen_tag_keys: set[tuple[str, int]],
) -> list[dict] | None:
    """Return records for one batch, or None if the batch must be skipped."""
    manifest = _load_json(batch_dir / "manifest.json")
    if manifest is None:
        print(f"warn: skipping batch {batch}: manifest missing or unreadable", file=sys.stderr)
        return None
    if not isinstance(manifest, dict):
        print(f"warn: skipping batch {batch}: manifest not an object", file=sys.stderr)
        return None
    if manifest.get("mode") == "view":
        return []
    items = manifest.get("items")
    if not isinstance(items, list):
        print(f"warn: skipping batch {batch}: manifest has no items list", file=sys.stderr)
        return None
    batch_time = manifest.get("created_at", "")
    result = _load_json(batch_dir / "result.json")
    if not isinstance(result, dict):
        result = {}

    def _idx_set(key: str) -> set[int]:
        values = result.get(key)
        return {v for v in values if isinstance(v, int)} if isinstance(values, list) else set()

    love, yes, no = _idx_set("l"), _idx_set("y"), _idx_set("n")
    raw_comments = result.get("c") if isinstance(result.get("c"), dict) else {}
    raw_reasons = result.get("r") if isinstance(result.get("r"), dict) else {}

    records: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("i"), int):
            continue
        i = item["i"]
        verdict = _verdict_for(i, love, yes, no)
        comment = None
        for key, text in raw_comments.items():
            if str(key).lstrip("-").isdigit() and int(key) == i and isinstance(text, str):
                comment = text
                break
        reasons: list[str] = []
        for key, value in raw_reasons.items():
            if str(key).lstrip("-").isdigit() and int(key) == i and isinstance(value, list):
                reasons = [r for r in value if isinstance(r, str)]
                break
        tag_key = (batch, i)
        seen_tag_keys.add(tag_key)
        issue_tags = list(tags.get(tag_key, []))
        records.append(_build_record(
            batch, batch_time, item, verdict, comment, issue_tags, reasons, nai_index,
        ))
    return records


def _iso(epoch: float) -> str:
    """A ledger timestamp from a POSIX time, in the same shape review batches use."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_time(path: Path) -> str:
    """The source file's own mtime, used when a source records no per-item time."""
    try:
        return _iso(path.stat().st_mtime)
    except OSError:
        return ""


def _ingested_record(
    source: str, batch: str, batch_time: str, i: int, rel_path: str,
    verdict: str, nai_index: NaiIndex, comment: str | None = None,
) -> dict:
    """One ledger record from a source with no review batch behind it.

    Goes through ``_build_record`` so line inference, recipe lookup and the existence
    check stay in one place; only the provenance fields differ.
    """
    path = Path(rel_path)
    item = {
        "i": i,
        "name": path.name,
        "path": rel_path,
        "source_path": rel_path if path.is_absolute() else None,
        "media": "image",
        "caption": "",
    }
    record = _build_record(batch, batch_time, item, verdict, comment, [], [], nai_index)
    record["source"] = source
    return record


def _was_picked(log: list[str], slot: str, path: str) -> bool:
    """True when the atelier log still shows a human picking ``path`` into ``slot``.

    ``pick`` writes "picked <slot>: <filename>"; ``start_fill``'s auto-adopt writes
    "<view> view ready" instead, which is how the two are told apart.
    """
    # The marker must end the line: a substring match would let a log entry for
    # "a.png.webp" in the same slot mark a sheet path of "a.png" as picked.
    marker = f"picked {slot}: {Path(path).name}"
    return any(line.rstrip().endswith(marker) for line in log)


def atelier_records(characters_root: Path, nai_index: NaiIndex) -> list[dict]:
    """Ledger records from every character atelier state file.

    A released image is a rejection even if it was picked first, because ``release``
    clears the sheet slot as it files the path under ``rejected``. Candidates the human
    has neither picked nor released stay ``pending``, exactly as an undecided batch item
    does -- dropping them would report the atelier's love rate as 100%.

    A filled sheet slot is not by itself a pick: ``start_fill`` adopts the first image it
    produces into any slot still empty, with no human involved. Only a slot the log still
    shows as picked counts as love. The log keeps its last LOG_KEEP lines, so an old pick
    reads as ``pending`` once its line ages out -- that under-reports love, which is the
    safe direction. Recording a machine's auto-adopt as love would invent a judgement.
    """
    from loramake.character_atelier import SLOTS

    records: list[dict] = []
    if not characters_root.is_dir():
        return records
    for state_path in sorted(characters_root.glob("*/atelier.json")):
        state = _load_json(state_path)
        if not isinstance(state, dict):
            print(f"warn: skipping atelier state {state_path}: not an object", file=sys.stderr)
            continue
        card_id = state.get("id") or state_path.parent.name
        batch, batch_time = f"atelier:{card_id}", _file_time(state_path)
        rejected = {p for p in (state.get("rejected") or []) if isinstance(p, str)}
        sheet = state.get("sheet") if isinstance(state.get("sheet"), dict) else {}
        log = [line for line in (state.get("log") or []) if isinstance(line, str)]
        # Everything on the sheet is in the ledger; the log only decides whether a slot is
        # a pick or a machine's auto-adopt, and an auto-adopt is simply still undecided.
        on_sheet = {slot: sheet[slot] for slot in SLOTS if isinstance(sheet.get(slot), str)}
        picked = {path for slot, path in on_sheet.items() if _was_picked(log, slot, path)}
        candidates: set[str] = set()
        raw_candidates = state.get("candidates") if isinstance(state.get("candidates"), dict) else {}
        for items in raw_candidates.values():
            if isinstance(items, list):
                candidates |= {c["path"] for c in items
                               if isinstance(c, dict) and isinstance(c.get("path"), str)}
        for i, path in enumerate(sorted(rejected | set(on_sheet.values()) | candidates)):
            if path in rejected:
                verdict = ATELIER_VERDICTS["released"]
            elif path in picked:
                verdict = ATELIER_VERDICTS["picked"]
            else:
                verdict = "pending"
            records.append(_ingested_record("atelier", batch, batch_time, i, path, verdict, nai_index))
    return records


def gallery_records(ratings_path: Path, nai_index: NaiIndex) -> list[dict]:
    """Ledger records from the gallery's star ratings, mapped through STARS_TO_VERDICT."""
    ratings = _load_json(ratings_path)
    if ratings is None:
        return []
    if not isinstance(ratings, dict):
        print(f"warn: skipping ratings {ratings_path}: not an object", file=sys.stderr)
        return []
    batch_time = _file_time(ratings_path)
    records: list[dict] = []
    for i, (rel_path, stars) in enumerate(sorted(ratings.items())):
        verdict = STARS_TO_VERDICT.get(stars) if isinstance(stars, int) else None
        if verdict is None:
            print(f"warn: skipping rating {rel_path!r}: stars={stars!r} is not 1-5", file=sys.stderr)
            continue
        records.append(_ingested_record(
            "gallery", GALLERY_BATCH, batch_time, i, rel_path, verdict, nai_index,
            comment=f"{stars} stars",
        ))
    return records


def studio_records(db_path: Path, nai_index: NaiIndex) -> list[dict]:
    """Ledger records from the Studio's per-job up/down verdicts.

    The database is opened read-only: this is an ingest, and the Studio owns that file.
    A missing database is not an error -- the Studio simply has not run on this machine.
    """
    from app.jobs import result_image_candidates

    if not db_path.exists():
        print(f"warn: no Studio database at {db_path}; skipping", file=sys.stderr)
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as exc:
        print(f"warn: cannot open Studio database {db_path}: {exc}", file=sys.stderr)
        return []
    records: list[dict] = []
    try:
        rows = conn.execute(
            "SELECT r.job_id, r.verdict, r.note, r.updated_at, j.result "
            "FROM job_reviews r JOIN jobs j ON j.id = r.job_id ORDER BY r.updated_at"
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"warn: cannot read job_reviews from {db_path}: {exc}", file=sys.stderr)
        return []
    finally:
        conn.close()
    for i, (job_id, verdict, note, updated_at, raw_result) in enumerate(rows):
        mapped = STUDIO_VERDICTS.get(verdict)
        if mapped is None:
            print(f"warn: skipping job {job_id}: unknown verdict {verdict!r}", file=sys.stderr)
            continue
        try:
            result = json.loads(raw_result) if raw_result else {}
        except json.JSONDecodeError:
            result = {}
        # The first image the job names, not the first that still exists: a verdict on an
        # image since archived is still a verdict, and `exists` records the difference.
        candidates = result_image_candidates(result) if isinstance(result, dict) else []
        if not candidates:
            print(f"warn: skipping job {job_id}: its result names no image", file=sys.stderr)
            continue
        records.append(_ingested_record(
            "studio", STUDIO_BATCH, _iso(updated_at), i, candidates[0], mapped, nai_index,
            comment=note or None,
        ))
    return records


def review_event_records(db_path: Path, nai_index: NaiIndex) -> list[dict]:
    """Ledger records from the per-image yes/no marks in the review-event ledger.

    The /sessions/ and /moss/ pages append one review event per mark to the experiments
    database and never delete one: a withdrawal is a comment event that supersedes the
    mark. A sample's verdict is its current event as the app projects it, the latest event
    by any actor on the experiment's current evidence revision. Only a human yes or no makes
    a record. A comment, the withdrawal included, reads as unmarked, as the pages render it,
    and an agent's mark is not the user's taste.

    Only ``sample`` targets count: the Results page can also mark the same image as an
    ``asset``, and reading both would count it twice. Only the sfw zone is read, because the
    ledger lives in the SFW gallery. An image marked here and judged on its Studio job keeps
    both records; ``stats --by source`` tells them apart.
    """
    # The app's projection is the one definition of a target's current event. Importing it
    # needs the app's environment, as --include-studio already does. The database is opened
    # read-only here, never through the app's own connect, which migrates and writes.
    from app.experiments import _review_projection

    if not db_path.exists():
        print(f"warn: no experiments database at {db_path}; skipping", file=sys.stderr)
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as exc:
        print(f"warn: cannot open experiments database {db_path}: {exc}", file=sys.stderr)
        return []
    conn.row_factory = sqlite3.Row
    marks: list[tuple[str, str, int, str, str]] = []
    try:
        sources = conn.execute(
            "SELECT experiment_id, source_revision, payload_json FROM experiment_sources "
            "WHERE zone = 'sfw' AND payload_json IS NOT NULL ORDER BY experiment_id"
        ).fetchall()
        asset_paths = {
            (row["experiment_id"], row["asset_id"]): row["path_rel"]
            for row in conn.execute(
                "SELECT experiment_id, asset_id, path_rel FROM experiment_assets WHERE zone = 'sfw'"
            )
        }
        for source in sources:
            experiment_id = source["experiment_id"]
            projection = _review_projection(conn, experiment_id, source["source_revision"])
            current = projection["current_by_target"]
            for i, sample in enumerate(json.loads(source["payload_json"]).get("samples") or []):
                event = current.get(f"sample:{sample['sample_id']}")
                if event is None or event["actor"]["kind"] != "human":
                    continue
                verdict = REVIEW_EVENT_VERDICTS.get(event["decision"])
                if verdict is None:
                    continue
                path = asset_paths.get((experiment_id, sample["asset"]["asset_id"]))
                if path is None:
                    print(f"warn: skipping {sample['sample_id']}: its asset is not registered",
                          file=sys.stderr)
                    continue
                marks.append((experiment_id, event["created_at"], i, path, verdict))
    except sqlite3.Error as exc:
        print(f"warn: cannot read review events from {db_path}: {exc}", file=sys.stderr)
        return []
    finally:
        conn.close()
    return [
        _ingested_record("review-events", f"experiment:{experiment_id}", created_at, i, path,
                         verdict, nai_index)
        for experiment_id, created_at, i, path, verdict in marks
    ]


def _resolved_image(record: dict) -> str | None:
    """A record's image as one absolute path, so two sources can be compared.

    Both sides must anchor a relative path to the repo root. Resolving one of them
    against the process cwd instead would make the comparison depend on where the
    command was run, and the match would silently never happen outside the repo root.
    """
    key = record.get("source_path") or record.get("path")
    if not key:
        return None
    path = Path(key)
    return str((path if path.is_absolute() else ROOT / path).resolve(strict=False))


def mark_superseded(records: list[dict]) -> None:
    """Tag every ingested record whose image a review batch already judged.

    The review batch is canonical: it is the only source carrying issue tags and reasons.
    The ingested record is kept rather than dropped, because dropping it would silently
    rewrite what a source says -- this ingest is read-only in both directions.
    """
    judged: dict[str, str] = {}
    for record in records:
        # An undecided batch row cannot be the canonical one. Letting it supersede would
        # hide a real gallery or Studio verdict behind an item nobody judged. Testing
        # against DECIDED_VERDICTS rather than the literal "pending" also covers a row
        # that reaches here with no verdict at all.
        if record.get("source") != SOURCE_REVIEW:
            continue
        if record.get("verdict") not in DECIDED_VERDICTS:
            continue
        key = _resolved_image(record)
        if key:
            judged.setdefault(key, record.get("batch", ""))
    for record in records:
        if record.get("source") == SOURCE_REVIEW:
            continue
        key = _resolved_image(record)
        batch = judged.get(key) if key else None
        if batch:
            record["superseded_by"] = batch


def build_ledger(review_root: Path, nai_manifest: Path, tags_path: Path) -> list[dict]:
    """Build the full ledger from all review batches under ``review_root``.

    ``nai_manifest`` is the fallback NAI recipe source: a manifest.jsonl file, or a directory
    searched recursively for them. Manifests beside the reviewed images are always tried first.
    """
    nai_index = NaiIndex(nai_manifest)
    tags = _load_tags(tags_path)
    seen_tag_keys: set[tuple[str, int]] = set()
    records: list[dict] = []
    if not review_root.is_dir():
        return records
    for entry in sorted(review_root.iterdir()):
        if not entry.is_dir():
            continue
        batch = entry.name
        if not BATCH_RE.fullmatch(batch):
            continue
        batch_records = _process_batch(entry, batch, tags, nai_index, seen_tag_keys)
        if batch_records is None:
            continue
        records.extend(batch_records)
    for key in set(tags) - seen_tag_keys:
        print(f"warn: tags entry has no matching batch item: {key}", file=sys.stderr)
    records.sort(key=lambda r: (r.get("batch_time") or "", r.get("batch") or "", r.get("i") or 0))
    return records


def _rates(love: int, yes: int, no: int, decided: int) -> tuple[float, float]:
    if decided <= 0:
        return 0.0, 0.0
    return love / decided, no / decided


def _stats_lineish(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, dict[str, int]] = {}
    for r in rows:
        k = r[key]
        g = groups.setdefault(k, {"total": 0, "love": 0, "yes": 0, "no": 0, "pending": 0, "comments": 0})
        g[r["verdict"]] += 1
        if r["verdict"] != "pending":
            g["total"] += 1
        if r.get("comment"):
            g["comments"] += 1
    out: list[dict] = []
    for k in sorted(groups):
        g = groups[k]
        love_rate, no_rate = _rates(g["love"], g["yes"], g["no"], g["total"])
        out.append({
            key: k, "total": g["total"], "love": g["love"], "yes": g["yes"], "no": g["no"],
            "love_rate": love_rate, "no_rate": no_rate, "comments": g["comments"],
        })
    return out


def _stats_tag(rows: list[dict]) -> list[dict]:
    """Count issue tags across every verdict.

    Pending records are included on purpose: a free-text comment is a taste signal even when
    the reviewer never clicked a verdict, and in practice most commented items are pending.
    """
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        for tag in r.get("issue_tags") or []:
            g = counts.setdefault(tag, {v: 0 for v in VERDICTS})
            g[r["verdict"]] += 1
    out: list[dict] = []
    for tag in sorted(counts):
        g = counts[tag]
        out.append({"tag": tag, "total": sum(g.values()), **g})
    return out


def _stats_reason(rows: list[dict]) -> list[dict]:
    """Count one-click rejection reasons across ``no`` records only."""
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("verdict") != "no":
            continue
        for reason in r.get("reasons") or []:
            counts[reason] = counts.get(reason, 0) + 1
    return [{"reason": reason, "count": counts[reason]} for reason in sorted(counts)]


def compute_stats(records: list[dict], by: str) -> list[dict]:
    """Compute aggregate stats from ledger records."""
    if by == "tag":
        return _stats_tag(records)
    if by == "reason":
        return _stats_reason(records)
    if by == "line":
        return _stats_lineish(records, "line")
    if by == "batch":
        return _stats_lineish(records, "batch")
    if by == "source":
        return _stats_lineish([{**r, "source": r.get("source", SOURCE_REVIEW)} for r in records], "source")
    if by == "month":
        month_rows = [{**r, "month": (r.get("batch_time") or "")[:7]} for r in records]
        return _stats_lineish(month_rows, "month")
    raise ValueError(f"unknown --by value: {by}")


def _fmt_rate(x: float) -> str:
    return f"{x:.2f}"


def _print_table(rows: list[dict], by: str) -> None:
    if by == "tag":
        print(f"{'tag':<20} {'total':>6} {'love':>5} {'yes':>5} {'no':>5} {'pending':>8}")
        for r in rows:
            print(f"{r['tag']:<20} {r['total']:>6} {r['love']:>5} {r['yes']:>5} {r['no']:>5} "
                  f"{r['pending']:>8}")
        return
    if by == "reason":
        print(f"{'reason':<20} {'count':>6}")
        for r in rows:
            print(f"{r['reason']:<20} {r['count']:>6}")
        return
    print(f"{by:<40} {'total':>6} {'love':>5} {'yes':>5} {'no':>5} "
          f"{'love_rate':>9} {'no_rate':>8} {'comments':>9}")
    for r in rows:
        print(f"{str(r[by]):<40} {r['total']:>6} {r['love']:>5} {r['yes']:>5} {r['no']:>5} "
              f"{_fmt_rate(r['love_rate']):>9} {_fmt_rate(r['no_rate']):>8} {r['comments']:>9}")


def _cmd_build(args: argparse.Namespace) -> int:
    review_root = _review_root()
    records = build_ledger(review_root, DEFAULT_NAI_MANIFEST, DEFAULT_TAGS)
    if (args.include_atelier or args.include_gallery or args.include_studio
            or args.include_review_events):
        nai_index = NaiIndex(DEFAULT_NAI_MANIFEST)
        if args.include_atelier:
            records.extend(atelier_records(OUTPUTS / "characters", nai_index))
        if args.include_gallery:
            records.extend(gallery_records(DEFAULT_RATINGS, nai_index))
        if args.include_studio:
            records.extend(studio_records(DEFAULT_JOBS_DB, nai_index))
        if args.include_review_events:
            records.extend(review_event_records(DEFAULT_EXPERIMENTS_DB, nai_index))
        mark_superseded(records)
        records.sort(key=lambda r: (r.get("batch_time") or "", r.get("batch") or "", r.get("i") or 0))
    out_path = Path(args.out) if args.out else DEFAULT_LEDGER
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    counts = {"love": 0, "yes": 0, "no": 0, "pending": 0, "comments": 0}
    batches: set[str] = set()
    by_source: dict[str, int] = {}
    superseded = 0
    for r in records:
        batches.add(r["batch"])
        counts[r["verdict"]] += 1
        if r.get("comment"):
            counts["comments"] += 1
        source = r.get("source", SOURCE_REVIEW)
        by_source[source] = by_source.get(source, 0) + 1
        if r.get("superseded_by"):
            superseded += 1
    print(f"batches={len(batches)} items={len(records)} "
          f"love={counts['love']} yes={counts['yes']} no={counts['no']} "
          f"pending={counts['pending']} comments={counts['comments']}")
    print("sources: " + " ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
          + f" superseded={superseded}")
    print(f"wrote {out_path}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger) if args.ledger else DEFAULT_LEDGER
    if not ledger_path.exists():
        print(f"taste_ledger.py: error: ledger not found: {ledger_path}", file=sys.stderr)
        return 1
    records: list[dict] = []
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"taste_ledger.py: error: cannot read ledger: {exc}", file=sys.stderr)
        return 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not args.include_superseded:
        records = [r for r in records if not r.get("superseded_by")]
    rows = compute_stats(records, args.by)
    _print_table(rows, args.by)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate gallery review verdicts into a taste ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="build the JSONL ledger")
    build_p.add_argument("--out", help="output ledger path")
    build_p.add_argument("--include-atelier", action="store_true",
                         help="also ingest atelier pick/release verdicts (read-only)")
    build_p.add_argument("--include-gallery", action="store_true",
                         help="also ingest gallery star ratings (read-only)")
    build_p.add_argument("--include-studio", action="store_true",
                         help="also ingest Studio per-job up/down verdicts (read-only)")
    build_p.add_argument("--include-review-events", action="store_true",
                         help="also ingest the per-image yes/no marks of /sessions/ and "
                              "/moss/ (read-only)")
    build_p.set_defaults(func=_cmd_build)
    stats_p = sub.add_parser("stats", help="print statistics from the ledger")
    stats_p.add_argument("--ledger", help="ledger path")
    stats_p.add_argument("--by", choices=("line", "batch", "tag", "reason", "month", "source"),
                         default="line")
    stats_p.add_argument("--include-superseded", action="store_true",
                         help="also count ingested items a review batch already judged")
    stats_p.set_defaults(func=_cmd_stats)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
