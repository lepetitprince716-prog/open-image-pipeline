#!/usr/bin/env python3
"""Strict offline Phase-C evaluation for Anima Step 2a.

``plan`` deterministically expands the checked-in bank into the exact
prompt x model x seed XY matrix.  ``score`` accepts only a complete hash-bound
image manifest with checkpoint/runner/prompt/render provenance, reuses the
existing WD14/QC/CCIP bridges (or manifest-embedded results), and publishes an
atomic no-clobber evidence directory:

    metrics.json  evidence.json  RESULTS.md  contact_sheet.png

Manifest paths are relative to the manifest directory and must equal each
cell's planned output.  Every entry needs ``cell_id/model_id/prompt_id/seed``
plus ``path/sha256/bytes`` and the generation-contract hashes. Precomputed results live under
``scores.wd14``, ``scores.qc``, and ``scores.ccip`` using the existing bridge
shapes and require exact input hash binding. Any missing/extra cell, stale
hash, analyzer failure, partial CCIP
matrix, or missing contact-sheet slot is fatal.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
DEFAULT_BANK = REPO_ROOT / "configs/anima_step2a_eval_bank.json"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from consistency_check import DEFAULT_VENV_PY, ccip_one  # noqa: E402
from qc_score import flags_for, qc_one  # noqa: E402


BANK_SCHEMA = "anima_step2a_eval_bank_v1"
PLAN_SCHEMA = "anima_step2a_eval_plan_v1"
IMAGE_MANIFEST_SCHEMA = "anima_step2a_eval_images_v1"
METRICS_SCHEMA = "anima_step2a_eval_metrics_v1"
EVIDENCE_SCHEMA = "anima_step2a_eval_evidence_v1"
RATINGS = ("general", "sensitive", "questionable", "explicit")
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHEET_ROW_LABEL_W = 240
SHEET_HEADER_H = 44


class EvalContractError(ValueError):
    pass


def _render_eval_sheet(
    paths: list[str],
    labels: list[dict[str, Any]],
    out: str,
    *,
    cols: int,
    thumb: int,
) -> str:
    """Reuse contact_sheet tiles, then add readable XY row/column identities."""

    from PIL import Image, ImageDraw, ImageFont
    from contact_sheet import PAD, build_sheet

    if len(paths) != len(labels):
        raise EvalContractError("contact-sheet paths/labels length mismatch")
    build_sheet(paths, out, cols=cols, thumb=thumb, label=False)
    with Image.open(out) as opened:
        tile_sheet = opened.convert("RGB")
    cell_w, cell_h = thumb + 2 * PAD, thumb + 2 * PAD
    sheet = Image.new(
        "RGB",
        (SHEET_ROW_LABEL_W + tile_sheet.width, SHEET_HEADER_H + tile_sheet.height),
        (24, 24, 28),
    )
    sheet.paste(tile_sheet, (SHEET_ROW_LABEL_W, SHEET_HEADER_H))
    draw = ImageDraw.Draw(sheet)
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, 12)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()

    first_row = labels[:cols]
    for column, label in enumerate(first_row):
        header = f"{label['model_id']}\nseed {label['seed']}"
        bbox = draw.multiline_textbbox((0, 0), header, font=font, spacing=1)
        if bbox[2] - bbox[0] > cell_w - 8:
            raise EvalContractError("contact-sheet column label does not fit")
        draw.multiline_text(
            (SHEET_ROW_LABEL_W + column * cell_w + 4, 3),
            header,
            fill=(235, 235, 235),
            font=font,
            spacing=1,
        )
    rows = math.ceil(len(labels) / cols)
    for row in range(rows):
        label = labels[row * cols]["prompt_id"]
        bbox = draw.textbbox((0, 0), label, font=font)
        if bbox[2] - bbox[0] > SHEET_ROW_LABEL_W - 12:
            raise EvalContractError("contact-sheet prompt row label does not fit")
        draw.text(
            (6, SHEET_HEADER_H + row * cell_h + 6),
            label,
            fill=(235, 235, 235),
            font=font,
        )
    for slot, label in enumerate(labels):
        row, column = divmod(slot, cols)
        x0 = SHEET_ROW_LABEL_W + column * cell_w
        y0 = SHEET_HEADER_H + row * cell_h
        draw.rectangle((x0 + 2, y0 + 2, x0 + 34, y0 + 18), fill=(24, 24, 28))
        draw.text(
            (x0 + 4, y0 + 3),
            f"[{slot}]",
            fill=(235, 235, 235),
            font=font,
        )
        if (
            label["prompt_id"] != labels[row * cols]["prompt_id"]
            or label["model_id"] != first_row[column]["model_id"]
            or label["seed"] != first_row[column]["seed"]
        ):
            raise EvalContractError("contact-sheet labels do not form a prompt/model/seed XY grid")
    sheet.save(out)
    return out


def build_contact_sheet(
    paths: list[str],
    labels: list[dict[str, Any]],
    out: str,
    *,
    cols: int,
    thumb: int,
) -> str:
    """Public seam; plan remains usable without importing Pillow."""

    return _render_eval_sheet(paths, labels, out, cols=cols, thumb=thumb)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise EvalContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvalContractError(f"{label} has non-finite number {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalContractError(f"{label} is not readable strict UTF-8 JSON: {path}: {exc}") from exc
    return data, raw


def _object(value: Any, label: str, required: set[str], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalContractError(f"{label} must be an object")
    missing, extra = required - value.keys(), value.keys() - allowed
    if missing:
        raise EvalContractError(f"{label} missing: {', '.join(sorted(missing))}")
    if extra:
        raise EvalContractError(f"{label} unknown: {', '.join(sorted(extra))}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalContractError(f"{label} must be non-empty text")
    return value


def _slug(value: Any, label: str) -> str:
    value = _text(value, label)
    if not SLUG.fullmatch(value):
        raise EvalContractError(f"{label} is not a safe slug")
    return value


def _number(value: Any, label: str, low: float = 0.0, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalContractError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        raise EvalContractError(f"{label} is outside [{low}, {high or 'inf'}]")
    return value


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvalContractError(f"{label} must be a non-negative integer")
    return value


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _write_noclobber(path: Path, raw: bytes) -> None:
    """Hard-link publication gives atomic create-if-absent semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _tag(value: str) -> str:
    return re.sub(r"_+", "_", value.strip().lower().replace(" ", "_"))


def validate_bank(raw: Any) -> dict[str, Any]:
    keys = {
        "schema_version", "bank_id", "description", "models", "seeds", "render",
        "defaults", "contact_sheet", "ccip", "prompts",
    }
    bank = _object(raw, "bank", keys, keys)
    if bank["schema_version"] != BANK_SCHEMA:
        raise EvalContractError(f"bank schema must be {BANK_SCHEMA}")

    models: list[dict[str, str]] = []
    seen_models: set[str] = set()
    model_keys = {
        "id", "label", "artifact", "role", "expected_sha256", "expected_bytes",
    }
    for index, value in enumerate(bank["models"] if isinstance(bank["models"], list) else []):
        item = _object(value, f"models[{index}]", model_keys, model_keys)
        model_id = _slug(item["id"], f"models[{index}].id")
        if model_id in seen_models or item["role"] not in {"baseline", "candidate"}:
            raise EvalContractError(f"invalid/duplicate model at index {index}")
        expected_sha = item["expected_sha256"]
        expected_bytes = item["expected_bytes"]
        if expected_sha is not None and (
            not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha)
        ):
            raise EvalContractError(f"models[{index}].expected_sha256 must be null or SHA-256")
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise EvalContractError(f"models[{index}].expected_bytes must be null or positive")
        if item["role"] == "baseline" and (expected_sha is None or expected_bytes is None):
            raise EvalContractError("baseline model must freeze expected SHA-256 and byte count")
        if item["role"] == "candidate" and (
            expected_sha is not None or expected_bytes is not None
        ):
            raise EvalContractError("candidate identity is supplied by the image manifest")
        seen_models.add(model_id)
        models.append({
            "id": model_id,
            "label": _text(item["label"], f"models[{index}].label"),
            "artifact": _text(item["artifact"], f"models[{index}].artifact"),
            "role": item["role"],
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
        })
    baseline = [model["id"] for model in models if model["role"] == "baseline"]
    candidates = [model["id"] for model in models if model["role"] == "candidate"]
    if len(baseline) != 1 or not candidates:
        raise EvalContractError("bank needs exactly one baseline and at least one candidate")

    seeds = bank["seeds"]
    if (not isinstance(seeds, list) or not 2 <= len(seeds) <= 8
            or any(isinstance(seed, bool) or not isinstance(seed, int)
                   or not 0 <= seed < 2**63 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise EvalContractError("bank.seeds must be 2-8 unique non-negative integers")

    render_keys = {
        "runner", "runner_sha256", "width", "height", "steps", "cfg", "sampler",
        "scheduler", "shift", "no_scaffold", "no_wildcard", "no_trigger_inject",
        "turbo", "content_loras", "clip_name", "clip_type", "vae_name",
    }
    render = _object(bank["render"], "render", render_keys, render_keys)
    render = {
        "runner": _text(render["runner"], "render.runner"),
        "runner_sha256": _text(render["runner_sha256"], "render.runner_sha256"),
        "clip_name": _text(render["clip_name"], "render.clip_name"),
        "clip_type": _text(render["clip_type"], "render.clip_type"),
        "vae_name": _text(render["vae_name"], "render.vae_name"),
        "width": _count(render["width"], "render.width"),
        "height": _count(render["height"], "render.height"),
        "steps": _count(render["steps"], "render.steps"),
        "cfg": _number(render["cfg"], "render.cfg"),
        "sampler": _text(render["sampler"], "render.sampler"),
        "scheduler": _text(render["scheduler"], "render.scheduler"),
        "shift": _number(render["shift"], "render.shift"),
        "no_scaffold": render["no_scaffold"],
        "no_wildcard": render["no_wildcard"],
        "no_trigger_inject": render["no_trigger_inject"],
        "turbo": render["turbo"],
        "content_loras": render["content_loras"],
    }
    if not SHA256.fullmatch(render["runner_sha256"]):
        raise EvalContractError("render.runner_sha256 must be lowercase SHA-256")
    if (
        any(
            not isinstance(render[name], bool)
            for name in ("no_scaffold", "no_wildcard", "no_trigger_inject", "turbo")
        )
        or not render["no_scaffold"]
        or not render["no_wildcard"]
        or not render["no_trigger_inject"]
        or render["turbo"]
        or render["content_loras"] != []
    ):
        raise EvalContractError("evaluation render must disable scaffold/wildcards/triggers/turbo/LoRAs")
    if min(render["width"], render["height"], render["steps"]) < 1:
        raise EvalContractError("render dimensions and steps must be positive")

    defaults = _object(bank["defaults"], "defaults",
                       {"negative_prompt", "required_tag_min_confidence"},
                       {"negative_prompt", "required_tag_min_confidence"})
    threshold = _number(defaults["required_tag_min_confidence"], "tag threshold", high=1.0)
    defaults = {
        "negative_prompt": _text(defaults["negative_prompt"], "negative prompt"),
        "required_tag_min_confidence": threshold,
    }
    sheet = _object(bank["contact_sheet"], "contact_sheet", {"columns", "thumb"},
                    {"columns", "thumb"})
    sheet = {"columns": _count(sheet["columns"], "sheet.columns"),
             "thumb": _count(sheet["thumb"], "sheet.thumb")}
    if (
        sheet["columns"] != len(models) * len(seeds)
        or not 64 <= sheet["thumb"] <= 512
    ):
        raise EvalContractError("invalid contact-sheet geometry")

    ccip_keys = {"mode", "baseline_model_id", "candidate_model_ids", "max_difference"}
    ccip = _object(bank["ccip"], "ccip", ccip_keys, ccip_keys)
    if (ccip["mode"] != "candidate_vs_baseline"
            or ccip["baseline_model_id"] != baseline[0]
            or not isinstance(ccip["candidate_model_ids"], list)
            or not ccip["candidate_model_ids"]
            or set(ccip["candidate_model_ids"]) - set(candidates)):
        raise EvalContractError("invalid candidate-vs-baseline CCIP configuration")
    ccip = {
        "mode": ccip["mode"],
        "baseline_model_id": baseline[0],
        "candidate_model_ids": list(ccip["candidate_model_ids"]),
        "max_difference": _number(ccip["max_difference"], "ccip.max_difference"),
    }

    values = bank["prompts"]
    if not isinstance(values, list) or not 30 <= len(values) <= 50:
        raise EvalContractError("bank must contain 30-50 prompts")
    prompt_required = {
        "id", "prompt", "evaluation_group", "expected_subjects", "expected_rating",
        "dimensions", "required_tags", "ccip_compare",
    }
    prompts: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    coverage = {axis: set() for axis in ("composition", "subjects", "style", "clothing", "rating")}
    for index, value in enumerate(values):
        item = _object(value, f"prompts[{index}]", prompt_required,
                       prompt_required | {"negative"})
        prompt_id = _slug(item["id"], f"prompts[{index}].id")
        if prompt_id in prompt_ids:
            raise EvalContractError(f"duplicate prompt id: {prompt_id}")
        prompt_ids.add(prompt_id)
        subjects = _count(item["expected_subjects"], f"{prompt_id}.expected_subjects")
        rating = item["expected_rating"]
        if rating not in RATINGS or not isinstance(item["ccip_compare"], bool):
            raise EvalContractError(f"invalid rating/CCIP flag for {prompt_id}")
        dims = _object(item["dimensions"], f"{prompt_id}.dimensions", set(coverage), set(coverage))
        dims = {axis: _slug(dims[axis], f"{prompt_id}.{axis}") for axis in coverage}
        if dims["subjects"] != str(subjects) or dims["rating"] != rating:
            raise EvalContractError(f"dimension contract mismatch for {prompt_id}")
        for axis, value in dims.items():
            coverage[axis].add(value)

        tag_values = item["required_tags"]
        if not isinstance(tag_values, list) or not tag_values:
            raise EvalContractError(f"{prompt_id}.required_tags must be non-empty")
        tags, seen_tags = [], set()
        for pos, value in enumerate(tag_values):
            if isinstance(value, str):
                name, minimum = _text(value, f"{prompt_id}.tag[{pos}]"), threshold
            else:
                value = _object(value, f"{prompt_id}.tag[{pos}]",
                                {"tag", "min_confidence"}, {"tag", "min_confidence"})
                name = _text(value["tag"], f"{prompt_id}.tag[{pos}]")
                minimum = _number(value["min_confidence"], f"{prompt_id}.tag[{pos}]", high=1.0)
            name = _tag(name)
            if not name or name in seen_tags:
                raise EvalContractError(f"invalid/duplicate required tag for {prompt_id}")
            seen_tags.add(name)
            tags.append({"tag": name, "min_confidence": round(minimum, 6)})
        prompts.append({
            "id": prompt_id,
            "prompt": _text(item["prompt"], f"{prompt_id}.prompt"),
            "negative": _text(item.get("negative", defaults["negative_prompt"]), f"{prompt_id}.negative"),
            "evaluation_group": _slug(item["evaluation_group"], f"{prompt_id}.group"),
            "expected_subjects": subjects,
            "expected_rating": rating,
            "dimensions": dims,
            "required_tags": tags,
            "ccip_compare": item["ccip_compare"],
        })

    minima = {"composition": 6, "subjects": 4, "style": 8, "clothing": 8, "rating": 4}
    if any(len(coverage[axis]) < minimum for axis, minimum in minima.items()):
        raise EvalContractError("prompt bank lacks required axis coverage")
    if coverage["rating"] != set(RATINGS):
        raise EvalContractError("prompt bank must cover all rating levels")
    if not {"retention", "target_domain"} <= {
        prompt["evaluation_group"] for prompt in prompts
    }:
        raise EvalContractError("prompt bank needs retention and target_domain groups")
    if not any(prompt["ccip_compare"] for prompt in prompts):
        raise EvalContractError("prompt bank needs at least one optional CCIP pair")
    return {
        "schema_version": BANK_SCHEMA,
        "bank_id": _slug(bank["bank_id"], "bank_id"),
        "description": _text(bank["description"], "description"),
        "models": models, "seeds": seeds, "render": render, "defaults": defaults,
        "contact_sheet": sheet, "ccip": ccip, "prompts": prompts,
    }


def load_bank(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = _read_json(path, "bank")
    bank = validate_bank(value)
    runner_value = bank["render"]["runner"]
    runner_posix = PurePosixPath(runner_value)
    if (
        "\\" in runner_value
        or runner_posix.is_absolute()
        or any(part in {"", ".", ".."} for part in runner_posix.parts)
    ):
        raise EvalContractError("render.runner must be a clean repo-relative POSIX path")
    runner = REPO_ROOT.joinpath(*runner_posix.parts).resolve()
    try:
        runner.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise EvalContractError("render.runner escapes the repository") from exc
    if not runner.is_file() or _sha_file(runner) != bank["render"]["runner_sha256"]:
        raise EvalContractError("render.runner bytes do not match frozen runner_sha256")
    return bank, _sha_bytes(raw)


def build_plan(bank: dict[str, Any], bank_sha256: str) -> dict[str, Any]:
    cells = []
    for prompt in bank["prompts"]:
        for model in bank["models"]:
            for seed in bank["seeds"]:
                slot = len(cells)
                cells.append({
                    "slot": slot,
                    "cell_id": f"{model['id']}__{prompt['id']}__seed_{seed}",
                    "model_id": model["id"], "model_artifact": model["artifact"],
                    "model_role": model["role"], "prompt_id": prompt["id"], "seed": seed,
                    "prompt": prompt["prompt"], "negative": prompt["negative"],
                    "evaluation_group": prompt["evaluation_group"],
                    "expected_subjects": prompt["expected_subjects"],
                    "expected_rating": prompt["expected_rating"],
                    "required_tags": prompt["required_tags"], "dimensions": prompt["dimensions"],
                    "ccip_compare": prompt["ccip_compare"],
                    "expected_output": f"images/{model['id']}/{prompt['id']}/seed_{seed}.png",
                })
    return {
        "schema_version": PLAN_SCHEMA, "bank_id": bank["bank_id"],
        "bank_sha256": bank_sha256, "matrix_order": ["prompt", "model", "seed"],
        "counts": {"models": len(bank["models"]), "prompts": len(bank["prompts"]),
                   "seeds": len(bank["seeds"]), "cells": len(cells)},
        "render": bank["render"], "models": bank["models"], "seeds": bank["seeds"],
        "ccip": bank["ccip"],
        "contact_sheet": {
            **bank["contact_sheet"],
            "layout": "one_prompt_per_row; model_then_seed_columns",
        },
        "cells": cells,
    }


def create_plan(bank_path: Path, out_path: Path) -> dict[str, Any]:
    bank, digest = load_bank(bank_path)
    plan = build_plan(bank, digest)
    _write_noclobber(out_path, _json_bytes(plan))
    return plan


def _load_plan(plan_path: Path, bank_path: Path) -> tuple[dict, str, str]:
    bank, bank_sha = load_bank(bank_path)
    plan, raw = _read_json(plan_path, "plan")
    if plan != build_plan(bank, bank_sha):
        raise EvalContractError("plan does not match the deterministic matrix from the current bank")
    return plan, _sha_bytes(raw), bank_sha


def _image_path(root: Path, value: Any, label: str) -> Path:
    value = _text(value, label)
    posix = PurePosixPath(value)
    if "\\" in value or posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise EvalContractError(f"{label} must be a clean POSIX-relative path")
    candidate = root.joinpath(*posix.parts)
    if candidate.is_symlink():
        raise EvalContractError(f"{label} must not be a symlink")
    try:
        candidate = candidate.resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvalContractError(f"{label} is missing or escapes the manifest directory") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise EvalContractError(f"{label} is not a supported regular image")
    return candidate


def _sidecar_path(root: Path, value: Any, label: str) -> Path:
    value = _text(value, label)
    posix = PurePosixPath(value)
    if "\\" in value or posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise EvalContractError(f"{label} must be a clean POSIX-relative path")
    candidate = root.joinpath(*posix.parts)
    if candidate.is_symlink():
        raise EvalContractError(f"{label} must not be a symlink")
    try:
        candidate = candidate.resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvalContractError(f"{label} is missing or escapes the manifest directory") from exc
    if not candidate.is_file() or not value.endswith(".png.meta.json"):
        raise EvalContractError(f"{label} is not a z3_anima_fast JSON sidecar")
    return candidate


def _inspect(path: Path, label: str) -> dict[str, Any]:
    before = path.stat()
    try:
        from PIL import Image
        with Image.open(path) as image:
            fmt, size = str(image.format or "").upper(), image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise EvalContractError(f"{label} decode failed: {type(exc).__name__}: {exc}") from exc
    digest, after = _sha_file(path), path.stat()
    before_sig = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_sig = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_sig != after_sig:
        raise EvalContractError(f"{label} changed during validation")
    return {
        "canonical_path": str(path), "sha256": digest, "bytes": after.st_size,
        "device": after.st_dev, "inode": after.st_ino, "format": fmt,
        "width": size[0], "height": size[1],
    }


def _contract_sha(value: Any) -> str:
    return _sha_bytes(_json_bytes(value))


def _expected_prompt_graph(cell: dict, model_artifact: dict, plan: dict) -> dict:
    render = plan["render"]
    return {
        "4": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_artifact["artifact"],
                "weight_dtype": "default",
            },
        },
        "7": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": render["clip_name"],
                "type": render["clip_type"],
                "device": "default",
            },
        },
        "8": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": render["vae_name"]},
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["7", 0], "text": cell["prompt"]},
        },
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["7", 0], "text": cell["negative"]},
        },
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": render["width"],
                "height": render["height"],
                "batch_size": 1,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["8", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "z3_anima_fast"},
        },
        "6": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["4", 0], "shift": render["shift"]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["11", 0],
                "seed": cell["seed"],
                "steps": render["steps"],
                "cfg": render["cfg"],
                "sampler_name": render["sampler"],
                "scheduler": render["scheduler"],
                "denoise": 1.0,
            },
        },
    }


def _validate_embedded_prompt_graph(
    path: Path, cell: dict, model_artifact: dict, plan: dict
) -> str:
    try:
        from PIL import Image

        with Image.open(path) as image:
            encoded = image.info.get("prompt")
    except Exception as exc:
        raise EvalContractError(f"{cell['cell_id']} PNG metadata read failed: {exc}") from exc
    if not isinstance(encoded, str):
        raise EvalContractError(f"{cell['cell_id']} PNG lacks embedded ComfyUI prompt graph")
    try:
        graph = json.loads(
            encoded,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvalContractError(
                    f"{cell['cell_id']} prompt graph has non-finite number {token}"
                )
            ),
        )
    except (json.JSONDecodeError, EvalContractError) as exc:
        raise EvalContractError(
            f"{cell['cell_id']} embedded ComfyUI prompt graph is invalid"
        ) from exc
    expected = _expected_prompt_graph(cell, model_artifact, plan)
    if graph != expected:
        raise EvalContractError(
            f"{cell['cell_id']} embedded ComfyUI prompt graph differs from the plan"
        )
    return _contract_sha(graph)


def _validate_generation_sidecar(
    path: Path, cell: dict, model_artifact: dict, plan: dict
) -> tuple[str, int]:
    value, raw = _read_json(path, f"{cell['cell_id']} sidecar")
    if not isinstance(value, dict):
        raise EvalContractError(f"{cell['cell_id']} sidecar must be an object")
    expected = {
        "seed": cell["seed"],
        "steps": plan["render"]["steps"],
        "cfg": plan["render"]["cfg"],
        "unet": model_artifact["artifact"],
        "shift": plan["render"]["shift"],
        "res": [plan["render"]["width"], plan["render"]["height"]],
        "sampler": plan["render"]["sampler"],
        "scheduler": plan["render"]["scheduler"],
        "prompt": cell["prompt"],
        "neg": cell["negative"],
        "lora": None,
        "turbo": None,
        "content_loras": [],
        "triggers_injected": "",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise EvalContractError(
                f"{cell['cell_id']} sidecar.{field} does not match the generation plan"
            )
    return _sha_bytes(raw), len(raw)


def _model_artifacts(value: Any, plan: dict) -> dict[str, dict]:
    if not isinstance(value, list):
        raise EvalContractError("image manifest.model_artifacts must be a list")
    planned = {model["id"]: model for model in plan["models"]}
    artifacts: dict[str, dict] = {}
    for index, raw in enumerate(value):
        item = _object(
            raw,
            f"model_artifacts[{index}]",
            {"model_id", "artifact", "sha256", "bytes"},
            {"model_id", "artifact", "sha256", "bytes"},
        )
        model_id = _slug(item["model_id"], f"model_artifacts[{index}].model_id")
        if model_id not in planned or model_id in artifacts:
            raise EvalContractError(f"invalid/duplicate model artifact: {model_id}")
        artifact = _text(item["artifact"], f"model_artifacts[{index}].artifact")
        digest = item["sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise EvalContractError(f"{model_id}.model sha256 must be lowercase SHA-256")
        size = _count(item["bytes"], f"{model_id}.model bytes")
        if not size:
            raise EvalContractError(f"{model_id}.model bytes must be positive")
        model = planned[model_id]
        expected_sha, expected_bytes = model["expected_sha256"], model["expected_bytes"]
        if expected_sha is not None and digest != expected_sha:
            raise EvalContractError(f"{model_id} checkpoint sha256 differs from frozen baseline")
        if expected_bytes is not None and size != expected_bytes:
            raise EvalContractError(f"{model_id} checkpoint bytes differ from frozen baseline")
        if model["role"] == "baseline" and artifact != model["artifact"]:
            raise EvalContractError(f"{model_id} artifact differs from frozen baseline path")
        if model["role"] == "candidate" and artifact == model["artifact"]:
            raise EvalContractError(f"{model_id} artifact still uses the unresolved placeholder")
        artifacts[model_id] = {
            "model_id": model_id,
            "artifact": artifact,
            "sha256": digest,
            "bytes": size,
        }
    if set(artifacts) != set(planned):
        raise EvalContractError("image manifest does not identify every planned model artifact")
    if len({item["sha256"] for item in artifacts.values()}) != len(artifacts):
        raise EvalContractError("baseline and candidate checkpoint SHA-256 must be distinct")
    return artifacts


def validate_image_manifest(manifest_path: Path, plan: dict, plan_sha: str,
                            bank_sha: str) -> tuple[dict[str, dict], str]:
    raw_manifest, raw = _read_json(manifest_path, "image manifest")
    manifest = _object(
        raw_manifest, "image manifest",
        {"schema_version", "bank_sha256", "plan_sha256", "model_artifacts", "entries"},
        {
            "schema_version", "bank_sha256", "plan_sha256", "model_artifacts",
            "entries", "metadata",
        },
    )
    if (manifest["schema_version"] != IMAGE_MANIFEST_SCHEMA
            or manifest["bank_sha256"] != bank_sha
            or manifest["plan_sha256"] != plan_sha):
        raise EvalContractError("image manifest does not bind the current schema/bank/plan")
    if not isinstance(manifest["entries"], list):
        raise EvalContractError("image manifest.entries must be a list")
    model_artifacts = _model_artifacts(manifest["model_artifacts"], plan)

    expected = {cell["cell_id"]: cell for cell in plan["cells"]}
    records, paths, inodes, image_hashes = {}, {}, {}, {}
    sidecar_paths, sidecar_inodes = {}, {}
    entry_required = {
        "cell_id", "model_id", "prompt_id", "seed", "path", "sha256", "bytes",
        "model_sha256", "runner_sha256", "prompt_sha256", "negative_sha256",
        "render_sha256", "prompt_graph_sha256", "sidecar_path", "sidecar_sha256",
        "sidecar_bytes",
    }
    render_sha = _contract_sha(plan["render"])
    for index, value in enumerate(manifest["entries"]):
        entry = _object(value, f"entries[{index}]", entry_required, entry_required | {"scores"})
        cell_id = _slug(entry["cell_id"], f"entries[{index}].cell_id")
        if cell_id in records:
            raise EvalContractError(f"duplicate matrix cell: {cell_id}")
        if cell_id not in expected:
            raise EvalContractError(f"extra matrix cell: {cell_id}")
        cell = expected[cell_id]
        if any(entry[key] != cell[key] for key in ("model_id", "prompt_id", "seed")):
            raise EvalContractError(f"{cell_id} identity does not match plan")
        provenance = {
            "model_sha256": model_artifacts[cell["model_id"]]["sha256"],
            "runner_sha256": plan["render"]["runner_sha256"],
            "prompt_sha256": _sha_bytes(cell["prompt"].encode("utf-8")),
            "negative_sha256": _sha_bytes(cell["negative"].encode("utf-8")),
            "render_sha256": render_sha,
        }
        for field, expected_value in provenance.items():
            if entry[field] != expected_value:
                raise EvalContractError(f"{cell_id}.{field} does not bind the generation contract")
        if entry["path"] != cell["expected_output"]:
            raise EvalContractError(f"{cell_id}.path must equal planned path {cell['expected_output']}")
        if not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"]):
            raise EvalContractError(f"{cell_id}.sha256 must be lowercase SHA-256")
        declared_bytes = _count(entry["bytes"], f"{cell_id}.bytes")
        if not declared_bytes:
            raise EvalContractError(f"{cell_id}.bytes must be positive")
        path = _image_path(manifest_path.parent, entry["path"], f"{cell_id}.path")
        identity = _inspect(path, cell_id)
        if identity["sha256"] != entry["sha256"]:
            raise EvalContractError(f"{cell_id} sha256 mismatch")
        if identity["bytes"] != declared_bytes:
            raise EvalContractError(f"{cell_id} byte-count mismatch")
        if (
            identity["format"] != "PNG"
            or identity["width"] != plan["render"]["width"]
            or identity["height"] != plan["render"]["height"]
        ):
            raise EvalContractError(
                f"{cell_id} render geometry/format differs from "
                f"PNG {plan['render']['width']}x{plan['render']['height']}"
            )
        if identity["canonical_path"] in paths:
            raise EvalContractError(f"image alias reused by {paths[identity['canonical_path']]} and {cell_id}")
        paths[identity["canonical_path"]] = cell_id
        inode = (identity["device"], identity["inode"])
        if inode in inodes:
            raise EvalContractError(f"image inode reused by {inodes[inode]} and {cell_id}")
        inodes[inode] = cell_id
        if identity["sha256"] in image_hashes:
            raise EvalContractError(
                f"image bytes reused by {image_hashes[identity['sha256']]} and {cell_id}"
            )
        image_hashes[identity["sha256"]] = cell_id
        prompt_graph_sha = _validate_embedded_prompt_graph(
            path, cell, model_artifacts[cell["model_id"]], plan
        )
        if entry["prompt_graph_sha256"] != prompt_graph_sha:
            raise EvalContractError(
                f"{cell_id}.prompt_graph_sha256 does not bind the embedded graph"
            )
        expected_sidecar = cell["expected_output"] + ".meta.json"
        if entry["sidecar_path"] != expected_sidecar:
            raise EvalContractError(
                f"{cell_id}.sidecar_path must equal planned path {expected_sidecar}"
            )
        sidecar = _sidecar_path(
            manifest_path.parent, entry["sidecar_path"], f"{cell_id}.sidecar_path"
        )
        sidecar_sha, sidecar_bytes = _validate_generation_sidecar(
            sidecar, cell, model_artifacts[cell["model_id"]], plan
        )
        declared_sidecar_sha = entry["sidecar_sha256"]
        declared_sidecar_bytes = _count(entry["sidecar_bytes"], f"{cell_id}.sidecar_bytes")
        if (
            not isinstance(declared_sidecar_sha, str)
            or not SHA256.fullmatch(declared_sidecar_sha)
            or not declared_sidecar_bytes
            or declared_sidecar_sha != sidecar_sha
            or declared_sidecar_bytes != sidecar_bytes
        ):
            raise EvalContractError(f"{cell_id} sidecar hash/byte-count mismatch")
        sidecar_stat = sidecar.stat()
        sidecar_canonical = str(sidecar)
        sidecar_inode = (sidecar_stat.st_dev, sidecar_stat.st_ino)
        if sidecar_canonical in sidecar_paths or sidecar_inode in sidecar_inodes:
            raise EvalContractError(f"{cell_id} reuses another cell's generation sidecar")
        sidecar_paths[sidecar_canonical] = cell_id
        sidecar_inodes[sidecar_inode] = cell_id
        scores = entry.get("scores", {})
        if not isinstance(scores, dict) or set(scores) - {"wd14", "qc", "ccip"}:
            raise EvalContractError(f"{cell_id}.scores has invalid fields")
        records[cell_id] = {
            "cell": cell, "declared_path": entry["path"], "path": path,
            "identity": identity, "scores": scores,
            "model_artifact": model_artifacts[cell["model_id"]],
            "generation_contract": provenance,
            "prompt_graph_sha256": prompt_graph_sha,
            "sidecar": {
                "path": sidecar,
                "declared_path": entry["sidecar_path"],
                "sha256": sidecar_sha,
                "bytes": sidecar_bytes,
            },
        }
    missing = set(expected) - set(records)
    if missing or len(records) != plan["counts"]["cells"]:
        raise EvalContractError(f"image manifest matrix mismatch: missing {len(missing)} cells")
    return records, _sha_bytes(raw)


def _probabilities(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise EvalContractError(f"{label} must be an object")
    out = {}
    for name, score in value.items():
        name = _tag(_text(name, f"{label}.key"))
        if name in out:
            raise EvalContractError(f"{label} has normalized duplicate {name}")
        out[name] = round(_number(score, f"{label}.{name}", high=1.0), 6)
    return out


def _wd14(value: Any, label: str) -> dict:
    item = _object(value, label, {"ok", "rating", "general", "character"},
                   {"ok", "rating", "general", "character", "reason", "input_identity",
                    "input_sha256", "provenance"})
    rating = _probabilities(item["rating"], f"{label}.rating")
    if item["ok"] is not True or set(RATINGS) - set(rating):
        raise EvalContractError(f"{label} is not a complete successful WD14 result")
    return {"ok": True, "rating": rating, "general": _probabilities(item["general"], f"{label}.general"),
            "character": _probabilities(item["character"], f"{label}.character"),
            "reason": str(item.get("reason", ""))}


def _qc(value: Any, label: str) -> dict:
    required = {"ok", "aesthetic_percentile", "aesthetic_label", "completeness",
                "monochrome", "faces", "hands", "persons"}
    item = _object(value, label, required,
                   required | {"reason", "input_sha256", "provenance"})
    if item["ok"] is not True:
        raise EvalContractError(f"{label}.ok must be true")
    return {
        "ok": True,
        "aesthetic_percentile": round(_number(item["aesthetic_percentile"],
                                               f"{label}.aesthetic", high=1.0), 6),
        "aesthetic_label": _text(item["aesthetic_label"], f"{label}.label"),
        "completeness": _text(item["completeness"], f"{label}.completeness"),
        "monochrome": round(_number(item["monochrome"], f"{label}.monochrome", high=1.0), 6),
        "faces": _count(item["faces"], f"{label}.faces"),
        "hands": _count(item["hands"], f"{label}.hands"),
        "persons": _count(item["persons"], f"{label}.persons"),
        "reason": str(item.get("reason", "")),
    }


def _ccip(value: Any, label: str) -> dict:
    required = {"ok", "difference", "threshold", "same_character"}
    item = _object(value, label, required,
                   required | {"reason", "input_sha256", "anchor_sha256",
                               "anchor_cell_id", "provenance"})
    if item["ok"] is not True or not isinstance(item["same_character"], bool):
        raise EvalContractError(f"{label} is not a successful CCIP result")
    return {
        "ok": True, "difference": round(_number(item["difference"], f"{label}.difference"), 6),
        "threshold": round(_number(item["threshold"], f"{label}.threshold", low=1e-12), 6),
        "same_character": item["same_character"], "reason": str(item.get("reason", "")),
    }


def _bound(raw: dict, record: dict, label: str, anchor: dict | None = None) -> None:
    if raw.get("input_sha256") != record["identity"]["sha256"]:
        raise EvalContractError(f"{label}.input_sha256 does not bind the image")
    if anchor:
        if raw.get("anchor_sha256") != anchor["identity"]["sha256"]:
            raise EvalContractError(f"{label}.anchor_sha256 does not bind the baseline")
        if raw.get("anchor_cell_id") != anchor["cell"]["cell_id"]:
            raise EvalContractError(f"{label}.anchor_cell_id does not bind the baseline")


def run_wd14_batch(venv: str, paths: list[str]) -> dict[str, dict]:
    if not paths:
        return {}
    try:
        proc = subprocess.run(
            [venv, str(SCRIPTS / "imgutils_bridge.py"), "wd14", *paths],
            capture_output=True, text=True, timeout=max(300, len(paths) * 60),
        )
        data = json.loads(proc.stdout)
    except Exception as exc:
        raise EvalContractError(f"WD14 bridge failed: {type(exc).__name__}: {exc}") from exc
    if proc.returncode:
        raise EvalContractError(f"WD14 bridge exited {proc.returncode}: {proc.stderr[:300]}")
    if len(paths) == 1:
        return {paths[0]: data}
    if not isinstance(data, dict) or data.get("ok") is not True or set(data.get("results", {})) != set(paths):
        raise EvalContractError(f"WD14 batch incomplete: {data.get('reason', '') if isinstance(data, dict) else ''}")
    return data["results"]


def _analyze_records(
    records: dict[str, dict], plan: dict, venv: str
) -> tuple[list[dict], dict[str, int]]:
    ordered = [records[cell["cell_id"]] for cell in plan["cells"]]
    wd_missing = [record for record in ordered if "wd14" not in record["scores"]]
    qc_missing = [record for record in ordered if "qc" not in record["scores"]]
    if (wd_missing or qc_missing) and not os.path.isfile(venv):
        raise EvalContractError("WD14/QC results are missing and imgutils interpreter is unavailable")
    live_wd = run_wd14_batch(venv, [str(record["path"]) for record in wd_missing])

    baseline_id = plan["ccip"]["baseline_model_id"]
    candidates = set(plan["ccip"]["candidate_model_ids"])
    by_key = {(r["cell"]["model_id"], r["cell"]["prompt_id"], r["cell"]["seed"]): r for r in ordered}
    pairs = []
    for record in ordered:
        cell = record["cell"]
        if cell["model_id"] in candidates and cell["ccip_compare"]:
            anchor = by_key.get((baseline_id, cell["prompt_id"], cell["seed"]))
            if not anchor:
                raise EvalContractError(f"no CCIP baseline for {cell['cell_id']}")
            pairs.append((record, anchor))
    eligible = {record["cell"]["cell_id"] for record, _ in pairs}
    extra_ccip = [record["cell"]["cell_id"] for record in ordered
                  if "ccip" in record["scores"] and record["cell"]["cell_id"] not in eligible]
    if extra_ccip:
        raise EvalContractError(f"CCIP result on non-eligible cells: {extra_ccip[:3]}")
    if any("ccip" not in record["scores"] for record, _ in pairs) and not os.path.isfile(venv):
        raise EvalContractError("CCIP matrix is partial and imgutils interpreter is unavailable")

    sources = {f"{op}_{source}": 0 for op in ("wd14", "qc", "ccip")
               for source in ("precomputed", "bridge")}
    scored_ccip, ccip_source = {}, {}
    for record, anchor in pairs:
        cell_id = record["cell"]["cell_id"]
        if "ccip" in record["scores"]:
            raw = record["scores"]["ccip"]
            if not isinstance(raw, dict):
                raise EvalContractError(f"{cell_id}.ccip must be an object")
            _bound(raw, record, f"{cell_id}.ccip", anchor)
            result, source = _ccip(raw, f"{cell_id}.ccip"), "precomputed"
        else:
            raw = ccip_one(venv, str(anchor["path"]), str(record["path"]))
            if raw is None:
                raise EvalContractError(f"CCIP bridge failed for {cell_id}")
            result, source = _ccip(raw, f"{cell_id}.ccip"), "bridge"
        result.update({
            "anchor_cell_id": anchor["cell"]["cell_id"],
            "anchor_sha256": anchor["identity"]["sha256"],
            "held_at_configured_max": result["difference"] <= plan["ccip"]["max_difference"],
        })
        scored_ccip[cell_id], ccip_source[cell_id] = result, source
        sources[f"ccip_{source}"] += 1

    rows = []
    for record in ordered:
        cell, cell_id = record["cell"], record["cell"]["cell_id"]
        if "wd14" in record["scores"]:
            raw = record["scores"]["wd14"]
            if not isinstance(raw, dict):
                raise EvalContractError(f"{cell_id}.wd14 must be an object")
            _bound(raw, record, f"{cell_id}.wd14")
            wd, wd_source = _wd14(raw, f"{cell_id}.wd14"), "precomputed"
        else:
            wd, wd_source = _wd14(live_wd[str(record["path"])], f"{cell_id}.wd14"), "bridge"
        sources[f"wd14_{wd_source}"] += 1

        if "qc" in record["scores"]:
            raw = record["scores"]["qc"]
            if not isinstance(raw, dict):
                raise EvalContractError(f"{cell_id}.qc must be an object")
            _bound(raw, record, f"{cell_id}.qc")
            quality, qc_source = _qc(raw, f"{cell_id}.qc"), "precomputed"
        else:
            raw = qc_one(venv, str(record["path"]))
            if raw is None:
                raise EvalContractError(f"QC bridge failed for {cell_id}")
            quality, qc_source = _qc(raw, f"{cell_id}.qc"), "bridge"
        sources[f"qc_{qc_source}"] += 1

        tags = dict(wd["general"])
        for name, confidence in wd["character"].items():
            tags[name] = max(tags.get(name, 0.0), confidence)
        checks = [
            {**expected, "confidence": tags.get(expected["tag"], 0.0),
             "hit": tags.get(expected["tag"], 0.0) >= expected["min_confidence"]}
            for expected in cell["required_tags"]
        ]
        predicted = max(RATINGS, key=lambda name: (wd["rating"][name], -RATINGS.index(name)))
        quality["flags"] = flags_for(quality, cell["expected_subjects"])
        quality["unflagged"] = not quality["flags"]
        rows.append({
            "slot": cell["slot"], "cell_id": cell_id, "model_id": cell["model_id"],
            "model_role": cell["model_role"], "prompt_id": cell["prompt_id"], "seed": cell["seed"],
            "evaluation_group": cell["evaluation_group"], "dimensions": cell["dimensions"],
            "path": record["declared_path"], "sha256": record["identity"]["sha256"],
            "bytes": record["identity"]["bytes"], "expected_subjects": cell["expected_subjects"],
            "expected_rating": cell["expected_rating"],
            "required_tag_hits": sum(check["hit"] for check in checks),
            "required_tag_total": len(checks),
            "required_tag_hit_rate": round(sum(check["hit"] for check in checks) / len(checks), 6),
            "required_tags": checks, "rating_prediction": predicted,
            "rating_hit": predicted == cell["expected_rating"], "rating_scores": wd["rating"],
            "qc": quality, "ccip": scored_ccip.get(cell_id),
            "analysis_source": {"wd14": wd_source, "qc": qc_source,
                                "ccip": ccip_source.get(cell_id, "not_requested")},
        })
    sources.update({"ccip_eligible": len(pairs), "ccip_scored": len(scored_ccip)})
    if sources["ccip_scored"] != sources["ccip_eligible"]:
        raise EvalContractError("CCIP matrix is incomplete")
    return rows, sources


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _aggregate(rows: list[dict]) -> dict:
    hits, total = sum(r["required_tag_hits"] for r in rows), sum(r["required_tag_total"] for r in rows)
    ccip = [r["ccip"] for r in rows if r["ccip"]]
    return {
        "cells": len(rows), "required_tag_hits": hits, "required_tag_total": total,
        "required_tag_hit_rate": round(hits / total, 6) if total else None,
        "mean_aesthetic_percentile": _mean([r["qc"]["aesthetic_percentile"] for r in rows]),
        "qc_unflagged_cells": sum(r["qc"]["unflagged"] for r in rows),
        "qc_unflagged_rate": round(sum(r["qc"]["unflagged"] for r in rows) / len(rows), 6) if rows else None,
        "rating_hits": sum(r["rating_hit"] for r in rows),
        "rating_accuracy": round(sum(r["rating_hit"] for r in rows) / len(rows), 6) if rows else None,
        "ccip_scored_cells": len(ccip), "ccip_mean_difference": _mean([r["difference"] for r in ccip]),
        "ccip_held_cells": sum(r["held_at_configured_max"] for r in ccip),
        "ccip_hold_rate": round(sum(r["held_at_configured_max"] for r in ccip) / len(ccip), 6) if ccip else None,
    }


def _group(rows: list[dict], key) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    return {name: _aggregate(groups[name]) for name in sorted(groups)}


def build_metrics(rows: list[dict], plan: dict, plan_sha: str, bank_sha: str,
                  manifest_sha: str, sources: dict) -> dict:
    by_model = _group(rows, lambda row: row["model_id"])
    axes = {
        axis: _group(rows, lambda row, axis=axis: f"{row['model_id']}::{row['dimensions'][axis]}")
        for axis in ("composition", "subjects", "style", "clothing", "rating")
    }
    baseline = plan["ccip"]["baseline_model_id"]
    deltas = {}
    ccip_comparisons = {}
    for candidate in plan["ccip"]["candidate_model_ids"]:
        deltas[candidate] = {}
        for metric in ("required_tag_hit_rate", "mean_aesthetic_percentile",
                       "qc_unflagged_rate", "rating_accuracy"):
            left, right = by_model[baseline][metric], by_model[candidate][metric]
            deltas[candidate][metric] = (
                round(right - left, 6) if left is not None and right is not None else None
            )
        ccip_comparisons[candidate] = {
            "scored_cells": by_model[candidate]["ccip_scored_cells"],
            "mean_difference": by_model[candidate]["ccip_mean_difference"],
            "held_cells": by_model[candidate]["ccip_held_cells"],
            "hold_rate": by_model[candidate]["ccip_hold_rate"],
            "configured_max_difference": plan["ccip"]["max_difference"],
        }
    return {
        "schema_version": METRICS_SCHEMA, "status": "complete", "bank_sha256": bank_sha,
        "plan_sha256": plan_sha, "image_manifest_sha256": manifest_sha,
        "matrix_counts": plan["counts"], "analysis_sources": sources,
        "overall": _aggregate(rows), "by_model": by_model,
        "by_model_and_evaluation_group": _group(
            rows, lambda row: f"{row['model_id']}::{row['evaluation_group']}"
        ),
        "by_model_and_axis": axes,
        "candidate_minus_baseline": deltas,
        "candidate_vs_baseline_ccip": ccip_comparisons,
        "cells": rows,
    }


def _same(before: dict, after: dict) -> bool:
    return all(before[key] == after[key]
               for key in ("sha256", "bytes", "device", "inode", "format", "width", "height"))


def _sheet_geometry(count: int, cols: int, thumb: int) -> tuple[int, int, int]:
    from contact_sheet import PAD

    rows = math.ceil(count / cols)
    return (
        SHEET_ROW_LABEL_W + cols * (thumb + 2 * PAD),
        SHEET_HEADER_H + rows * (thumb + 2 * PAD),
        rows,
    )


def _validate_contact_sheet(
    path: Path, plan: dict, paths: list[str], labels: list[dict[str, Any]]
) -> dict[str, int]:
    cols, thumb = (
        plan["contact_sheet"]["columns"],
        plan["contact_sheet"]["thumb"],
    )
    width, height, rows = _sheet_geometry(plan["counts"]["cells"], cols, thumb)
    identity = _inspect(path, "contact_sheet")
    if (
        identity["format"] != "PNG"
        or identity["width"] != width
        or identity["height"] != height
        or rows != plan["counts"]["prompts"]
    ):
        raise EvalContractError(
            "contact sheet is not the complete one-prompt-per-row XY layout"
        )
    reference: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=".contact-sheet-reference-", suffix=".png", dir=path.parent
        )
        os.close(fd)
        reference = Path(name)
        _render_eval_sheet(paths, labels, str(reference), cols=cols, thumb=thumb)
        if _sha_file(reference) != identity["sha256"]:
            raise EvalContractError(
                "contact sheet pixels/labels do not match the complete matrix"
            )
    finally:
        if reference is not None:
            reference.unlink(missing_ok=True)
    return {"columns": cols, "rows": rows, "width": width, "height": height}


def _rename_dir_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a racing destination."""

    src, dst = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise EvalContractError("atomic no-replace directory publish is unavailable") from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-2, src, -2, dst, 0x00000004)  # AT_FDCWD, RENAME_EXCL
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise EvalContractError("atomic no-replace directory publish is unavailable") from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, src, -100, dst, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        move.restype = ctypes.c_int
        if move(str(source), str(destination), 0):
            return
        error = ctypes.get_last_error()
        if error in {80, 183}:
            raise FileExistsError(f"refusing to overwrite existing output directory: {destination}")
        raise OSError(error, os.strerror(error), str(destination))
    else:
        raise EvalContractError("atomic no-replace directory publish is unavailable")

    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to overwrite existing output directory: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _results(metrics: dict, evidence_sha: str, metrics_sha: str, sheet_sha: str) -> str:
    overall = metrics["overall"]
    lines = [
        "# Anima Step 2a Phase C Evaluation Results", "",
        "- Status: **COMPLETE** (matrix/evidence contract complete; this is not an aesthetic verdict)",
        f"- Cells: {overall['cells']}",
        f"- Required expected-tag hit rate: {overall['required_tag_hit_rate']:.4f}",
        f"- Mean QC aesthetic percentile: {overall['mean_aesthetic_percentile']:.4f}",
        f"- QC unflagged rate: {overall['qc_unflagged_rate']:.4f}",
        f"- WD14 rating accuracy: {overall['rating_accuracy']:.4f}",
    ]
    if overall["ccip_scored_cells"]:
        lines += [
            f"- Required CCIP cells: {overall['ccip_scored_cells']}",
            f"- Required CCIP mean difference: {overall['ccip_mean_difference']:.4f}",
            f"- Required CCIP hold rate: {overall['ccip_hold_rate']:.4f}",
        ]
    else:
        lines.append("- Required CCIP: incomplete")
    lines += ["", "## By model", ""]
    for name, value in metrics["by_model"].items():
        lines.append(
            f"- `{name}`: tag={value['required_tag_hit_rate']:.4f}, "
            f"aesthetic={value['mean_aesthetic_percentile']:.4f}, "
            f"qc_unflagged={value['qc_unflagged_rate']:.4f}, rating={value['rating_accuracy']:.4f}"
        )
    lines += [
        "", "## Evidence artifacts", "",
        f"- `metrics.json` sha256 `{metrics_sha}`",
        f"- `evidence.json` sha256 `{evidence_sha}`",
        f"- `contact_sheet.png` sha256 `{sheet_sha}`",
        f"- Plan sha256 `{metrics['plan_sha256']}`",
        f"- Image manifest sha256 `{metrics['image_manifest_sha256']}`", "",
        "The contact sheet uses plan slot order. Missing cells are fatal before sheet construction.", "",
    ]
    return "\n".join(lines)


def score_manifest(
    bank_path: Path,
    plan_path: Path,
    manifest_path: Path,
    out_dir: Path,
    *,
    venv: str = DEFAULT_VENV_PY,
) -> dict:
    if os.path.lexists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    plan, plan_sha, bank_sha = _load_plan(plan_path, bank_path)
    records, manifest_sha = validate_image_manifest(manifest_path, plan, plan_sha, bank_sha)
    rows, sources = _analyze_records(records, plan, venv)
    for record in records.values():
        if not _same(record["identity"], _inspect(record["path"], record["cell"]["cell_id"])):
            raise EvalContractError(f"{record['cell']['cell_id']} changed after analysis")
    metrics = build_metrics(rows, plan, plan_sha, bank_sha, manifest_sha, sources)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    published = False
    try:
        sheet = staging / "contact_sheet.png"
        sheet_records = [records[cell["cell_id"]] for cell in plan["cells"]]
        sheet_paths = [str(record["path"]) for record in sheet_records]
        sheet_labels = [
            {
                "model_id": record["cell"]["model_id"],
                "prompt_id": record["cell"]["prompt_id"],
                "seed": record["cell"]["seed"],
            }
            for record in sheet_records
        ]
        build_contact_sheet(
            sheet_paths,
            sheet_labels,
            str(sheet),
            cols=plan["contact_sheet"]["columns"],
            thumb=plan["contact_sheet"]["thumb"],
        )
        if not sheet.is_file():
            raise EvalContractError("contact-sheet helper produced no output")
        sheet_layout = _validate_contact_sheet(
            sheet, plan, sheet_paths, sheet_labels
        )
        for record in records.values():
            if not _same(record["identity"], _inspect(record["path"], record["cell"]["cell_id"])):
                raise EvalContractError(f"{record['cell']['cell_id']} changed during contact sheet")

        metrics_path = staging / "metrics.json"
        metrics_path.write_bytes(_json_bytes(metrics))
        metrics_sha, sheet_sha = _sha_file(metrics_path), _sha_file(sheet)
        evidence = {
            "schema_version": EVIDENCE_SCHEMA, "status": "complete",
            "bank": {"path": str(bank_path), "sha256": bank_sha},
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "image_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "model_artifacts": [
                records[next(
                    cell["cell_id"]
                    for cell in plan["cells"]
                    if cell["model_id"] == model["id"]
                )]["model_artifact"]
                for model in plan["models"]
            ],
            "matrix": {"expected_cells": plan["counts"]["cells"], "verified_cells": len(records),
                       "extra_cells": 0, "missing_cells": 0},
            "inputs": [
                {"slot": r["cell"]["slot"], "cell_id": r["cell"]["cell_id"],
                 "path": r["declared_path"], "sha256": r["identity"]["sha256"],
                 "bytes": r["identity"]["bytes"], "format": r["identity"]["format"],
                 "width": r["identity"]["width"], "height": r["identity"]["height"],
                 "sidecar_path": r["sidecar"]["declared_path"],
                 "sidecar_sha256": r["sidecar"]["sha256"],
                 "sidecar_bytes": r["sidecar"]["bytes"],
                 "prompt_graph_sha256": r["prompt_graph_sha256"],
                 "generation_contract": r["generation_contract"]}
                for r in (records[cell["cell_id"]] for cell in plan["cells"])
            ],
            "contact_sheet": {
                "path": sheet.name, "sha256": sheet_sha,
                **sheet_layout,
                "thumb": plan["contact_sheet"]["thumb"],
                "layout": plan["contact_sheet"]["layout"],
                "slots": [
                    {
                        "slot": cell["slot"],
                        "row": cell["slot"] // sheet_layout["columns"],
                        "column": cell["slot"] % sheet_layout["columns"],
                        "cell_id": cell["cell_id"],
                        "label": (
                            f"{sheet_labels[cell['slot']]['model_id']} "
                            f"{sheet_labels[cell['slot']]['prompt_id']} "
                            f"seed {sheet_labels[cell['slot']]['seed']}"
                        ),
                    }
                    for cell in plan["cells"]
                ],
            },
            "artifacts": {
                "metrics.json": {"sha256": metrics_sha, "bytes": metrics_path.stat().st_size},
                "contact_sheet.png": {"sha256": sheet_sha, "bytes": sheet.stat().st_size},
            },
            "analysis_sources": sources,
        }
        evidence_path = staging / "evidence.json"
        evidence_path.write_bytes(_json_bytes(evidence))
        evidence_sha = _sha_file(evidence_path)
        (staging / "RESULTS.md").write_text(
            _results(metrics, evidence_sha, metrics_sha, sheet_sha), encoding="utf-8"
        )
        _rename_dir_noreplace(staging, out_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="strict offline Anima Step-2a Phase-C evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write deterministic model x prompt x seed plan")
    plan.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    plan.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("score", help="score a complete hash-bound image matrix")
    score.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--venv", default=os.environ.get("IMGUTILS_VENV_PY", DEFAULT_VENV_PY))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = create_plan(args.bank, args.out)
            print(f"PLAN {args.out} cells={plan['counts']['cells']} sha256={_sha_file(args.out)}")
        else:
            metrics = score_manifest(
                args.bank, args.plan, args.manifest, args.out_dir,
                venv=args.venv,
            )
            print(f"COMPLETE {args.out_dir} cells={metrics['overall']['cells']} "
                  f"tag_hit_rate={metrics['overall']['required_tag_hit_rate']:.4f}")
    except (EvalContractError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
