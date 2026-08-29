#!/usr/bin/env python3
"""Hermes-facing image-generation facade.

One tool: ``studio_generate``. That name is the local ASCII-diffusion trigger
(``mcp__loramake_image__studio_generate``). xAI Imagine stays
``image_generate`` — do not reuse that name. New local edit/gen lines dispatch
*inside* this function; do not add sibling MCP tools.

The engines stay in ``scripts/gen.py``. This file only shells out — except the
``region`` phases ``inspect`` and ``preview``, which are pure local PIL work
(overlays and mask compilation) and deliberately never touch a GPU: the model
looks, selects, and shows its selection for confirmation before ``commit``
spends a render.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GEN_PY = REPO / "scripts" / "gen.py"
NSFW_GEN_PY = REPO / "nsfw" / "scripts" / "gen.py"
FRESH_INTENT = "draft"
EDIT_INTENT = "promote"
REGION_INTENT = "region"
RESCUE_INTENT = "rescue"
RESCUE_BOX = "zzz"  # pose lock is verified on ZZZ/iLustMix only
RUN_TIMEOUT_S = 900

REGION_PHASES = ("inspect", "crops", "preview", "commit")

REGION_HELP = (
    'region JSON: {"phase": "inspect"|"crops"|"preview"|"commit", '
    '"boxes": [[x,y,w,h], ...], "cells": ["E9", ...], '
    '"detect": [{"class": "feet", "index": 0}, ...], '
    '"keep_outs": [[x,y,w,h], ...], "mode": "refine"|"replace"|"polish"}. '
    "inspect returns a labelled coordinate grid (plus detector hits when "
    '"detect" names classes); "cells" are grid cell ids read off that overlay '
    "(each becomes one box); crops returns 2x zooms of the selected boxes so "
    "details (like toe count) can be verified up close; preview compiles the "
    "mask and returns a red overlay to confirm; commit runs the repaint."
)


def gen_script(source: str | None) -> Path:
    """Pick the existing router. Source under nsfw/outputs uses the NSFW copy."""
    override = os.environ.get("LORAMAKE_GEN_PY")
    if override:
        return Path(override)
    if source:
        resolved = Path(source).expanduser().resolve()
        nsfw_root = (REPO / "nsfw" / "outputs").resolve()
        try:
            resolved.relative_to(nsfw_root)
            if NSFW_GEN_PY.is_file():
                return NSFW_GEN_PY
        except ValueError:
            pass
    return DEFAULT_GEN_PY


def build_gen_argv(
    prompt: str,
    source: str | None,
    json_out: Path,
    *,
    gen_py: Path | None = None,
) -> list[str]:
    script = gen_py or gen_script(source)
    argv = [sys.executable, str(script)]
    if source:
        argv += [EDIT_INTENT, "--source", source, "--prompt", prompt]
    else:
        argv += [FRESH_INTENT, "--prompt", prompt]
    argv += ["--run", "--json-out", str(json_out)]
    return argv


def build_rescue_argv(
    prompt: str | None,
    source: str,
    json_out: Path,
    *,
    gen_py: Path | None = None,
) -> list[str]:
    """Structure rescue: gen.py rescue -> scripts/rescue_gen.py (one shot, fixed recipe)."""
    script = gen_py or gen_script(source)
    argv = [sys.executable, str(script), RESCUE_INTENT, "--source", source]
    if prompt:
        argv += ["--prompt", prompt]
    argv += ["--box", RESCUE_BOX, "--run", "--json-out", str(json_out)]
    return argv


def build_region_argv(
    prompt: str,
    source: str,
    spec_json: str,
    mode: str,
    json_out: Path,
    *,
    gen_py: Path | None = None,
) -> list[str]:
    """The commit-phase dispatch: gen.py region -> scripts/region_fix.py."""
    script = gen_py or gen_script(source)
    argv = [sys.executable, str(script), REGION_INTENT, "--source", source, "--prompt", prompt]
    argv += ["--run", "--json-out", str(json_out)]
    argv += ["--", "--spec-json", spec_json, "--mode", mode]
    return argv


def pick_image_path(result: dict) -> str | None:
    """One displayable path from a ``loramake.gen-result`` blob."""
    for key in ("best",):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("survivors", "new_files"):
        files = result.get(key)
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str) and item.strip():
                    return item
    return None


def display_envelope(path: str, result: dict) -> dict:
    return {
        "success": True,
        "host_image": path,
        "image": path,
        "intent": result.get("intent"),
        "box": result.get("box"),
        "rc": result.get("rc", 0),
    }


def fail_envelope(error: str, **extra: object) -> dict:
    payload = {"success": False, "error": error}
    payload.update(extra)
    return payload


def _spawn_gen(argv: list[str], json_out: Path) -> dict:
    """Run one gen.py subprocess and translate its json-out into an envelope."""
    env = {**os.environ, "DRIVING_MODEL": os.environ.get("DRIVING_MODEL", "hermes")}
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
        raw = json_out.read_text(encoding="utf-8") if json_out.is_file() else ""
        result = json.loads(raw) if raw.strip() else {}
        if not isinstance(result, dict):
            result = {}
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"gen.py exited {proc.returncode}").strip()
            return fail_envelope(err[-2000:], rc=proc.returncode, intent=result.get("intent"))
        path = pick_image_path(result)
        if not path:
            return fail_envelope("gen.py produced no image", rc=proc.returncode, result=result)
        return display_envelope(path, result)
    except subprocess.TimeoutExpired:
        return fail_envelope(f"gen.py timed out after {RUN_TIMEOUT_S}s")
    except json.JSONDecodeError as exc:
        return fail_envelope(f"gen.py json-out unreadable: {exc}")
    finally:
        json_out.unlink(missing_ok=True)


def run_generate(prompt: str, source: str | None = None) -> dict:
    text = (prompt or "").strip()
    if not text:
        return fail_envelope("empty prompt")
    if source:
        src = Path(source).expanduser()
        if not src.is_file():
            return fail_envelope(f"source not found: {source}")
        source = str(src)

    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_gen_argv(text, source, json_out)
    return _spawn_gen(argv, json_out)


def run_rescue(prompt: str | None, source: str | None) -> dict:
    """Pose-locked re-render of a structurally broken image (extra limbs, fused
    objects, impossible joints). The prompt is optional: without it the source
    sidecar prompt is reused; pass one only to correct a content-level error
    (an OOD token literalised into an object)."""
    if not source:
        return fail_envelope("rescue needs a source image path")
    src = Path(source).expanduser()
    if not src.is_file():
        return fail_envelope(f"source not found: {source}")

    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_rescue_argv((prompt or "").strip() or None, str(src), json_out)
    envelope = _spawn_gen(argv, json_out)
    envelope.setdefault("mode", "rescue")
    return envelope


# --------------------------------------------------------------------------------------- #
# Region phases
# --------------------------------------------------------------------------------------- #

def _scratch_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "loramake_region"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scratch_path(stem: str, kind: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return _scratch_dir() / f"{stem}_{kind}_{stamp}.png"


def _parse_detect_classes(value: object) -> list[str]:
    sys.path.insert(0, str(REPO))
    from loramake.mask_spec import DETECT_CLASSES
    from loramake.region_mask import MaskError

    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise MaskError('inspect "detect" must be a list of class names')
    unknown = [v for v in value if v not in DETECT_CLASSES]
    if unknown:
        raise MaskError(
            f"unknown detector classes {unknown}; choose from {', '.join(DETECT_CLASSES)}"
        )
    return list(value)


def _region_inspect(source: str, payload: dict) -> tuple[dict, str | None]:
    from PIL import Image

    from loramake import region_detect, region_overlay
    from loramake.region_mask import MaskError

    extra = set(payload) - {"phase", "detect"}
    if extra:
        raise MaskError(f"inspect accepts only \"detect\"; unknown keys: {sorted(extra)}")
    classes = _parse_detect_classes(payload.get("detect"))

    img = Image.open(source)
    img.load()
    overlay = region_overlay.grid_overlay(img)
    detections: list[dict] = []
    if classes:
        detections = [
            d for d in region_detect.detect_regions(source) if d["label"] in classes
        ]
        overlay = region_overlay.hits_overlay(overlay, detections)
    out_path = _scratch_path(Path(source).stem, "inspect")
    overlay.save(out_path)
    step_x, step_y = region_overlay.grid_steps((img.width, img.height))
    envelope = {
        "success": True,
        "phase": "inspect",
        "source": source,
        "size": [img.width, img.height],
        "grid_step": [step_x, step_y],
        "detections": detections,
        "overlay": str(out_path),
        "next": (
            "read the target region off the grid (cell ids like E9, or pixel boxes), then "
            "call phase=crops to verify the detail up close before phase=preview"
        ),
    }
    return envelope, str(out_path)


def _spec_fields(source: str, payload: dict) -> dict:
    """Region payload -> mask-spec fields: strip the phase, resolve cells to boxes.

    Cells are the coarse answer format for models that read the grid overlay but cannot
    report reliable pixel coordinates; the conversion must use the same grid step the
    overlay drew, so it goes through region_overlay.
    """
    from PIL import Image

    from loramake import region_overlay
    from loramake.region_mask import MaskError

    fields = {k: v for k, v in payload.items() if k != "phase"}
    cells = fields.pop("cells", None)
    if cells is not None:
        if not isinstance(cells, list) or not all(isinstance(c, str) for c in cells):
            raise MaskError('"cells" must be a list of grid cell ids like "E9"')
        with Image.open(source) as img:
            size = (img.width, img.height)
        try:
            boxes = region_overlay.cells_to_boxes(size, cells)
        except ValueError as exc:
            raise MaskError(str(exc)) from exc
        fields["boxes"] = list(fields.get("boxes") or []) + [list(b.as_tuple()) for b in boxes]
    fields.setdefault("source", source)
    return fields


def _compile_from_payload(source: str, payload: dict):
    """Shared preview/commit spec handling: parse, detect if needed, compile."""
    from loramake import mask_spec, region_detect

    spec = mask_spec.parse_spec(_spec_fields(source, payload))
    detections = region_detect.detect_regions(source) if spec.wants_detection() else []
    from PIL import Image

    with Image.open(source) as img:
        size = (img.width, img.height)
        src = img.convert("RGB")
    mask = mask_spec.compile_mask(spec, size, detections)
    return spec, src, mask, detections


def _region_crops(source: str, payload: dict) -> tuple[dict, str | None]:
    """2x zooms of the selected boxes, tiled into one sheet — the look-closer step.

    Fine judgments (a toe count, a finger boundary) are unreliable on a full frame even for
    strong vision models; on a 2x crop they are cheap. The protocol therefore counts on
    crops, never on the full image.
    """
    from PIL import Image

    from loramake import mask_spec, region_detect, region_overlay
    from loramake.region_mask import MaskError

    spec = mask_spec.parse_spec({**_spec_fields(source, payload), "keep_outs": []})
    with Image.open(source) as img:
        size = (img.width, img.height)
        src = img.convert("RGB")
    boxes = list(spec.boxes)
    if spec.detect:
        detections = region_detect.detect_regions(source)
        boxes += mask_spec.select_detections(spec, detections)
    if not boxes:
        raise MaskError("crops needs at least one boxes/cells/detect selection")

    pad = 24
    # Touching cells describe one region; crop the merged extent, not each cell,
    # or the subject gets cut at every cell boundary and counts become unanswerable.
    boxes = region_overlay.merge_boxes(boxes, gap=pad)
    tiles = []
    at_border: list[list[str]] = []
    for box in boxes:
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1 = min(size[0], box.x + box.width + pad)
        y1 = min(size[1], box.y + box.height + pad)
        # The enclosure gate needs to know which edges cannot be widened. The clamp above
        # already decides that, so report it instead of leaving the caller to judge it.
        at_border.append(
            [
                name
                for name, hit in (
                    ("left", x0 == 0),
                    ("top", y0 == 0),
                    ("right", x1 == size[0]),
                    ("bottom", y1 == size[1]),
                )
                if hit
            ]
        )
        crop = src.crop((x0, y0, x1, y1))
        tiles.append(crop.resize((crop.width * 2, crop.height * 2)))
    sheet_w = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    sheet_h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    out_path = _scratch_path(Path(source).stem, "crops")
    sheet.save(out_path)
    envelope = {
        "success": True,
        "phase": "crops",
        "source": source,
        "boxes": [list(b.as_tuple()) for b in boxes],
        "at_image_border": at_border,
        "crops": str(out_path),
        "next": (
            "first check enclosure on all four edges of each crop: the outermost content "
            "must be non-subject, and a crop whose subject touches an edge is invalid — "
            "add the neighbouring cell in that direction and re-run crops. The exception "
            'is an edge listed in "at_image_border": the source image ends there, so it '
            "encloses the subject and cannot be widened. Then verify the detail up close "
            "(e.g. count toes per foot twice, by tips and by toenails, per the delegation "
            "playbook; a healthy foot counts 5) — in a style that draws no consistent nail "
            "plate the tip count alone is authoritative and the nail count is void. Then "
            "call phase=preview with the confirmed selection"
        ),
    }
    return envelope, str(out_path)


def _region_preview(source: str, payload: dict) -> tuple[dict, str | None]:
    from loramake import mask_spec, region_overlay
    from loramake.region_mask import coverage

    spec, src, mask, _ = _compile_from_payload(source, payload)
    box = mask_spec.derive_bbox(mask)
    preview = region_overlay.mask_preview(
        src, mask, boxes=spec.boxes, keep_outs=spec.keep_outs, work_box=box
    )
    stem = Path(source).stem
    mask_path = _scratch_path(stem, "mask")
    preview_path = _scratch_path(stem, "preview")
    mask.save(mask_path)
    preview.save(preview_path)
    envelope = {
        "success": True,
        "phase": "preview",
        "source": source,
        "spec": json.loads(mask_spec.canonical_json(spec)),
        "bbox": list(box.as_tuple()),
        "coverage": round(coverage(mask), 4),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "mask_path": str(mask_path),
        "preview": str(preview_path),
        "next": (
            "if the red area is exactly what should be repainted, call again with the "
            "same selection and phase=commit; otherwise adjust boxes/keep_outs"
        ),
    }
    return envelope, str(preview_path)


def _region_commit(prompt: str, source: str, payload: dict) -> tuple[dict, str | None]:
    from loramake import mask_spec

    spec = mask_spec.parse_spec(_spec_fields(source, payload))

    fd, tmp = tempfile.mkstemp(prefix="loramake_image_", suffix=".json")
    os.close(fd)
    json_out = Path(tmp)
    argv = build_region_argv(
        prompt, source, mask_spec.canonical_json(spec), spec.mode, json_out
    )
    envelope = _spawn_gen(argv, json_out)
    envelope.setdefault("phase", "commit")
    path = envelope.get("host_image")
    return envelope, path if envelope.get("success") else None


def run_region(prompt: str, source: str | None, region: str) -> tuple[dict, str | None]:
    """Dispatch one region phase. Returns (envelope, image path to attach or None)."""
    sys.path.insert(0, str(REPO))
    from loramake.region_mask import MaskError

    try:
        payload = json.loads(region)
    except (TypeError, json.JSONDecodeError) as exc:
        return fail_envelope(f"region is not valid JSON: {exc}", help=REGION_HELP), None
    if not isinstance(payload, dict):
        return fail_envelope("region must be a JSON object", help=REGION_HELP), None

    phase = payload.get("phase")
    if phase not in REGION_PHASES:
        return (
            fail_envelope(
                f"region phase must be one of {', '.join(REGION_PHASES)}", help=REGION_HELP
            ),
            None,
        )
    if not source:
        return fail_envelope("region editing needs a source image path", help=REGION_HELP), None
    src = Path(source).expanduser()
    if not src.is_file():
        return fail_envelope(f"source not found: {source}"), None
    source = str(src)

    try:
        if phase == "inspect":
            return _region_inspect(source, payload)
        if phase == "crops":
            return _region_crops(source, payload)
        if phase == "preview":
            return _region_preview(source, payload)
        text = (prompt or "").strip()
        if not text:
            return fail_envelope("commit needs a prompt describing the repaint"), None
        return _region_commit(text, source, payload)
    except MaskError as exc:
        return fail_envelope(str(exc), phase=phase, help=REGION_HELP), None


def main() -> None:
    from mcp.server.mcpserver import Image, MCPServer

    mcp = MCPServer(
        "loramake-image",
        instructions=(
            "Local Z3Comfy Studio image generation. Call studio_generate for "
            "on-machine ComfyUI images and follow-up edits. "
            "xAI/Grok cloud images still use the built-in image_generate tool — "
            "do not replace that path with this one. "
            "Pass the previous image path as source when editing. "
            "For a localized repaint, keep source and walk the region phases: "
            "inspect (labelled grid + detector hits) -> preview (red mask "
            "overlay to confirm) -> commit (repaint only the masked area). "
            "For a structurally broken image (extra limbs, impossible joints, "
            "an object fused into the body, broken proportions) pass "
            "rescue=true with the source instead: one call re-renders it with "
            "its pose locked; add prompt only to correct a literalised token. "
            "Do not use terminal/gen.py for chat-visible local generation."
        ),
    )

    @mcp.tool(name="studio_generate", structured_output=False)
    def studio_generate(
        prompt: str,
        source: str | None = None,
        region: str | None = None,
        rescue: bool = False,
    ) -> list:
        """Generate or edit an image on the local Studio/ComfyUI pipeline.

        Not the xAI image_generate tool. Fresh image: pass only `prompt`.
        Whole-image follow-up edit: pass the previous image path as `source`
        plus the new instruction in `prompt`. Localized repaint: also pass
        `region` (JSON) and walk phases inspect -> preview -> commit; each
        phase returns an image to look at before any GPU is spent. Structure
        rescue (extra limbs, impossible joints, object fused into the body,
        broken proportions): pass `rescue=true` with `source` — one call
        re-renders the image with its pose locked; leave `prompt` empty to
        reuse the source prompt, or pass a corrected prompt when a token was
        literalised into an object. Keep using this same tool for every later
        local generation or edit.
        """
        if rescue and region is not None:
            envelope, image_path = (
                fail_envelope("pass either rescue or region, not both"),
                None,
            )
        elif rescue:
            envelope = run_rescue(prompt, source)
            image_path = envelope.get("host_image") if envelope.get("success") else None
        elif region is not None:
            envelope, image_path = run_region(prompt, source, region)
        else:
            envelope = run_generate(prompt, source)
            image_path = envelope.get("host_image") if envelope.get("success") else None
        blocks: list = [json.dumps(envelope, ensure_ascii=False)]
        if image_path and Path(image_path).is_file():
            blocks.append(Image(path=image_path))
        return blocks

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
