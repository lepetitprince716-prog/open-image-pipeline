#!/usr/bin/env python3
"""Six-seed A/B acceptance harness for the isolated Anima structured Pilot.

The harness exercises the public ``gen.py anima_structured_pilot`` route, pairs
control-off and control-on renders by seed, re-detects body keypoints with the
live ZZZ DWPose node, and checks the pinned identity anchor with CCIP.  It never
changes the frozen thresholds and leaves the Pilot feature flag off outside its
child generation processes.

Default mode prints a deterministic plan.  ``--run`` is the only GPU-mutating
mode and refuses to reuse an existing evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake import anima_structured_pilot as _anima_contract  # noqa: E402
from loramake import sdxl_structured_pilot as _sdxl_contract  # noqa: E402
from loramake.anima_structured_pilot import (  # noqa: E402
    ContractError,
    canonical_json,
    sha256_file,
)
from loramake.comfy import free_vram  # noqa: E402
from loramake.config import OUTPUTS  # noqa: E402
from scripts.anima_structured_pilot import resolve_zzz_host, upload_control  # noqa: E402
from scripts.consistency_check import DEFAULT_VENV_PY, ccip_many  # noqa: E402
from scripts.contact_sheet import build_sheet  # noqa: E402

SCHEMA = "loramake.anima-structured-eval.v1"
FAMILIES = {
    "anima": {
        "contract": _anima_contract,
        "intent": "anima_structured_pilot",
        "registry_file": "anima_structured_pilot.json",
        "evidence_subdir": "anima_structured_pilot",
        "eval_schema": "loramake.anima-structured-eval.v1",
    },
    "sdxl": {
        "contract": _sdxl_contract,
        "intent": "sdxl_structured_pilot",
        "registry_file": "sdxl_structured_pilot.json",
        "evidence_subdir": "sdxl_structured_pilot",
        "eval_schema": "loramake.sdxl-structured-eval.v1",
    },
}
DEFAULT_SEEDS = (804001, 804002, 804003, 804004, 804005, 804006)
DEFAULT_SOURCE = (
    "one adult woman, solo, full body, centered in a quiet neutral studio, "
    "fully clothed, eye-level camera"
)
ABSOLUTE_POSE_LIMIT = 0.12
RELATIVE_RATIO_LIMIT = 0.85
LOW_BASELINE_DEGRADATION_LIMIT = 0.01
MIN_IMPROVED_PAIRS = 4
MIN_ABSOLUTE_PAIRS_WHEN_BASELINE_LOW = 5
MIN_CCIP_HOLDS = 5
BODY_JOINT_COUNT = 18


class EvaluationError(RuntimeError):
    """Evidence could not be produced or did not satisfy a measurable precondition."""


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _median_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(present) if present else None


def _request_json(host: str, path: str, *, payload: dict[str, Any] | None = None,
                  timeout: float = 30.0) -> dict[str, Any]:
    raw = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=raw,
        headers={"Content-Type": "application/json"} if raw is not None else {},
        method="POST" if raw is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} returned {type(value).__name__}, expected object")
    return value


def _request_bytes(host: str, path: str, *, timeout: float = 60.0) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(host.rstrip("/") + path, timeout=timeout) as response:
        return response.read()


def _upload(host: str, path: Path, prefix: str) -> str:
    return upload_control(
        host,
        path,
        _sha256(path),
        prefix=prefix,
        suffix=path.suffix.lower() if path.suffix else ".png",
    )


def _dwpose_graph(image_name: str, resolution: int,
                  bbox_detector: str = "yolox_l.onnx") -> dict[str, Any]:
    # Deliberately narrower than pose_library._extract_graph, which enables hands and face so the
    # stored skeleton PNG carries them. Scoring reads only the 18 body joints, and the hand/face
    # estimators cost time without changing a single body coordinate.
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "2": {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["1", 0],
                "detect_hand": "disable",
                "detect_body": "enable",
                "detect_face": "disable",
                "resolution": resolution,
                "bbox_detector": bbox_detector,
                # pose_library.py omitted this optional input, so the Python default was ONNX.
                "pose_estimator": "dw-ll_ucoco_384.onnx",
                "scale_stick_for_xinsr_cn": "disable",
            },
        },
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
    }


def _wait_history(host: str, prompt_id: str, timeout: float = 360.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _request_json(host, f"/history/{urllib.parse.quote(prompt_id)}", timeout=20)
        item = history.get(prompt_id)
        if isinstance(item, dict) and item.get("outputs"):
            return item
        status = item.get("status", {}) if isinstance(item, dict) else {}
        if status.get("status_str") == "error":
            raise EvaluationError(f"DWPose prompt {prompt_id} failed: {status}")
        time.sleep(1.0)
    raise EvaluationError(f"DWPose prompt timed out: {prompt_id}")


def detect_pose(host: str, path: Path, *, resolution: int = 1024,
                bbox_detector: str = "yolox_l.onnx") -> dict[str, Any]:
    uploaded = _upload(host, path, "anima_eval")
    submitted = _request_json(
        host, "/prompt",
        payload={"prompt": _dwpose_graph(uploaded, resolution, bbox_detector)},
    )
    prompt_id = submitted.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise EvaluationError(f"DWPose submit returned no prompt_id for {path.name}")
    item = _wait_history(host, prompt_id)
    outputs = item.get("outputs", {})
    pose_output = outputs.get("2", {})
    encoded = pose_output.get("openpose_json") if isinstance(pose_output, dict) else None
    if not isinstance(encoded, list) or not encoded or not isinstance(encoded[0], str):
        raise EvaluationError(f"DWPose history lacks openpose_json for {path.name}")
    try:
        documents = json.loads(encoded[0])
        document = documents[0]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise EvaluationError(f"DWPose returned malformed keypoints for {path.name}") from exc
    if not isinstance(document, dict):
        raise EvaluationError(f"DWPose keypoint document is not an object for {path.name}")

    preview = outputs.get("3", {})
    images = preview.get("images") if isinstance(preview, dict) else None
    preview_size = None
    if isinstance(images, list) and images and isinstance(images[0], dict):
        descriptor = images[0]
        query = urllib.parse.urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "temp"),
            }
        )
        raw = _request_bytes(host, "/view?" + query, timeout=90)
        with Image.open(io.BytesIO(raw)) as image:
            preview_size = [int(image.width), int(image.height)]
    return {
        "prompt_id": prompt_id,
        "uploaded": uploaded,
        "document": document,
        "preview_size": preview_size,
    }


def _triples(raw: object) -> list[tuple[float, float, float]]:
    if not isinstance(raw, list):
        return []
    if raw and all(isinstance(item, list) and len(item) >= 3 for item in raw):
        values = raw
    else:
        if len(raw) % 3:
            return []
        values = [raw[index:index + 3] for index in range(0, len(raw), 3)]
    out: list[tuple[float, float, float]] = []
    for item in values[:BODY_JOINT_COUNT]:
        try:
            x, y, confidence = float(item[0]), float(item[1]), float(item[2])
        except (TypeError, ValueError):
            out.append((0.0, 0.0, 0.0))
            continue
        out.append((x, y, confidence))
    return out


def _largest_person(document: dict[str, Any]) -> list[tuple[float, float, float]]:
    people = document.get("people")
    if not isinstance(people, list):
        return []
    ranked: list[tuple[float, list[tuple[float, float, float]]]] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        points = _triples(person.get("pose_keypoints_2d"))
        visible = [(x, y) for x, y, confidence in points if confidence > 0 and (x != 0 or y != 0)]
        if len(visible) < 4:
            continue
        xs = [point[0] for point in visible]
        ys = [point[1] for point in visible]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        ranked.append((area, points))
    return max(ranked, key=lambda item: item[0])[1] if ranked else []


def normalize_target(detection: dict[str, Any], *, width: int, height: int,
                     frame: str, offset: float) -> list[tuple[float, float, float]]:
    if frame != "full":
        raise EvaluationError("Pilot evaluator currently accepts only frame=full references")
    document = detection["document"]
    points = _largest_person(document)
    if not points:
        raise EvaluationError("target DWPose document has no measurable person")
    canvas_width = int(document.get("canvas_width") or 0)
    canvas_height = int(document.get("canvas_height") or 0)
    preview_size = detection.get("preview_size")
    if canvas_width <= 0 or canvas_height <= 0 or not preview_size:
        raise EvaluationError("target DWPose canvas/preview dimensions are missing")
    preview_width, preview_height = map(int, preview_size)

    # First map JSON coordinates onto the raw skeleton PNG returned by DWPose.
    raw_points = [
        (x * preview_width / canvas_width, y * preview_height / canvas_height, confidence)
        for x, y, confidence in points
    ]
    # Then reproduce pose_prep.prep(frame=full): fill height without distortion and center crop/pad.
    scale_y = height / preview_height
    normalized_width = max(1, round(preview_width * scale_y))
    scale_x = normalized_width / preview_width
    if normalized_width >= width:
        free = normalized_width - width
        left = round((free / 2) * (1 - offset))
        x_shift = -min(max(left, 0), free)
    else:
        free = width - normalized_width
        x = round((free / 2) * (1 + offset))
        x_shift = min(max(x, 0), free)
    return [(x * scale_x + x_shift, y * scale_y, confidence) for x, y, confidence in raw_points]


def pose_error(target: list[tuple[float, float, float]],
               candidate_document: dict[str, Any]) -> dict[str, Any]:
    candidate = _largest_person(candidate_document)
    if not candidate:
        return {"ok": False, "reason": "no_person", "joint_count": 0, "median_error": None}
    target_visible = [(x, y) for x, y, confidence in target if confidence > 0 and (x != 0 or y != 0)]
    if len(target_visible) < 4:
        raise EvaluationError("target has fewer than four visible body joints")
    xs = [point[0] for point in target_visible]
    ys = [point[1] for point in target_visible]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diagonal <= 0:
        raise EvaluationError("target body bounding box has zero diagonal")
    pairs = []
    for target_point, candidate_point in zip(target, candidate, strict=False):
        tx, ty, tc = target_point
        cx, cy, cc = candidate_point
        if tc > 0 and cc > 0 and (tx != 0 or ty != 0) and (cx != 0 or cy != 0):
            pairs.append(((tx, ty), (cx, cy)))
    distances = [math.hypot(cx - tx, cy - ty) / diagonal for (tx, ty), (cx, cy) in pairs]
    if len(distances) < 4:
        return {
            "ok": False,
            "reason": "too_few_common_joints",
            "joint_count": len(distances),
            "median_error": None,
        }
    return {
        "ok": True,
        "reason": "",
        "joint_count": len(distances),
        "target_bbox_diagonal": diagonal,
        "median_error": statistics.median(distances),
        "joint_errors": distances,
        "aligned_median_error": aligned_pose_error(pairs, diagonal),
    }


def aligned_pose_error(pairs: list[tuple[tuple[float, float], tuple[float, float]]],
                       diagonal: float) -> float | None:
    """Median residual after removing translation and uniform scale. Diagnostic only.

    ``median_error`` answers the gate's question: did the render land on the target skeleton,
    placement and size included. That single scalar cannot say whether a wrong-but-recognisable
    pose is emerging, because a perfect limb configuration drawn at a different position or size
    still scores badly. This aligns the candidate onto the target by centroid and best-fit uniform
    scale first, so what remains is limb configuration alone. Rotation is deliberately not removed:
    a rotated figure is a different pose in image space, not a framing difference.
    """
    if len(pairs) < 4 or diagonal <= 0:
        return None
    target_mean_x = statistics.fmean(point[0][0] for point in pairs)
    target_mean_y = statistics.fmean(point[0][1] for point in pairs)
    candidate_mean_x = statistics.fmean(point[1][0] for point in pairs)
    candidate_mean_y = statistics.fmean(point[1][1] for point in pairs)
    centered = [
        ((tx - target_mean_x, ty - target_mean_y), (cx - candidate_mean_x, cy - candidate_mean_y))
        for (tx, ty), (cx, cy) in pairs
    ]
    spread = sum(cx * cx + cy * cy for _, (cx, cy) in centered)
    if spread <= 0:
        return None
    dot = sum(tx * cx + ty * cy for (tx, ty), (cx, cy) in centered)
    scale = dot / spread
    if scale <= 0:
        # Anti-correlated with the target: no uniform scale maps one onto the other.
        return None
    residuals = [
        math.hypot(scale * cx - tx, scale * cy - ty) / diagonal for (tx, ty), (cx, cy) in centered
    ]
    return statistics.median(residuals)


def acceptance(off_errors: list[float], on_errors: list[float], ccip_holds: list[bool]) -> dict[str, Any]:
    if len(off_errors) != 6 or len(on_errors) != 6 or len(ccip_holds) != 6:
        raise EvaluationError("acceptance requires exactly six complete A/B/CCIP rows")
    off_median = statistics.median(off_errors)
    on_median = statistics.median(on_errors)
    improved = sum(on < off for on, off in zip(on_errors, off_errors, strict=True))
    identity_holds = sum(ccip_holds)
    absolute_pairs = sum(value <= ABSOLUTE_POSE_LIMIT for value in on_errors)
    baseline_low = off_median <= ABSOLUTE_POSE_LIMIT
    if baseline_low:
        pose_checks = {
            "absolute_median": on_median <= ABSOLUTE_POSE_LIMIT,
            "degradation_limit": on_median <= off_median + LOW_BASELINE_DEGRADATION_LIMIT,
            "absolute_pairs": absolute_pairs >= MIN_ABSOLUTE_PAIRS_WHEN_BASELINE_LOW,
        }
    else:
        pose_checks = {
            "absolute_median": on_median <= ABSOLUTE_POSE_LIMIT,
            "relative_ratio": on_median <= RELATIVE_RATIO_LIMIT * off_median,
            "improved_pairs": improved >= MIN_IMPROVED_PAIRS,
        }
    identity_check = identity_holds >= MIN_CCIP_HOLDS
    passed = all(pose_checks.values()) and identity_check
    return {
        "status": "PASS" if passed else "HOLD",
        "off_median": off_median,
        "on_median": on_median,
        "on_to_off_ratio": on_median / off_median if off_median else None,
        "baseline_low": baseline_low,
        "improved_pairs": improved,
        "absolute_pairs": absolute_pairs,
        "identity_holds": identity_holds,
        "pose_checks": pose_checks,
        "identity_check": identity_check,
    }


def _run_generation(task: dict[str, Any], *, run_id: str, mode: str,
                    result_path: Path, family: dict[str, Any]) -> Path:
    # DRIVING_MODEL is inherited, not forced: the engine already defaults to codex when it is
    # unset, and forcing it here filed every run under codex whoever actually drove it.
    env = {**os.environ, family["contract"].FEATURE_ENV: "1"}
    # --no-wait: the harness is strictly serial and verifies an empty queue before the first
    # render, so the only thing the router's idle probe would see mid-run is our own resident
    # model memory (--no-free keeps it loaded). The SDXL family's ~10GB residency sits below
    # the probe's free-VRAM heuristic and would deadlock the run; queue emptiness is the
    # anti-clobber property that actually matters here.
    command = [
        sys.executable,
        str(ROOT / "scripts" / "gen.py"),
        family["intent"],
        "--box", "zzz",
        "--task-json", canonical_json(task),
        "--run",
        "--no-free",
        "--no-wait",
        "--json-out", str(result_path),
        "--",
        "--control-mode", mode,
        "--run-id", run_id,
    ]
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, timeout=2100)
    if proc.returncode != 0:
        raise EvaluationError(f"gen.py {mode} seed={task['seed']} exited {proc.returncode}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        images = result["new_files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvaluationError(f"invalid gen result for {mode} seed={task['seed']}") from exc
    if result.get("rc") != 0 or not isinstance(images, list) or len(images) != 1:
        raise EvaluationError(f"gen result is incomplete for {mode} seed={task['seed']}: {result}")
    image = Path(images[0]).resolve()
    if not image.is_file() or image.suffix.lower() != ".png":
        raise EvaluationError(f"generated image is missing for {mode} seed={task['seed']}")
    return image


def _verify_reference(entry: dict[str, Any]) -> Path:
    reference = entry.get("evaluation_reference")
    if not isinstance(reference, dict):
        raise EvaluationError("pose registry lacks evaluation_reference")
    path = (ROOT / str(reference.get("path", ""))).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise EvaluationError("evaluation_reference path is missing or escapes the repository")
    if path.stat().st_size != reference.get("bytes") or _sha256(path) != reference.get("sha256"):
        raise EvaluationError("evaluation_reference bytes/hash do not match the registry")
    with Image.open(path) as image:
        if [image.width, image.height] != [reference.get("width"), reference.get("height")]:
            raise EvaluationError("evaluation_reference dimensions do not match the registry")
    return path


def _write_overlay(path: Path, skeleton: Path, points: list[tuple[float, float, float]]) -> None:
    with Image.open(skeleton).convert("RGB") as base:
        draw = ImageDraw.Draw(base)
        for index, (x, y, confidence) in enumerate(points):
            if confidence > 0 and (x != 0 or y != 0):
                radius = 5
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="white", width=2)
                draw.text((x + 6, y - 6), str(index), fill="white")
        base.save(path)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(values) != 6 or len(set(values)) != 6 or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("exactly six distinct non-negative seeds are required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="unique evidence directory slug")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="exact source text shared by every pair")
    parser.add_argument(
        "--source-domain",
        choices=("tags", "natural_language", "mixed"),
        default="mixed",
        help="prompt domain of --source, recorded in the report. Anima treats the three domains as "
             "near-equivalent on aestheticV11, but on Miao8 pure natural language reproduced a pose "
             "drift (3/3 seeds), so a pose experiment must state which domain it ran in.",
    )
    parser.add_argument("--pose-id", default="arabesque_pointe_centered")
    parser.add_argument("--family", choices=sorted(FAMILIES), default="anima",
                        help="which structured-Pilot family to evaluate (engine route + registry)")
    parser.add_argument("--identity-id", default=None,
                        help="registered identity id; defaults to the family registry's sole identity")
    parser.add_argument("--run", action="store_true", help="execute 12 renders and live scoring")
    args = parser.parse_args()

    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in args.run_id):
        raise SystemExit("ERROR: --run-id must contain only lowercase ASCII letters, digits, - and _")
    family = FAMILIES[args.family]
    contract = family["contract"]
    try:
        registry = contract.load_registry()
        host = resolve_zzz_host()
    except ContractError as exc:
        raise SystemExit("ERROR: " + "; ".join(exc.problems)) from exc
    if args.pose_id not in registry["poses"]:
        raise SystemExit(f"ERROR: unregistered pose: {args.pose_id}")
    identity_id = args.identity_id or next(iter(registry["identities"]))
    if identity_id not in registry["identities"]:
        raise SystemExit(f"ERROR: unregistered identity: {identity_id}")
    selection = {
        "schema": contract.SELECTION_SCHEMA,
        "identity_id": identity_id,
        "pose_id": args.pose_id,
        "layout": {"canvas": "portrait", "framing": "full_body", "placement": "center"},
    }
    tasks = [contract.compile_task(args.source, selection, seed, registry=registry)
             for seed in args.seeds]
    evidence_dir = OUTPUTS / "_analysis" / family["evidence_subdir"] / args.run_id
    plan = {
        "schema": family["eval_schema"],
        "status": "planned",
        "run_id": args.run_id,
        "family": args.family,
        "identity_id": identity_id,
        "host": host,
        "seeds": list(args.seeds),
        "source": args.source,
        "source_sha256": hashlib.sha256(args.source.encode("utf-8")).hexdigest(),
        "source_domain": args.source_domain,
        "pose_id": args.pose_id,
        "registry_sha256": _sha256(ROOT / "scripts" / family["registry_file"]),
        "renders": 12,
        "evidence_dir": str(evidence_dir.resolve()),
        "thresholds": {
            "absolute_pose_limit": ABSOLUTE_POSE_LIMIT,
            "relative_ratio_limit": RELATIVE_RATIO_LIMIT,
            "low_baseline_degradation_limit": LOW_BASELINE_DEGRADATION_LIMIT,
            "min_improved_pairs": MIN_IMPROVED_PAIRS,
            "min_absolute_pairs_when_baseline_low": MIN_ABSOLUTE_PAIRS_WHEN_BASELINE_LOW,
            "min_ccip_holds": MIN_CCIP_HOLDS,
        },
    }
    print("[plan] " + canonical_json(plan), flush=True)
    if not args.run:
        return
    if evidence_dir.exists():
        raise SystemExit(f"ERROR: evidence directory already exists: {evidence_dir}")
    # Verify the local scoring reference BEFORE burning twelve renders: the target photo is an
    # ignored workstation asset, and discovering it missing after the GPU pass wastes the run.
    pose_entry = registry["poses"][args.pose_id]
    reference_path = _verify_reference(pose_entry)
    # The renders below pass --no-wait to the router (our own --no-free model residency would
    # otherwise trip its free-VRAM heuristic), so the anti-clobber check happens once here:
    # refuse to start over anyone else's queued work.
    queue = _request_json(host, "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise SystemExit(
            f"ERROR: {host} queue is not empty "
            f"(running={len(queue.get('queue_running', []))}, "
            f"pending={len(queue.get('queue_pending', []))}); refusing to start"
        )
    evidence_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    try:
        for task in tasks:
            seed = task["seed"]
            row: dict[str, Any] = {"seed": seed, "task": task}
            for mode in ("off", "on"):
                result_path = evidence_dir / f"gen_{seed}_{mode}.json"
                image = _run_generation(
                    task,
                    run_id=f"{args.run_id}-{seed}",
                    mode=mode,
                    result_path=result_path,
                    family=family,
                )
                row[mode] = {"image": str(image), "sha256": _sha256(image)}
            rows.append(row)

        visual_review = evidence_dir / "visual_review.png"
        build_sheet(
            [row[mode]["image"] for row in rows for mode in ("off", "on")],
            str(visual_review),
            cols=2,
            thumb=512,
        )

        # The render residency (--no-free) is no longer needed once all twelve images exist.
        # Give the DWPose ONNX stack a clean box before scoring: under the SDXL family's ~10GB
        # residual, the bbox detector intermittently returned zero detections on obviously
        # valid full-body figures.
        free_vram(host)

        target_detection = detect_pose(
            host,
            reference_path,
            resolution=int(pose_entry["evaluation_reference"]["detect_resolution"]),
        )
        target_points = normalize_target(
            target_detection,
            width=int(registry["profile"]["width"]),
            height=int(registry["profile"]["height"]),
            frame=str(pose_entry["evaluation_reference"]["frame"]),
            offset=float(pose_entry["evaluation_reference"]["offset"]),
        )
        overlay = evidence_dir / "target_keypoints_overlay.png"
        _write_overlay(overlay, (ROOT / pose_entry["path"]).resolve(), target_points)

        for row in rows:
            for mode in ("off", "on"):
                detected = detect_pose(host, Path(row[mode]["image"]), resolution=1024)
                score = pose_error(target_points, detected["document"])
                if not score.get("ok") and score.get("reason") == "no_person":
                    # The yolox person detector has a deterministic blind spot on some
                    # obviously valid single-figure renders (observed on iLustMix output:
                    # zero detections on a clean centered full body). Fall back to
                    # whole-image pose estimation, which changes the graph — an identical
                    # resubmission would only hit ComfyUI's execution cache and return
                    # empty outputs. The fallback is recorded per row.
                    detected = detect_pose(
                        host, Path(row[mode]["image"]), resolution=1024, bbox_detector="None"
                    )
                    score = pose_error(target_points, detected["document"])
                    score["bbox_fallback"] = "None"
                row[mode]["pose"] = score
                row[mode]["dwpose_prompt_id"] = detected["prompt_id"]

        on_images = [row["on"]["image"] for row in rows]
        anchor = (ROOT / registry["identities"][identity_id]["ccip_anchor"]["path"]).resolve()
        ccip = ccip_many(os.environ.get("IMGUTILS_VENV_PY", DEFAULT_VENV_PY), str(anchor), on_images)
        for row in rows:
            score = ccip.get(row["on"]["image"])
            if not isinstance(score, dict) or not isinstance(score.get("same_character"), bool):
                raise EvaluationError(f"CCIP failed for seed {row['seed']}")
            row["on"]["ccip"] = score

        off_errors = [row["off"]["pose"]["median_error"] for row in rows]
        on_errors = [row["on"]["pose"]["median_error"] for row in rows]
        if not all(isinstance(value, (int, float)) for value in off_errors + on_errors):
            raise EvaluationError("one or more DWPose scores are incomplete")
        ccip_holds = [bool(row["on"]["ccip"]["same_character"]) for row in rows]
        gate = acceptance(off_errors, on_errors, ccip_holds)
        aligned = {
            mode: [row[mode]["pose"].get("aligned_median_error") for row in rows]
            for mode in ("off", "on")
        }
        report = {
            **plan,
            "status": gate["status"],
            "aligned_diagnostic": {
                "note": "translation+scale removed; reported for interpretation, never gates",
                "off_median": _median_or_none(aligned["off"]),
                "on_median": _median_or_none(aligned["on"]),
                "off": aligned["off"],
                "on": aligned["on"],
            },
            "completed_at_epoch": time.time(),
            "target": {
                "skeleton": pose_entry,
                "reference_path": str(reference_path),
                "reference_sha256": _sha256(reference_path),
                "dwpose_prompt_id": target_detection["prompt_id"],
                "json_canvas": [
                    target_detection["document"].get("canvas_width"),
                    target_detection["document"].get("canvas_height"),
                ],
                "preview_size": target_detection["preview_size"],
                "overlay": str(overlay),
            },
            "rows": rows,
            "gate": gate,
            "ccip_anchor": {"path": str(anchor), "sha256": _sha256(anchor)},
            "visual_review": {"path": str(visual_review), "sha256": _sha256(visual_review)},
        }
        report_path = evidence_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[result] " + canonical_json({"status": gate["status"], "gate": gate}), flush=True)
        print(report_path.resolve())
    except Exception as exc:
        failure = {**plan, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "rows": rows}
        (evidence_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    finally:
        free_vram(host)


if __name__ == "__main__":
    main()
