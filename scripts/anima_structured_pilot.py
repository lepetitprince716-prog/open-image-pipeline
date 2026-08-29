#!/usr/bin/env python3
"""Isolated Anima structured-conditioning Pilot (ZZZ only).

The engine accepts one strict ``loramake.anima-structured-task.v1`` JSON
document.  It never asks an LLM to write prompt text: the exact user source is
deterministically joined with the server-owned safety token and registered
identity trigger, then sent through one positive ``CLIPTextEncode`` node so
Qwen3 retains full cross-token context.

Default mode is plan-only.  ``--run`` requires a unique run id, a hash-valid
local pose asset, exact live loader enums, and the pinned custom-node signature.
``--control-mode off`` exists only for the bounded Pilot A/B and omits the
LLLite node completely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake.anima_structured_pilot import (  # noqa: E402
    ContractError,
    REGISTRY_PATH,
    canonical_json,
    load_registry,
    pilot_enabled,
    require_task,
    resolve_runtime,
    sha256_file,
)
from loramake.comfy import (  # noqa: E402
    atomic_write_text,
    emit_event,
    save_image_with_meta,
    submit_and_fetch,
)
from loramake.comfy_endpoints import (  # noqa: E402
    CLAUDE_LOOPBACK_COMFY_HOSTS,
    DIRECT_COMFY_HOSTS,
)
from loramake.config import OUTPUTS, ZZZ_HOST  # noqa: E402

OUTDIR = OUTPUTS / "anima" / "by_model"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
EXPECTED_LLLITE_INPUTS = {
    "model": "MODEL",
    "lllite_name": None,
    "image": "IMAGE",
    "strength": "FLOAT",
    "start_percent": "FLOAT",
    "end_percent": "FLOAT",
    "preserve_wrapper": "BOOLEAN",
}


def resolve_zzz_host() -> str:
    """Resolve only a ZZZ adapter; never let the generic override select Z3."""
    box_specific = (os.environ.get("ZZZ_HOST") or "").strip()
    candidate = (box_specific or os.environ.get("Z3_HOST") or ZZZ_HOST).rstrip("/")
    known_z3 = {
        DIRECT_COMFY_HOSTS["z3"].rstrip("/"),
        CLAUDE_LOOPBACK_COMFY_HOSTS["z3"].rstrip("/"),
    }
    allowed_zzz = {
        ZZZ_HOST.rstrip("/"),
        DIRECT_COMFY_HOSTS["zzz"].rstrip("/"),
        CLAUDE_LOOPBACK_COMFY_HOSTS["zzz"].rstrip("/"),
    }
    if box_specific:
        allowed_zzz.add(box_specific.rstrip("/"))
    if candidate in known_z3 or candidate not in allowed_zzz:
        raise ContractError([
            "This structured Pilot line is ZZZ-only; the configured endpoint is not an approved "
            "ZZZ adapter. Use the box-specific ZZZ_HOST override when required."
        ])
    return candidate


def _request_json(host: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(host.rstrip("/") + path, timeout=timeout) as response:
        raw = response.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned {type(value).__name__}, expected object")
    return value


def _combo_values(spec: object) -> set[str]:
    if not isinstance(spec, list) or not spec:
        return set()
    head = spec[0]
    if isinstance(head, list):
        return {str(item) for item in head}
    if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return {str(item) for item in spec[1].get("options", [])}
    return set()


def _required_input(info: dict[str, Any], node: str, field: str) -> object:
    try:
        return info[node]["input"]["required"][field]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"live node {node}.{field} is missing") from exc


def _assert_loader_value(host: str, node: str, field: str, value: str) -> None:
    info = _request_json(host, f"/object_info/{urllib.parse.quote(node)}")
    values = _combo_values(_required_input(info, node, field))
    if not values:
        raise RuntimeError(f"cannot read live enum {node}.{field}")
    if value not in values:
        raise RuntimeError(f"required asset absent from {node}.{field}: {value!r} ({len(values)} available)")


def live_preflight(host: str, runtime: dict[str, Any], *, control_mode: str) -> dict[str, Any]:
    profile = runtime["profile"]
    identity = runtime["identity"]
    _assert_loader_value(host, "UNETLoader", "unet_name", profile["base"]["loader_name"])
    _assert_loader_value(host, "LoraLoaderModelOnly", "lora_name", identity["loader_name"])
    _assert_loader_value(host, "CLIPLoader", "clip_name", profile["clip"]["loader_name"])
    _assert_loader_value(host, "VAELoader", "vae_name", profile["vae"]["loader_name"])

    observed: dict[str, Any] = {"host": host, "control_mode": control_mode}
    if control_mode == "on":
        node_id = profile["lllite"]["node_id"]
        info = _request_json(host, f"/object_info/{urllib.parse.quote(node_id)}")
        node = info.get(node_id)
        if not isinstance(node, dict):
            raise RuntimeError(f"required custom node is not registered: {node_id}")
        required = node.get("input", {}).get("required", {})
        if not isinstance(required, dict):
            raise RuntimeError(f"live node {node_id} has no required-input contract")
        for field, expected_type in EXPECTED_LLLITE_INPUTS.items():
            spec = required.get(field)
            if not isinstance(spec, list) or not spec:
                raise RuntimeError(f"live node {node_id}.{field} is missing")
            if expected_type is not None and spec[0] != expected_type:
                raise RuntimeError(
                    f"live node {node_id}.{field} type is {spec[0]!r}, expected {expected_type!r}"
                )
        values = _combo_values(required["lllite_name"])
        wanted = profile["lllite"]["loader_name"]
        if wanted not in values:
            raise RuntimeError(f"required LLLite weight absent from {node_id}.lllite_name: {wanted!r}")
        output = node.get("output")
        if output != ["MODEL"]:
            raise RuntimeError(f"live node {node_id} output contract is {output!r}, expected ['MODEL']")
        observed["lllite"] = {
            "node_id": node_id,
            "required_fields": sorted(required),
            "output": output,
            "weight_present": True,
        }
    return observed


def upload_control(host: str, path: Path, digest: str, *, prefix: str = "anima_pilot_pose",
                   suffix: str = ".png") -> str:
    """Upload one hash-named image to the box's ComfyUI input dir; return its server name."""
    name = f"{prefix}_{digest[:16]}{suffix}"
    hostname = urllib.parse.urlparse(host).hostname or "127.0.0.1"
    proc = subprocess.run(
        [
            "curl", "-sS", "--fail", "--noproxy", hostname, "--max-time", "120",
            "-F", f"image=@{path};filename={name}",
            "-F", "overwrite=true",
            f"{host}/upload/image",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "no response").strip()[-300:]
        raise RuntimeError(f"pose upload failed rc={proc.returncode}: {detail}")
    try:
        response = json.loads(proc.stdout)
        returned = response["name"]
        subfolder = response.get("subfolder") or ""
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"pose upload returned invalid JSON: {proc.stdout[:300]!r}") from exc
    if returned != name:
        raise RuntimeError(f"pose upload name mismatch: returned {returned!r}, expected {name!r}")
    return f"{subfolder}/{returned}" if subfolder else returned


def build_graph(task: dict[str, Any], runtime: dict[str, Any], *, control_mode: str,
                uploaded_pose: str | None = None) -> dict[str, Any]:
    if control_mode not in {"on", "off"}:
        raise ValueError("control_mode must be on or off")
    if control_mode == "on" and not uploaded_pose:
        raise ValueError("uploaded_pose is required when control_mode=on")
    profile = runtime["profile"]
    identity = runtime["identity"]
    conditioning = runtime["conditioning"]
    sampling = profile["sampling"]
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": profile["base"]["loader_name"], "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": identity["loader_name"],
                "strength_model": task["identity"]["strength"],
            },
        },
        "3": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["2", 0], "shift": sampling["shift"]},
        },
        "6": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": profile["clip"]["loader_name"],
                "type": profile["clip"]["type"],
                "device": "default",
            },
        },
        # F1 contract: trigger + exact source share this one positive encoder call.
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 0], "text": conditioning["text"]},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 0], "text": profile["negative_text"]},
        },
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": profile["width"], "height": profile["height"], "batch_size": 1},
        },
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]["loader_name"]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["11", 0]}},
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "anima_structured_pilot"},
        },
    }
    model_ref: list[Any] = ["3", 0]
    if control_mode == "on":
        graph["4"] = {"class_type": "LoadImage", "inputs": {"image": uploaded_pose, "upload": "image"}}
        graph["5"] = {
            "class_type": profile["lllite"]["node_id"],
            "inputs": {
                "model": model_ref,
                "lllite_name": profile["lllite"]["loader_name"],
                "image": ["4", 0],
                "strength": task["pose"]["strength"],
                "start_percent": task["pose"]["start_percent"],
                "end_percent": task["pose"]["end_percent"],
                "preserve_wrapper": True,
            },
        }
        model_ref = ["5", 0]
    graph["10"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["7", 0],
            "negative": ["8", 0],
            "latent_image": ["9", 0],
            "seed": task["seed"],
            "steps": sampling["steps"],
            "cfg": sampling["cfg"],
            "sampler_name": sampling["sampler"],
            "scheduler": sampling["scheduler"],
            "denoise": sampling["denoise"],
        },
    }
    return graph


def _load_task(args: argparse.Namespace) -> dict[str, Any]:
    try:
        raw = args.task_json if args.task_json is not None else Path(args.task_file).read_text(encoding="utf-8")
        task = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError([f"task JSON is unreadable: {type(exc).__name__}: {exc}"]) from exc
    return require_task(task, verify_local_assets=True)


def _write_direct_result(path: str | None, result: dict[str, Any]) -> None:
    if path:
        atomic_write_text(Path(path), json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-json", help="strict task JSON object")
    source.add_argument("--task-file", help="UTF-8 file containing one strict task JSON object")
    parser.add_argument("--control-mode", choices=("on", "off"), default="on",
                        help="on for production Pilot; off is evaluation-only A/B baseline")
    parser.add_argument("--run-id", default=None,
                        help="unique lowercase slug; required for --run unless LORAMAKE_ATTEMPT_ID is set")
    parser.add_argument("--run", action="store_true", help="submit to ZZZ (default prints a plan only)")
    parser.add_argument("--json-out", default=None, help="optional direct-engine result JSON")
    args = parser.parse_args()

    try:
        registry = load_registry()
        task = _load_task(args)
        runtime = resolve_runtime(task, registry=registry)
        host = resolve_zzz_host()
    except ContractError as exc:
        raise SystemExit("ERROR: " + "; ".join(exc.problems)) from exc

    run_id = args.run_id or os.environ.get("LORAMAKE_ATTEMPT_ID")
    if args.run and (not run_id or not RUN_ID_RE.fullmatch(run_id)):
        raise SystemExit("ERROR: --run requires --run-id matching [a-z0-9][a-z0-9_-]{0,63} "
                         "(app jobs supply LORAMAKE_ATTEMPT_ID automatically)")
    if args.run and not pilot_enabled(registry=registry):
        raise SystemExit(
            "ERROR: Anima structured Pilot is disabled; set "
            "LORAMAKE_ANIMA_STRUCTURED_PILOT_ENABLED=1 for an explicit Pilot run"
        )
    plan = {
        "schema": "loramake.anima-structured-plan.v1",
        "task_sha256": runtime["task_sha256"],
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "box": "zzz",
        "host": host,
        "control_mode": args.control_mode,
        "run_id": run_id,
        "profile": runtime["profile"]["id"],
        "identity_id": task["identity"]["lora_id"],
        "pose_id": task["pose"]["asset_id"],
        "source_sha256": task["source"]["sha256"],
        "effective_text_sha256": runtime["conditioning"]["sha256"],
        "positive_clip_text_encode_nodes": 1,
    }
    print("[plan] " + canonical_json(plan))
    if not args.run:
        _write_direct_result(args.json_out, {**plan, "status": "planned", "images": []})
        return

    started = time.monotonic()
    events: list[dict[str, Any]] = []
    try:
        observed = live_preflight(host, runtime, control_mode=args.control_mode)
        uploaded = None
        if args.control_mode == "on":
            uploaded = upload_control(host, Path(runtime["pose"]["absolute_path"]), runtime["pose"]["sha256"])
        graph = build_graph(task, runtime, control_mode=args.control_mode, uploaded_pose=uploaded)
        graph_sha256 = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
        fetched = submit_and_fetch(
            graph,
            label=f"anima-pilot-{args.control_mode}",
            host=host,
            timeout=1800,
            event_sink=events.append,
        )
        if not fetched:
            raise RuntimeError("ComfyUI completed without a valid PNG")
        raw, remote_image = fetched[0]
        model = (os.environ.get("DRIVING_MODEL") or "codex").strip().lower() or "codex"
        outdir = OUTDIR / model / "structured_pilot"
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"anima_pilot_{run_id}_{args.control_mode}_{task['seed']}"
        output = outdir / f"{stem}.png"
        if output.exists() or Path(str(output) + ".meta.json").exists():
            raise RuntimeError(f"no-clobber output already exists: {output}")
        meta = {
            "schema": "loramake.anima-structured-evidence.v1",
            "task": task,
            "task_sha256": runtime["task_sha256"],
            "registry_path": str(REGISTRY_PATH.relative_to(ROOT)),
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "effective_conditioning": runtime["conditioning"],
            "runtime": {
                "profile": runtime["profile"],
                "identity": runtime["identity"],
                "pose": {key: value for key, value in runtime["pose"].items() if key != "absolute_path"},
            },
            "control_mode": args.control_mode,
            "uploaded_pose": uploaded,
            "graph_sha256": graph_sha256,
            "live_preflight": observed,
            "remote_image": remote_image,
            "events": events,
            "elapsed_s": round(time.monotonic() - started, 3),
            "driving_model": model,
        }
        save_image_with_meta(output, raw, meta, overwrite=False)
        # save_image_with_meta already reports the PNG; keep this explicit call out of the path.
        result = {
            **plan,
            "status": "completed",
            "images": [str(output.resolve())],
            "sidecars": [str(Path(str(output) + ".meta.json").resolve())],
            "events": events,
            "elapsed_s": meta["elapsed_s"],
        }
        _write_direct_result(args.json_out, result)
        print(str(output.resolve()))
    except Exception as exc:
        emit_event("anima_pilot_error", host=host, label="anima-structured-pilot",
                   error_class=type(exc).__name__)
        _write_direct_result(
            args.json_out,
            {**plan, "status": "error", "images": [], "error": f"{type(exc).__name__}: {exc}"},
        )
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
