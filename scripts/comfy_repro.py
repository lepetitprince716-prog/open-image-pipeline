#!/usr/bin/env python3
"""Replay a ComfyUI-generated image from its embedded API graph — same-kernel repro.

Every ComfyUI render embeds its full API graph in the PNG (`prompt` text chunk,
plus the UI `workflow`). This engine extracts that graph, preflights it against
the live box, and resubmits it verbatim — the exact-recipe sibling of the Forge
line's metadata repro (docs/forge_line.md): same kernel, same graph, same seed.
Anima-line images (flow-matching, Z3) replay through this path; the SDE
cross-kernel caveat does not apply because nothing is translated.

Graph adaptations (each recorded in the repro report):
  - SaveImage filename_prefix -> "comfy_repro" (server-side naming only)
  - --seed replaces every integer `seed` / `noise_seed` input
  - --n > 1 re-rolls copies as base_seed + i (exact replay is copy 0); note that a
    graph carrying multiple DISTINCT seeds collapses to one uniform seed per copy

Fail-closed preflight: every class_type must exist on the box, and every
enum-typed input value (checkpoints, LoRAs, VAEs, samplers, detector models,
LoadImage inputs, ...) must be present in that node's live /object_info enum.

Usage:
  python3 scripts/comfy_repro.py --source render.png            # PLAN + preflight
  python3 scripts/comfy_repro.py --source render.png --run
  python3 scripts/gen.py comfy_repro --source render.png --run  # via the router
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_CHARACTER, by_model_dir  # noqa: E402  (bootstraps repo root)
from z3_comfy import (Z3_HOST, curl, submit_and_fetch,  # noqa: E402
                      save_image_with_meta, suggest_assets)
import image_recipe  # noqa: E402  (sibling; PNG chunk parsing + imgutils bridge)

import os  # noqa: E402

MODEL = os.environ.get("DRIVING_MODEL", "claude").strip().lower() or "claude"
REPORT_SCHEMA = "loramake.comfy-repro-report/v1"
SEED_FIELDS = ("seed", "noise_seed")
HONESTY = (
    "Same-kernel graph replay: with every asset present and the same box, the result is "
    "expected near-identical; residual drift comes from GPU/driver nondeterminism and "
    "custom-node version deltas. Cross-box replay additionally shifts with hardware."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a ComfyUI image from its embedded API graph (same-kernel repro)")
    parser.add_argument("--source", required=True, help="PNG with an embedded ComfyUI prompt graph")
    parser.add_argument("--n", type=int, default=1,
                        help="copies (copy 0 replays the source seed; extras re-roll base+i)")
    parser.add_argument("--seed", type=int, help="override every seed/noise_seed input")
    parser.add_argument("--host", help="ComfyUI endpoint (default Z3_HOST resolution)")
    parser.add_argument("--out-dir", help="output dir (default outputs/<char>/by_model/<model>/comfy_repro)")
    parser.add_argument("--timeout", type=float, help="poll budget per submission (default Z3_POLL_TIMEOUT)")
    parser.add_argument("--repro-report", help="report path (default <out-dir>/comfy_repro_<stamp>.json)")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="downgrade an unreachable /object_info preflight to a warning")
    parser.add_argument("--json", help="write the adapted graph + preflight notes to this path")
    parser.add_argument("--run", action="store_true", help="execute (default: PLAN only)")
    args = parser.parse_args()
    if not 1 <= args.n <= 16:
        parser.error("--n must be in 1..16")
    return args


def read_source_graph(path: str) -> dict:
    """The embedded API graph — the authoritative recipe for a ComfyUI render."""
    source = Path(path)
    if not source.is_file():
        sys.exit(f"ERROR: source image not found: {path}")
    try:
        chunks = image_recipe.decode_text_chunks(image_recipe.read_png_chunks(str(source)))
    except Exception as exc:
        sys.exit(f"ERROR: cannot read PNG chunks from {path}: {type(exc).__name__}")
    raw = chunks.get("prompt")
    if not raw:
        hint = " (an A1111/Forge 'parameters' image? use gen.py forge_repro)" \
            if "parameters" in chunks else ""
        sys.exit(f"ERROR: source carries no ComfyUI prompt graph{hint}.")
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit("ERROR: embedded prompt chunk is not valid JSON.")
    if not isinstance(graph, dict) or not graph:
        sys.exit("ERROR: embedded prompt graph is empty or not an object.")
    return graph


def adapt_graph(graph: dict, *, seed: int | None) -> tuple[dict, list[str]]:
    """Return (adapted copy, notes). Only server-side naming and seeds are touched."""
    adapted = copy.deepcopy(graph)
    notes: list[str] = []
    for node_id, node in adapted.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if node.get("class_type") == "SaveImage" and "filename_prefix" in inputs:
            notes.append(f"node {node_id}: filename_prefix "
                         f"{inputs['filename_prefix']!r} -> 'comfy_repro'")
            inputs["filename_prefix"] = "comfy_repro"
        if seed is not None:
            for field in SEED_FIELDS:
                if isinstance(inputs.get(field), int):
                    notes.append(f"node {node_id}: {field} {inputs[field]} -> {seed}")
                    inputs[field] = seed
    return adapted, notes


def source_seed(graph: dict) -> int | None:
    for node in graph.values():
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
            for field in SEED_FIELDS:
                value = node["inputs"].get(field)
                if isinstance(value, int):
                    return value
    return None


def _object_info(host: str, class_type: str) -> dict | None:
    """The /object_info spec for one node class; None = probe failed, {} = class absent."""
    from urllib.parse import quote
    raw = curl([f"{host.rstrip('/')}/object_info/{quote(class_type, safe='')}"])
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return info if isinstance(info, dict) else None


def preflight_graph(host: str, graph: dict, *, allow_unverified: bool) -> tuple[list[str], list[str]]:
    """Verify every node class exists and every enum-typed input value is on the box."""
    problems: list[str] = []
    notes: list[str] = []
    specs: dict[str, dict | None] = {}
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        if not class_type:
            problems.append(f"node {node_id} has no class_type")
            continue
        if class_type not in specs:
            specs[class_type] = _object_info(host, class_type)
        info = specs[class_type]
        if info is None:
            message = f"cannot verify node class {class_type!r} (/object_info unreachable)"
            (notes if allow_unverified else problems).append(
                message + ("" if allow_unverified else " — pass --allow-unverified to proceed"))
            continue
        if class_type not in info:
            problems.append(f"node {node_id}: class {class_type!r} is not installed on this box")
            continue
        spec_inputs = info[class_type].get("input") or {}
        merged: dict = {}
        for section in ("required", "optional"):
            section_spec = spec_inputs.get(section)
            if isinstance(section_spec, dict):
                merged.update(section_spec)
        node_inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        for field, value in node_inputs.items():
            if not isinstance(value, str):
                continue  # links and numbers are not enum values
            field_spec = merged.get(field)
            if not (isinstance(field_spec, (list, tuple)) and field_spec):
                continue
            head = field_spec[0]
            if isinstance(head, list):
                choices = {str(c) for c in head}
            elif head == "COMBO" and len(field_spec) > 1 and isinstance(field_spec[1], dict):
                choices = {str(c) for c in field_spec[1].get("options", [])}
            else:
                continue  # free string input, nothing to validate
            if choices and value not in choices:
                close = suggest_assets(value, choices)
                hint = f"; closest: {', '.join(close)}" if close else ""
                problems.append(f"node {node_id} ({class_type}.{field}): {value!r} "
                                f"not on this box ({len(choices)} available{hint})")
            elif choices:
                notes.append(f"{class_type}.{field}: {value!r} present")
    return problems, notes


def graph_summary(graph: dict) -> str:
    counts: dict[str, int] = {}
    for node in graph.values():
        if isinstance(node, dict):
            counts[str(node.get("class_type"))] = counts.get(str(node.get("class_type")), 0) + 1
    return ", ".join(f"{k}x{v}" if v > 1 else k for k, v in sorted(counts.items()))


def read_produced_graph(path: Path) -> dict | None:
    try:
        chunks = image_recipe.decode_text_chunks(image_recipe.read_png_chunks(str(path)))
        return json.loads(chunks.get("prompt") or "null")
    except Exception:
        return None


def verify(source: str, submitted: dict, saved: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for path in saved:
        produced_graph = read_produced_graph(path)
        graph_match = produced_graph == submitted if produced_graph is not None else None
        ccip = image_recipe.run_bridge("ccip", source, str(path))
        gray = image_recipe.run_bridge("graycorr", source, str(path))
        corr = gray.get("correlation") if isinstance(gray, dict) and gray.get("ok") else None
        ccip_diff = ccip.get("difference") if isinstance(ccip, dict) and ccip.get("ok") else None
        entries.append({"path": str(path), "graph_roundtrip_match": graph_match,
                        "graycorr": gray, "ccip": ccip})
        print(f"  [repro] {path.name}: graph_roundtrip="
              + {True: "MATCH", False: "DIFF", None: "unavailable"}[graph_match]
              + (f", graycorr {corr}" if corr is not None else ", graycorr unavailable")
              + (f", ccip diff {ccip_diff}" if ccip_diff is not None else ""))
    return entries


def main() -> None:
    args = parse_args()
    host = (args.host or Z3_HOST).rstrip("/")
    graph = read_source_graph(args.source)
    base_seed = args.seed if args.seed is not None else source_seed(graph)
    adapted, adapt_notes = adapt_graph(graph, seed=args.seed)
    problems, asset_notes = preflight_graph(host, adapted, allow_unverified=args.allow_unverified)

    print(f"[comfy-repro] host={host} nodes={len(adapted)} seed={base_seed}")
    print(f"  graph: {graph_summary(adapted)}")
    for note in adapt_notes:
        print(f"  [adapt] {note}")
    for problem in problems:
        print(f"  [PREFLIGHT-FAIL] {problem}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"graph": adapted, "problems": problems, "adaptations": adapt_notes},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [json] -> {args.json}")
    if problems:
        sys.exit(2)
    if not args.run:
        print("PLAN only — add --run to execute.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else by_model_dir(
        OUTPUT_CHARACTER, "comfy_repro", model=MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved: list[Path] = []
    submitted_first: dict | None = None
    started = time.monotonic()
    for index in range(args.n):
        copy_graph = adapted
        copy_seed = base_seed
        if index > 0:
            if base_seed is None:
                print("  [warn] no seed field in graph; extra copies would be identical — stopping at 1")
                break
            copy_seed = base_seed + index
            copy_graph, _ = adapt_graph(adapted, seed=copy_seed)
        if submitted_first is None:
            submitted_first = copy_graph
        results = submit_and_fetch(copy_graph, label=f"comfy_repro[{index}]", host=host,
                                   timeout=args.timeout, first_only=False)
        if not results:
            sys.exit(f"ERROR: submission {index} produced no output (see driver log above).")
        for j, (raw, item) in enumerate(results):
            path = out_dir / f"comfy_repro_{stamp}_s{copy_seed}_{index:02d}"
            path = path.with_name(path.name + (f"_{j}" if j else "") + ".png")
            meta = {"engine": "comfy_repro", "host": host, "source": args.source,
                    "seed": copy_seed, "character": OUTPUT_CHARACTER, "driving_model": MODEL,
                    "server_item": item}
            save_image_with_meta(path, raw, meta)
            saved.append(path)
    elapsed = round(time.monotonic() - started, 1)
    print(f"  [done] {len(saved)} image(s) in {elapsed}s")
    for path in saved:
        print(f"  {path}")

    entries = verify(args.source, submitted_first or adapted, saved)
    report_path = (Path(args.repro_report) if args.repro_report
                   else out_dir / f"comfy_repro_{stamp}.json")
    report_path.write_text(json.dumps({
        "schema": REPORT_SCHEMA, "source": args.source, "host": host,
        "honesty": HONESTY, "seed": base_seed, "adaptations": adapt_notes,
        "produced": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [repro] report -> {report_path}")


if __name__ == "__main__":
    main()
