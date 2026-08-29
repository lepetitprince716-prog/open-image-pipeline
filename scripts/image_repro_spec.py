#!/usr/bin/env python3
"""Build a reproduction spec for a non-AI / artist image (branch B).

When an image carries no recoverable generation metadata (see image_recipe.py),
faithful parameter recovery is impossible — the image was never produced by a
seed / sampler / checkpoint. This tool instead assembles the *mechanical* half of
a ``reproduction_spec_v1``:
  - WD14 booru tag candidates (general / character / rating) via the imgutils venv
  - a dominant-colour palette (hex + proportion) from the pixels
  - a tag-joined ``prompt_candidate`` (the tagger's literal output)

It deliberately does NOT pre-write the subjective ``visual_spec`` narrative fields
(subject / composition / must_preserve / ...). Per the image-analysis skill those
are visual-judgment fields filled by Grok / a human, not by this tool — the
``content_executor`` stays ``grok-in-build`` and those slots are left empty.
Control-map generation (canny / depth / openpose) and the actual reproduction loop
are left to the existing gen pipeline (z3_quality_gen.py); this tool only records
the recommended ``reference_strategy``.

SFW only. Reuses image_recipe.py for file facts and the imgutils venv bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_recipe as ir
import recipe_guess as rg

SCHEMA = "reproduction_spec_v1"
EXPERIMENT_SCHEMA = "image_experiment_record_v1"
_VISUAL_SPEC_FIELDS = (
    "subject", "composition", "pose_action", "camera_framing", "linework",
    "color_palette", "lighting", "background", "must_preserve", "avoid",
)
_REFERENCE_STRATEGIES = (
    "prompt_only", "ipadapter", "openpose", "canny", "depth", "regional", "inpaint", "mixed",
)
_CANDIDATE_FAMILIES = ("illustrious", "noobai", "pony", "sdxl", "anima", "flux", "unknown")


def extract_palette(path: str, colors: int = 6) -> list[dict[str, Any]]:
    """Dominant colours as ``{hex, proportion}``, largest share first. Purely
    mechanical — quantises a downscaled copy and counts cluster occupancy."""
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((256, 256))
            quant = rgb.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
            palette = quant.getpalette() or []
            counts = quant.getcolors() or []
    except Exception:
        return []
    total = sum(count for count, _ in counts) or 1
    out: list[dict[str, Any]] = []
    for count, index in sorted(counts, key=lambda c: c[0], reverse=True):
        if not isinstance(index, int):  # P-mode palette index is an int; guard the stubs
            continue
        base = index * 3
        r, g, b = palette[base:base + 3]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "proportion": round(count / total, 4)})
    return out


def wd14_tags(path: str) -> dict[str, Any] | None:
    result = ir.run_bridge("wd14", path)
    if result is None or not result.get("ok"):
        return None
    return {
        "rating": result.get("rating", {}),
        "general": result.get("general", {}),
        "character": result.get("character", {}),
    }


def tag_prompt_candidate(tags: dict[str, Any] | None, threshold: float = 0.35) -> str:
    """Comma-join general tags above a confidence threshold — the tagger's literal
    booru-prompt output, ordered by confidence. Not an aesthetic judgement."""
    if not tags:
        return ""
    general = tags.get("general", {})
    ranked = sorted(general.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [name.replace("_", " ") for name, score in ranked if score >= threshold]
    return ", ".join(chosen)


def build_spec(path: str, source_type: str, reference_strategy: str,
               candidate_family: str) -> dict[str, Any]:
    facts = ir.file_facts(path)
    tags = wd14_tags(path)
    palette = extract_palette(path)
    origin = ["file_system"]
    if tags is not None:
        origin.append("wd14")
    if palette:
        origin.append("palette")

    visual_spec: dict[str, Any] = {field: [] for field in _VISUAL_SPEC_FIELDS}
    visual_spec["color_palette"] = palette  # mechanical; the rest await Grok / human

    spec: dict[str, Any] = {
        "schema": SCHEMA,
        "source_path": facts["source_path"],
        "sha256": facts["sha256"],
        "source_type": source_type,
        "metadata_origin": origin,
        "visual_spec": visual_spec,
        "tag_candidates": tags,  # raw WD14 model output (general/character/rating)
        "prompt_candidate": tag_prompt_candidate(tags),
        "negative_candidate": "",
        "reference_strategy": reference_strategy,
        "candidate_family": candidate_family,
        "confidence": "low",
        "content_executor": "grok-in-build",
        "codex_content_review": False,
    }
    chars = [tag for tag, _ in rg.character_candidates((tags or {}).get("character"))]
    spec["inferred"] = rg.build_inferred(tags, lora_hints=rg.lookup_character_loras(chars))
    if tags is None:
        spec["warnings"] = ["imgutils venv unavailable — WD14 tags not extracted"]
    return spec


def new_experiment_record(spec: dict[str, Any]) -> dict[str, Any]:
    """Scaffold an image_experiment_record_v1 from a spec. The gen loop fills
    prompt_id / history_status / output_path / verdict after an actual run."""
    return {
        "schema": EXPERIMENT_SCHEMA,
        "source_sha256": spec["sha256"],
        "spec_schema": spec["schema"],
        "workflow_id": "",
        "asset_query": {},
        "prompt_summary": spec.get("prompt_candidate", ""),
        "negative_summary": spec.get("negative_candidate", ""),
        "prompt_id": "",
        "history_status": "",
        "output_path": "",
        "review_source": "user",
        "verdict": "",
        "approved": False,
    }


def write_outputs(spec: dict[str, Any], out_dir: str | None, log_experiment: bool) -> str:
    out_dir = out_dir or os.path.join(ir.OUTPUT_BASE, spec["sha256"][:12])
    os.makedirs(out_dir, exist_ok=True)
    ir._dump(os.path.join(out_dir, "repro_spec.json"), spec)
    ir._dump(os.path.join(out_dir, "palette.json"), spec["visual_spec"]["color_palette"])
    if spec.get("tag_candidates") is not None:
        ir._dump(os.path.join(out_dir, "tags.json"), spec["tag_candidates"])
    if log_experiment:
        log_path = os.path.join(out_dir, "iteration_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(new_experiment_record(spec), ensure_ascii=False) + "\n")
    return out_dir


def _print_summary(spec: dict[str, Any], out_dir: str | None) -> None:
    tags = spec.get("tag_candidates")
    n_general = len(tags.get("general", {})) if tags else 0
    print(f"Schema: {spec['schema']} | source_type: {spec['source_type']}")
    print(f"WD14 general tags: {n_general} | palette colours: {len(spec['visual_spec']['color_palette'])} "
          f"| strategy: {spec['reference_strategy']} | family: {spec['candidate_family']}")
    if spec.get("prompt_candidate"):
        print(f"Prompt candidate: {spec['prompt_candidate'][:120]}")
    if spec.get("warnings"):
        print(f"Warnings: {'; '.join(spec['warnings'])}")
    if out_dir:
        print(f"Written to: {out_dir}")


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Build a reproduction_spec_v1 (mechanical fields) for a non-AI / artist image.")
    parser.add_argument("image", help="path to a local image file")
    parser.add_argument("--artist", action="store_true",
                        help="mark source_type=artist_drawn_declared (else unknown)")
    parser.add_argument("--strategy", choices=_REFERENCE_STRATEGIES, default="ipadapter",
                        help="recommended reference strategy (default ipadapter)")
    parser.add_argument("--family", choices=_CANDIDATE_FAMILIES, default="unknown",
                        help="candidate base-model family (default unknown)")
    parser.add_argument("--json", action="store_true", help="print the spec to stdout")
    parser.add_argument("--log-experiment", action="store_true",
                        help="append an image_experiment_record_v1 scaffold to iteration_log.jsonl")
    parser.add_argument("--out-dir", help="output directory (default outputs/image_repro/<sha12>/)")
    parser.add_argument("--no-write", action="store_true", help="do not write output files")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"not a file: {args.image}")

    source_type = "artist_drawn_declared" if args.artist else "unknown"
    spec = build_spec(args.image, source_type, args.strategy, args.family)
    out_dir = None if args.no_write else write_outputs(spec, args.out_dir, args.log_experiment)

    if args.json:
        json.dump(spec, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        _print_summary(spec, out_dir)


if __name__ == "__main__":
    main()
