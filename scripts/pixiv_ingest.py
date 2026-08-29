#!/usr/bin/env python3
"""Ingest a Pixiv work → site metadata + opportunistic recipe + reproduction spec.

Pixiv strips embedded generation metadata on upload, and AI artists almost never
disclose their recipe in the caption (0/14 on the daily_ai ranking — verified).
So an arbitrary Pixiv AI image's true recipe is generally **unrecoverable**; the
realistic target is visual reproduction. For a Pixiv work this tool:

  1. Fetches the image + site metadata (caption / tags / illust_ai_type / artist)
     via gallery-dl (`--cookies-from-browser`, Safari by default).
  2. Opportunistically scans the caption for a disclosed recipe (inline `<lora:>`
     tags, an A1111 param block, base-model mentions) — kept because a minority of
     artists do paste it.
  3. Routes:
       - a disclosed recipe  → `ai_image_forensics_v1` (metadata_origin pixiv_site)
       - otherwise           → `reproduction_spec_v1`, enriched with the Pixiv
         content tags (subject/character/series) and **Civitai artist-style-LoRA
         candidates** — people publish style LoRAs of popular Pixiv AI artists, the
         most reliable reproduction lever in this domain.

Auth: `PIXIV_COOKIES_BROWSER` (default `safari`). SFW only; Civitai stays on
civitai.com. The actual reproduction gen loop is a separate later step
(z3_quality_gen). Style-LoRA candidates are search hits by artist name — they are
*unconfirmed* and need human/visual confirmation before use.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import civitai
import civitai_recipe as cr
import image_recipe as ir
import image_repro_spec as rs

COOKIES_BROWSER = os.environ.get("PIXIV_COOKIES_BROWSER", "safari")
DEFAULT_SRC_DIR = os.path.join(ir.OUTPUT_BASE, "_pixiv_src")
_ARTWORK_ID = re.compile(r"(?:artworks/|illust_id=)(\d{4,})|^(\d{4,})$")
_BASE_MODEL_HINTS = (
    "novelai", "nai diffusion", "animagine", "illustrious", "noobai", "pony",
    "pony diffusion", "flux", "stable diffusion", "sdxl", "kohaku",
)


def parse_artwork_id(arg: str) -> str | None:
    match = _ARTWORK_ID.search(arg.strip())
    if match:
        return match.group(1) or match.group(2)
    return None


def fetch_pixiv(arg: str, dest: str) -> tuple[str | None, dict[str, Any]]:
    """Download the first image of a Pixiv work + its metadata sidecar. Returns
    (image_path, metadata) or (None, {}) on failure."""
    artwork_id = parse_artwork_id(arg)
    if not artwork_id:
        return None, {}
    url = f"https://www.pixiv.net/artworks/{artwork_id}"
    os.makedirs(dest, exist_ok=True)
    cmd = ["gallery-dl", "--cookies-from-browser", COOKIES_BROWSER, "-D", dest,
           "--write-metadata", "--range", "1-1", url]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except Exception:
        return None, {}
    images = [f for f in os.listdir(dest)
              if f.startswith(artwork_id) and not f.endswith(".json")]
    if not images:
        return None, {}
    image_path = os.path.join(dest, sorted(images)[0])
    metadata: dict[str, Any] = {}
    sidecar = image_path + ".json"
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as fh:
                metadata = json.load(fh)
        except Exception:
            metadata = {}
    return image_path, metadata


def clean_caption(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return re.sub(r"<[^>]+>", "", html.unescape(raw)).strip()


def scan_caption_recipe(caption: str) -> dict[str, Any]:
    """Best-effort recipe disclosure scan over the artist's caption."""
    out: dict[str, Any] = {
        "disclosed": False, "prompt": None, "negative_prompt": None,
        "parameters": None, "lora_hints": [], "model_hints": [],
    }
    if not caption:
        return out
    low = caption.lower()
    inline = cr.inline_lora_weights(caption)
    if inline:
        out["lora_hints"] = [{"name": name, "weight": weight} for name, weight in inline.items()]
    out["model_hints"] = [hint for hint in _BASE_MODEL_HINTS if hint in low]
    if "negative prompt:" in low or "steps:" in low:
        a1111 = ir.parse_a1111_parameters(caption)
        out["prompt"] = a1111.get("prompt")
        out["negative_prompt"] = a1111.get("negative_prompt")
        out["parameters"] = a1111.get("parameters")
    out["disclosed"] = bool(inline or out["parameters"])
    return out


def artist_style_loras(name: str | None, limit: int = 8) -> list[dict[str, Any]]:
    """Civitai LoRA search by Pixiv artist name. Hits are UNCONFIRMED candidates
    (fuzzy name match) and need visual confirmation before use."""
    if not name:
        return []
    try:
        code, data = civitai.search_models(query=name, types="LORA", limit=limit)
    except Exception:
        return []
    if code != 200 or not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for model in data.get("items") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        out.append({
            "name": model.get("name"),
            "model_id": model_id,
            "url": f"https://civitai.com/models/{model_id}" if model_id else None,
            "nsfw": model.get("nsfw"),
            "tags": (model.get("tags") or [])[:6],
            "match": "unconfirmed — Civitai search by artist name",
        })
    return out


def _pixiv_block(metadata: dict[str, Any], caption: str) -> dict[str, Any]:
    raw_user = metadata.get("user")
    user = raw_user if isinstance(raw_user, dict) else {}
    return {
        "artwork_id": metadata.get("id"),
        "title": metadata.get("title"),
        "artist": user.get("name"),
        "artist_account": user.get("account"),
        "artist_id": user.get("id"),
        "illust_ai_type": metadata.get("illust_ai_type"),
        "tags": metadata.get("tags") or [],
        "caption_excerpt": caption[:500],
    }


def build(image_path: str, metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    facts = ir.file_facts(image_path)
    caption = clean_caption(metadata.get("caption"))
    ai_type = metadata.get("illust_ai_type")
    pixiv = _pixiv_block(metadata, caption)
    scan = scan_caption_recipe(caption)

    if scan["disclosed"]:
        params = scan["parameters"] or {}
        extracted = {
            "prompt": scan["prompt"], "negative_prompt": scan["negative_prompt"],
            "workflow": None, "seed": params.get("Seed"), "sampler": params.get("Sampler"),
            "model": params.get("Model"), "loras": scan["lora_hints"], "parameters": params or None,
        }
        record = {
            "schema": ir.SCHEMA, "source_path": facts["source_path"], "sha256": facts["sha256"],
            "source_type": "ai_generated_declared" if ai_type == 2 else "ai_generated_site_metadata",
            "metadata_origin": ["pixiv_site"], "extracted": extracted,
            "inferred": {"model_family_guess": None, "prompt_slot_guess": [],
                         "lora_hints": scan["model_hints"], "control_hints": [], "confidence": "low"},
            "cannot_recover": ir.compute_cannot_recover(extracted),
            "content_executor": "grok-in-build", "codex_content_review": False,
            "pixiv": pixiv, "caption_scan": scan,
        }
        return record, "forensics"

    source_type = ("ai_generated_declared" if ai_type == 2
                   else "artist_drawn_declared" if ai_type == 1 else "unknown")
    spec = rs.build_spec(image_path, source_type, reference_strategy="ipadapter",
                         candidate_family="unknown")
    spec["pixiv"] = pixiv
    spec["content_tags"] = pixiv["tags"]
    spec["caption_scan"] = scan
    spec["style_lora_candidates"] = artist_style_loras(pixiv["artist"])
    return spec, "repro_spec"


def write_outputs(record: dict[str, Any], kind: str) -> str:
    out_dir = os.path.join(ir.OUTPUT_BASE, record["sha256"][:12])
    os.makedirs(out_dir, exist_ok=True)
    ir._dump(os.path.join(out_dir, "pixiv_meta.json"), record.get("pixiv", {}))
    ir._dump(os.path.join(out_dir, f"{'forensics' if kind == 'forensics' else 'repro_spec'}.json"), record)
    if kind == "repro_spec":
        ir._dump(os.path.join(out_dir, "palette.json"), record["visual_spec"]["color_palette"])
        if record.get("style_lora_candidates"):
            ir._dump(os.path.join(out_dir, "style_lora_candidates.json"), record["style_lora_candidates"])
    return out_dir


def _print_summary(record: dict[str, Any], kind: str, out_dir: str | None) -> None:
    pixiv = record.get("pixiv", {})
    print(f"Pixiv {pixiv.get('artwork_id')} \"{pixiv.get('title')}\" by {pixiv.get('artist')} "
          f"(ai_type={pixiv.get('illust_ai_type')})")
    if kind == "forensics":
        print(f"-> recipe DISCLOSED in caption ({record['source_type']}); "
              f"prompt: {'yes' if record['extracted'].get('prompt') else 'no'}")
    else:
        scan = record.get("caption_scan", {})
        cands = record.get("style_lora_candidates", [])
        print(f"-> no recipe disclosed; reproduction_spec ({record['source_type']})")
        print(f"   model hints in caption: {scan.get('model_hints') or 'none'}")
        print(f"   content tags: {(record.get('content_tags') or [])[:8]}")
        print(f"   Civitai style-LoRA candidates (unconfirmed): {len(cands)}")
        for c in cands[:5]:
            print(f"     - {c['name']}  {c['url']}")
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
        description="Ingest a Pixiv work → site metadata + opportunistic recipe + reproduction spec.")
    parser.add_argument("work", help="Pixiv artwork URL or numeric ID")
    parser.add_argument("--dest", default=DEFAULT_SRC_DIR, help="download dir for the source image")
    parser.add_argument("--json", action="store_true", help="print the record to stdout")
    parser.add_argument("--no-write", action="store_true", help="do not write output files")
    args = parser.parse_args()

    image_path, metadata = fetch_pixiv(args.work, args.dest)
    if not image_path:
        parser.error(f"could not fetch Pixiv work: {args.work} "
                     f"(check the ID/URL and that {COOKIES_BROWSER} has a Pixiv login)")

    record, kind = build(image_path, metadata)
    out_dir = None if args.no_write else write_outputs(record, kind)

    if args.json:
        json.dump(record, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        _print_summary(record, kind, out_dir)


if __name__ == "__main__":
    main()
