#!/usr/bin/env python3
"""Extract a LOCAL image's generation recipe and render reproduction packs.

P0 — read facts (sha256 / dimensions / format) plus any embedded generation
metadata from a local image file:
  - PNG text chunks (tEXt / zTXt / iTXt)
  - Automatic1111 parameter blocks (prompt / negative / trailing param line)
  - ComfyUI ``prompt`` (API graph) and ``workflow`` (UI graph) JSON
  - NovelAI ``Comment`` JSON
  - NovelAI stealth alpha-LSB metadata (via the imgutils venv bridge)
  - EXIF ``UserComment`` (JPEG / WEBP A1111 exports)
Embedded checkpoint / LoRA hashes are resolved against Civitai (reusing
``civitai_recipe.py``) and optionally matched against the local LoRA catalog.
Emits an ``ai_image_forensics_v1`` record.

P1 — render ``repro_a1111.txt`` (paste-ready params) and
``repro_comfy_workflow.json`` (the embedded graph verbatim, else a labelled
non-runnable scaffold) from that record.

Pixiv note: Pixiv strips all embedded metadata on upload, so this tool recovers
nothing from Pixiv-sourced images. Its real input domain is local images from
Civitai downloads, local generation, NovelAI saves, etc. For artist / non-AI
images use ``image_repro_spec.py`` (reproduction_spec_v1) instead.

SFW only; Civitai access stays on civitai.com through civitai_recipe.py's
transport. The recipe content is independent of where output files are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zlib
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import civitai_recipe as cr  # reuse resource resolution + local-catalog match

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCHEMA = "ai_image_forensics_v1"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
OUTPUT_BASE = os.path.join(_REPO, "outputs", "image_repro")
BRIDGE = os.path.join(_SCRIPT_DIR, "imgutils_bridge.py")
# imgutils venv (comfyui新), consumed read-only. Override via IMGUTILS_VENV_PY.
DEFAULT_VENV_PY = "/opt/comfyui-boxa/tools/visual_features/.venv/bin/python"

_EXIF_IFD = 0x8769          # pointer to the Exif sub-IFD
_EXIF_USERCOMMENT = 0x9286  # UserComment tag inside the Exif sub-IFD
_KEY_PATTERN = re.compile(r"([A-Za-z0-9 _./()-]+?):\s*")


# --------------------------------------------------------------------------- #
# Fact layer
# --------------------------------------------------------------------------- #
def file_facts(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        data = fh.read()
    facts: dict[str, Any] = {
        "source_path": os.path.abspath(path),
        "file_name": os.path.basename(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "metadata_origin": ["file_system"],
    }
    try:
        with Image.open(path) as im:
            facts["width"], facts["height"] = im.width, im.height
            facts["format"], facts["mode"] = im.format, im.mode
    except Exception as exc:
        facts["image_error"] = type(exc).__name__
    return facts


# --------------------------------------------------------------------------- #
# PNG text chunks (ported from the spell-parse parser, kept dependency-free)
# --------------------------------------------------------------------------- #
def read_png_chunks(path: str) -> list[tuple[str, bytes]]:
    chunks: list[tuple[str, bytes]] = []
    with open(path, "rb") as fh:
        if fh.read(8) != PNG_SIGNATURE:
            raise ValueError("unsupported_format")
        while True:
            raw_len = fh.read(4)
            if not raw_len:
                break
            if len(raw_len) != 4:
                raise ValueError("corrupt_png")
            length = struct.unpack(">I", raw_len)[0]
            chunk_type = fh.read(4)
            data = fh.read(length)
            crc = fh.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(crc) != 4:
                raise ValueError("corrupt_png")
            name = chunk_type.decode("latin-1")
            chunks.append((name, data))
            if name == "IEND":
                break
    return chunks


def decode_text_chunks(chunks: list[tuple[str, bytes]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for name, data in chunks:
        try:
            if name == "tEXt":
                key, value = data.split(b"\x00", 1)
                metadata[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
            elif name == "zTXt":
                key, rest = data.split(b"\x00", 1)
                if rest:
                    metadata[key.decode("utf-8", "replace")] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
            elif name == "iTXt":
                parts = data.split(b"\x00", 5)
                if len(parts) == 6:
                    key, comp_flag, text = parts[0], parts[1], parts[5]
                    if comp_flag == b"\x01":
                        text = zlib.decompress(text)
                    metadata[key.decode("utf-8", "replace")] = text.decode("utf-8", "replace")
        except Exception:
            continue
    return metadata


# --------------------------------------------------------------------------- #
# A1111 parameter block
# --------------------------------------------------------------------------- #
def parse_param_line(line: str) -> dict[str, str]:
    """Split an A1111 ``k: v, k: v`` trailing line into a dict, keeping JSON
    braces and quoted strings intact so ``Lora hashes: "a: h1, b: h2"`` and
    ``Hashes: {...}`` survive (the naive comma-split upstream truncates them)."""
    fields: dict[str, str] = {}
    pos = 0
    while pos < len(line):
        match = _KEY_PATTERN.match(line, pos)
        if not match:
            nxt = line.find(",", pos)
            if nxt == -1:
                break
            pos = nxt + 1
            continue
        key = match.group(1).strip()
        start = match.end()
        depth, in_quote, j = 0, False, start
        while j < len(line):
            ch = line[j]
            if ch == '"':
                in_quote = not in_quote
            elif not in_quote and ch in "{[":
                depth += 1
            elif not in_quote and ch in "}]":
                depth -= 1
            elif not in_quote and ch == "," and depth == 0:
                break
            j += 1
        fields[key] = line[start:j].strip()
        pos = j + 1
    return fields


def parse_a1111_parameters(text: str) -> dict[str, Any]:
    block = (text or "").strip()
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return {}
    last_line, body = lines[-1], lines[:-1]
    prompt_lines: list[str] = []
    negative_lines: list[str] = []
    in_negative = False
    for line in body:
        if line.startswith("Negative prompt:"):
            in_negative = True
            negative_lines.append(line.split(":", 1)[1].strip())
        elif in_negative:
            negative_lines.append(line)
        else:
            prompt_lines.append(line)

    result: dict[str, Any] = {}
    # The last line is the param line only when it actually looks like one.
    fields = parse_param_line(last_line) if ":" in last_line else {}
    if not fields:  # no params parsed → the "last line" is really prompt text
        prompt_lines.append(last_line)
    if prompt_lines:
        result["prompt"] = "\n".join(prompt_lines).strip()
    if negative_lines:
        result["negative_prompt"] = "\n".join(negative_lines).strip()
    if fields:
        result["parameters"] = fields
    return result


def parse_novelai_comment(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {"novelai": payload}
    prompt, negative = payload.get("prompt"), payload.get("uc")
    if isinstance(prompt, str) and prompt.strip():
        out["prompt"] = prompt.strip()
    if isinstance(negative, str) and negative.strip():
        out["negative_prompt"] = negative.strip()
    return out


# --------------------------------------------------------------------------- #
# ComfyUI graph
# --------------------------------------------------------------------------- #
def parse_comfy(metadata: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("prompt", "workflow"):
        raw = metadata.get(key)
        if raw:
            try:
                out[key] = json.loads(raw)
            except Exception:
                continue
    return out


def comfy_resources(graph: Any) -> tuple[str | None, list[dict[str, Any]]]:
    """Best-effort checkpoint + LoRA extraction from a ComfyUI API graph. Graphs
    carry file names, not hashes, so these resolve by name only (unresolved)."""
    checkpoint: str | None = None
    loras: list[dict[str, Any]] = []
    nodes = graph.values() if isinstance(graph, dict) else []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ctype = str(node.get("class_type", ""))
        inputs = node.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        if "CheckpointLoader" in ctype and inputs.get("ckpt_name"):
            checkpoint = checkpoint or str(inputs["ckpt_name"])
        if "Lora" in ctype and inputs.get("lora_name"):
            weight = inputs.get("strength_model", inputs.get("strength", 1.0))
            loras.append({"name": str(inputs["lora_name"]),
                          "weight": cr._coerce_weight(weight, 1.0)})
    return checkpoint, loras


# String fields that hold a literal prompt (``populated_text`` = a wildcard
# processor's resolved output); link keys to follow upstream toward that literal.
_STRING_FIELDS = ("populated_text", "text", "string", "value", "wildcard_text")
_FOLLOW_KEYS = ("text", "string", "input1", "positive", "conditioning", "value", "samples")


def _resolve_text_node(graph: dict, ref: Any, depth: int = 0,
                       seen: frozenset[str] = frozenset()) -> str | None:
    """Follow a ComfyUI link to a literal prompt string, traversing common
    wrappers (CLIPTextEncode, ImpactSwitch, RegexReplace, StringConcatenate,
    wildcard processors). Bounded; returns None when no literal is reachable —
    e.g. a fully runtime-dynamic wildcard chain. The workflow JSON is preserved
    regardless, so a None here never means lost provenance."""
    if isinstance(ref, str):
        return ref.strip() or None
    if not (isinstance(ref, list) and ref):
        return None
    nid = str(ref[0])
    if depth > 10 or nid in seen:
        return None
    node = graph.get(nid)
    if not isinstance(node, dict):
        return None
    raw_inputs = node.get("inputs")
    inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
    for field in _STRING_FIELDS:  # a literal here is the answer
        value = inputs.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in _FOLLOW_KEYS:  # else follow the single most likely upstream link
        if isinstance(inputs.get(key), list):
            resolved = _resolve_text_node(graph, inputs[key], depth + 1, seen | {nid})
            if resolved:
                return resolved
    # concat-style nodes (StringConcatenate, etc.): join every remaining link in order
    parts: list[str] = []
    for key, value in inputs.items():
        if isinstance(value, list) and value:
            resolved = _resolve_text_node(graph, value, depth + 1, seen | {nid})
            if resolved:
                parts.append(resolved)
    joined = ", ".join(dict.fromkeys(parts)).strip()
    return joined or None


def comfy_prompt_texts(graph: Any) -> tuple[str | None, str | None]:
    """Resolve positive / negative prompt text by following each sampler node's
    positive / negative conditioning link back to its literal text source."""
    if not isinstance(graph, dict):
        return None, None
    for node in graph.values():
        if not isinstance(node, dict) or "Sampler" not in str(node.get("class_type", "")):
            continue
        raw_inputs = node.get("inputs")
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        positive = _resolve_text_node(graph, inputs.get("positive"))
        negative = _resolve_text_node(graph, inputs.get("negative"))
        if positive or negative:
            return positive, negative
    return None, None


_SAMPLER_FIELDS = ("sampler_name", "scheduler", "cfg", "steps", "seed", "denoise")


def _resolve_scalar(graph: dict, value: Any, depth: int = 0) -> Any:
    """Resolve a scalar sampler input, following one link (e.g. a Seed node)."""
    if isinstance(value, (str, int, float)):
        return value
    if depth > 6 or not (isinstance(value, list) and value):
        return None
    node = graph.get(str(value[0]))
    if not isinstance(node, dict):
        return None
    raw_inputs = node.get("inputs")
    inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
    for field in ("seed", "value", "int", "number", "Value"):
        if isinstance(inputs.get(field), (int, float, str)):
            return inputs[field]
        if isinstance(inputs.get(field), list):
            return _resolve_scalar(graph, inputs[field], depth + 1)
    return None


def comfy_sampler_params(graph: Any) -> dict[str, Any]:
    """Pull sampler_name / scheduler / cfg / steps / seed / denoise from the first
    sampler node (resolving link-valued inputs such as a Seed node to literals)."""
    if not isinstance(graph, dict):
        return {}
    for node in graph.values():
        if not isinstance(node, dict) or "Sampler" not in str(node.get("class_type", "")):
            continue
        raw_inputs = node.get("inputs")
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        out: dict[str, Any] = {}
        for field in _SAMPLER_FIELDS:
            if field in inputs:
                resolved = _resolve_scalar(graph, inputs[field])
                if resolved is not None:
                    out[field] = resolved
        if out:
            return out
    return {}


# --------------------------------------------------------------------------- #
# EXIF UserComment (JPEG / WEBP)
# --------------------------------------------------------------------------- #
def _decode_user_comment(raw: bytes) -> str | None:
    for prefix, enc in ((b"UNICODE\x00", "utf-16-be"), (b"ASCII\x00\x00\x00", "ascii")):
        if raw.startswith(prefix):
            try:
                return raw[len(prefix):].decode(enc, "replace").strip("\x00") or None
            except Exception:
                return None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc).strip("\x00")
            if text:
                return text
        except Exception:
            continue
    return None


def read_exif_parameters(path: str) -> str | None:
    try:
        with Image.open(path) as im:
            user_comment = im.getexif().get_ifd(_EXIF_IFD).get(_EXIF_USERCOMMENT)
    except Exception:
        return None
    if not user_comment:
        return None
    if isinstance(user_comment, bytes):
        return _decode_user_comment(user_comment)
    text = str(user_comment).strip("\x00")
    return text or None


# --------------------------------------------------------------------------- #
# imgutils venv bridge
# --------------------------------------------------------------------------- #
def run_bridge(op: str, path: str, path2: str | None = None) -> dict[str, Any] | None:
    venv = os.environ.get("IMGUTILS_VENV_PY", DEFAULT_VENV_PY)
    if not os.path.exists(venv):
        return None
    argv = [venv, BRIDGE, op, path]
    if path2 is not None:
        argv.append(path2)
    try:
        proc = subprocess.run(argv,
                              capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    if not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Resource resolution (reuses civitai_recipe)
# --------------------------------------------------------------------------- #
def _hashes_from_params(params: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    lora_hashes = params.get("Lora hashes")
    if isinstance(lora_hashes, str):
        for pair in lora_hashes.strip().strip('"').split(","):
            if ":" in pair:
                name, value = pair.split(":", 1)
                if name.strip() and value.strip():
                    out[f"lora:{name.strip()}"] = value.strip()
    hashes_field = params.get("Hashes")
    if isinstance(hashes_field, str):
        try:
            parsed = json.loads(hashes_field)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(value, str) and value:
                    out[str(key)] = value
    return out


def synth_meta(prompt: str | None, params: dict[str, Any]) -> dict[str, Any]:
    """Build a Civitai-shaped meta dict from parsed A1111 params so
    civitai_recipe.collect_resources can merge checkpoint + LoRA stubs."""
    meta: dict[str, Any] = {"prompt": prompt or ""}
    if params.get("Model"):
        meta["Model"] = params["Model"]
    model_hash = params.get("Model hash")
    hashes = _hashes_from_params(params)
    if hashes.get("model") and not model_hash:
        model_hash = hashes["model"]
    if model_hash:
        meta["Model hash"] = model_hash
    if hashes:
        meta["hashes"] = hashes
    return meta


def resolve_resources(meta: dict[str, Any], local: bool, offline: bool) -> tuple[dict | None, list[dict]]:
    checkpoint, loras = cr.collect_resources(meta)
    catalog = cr.load_local_catalog() if local else None

    def finish(stub: dict) -> dict:
        if not offline and (stub.get("hash") or stub.get("version_id")):
            stub = cr.resolve_stub(stub)
        else:
            stub = dict(stub, resolved=False, civitai=None,
                        reason="offline" if offline else "no hash or version id")
        if catalog is not None:
            stub["local"] = cr.local_match(stub, catalog)
        return stub

    resolved_checkpoint = finish(checkpoint) if checkpoint is not None else None
    return resolved_checkpoint, [finish(lora) for lora in loras]


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #
def _lora_record(lora: dict[str, Any]) -> dict[str, Any]:
    civ = lora.get("civitai") or {}
    out: dict[str, Any] = {
        "name": (civ.get("model_name") or lora.get("name")),
        "weight": lora.get("weight"),
        "hash": lora.get("hash"),
        "version_id": civ.get("version_id") or lora.get("version_id"),
        "resolved": bool(lora.get("resolved")),
    }
    if "local" in lora:
        out["local_present"] = lora["local"].get("present")
    return out


def classify(metadata_origin: list[str], extracted: dict[str, Any]) -> str:
    """A bare ``png_metadata`` / ``exif`` origin (e.g. a lone ``Software`` tag on a
    hand-drawn PNG) is NOT a generation signal — only embedded gen artifacts are."""
    gen_signals = {"comfy_workflow", "novelai_comment", "stealth_lsb"}
    if gen_signals.intersection(metadata_origin):
        return "ai_generated_extracted"
    if extracted.get("prompt") or extracted.get("parameters") or extracted.get("workflow"):
        return "ai_generated_extracted"
    return "unknown"


def compute_cannot_recover(extracted: dict[str, Any]) -> list[str]:
    missing = []
    if not extracted.get("prompt"):
        missing.append("prompt")
    if extracted.get("seed") in (None, ""):
        missing.append("seed")
    if not extracted.get("sampler"):
        missing.append("sampler")
    if not extracted.get("model"):
        missing.append("model")
    if not extracted.get("loras"):
        missing.append("loras")
    return missing


def build_forensics(parsed: dict[str, Any], checkpoint: dict | None,
                    loras: list[dict]) -> dict[str, Any]:
    facts = parsed["facts"]
    params = parsed.get("params") or {}
    model_name = None
    if checkpoint:
        civ = checkpoint.get("civitai") or {}
        model_name = civ.get("model_name") or checkpoint.get("name")
    model_name = model_name or params.get("Model") or parsed.get("comfy_checkpoint")
    extracted = {
        "prompt": parsed.get("prompt"),
        "negative_prompt": parsed.get("negative_prompt"),
        "workflow": parsed.get("workflow"),
        "seed": params.get("Seed"),
        "sampler": params.get("Sampler"),
        "model": model_name,
        "loras": [_lora_record(lora) for lora in loras],
        "parameters": params or None,
    }
    source_type = classify(facts["metadata_origin"], extracted)
    inferred: dict[str, Any] = {
        "model_family_guess": None,
        "prompt_slot_guess": [],
        "lora_hints": [],
        "control_hints": [],
        "confidence": "low",
    }
    if source_type == "unknown":
        # Pixel-side inference matters only when no metadata was recovered; an
        # extracted recipe is exact and needs no guessing (and skipping the WD14
        # call keeps forensics cheap for metadata-bearing images).
        import recipe_guess as rg  # lazy: keeps image_recipe importable standalone
        wd14 = run_bridge("wd14", facts["source_path"])
        if wd14 and wd14.get("ok"):
            chars = [tag for tag, _ in rg.character_candidates(wd14.get("character"))]
            inferred = rg.build_inferred(wd14, lora_hints=rg.lookup_character_loras(chars))
    return {
        "schema": SCHEMA,
        "source_path": facts["source_path"],
        "sha256": facts["sha256"],
        "source_type": source_type,
        "metadata_origin": facts["metadata_origin"],
        "extracted": extracted,
        "inferred": inferred,
        "cannot_recover": compute_cannot_recover(extracted),
        "content_executor": "grok-in-build",
        "codex_content_review": False,
    }


# --------------------------------------------------------------------------- #
# Parse orchestration
# --------------------------------------------------------------------------- #
def parse_image(path: str) -> dict[str, Any]:
    facts = file_facts(path)
    origin = facts["metadata_origin"]
    fmt = (facts.get("format") or "").upper()
    raw_metadata: dict[str, Any] = {}
    warnings: list[str] = []

    chunk_meta: dict[str, str] = {}
    if fmt == "PNG":
        try:
            chunk_meta = decode_text_chunks(read_png_chunks(path))
        except Exception as exc:
            warnings.append(f"png chunk read failed: {type(exc).__name__}")
    if chunk_meta:
        raw_metadata.update(chunk_meta)
        origin.append("png_metadata")

    prompt = negative = None
    params: dict[str, Any] = {}

    if "parameters" in chunk_meta:
        a1111 = parse_a1111_parameters(chunk_meta["parameters"])
        prompt = a1111.get("prompt")
        negative = a1111.get("negative_prompt")
        params = a1111.get("parameters") or {}

    comfy = parse_comfy(chunk_meta)
    workflow = comfy.get("workflow")
    comfy_prompt = comfy.get("prompt")
    if workflow is not None or comfy_prompt is not None:
        origin.append("comfy_workflow")
    comfy_checkpoint, comfy_loras = comfy_resources(comfy_prompt) if comfy_prompt is not None else (None, [])
    if comfy_prompt is not None and not prompt:
        comfy_pos, comfy_neg = comfy_prompt_texts(comfy_prompt)
        prompt = prompt or comfy_pos
        negative = negative or comfy_neg
    if comfy_prompt is not None and not params:
        sampler = comfy_sampler_params(comfy_prompt)
        params = {key: value for key, value in {
            "Sampler": sampler.get("sampler_name"),
            "Seed": sampler.get("seed"),
            "Steps": sampler.get("steps"),
            "CFG scale": sampler.get("cfg"),
            "Schedule": sampler.get("scheduler"),
            "Denoising strength": sampler.get("denoise"),
        }.items() if value is not None}

    if "Comment" in chunk_meta:
        nai = parse_novelai_comment(chunk_meta["Comment"])
        if nai:
            origin.append("novelai_comment")
            prompt = prompt or nai.get("prompt")
            negative = negative or nai.get("negative_prompt")

    # EXIF for non-PNG, or as a fallback when PNG carried no prompt.
    if fmt in ("JPEG", "WEBP", "TIFF") or not prompt:
        exif_text = read_exif_parameters(path)
        if exif_text:
            origin.append("exif")
            raw_metadata["exif:UserComment"] = exif_text
            if not prompt:
                a1111 = parse_a1111_parameters(exif_text)
                prompt = a1111.get("prompt")
                negative = negative or a1111.get("negative_prompt")
                params = params or a1111.get("parameters") or {}

    # NovelAI stealth alpha-LSB (venv bridge).
    lsb = run_bridge("lsb", path)
    if lsb is None:
        warnings.append("imgutils venv unavailable — NovelAI stealth-LSB not checked")
    elif lsb.get("ok") and lsb.get("metadata"):
        origin.append("stealth_lsb")
        stealth = lsb["metadata"]
        raw_metadata["stealth_lsb"] = stealth
        if not prompt and isinstance(stealth, dict):
            comment = stealth.get("Comment")
            nai = parse_novelai_comment(comment) if isinstance(comment, str) else {}
            prompt = prompt or nai.get("prompt")
            negative = negative or nai.get("negative_prompt")

    return {
        "facts": facts,
        "prompt": prompt,
        "negative_prompt": negative,
        "params": params,
        "workflow": workflow,
        "comfy_prompt": comfy_prompt,
        "comfy_checkpoint": comfy_checkpoint,
        "comfy_loras": comfy_loras,
        "raw_metadata": raw_metadata,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# P1 renderers
# --------------------------------------------------------------------------- #
def render_a1111_txt(record: dict[str, Any]) -> str:
    extracted = record["extracted"]
    lines: list[str] = []
    if extracted.get("prompt"):
        lines.append(extracted["prompt"])
    if extracted.get("negative_prompt"):
        lines.append(f"Negative prompt: {extracted['negative_prompt']}")
    params = extracted.get("parameters") or {}
    if params:
        lines.append(", ".join(f"{k}: {v}" for k, v in params.items()))
    return ("\n".join(lines) + "\n") if lines else ""


def render_comfy_workflow(record: dict[str, Any]) -> dict[str, Any]:
    extracted = record["extracted"]
    workflow = extracted.get("workflow")
    if isinstance(workflow, (dict, list)):
        return {"embedded_workflow": workflow}
    return {
        "_note": "No embedded ComfyUI graph. This is a non-runnable scaffold of the "
                 "extracted recipe — wire it into a loader/sampler graph manually.",
        "checkpoint": extracted.get("model"),
        "loras": extracted.get("loras"),
        "positive": extracted.get("prompt"),
        "negative": extracted.get("negative_prompt"),
        "params": extracted.get("parameters") or {},
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_outputs(record: dict[str, Any], parsed: dict[str, Any], out_dir: str | None) -> str:
    out_dir = out_dir or os.path.join(OUTPUT_BASE, record["sha256"][:12])
    os.makedirs(out_dir, exist_ok=True)
    _dump(os.path.join(out_dir, "facts.json"), parsed["facts"])
    _dump(os.path.join(out_dir, "forensics.json"), record)
    _dump(os.path.join(out_dir, "raw_metadata.json"), parsed["raw_metadata"])
    with open(os.path.join(out_dir, "repro_a1111.txt"), "w", encoding="utf-8") as fh:
        fh.write(render_a1111_txt(record))
    _dump(os.path.join(out_dir, "repro_comfy_workflow.json"), render_comfy_workflow(record))
    if parsed.get("warnings"):
        _dump(os.path.join(out_dir, "warnings.json"), parsed["warnings"])
    return out_dir


# --------------------------------------------------------------------------- #
# Public entry point + CLI
# --------------------------------------------------------------------------- #
def get_recipe(path: str, local: bool = False, offline: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_image(path)
    meta = synth_meta(parsed.get("prompt"), parsed.get("params") or {})
    checkpoint, loras = resolve_resources(meta, local, offline)
    # Fall back to name-only LoRAs when hash resolution found none: inline
    # <lora:name:weight> tags in the prompt (richest — carries weights), else the
    # ComfyUI graph's LoraLoader names.
    if not loras:
        inline = cr.inline_lora_weights(parsed.get("prompt") or "")
        if inline:
            loras = [{"name": name, "weight": weight, "hash": None, "version_id": None,
                      "resolved": False, "civitai": None, "reason": "inline <lora:> tag — name only"}
                     for name, weight in inline.items()]
        elif parsed.get("comfy_loras"):
            loras = [dict(lora, resolved=False, civitai=None, reason="comfy graph: name only")
                     for lora in parsed["comfy_loras"]]
    return build_forensics(parsed, checkpoint, loras), parsed


def _print_summary(record: dict[str, Any], out_dir: str | None) -> None:
    ex = record["extracted"]
    print(f"Source type: {record['source_type']}")
    print(f"Prompt: {'yes' if ex.get('prompt') else 'no'} | "
          f"Model: {ex.get('model') or '?'} | LoRAs: {len(ex.get('loras') or [])} | "
          f"Seed: {ex.get('seed') or '?'} | Sampler: {ex.get('sampler') or '?'}")
    if record["cannot_recover"]:
        print(f"Cannot recover: {', '.join(record['cannot_recover'])}")
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
        description="Extract a local image's generation recipe (ai_image_forensics_v1) "
                    "and render reproduction packs.")
    parser.add_argument("image", help="path to a local image file")
    parser.add_argument("--json", action="store_true", help="print the forensics record to stdout")
    parser.add_argument("--local", action="store_true",
                        help="flag which resolved resources are present in the local LoRA catalog")
    parser.add_argument("--offline", action="store_true",
                        help="skip Civitai hash resolution (no network)")
    parser.add_argument("--out-dir", help="output directory (default outputs/image_repro/<sha12>/)")
    parser.add_argument("--no-write", action="store_true", help="do not write output files")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"not a file: {args.image}")

    record, parsed = get_recipe(args.image, local=args.local, offline=args.offline)
    out_dir = None if args.no_write else write_outputs(record, parsed, args.out_dir)

    if args.json:
        json.dump(record, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        _print_summary(record, out_dir)


if __name__ == "__main__":
    main()
