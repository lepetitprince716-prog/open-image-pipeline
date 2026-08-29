#!/usr/bin/env python3
"""NovelAI image generation client (txt2img).

Minimal stdlib-only client for the NovelAI image API. Written ahead of the first
subscription; the request format follows the ComfyUI_NAIDGenerator node and the
LlmKira/novelai-python SDK.

Smoke-tested against the live API on 2026-08-25: V4.5 single, V4.5 two
characters with centers, and V5 full all returned images. A client User-Agent
is required; the default Python-urllib signature gets a Cloudflare 403 (1010).

Token: set NAI_ACCESS_TOKEN in the environment (User Settings -> Account ->
Get Persistent API Token). The token is never printed or logged.

Usage:
  python3 scripts/nai_gen.py --prompt "1girl, cafe, afternoon light" --dry-run
  python3 scripts/nai_gen.py --job job.json --out outputs/nai
  python3 scripts/nai_gen.py --prompt "1girl, rooftop, sunset" --model v5-full \
      --artist-mix <card_id>   # prepend an artist string card (nai_artist_strings.py)
  python3 scripts/nai_gen.py --prompt "2girls, rooftop" \
      --char "girl, silver hair, red eyes" --char-pos 0.3,0.5 \
      --char "girl, black hair, amber eyes" --char-pos 0.7,0.5
  python3 scripts/nai_gen.py --prompt "1girl, standing" \
      --vibe style_01.jpg --vibe-strength 0.6   # vibe transfer (style reference)

Vibe transfer: --vibe takes a reference image, encodes it through
/ai/encode-vibe and passes the block to the generator. Encoding costs 2 Anlas
per (image, information_extracted, model) combination, so blocks are cached
under --vibe-cache and reused. Only the V4/V4.5 families encode; V5 returns
500 (probed 2026-08-28).

Job file format (matches docs/novelai_line/PROMPT_DELEGATION_PLAYBOOK.md):
  {"model": "...", "base": "...", "characters": [{"prompt": "...",
   "center": {"x": 0.3, "y": 0.5}}], "negative": "..."}
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API_URL = "https://image.novelai.net/ai/generate-image"
ENCODE_VIBE_URL = "https://image.novelai.net/ai/encode-vibe"

# Encoding costs 2 Anlas per (image, information_extracted, model); cache blocks
# on disk so a re-run never pays twice.
DEFAULT_VIBE_CACHE = "outputs/nai/vibes"

# NovelAI's own guidance: four or fewer vibes add no generation cost, and the
# strengths should sum to at most this.
VIBE_STRENGTH_SUM_ADVICE = 1.0
VIBE_FREE_COUNT = 4
DEFAULT_VIBE_STRENGTH = 0.6
DEFAULT_VIBE_INFO = 1.0

# Cloudflare rejects the default Python-urllib signature (403 / 1010); identify
# as a normal API client the way the SDK ecosystem does.
USER_AGENT = "nai-gen/0.1 (loramake; +https://github.com/lepetitprince716-prog/z3comfy-studio)"

MODELS = {
    "v5-full": "nai-diffusion-5-full",
    "v5-curated": "nai-diffusion-5-curated",
    "v4.5-full": "nai-diffusion-4-5-full",
    "v4.5-curated": "nai-diffusion-4-5-curated",
    "v4-full": "nai-diffusion-4-full",
    "v4-curated": "nai-diffusion-4-curated-preview",
    "v3": "nai-diffusion-3",
    "furry-v3": "nai-diffusion-furry-3",
}

SAMPLERS = (
    "k_euler",
    "k_euler_ancestral",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m_sde",
    "k_dpmpp_2m",
    "k_dpmpp_sde",
    "ddim",
)

NOISE_SCHEDULES = ("native", "karras", "exponential", "polyexponential")

# Opus unlimited tier: normal resolutions at <= 28 steps on V4.5 and older.
DEFAULT_SIZES = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

DEFAULT_NEGATIVE = (
    "lowres, worst quality, bad quality, jpeg artifacts, artistic error, watermark"
)


def resolve_model(name):
    if name in MODELS:
        return MODELS[name]
    # Any raw nai-diffusion-* id passes through so new models work before the
    # alias table catches up.
    if name.startswith("nai-diffusion-"):
        return name
    raise SystemExit(
        f"unknown model {name!r}; use one of {', '.join(MODELS)} or a raw nai-diffusion-* id"
    )


def is_v4_family(model_id):
    return model_id.startswith(("nai-diffusion-4", "nai-diffusion-5"))


def validate_center(center, origin):
    try:
        x, y = float(center["x"]), float(center["y"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"bad center in {origin}; expected {{'x': 0..1, 'y': 0..1}}")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise SystemExit(f"center out of range in {origin}; coordinates are 0..1")
    return {"x": x, "y": y}


def parse_center(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"bad --char-pos {text!r}; expected x,y like 0.3,0.5")
    return validate_center({"x": x, "y": y}, f"--char-pos {text!r}")


def build_characters(args):
    characters = []
    positions = list(args.char_pos or [])
    for i, prompt in enumerate(args.char or []):
        entry = {"prompt": prompt}
        if i < len(positions):
            entry["center"] = parse_center(positions[i])
        characters.append(entry)
    return characters


def load_job(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "base" not in data:
        # Playbook authors deliver a batch wrapper like {"B1": {...}}. Unwrap a
        # single entry automatically; multiple entries need an explicit pick.
        entries = {
            k: v for k, v in data.items() if isinstance(v, dict) and "base" in v
        }
        if len(entries) == 1:
            return next(iter(entries.values()))
        if entries:
            raise SystemExit(
                f"job file holds multiple entries ({', '.join(sorted(entries))}); "
                "split it and pass one entry per --job file"
            )
        raise SystemExit("job file missing keys: ['base']")
    return data


def post_json(url, payload, token, timeout=180):
    """POST a JSON body and return the raw response bytes."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        detail = err.read()[:500].decode("utf-8", "replace")
        hints = {
            401: "token invalid or expired; regenerate the persistent token",
            402: "insufficient Anlas or no active subscription for this model/settings",
            429: "rate limited; wait and retry",
        }
        hint = hints.get(err.code, "")
        raise SystemExit(
            f"API error {err.code}: {detail}" + (f" ({hint})" if hint else "")
        )
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise SystemExit(f"network error reaching {url}: {err}")


def vibe_cache_path(cache_dir, image_bytes, information_extracted, model_id):
    # The model is part of the key: a block encoded for V4.5 is not valid for
    # another family, and a collision here would silently mis-transfer style.
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    # resolve_model passes any nai-diffusion-* id through, so flatten the id to
    # one safe filename component: a separator in it must never let the cache
    # file land outside cache_dir.
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return Path(cache_dir) / f"{digest}_ie{information_extracted:.2f}_{model_part}.bin"


def resolve_vibe(
    path, information_extracted, model_id, token, cache_dir, offline=False
):
    """Cache-first vibe block for one reference image.

    Returns the provenance dict the manifest records; "block" is None when an
    offline caller asked for a block that has not been encoded yet.
    """
    if not (0.0 <= information_extracted <= 1.0):
        raise SystemExit(
            f"--vibe-info {information_extracted} out of range; expected 0..1"
        )
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"vibe reference not found: {source}")
    raw = source.read_bytes()
    cached = vibe_cache_path(cache_dir, raw, information_extracted, model_id)
    record = {
        "source": str(source),
        "cache": str(cached),
        "information_extracted": information_extracted,
        "model": model_id,
    }
    if cached.is_file():
        record["block"] = cached.read_bytes()
        record["cached"] = True
        return record
    if offline:
        record["block"] = None
        record["cached"] = False
        return record
    blob = post_json(
        ENCODE_VIBE_URL,
        {
            "image": base64.b64encode(raw).decode("ascii"),
            "model": model_id,
            "information_extracted": information_extracted,
            # focus_seed / info_extract_seed left unset: the API then derives
            # them from the image content, which is what makes caching correct.
        },
        token,
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(blob)
    record["block"] = blob
    record["cached"] = False
    return record


def build_vibes(args, model_id, token, offline=False):
    """Resolve every --vibe into an encoded block plus its strength."""
    paths = list(args.vibe or [])
    if not paths:
        return []
    if model_id.startswith("nai-diffusion-5"):
        # Deliberately a warning, not a hard failure: the 500 is server-side
        # state, and warn-and-proceed means V5 vibe support appearing upstream
        # needs no client change. The encode call itself still fails loudly.
        print(
            f"[vibe] warning: /ai/encode-vibe returned 500 for {model_id} when "
            "probed on 2026-08-28; the V4/V4.5 families are the verified path",
            file=sys.stderr,
        )
    if len(paths) > VIBE_FREE_COUNT:
        print(
            f"[vibe] warning: {len(paths)} references; more than {VIBE_FREE_COUNT} "
            "adds a per-generation Anlas cost",
            file=sys.stderr,
        )
    strengths = spread_vibe_values(args.vibe_strength, len(paths), "--vibe-strength")
    infos = spread_vibe_values(args.vibe_info, len(paths), "--vibe-info")
    vibes = []
    for path, strength, info in zip(paths, strengths, infos):
        if not (0.0 <= strength <= 1.0):
            raise SystemExit(f"--vibe-strength {strength} out of range; expected 0..1")
        record = resolve_vibe(
            path, info, model_id, token, args.vibe_cache, offline=offline
        )
        record["strength"] = strength
        vibes.append(record)
        if record["block"] is None:
            state = "not encoded yet (offline)"
        elif record["cached"]:
            state = f"cached {record['cache']}"
        else:
            state = f"encoded -> {record['cache']} (2 Anlas)"
        # stderr: --dry-run writes the payload JSON to stdout for piping.
        print(
            f"[vibe] {Path(path).name} ie={info} strength={strength}: {state}",
            file=sys.stderr,
        )
    total = sum(v["strength"] for v in vibes)
    if total > VIBE_STRENGTH_SUM_ADVICE:
        print(
            f"[vibe] warning: strengths sum to {total:.2f}; NovelAI advises "
            f"<= {VIBE_STRENGTH_SUM_ADVICE:.1f}",
            file=sys.stderr,
        )
    return vibes


def spread_vibe_values(values, count, flag):
    """Broadcast one value across every vibe, or require one value each."""
    values = list(values or [])
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise SystemExit(
            f"{flag} takes one value or one per --vibe; got {len(values)} for {count}"
        )
    return values


def build_payload(model_id, base, characters, negative, args, vibes=None):
    with_center = sum(1 for c in characters if "center" in c)
    if 0 < with_center < len(characters):
        # Mixing placed and AI's-Choice characters would silently pin the
        # center-less ones to (0.5, 0.5); force an explicit decision instead.
        raise SystemExit(
            "give every character a center or none; mixed placement is ambiguous"
        )
    use_coords = with_center > 0
    params = {
        "params_version": 1,
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "sampler": args.sampler,
        "steps": args.steps,
        "seed": args.seed,
        "n_samples": 1,
        "ucPreset": args.uc_preset,
        "qualityToggle": args.quality_toggle,
        "dynamic_thresholding": False,
        "controlnet_strength": 1.0,
        "legacy": False,
        "add_original_image": False,
        "cfg_rescale": args.cfg_rescale,
        "noise_schedule": args.noise_schedule,
        "negative_prompt": negative,
        # Defaults mirrored from ComfyUI_NAIDGenerator so the request matches a
        # known-working client. SMEA (sm/sm_dyn) is a V3-era option, off here.
        "sm": False,
        "sm_dyn": False,
        "uncond_scale": 1.0,
        "extra_noise_seed": args.seed,
        "legacy_v3_extend": False,
    }
    if is_v4_family(model_id):
        char_captions = [
            {
                "char_caption": c["prompt"],
                "centers": [c["center"]] if "center" in c else [{"x": 0.5, "y": 0.5}],
            }
            for c in characters
        ]
        params["v4_prompt"] = {
            "use_coords": use_coords,
            "use_order": True,
            "caption": {"base_caption": base, "char_captions": char_captions},
        }
        params["v4_negative_prompt"] = {
            "use_coords": False,
            "use_order": False,
            "caption": {
                "base_caption": negative,
                "char_captions": [
                    {"char_caption": "", "centers": [{"x": 0.5, "y": 0.5}]}
                    for _ in characters
                ],
            },
        }
    for key, value in vibe_parameters(vibes or []).items():
        params[key] = value
    return {
        "input": base,
        "model": model_id,
        "action": "generate",
        "parameters": params,
    }


def vibe_parameters(vibes):
    """The three parallel arrays the generator expects for vibe transfer.

    information_extracted must match the value the block was encoded with, so
    it is carried on the record rather than re-read from the CLI.
    """
    if not vibes:
        return {}
    images = []
    for vibe in vibes:
        block = vibe.get("block")
        if block is None:
            # Offline dry-run: show the shape without paying for an encode.
            images.append(f"<unencoded vibe: {vibe['source']}>")
        else:
            images.append(base64.b64encode(block).decode("ascii"))
    return {
        "reference_image_multiple": images,
        "reference_strength_multiple": [v["strength"] for v in vibes],
        "reference_information_extracted_multiple": [
            v["information_extracted"] for v in vibes
        ],
    }


def unique_path(out_dir, stem):
    path = out_dir / f"{stem}.png"
    n = 1
    while path.exists():
        path = out_dir / f"{stem}-{n}.png"
        n += 1
    return path


def save_response(body, out_dir, stamp):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            for i, name in enumerate(zf.namelist()):
                path = unique_path(out_dir, f"nai_{stamp}_{i}")
                path.write_bytes(zf.read(name))
                saved.append(path)
    else:
        path = unique_path(out_dir, f"nai_{stamp}")
        path.write_bytes(body)
        saved.append(path)
    return saved


# Vibe blocks are megabytes of base64; the manifest and the dry-run dump both
# keep the provenance and drop the payload.
VIBE_BULK_KEYS = ("reference_image_multiple", "reference_image")


def append_manifest(out_dir, payload, saved, vibes=None):
    # One JSONL line per generation so prompts/params stay searchable without
    # opening the PNGs (NAI keeps no server-side history for API calls).
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": payload["model"],
        "prompt": payload["input"],
        "parameters": {
            k: v
            for k, v in payload["parameters"].items()
            if k != "negative_prompt" and k not in VIBE_BULK_KEYS
        },
        "negative": payload["parameters"]["negative_prompt"],
        "files": [str(p) for p in saved],
    }
    if vibes:
        record["vibes"] = [
            {
                "source": v["source"],
                "cache": v["cache"],
                "strength": v["strength"],
                "information_extracted": v["information_extracted"],
                "model": v["model"],
            }
            for v in vibes
        ]
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def elide_bulk(payload):
    """Copy of the payload with vibe base64 replaced by a size marker."""
    params = dict(payload["parameters"])
    for key in VIBE_BULK_KEYS:
        if key not in params:
            continue
        value = params[key]
        items = value if isinstance(value, list) else [value]
        # Offline dry-run placeholders are already human-readable, not base64;
        # measuring them would report the placeholder's length as a size.
        marks = [
            i if isinstance(i, str) and i.startswith("<unencoded vibe:")
            else f"<base64 {len(i)} chars>"
            for i in items
        ]
        params[key] = marks if isinstance(value, list) else marks[0]
    return {**payload, "parameters": params}


def generate(payload, token, out_dir, vibes=None):
    body = post_json(API_URL, payload, token)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = save_response(body, out_dir, stamp)
    append_manifest(out_dir, payload, saved, vibes)
    return saved


def main():
    parser = argparse.ArgumentParser(description="NovelAI txt2img client")
    parser.add_argument("--prompt", help="base prompt (scene, style, count tags)")
    parser.add_argument(
        "--job", help="JSON job file (playbook format); overrides --prompt/--char"
    )
    parser.add_argument(
        "--model",
        default="v4.5-full",
        help=f"model alias or raw id (default v4.5-full); aliases: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--char",
        action="append",
        help="character prompt; repeat for multiple characters",
    )
    parser.add_argument(
        "--char-pos",
        action="append",
        help="x,y in 0..1 for the matching --char (order-paired)",
    )
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument(
        "--artist-mix",
        help="artist string card id; the card's style block is prepended to the "
        "base prompt (see scripts/nai_artist_strings.py list)",
    )
    parser.add_argument(
        "--vibe",
        action="append",
        help="vibe transfer reference image; repeat for up to 4 references",
    )
    parser.add_argument(
        "--vibe-strength",
        action="append",
        type=float,
        default=None,
        help="0..1 per --vibe, or one value for all (default 0.6); NovelAI "
        "advises the strengths sum to <= 1.0",
    )
    parser.add_argument(
        "--vibe-info",
        action="append",
        type=float,
        default=None,
        help="information_extracted 0..1 per --vibe, or one value for all "
        "(default 1.0); part of the encode cache key",
    )
    parser.add_argument(
        "--vibe-cache",
        default=DEFAULT_VIBE_CACHE,
        help=f"directory for cached vibe blocks (default {DEFAULT_VIBE_CACHE})",
    )
    parser.add_argument(
        "--quality-toggle",
        action="store_true",
        help="send qualityToggle=true (server-side Add Quality Tags; default off)",
    )
    parser.add_argument(
        "--uc-preset",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
        help="ucPreset id (default 3, matching the previous hardcoded value)",
    )
    parser.add_argument(
        "--size",
        choices=sorted(DEFAULT_SIZES),
        default="portrait",
        help="named size (default portrait 832x1216)",
    )
    parser.add_argument("--width", type=int, help="override width")
    parser.add_argument("--height", type=int, help="override height")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--scale", type=float, default=5.0, help="prompt guidance")
    parser.add_argument("--cfg-rescale", type=float, default=0.0)
    parser.add_argument("--sampler", choices=SAMPLERS, default="k_euler_ancestral")
    parser.add_argument(
        "--noise-schedule", choices=NOISE_SCHEDULES, default="karras"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", default="outputs/nai", help="output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request payload and exit; no network, no token needed",
    )
    args = parser.parse_args()

    if args.job:
        job = load_job(args.job)
        model_id = resolve_model(job.get("model", args.model))
        base = job["base"]
        characters = job.get("characters", [])
        for i, c in enumerate(characters):
            if "prompt" not in c:
                raise SystemExit("every job character needs a 'prompt' key")
            if "center" in c:
                c["center"] = validate_center(c["center"], f"job character {i + 1}")
        negative = job.get("negative", args.negative)
    else:
        if not args.prompt:
            parser.error("--prompt or --job is required")
        model_id = resolve_model(args.model)
        base = args.prompt
        characters = build_characters(args)
        negative = args.negative

    if args.artist_mix:
        # Lazy sibling import: the card library is optional unless the flag is used.
        import nai_artist_strings

        card = nai_artist_strings.get(args.artist_mix)
        if card is None:
            raise SystemExit(
                f"unknown artist mix {args.artist_mix!r}; "
                "run scripts/nai_artist_strings.py list"
            )
        block = nai_artist_strings.compose_block(card)
        # Style-first ordering (V45_ARTIST_MIX_GUIDE): the block leads the base
        # prompt and never enters char_captions -- style is a global layer.
        base = f"{block}, {base}"
        if card.get("neg_add"):
            negative = f"{negative}, {card['neg_add']}" if negative else card["neg_add"]
        print(f"[artist-mix] {card['id']} · {card['name_cn']} — {card['intent']}")
        if args.quality_toggle and card.get("pairing", {}).get("quality_tags") == "off":
            print(
                f"[artist-mix] warning: {card['id']} advises quality tags OFF; "
                "--quality-toggle may flatten this style",
                file=sys.stderr,
            )

    default_w, default_h = DEFAULT_SIZES[args.size]
    args.width = args.width or default_w
    args.height = args.height or default_h
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    token = os.environ.get("NAI_ACCESS_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit(
            "NAI_ACCESS_TOKEN is not set; export the persistent API token first "
            "(or use --dry-run to inspect the request offline)"
        )

    if args.vibe_strength is None:
        args.vibe_strength = [DEFAULT_VIBE_STRENGTH]
    if args.vibe_info is None:
        args.vibe_info = [DEFAULT_VIBE_INFO]
    # Dry-run stays offline and free: an uncached block is left unencoded.
    vibes = build_vibes(args, model_id, token, offline=args.dry_run)

    payload = build_payload(model_id, base, characters, negative, args, vibes)

    if args.dry_run:
        print(json.dumps(elide_bulk(payload), indent=2, ensure_ascii=False))
        return

    saved = generate(payload, token, Path(args.out), vibes)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
