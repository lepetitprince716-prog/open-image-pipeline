#!/usr/bin/env python3
"""NovelAI image generation client (txt2img, img2img / Enhance).

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
  python3 scripts/nai_gen.py --prompt "1girl, upper body" \
      --char-ref anchor.png --char-ref-fidelity 0.5   # Precise Reference (V4.5, +5 Anlas)
  .venv/bin/python scripts/nai_gen.py --job job.json --seed 42 \
      --image draft.png --strength 0.4 --width 1280 --height 1856   # img2img / Enhance

img2img: --image is resized to --width x --height and sent with action
img2img. Enhance = the draft's own job and seed at a larger size, strength
0.4, noise 0 (V5 verified 2026-09-23). A same-seed txt2img at a new size is a
new composition, not an upscale. Resizing needs Pillow (the repo venv);
--dry-run skips it and works on the system python3.

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
import contextlib
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
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
USER_AGENT = "nai-gen/0.1 (loramake; +https://github.com/lepetitprince716-prog/open-image-pipeline)"

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

# Named sizes. All three sit inside the Opus free tier at <= 28 steps
# (is_opus_free_eligible).
DEFAULT_SIZES = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

DEFAULT_NEGATIVE = (
    "lowres, worst quality, bad quality, jpeg artifacts, artistic error, watermark"
)

# Anlas daily-budget ledger. Fail-closed: the estimated cost of every paid
# operation is reserved in the ledger under an exclusive lock before the request
# that would spend it, so concurrent invocations cannot both pass the same cap.
DEFAULT_ANLAS_LEDGER = "outputs/nai/anlas_ledger.jsonl"
DEFAULT_ANLAS_BUDGET = 300
ANLAS_FREE_MODEL_PREFIXES = (
    "nai-diffusion-5",
    "nai-diffusion-4",
    "nai-diffusion-3",
    "nai-diffusion-furry-3",
)
# Opus free tier as the novelai.net Generate button showed it on 2026-09-30:
# one image of at most this many pixels in any shape, <= 28 steps, no base image.
OPUS_FREE_MAX_PIXELS = 1024 * 1024
# Outside the free tier one image costs
# ceil(ceil(PIXEL_RATE * px + PIXEL_STEP_RATE * px * steps) * factor).
# The rates come from community clients of the NovelAI web app; the Generate
# button readings in tests/test_nai_gen.py pin them.
ANLAS_PIXEL_RATE = 2.951823174884865e-6
ANLAS_PIXEL_STEP_RATE = 5.753298233447344e-7
# V3 to V4.5 pay the plain formula and V5 pays 1.5x. A family missing from this
# list is priced like V5 until it has been measured.
ANLAS_PLAIN_PRICE_PREFIXES = (
    "nai-diffusion-4",
    "nai-diffusion-3",
    "nai-diffusion-furry-3",
)
ANLAS_V5_PRICE_FACTOR = 1.5
# /dev/null cannot hold a lock file next to it, and a bit bucket ledger records
# nothing anyway: treat it like an explicit opt-out.
ANLAS_LEDGER_OFF = ("", "none", "None", os.devnull)
VIBE_ENCODE_ANLAS = 2

# Precise Reference ("director reference" in the request): V4.5 models only,
# +5 Anlas per reference per generation, never covered by the Opus free tier,
# and incompatible with Vibe Transfer (docs.novelai.net/en/image/precisereference).
DIRECTOR_REFERENCE_ANLAS = 5
DIRECTOR_REFERENCE_TYPES = ("character", "style", "character&style")
PRECISE_REFERENCE_MODEL_PREFIX = "nai-diffusion-4-5"
DEFAULT_REFERENCE_STRENGTH = 1.0
DEFAULT_REFERENCE_FIDELITY = 0.5
# The web UI and the maintained SDKs letterbox every reference into this canvas
# (aspect kept, black padding); doing the same keeps results comparable to the site.
REFERENCE_CANVAS = (1024, 1536)


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


def is_opus_free_eligible(
    model_id, width, height, steps, *, n_samples=1, base_image=False
):
    """True inside the Opus free tier: a V3 to V5 model, at most
    OPUS_FREE_MAX_PIXELS in any shape, <= 28 steps, one image, no base image."""
    if not any(model_id.startswith(p) for p in ANLAS_FREE_MODEL_PREFIXES):
        return False
    if steps > 28:
        return False
    if width * height > OPUS_FREE_MAX_PIXELS:
        return False
    return n_samples == 1 and not base_image


def estimate_generation_cost(
    model_id, width, height, steps, vibe_count=0, reference_count=0, *, n_samples=1, base_image=False
):
    """Anlas upper bound for one request, reserved by the fail-closed ledger.

    Inside the Opus free tier the base is 0. Otherwise each image costs
    ceil(ceil(ANLAS_PIXEL_RATE * px + ANLAS_PIXEL_STEP_RATE * px * steps) * factor)
    with factor 1.5 on V5 and 1.0 on V3 to V4.5, which reproduces every Generate
    button reading taken on novelai.net on 2026-09-30 (tests/test_nai_gen.py).

    Every image of a multi-image request is charged, although the site waives
    one of them on a free canvas. A base image (img2img) is charged the txt2img
    price; the site's img2img price was not measured. Each vibe beyond the
    first 4 adds a per-generation cost, and every Precise Reference adds its
    flat per-generation charge on top of either base.

    Blind spot: V5 free renders also draw on the account's Opus usage limit (the
    site showed "99% of Opus Generations remaining"). What the site bills once
    that runs out was not measured, and this estimate cannot see the limit.
    """
    if is_opus_free_eligible(
        model_id, width, height, steps, n_samples=n_samples, base_image=base_image
    ):
        base = 0
    else:
        pixels = width * height
        raw = ANLAS_PIXEL_RATE * pixels + ANLAS_PIXEL_STEP_RATE * pixels * steps
        factor = (
            1.0
            if model_id.startswith(ANLAS_PLAIN_PRICE_PREFIXES)
            else ANLAS_V5_PRICE_FACTOR
        )
        base = math.ceil(math.ceil(raw) * factor) * n_samples
    extra_vibes = max(0, vibe_count - 4)
    # Extra vibes are billed per generation, on top of the per-encode charge.
    return (
        base
        + extra_vibes * VIBE_ENCODE_ANLAS
        + reference_count * DIRECTOR_REFERENCE_ANLAS
    )


def _ledger_date():
    return datetime.date.today().isoformat()


def _ledger_path(args):
    if args.anlas_ledger in ANLAS_LEDGER_OFF:
        return None
    return Path(args.anlas_ledger)


def _load_ledger(path):
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as err:
            raise SystemExit(f"invalid Anlas ledger entry at line {len(records) + 1}: {err}") from err
    return records


@contextlib.contextmanager
def _file_lock(path):
    """Exclusive inter-process lock guarding one file (ledger or cache entry)."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _write_records(path, records):
    """Append whole records durably; the caller must hold the ledger lock."""
    if not records:
        return
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())


def ledger_total(records, date):
    """Conservative Anlas total for one date.

    A reservation counts at its reserved cost until a `resolve` record settles
    it: `charged` swaps in the final cost, `released` drops it, and anything
    still unresolved (a crashed or killed run) keeps counting, so an interrupted
    charge can never come back as free budget.
    """
    reserved = {}
    resolutions = {}
    total = 0.0
    for record in records:
        if record.get("date") != date:
            continue
        if record.get("kind") == "resolve":
            resolutions[record.get("ref")] = record
            continue
        key = record.get("id")
        if key is None:
            total += float(record.get("cost", 0))
            continue
        reserved[key] = float(record.get("cost", 0))
    for key, cost in reserved.items():
        resolution = resolutions.get(key)
        if resolution is None:
            total += cost
        elif resolution.get("status") != "released":
            total += float(resolution.get("cost", cost))
    return total


def used_today(path, date=None):
    if path is None:
        return 0
    return ledger_total(_load_ledger(path), date or _ledger_date())


def reserve_anlas(path, budget, items):
    """Atomically check today's budget and reserve every item's estimated cost.

    `items` is a sequence of (kind, cost, details). The whole batch is checked
    and written under one exclusive lock, so two concurrent runs can never both
    read the same pre-spend total. Returns (used_before, reservation_ids) with
    one id per item (None when the ledger is disabled).
    """
    items = list(items)
    total = sum(cost for _, cost, _ in items)
    if path is None:
        return 0, [None] * len(items)
    date = _ledger_date()
    with _file_lock(path):
        used = ledger_total(_load_ledger(path), date)
        if used + total > budget:
            raise SystemExit(
                f"Anlas budget exceeded: used {used:.1f}/{budget}, "
                f"this call estimated {total:.1f}; "
                f"use --anlas-budget to raise the limit or --anlas-ledger none to disable"
            )
        now = datetime.datetime.now().isoformat()
        records = []
        ids = []
        for kind, cost, details in items:
            key = uuid.uuid4().hex[:16]
            ids.append(key)
            record = {
                "date": date,
                "time": now,
                "id": key,
                "kind": kind,
                "cost": float(cost),
                "status": "reserved",
            }
            if details:
                record["details"] = details
            records.append(record)
        _write_records(path, records)
    return used, ids


def resolve_reservation(path, reservation_id, status, cost):
    """Settle one reservation as charged, failed, or released.

    `failed` keeps the cost counted: a request that errored out after leaving
    this process may still have been billed.
    """
    if path is None or reservation_id is None:
        return
    record = {
        "date": _ledger_date(),
        "time": datetime.datetime.now().isoformat(),
        "kind": "resolve",
        "ref": reservation_id,
        "status": status,
        "cost": float(cost),
    }
    with _file_lock(path):
        _write_records(path, [record])


def check_anlas_budget(path, budget, estimated_cost):
    """Read-only budget check for --dry-run; reserves nothing."""
    used = used_today(path)
    if path is not None and used + estimated_cost > budget:
        raise SystemExit(
            f"Anlas budget exceeded: used {used:.1f}/{budget}, "
            f"this call estimated {estimated_cost:.1f}; "
            f"use --anlas-budget to raise the limit or --anlas-ledger none to disable"
        )
    return used


def preflight_vibes(args, model_id):
    """Offline cache check for every --vibe; returns uncached count without spending."""
    paths = list(args.vibe or [])
    if not paths:
        return []
    strengths = spread_vibe_values(args.vibe_strength, len(paths), "--vibe-strength")
    infos = spread_vibe_values(args.vibe_info, len(paths), "--vibe-info")
    records = []
    for path, strength, info in zip(paths, strengths, infos, strict=False):
        record = resolve_vibe(path, info, model_id, None, args.vibe_cache, offline=True)
        record["strength"] = strength
        records.append(record)
    return records


def is_v4_family(model_id):
    return model_id.startswith(("nai-diffusion-4", "nai-diffusion-5"))


def validate_center(center, origin):
    try:
        x, y = float(center["x"]), float(center["y"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"bad center in {origin}; expected {{'x': 0..1, 'y': 0..1}}") from None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise SystemExit(f"center out of range in {origin}; coordinates are 0..1")
    return {"x": x, "y": y}


def parse_center(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"bad --char-pos {text!r}; expected x,y like 0.3,0.5") from None
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
        ) from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise SystemExit(f"network error reaching {url}: {err}") from err


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
    cached.parent.mkdir(parents=True, exist_ok=True)
    # One paid encode per cache entry, even across parallel runs: the second
    # process waits here and then finds the block the first one wrote.
    with _file_lock(cached):
        if cached.is_file():
            record["block"] = cached.read_bytes()
            record["cached"] = True
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
        # Publish atomically: a reader on the unlocked fast path must never see
        # a half-written block, which would silently mis-transfer style.
        tmp = cached.with_name(f"{cached.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            with tmp.open("wb") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, cached)
        finally:
            tmp.unlink(missing_ok=True)
    record["block"] = blob
    record["cached"] = False
    return record


def build_vibes(args, model_id, token, offline=False, settle=None):
    """Resolve every --vibe into an encoded block plus its strength.

    `settle(index, status)` is called for each reference, keyed by its position
    in `--vibe`, as soon as its encode call returns or raises, so a failure
    part-way through the batch cannot lose the accounting of the encodes that
    already happened. Positions, not (source, info) pairs, keep repeated
    references from settling one another's reservation.
    """
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
    for index, (path, strength, info) in enumerate(zip(paths, strengths, infos, strict=False)):
        if not (0.0 <= strength <= 1.0):
            raise SystemExit(f"--vibe-strength {strength} out of range; expected 0..1")
        try:
            record = resolve_vibe(
                path, info, model_id, token, args.vibe_cache, offline=offline
            )
        except BaseException:
            if settle is not None:
                settle(index, "failed")
            raise
        if settle is not None:
            settle(index, "released" if record["cached"] else "charged")
        record["strength"] = strength
        vibes.append(record)
        if record["block"] is None:
            state = "not encoded yet (offline)"
        elif record["cached"]:
            state = f"cached {record['cache']}"
        else:
            state = f"encoded -> {record['cache']} ({VIBE_ENCODE_ANLAS} Anlas)"
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


def spread_values(values, count, flag, per="--vibe"):
    """Broadcast one value across every item, or require one value each."""
    values = list(values or [])
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise SystemExit(
            f"{flag} takes one value or one per {per}; got {len(values)} for {count}"
        )
    return values


def spread_vibe_values(values, count, flag):
    return spread_values(values, count, flag, per="--vibe")


def letterbox_reference(raw, canvas=REFERENCE_CANVAS):
    """Fit one reference image into the canvas (aspect kept, black padding) as PNG bytes."""
    from PIL import Image  # lazy: only Precise Reference needs Pillow

    with Image.open(io.BytesIO(raw)) as src:
        image = src.convert("RGB")
    target_w, target_h = canvas
    src_w, src_h = image.size
    if src_w * target_h > target_w * src_h:
        new_w, new_h = target_w, max(1, round(target_w * src_h / src_w))
    else:
        new_w, new_h = max(1, round(target_h * src_w / src_h)), target_h
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    background = Image.new("RGB", canvas, (0, 0, 0))
    background.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    buffer = io.BytesIO()
    background.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_init_image(path, width, height):
    """img2img source resized to the target size, as base64 PNG."""
    from PIL import Image  # lazy: only img2img needs Pillow

    with Image.open(path) as src:
        image = src.convert("RGB").resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def apply_img2img(payload, image, strength, noise):
    """Copy of a generate payload turned into an img2img request."""
    params = {**payload["parameters"], "image": image, "strength": strength, "noise": noise}
    return {**payload, "action": "img2img", "parameters": params}


def _reference_specs(args):
    """Job `character_references` entries first, then the CLI --char-ref flags."""
    specs = []
    for i, spec in enumerate(getattr(args, "character_references", None) or []):
        if not isinstance(spec, dict) or "image" not in spec:
            raise SystemExit(
                f"job character_references[{i}] needs an 'image' key "
                "(plus optional type / strength / fidelity)"
            )
        specs.append(
            {
                "image": spec["image"],
                "type": spec.get("type", "character"),
                "strength": spec.get("strength", DEFAULT_REFERENCE_STRENGTH),
                "fidelity": spec.get("fidelity", DEFAULT_REFERENCE_FIDELITY),
            }
        )
    paths = list(getattr(args, "char_ref", None) or [])
    if paths:
        types = spread_values(
            getattr(args, "char_ref_type", None) or ["character"],
            len(paths), "--char-ref-type", per="--char-ref",
        )
        strengths = spread_values(
            getattr(args, "char_ref_strength", None) or [DEFAULT_REFERENCE_STRENGTH],
            len(paths), "--char-ref-strength", per="--char-ref",
        )
        fidelities = spread_values(
            getattr(args, "char_ref_fidelity", None) or [DEFAULT_REFERENCE_FIDELITY],
            len(paths), "--char-ref-fidelity", per="--char-ref",
        )
        for path, kind, strength, fidelity in zip(paths, types, strengths, fidelities, strict=False):
            specs.append(
                {"image": path, "type": kind, "strength": strength, "fidelity": fidelity}
            )
    return specs


def build_references(args, model_id):
    """Resolve Precise Reference inputs into records with letterboxed PNG bytes.

    Fail-closed before anything is spent: wrong model family, a vibe in the same
    request, an unknown type, an out-of-range slider or a missing file all exit.
    """
    specs = _reference_specs(args)
    if not specs:
        return []
    if not model_id.startswith(PRECISE_REFERENCE_MODEL_PREFIX):
        raise SystemExit(
            f"Precise Reference is only available on the V4.5 models, not {model_id}; "
            "use --model v4.5-full or v4.5-curated"
        )
    if getattr(args, "vibe", None):
        raise SystemExit(
            "Precise Reference and Vibe Transfer cannot share one request on NovelAI; "
            "drop --vibe or --char-ref"
        )
    records = []
    for spec in specs:
        kind = spec["type"]
        if kind not in DIRECTOR_REFERENCE_TYPES:
            raise SystemExit(
                f"reference type {kind!r} unknown; expected one of {DIRECTOR_REFERENCE_TYPES}"
            )
        strength, fidelity = float(spec["strength"]), float(spec["fidelity"])
        for name, value in (("strength", strength), ("fidelity", fidelity)):
            if not -1.0 <= value <= 1.0:
                raise SystemExit(
                    f"reference {name} {value} out of range; expected -1..1 "
                    "(negative values are the site's 'ugly reference' trick)"
                )
        source = Path(spec["image"])
        if not source.is_file():
            raise SystemExit(f"reference image not found: {source}")
        records.append(
            {
                "source": str(source),
                "type": kind,
                "strength": strength,
                "fidelity": fidelity,
                "image": letterbox_reference(source.read_bytes()),
            }
        )
    return records


def reference_parameters(references):
    """The five parallel arrays of a Precise Reference request.

    Shape pinned from the maintained SDKs that mirror the web UI: the Fidelity
    slider travels inverted as the secondary strength, information_extracted is
    always 1, and each description only names the reference type.
    """
    if not references:
        return {}
    return {
        "director_reference_images": [
            base64.b64encode(r["image"]).decode("ascii") for r in references
        ],
        "director_reference_descriptions": [
            {"caption": {"base_caption": r["type"], "char_captions": []}, "legacy_uc": False}
            for r in references
        ],
        "director_reference_strength_values": [round(r["strength"], 2) for r in references],
        "director_reference_secondary_strength_values": [
            round(1.0 - r["fidelity"], 2) for r in references
        ],
        "director_reference_information_extracted": [1.0 for _ in references],
    }


def build_payload(
    model_id, base, characters, negative, args, vibes=None, references=None
):
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
    for key, value in reference_parameters(references or []).items():
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


# Vibe blocks and the img2img source are megabytes of base64; the manifest and
# the dry-run dump both keep the provenance and drop the payload.
VIBE_BULK_KEYS = (
    "reference_image_multiple",
    "reference_image",
    "director_reference_images",
    "image",
)


def append_manifest(
    out_dir, payload, saved, vibes=None, references=None, image_source=None
):
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
    if image_source:
        record["image_source"] = str(Path(image_source).resolve())
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
    if references:
        record["character_references"] = [
            {
                "source": r["source"],
                "type": r["type"],
                "strength": r["strength"],
                "fidelity": r["fidelity"],
            }
            for r in references
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
            i if isinstance(i, str) and i.startswith("<unencoded ")
            else f"<base64 {len(i)} chars>"
            for i in items
        ]
        params[key] = marks if isinstance(value, list) else marks[0]
    return {**payload, "parameters": params}


def save_generation(
    body, payload, out_dir, vibes=None, references=None, image_source=None
):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = save_response(body, out_dir, stamp)
    append_manifest(out_dir, payload, saved, vibes, references, image_source)
    return saved


def generate(payload, token, out_dir, vibes=None, references=None):
    body = post_json(API_URL, payload, token)
    return save_generation(body, payload, out_dir, vibes, references)


def main():
    parser = argparse.ArgumentParser(description="NovelAI txt2img / img2img client")
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
        "--char-ref",
        action="append",
        help="Precise Reference image (V4.5 only, +5 Anlas each per generation); "
        "repeat for several references, which the site blends together",
    )
    parser.add_argument(
        "--char-ref-type",
        action="append",
        choices=DIRECTOR_REFERENCE_TYPES,
        help="reference type per --char-ref, or one for all (default character)",
    )
    parser.add_argument(
        "--char-ref-strength",
        action="append",
        type=float,
        help=f"Strength slider per --char-ref, or one for all (default {DEFAULT_REFERENCE_STRENGTH})",
    )
    parser.add_argument(
        "--char-ref-fidelity",
        action="append",
        type=float,
        help=f"Fidelity slider per --char-ref, or one for all (default {DEFAULT_REFERENCE_FIDELITY})",
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
    parser.add_argument(
        "--image",
        help="img2img source; resized to --width x --height (Enhance: draft's job "
        "and seed, larger size, --strength 0.4 --noise 0)",
    )
    parser.add_argument(
        "--strength", type=float, default=0.4, help="img2img strength 0..1 (default 0.4)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, help="img2img added noise 0..1 (default 0)"
    )
    parser.add_argument("--out", default="outputs/nai", help="output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request payload and exit; no network, no token needed",
    )
    parser.add_argument(
        "--anlas-ledger",
        default=os.environ.get("NAI_ANLAS_LEDGER", DEFAULT_ANLAS_LEDGER),
        help="JSONL path for the daily Anlas ledger (default outputs/nai/anlas_ledger.jsonl; "
        "set to 'none' or NAI_ANLAS_LEDGER=none to disable)",
    )
    parser.add_argument(
        "--anlas-budget",
        type=float,
        default=float(os.environ.get("NAI_ANLAS_BUDGET", DEFAULT_ANLAS_BUDGET)),
        help="daily Anlas budget (default 300, set NAI_ANLAS_BUDGET)",
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
        args.character_references = job.get("character_references", [])
    else:
        if not args.prompt:
            parser.error("--prompt or --job is required")
        model_id = resolve_model(args.model)
        base = args.prompt
        characters = build_characters(args)
        negative = args.negative
        args.character_references = []

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
    if args.image:
        if not Path(args.image).is_file():
            raise SystemExit(f"--image not found: {args.image}")
        if args.width % 64 or args.height % 64:
            raise SystemExit("img2img width and height must be multiples of 64")
        if not 0.0 <= args.strength <= 1.0 or not 0.0 <= args.noise <= 1.0:
            raise SystemExit("--strength and --noise must be within 0..1")

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

    ledger_path = _ledger_path(args)
    references = build_references(args, model_id)
    if references:
        print(
            f"[reference] {len(references)} precise reference(s): "
            f"+{len(references) * DIRECTOR_REFERENCE_ANLAS} Anlas per generation",
            file=sys.stderr,
        )
    preflight = preflight_vibes(args, model_id)
    uncached = [(i, v) for i, v in enumerate(preflight) if not v["cached"]]
    gen_cost = estimate_generation_cost(
        model_id, args.width, args.height, args.steps, len(preflight), len(references),
        base_image=bool(args.image),
    )
    vibe_cost = len(uncached) * VIBE_ENCODE_ANLAS
    total_cost = gen_cost + vibe_cost

    if args.dry_run:
        used = check_anlas_budget(ledger_path, args.anlas_budget, total_cost)
        payload = build_payload(
            model_id, base, characters, negative, args, preflight, references
        )
        if args.image:
            # Offline: show the shape without needing Pillow for the resize.
            payload = apply_img2img(
                payload, f"<unencoded image: {args.image}>", args.strength, args.noise
            )
        print(json.dumps(elide_bulk(payload), indent=2, ensure_ascii=False))
        print(
            f"[anlas] dry-run estimate: {total_cost:.1f} "
            f"(used {used:.1f}/{args.anlas_budget})",
            file=sys.stderr,
        )
        return

    # Encode before reserving: a bad source or missing Pillow fails for free.
    init_image = (
        encode_init_image(args.image, args.width, args.height) if args.image else None
    )

    # Reserve the whole estimated spend before the first paid request: the check
    # and the write happen under one ledger lock, so a parallel run sees this
    # call's cost even though nothing has been charged yet.
    encode_items = [
        (
            "vibe_encode",
            VIBE_ENCODE_ANLAS,
            {
                "source": v["source"],
                "model": v["model"],
                "information_extracted": v["information_extracted"],
            },
        )
        for _, v in uncached
    ]
    gen_item = (
        "generation",
        gen_cost,
        {
            "model": model_id,
            "size": f"{args.width}x{args.height}",
            "steps": args.steps,
            "references": len(references),
            "action": "img2img" if args.image else "generate",
        },
    )
    used, reservations = reserve_anlas(
        ledger_path, args.anlas_budget, encode_items + [gen_item]
    )
    encode_reservations = {
        index: rid for (index, _), rid in zip(uncached, reservations, strict=False)
    }
    gen_reservation = reservations[-1]
    print(
        f"[anlas] budget {used:.1f}/{args.anlas_budget}, "
        f"this call {total_cost:.1f} ({gen_cost} gen + {vibe_cost} encode) reserved",
        file=sys.stderr,
    )

    def settle_encode(index, status):
        rid = encode_reservations.pop(index, None)
        resolve_reservation(ledger_path, rid, status, VIBE_ENCODE_ANLAS)

    try:
        vibes = build_vibes(args, model_id, token, offline=False, settle=settle_encode)
        payload = build_payload(
            model_id, base, characters, negative, args, vibes, references
        )
        if init_image:
            payload = apply_img2img(payload, init_image, args.strength, args.noise)
    except BaseException:
        # The generation request never left, and neither did the encodes still
        # holding a reservation, so that part of the spend is free again.
        for index in list(encode_reservations):
            settle_encode(index, "released")
        resolve_reservation(ledger_path, gen_reservation, "released", gen_cost)
        raise

    try:
        body = post_json(API_URL, payload, token)
    except BaseException:
        # The request may still have been billed; keep counting it.
        resolve_reservation(ledger_path, gen_reservation, "failed", gen_cost)
        raise
    # Settle before saving: the image is already billed once the POST returns.
    resolve_reservation(ledger_path, gen_reservation, "charged", gen_cost)

    saved = save_generation(
        body, payload, Path(args.out), vibes, references, image_source=args.image
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
