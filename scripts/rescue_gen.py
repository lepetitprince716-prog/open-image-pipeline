#!/usr/bin/env python3
"""Structure rescue: one-shot pose-locked re-render of a structurally broken image.

For drafts whose composition is good but whose structure failed (extra limbs,
impossible joints, object-body fusion, broken proportions). The openpose
preprocessor fits ONE plausible human skeleton to the source, so structural
garbage never enters the control map; re-rendering against that skeleton drops
it while keeping the pose and framing. Content-level errors — an out-of-vocab
token literalised into an object (e.g. "cartwheel" -> an actual wheel) — are
NOT structural and survive the re-render; fix those by passing a corrected
--prompt instead of the sidecar prompt.

Recipe frozen from the 2026-08-28 probe (evidence:
outputs/_analysis/rescue_probe_20260828/): openpose ControlNet strength 1.0 /
end 1.0, preprocessing in-graph on the broken source, resolution matched to the
source (capped), source prompt + source LoRAs from the sidecar, wd14
clothing/colour tags appended so an outfit-less prompt cannot drift into
swimwear. Pose lock is verified on ZZZ/iLustMix only (Z3 Illustrious responds
weakly to openpose) — run through `gen.py rescue --box zzz`, which resolves the
host and forces the checkpoint.

Deliberately NO iteration loop: this line is called by LLMs mid-chat, so it is
one call with a fixed recipe. lineart control is not offered here — measured to
reproduce the structural errors faithfully (see the probe README).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from foot_fix import read_meta, source_loras  # noqa: E402

DEFAULT_N = 3
MAX_SIDE = 1344  # cap the render at draft scale; a hires source must not trigger a hires-sized gen
RESCUE_NEG = "2girls, multiple views, duplicate, clone, nsfw, nude, explicit"

# A z3_quality_gen-authored sidecar prompt already embeds the quality scaffold and the
# LoRA trigger words; re-rendering through the same engine would prepend both again.
# Detect it on the prompt text itself (engine-agnostic), and disable the doubling.
SCAFFOLD_MARK = "masterpiece, best quality"

# wd14 general tags worth carrying into the re-render: clothing and worn accessories.
# Matching is on '_'-delimited words, not raw substrings, so "elbow" is not an outfit
# because of "bow" while "hair_bow" and "blue_dress" still match. Advisory only: a
# bridge failure degrades to no extra tags.
OUTFIT_TOKENS = frozenset((
    "dress", "skirt", "shirt", "blouse", "jacket", "coat", "hoodie", "sweater",
    "pants", "jeans", "shorts", "leggings", "pantyhose", "thighhighs", "socks",
    "shoes", "boots", "sneakers", "heels", "sandals", "uniform", "suit",
    "camisole", "bikini", "swimsuit", "leotard",
    "kimono", "apron", "gloves", "scarf", "hat", "ribbon", "bow", "top",
))


def is_outfit_tag(tag: str) -> bool:
    return any(word in OUTFIT_TOKENS for word in tag.split("_"))


def normalize_res(width: int, height: int, max_side: int = MAX_SIDE) -> str:
    """Source resolution scaled so max(side) <= max_side, snapped to /8."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad source size {width}x{height}")
    scale = min(1.0, max_side / max(width, height))
    w = max(8, int(width * scale) // 8 * 8)
    h = max(8, int(height * scale) // 8 * 8)
    return f"{w}x{h}"


def outfit_tags(source: str) -> list[str]:
    """Clothing tags from the source image via the imgutils bridge. [] on any failure."""
    try:
        from image_recipe import run_bridge

        res = run_bridge("wd14", source)
    except Exception:
        return []
    if not isinstance(res, dict) or not res.get("ok"):
        return []
    general = res.get("general") or {}
    if not isinstance(general, dict):
        return []
    return sorted(t for t in general if is_outfit_tag(t))


def sidecar_loras(meta: dict) -> list[tuple[str, float]]:
    """LoRAs from either sidecar shape: the quality engine's ``content_loras`` /
    ``three_step.role_loras`` (via foot_fix.source_loras) or the draft engine's
    plain ``loras`` list. Tolerant: a malformed entry is skipped, never a traceback."""
    try:
        loras = source_loras(meta)
    except (TypeError, ValueError):
        loras = []
    if loras:
        return loras
    out: list[tuple[str, float]] = []
    for item in meta.get("loras") or []:
        if not (isinstance(item, dict) and item.get("file")):
            continue
        try:
            strength = float(item.get("strength") or 0.8)
        except (TypeError, ValueError):
            strength = 0.8
        out.append((str(item["file"]), strength))
    return out


def compose_prompt(meta: dict, override: str | None, tags: list[str]) -> str:
    """Corrected prompt wins over the sidecar prompt; outfit tags append if absent."""
    base = (override or "").strip() or str(meta.get("prompt") or "").strip()
    if not base:
        raise SystemExit(
            "rescue needs a prompt: the source has no .meta.json prompt and no "
            "--prompt was passed"
        )
    extra = []
    for tag in tags:
        text = tag.replace("_", " ")
        if tag not in base and text not in base:
            extra.append(text)
    return ", ".join([base, *extra]) if extra else base


def build_engine_argv(
    source: str,
    prompt: str,
    loras: list[str],
    res: str,
    n: int,
    seed: int | None,
    base: str | None,
    neg_add: str | None,
) -> list[str]:
    argv = [
        sys.executable, str(SCRIPTS / "z3_quality_gen.py"),
        "--family", "illustrious",
        "--prompt", prompt,
    ]
    if SCAFFOLD_MARK in prompt:
        argv += ["--no-scaffold", "--no-trigger-inject"]
    argv += [
        "--control", "openpose",
        "--control-image", source,
        "--control-strength", "1.0",
        "--control-end", "1.0",
        "--res", res,
        "--n", str(n),
    ]
    if base:
        argv += ["--base", base]
    if seed is not None:
        argv += ["--seed", str(seed)]
    for spec in loras:
        argv += ["--lora", spec]
    neg = f"{RESCUE_NEG}, {neg_add}" if neg_add else RESCUE_NEG
    argv += ["--neg-add", neg]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-shot pose-locked re-render of a structurally broken image"
    )
    ap.add_argument("--source", required=True, help="the broken image (PNG)")
    ap.add_argument("--prompt", default=None,
                    help="corrected prompt; default is the source sidecar prompt")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lora", action="append", default=None,
                    help="FILE[:STRENGTH] (repeatable); default: the source sidecar LoRAs")
    ap.add_argument("--base", default=None, help="checkpoint override (router forces iLustMix on zzz)")
    ap.add_argument("--neg-add", default=None, help="extra negatives on top of the rescue set")
    ap.add_argument("--no-outfit-tags", action="store_true",
                    help="skip the wd14 clothing-tag injection")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    if not src.is_file():
        sys.exit(f"source not found: {args.source}")
    from PIL import Image

    with Image.open(src) as img:
        res = normalize_res(img.width, img.height)

    meta = read_meta(str(src))
    loras = list(args.lora or [])
    if not loras:
        loras = [f"{f}:{s}" for f, s in sidecar_loras(meta)]
    if not loras:
        print(" warn: no LoRA from --lora or the source sidecar — identity is unanchored",
              file=sys.stderr)
    tags = [] if args.no_outfit_tags else outfit_tags(str(src))
    prompt = compose_prompt(meta, args.prompt, tags)

    argv = build_engine_argv(str(src), prompt, loras, res, args.n, args.seed,
                             args.base, args.neg_add)
    print(f"rescue src={src.name} res={res} n={args.n} loras={len(loras)} "
          f"outfit_tags={len(tags)} prompt_override={bool(args.prompt)}")
    proc = subprocess.run(argv, cwd=str(SCRIPTS.parent))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
