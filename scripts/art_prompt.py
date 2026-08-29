#!/usr/bin/env python3
"""Artistic prompt builder for the loramake single-character line.

Turns a character/subject into an anti-AI-slop positive prompt BODY plus a complete SFW negative, and
recommends an aspect (WxH) and an optional detail sampler. The body keeps composition / lighting / pose
/ medium / palette as __wildcards__ so the quality engine varies them per seed — escaping the
centred-bust, flat-light, plastic-skin sameness that reads as "AI slop".

It is consumed two ways:
  - imported by gen.py's `artistic` intent: build() returns the pieces; gen.py routes them to
    z3_quality_gen.py and reuses its idle-probe / best-of-N / box quirks.
  - run directly to print the pieces, or a ready z3_quality_gen.py command (--as-cmd).

The engine (z3_quality_gen.py) prepends its own quality scaffold and any LoRA trigger words, then
expands the wildcards per image (z3_quality_gen.py:805,559). The final positive per image is therefore
    [quality scaffold] + [LoRA triggers] + [this body]
which matches the research-backed tag order: quality -> style -> medium/composition -> content.

Style modes (the per-image policy is the caller's; all three are supported):
  lora    : no artist tags in the prompt; a style LoRA (passed to the engine via --lora) carries the
            look through its trigger. Traceable; does not name an artist. (default)
  artist  : inject Danbooru `(by name:weight)` tags — the strongest single style steer, but it bakes a
            specific (often living) artist's style into the image. Requires --artist.
  none    : neither; rely on medium / composition / lighting tags alone.

Usage:
  art_prompt.py "1girl, solo, long silver hair, red eyes"
  art_prompt.py "1girl, solo, ..." --style-mode artist --artist "by ke-ta:1.0, by wlop:0.5" --aspect tall
  art_prompt.py "1girl, ..." --medium "oil painting" --camera "cowboy shot" --as-cmd --lora "char:0.8"
"""
from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# SDXL-optimal resolutions (~1 MP); non-square ratios bias away from the centred-bust square default.
ASPECTS: dict[str, tuple[int, int]] = {
    "portrait": (832, 1216),      # default character framing
    "tall": (768, 1344),          # dramatic vertical
    "square": (1024, 1024),       # face / detail study (most slop-prone — use deliberately)
    "landscape": (1216, 832),     # environmental
    "wide": (1344, 768),          # cinematic establishing
    "hi-portrait": (1024, 1536),  # engine default; more native pixels, heavier upscale
}
DEFAULT_ASPECT = "portrait"

# richer-texture sampler for a deliberate "hero" shot (research: dpmpp_3m_sde = richest detail);
# None keeps the recipe default (dpmpp_2m, clean / controlled).
DETAIL_SAMPLER = "dpmpp_3m_sde"

STYLE_MODES = ("lora", "artist", "none")

# Complete SFW negative: research baseline + the anatomy guards the showcase line has but the quality
# recipe lacks (asymmetrical eyes, deformed, mutated). `chromatic aberration` and `artist name` belong
# in the negative for Illustrious even when artist tags are used in the positive — they suppress literal
# signatures / lens artefacts, not the artist's style (which comes from the positive `(by ...)` tags).
NEGATIVE = (
    "worst quality, low quality, bad quality, normal quality, lowres, jpeg artifacts, "
    "signature, watermark, username, artist name, "
    "bad anatomy, bad hands, malformed hands, malformed feet, bad proportions, "
    "extra fingers, extra toes, missing fingers, missing toes, fused fingers, fused toes, "
    "extra digits, fewer digits, missing limbs, "
    "asymmetrical eyes, uneven eyes, cross-eyed, deformed, mutated, "
    "blurry, out of focus, chromatic aberration, sketch, censor, "
    "text, 3d, realistic, photo, multiple views, "
    "nsfw, child, loli, petite, childish, baby face, aged down, chibi"  # guard the base's young/loli drift
)

# SFW line-strategy guards (five-layer decoupling plan N1: 线策略守卫). These are line
# policy, not card aesthetics: they migrated OUT of card neg_add (blue_hour_silhouette,
# white_coat_nocturne, stained_glass_nouveau, neon_rain_wash, overhead_repose) so a card
# face declares aesthetics only. hands_macro_story still repeats see-through/skin tight
# because the art_directions.py sync rule forces the primary base's quirks neg_add onto
# the card — the rule keeps them for the non-artistic lines that consume quirks directly.
# The purity lint keeps identity/body language out, and validate() is the aesthetic gate.
# Rollback = drop this constant.
LINE_GUARDS = (
    "bare legs",
    "thigh slit",
    "cleavage",
    "see-through",
    "suggestive pose",
    "skin tight",
)

# per-segment default value = the matching wildcard token (engine varies it per image); the order here
# is the body order (after any artist tags, before the subject). `composition` (figure-vs-frame) is a
# default slot since v2; `expression` is opt-in (see build()) so pre-v2 commands keep their prompts.
_SEGMENT_TOKENS = (
    ("medium", "__medium__"),
    ("camera", "__camera__"),
    ("composition", "__composition__"),
    ("lighting", "__lighting__"),
    ("pose", "__pose__"),
    ("color_mood", "__color_mood__"),
)

EXPRESSION_TOKEN = "__expression__"


@dataclass(frozen=True)
class ArtPrompt:
    """The artistic prompt pieces. `positive` is a body with wildcards; the engine prepends its quality
    scaffold + LoRA triggers and expands the wildcards per seed."""
    positive: str
    negative: str
    width: int
    height: int
    sampler: str | None  # None = keep the recipe default sampler


def _segment(value: str | None, token: str) -> str | None:
    """Resolve one segment: None -> wildcard token (vary per image); '' / 'none' -> drop; else literal."""
    if value is None:
        return token
    v = value.strip()
    if not v or v.lower() == "none":
        return None
    return v


def _artist_tags(spec: str) -> str:
    """Normalise a comma-separated artist spec into weighted Danbooru tags.
    'by ke-ta:1.0, wlop:0.5' -> '(by ke-ta:1.0), (by wlop:0.5)'; a bare 'ke-ta' -> '(by ke-ta)'."""
    out: list[str] = []
    for raw in spec.split(","):
        tag = raw.strip()
        if not tag:
            continue
        name, _, weight = tag.partition(":")
        name = name.strip()
        if not name.lower().startswith("by "):
            name = "by " + name
        weight = weight.strip()
        out.append(f"({name}:{weight})" if weight else f"({name})")
    if not out:
        raise ValueError("artist spec parsed to nothing")
    return ", ".join(out)


def build(subject: str, *, style_mode: str = "lora", artist: str | None = None,
          medium: str | None = None, camera: str | None = None, composition: str | None = None,
          lighting: str | None = None, pose: str | None = None, color_mood: str | None = None,
          expression: str | None = None, aspect: str = DEFAULT_ASPECT,
          detail: bool = False) -> ArtPrompt:
    """Assemble the artistic prompt body + negative + aspect + sampler. Pure string logic, no GPU.

    Each composition segment defaults to its __wildcard__ token (varied per seed by the engine); pass a
    literal to fix it, or 'none' to drop it. Segment order matches the research-backed tag order; the
    engine adds quality tags ahead of all of this via its scaffold.

    `expression` is opt-in (the slot is absent by default): 'vary' -> __expression__ wildcard, a
    literal pins it, None / '' / 'none' leave the prompt without the slot. It sits after color_mood,
    right before the subject (character-adjacent).
    """
    subject = (subject or "").strip()
    if not subject:
        raise ValueError("subject prompt is required (e.g. '1girl, solo, long silver hair, red eyes')")
    if style_mode not in STYLE_MODES:
        raise ValueError(f"unknown style_mode {style_mode!r}; choose {' | '.join(STYLE_MODES)}")
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}; choose from {', '.join(ASPECTS)}")
    if style_mode == "artist" and not (artist and artist.strip()):
        raise ValueError("style_mode 'artist' needs --artist (e.g. 'by ke-ta:1.0, by wlop:0.5'); "
                         "auto-picking an artist is unreliable, so it is not done")

    parts: list[str] = []
    if style_mode == "artist":
        assert artist is not None  # guaranteed by the validation above; narrows for the type checker
        parts.append(_artist_tags(artist))
    values = {"medium": medium, "camera": camera, "composition": composition,
              "lighting": lighting, "pose": pose, "color_mood": color_mood}
    for key, token in _SEGMENT_TOKENS:
        seg = _segment(values[key], token)
        if seg:
            parts.append(seg)
    if expression is not None:  # opt-in slot: absent unless asked for
        expr = EXPRESSION_TOKEN if expression.strip().lower() == "vary" \
            else _segment(expression, EXPRESSION_TOKEN)
        if expr:
            parts.append(expr)
    parts.append(subject)

    w, h = ASPECTS[aspect]
    return ArtPrompt(positive=", ".join(parts),
                     negative=NEGATIVE + ", " + ", ".join(LINE_GUARDS), width=w, height=h,
                     sampler=DETAIL_SAMPLER if detail else None)


def to_command(ap: ArtPrompt, loras: list[str] | None = None) -> list[str]:
    """A ready argv invoking z3_quality_gen.py (illustrious --max) with these pieces."""
    cmd = [sys.executable, str(ROOT / "z3_quality_gen.py"), "--family", "illustrious", "--max",
           "--prompt", ap.positive, "--neg", ap.negative, "--res", f"{ap.width}x{ap.height}"]
    if ap.sampler:
        cmd += ["--sampler", ap.sampler]
    for lora in loras or []:
        cmd += ["--lora", lora]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Artistic (anti-slop) prompt builder for the single-character line.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="subject/character tags, e.g. '1girl, solo, long silver hair, red eyes'")
    ap.add_argument("--style-mode", choices=STYLE_MODES, default="lora",
                    help="lora (style LoRA carries the look; default) | artist (by-name tags) | none")
    ap.add_argument("--artist", default=None,
                    help="artist mode: comma list, e.g. 'by ke-ta:1.0, by wlop:0.5'")
    ap.add_argument("--medium", default=None, help="fix the medium (default: __medium__ wildcard; 'none' drops it)")
    ap.add_argument("--camera", default=None, help="fix framing (default: __camera__ wildcard)")
    ap.add_argument("--composition", default=None,
                    help="fix figure-vs-frame composition (default: __composition__ wildcard; 'none' drops)")
    ap.add_argument("--lighting", default=None, help="fix lighting (default: __lighting__ wildcard)")
    ap.add_argument("--pose", default=None, help="fix pose (default: __pose__ wildcard)")
    ap.add_argument("--color-mood", dest="color_mood", default=None,
                    help="fix palette mood (default: __color_mood__ wildcard)")
    ap.add_argument("--expression", default=None,
                    help="opt-in slot: omit = no expression segment (unchanged behavior); "
                         "'vary' = __expression__ per image; a literal pins it; 'none' = same as omit")
    ap.add_argument("--aspect", choices=tuple(ASPECTS), default=DEFAULT_ASPECT,
                    help=f"output aspect (default {DEFAULT_ASPECT})")
    ap.add_argument("--detail", action="store_true",
                    help=f"use the richer-detail sampler ({DETAIL_SAMPLER}) instead of the recipe default")
    ap.add_argument("--lora", action="append", default=None,
                    help="--as-cmd only: LoRA spec to include in the printed command (repeatable)")
    ap.add_argument("--as-cmd", action="store_true", help="print a ready z3_quality_gen.py command")
    args = ap.parse_args()

    try:
        result = build(args.subject, style_mode=args.style_mode, artist=args.artist, medium=args.medium,
                       camera=args.camera, composition=args.composition, lighting=args.lighting,
                       pose=args.pose, color_mood=args.color_mood, expression=args.expression,
                       aspect=args.aspect, detail=args.detail)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    if args.as_cmd:
        print(" ".join(shlex.quote(c) for c in to_command(result, args.lora)))
    else:
        print(f"positive: {result.positive}")
        print(f"negative: {result.negative}")
        print(f"res     : {result.width}x{result.height}")
        print(f"sampler : {result.sampler or '(recipe default)'}")


if __name__ == "__main__":
    main()
