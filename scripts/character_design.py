#!/usr/bin/env python3
"""Character design cards: visual-only character sheets that compile into T2I prompts.

A design card is the unit of character work the way a concept card is the unit of scene
work. It holds nothing but what a viewer can see — silhouette, face, hair, eyes, one
signature element, one accent colour, a register written as visible expressions, outfit
layers in character-neutral wording, accessories with a scene class — and a render ledger
that records, element by element, whether the model actually drew what the card says.
There is no room on the card for backstory, personality adjectives or lore: the lint
refuses them, which is how the card stays a design sheet and never becomes a novel sheet.

The script never authors a card (a person or a delegated author model does), never
renders (``probe`` writes briefs and prints the exact commands), and never edits
character packs or outfit cards (``promote`` prints fragments to paste).

Usage:
    python3 scripts/character_design.py fields
    python3 scripts/character_design.py new ID [--subject female] [--lines anima,nai] [--out PATH]
    python3 scripts/character_design.py lint CARD [--json]
    python3 scripts/character_design.py compile CARD --line anima|illustrious|nai|prose
                                         [--surface tag|nl|both] [--frame full|bust]
                                         [--set id|sig|outfit|acc] [--outfit ID]
                                         [--expression N] [--scene-class default] [--json]
    python3 scripts/character_design.py probe CARD --out DIR [--line anima] [--seeds 7,8]
                                         [--sets id,sig,outfit,acc] [--frames full,bust] [--outfit ID]
    python3 scripts/character_design.py spec DIR --out SPEC.json
    python3 scripts/character_design.py ledger CARD (--batch ID | --result-json FILE)
                                         [--write] [--set element=action ...] [--json]
    python3 scripts/character_design.py promote CARD [--out FILE] [--force]

Exit codes: 0 ok · 1 lint errors / refused status · 2 usage or refused surface.
Authority: docs/runbooks/character_design_runbook.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The schema tuples, the outfit/accessory lexicons and the pure lint rules live in the
# stdlib-only leaf loramake/design_lint.py (Phase C, item C2). They are re-exported here so
# every importer keeps reading them from character_design.
from loramake import design_lint  # noqa: E402
from loramake.design_lint import (  # noqa: E402
    ANCHOR_OPTIONAL, ANCHOR_REQUIRED, BODY_LANGUAGE, CARD_LINES, CONCEPT_MAX, Finding, ID_RE, IMPRESSION_MAX,
    INTENT_MAX, MATERIAL_WORDS, MOTION_CLASSES, OPTIONAL_KEYS, OUTFIT_COVERAGES, OUTFIT_REQUIRED, OUTFIT_STATUSES,
    REQUIRED_KEYS, SCENE_CLASSES, SCHEMA, STATUSES, SUBJECTS, WHOLE_OBJECTS, _color_words, _dedup, _enum_ok,
    _lint_accessories, _lint_design_notes, _lint_outfits, _lint_schema, _mentions, _str_list, _strs, _word_count,
    color_lock, identity_tags, prompt_phrases,
)

PROBE_SCHEMA = "loramake.character-design-probe/v1"
PROMOTE_SCHEMA = "loramake.character-design-promote/v1"
CARD_HOME = Path("configs") / "character_designs"
PACK_SCHEMA = HERE / "character_pack.schema.json"

LINES = CARD_LINES + ("prose",)  # prose = the NL surface with no line rules
SURFACES = ("tag", "nl", "both")
FRAMES = ("full", "bust")
ACTIONS = ("keep", "rephrase", "demote", "drop")
SET_NAMES = ("id", "sig", "outfit", "acc")
REGISTER_MAX_WORDS = 12
NAI_CHARACTER_MAX = 70   # concept round 3: the character block that wins is 60-70 words
ANIMA_MAX = 120          # HAREM_PROMPT_GUIDE: ~117 words is a step-function failure
NEGATIVE_MAX_WORDS = 10  # aestheticV11 short negative
NAI_MODEL = "nai-diffusion-5-full"
FRAME_TOKENS = {
    "full": ("standing", "full body", "simple background", "white background"),
    "bust": ("upper body", "portrait", "simple background", "white background"),
}
FRAME_RES = {"anima": {"full": "portrait", "bust": "square"},
             "illustrious": {"full": "896x1152", "bust": "1024x1024"}}

# ---------------------------------------------------------------------------
# Lexicons. `fields` prints them; the runbook points here instead of restating.
# BODY_LANGUAGE, MATERIAL_WORDS and WHOLE_OBJECTS live in loramake/design_lint.py next to
# the rules that read them; the rest stay here with lint_card.
# ---------------------------------------------------------------------------

# Words that describe a person instead of a picture. Any of these on a prompt-facing
# field is the card turning into a novel sheet.
NARRATIVE_LEXICON = (
    # personality adjectives — the visible form belongs in register.allowed_expressions
    "kind", "loyal", "brave", "cowardly", "shy", "cruel", "gentle", "clever", "smart",
    "genius", "intelligent", "lonely", "friendly", "honest", "arrogant", "humble",
    "ambitious", "stubborn", "caring", "cheerful", "moody", "timid", "ruthless", "noble",
    "kind-hearted", "cold-hearted", "warm-hearted", "sociable", "introverted", "extroverted",
    # backstory
    "past", "history", "childhood", "born", "raised", "grew up", "became", "betrayed",
    "dreams", "wants", "believes", "secretly", "personality", "backstory", "lore",
    "memories", "trauma", "revenge", "destiny", "fate", "mission", "years old", "age",
    "works as", "lives in", "came from", "orphan", "survivor",
    # institutions and relationships
    "guild", "faction", "kingdom", "empire", "corporation", "organisation", "organization",
    "sister", "brother", "mother", "father", "daughter", "son", "rival", "lover",
    "boyfriend", "girlfriend", "servant", "heir", "princess of", "queen of", "leader of",
)
NARRATIVE_CLAUSE_RE = re.compile(
    r"\b(she|he|they|her|his|their)\s+(is|was|has|had|wants|loves|hates|fears|likes|"
    r"dislikes|works|lives|grew|became|believes)\b", re.IGNORECASE)

# Medium and style words: identity must not carry style (that is the direction layer).
STYLE_WORDS = (
    "watercolor", "watercolour", "oil painting", "sketch", "2d", "3d", "anime style",
    "thick painting", "impasto", "risograph", "charcoal drawing", "pastel drawing",
    "cel shading", "cel-shaded", "flat color", "lineart", "monochrome", "pixel art",
    "realistic", "photorealistic", "screentone", "ink wash", "gouache", "anime screencap",
)
QUALITY_WORDS = (  # union of concept_cards.QUALITY_TAGS and the catalog landing ban list
    "masterpiece", "best quality", "amazing quality", "very aesthetic", "absurdres",
    "highres", "score_9", "score_8_up", "score_", "newest", "very awa",
)
WEIGHT_RE = re.compile(r"\(\s*[^()]+:\s*-?\d+(?:\.\d+)?\s*\)|\(\([^()]+\)\)|\{[^{}]+\}|-?\d+(?:\.\d+)?::")

HAIR_COLORS = (
    "blonde", "ash blonde", "platinum", "silver", "white", "grey", "gray", "black", "brown",
    "dark brown", "auburn", "red", "ginger", "orange", "pink", "purple", "violet", "lavender",
    "blue", "navy", "teal", "green", "mint", "wine", "burgundy", "indigo", "chestnut",
)
EYE_COLORS = (
    "amber", "gold", "golden", "yellow", "crimson", "red", "scarlet", "emerald", "green",
    "grey", "gray", "hazel", "brown", "ice blue", "blue", "teal", "violet", "purple", "pink",
    "black", "silver", "ruby",
)
COLOR_WORDS = tuple(sorted(set(HAIR_COLORS) | set(EYE_COLORS) | {
    "charcoal", "ash grey", "ash gray", "cream", "ivory", "beige", "tan", "khaki", "olive",
    "maroon", "coral", "peach", "turquoise", "cyan", "magenta", "bronze", "copper",
}))
# Neutrals do not fight a direction card's color_mood; only saturated colours count towards
# an outfit's palette_strong flag.
NEUTRAL_COLORS = ("black", "white", "grey", "gray", "charcoal", "ash grey", "ash gray", "cream",
                  "ivory", "beige", "silver")
HETEROCHROMIA_WORDS = ("heterochromia", "odd eyes", "odd-eyed", "mismatched eyes",
                       "different colored eyes", "different coloured eyes")
GLOVE_HALF_WORDS = ("half-worn", "half worn", "half-on", "half on", "half-off", "half off",
                    "removing", "pulling off", "taking off", "dangling", "one glove off")
# Concept round 4 (2026-09-02): NAI draws a mechanism as a surface, never as internals.
MECHANISM_WORDS = (
    "socket", "exposed joint", "ball joint", "internals", "exposed wiring", "wiring",
    "drivetrain", "bicycle chain", "drive chain", "heat vent", "open panel", "exposed gears",
    "gears", "piston", "engine", "circuitry", "exposed mechanism", "cable port",
    "charging port", "hydraulics",
)
VISIBLE_ANCHORS = (
    "gaze", "look", "looking", "eyes", "eyelids", "half-lidded", "brow", "brows", "eyebrow",
    "eyebrows", "mouth", "lips", "lip", "smile", "smiling", "smirk", "grin", "frown", "pout",
    "teeth", "chin", "head tilt", "head tilted", "cheek", "cheeks", "blush", "stare",
    "staring", "glance", "glancing", "wink", "squint", "jaw", "expression", "expressionless",
    "parted lips", "closed eyes", "eyes closed", "tears", "nose", "forehead",
)
GAZE_WORDS = ("looking", "gaze", "glance", "glancing", "stare", "staring", "eyes closed",
              "closed eyes", "eyes lowered", "wink")


class DesignError(ValueError):
    """A user-facing card problem."""


# ---------------------------------------------------------------------------
# Card IO
# ---------------------------------------------------------------------------

def load_card(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DesignError(f"cannot read card: {exc}") from None
    except json.JSONDecodeError as exc:
        raise DesignError(f"card is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise DesignError("card must be a JSON object")
    return data


def dump_card(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False, indent=2) + "\n"


# Fields that do not change what a probe rendered: the ledger, the status the ledger
# advances, and provenance notes. Everything else is the design, and a probe is bound to it.
# The one part of the ledger that does change the compiled prompt — which elements are
# demoted or dropped — is folded back into the hash, so a demotion is a new revision.
SHA_EXCLUDED = ("render_ledger", "status", "notes", "concept", "impression_keywords", "anchor")


def card_sha256(card: dict) -> str:
    design = {k: v for k, v in card.items() if k not in SHA_EXCLUDED}
    design["_compiled_out"] = sorted(_demoted(card))
    canonical = json.dumps(design, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def card_sha8(card: dict) -> str:
    return card_sha256(card)[:8]


def skeleton(design_id: str, subject: str, lines: Sequence[str]) -> dict:
    return {
        "schema": SCHEMA,
        "id": design_id,
        "name_cn": "TODO 中文名",
        "intent": "TODO 一句视觉意图",
        "subject": subject,
        "target_lines": list(lines),
        "silhouette": ["TODO build", "TODO height"],
        "face": ["TODO face shape"],
        "hair": {"color": "TODO colour hair", "length": "TODO length hair",
                 "shape": "TODO hairstyle", "detail": "TODO one hair detail"},
        "eyes": {"color": "TODO colour eyes", "shape": "TODO eye shape"},
        "skin_marks": [],
        "signature_element": "TODO the one recognisable visual hook",
        "palette": {"base": ["TODO base colour"], "accent": "TODO accent colour"},
        "register": {
            "description": "TODO visible demeanour in under twelve words",
            "allowed_expressions": ["TODO calm gaze, looking at viewer", "TODO second visible expression"],
            "excluded": [],
        },
        "outfits": [{
            "id": "default", "name_cn": "TODO 服装名",
            "tags": ["TODO garment", "TODO material word", "TODO second material word"],
            "sfw": True, "coverage": "full", "motion_allow": ["static", "seated"],
            "neg_add": "", "status": "unvalidated",
        }],
        "accessories": [],
        "neg_guards": [],
        "forbidden": [],
        "notes": [],
        "status": "draft",
        "render_ledger": {"elements": {}, "rows": []},
    }


# ---------------------------------------------------------------------------
# Lint. The pure rules live in loramake/design_lint.py; this block composes them.
# ---------------------------------------------------------------------------

def pack_token_table(characters_dir: Path | None = None) -> list[str]:
    """Identity tokens owned by the existing packs (identity_tags + color_lock).

    Facade over ``design_lint.pack_token_table`` that keeps the repository default directory.
    Mirrors scripts/test_card_purity.py token_table(); the test asserts both agree.
    """
    return design_lint.pack_token_table(characters_dir or ROOT / "characters")


def lint_card(card: dict, pack_tokens: Sequence[str] | None = None) -> list[Finding]:
    """Return every finding (errors and warnings) for one card."""
    out: list[Finding] = []
    if not _lint_schema(card, out):
        return out
    own_tokens = identity_tags(card)
    packs = list(pack_tokens) if pack_tokens is not None else pack_token_table()
    # An own token that is also an existing pack token is a collision, not a double report.
    packs = [t for t in packs if t not in own_tokens]

    for path, text in prompt_phrases(card):
        lowered = text.lower()
        for word in NARRATIVE_LEXICON:
            if _mentions(text, word):
                out.append(Finding("error", path, "visual_language",
                                   f"narrative word {word!r} (write what is visible instead)"))
        if NARRATIVE_CLAUSE_RE.search(text):
            out.append(Finding("error", path, "visual_language",
                               "pronoun clause reads like a character biography"))
        if re.search(r"\d", text):
            out.append(Finding("error", path, "no_digits", "digits do not render (no ages, heights, head counts)"))
        if not text.isascii():
            out.append(Finding("error", path, "english_only", "prompt-facing text must be ASCII English"))
        if WEIGHT_RE.search(text):
            out.append(Finding("error", path, "no_weight_syntax", "no (x:1.2), ((x)), {x} or 1.3:: weights"))
        for word in QUALITY_WORDS:
            if word in lowered:
                out.append(Finding("error", path, "no_quality_tags", f"quality/score tag {word!r}"))
                break
        if "_" in text or "@" in text:
            out.append(Finding("error", path, "no_underscore_at",
                               "no '_' or '@' (the card carries no artist; style is a direction layer)"))
        for word in STYLE_WORDS:
            if _mentions(text, word):
                out.append(Finding("error", path, "no_style_words",
                                   f"style word {word!r} (identity must not carry style)"))
        if "glove" in lowered:
            if any(w in lowered for w in GLOVE_HALF_WORDS):
                out.append(Finding("error", path, "render_priors",
                                   "half-worn gloves never render (HAREM_PROMPT_GUIDE prior ceiling)"))
            else:
                out.append(Finding("warning", path, "render_priors",
                                   "gloves: only fully worn renders; plural 'gloved hands' fuses hands"))
    free_text = [("intent", card.get("intent"))]
    for oi, outfit in enumerate(card["outfits"]):
        free_text += [(f"outfits[{oi}].{f}", outfit.get(f)) for f in ("intent", "note")]
    for path, text in free_text:
        if not isinstance(text, str):
            continue
        for word in NARRATIVE_LEXICON:
            if _mentions(text, word):
                out.append(Finding("error", path, "visual_language",
                                   f"narrative word {word!r} (write what is visible instead)"))
        if NARRATIVE_CLAUSE_RE.search(text):
            out.append(Finding("error", path, "visual_language", "pronoun clause reads like a character biography"))
        if WEIGHT_RE.search(text):
            out.append(Finding("error", path, "no_weight_syntax", "no (x:1.2), ((x)), {x} or 1.3:: weights"))
        for word in QUALITY_WORDS:
            if word in text.lower():
                out.append(Finding("error", path, "no_quality_tags", f"quality/score tag {word!r}"))
                break
        for word in STYLE_WORDS:
            if _mentions(text, word):
                out.append(Finding("error", path, "no_style_words", f"style word {word!r} (identity must not carry style)"))

    hair, eyes = card["hair"], card["eyes"]
    if not _color_words(hair["color"], HAIR_COLORS):
        out.append(Finding("error", "hair.color", "explicit_colors", "must name a hair colour"))
    eye_colors = _color_words(eyes["color"], EYE_COLORS)
    if not eye_colors:
        out.append(Finding("error", "eyes.color", "explicit_colors", "must name an eye colour"))
    hetero = any(_mentions(eyes["color"], w) for w in HETEROCHROMIA_WORDS) or len(eye_colors) > 1
    if hetero and card.get("allow_heterochromia") is not True:
        out.append(Finding("error", "eyes.color", "render_priors",
                           "heterochromia never renders (prior ceiling); set allow_heterochromia to override"))

    for path in ("signature_element",) + tuple(f"skin_marks[{i}]" for i in range(len(card.get("skin_marks") or []))):
        text = card["signature_element"] if path == "signature_element" else card["skin_marks"][int(path[11:-1])]
        for word in MECHANISM_WORDS:
            if _mentions(text, word):
                out.append(Finding("error", path, "mechanism_words",
                                   f"{word!r}: mechanism only as an outer shell or lit readout (concept round 4)"))
    for oi, outfit in enumerate(card.get("outfits") or []):
        if isinstance(outfit, dict) and _str_list(outfit.get("tags")):
            for ti, tag in enumerate(outfit["tags"]):
                for word in MECHANISM_WORDS:
                    if _mentions(tag, word):
                        out.append(Finding("error", f"outfits[{oi}].tags[{ti}]", "mechanism_words",
                                           f"{word!r}: mechanism only as an outer shell (concept round 4)"))
    for ai, acc in enumerate(card.get("accessories") or []):
        if isinstance(acc, dict) and isinstance(acc.get("phrase"), str):
            for word in MECHANISM_WORDS:
                if _mentions(acc["phrase"], word):
                    out.append(Finding("error", f"accessories[{ai}].phrase", "mechanism_words",
                                       f"{word!r}: mechanism only as an outer shell (concept round 4)"))

    sig = card["signature_element"].strip()
    words = _word_count(sig)
    if not 2 <= words <= 8:
        out.append(Finding("error", "signature_element", "one_signature", "one phrase of 2-8 words"))
    elif "," in sig or " and " in f" {sig.lower()} ":
        out.append(Finding("error", "signature_element", "one_signature",
                           "exactly one element (no commas, no 'and')"))
    others = [hair.get(k, "") for k in ("color", "length", "shape", "detail")] + [eyes.get("color", "")]
    for outfit in card.get("outfits") or []:
        if isinstance(outfit, dict):
            others += list(outfit.get("tags") or [])
    if any(sig.lower() == o.strip().lower() for o in others if isinstance(o, str)):
        out.append(Finding("error", "signature_element", "one_signature",
                           "must differ from hair, eye and outfit phrases"))

    palette = card["palette"]
    base = palette["base"]
    accent = palette["accent"].strip()
    if not 1 <= len(base) <= 3:
        out.append(Finding("error", "palette.base", "one_accent", "1-3 base colours"))
    if _word_count(accent) != 1 or not _color_words(accent, COLOR_WORDS):
        out.append(Finding("error", "palette.accent", "one_accent", "exactly one colour word"))
    elif any(_mentions(b, accent) for b in base):
        out.append(Finding("error", "palette.accent", "one_accent", "accent must not appear in base"))

    register = card["register"]
    if _word_count(register["description"]) > REGISTER_MAX_WORDS:
        out.append(Finding("error", "register.description", "register_cap",
                           f"at most {REGISTER_MAX_WORDS} words; personality lives only as visible demeanour"))
    exprs = register["allowed_expressions"]
    if not 2 <= len(exprs) <= 6:
        out.append(Finding("error", "register.allowed_expressions", "register_cap", "2-6 visible expressions"))
    for i, expr in enumerate(exprs):
        if not any(_mentions(expr, a) for a in VISIBLE_ANCHORS):
            out.append(Finding("error", f"register.allowed_expressions[{i}]", "register_cap",
                               "must name something visible (gaze, brows, mouth, lips, smile, head tilt ...)"))

    _lint_outfits(card, own_tokens, packs, out)
    _lint_accessories(card, own_tokens, packs, out)

    if not any(f.level == "error" for f in out):
        _lint_budgets(card, out)
        _lint_status(card, out)
    return out


def _lint_budgets(card: dict, out: list[Finding]) -> None:
    try:
        from concept_cards import BLACKLIST, WORD_BAND  # noqa: WPS433
    except ImportError:  # pragma: no cover - concept_cards is part of the repo
        BLACKLIST, WORD_BAND = {}, (30, 130)
    for outfit in card["outfits"]:
        oid = outfit["id"]
        for line in card["target_lines"]:
            for frame in FRAMES:
                where = f"compile.{line}.{frame}.{oid}"
                compiled = compile_card(card, line, "tag", frame=frame, set_name=top_set(card, oid), outfit_id=oid)
                text = compiled["tag"].lower()
                for token in BLACKLIST.get(line, ()):
                    if token in text:
                        out.append(Finding("error", where, "line_blacklist",
                                           f"blacklisted token for {line}: {token!r}"))
                words = compiled["word_count"]
                if not WORD_BAND[0] <= words <= WORD_BAND[1]:
                    out.append(Finding("error", where, "word_budget", f"{words} words outside band {WORD_BAND}"))
                if line == "anima" and words > ANIMA_MAX:
                    out.append(Finding("error", where, "word_budget",
                                       f"{words} words over the Anima full-body ceiling {ANIMA_MAX}"))
                if line == "nai":
                    char_words = _word_count(compiled["job"]["characters"][0]["prompt"])
                    if char_words > NAI_CHARACTER_MAX:
                        out.append(Finding("warning", where, "word_budget",
                                           f"character block {char_words} words; round-3 winners stay under "
                                           f"{NAI_CHARACTER_MAX}"))


def _lint_status(card: dict, out: list[Finding]) -> None:
    ledger = card.get("render_ledger") or {}
    elements = ledger.get("elements") or {}
    rows = ledger.get("rows") or []
    status = card["status"]
    if status == "probed" and not rows and not elements:
        out.append(Finding("error", "status", "status_gate", "probed needs at least one ledger row"))
    if status == "verified":
        for problem in verified_problems(card):
            out.append(Finding("error", "status", "status_gate", problem))
    for key, entry in elements.items():
        if not isinstance(entry, dict) or entry.get("action") not in ACTIONS:
            out.append(Finding("error", f"render_ledger.elements.{key}", "ledger_shape",
                               f"action must be one of {ACTIONS}"))


VERIFIED_MIN_SEEDS = 2


def active_outfits(card: dict) -> list[dict]:
    """Outfits that still count: not demoted or dropped in the ledger, not rejected on the card."""
    skipped = _demoted(card)
    return [o for o in card.get("outfits") or []
            if isinstance(o, dict) and f"outfit:{o.get('id')}" not in skipped and o.get("status") != "rejected"]


def verified_problems(card: dict) -> list[str]:
    """Why the card is not verified: every live element of every active outfit needs action
    keep plus positive verdicts from two distinct seeds on every frame it compiles into, at
    its current phrase and rendered from the current card revision (same card hash).
    Comment-only rows and renders of an earlier revision never count."""
    elements = (card.get("render_ledger") or {}).get("elements") or {}
    problems: list[str] = []
    outfits = active_outfits(card)
    if not outfits:
        problems.append("no active outfit (every outfit is demoted, dropped or rejected)")
    current = card_sha8(card)  # only renders of this exact revision are evidence
    checked: set[tuple[str, str]] = set()
    for outfit in outfits:
        for frame in FRAMES:
            for key, toks in element_tokens(card, frame, top_set(card, outfit["id"])):
                if (key, frame) in checked:
                    continue
                checked.add((key, frame))
                entry = elements.get(key) if isinstance(elements.get(key), dict) else None
                if entry is None or entry.get("action") != "keep":
                    problems.append(f"{key}: no ledger action keep")
                    continue
                if entry.get("phrase") != ", ".join(toks):
                    problems.append(f"{key}: phrase changed since its last probe")
                    continue
                seeds = {v.get("seed") for v in entry.get("verdicts") or []
                         if v.get("frame") == frame and v.get("verdict") in ("love", "yes")
                         and v.get("revision") == entry.get("revision") and v.get("card_sha8") == current}
                if len(seeds) < VERIFIED_MIN_SEEDS:
                    problems.append(f"{key}/{frame}: {len(seeds)} positive seed(s) at card {current}, "
                                    f"need {VERIFIED_MIN_SEEDS} (renders of an earlier revision do not count)")
    return sorted(set(problems))


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def outfit_for(card: dict, outfit_id: str | None = None) -> dict:
    """Public accessor for one outfit (the first when ``outfit_id`` is None); raises DesignError."""
    return _outfit(card, outfit_id)


def _outfit(card: dict, outfit_id: str | None) -> dict:
    outfits = [o for o in card.get("outfits") or [] if isinstance(o, dict)]
    if not outfits:
        raise DesignError("card has no outfits")
    if outfit_id is None:
        return outfits[0]
    for outfit in outfits:
        if outfit.get("id") == outfit_id:
            return outfit
    raise DesignError(f"no outfit {outfit_id!r} on this card (have {[o.get('id') for o in outfits]})")


def accessories_for(card: dict, scene_class: str = "default") -> list[tuple[int, dict]]:
    """Accessories that compile for a scene class: the default ones plus that class's own."""
    return [(i, a) for i, a in enumerate(card.get("accessories") or [])
            if isinstance(a, dict) and a.get("scene_class", "default") in ("default", scene_class)]


def top_set(card: dict, outfit_id: str | None = None, scene_class: str = "default") -> str:
    outfit = _outfit(card, outfit_id)
    return f"acc:{outfit['id']}" if accessories_for(card, scene_class) else f"outfit:{outfit['id']}"


def set_parent(set_name: str) -> str | None:
    if set_name == "id":
        return None
    if set_name == "sig":
        return "id"
    kind, _, oid = set_name.partition(":")
    if kind == "outfit":
        return "sig"
    if kind == "acc":
        return f"outfit:{oid}"
    raise DesignError(f"unknown element set {set_name!r}")


def resolve_set(card: dict, set_name: str, outfit_id: str | None, scene_class: str = "default") -> str:
    """Expand the short set names (outfit / acc) to their outfit-qualified form."""
    if set_name in ("id", "sig"):
        return set_name
    if set_name in ("outfit", "acc"):
        oid = _outfit(card, outfit_id)["id"]
        if set_name == "acc" and not accessories_for(card, scene_class):
            raise DesignError(f"no accessory compiles for scene class {scene_class!r}; "
                              "the acc set would only duplicate the outfit set")
        return f"{set_name}:{oid}"
    set_parent(set_name)  # validates the shape
    return set_name


def _demoted(card: dict) -> set[str]:
    elements = (card.get("render_ledger") or {}).get("elements") or {}
    return {k for k, e in elements.items() if isinstance(e, dict) and e.get("action") in ("demote", "drop")}


def element_tokens(card: dict, frame: str, set_name: str, expression_index: int = 0,
                   scene_class: str = "default") -> list[tuple[str, list[str]]]:
    """Ordered (element key, tokens) pairs for one frame and element set, demotions applied."""
    if frame not in FRAMES:
        raise DesignError(f"frame must be one of {FRAMES}")
    kind, _, oid = set_name.partition(":")
    level = {"id": 0, "sig": 1, "outfit": 2, "acc": 3}[kind]
    hair, eyes = card["hair"], card["eyes"]
    exprs = card["register"]["allowed_expressions"]
    expr = exprs[expression_index % len(exprs)]
    expr_tokens = [expr]
    if not any(_mentions(expr, g) for g in GAZE_WORDS):
        expr_tokens.append("looking at viewer")
    ordered: list[tuple[str, list[str]]] = [
        ("silhouette", list(card["silhouette"])),
        ("hair", [hair[k] for k in ("color", "length", "shape", "detail") if hair.get(k)]),
        ("eyes", [eyes[k] for k in ("color", "shape") if eyes.get(k)]),
        ("face", list(card["face"])),
    ]
    if level >= 1:
        if frame == "bust" and card.get("skin_marks"):
            ordered.append(("skin_marks", list(card["skin_marks"])))
        ordered.append(("signature_element", [card["signature_element"]]))
    if level >= 2:
        outfit = _outfit(card, oid)
        ordered.append((f"outfit:{outfit['id']}", list(outfit["tags"])))
    if level >= 3:
        for ai, acc in accessories_for(card, scene_class):
            ordered.append((f"accessory:{ai}", [acc["phrase"]] + ([acc["placement"]] if acc.get("placement") else [])))
    ordered.append(("expression", expr_tokens))
    demoted = _demoted(card)
    return [(key, toks) for key, toks in ordered if key not in demoted]


def set_tokens(card: dict, frame: str, set_name: str, **kw: object) -> list[str]:
    tokens: list[str] = []
    for _, toks in element_tokens(card, frame, set_name, **kw):  # type: ignore[arg-type]
        tokens += toks
    return _dedup(tokens)


def set_delta(card: dict, frame: str, set_name: str, **kw: object) -> list[str]:
    """Tokens this set adds on top of its parent (what a probe render is testing)."""
    parent = set_parent(set_name)
    own = set_tokens(card, frame, set_name, **kw)
    if parent is None:
        return own
    base = {t.lower() for t in set_tokens(card, frame, parent, **kw)}
    return [t for t in own if t.lower() not in base]


def _guards(card: dict) -> list[str]:
    return _dedup(list(card.get("neg_guards") or []) + list(card.get("forbidden") or []))


def _outfit_neg_add(outfit: dict) -> list[str]:
    extra = outfit.get("neg_add") or ""
    if isinstance(extra, str):
        return [part.strip() for part in extra.split(",") if part.strip()]
    if isinstance(extra, list):
        return [str(part).strip() for part in extra if str(part).strip()]
    return []


def _compile_negatives(card: dict, set_name: str) -> list[str]:
    """Card-level guards, plus the selected outfit's neg_add once the set includes that outfit.

    Identity and signature sets stay outfit-free: garment defect tokens must not leak into
    the ablation rungs that are proving the figure, not the clothes.
    """
    tokens = list(_guards(card))
    kind, _, oid = set_name.partition(":")
    if kind in ("outfit", "acc"):
        tokens.extend(_outfit_neg_add(_outfit(card, oid or None)))
    return _dedup(tokens)


def _nl_sentence(prefix: str, tokens: Sequence[str]) -> str:
    text = ", ".join(tokens)
    if not text:
        return ""
    text = f"{prefix}{text}" if prefix else text
    return text[0].upper() + text[1:] + "."


def natural_language(card: dict, frame: str, set_name: str, **kw: object) -> str:
    """The NL surface: the same tokens as the tag surface joined by connectives only."""
    noun, pronoun = SUBJECTS[card["subject"]][2], SUBJECTS[card["subject"]][3]
    groups = dict(element_tokens(card, frame, set_name, **kw))  # type: ignore[arg-type]
    sentences = [
        _nl_sentence(f"one {noun}, ", groups.get("silhouette", [])),
        _nl_sentence("", groups.get("hair", [])),
        _nl_sentence("", groups.get("eyes", [])),
        _nl_sentence("", groups.get("face", [])),
        _nl_sentence("", groups.get("skin_marks", [])),
        _nl_sentence("", groups.get("signature_element", [])),
    ]
    outfit_tokens = [t for k, toks in groups.items() if k.startswith("outfit:") for t in toks]
    sentences.append(_nl_sentence(f"{pronoun} wears ", outfit_tokens))
    acc_tokens = [t for k, toks in groups.items() if k.startswith("accessory:") for t in toks]
    sentences.append(_nl_sentence("", acc_tokens))
    sentences.append(_nl_sentence("", groups.get("expression", [])))
    sentences.append(_nl_sentence("", FRAME_TOKENS[frame]))
    return " ".join(s for s in sentences if s)


def compile_card(card: dict, line: str, surface: str = "tag", frame: str = "full",
                 set_name: str | None = None, outfit_id: str | None = None,
                 expression_index: int = 0, scene_class: str = "default") -> dict:
    """Deterministic prompt assembly for one line, surface, frame and element set."""
    if line not in LINES:
        raise DesignError(f"line must be one of {LINES}")
    if surface not in SURFACES:
        raise DesignError(f"surface must be one of {SURFACES}")
    if frame not in FRAMES:
        raise DesignError(f"frame must be one of {FRAMES}")
    if scene_class not in SCENE_CLASSES:
        raise DesignError(f"scene-class must be one of {SCENE_CLASSES}")
    if line in ("nai", "illustrious") and surface != "tag":
        why = "tag lists beat sentences on V5 (concept round 5)" if line == "nai" \
            else "Illustrious/NoobAI read Danbooru tags, not prose"
        raise DesignError(f"{line} has no NL surface: {why}; see the runbook dialect table")
    if line == "prose" and surface != "nl":
        raise DesignError("prose is the NL-only surface for NL-first engines; use --surface nl")
    if line in CARD_LINES and line not in (card.get("target_lines") or []):
        raise DesignError(f"{line} is not in the card's target_lines {card.get('target_lines')}; "
                          "add it there so lint covers that dialect before anything compiles for it")
    set_name = resolve_set(card, set_name or top_set(card, outfit_id, scene_class), outfit_id, scene_class)
    kw = {"expression_index": expression_index, "scene_class": scene_class}
    elements = element_tokens(card, frame, set_name, **kw)
    body = _dedup(t for _, toks in elements for t in toks)
    count_tag, char_tag = SUBJECTS[card["subject"]][0], SUBJECTS[card["subject"]][1]
    frame_tokens = list(FRAME_TOKENS[frame])
    guards = _compile_negatives(card, set_name)
    result: dict = {
        "design": card["id"], "line": line, "surface": surface, "frame": frame, "set": set_name,
        "elements": [key for key, _ in elements], "tag": None, "nl": None, "negative": "",
        "qc_cull": [], "job": None, "word_count": 0,
    }
    tag_tokens = [count_tag, "solo"] + body + frame_tokens
    if line == "anima":
        result["tag"] = ", ".join(tag_tokens)
        negative = []
        for g in guards:
            if _word_count(" ".join(negative + [g])) > NEGATIVE_MAX_WORDS:
                break
            negative.append(g)
        result["negative"] = ", ".join(negative)
        result["qc_cull"] = list(card.get("forbidden") or [])
    elif line == "illustrious":
        result["tag"] = ", ".join(tag_tokens)
        result["negative"] = ", ".join(guards)
    elif line == "nai":
        try:
            from concept_cards import DEFAULT_NEGATIVE
        except ImportError:  # pragma: no cover
            DEFAULT_NEGATIVE = "lowres, worst quality, bad quality, jpeg artifacts, artistic error, watermark"
        base = ", ".join([count_tag, "solo"] + frame_tokens)
        character = ", ".join([char_tag] + body)
        result["negative"] = ", ".join(_dedup([DEFAULT_NEGATIVE] + guards))
        result["job"] = {"model": NAI_MODEL, "base": base,
                         "characters": [{"prompt": character}], "negative": result["negative"]}
        result["tag"] = f"{base}\n{character}"
    if surface in ("nl", "both") and line in ("anima", "prose"):
        result["nl"] = natural_language(card, frame, set_name, **kw)
    if surface == "nl" and line == "anima":
        result["tag"] = None
    result["word_count"] = _word_count(result["tag"] or result["nl"] or "")
    return result


# ---------------------------------------------------------------------------
# Probe / spec
# ---------------------------------------------------------------------------

def probe_sets(card: dict, requested: Sequence[str] | None, outfit_id: str | None,
               scene_class: str = "default") -> list[str]:
    """The ablation ladder for one outfit; the acc rung exists only when an accessory
    actually compiles for the scene class (otherwise it would re-render the outfit rung)."""
    outfit = _outfit(card, outfit_id)
    ladder = ["id", "sig", f"outfit:{outfit['id']}"]
    if accessories_for(card, scene_class):
        ladder.append(f"acc:{outfit['id']}")
    if not requested:
        return ladder
    wanted = [resolve_set(card, s, outfit["id"], scene_class) for s in requested]
    unknown = [s for s in wanted if s not in ladder]
    if unknown:
        raise DesignError(f"sets {unknown} are not on this card's ladder {ladder}")
    return [s for s in ladder if s in wanted]


def _brief_for(card: dict, line: str, compiled: dict, frame: str, seed: int) -> dict:
    if line == "anima":
        return {"tier": "draft", "box": "boxa", "res": FRAME_RES["anima"][frame], "n": 1,
                "seed": seed, "sfw": True, "prompt": compiled["tag"]}
    brief = {"tier": "best", "box": "boxb", "res": FRAME_RES["illustrious"][frame], "n": 1,
             "seed": seed, "sfw": True, "prompt": compiled["tag"]}
    if compiled["negative"]:
        brief["neg_add"] = compiled["negative"]
    return brief


def _validate_brief(brief: dict) -> list[str]:
    from app import brief as brief_mod  # lazy: pulls loramake, only needed here
    return brief_mod.validate(brief)


def write_probe(card: dict, line: str, out_dir: Path, seeds: Sequence[int], sets: Sequence[str],
                frames: Sequence[str], scene_class: str = "default") -> dict:
    """Write briefs / jobs, the probe index and commands.sh. Renders nothing."""
    if line not in CARD_LINES:
        raise DesignError(f"probe line must be one of {CARD_LINES}")
    if line not in card["target_lines"]:
        raise DesignError(f"{line} is not in the card's target_lines {card['target_lines']}")
    for set_name in sets:
        _, _, oid = set_name.partition(":")
        if oid and _outfit(card, oid).get("sfw") is not True:
            raise DesignError(f"outfit {oid!r} is sfw:false; the design probe is SFW-only "
                              "(the outfit is linted, never rendered on this line)")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise DesignError(f"{out_dir} is not an empty directory; probes never mix, pick a new --out")
    # Plan first, write last: a brief the app rejects must not leave a half-written round
    # behind that blocks a retry into the same --out.
    files: list[tuple[Path, str]] = []
    items: list[dict] = []
    commands: list[str] = []
    for set_name in sets:
        for frame in frames:
            compiled = compile_card(card, line, "tag", frame=frame, set_name=set_name, scene_class=scene_class)
            stem = f"{set_name.replace(':', '-')}__{frame}"
            delta = set_delta(card, frame, set_name, scene_class=scene_class)
            if line == "nai":
                job_path = out_dir / "jobs" / f"{stem}.job.json"
                files.append((job_path, json.dumps(compiled["job"], ensure_ascii=False, indent=2) + "\n"))
                for seed in seeds:
                    log = out_dir / "results" / f"{stem}__s{seed}.log"
                    render_dir = out_dir / "renders" / stem
                    commands.append(" ".join(shlex.quote(a) for a in (
                        "python3", "scripts/nai_gen.py", "--job", str(job_path), "--seed", str(seed),
                        "--out", str(render_dir))) + f" | tee {shlex.quote(str(log))}")
                    items.append({"set": set_name, "frame": frame, "seed": seed, "job": str(job_path),
                                  "log": str(log), "elements": compiled["elements"], "delta": delta})
            else:
                for seed in seeds:
                    brief = _brief_for(card, line, compiled, frame, seed)
                    problems = _validate_brief(brief)
                    if problems:
                        raise DesignError(f"{stem} seed {seed}: brief rejected by app.brief: {problems}")
                    brief_path = out_dir / "briefs" / f"{stem}__s{seed}.brief.json"
                    files.append((brief_path, json.dumps(brief, ensure_ascii=False, indent=2) + "\n"))
                    result = out_dir / "results" / f"{stem}__s{seed}.json"
                    render_dir = out_dir / "renders" / f"{stem}__s{seed}"
                    # --keep-into: the engines name outputs by seed, so the ladder's sets would
                    # overwrite each other in the shared output tree without a copy per item.
                    commands.append(" ".join(shlex.quote(a) for a in (
                        "python3", "scripts/gen_job.py", str(brief_path), "--json-out", str(result),
                        "--keep-into", str(render_dir))))
                    items.append({"set": set_name, "frame": frame, "seed": seed, "brief": str(brief_path),
                                  "result": str(result), "render_dir": str(render_dir),
                                  "elements": compiled["elements"], "delta": delta})
    index = {"schema": PROBE_SCHEMA, "design": card["id"], "name_cn": card["name_cn"], "line": line,
             "card_sha256": card_sha256(card), "scene_class": scene_class, "items": items}
    files.append((out_dir / "probe_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"))
    files.append((out_dir / "commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n"))
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = commands
    return index


def _rendered_files(item: dict) -> list[str]:
    if item.get("result"):
        path = Path(item["result"])
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        files = data.get("new_files") if isinstance(data, dict) else None
        return [str(f) for f in files or [] if isinstance(f, str)]
    path = Path(item.get("log") or "")
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".png")]


def caption_for(index: dict, item: dict) -> str:
    try:
        from concept_cards import CAPTION_LIMIT
    except ImportError:  # pragma: no cover
        CAPTION_LIMIT = 200
    scene = str(index.get("scene_class") or "default")
    key = f"{index['design']}/{item['set']}/{item['frame']}/{scene}"
    sha8 = str(index.get("card_sha256") or "")[:8] or "unbound0"
    text = (f"[{key}] {index.get('name_cn', '')} · {index['line']} · seed {item['seed']} · card {sha8} — "
            f"+{', '.join(item.get('delta') or [])}")
    if len(text) > CAPTION_LIMIT:
        text = text[:CAPTION_LIMIT - 1] + "…"
    return text


def build_spec(out_dir: Path) -> list[dict]:
    index_path = out_dir / "probe_index.json"
    if not index_path.is_file():
        raise DesignError(f"no probe_index.json under {out_dir}; run probe first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    spec: list[dict] = []
    for item in index.get("items", []):
        for file_path in _rendered_files(item):
            spec.append({"path": file_path, "caption": caption_for(index, item)})
    return spec


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

CAPTION_LINE_RE = re.compile(r"· (anima|illustrious|nai) ·")
CAPTION_SHA_RE = re.compile(r"· card ([0-9a-f]{8})")
SUBSTITUTION_CHIPS = ("foreign_object", "nonsense")
ID_CHIP_MAP = {"face": ("face", "eyes"), "figure": ("silhouette",), "foreign_object": ("hair",),
               "nonsense": ("hair",)}
NOTE_CHIPS = ("style", "composition", "hands")


def parse_probe_caption(caption: str) -> dict | None:
    try:
        from concept_cards import parse_caption_key
    except ImportError:  # pragma: no cover
        return None
    key = parse_caption_key(caption)
    if key is None:
        return None
    composite, seed = key
    parts = composite.split("/")
    if len(parts) == 3:
        design, set_name, frame = parts
        scene = "default"
    elif len(parts) == 4:
        design, set_name, frame, scene = parts
    else:
        return None
    match = CAPTION_LINE_RE.search(caption)
    sha = CAPTION_SHA_RE.search(caption)
    return {"design": design, "set": set_name, "frame": frame, "seed": seed,
            "line": match.group(1) if match else "?", "card_sha8": sha.group(1) if sha else None,
            "scene_class": scene if scene in SCENE_CLASSES else "default"}


def join_probe_verdicts(card: dict, result: dict, allow_stale: bool = False) -> list[dict]:
    """Rows of verdicts for this design from a review_result.read_result() report.

    A render is evidence only for the card revision that produced it: rows whose caption
    carries a different card hash (or none) are stale and skipped unless ``allow_stale``.
    """
    rows: dict[int, dict] = {}
    batch = str(result.get("batch", ""))
    current = card_sha8(card)

    def add(item: dict, verdict: str) -> None:
        parsed = parse_probe_caption(str(item.get("caption", "")))
        if parsed is None or parsed["design"] != card["id"]:
            return
        if not allow_stale and parsed["card_sha8"] != current:
            return
        rows[int(item["i"])] = {
            "i": int(item["i"]), "batch": batch, "line": parsed["line"], "set": parsed["set"],
            "frame": parsed["frame"], "seed": parsed["seed"], "verdict": verdict,
            "reasons": list(item.get("reasons") or []), "comment": str(item.get("comment") or ""),
            "card_sha8": parsed["card_sha8"],
            "scene_class": parsed.get("scene_class") or "default",
        }

    for verdict in ("love", "yes", "no"):
        for item in result.get(verdict) or []:
            add(item, verdict)
    for item in (result.get("comments") or {}).values():
        if int(item["i"]) in rows:
            rows[int(item["i"])]["comment"] = str(item.get("comment") or "")
        else:
            add(item, "comment")
    return [rows[i] for i in sorted(rows)]


def stale_probe_rows(card: dict, result: dict) -> int:
    """How many of the report's rows for this design belong to another card revision."""
    return len(join_probe_verdicts(card, result, allow_stale=True)) - len(join_probe_verdicts(card, result))


def _set_stats(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    stats: dict[tuple[str, str], dict] = {}
    for row in rows:
        s = stats.setdefault((row["set"], row["frame"]),
                             {"decided": 0, "kept": 0, "chips": Counter(), "rows": []})
        s["rows"].append(row)
        if row["verdict"] in ("love", "yes", "no"):
            s["decided"] += 1
            s["kept"] += row["verdict"] != "no"
        for chip in row["reasons"]:
            s["chips"][chip] += 1
    for s in stats.values():
        s["rate"] = round(s["kept"] / s["decided"], 3) if s["decided"] else None
    return stats


def element_actions(card: dict, rows: Sequence[dict], overrides: dict[str, str] | None = None,
                    scene_class: str = "default") -> dict[str, dict]:
    """Per-element ledger entries derived from the verdict rows (the human may override)."""
    overrides = overrides or {}
    stats = _set_stats(rows)
    existing = (card.get("render_ledger") or {}).get("elements") or {}
    entries: dict[str, dict] = {}
    batch = rows[0]["batch"] if rows else ""

    def entry_for(key: str, set_name: str, phrase: str) -> dict:
        if key in entries:
            return entries[key]
        old = existing.get(key) if isinstance(existing.get(key), dict) else None
        revision = int(old.get("revision", 1)) if old else 1
        if old and old.get("phrase") != phrase:
            revision += 1
        entries[key] = {"phrase": phrase, "set": set_name, "revision": revision,
                        "verdicts": [], "reasons": Counter(), "failed": False, "notes": []}
        return entries[key]

    for (set_name, frame), stat in stats.items():
        if stat["decided"] == 0:
            continue  # comment-only rows are history, never evidence for an action
        try:
            elements = element_tokens(card, frame, set_name, scene_class=scene_class)
        except (DesignError, KeyError):
            continue
        own_keys = {key for key, _ in elements}
        parent = set_parent(set_name)
        parent_keys = {key for key, _ in element_tokens(card, frame, parent, scene_class=scene_class)} \
            if parent else set()
        tested = [(key, toks) for key, toks in elements if key not in parent_keys]
        parent_rate = stats.get((parent, frame), {}).get("rate") if parent else None
        lost = parent_rate is not None and stat["rate"] is not None and stat["rate"] < parent_rate
        collapsed = stat["rate"] == 0 and stat["decided"] >= 2
        # A set probed without its parent (--sets outfit / acc) has nothing to be compared
        # against, so any rejection counts against it: fail closed, the human can --set keep.
        isolated = parent is not None and (parent, frame) not in stats
        fail_closed = isolated and stat["rate"] is not None and stat["rate"] < 1.0
        for key, toks in tested:
            e = entry_for(key, set_name, ", ".join(toks))
            for row in stat["rows"]:
                e["verdicts"].append({"batch": row["batch"], "frame": frame, "seed": row["seed"],
                                      "verdict": row["verdict"], "reasons": row["reasons"],
                                      "revision": e["revision"], "card_sha8": row.get("card_sha8")})
            attributed = Counter()
            for chip, n in stat["chips"].items():
                if chip in NOTE_CHIPS:
                    e["notes"].append(f"{chip} x{n} on {set_name}/{frame} (line note, not this element)")
                    continue
                if set_name == "id":
                    if key in ID_CHIP_MAP.get(chip, ()):
                        attributed[chip] += n
                elif chip in SUBSTITUTION_CHIPS or chip not in ID_CHIP_MAP:
                    attributed[chip] += n
            e["reasons"].update(attributed)
            if attributed or lost or collapsed or fail_closed:
                e["failed"] = True
        _ = own_keys

    for key, e in entries.items():
        if key in overrides:
            action = overrides[key]
        elif not e["failed"]:
            action = "keep"
        elif e["revision"] <= 2:
            action = "rephrase"
        else:
            action = "demote"
        e["action"] = action
        e["reasons"] = dict(e["reasons"])
        e["last_batch"] = batch
        del e["failed"]
    for key, action in overrides.items():
        if key not in entries:
            old = existing.get(key) if isinstance(existing.get(key), dict) else {}
            entries[key] = {"phrase": old.get("phrase", ""), "set": old.get("set", "?"),
                            "revision": int(old.get("revision", 1)), "verdicts": [], "reasons": {},
                            "notes": [], "action": action, "last_batch": batch}
    return entries


def merge_ledger(card: dict, entries: dict[str, dict], rows: Sequence[dict]) -> dict:
    """Return a copy of the card whose render_ledger absorbs the new entries and rows."""
    updated = json.loads(json.dumps(card, ensure_ascii=False))
    ledger = updated.setdefault("render_ledger", {"elements": {}, "rows": []})
    ledger.setdefault("elements", {})
    ledger.setdefault("rows", [])
    live_keys: set[str] = set()
    for outfit in updated.get("outfits") or []:
        for frame in FRAMES:
            live_keys |= {k for k, _ in element_tokens(updated, frame, top_set(updated, outfit["id"]))}
    for key, new in entries.items():
        old = ledger["elements"].get(key) if isinstance(ledger["elements"].get(key), dict) else None
        seen = {(v["batch"], v["frame"], v["seed"]) for v in (old or {}).get("verdicts", [])}
        verdicts = list((old or {}).get("verdicts", []))
        verdicts += [v for v in new["verdicts"] if (v["batch"], v["frame"], v["seed"]) not in seen]
        reasons = Counter((old or {}).get("reasons", {}))
        reasons.update(new["reasons"])
        ledger["elements"][key] = {
            "phrase": new["phrase"], "set": new["set"], "revision": new["revision"],
            "verdicts": verdicts, "reasons": dict(reasons), "action": new["action"],
            "last_batch": new["last_batch"], "notes": _dedup((old or {}).get("notes", []) + new["notes"]),
        }
    for key in list(ledger["elements"]):
        if key not in live_keys and key not in entries and key not in _demoted(updated):
            del ledger["elements"][key]
    seen_rows = {(r.get("batch"), r.get("i")) for r in ledger["rows"]}
    for row in rows:
        if (row["batch"], row["i"]) not in seen_rows:
            ledger["rows"].append({k: row.get(k) for k in ("batch", "i", "line", "set", "frame", "seed",
                                                           "verdict", "reasons", "comment", "card_sha8")})
    if updated.get("status") == "draft":
        updated["status"] = "probed"
    return updated


def ledger_table(card: dict, entries: dict[str, dict]) -> str:
    lines = [f"{'element':<26} {'set':<20} {'keep(full)':>10} {'keep(bust)':>10}  chips  action"]
    for key in sorted(entries, key=lambda k: (entries[k]["set"], k)):
        e = entries[key]
        rates = {}
        for frame in FRAMES:
            decided = [v for v in e["verdicts"] if v["frame"] == frame and v["verdict"] in ("love", "yes", "no")]
            kept = sum(v["verdict"] != "no" for v in decided)
            rates[frame] = f"{kept}/{len(decided)}" if decided else "-"
        chips = ",".join(f"{c}x{n}" for c, n in sorted(e["reasons"].items())) or "-"
        lines.append(f"{key:<26} {e['set']:<20} {rates['full']:>10} {rates['bust']:>10}  {chips}  {e['action']}")
    return "\n".join(lines)


def outfit_evidence_skip(card: dict, outfit: dict, entry: dict | None) -> str | None:
    """Why this outfit must not be exported. None means the current keep still holds."""
    if entry is None or entry.get("action") != "keep":
        return "no ledger evidence (probe it, or promote --force)"
    phrase = ", ".join(outfit.get("tags") or [])
    if entry.get("phrase") != phrase:
        return "outfit phrase changed since its last probe"
    if not (entry.get("verdicts") or []):
        return "no ledger evidence (probe it, or promote --force)"
    return None


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------

def promote(card: dict, pack_tokens: Sequence[str] | None = None, require_evidence: bool = True) -> dict:
    """Pack prompt_identity fragment + outfit card snippets. Prints; never writes packs.

    Honours the ledger the way compile does: demoted or dropped elements stay out, a demoted
    one leaves a prompt_notes line. Outfits are exported only with ledger evidence (action
    keep) unless ``require_evidence`` is off (the ``--force`` escape hatch); rejected outfits
    are never exported. Everything left out is listed under ``skipped_outfits``.
    """
    ledger = (card.get("render_ledger") or {}).get("elements") or {}
    skipped = _demoted(card)  # demote | drop — exactly what compile leaves out
    for core in ("hair", "eyes"):
        if core in skipped:
            raise DesignError(f"{core} carries the colour lock and cannot be demoted or dropped; "
                              "rephrase it and probe again")
    notes = list(card.get("notes") or [])
    for key, entry in sorted(ledger.items()):
        if isinstance(entry, dict) and entry.get("action") == "demote":
            chips = ", ".join(f"{c}x{n}" for c, n in sorted((entry.get("reasons") or {}).items())) or "no chips"
            when = "close-ups only" if key == "skin_marks" else "unreliable"
            notes.append(f"{entry.get('phrase', key)}: {when} — demoted after {entry.get('revision', '?')} "
                         f"rewrites ({chips}; batch {entry.get('last_batch', '?')})")
    hair, eyes = card["hair"], card["eyes"]
    skin_marks = list(card.get("skin_marks") or []) if "skin_marks" not in skipped else []
    for mark in skin_marks:
        notes.append(f"{mark}: close-ups only (bust frame); not part of identity_tags")
    anchors = _dedup([
        " ".join(t for t in (hair.get("color"), hair.get("shape"), hair.get("detail")) if t),
        eyes["color"],
        card["signature_element"] if "signature_element" not in skipped else "",
        ", ".join(card["silhouette"]) if "silhouette" not in skipped else "",
        *skin_marks,
    ])
    fragment = {
        "id": card["id"],
        "identity_anchors": anchors,
        "prompt_identity": {
            "identity_tags": identity_tags(card, skip=sorted(skipped)),
            "color_lock": color_lock(card),
            "neg_guards": _guards(card),
            "register": {"description": card["register"]["description"],
                         "exclude": list(card["register"].get("excluded") or [])},
        },
        "prompt_notes": notes,
    }
    _check_pack_fragment(fragment)
    own = fragment["prompt_identity"]["identity_tags"] + fragment["prompt_identity"]["color_lock"]
    packs = list(pack_tokens) if pack_tokens is not None else pack_token_table()
    accent = card["palette"]["accent"].strip().lower()
    outfit_cards: dict[str, dict] = {}
    skipped_outfits: dict[str, str] = {}
    for outfit in card["outfits"]:
        oid = outfit["id"]
        key = f"outfit:{oid}"
        if outfit.get("status") == "rejected":
            skipped_outfits[oid] = "rejected on the card; never exported"
            continue
        if key in skipped:
            skipped_outfits[oid] = f"ledger action {ledger[key].get('action')}"
            continue
        entry = ledger.get(key) if isinstance(ledger.get(key), dict) else None
        stale = outfit_evidence_skip(card, outfit, entry) if require_evidence else None
        if stale:
            skipped_outfits[oid] = stale
            continue
        kept = entry is not None and entry.get("action") == "keep"
        tags = list(outfit["tags"])
        joined = " ".join(tags)
        colors = [c for c in _color_words(joined, COLOR_WORDS) if c not in NEUTRAL_COLORS]
        snippet = {
            "name_cn": outfit["name_cn"],
            "intent": outfit.get("intent") or f"{outfit['name_cn']}：{', '.join(tags[:3])}",
            "tags": tags,
            "neg_add": outfit.get("neg_add", ""),
            "sfw": bool(outfit["sfw"]),
            "coverage": outfit["coverage"],
            "motion_allow": list(outfit["motion_allow"]),
            "palette_strong": _mentions(joined, accent) or len(colors) >= 2,
            "source": f"{CARD_HOME.as_posix()}/{card['id']}.json",
            # verified is user-only on the deck: it survives only when the user set it on the
            # card AND the ledger kept the outfit; everything else lands as unvalidated.
            "status": "verified" if (outfit.get("status") == "verified" and kept) else "unvalidated",
        }
        face = " | ".join([snippet["name_cn"], snippet["intent"], joined, snippet["neg_add"]])
        for token in _dedup(own + packs):
            if _mentions(face, token):
                raise DesignError(f"outfit {oid!r} carries identity token {token!r}; "
                                  "it would fail scripts/test_card_purity.py after pasting")
        for term in BODY_LANGUAGE:
            if _mentions(face, term):
                raise DesignError(f"outfit {oid!r} carries body language {term!r}; "
                                  "it would fail scripts/test_outfit_cards.py after pasting")
        outfit_cards[oid] = snippet
    return {
        "schema": PROMOTE_SCHEMA,
        "design": card["id"],
        "status": card["status"],
        "pack_fragment": fragment,
        "outfit_cards": outfit_cards,
        "skipped_outfits": skipped_outfits,
        "paste": {
            "pack": f"characters/{card['id']}.pack.json — merge identity_anchors, prompt_identity and "
                    "prompt_notes; trigger, image_recipe and anchor_image stay yours",
            "outfits": "scripts/outfit_cards.json cards{} — then run scripts/test_outfit_cards.py "
                       "and scripts/test_card_purity.py",
        },
    }


def _check_pack_fragment(fragment: dict) -> None:
    """Hand check against character_pack.schema.json (jsonschema is a dev-only dependency)."""
    schema = json.loads(PACK_SCHEMA.read_text(encoding="utf-8"))
    identity_schema = schema["properties"]["prompt_identity"]
    identity = fragment["prompt_identity"]
    missing = [k for k in identity_schema["required"] if k not in identity]
    extra = sorted(set(identity) - set(identity_schema["properties"]))
    if missing or extra:
        raise DesignError(f"prompt_identity fragment mismatch: missing {missing}, extra {extra}")
    for key in ("identity_tags", "color_lock", "neg_guards"):
        if not isinstance(identity[key], list) or not all(isinstance(t, str) for t in identity[key]):
            raise DesignError(f"prompt_identity.{key} must be a list of strings")
    register = identity["register"]
    reg_schema = identity_schema["properties"]["register"]
    if sorted(register) != sorted(reg_schema["properties"]) or not isinstance(register["description"], str) \
            or not isinstance(register["exclude"], list):
        raise DesignError("prompt_identity.register must be {description, exclude}")
    if not identity["identity_tags"] or not identity["color_lock"]:
        raise DesignError("identity_tags and color_lock must not be empty")
    if not isinstance(fragment["identity_anchors"], list) or not fragment["identity_anchors"]:
        raise DesignError("identity_anchors must be a non-empty list")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_findings(findings: Sequence[Finding], as_json: bool) -> None:
    if as_json:
        print(json.dumps([f._asdict() for f in findings], ensure_ascii=False, indent=2))
        return
    for f in findings:
        print(f"{f.level.upper():<7} {f.render()}")


def _cmd_fields(_args: argparse.Namespace) -> int:
    print(f"schema        : {SCHEMA}")
    print(f"required      : {', '.join(REQUIRED_KEYS)}")
    print(f"optional      : {', '.join(OPTIONAL_KEYS)}")
    print(f"subject       : {', '.join(SUBJECTS)}")
    print(f"target_lines  : {', '.join(CARD_LINES)}   (compile also: prose = NL only)")
    print(f"frames        : {', '.join(FRAMES)}")
    print(f"element sets  : {' -> '.join(SET_NAMES)}   (id = silhouette+hair+eyes+face+expression)")
    print(f"status        : {' -> '.join(STATUSES[:3])} | rejected")
    print(f"ledger action : {', '.join(ACTIONS)}")
    print(f"outfit        : {', '.join(OUTFIT_REQUIRED)}; coverage {OUTFIT_COVERAGES}; "
          f"motion {MOTION_CLASSES}; status {OUTFIT_STATUSES}")
    print(f"accessory     : phrase, scene_class {SCENE_CLASSES}, placement")
    print(f"design notes  : concept <= {CONCEPT_MAX} chars; impression_keywords 1-{IMPRESSION_MAX}; "
          f"anchor {{{', '.join(ANCHOR_REQUIRED)} | {', '.join(ANCHOR_OPTIONAL)}}} (none compile, none hashed)")
    print(f"budgets       : intent <= {INTENT_MAX} chars; register.description <= {REGISTER_MAX_WORDS} words; "
          f"anima <= {ANIMA_MAX} words; nai character <= {NAI_CHARACTER_MAX} (warn); negative <= "
          f"{NEGATIVE_MAX_WORDS} words on anima")
    print()
    print("narrative words (refused)  :", ", ".join(NARRATIVE_LEXICON))
    print("style words (refused)      :", ", ".join(STYLE_WORDS))
    print("quality words (refused)    :", ", ".join(QUALITY_WORDS))
    print("mechanism words (refused)  :", ", ".join(MECHANISM_WORDS))
    print("body language (outfit face):", ", ".join(BODY_LANGUAGE))
    print("visible anchors (register) :", ", ".join(VISIBLE_ANCHORS))
    print("hair colours               :", ", ".join(HAIR_COLORS))
    print("eye colours                :", ", ".join(EYE_COLORS))
    print("material words (outfits)   :", ", ".join(MATERIAL_WORDS))
    print("whole objects (holding)    :", ", ".join(WHOLE_OBJECTS))
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    if not ID_RE.match(args.id):
        raise DesignError(f"id must match {ID_RE.pattern}")
    lines = [s.strip() for s in args.lines.split(",") if s.strip()]
    if not set(lines) <= set(CARD_LINES):
        raise DesignError(f"--lines must be a subset of {', '.join(CARD_LINES)}")
    if args.subject not in SUBJECTS:
        raise DesignError(f"--subject must be one of {', '.join(SUBJECTS)}")
    out = Path(args.out) if args.out else ROOT / CARD_HOME / f"{args.id}.json"
    if out.exists():
        raise DesignError(f"{out} exists; refusing to overwrite a card")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_card(skeleton(args.id, args.subject, lines)), encoding="utf-8")
    print(f"wrote skeleton {out}\nfill every TODO, then: python3 scripts/character_design.py lint {out}")
    return 0


def _load_linted(path: Path, as_json: bool = False) -> dict:
    card = load_card(path)
    findings = lint_card(card)
    errors = [f for f in findings if f.level == "error"]
    if errors:
        _print_findings(findings, as_json)
        raise SystemExit(1)
    return card


def _cmd_lint(args: argparse.Namespace) -> int:
    card = load_card(Path(args.card))
    findings = lint_card(card)
    _print_findings(findings, args.json)
    errors = sum(f.level == "error" for f in findings)
    if not args.json:
        print(f"{'FAIL' if errors else 'ok'}: {errors} error(s), {len(findings) - errors} warning(s)")
    return 1 if errors else 0


def _cmd_compile(args: argparse.Namespace) -> int:
    card = _load_linted(Path(args.card))
    result = compile_card(card, args.line, args.surface, frame=args.frame, set_name=args.set,
                          outfit_id=args.outfit, expression_index=args.expression,
                          scene_class=args.scene_class)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(f"# {result['design']} · {result['line']} · {result['frame']} · {result['set']} · "
          f"{result['word_count']} words")
    if result["tag"] is not None:
        print(result["tag"])
    if result["nl"] is not None:
        print(("\n" if result["tag"] else "") + result["nl"])
    if result["negative"]:
        print(f"negative: {result['negative']}")
    if result["qc_cull"]:
        print(f"qc_cull: {', '.join(result['qc_cull'])}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    card = _load_linted(Path(args.card))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    if not seeds or not set(frames) <= set(FRAMES) or not frames:
        raise DesignError(f"--seeds needs integers and --frames a subset of {FRAMES}")
    requested = [s.strip() for s in args.sets.split(",") if s.strip()] if args.sets else None
    sets = probe_sets(card, requested, args.outfit, args.scene_class)
    index = write_probe(card, args.line, Path(args.out), seeds, sets, frames, args.scene_class)
    print(f"probe {index['design']} · {index['line']} · {len(index['items'])} renders "
          f"({len(sets)} sets x {len(frames)} frames x {len(seeds)} seeds) -> {args.out}")
    for set_name in sets:
        for frame in frames:
            delta = next(i["delta"] for i in index["items"] if i["set"] == set_name and i["frame"] == frame)
            print(f"  {set_name:<20} {frame:<5} +{', '.join(delta)}")
    print("\nrun (nothing was rendered; the commands below need the box online):")
    for cmd in index["commands"]:
        print(f"  {cmd}")
    print("\nthe briefs are Studio-compatible job briefs: submit them to the console queue instead "
          "when the console record is required; the commands above are the design-sheet loop")
    print(f"\nthen: python3 scripts/character_design.py spec {args.out} --out {Path(args.out) / 'spec.json'}")
    return 0


def _cmd_spec(args: argparse.Namespace) -> int:
    spec = build_spec(Path(args.dir))
    if not spec:
        raise DesignError("no rendered files found under results/; run commands.sh first")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(spec)} items -> {out}")
    print(f"python3 scripts/review_batch.py --mode review --spec {shlex.quote(str(out))} --json")
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    path = Path(args.card)
    card = load_card(path)
    if args.result_json:
        result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    else:
        from review_result import read_result  # lazy: needs the gallery tree
        result = read_result(args.batch)
    overrides: dict[str, str] = {}
    for item in args.set or []:
        key, _, action = item.partition("=")
        if action not in ACTIONS or not key:
            raise DesignError(f"--set expects element=action with action in {ACTIONS}: {item!r}")
        overrides[key] = action
    rows = join_probe_verdicts(card, result, allow_stale=args.allow_stale)
    stale = 0 if args.allow_stale else stale_probe_rows(card, result)
    if stale:
        print(f"skipped {stale} stale row(s) rendered from another revision of this card "
              f"(current card {card_sha8(card)}); pass --allow-stale to count them anyway", file=sys.stderr)
    if not rows and not overrides:
        raise DesignError(f"no verdicts for design {card['id']!r} at card revision {card_sha8(card)} "
                          "in that batch")
    recovered = {row.get("scene_class") or "default" for row in rows}
    if args.scene_class is None:
        if len(recovered) > 1:
            raise DesignError(f"probe rows mix scene classes {sorted(recovered)}; pass --scene-class")
        scene_class = recovered.pop() if recovered else "default"
    elif recovered and recovered != {args.scene_class}:
        raise DesignError(f"--scene-class {args.scene_class} does not match the probe scene "
                          f"{sorted(recovered)}")
    else:
        scene_class = args.scene_class
    entries = element_actions(card, rows, overrides, scene_class=scene_class)
    if args.json:
        print(json.dumps({"rows": rows, "elements": entries}, ensure_ascii=False, indent=2))
    else:
        print(ledger_table(card, entries))
    if args.write:
        updated = merge_ledger(card, entries, rows)
        path.write_text(dump_card(updated), encoding="utf-8")
        print(f"\nledger written to {path} (status {updated['status']})")
    else:
        print("\n(dry run — add --write to store the ledger on the card)")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    card = _load_linted(Path(args.card))
    if card["status"] == "rejected":
        print("card is rejected: nothing to promote (and --force does not apply)", file=sys.stderr)
        return 1
    require_evidence = True
    if card["status"] == "draft":
        if not args.force:
            print("card is still draft: probe it first, or pass --force to promote unprobed", file=sys.stderr)
            return 1
        require_evidence = False  # the only case --force changes: an unprobed draft
    payload = promote(card, require_evidence=require_evidence)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"promote fragment -> {out}")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fields", help="print the card vocabulary and lexicons").set_defaults(func=_cmd_fields)

    p = sub.add_parser("new", help="write a skeleton card with TODO markers")
    p.add_argument("id")
    p.add_argument("--subject", default="female")
    p.add_argument("--lines", default="anima", help="comma list of target lines")
    p.add_argument("--out", default=None, help=f"default {CARD_HOME}/<id>.json")
    p.set_defaults(func=_cmd_new)

    p = sub.add_parser("lint", help="visual-language and per-line rules")
    p.add_argument("card")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_lint)

    p = sub.add_parser("compile", help="deterministic prompts for one line and surface")
    p.add_argument("card")
    p.add_argument("--line", required=True, choices=LINES)
    p.add_argument("--surface", default="tag", choices=SURFACES)
    p.add_argument("--frame", default="full", choices=FRAMES)
    p.add_argument("--set", default=None, help="id | sig | outfit | acc (default: the top set)")
    p.add_argument("--outfit", default=None)
    p.add_argument("--expression", type=int, default=0, help="index into register.allowed_expressions")
    p.add_argument("--scene-class", default="default", choices=SCENE_CLASSES)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_compile)

    p = sub.add_parser("probe", help="write design-sheet briefs and the commands that render them")
    p.add_argument("card")
    p.add_argument("--out", required=True, help="probe directory (must not hold a probe yet)")
    p.add_argument("--line", default="anima", choices=CARD_LINES, help="local model first: anima on BOXA")
    p.add_argument("--seeds", default="7,8")
    p.add_argument("--sets", default=None, help="comma subset of id,sig,outfit,acc (default: the ladder)")
    p.add_argument("--frames", default="full,bust")
    p.add_argument("--outfit", default=None)
    p.add_argument("--scene-class", default="default", choices=SCENE_CLASSES)
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("spec", help="review_batch spec from a rendered probe directory")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_spec)

    p = sub.add_parser("ledger", help="join review verdicts back onto the card's elements")
    p.add_argument("card")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", help="review batch ID (reads the gallery tree)")
    g.add_argument("--result-json", help="a review_result.py --json report file (offline)")
    p.add_argument("--write", action="store_true", help="store the ledger on the card")
    p.add_argument("--allow-stale", action="store_true",
                   help="also count rows rendered from an earlier revision of the card")
    p.add_argument("--set", action="append", metavar="ELEMENT=ACTION",
                   help="override an element's action (keep|rephrase|demote|drop)")
    p.add_argument("--scene-class", default=None, choices=SCENE_CLASSES,
                   help="must match the probe that produced the review; omitted, recovered from captions")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_ledger)

    p = sub.add_parser("promote", help="pack prompt_identity fragment + outfit card snippets")
    p.add_argument("card")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="promote a draft card")
    p.set_defaults(func=_cmd_promote)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DesignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
