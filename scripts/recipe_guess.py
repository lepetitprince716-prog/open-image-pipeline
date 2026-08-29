#!/usr/bin/env python3
"""Recipe inference from pixels — the ``inferred`` half of the forensics schema.

``image_recipe.build_forensics`` declares an ``inferred`` block (model family,
prompt slots, LoRA hints, control hints) that was hardcoded empty; this module
fills it from a WD14 tagging result. It is a leaf module by design: both
``image_recipe`` and ``image_repro_spec`` import it, so it must import neither
(they already import each other's machinery and a cycle would follow).

Pure functions only in the core: every classifier input (the WD14 dict, catalog
records) arrives as an argument, and no model is ever loaded here. The single IO
shell, ``lookup_character_loras``, wraps the lora_catalog lookup and is kept
separate so the pure core stays unit-testable without the catalog on disk.

Honesty contract: pixel-level base-model identification is not possible. The
family guess is a routing default with declared low confidence, plus an
out-of-scope verdict for photographic sources; nothing here pretends otherwise.
"""

from __future__ import annotations

from typing import Any

# Tags that signal a human figure (openpose locks the pose) vs. a scene — kept in
# sync with image_reproduce's routing set.
PERSON_COUNT_TAGS = frozenset({
    "1girl", "2girls", "3girls", "1boy", "2boys", "multiple_girls", "multiple_boys",
    "solo", "solo_focus", "couple",
})
# WD14 general tags that indicate a photographic (non-anime) source.
PHOTO_HINT_TAGS = frozenset({"photorealistic", "realistic", "photo_(medium)", "photo_background"})
# Style/meta general tags routed to the style slot of the prompt guess.
STYLE_TAGS = frozenset({
    "official_art", "anime_screencap", "game_cg", "sketch", "monochrome", "greyscale",
    "watercolor_(medium)", "oil_painting_(medium)", "pixel_art", "retro_artstyle",
    "flat_color", "lineart", "traditional_media", "chibi",
})

CHARACTER_THRESHOLD = 0.6   # WD14 character-bucket confidence floor
GENERAL_THRESHOLD = 0.35    # tag considered "present" (matches tag_prompt_candidate)
SPURIOUS_THRESHOLD = 0.5    # candidate tag confident enough to negate
SOURCE_ABSENT_BELOW = 0.2   # source score under this = truly absent from the source
MAX_ADD_PER_ROUND = 5
MAX_NEG_PER_ROUND = 3


def _spaced(tag: str) -> str:
    return tag.replace("_", " ")


def _norm(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def character_candidates(wd14_character: dict[str, float] | None,
                         threshold: float = CHARACTER_THRESHOLD) -> list[tuple[str, float]]:
    """Known-character identification for free: the WD14 character bucket above a
    confidence floor, best first."""
    if not wd14_character:
        return []
    ranked = sorted(wd14_character.items(), key=lambda kv: kv[1], reverse=True)
    return [(tag, round(float(score), 4)) for tag, score in ranked if score >= threshold]


def drop_overlapping(tags: list[str]) -> list[str]:
    """Drop a tag whose words are wholly contained in another chosen tag (a local,
    dependency-free stand-in for imgutils' drop_overlap_tags: 'long hair' goes when
    'very long hair' is present). Order is preserved."""
    norm = [_norm(t) for t in tags]
    keep: list[str] = []
    for i, tag in enumerate(tags):
        words = f" {norm[i]} "
        contained = any(i != j and words in f" {other} " for j, other in enumerate(norm))
        if not contained:
            keep.append(tag)
    return keep


def prompt_slots(wd14: dict[str, Any] | None,
                 threshold: float = GENERAL_THRESHOLD) -> list[dict[str, Any]]:
    """Ordered prompt-slot guess: character tags, then subject content, then
    style/meta — each slot a list of spaced tags, confidence order."""
    if not wd14:
        return []
    chars = [_spaced(t) for t, _ in character_candidates(wd14.get("character"))]
    general = wd14.get("general") or {}
    ranked = [t for t, s in sorted(general.items(), key=lambda kv: kv[1], reverse=True)
              if s >= threshold]
    style = [_spaced(t) for t in ranked if t in STYLE_TAGS]
    subject = drop_overlapping([_spaced(t) for t in ranked if t not in STYLE_TAGS])
    slots: list[dict[str, Any]] = []
    if chars:
        slots.append({"slot": "character", "tags": chars})
    if subject:
        slots.append({"slot": "subject", "tags": subject})
    if style:
        slots.append({"slot": "style", "tags": style})
    return slots


def model_family_guess(wd14: dict[str, Any] | None) -> dict[str, Any]:
    """Honest heuristic only. Photographic hints => out of scope; anything else is
    the routing default with declared low confidence — pixel-level base-model
    identification is not possible and this never claims it is."""
    general = (wd14 or {}).get("general") or {}
    photo = {t: s for t, s in general.items() if t in PHOTO_HINT_TAGS and s >= 0.5}
    if photo:
        return {"guess": None, "confidence": "low", "out_of_scope": True,
                "reason": f"photographic hints {sorted(photo)} — non-anime sources are out of scope"}
    return {"guess": "illustrious", "confidence": "low", "out_of_scope": False,
            "reason": "pixel-level base-model identification is not possible; "
                      "illustrious is the reproduction pipeline's routing default"}


def character_lora_candidates(char_tags: list[str], records: list[dict[str, Any]],
                              k: int = 3) -> list[dict[str, Any]]:
    """Lexical match of WD14 character tags against catalog records (file name and
    trigger words). Pure: records come from the caller. Scoring is containment of
    normalized character-name words; series qualifiers like '(blue archive)' are
    stripped for matching."""
    if not char_tags or not records:
        return []
    names = []
    for tag in char_tags:
        base = _norm(tag.split("(")[0]).strip()
        if base:
            names.append(base)
    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        fields = []
        for key in ("file", "name", "triggers"):
            value = rec.get(key)
            fields.append(" ".join(str(v) for v in value) if isinstance(value, (list, tuple))
                          else str(value or ""))
        haystack = _norm(" ".join(fields))
        hits = sum(1 for name in names if name and name in haystack)
        if hits:
            scored.append((hits / len(names), rec))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"file": rec.get("file"), "strength": rec.get("strength_default", 0.8),
             "base_model": rec.get("base_model"), "content_rating": rec.get("content_rating"),
             "score": round(score, 3), "source": "wd14-character-lexical"}
            for score, rec in scored[:k]]


def lookup_character_loras(char_tags: list[str], k: int = 3) -> list[dict[str, Any]]:
    """IO shell: lexical pass over the Z3 LoRA catalog. Degrades to [] when the
    catalog is missing — never raises (the reproduction loop treats hints as
    optional; note _load_index sys.exits on a missing index, hence BaseException)."""
    if not char_tags:
        return []
    try:
        import lora_catalog
        records = lora_catalog._load_index()
    except BaseException:  # noqa: BLE001 — includes the loader's SystemExit
        return []
    return character_lora_candidates(char_tags, records or [], k=k)


def control_hints(wd14: dict[str, Any] | None) -> list[str]:
    general = (wd14 or {}).get("general") or {}
    return ["openpose"] if any(t in general for t in PERSON_COUNT_TAGS) else ["canny"]


def build_inferred(wd14: dict[str, Any] | None,
                   lora_hints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the full ``inferred`` block for the forensics/spec schemas."""
    chars = character_candidates((wd14 or {}).get("character"))
    return {
        "model_family_guess": model_family_guess(wd14),
        "prompt_slot_guess": prompt_slots(wd14),
        "character_candidates": [{"tag": t, "score": s} for t, s in chars],
        "lora_hints": lora_hints if lora_hints is not None else [],
        "control_hints": control_hints(wd14),
        "confidence": "low",
    }


# --------------------------------------------------------------------------- #
# Per-round prompt refinement (the tag-diff feedback lever)
# --------------------------------------------------------------------------- #
def protected_tags(trigger_text: str, pos_prefix: str,
                   source_general: dict[str, float] | None) -> frozenset[str]:
    """Tags the refinement must never negate or remove: LoRA trigger words, the
    family positive prefix, and person-count tags the source actually shows."""
    protected = set()
    for blob in (trigger_text, pos_prefix):
        for part in (blob or "").split(","):
            part = _norm(part).strip()
            if part:
                protected.add(part)
    for tag in source_general or {}:
        if tag in PERSON_COUNT_TAGS:
            protected.add(_norm(tag))
    return frozenset(protected)


def tag_diff(source_general: dict[str, float] | None,
             candidate_general: dict[str, float] | None,
             protected: frozenset[str] = frozenset(),
             max_add: int = MAX_ADD_PER_ROUND,
             max_neg: int = MAX_NEG_PER_ROUND) -> dict[str, list[str]]:
    """Source-vs-candidate tag diff, capped per round.

    missing  = source tags >= GENERAL_THRESHOLD the candidate lacks -> positive adds
    spurious = candidate tags >= SPURIOUS_THRESHOLD the source scores < SOURCE_ABSENT_BELOW
               -> negative adds
    """
    src = source_general or {}
    cand = candidate_general or {}
    missing = [t for t, s in sorted(src.items(), key=lambda kv: kv[1], reverse=True)
               if s >= GENERAL_THRESHOLD
               and cand.get(t, 0.0) < GENERAL_THRESHOLD
               and _norm(t) not in protected]
    spurious = [t for t, s in sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
                if s >= SPURIOUS_THRESHOLD
                and src.get(t, 0.0) < SOURCE_ABSENT_BELOW
                and _norm(t) not in protected]
    return {"missing": [_spaced(t) for t in drop_overlapping(missing)[:max_add]],
            "spurious": [_spaced(t) for t in drop_overlapping(spurious)[:max_neg]]}


def refine_prompt(pos: str, neg: str, diff: dict[str, list[str]]) -> tuple[str, str, list[str]]:
    """Apply a tag diff to the prompt pair, skipping anything already present.
    Returns (pos, neg, notes); empty notes = no change."""
    notes: list[str] = []
    pos_norm = _norm(pos)
    adds = [t for t in diff.get("missing", []) if _norm(t) not in pos_norm]
    if adds:
        pos = pos.rstrip().rstrip(",") + ", " + ", ".join(adds)
        notes.append(f"+pos: {', '.join(adds)}")
    neg_norm = _norm(neg)
    negs = [t for t in diff.get("spurious", []) if _norm(t) not in neg_norm]
    if negs:
        neg = neg.rstrip().rstrip(",") + ", " + ", ".join(negs)
        notes.append(f"+neg: {', '.join(negs)}")
    return pos, neg, notes
