"""Character design lint leaf: schema tuples, the face lexicons and the pure lint rules.

Extracted from ``scripts/character_design.py`` (Phase C, item C2) without a behaviour
change. This module imports only the standard library and knows nothing about the
repository layout: the one path it needs, the pack directory read by
``pack_token_table``, comes in as a parameter. ``scripts/character_design.py`` re-exports
every name defined here, so its importers keep reading them from ``character_design``;
the composed rule ``lint_card``, the budgets that need the compiler and the status gate
stay in the script until the compile block moves.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Sequence

SCHEMA = "loramake.character-design/v1"

CARD_LINES = ("anima", "illustrious", "nai")  # lines a card may target
SUBJECTS = {"female": ("1girl", "girl", "woman", "she"),
            "male": ("1boy", "boy", "man", "he"),
            "other": ("1other", "other", "person", "they")}
SCENE_CLASSES = ("default", "task", "home", "formal")
STATUSES = ("draft", "probed", "verified", "rejected")
# Same enums as scripts/outfit_cards.py; tests assert equality so they cannot drift.
OUTFIT_COVERAGES = ("full", "moderate", "minimal")
MOTION_CLASSES = ("static", "seated", "reclining", "dance", "dynamic")
OUTFIT_STATUSES = ("unvalidated", "verified", "rejected")
OUTFIT_REQUIRED = ("id", "name_cn", "tags", "sfw", "coverage", "motion_allow", "neg_add", "status")

REQUIRED_KEYS = ("schema", "id", "name_cn", "intent", "subject", "target_lines", "silhouette",
                 "face", "hair", "eyes", "signature_element", "palette", "register", "outfits",
                 "status")
OPTIONAL_KEYS = ("skin_marks", "accessories", "neg_guards", "forbidden", "notes",
                 "render_ledger", "allow_heterochromia",
                 # Standing-art set line (2026-09-05): a one-line concept and up to three
                 # impression keywords for the designer, and the anchor record written after
                 # the identity round. None of them compiles into a prompt.
                 "concept", "impression_keywords", "anchor")
CONCEPT_MAX = 80
IMPRESSION_MAX = 3
ANCHOR_REQUIRED = ("image",)
ANCHOR_OPTIONAL = ("line", "seed", "artist_card", "card_sha8", "batch", "ccip_mean", "notes")
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
INTENT_MAX = 40

# ---------------------------------------------------------------------------
# Lexicons read by the rules below. The other lexicons stay in scripts/character_design.py
# with lint_card; its `fields` command prints all of them.
# ---------------------------------------------------------------------------

# Body-shape language is pack-owned (scripts/test_outfit_cards.py BODY_LANGUAGE): outfit and
# accessory faces must never describe the figure. Copied verbatim; the test asserts parity.
BODY_LANGUAGE = (
    "slender",
    "voluptuous",
    "curvy",
    "hourglass",
    "plump",
    "chubby",
    "petite",
    "mature female",
    "mature woman",
    "athletic build",
    "muscular build",
)

# Concept round 6: hands holding a small part collapse; a whole object is fine.
WHOLE_OBJECTS = (
    "umbrella", "parasol", "bag", "satchel", "handbag", "backpack", "suitcase", "briefcase",
    "book", "notebook", "folder", "scroll", "letter", "tablet", "phone", "camera", "sword",
    "katana", "dagger", "staff", "spear", "bow", "shield", "rifle", "gun", "revolver",
    "lantern", "torch", "cup", "mug", "teacup", "bottle", "flask", "cane", "basket",
    "bouquet", "flower", "fan", "guitar", "violin", "instrument", "sign", "flag", "broom",
    "helmet", "hat", "mask", "case", "box", "jar", "plush", "doll", "cat", "kitten",
)
MATERIAL_WORDS = (
    "cotton", "linen", "wool", "silk", "satin", "velvet", "leather", "denim", "nylon", "lace",
    "chiffon", "tulle", "tweed", "knit", "ribbed", "matte", "glossy", "sheer", "mesh", "fur",
    "faux fur", "canvas", "corduroy", "suede", "pleated", "quilted", "brocade", "organza",
    "jersey", "fleece", "gauze", "metallic", "patent", "polyester", "rubber", "plastic",
    "polycarbonate", "chainmail", "embroidered", "sequined", "cashmere", "flannel",
)


class Finding(NamedTuple):
    level: str  # error | warning
    path: str
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.path}: {self.rule}: {self.detail}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len([w for w in re.split(r"[\s,]+", text) if w])


def _mentions(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text, re.IGNORECASE) is not None


def _color_words(text: str, palette: Sequence[str]) -> list[str]:
    found = [c for c in palette if _mentions(text, c)]
    # "ash blonde" also matches "blonde"; keep the longest, drop the words it contains
    keep = []
    for c in sorted(found, key=len, reverse=True):
        if not any(c != k and _mentions(k, c) for k in keep):
            keep.append(c)
    return keep


def _str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def _enum_ok(value: object, allowed: Sequence[str]) -> bool:
    """Membership test that a list or object value cannot crash (unhashable types)."""
    return isinstance(value, str) and value in allowed


def _strs(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _dedup(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        key = tok.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(tok.strip())
    return out


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def prompt_phrases(card: dict) -> Iterator[tuple[str, str]]:
    """Every prompt-facing string on the card with its JSON path (notes excluded)."""
    for i, s in enumerate(_strs(card.get("silhouette"))):
        yield f"silhouette[{i}]", s
    for i, s in enumerate(_strs(card.get("face"))):
        yield f"face[{i}]", s
    hair = card.get("hair") or {}
    for key in ("color", "length", "shape", "detail"):
        if isinstance(hair, dict) and isinstance(hair.get(key), str) and hair[key]:
            yield f"hair.{key}", hair[key]
    eyes = card.get("eyes") or {}
    for key in ("color", "shape"):
        if isinstance(eyes, dict) and isinstance(eyes.get(key), str) and eyes[key]:
            yield f"eyes.{key}", eyes[key]
    for i, s in enumerate(_strs(card.get("skin_marks"))):
        yield f"skin_marks[{i}]", s
    if isinstance(card.get("signature_element"), str) and card["signature_element"]:
        yield "signature_element", card["signature_element"]
    palette = card.get("palette") or {}
    if isinstance(palette, dict):
        for i, s in enumerate(_strs(palette.get("base"))):
            yield f"palette.base[{i}]", s
        if isinstance(palette.get("accent"), str) and palette["accent"]:
            yield "palette.accent", palette["accent"]
    register = card.get("register") or {}
    if isinstance(register, dict):
        if isinstance(register.get("description"), str) and register["description"]:
            yield "register.description", register["description"]
        for i, s in enumerate(_strs(register.get("allowed_expressions"))):
            yield f"register.allowed_expressions[{i}]", s
        for i, s in enumerate(_strs(register.get("excluded"))):
            yield f"register.excluded[{i}]", s
    for oi, outfit in enumerate(card.get("outfits") if isinstance(card.get("outfits"), list) else []):
        if not isinstance(outfit, dict):
            continue
        for ti, tag in enumerate(_strs(outfit.get("tags"))):
            yield f"outfits[{oi}].tags[{ti}]", tag
        if isinstance(outfit.get("neg_add"), str) and outfit["neg_add"]:
            yield f"outfits[{oi}].neg_add", outfit["neg_add"]
    for ai, acc in enumerate(card.get("accessories") if isinstance(card.get("accessories"), list) else []):
        if not isinstance(acc, dict):
            continue
        if isinstance(acc.get("phrase"), str) and acc["phrase"]:
            yield f"accessories[{ai}].phrase", acc["phrase"]
        if isinstance(acc.get("placement"), str) and acc["placement"]:
            yield f"accessories[{ai}].placement", acc["placement"]
    for i, s in enumerate(_strs(card.get("neg_guards"))):
        yield f"neg_guards[{i}]", s
    for i, s in enumerate(_strs(card.get("forbidden"))):
        yield f"forbidden[{i}]", s


def _lint_schema(card: dict, out: list[Finding]) -> bool:
    """Structural checks. Returns False when a missing key or a wrong type makes the semantic
    rules unsafe to run; value problems (unknown keys, lengths, enum values) are reported
    without stopping so a card gets its full finding list in one pass."""
    fatal: list[str] = []

    def err(path: str, detail: str) -> None:
        out.append(Finding("error", path, "schema", detail))

    def bad(path: str, detail: str) -> None:
        err(path, detail)
        fatal.append(path)

    def opt_str(obj: dict, key: str, path: str) -> None:
        if key in obj and not (isinstance(obj[key], str) and obj[key].strip()):
            bad(f"{path}.{key}", "non-empty string when present")

    if card.get("schema") != SCHEMA:
        bad("schema", f"must be {SCHEMA!r}")
    for key in sorted(set(card) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)):
        err(key, "unknown key (this card has no room for non-visual fields)")
    missing = [k for k in REQUIRED_KEYS if k not in card]
    for key in missing:
        bad(key, "missing required field")
    if missing:
        return False
    if not isinstance(card["id"], str) or not ID_RE.match(card["id"]):
        bad("id", f"must match {ID_RE.pattern}")
    for key in ("name_cn", "intent"):
        if not isinstance(card[key], str) or not card[key].strip():
            bad(key, "must be a non-empty string")
    if isinstance(card["intent"], str) and ("\n" in card["intent"] or len(card["intent"]) > INTENT_MAX):
        err("intent", f"one line, at most {INTENT_MAX} characters")
    _lint_design_notes(card, err, bad)
    if not _enum_ok(card["subject"], SUBJECTS):
        bad("subject", f"must be one of {', '.join(SUBJECTS)}")
    lines = card["target_lines"]
    if not _str_list(lines) or not lines or not set(lines) <= set(CARD_LINES):
        bad("target_lines", f"non-empty subset of {', '.join(CARD_LINES)}")
    for key, lo, hi in (("silhouette", 1, 4), ("face", 1, 4)):
        if not _str_list(card[key]):
            bad(key, "list of non-empty strings")
        elif not lo <= len(card[key]) <= hi:
            err(key, f"{lo}-{hi} phrases")
    hair = card["hair"]
    if not isinstance(hair, dict) or any(not isinstance(hair.get(k), str) or not hair[k].strip()
                                         for k in ("color", "length", "shape")):
        bad("hair", "object with color, length, shape (detail optional)")
    else:
        opt_str(hair, "detail", "hair")
        if set(hair) - {"color", "length", "shape", "detail"}:
            err("hair", "only color, length, shape, detail are allowed")
    eyes = card["eyes"]
    if not isinstance(eyes, dict) or not isinstance(eyes.get("color"), str) or not eyes["color"].strip():
        bad("eyes", "object with color (shape optional)")
    else:
        opt_str(eyes, "shape", "eyes")
        if set(eyes) - {"color", "shape"}:
            err("eyes", "only color and shape are allowed")
    marks = card.get("skin_marks", [])
    if not _str_list(marks):
        bad("skin_marks", "list of non-empty strings")
    elif len(marks) > 2:
        err("skin_marks", "0-2 phrases")
    if not isinstance(card["signature_element"], str):
        bad("signature_element", "one string")
    palette = card["palette"]
    if not isinstance(palette, dict) or not _str_list(palette.get("base")) \
            or not isinstance(palette.get("accent"), str):
        bad("palette", "object with base[] and accent")
    else:
        for key in sorted(set(palette) - {"base", "accent"}):
            err(f"palette.{key}", "unknown key (palette holds base[] and accent only)")
    register = card["register"]
    if not isinstance(register, dict) or not isinstance(register.get("description"), str) \
            or not _str_list(register.get("allowed_expressions")) \
            or not _str_list(register.get("excluded", [])):
        bad("register", "object with description, allowed_expressions[], excluded[]")
    else:
        for key in sorted(set(register) - {"description", "allowed_expressions", "excluded"}):
            err(f"register.{key}", "unknown key (register is visible demeanour only)")
    outfits = card["outfits"]
    if not isinstance(outfits, list) or not outfits:
        bad("outfits", "at least one outfit")
    else:
        for oi, outfit in enumerate(outfits):
            path = f"outfits[{oi}]"
            if not isinstance(outfit, dict):
                bad(path, "must be an object")
                continue
            missing_keys = [k for k in OUTFIT_REQUIRED if k not in outfit]
            if missing_keys:
                bad(path, f"missing {missing_keys}")
                continue
            for key in sorted(set(outfit) - set(OUTFIT_REQUIRED) - {"note", "intent"}):
                err(f"{path}.{key}", "unknown outfit key")
            for key in ("id", "name_cn", "coverage", "status"):
                if not isinstance(outfit[key], str) or not outfit[key].strip():
                    bad(f"{path}.{key}", "non-empty string")
            if not isinstance(outfit["neg_add"], str):
                bad(f"{path}.neg_add", "string")
            if not _str_list(outfit["tags"]):
                bad(f"{path}.tags", "list of non-empty strings")
            if not isinstance(outfit["sfw"], bool):
                bad(f"{path}.sfw", "boolean")
            if not _str_list(outfit["motion_allow"]):
                bad(f"{path}.motion_allow", "list of strings")
            opt_str(outfit, "intent", path)
            opt_str(outfit, "note", path)
    accessories = card.get("accessories", [])
    if not isinstance(accessories, list):
        bad("accessories", "list")
    else:
        for ai, acc in enumerate(accessories):
            path = f"accessories[{ai}]"
            if not isinstance(acc, dict) or not isinstance(acc.get("phrase"), str) or not acc["phrase"].strip():
                bad(path, "object with phrase (scene_class, placement optional)")
                continue
            for key in sorted(set(acc) - {"phrase", "scene_class", "placement"}):
                err(f"{path}.{key}", "unknown accessory key")
            opt_str(acc, "scene_class", path)
            opt_str(acc, "placement", path)
    for key in ("neg_guards", "forbidden", "notes"):
        if not _str_list(card.get(key, [])):
            bad(key, "list of strings")
    if not _enum_ok(card["status"], STATUSES):
        bad("status", f"must be one of {', '.join(STATUSES)}")
    ledger = card.get("render_ledger", {"elements": {}, "rows": []})
    if not isinstance(ledger, dict) or not isinstance(ledger.get("elements", {}), dict) \
            or not isinstance(ledger.get("rows", []), list):
        out.append(Finding("error", "render_ledger", "ledger_shape", "object with elements{} and rows[]"))
        fatal.append("render_ledger")
    if card.get("allow_heterochromia") not in (None, True, False):
        bad("allow_heterochromia", "boolean")
    for path, text in prompt_phrases(card):
        if "TODO" in text:
            err(path, "unfilled TODO from the skeleton")
    return not fatal


def _lint_design_notes(card: dict, err, bad) -> None:
    """The three non-compiling fields of the standing-art line: shape only, no lexicon rules
    (they are for the designer's eyes, never for the model)."""
    if "concept" in card:
        concept = card["concept"]
        if not isinstance(concept, str) or not concept.strip():
            bad("concept", "non-empty string when present")
        elif "\n" in concept or len(concept) > CONCEPT_MAX:
            err("concept", f"one line, at most {CONCEPT_MAX} characters")
    if "impression_keywords" in card:
        words = card["impression_keywords"]
        if not _str_list(words) or not words or len(words) > IMPRESSION_MAX:
            bad("impression_keywords", f"1 to {IMPRESSION_MAX} non-empty strings")
    if "anchor" in card:
        anchor = card["anchor"]
        if not isinstance(anchor, dict):
            bad("anchor", "object with image (path) and optional line/seed/artist_card/card_sha8/batch/ccip_mean")
            return
        for key in ANCHOR_REQUIRED:
            if not (isinstance(anchor.get(key), str) and anchor[key].strip()):
                bad(f"anchor.{key}", "non-empty string")
        for key in sorted(set(anchor) - set(ANCHOR_REQUIRED) - set(ANCHOR_OPTIONAL)):
            err(f"anchor.{key}", "unknown key")
        if "seed" in anchor and not isinstance(anchor["seed"], int):
            err("anchor.seed", "integer")
        if "ccip_mean" in anchor and not isinstance(anchor["ccip_mean"], (int, float)):
            err("anchor.ccip_mean", "number")
        for key in ("line", "artist_card", "card_sha8", "batch"):
            if key in anchor and not (isinstance(anchor[key], str) and anchor[key].strip()):
                err(f"anchor.{key}", "non-empty string when present")
        if "notes" in anchor and not _str_list(anchor["notes"]):
            err("anchor.notes", "list of strings")


def _lint_outfits(card: dict, own_tokens: list[str], pack_tokens: list[str], out: list[Finding]) -> None:
    """Value rules for outfits; shapes and types were settled by _lint_schema."""
    seen: set[str] = set()
    for oi, outfit in enumerate(card["outfits"]):
        path = f"outfits[{oi}]"
        oid = outfit["id"]
        if not ID_RE.match(oid):
            out.append(Finding("error", f"{path}.id", "outfit_shape", f"must match {ID_RE.pattern}"))
        elif oid in seen:
            out.append(Finding("error", f"{path}.id", "outfit_shape", f"duplicate outfit id {oid!r}"))
        seen.add(oid)
        tags = outfit["tags"]
        if not 3 <= len(tags) <= 10:
            out.append(Finding("error", f"{path}.tags", "outfit_shape", "3-10 phrases"))
        if not _enum_ok(outfit["coverage"], OUTFIT_COVERAGES):
            out.append(Finding("error", f"{path}.coverage", "outfit_shape", f"one of {OUTFIT_COVERAGES}"))
        motion = outfit["motion_allow"]
        if not motion or not set(motion) <= set(MOTION_CLASSES):
            out.append(Finding("error", f"{path}.motion_allow", "outfit_shape",
                               f"non-empty subset of {MOTION_CLASSES}"))
        if not _enum_ok(outfit["status"], OUTFIT_STATUSES):
            out.append(Finding("error", f"{path}.status", "outfit_shape", f"one of {OUTFIT_STATUSES}"))
        joined = " ".join(tags)
        if len([m for m in MATERIAL_WORDS if _mentions(joined, m)]) < 2:
            out.append(Finding("warning", f"{path}.tags", "outfit_materials",
                               "fewer than two material words (concept round 3 recipe)"))
        for field in ("name_cn", "tags", "neg_add", "intent", "note"):
            if outfit.get(field) is None:
                continue
            text = outfit[field] if isinstance(outfit[field], str) else " | ".join(outfit[field])
            for token in own_tokens:
                if _mentions(text, token):
                    out.append(Finding("error", f"{path}.{field}", "character_neutral",
                                       f"identity token {token!r} on an outfit face (identity is pack-owned)"))
            for token in pack_tokens:
                if _mentions(text, token):
                    out.append(Finding("error", f"{path}.{field}", "character_neutral",
                                       f"existing pack token {token!r} on an outfit face"))
            for term in BODY_LANGUAGE:
                if _mentions(text, term):
                    out.append(Finding("error", f"{path}.{field}", "character_neutral",
                                       f"body language {term!r} on an outfit face"))


def _lint_accessories(card: dict, own_tokens: list[str], pack_tokens: list[str], out: list[Finding]) -> None:
    """Value rules for accessories; shapes and types were settled by _lint_schema."""
    for ai, acc in enumerate(card.get("accessories") or []):
        path = f"accessories[{ai}]"
        if not _enum_ok(acc.get("scene_class", "default"), SCENE_CLASSES):
            out.append(Finding("error", f"{path}.scene_class", "schema", f"one of {SCENE_CLASSES}"))
        phrase = acc["phrase"]
        match = re.search(r"\bholding\b\s+(?:an?\s+|the\s+)?(.+)", phrase, re.IGNORECASE)
        if match and not any(_mentions(phrase, obj) for obj in WHOLE_OBJECTS):
            out.append(Finding("warning", f"{path}.phrase", "small_handheld",
                               "hands holding a small part collapse (concept round 6); name a whole object"))
        for token in own_tokens + pack_tokens:
            if _mentions(phrase, token):
                out.append(Finding("error", f"{path}.phrase", "character_neutral",
                                   f"identity token {token!r} on an accessory face"))
        for term in BODY_LANGUAGE:
            if _mentions(phrase, term):
                out.append(Finding("error", f"{path}.phrase", "character_neutral",
                                   f"body language {term!r} on an accessory face"))


def pack_token_table(characters_dir: Path) -> list[str]:
    """Identity tokens owned by the existing packs (identity_tags + color_lock).

    ``characters_dir`` holds the ``*.pack.json`` files; the facade in
    ``scripts/character_design.py`` supplies the repository default. Mirrors
    scripts/test_card_purity.py token_table(); the test asserts both agree.
    """
    tokens: set[str] = set()
    for pack_path in sorted(characters_dir.glob("*.pack.json")):
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        identity = pack.get("prompt_identity") or {}
        for key in ("identity_tags", "color_lock"):
            for token in identity.get(key) or []:
                token = str(token).strip()
                if token:
                    tokens.add(token)
    return sorted(tokens)


def identity_tags(card: dict, skip: Sequence[str] = ()) -> list[str]:
    """Stable identity tokens the card will own once promoted (hair, eyes, silhouette, face)."""
    hair = card.get("hair") or {}
    eyes = card.get("eyes") or {}
    tokens: list = []
    if "hair" not in skip:
        tokens += [hair.get(k) for k in ("color", "length", "shape", "detail")]
    if "eyes" not in skip:
        tokens += [eyes.get(k) for k in ("color", "shape")]
    if "silhouette" not in skip:
        tokens += _strs(card.get("silhouette"))
    if "face" not in skip:
        tokens += _strs(card.get("face"))
    return _dedup([t for t in tokens if isinstance(t, str) and t.strip()])


def color_lock(card: dict) -> list[str]:
    hair = card.get("hair") or {}
    eyes = card.get("eyes") or {}
    return _dedup([t for t in (hair.get("color"), eyes.get("color")) if isinstance(t, str) and t.strip()])
