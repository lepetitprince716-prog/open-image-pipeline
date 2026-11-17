"""Cross-line style registry: cards under ``catalogs/style`` resolved into render fragments.

A style card names one visual style once and lists how each line realizes it: a NovelAI
artist card, a local LoRA from the catalog, a t2i catalog item, a bare prompt block, a
Civitai AIR, or a Krea 2 caption. The registry only references those assets; it never
copies them. ``resolve`` turns ``(style_id, line)`` into a ``RenderFragment`` that a caller
folds into its own prompt and LoRA arguments; ``matrix`` reports coverage per line;
``lint`` checks every reference against the asset it points at.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loramake.config import ROOT

SCHEMA_ID = "loramake.style-card/v1"
CARDS_DIR = ROOT / "catalogs" / "style"
SCHEMA_PATH = ROOT / "scripts" / "schema" / "style_card.schema.json"
NAI_LIBRARY = ROOT / "scripts" / "nai_artist_strings.json"
LORA_INDEX = ROOT / "lora_catalog" / "boxa_index.json"
T2I_STYLE_DIR = ROOT / "catalogs" / "t2i" / "core" / "style.prompt"

LINES = ("illustrious", "anima", "nai", "krea2", "civitai")
TIERS = ("verified", "candidate", "blocked")
KINDS = ("nai_card", "lora", "t2i_item", "prompt", "civitai_air", "caption", "lora_stack")
FAMILIES = ("anime", "media", "photo", "digital", "theme", "house")
PRIORITIES = ("P1", "P2", "P3", "house")
NAI_DEFAULT_MODEL = "nai-diffusion-5-full"
STATUS_GLYPH = {"verified": "✅", "candidate": "◐", "blocked": "⛔", "gap": "✗", "na": "—"}

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_AIR_RE = re.compile(r"^urn:air:[a-z0-9]+:lora:civitai:(\d+)@(\d+)$")
_TIER_RANK = {tier: i for i, tier in enumerate(TIERS)}


class StyleRegistryError(ValueError):
    """A card, a reference or a request is malformed."""


class StyleUnavailable(StyleRegistryError):
    """The style has no usable realization on the requested line."""


# --------------------------------------------------------------------------- loading


def load_cards(cards_dir: Path | str | None = None) -> dict[str, dict]:
    """Read every ``*.json`` card, keyed by id, sorted by id.

    Raises ``StyleRegistryError`` on unreadable JSON, a duplicate id, or an id that does
    not match its file name; schema and reference problems are reported by ``lint``.
    """
    directory = Path(cards_dir) if cards_dir else CARDS_DIR
    cards: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "registers.json":  # aesthetic registers live beside the cards
            continue
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StyleRegistryError(f"{path.name}: invalid JSON ({exc})") from exc
        if not isinstance(card, dict):
            raise StyleRegistryError(f"{path.name}: card must be a JSON object")
        card_id = card.get("id")
        if card_id != path.stem:
            raise StyleRegistryError(f"{path.name}: id {card_id!r} does not match file name")
        if card_id in cards:
            raise StyleRegistryError(f"{path.name}: duplicate id {card_id!r}")
        cards[card_id] = card
    return dict(sorted(cards.items()))


def load_schema(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or SCHEMA_PATH).read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nai_library(path: Path | str | None = None) -> dict[str, dict]:
    data = _load_json(Path(path or NAI_LIBRARY))
    return data.get("cards", {}) if isinstance(data, dict) else {}


def lora_index(path: Path | str | None = None) -> dict[str, dict]:
    data = _load_json(Path(path or LORA_INDEX))
    records = data if isinstance(data, list) else data.get("items") or data.get("loras") or []
    return {str(rec.get("id")): rec for rec in records if isinstance(rec, dict) and rec.get("id")}


def t2i_item(ref: str, directory: Path | str | None = None) -> dict | None:
    path = Path(directory or T2I_STYLE_DIR) / f"{ref}.json"
    if not path.is_file():
        return None
    item = _load_json(path)
    return item if isinstance(item, dict) else None


# --------------------------------------------------------------------------- validation


def schema_problems(card: dict, schema: dict | None = None) -> list[str]:
    """Validate one card against the JSON schema; fall back to structural checks."""
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover - dev extra provides it
        return _basic_problems(card)
    validator = jsonschema.Draft202012Validator(schema or load_schema())
    problems = []
    for error in sorted(validator.iter_errors(card), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in error.absolute_path) or "<root>"
        problems.append(f"{where}: {error.message}")
    return problems


def _basic_problems(card: dict) -> list[str]:
    problems = []
    if card.get("schema") != SCHEMA_ID:
        problems.append(f"schema must be {SCHEMA_ID}")
    if not _ID_RE.match(str(card.get("id", ""))):
        problems.append("id must match ^[a-z][a-z0-9_]*$")
    if card.get("family") not in FAMILIES:
        problems.append("family out of range")
    if card.get("priority") not in PRIORITIES:
        problems.append("priority out of range")
    if not card.get("evidence"):
        problems.append("evidence must not be empty")
    for line, reals in (card.get("realizations") or {}).items():
        if line not in LINES:
            problems.append(f"unknown line {line!r}")
        for real in reals or []:
            if real.get("kind") not in KINDS or real.get("tier") not in TIERS:
                problems.append(f"{line}/{real.get('id')}: bad kind or tier")
    return problems


def compose_nai_block(components: list[dict]) -> str:
    """Render NovelAI card components the way the artist library does.

    ``W::artist:x ::`` for weighted components, bare tag otherwise, in card order.
    """
    parts = []
    for comp in components:
        tag = f"artist:{comp['tag']}" if comp.get("kind") == "artist" else comp["tag"]
        weight = comp.get("weight")
        parts.append(tag if weight is None else f"{weight:g}::{tag} ::")
    return ", ".join(parts)


def lint(cards: dict[str, dict], *, schema: dict | None = None, library: dict | None = None,
         loras: dict | None = None, t2i_dir: Path | str | None = None) -> list[str]:
    """Return every problem found across the registry; an empty list means clean."""
    schema = schema or load_schema()
    library = library if library is not None else nai_library()
    loras = loras if loras is not None else lora_index()
    problems: list[str] = []
    for card_id, card in cards.items():
        for problem in schema_problems(card, schema):
            problems.append(f"{card_id}: {problem}")
        lines = set(card.get("lines") or [])
        realized: set[str] = set()
        seen_ids: set[str] = set()
        for line, reals in (card.get("realizations") or {}).items():
            if line not in lines:
                problems.append(f"{card_id}: realizations for {line} but line not targeted")
            for real in reals or []:
                rid = str(real.get("id"))
                if rid in seen_ids:
                    problems.append(f"{card_id}: duplicate realization id {rid}")
                seen_ids.add(rid)
                if real.get("tier") != "blocked":
                    realized.add(line)
                problems.extend(f"{card_id}/{line}/{rid}: {p}"
                                for p in _reference_problems(line, real, library, loras, t2i_dir))
        for gap in card.get("gaps") or []:
            if gap.get("line") not in lines:
                problems.append(f"{card_id}: gap for {gap.get('line')} but line not targeted")
            elif gap.get("line") in realized:
                problems.append(f"{card_id}: gap listed for {gap['line']} which has a realization")
        gap_lines = {g.get("line") for g in card.get("gaps") or []}
        for line in sorted(lines - realized - gap_lines):
            problems.append(f"{card_id}: line {line} has neither a realization nor a gap entry")
    return problems


def _reference_problems(line: str, real: dict, library: dict, loras: dict,
                        t2i_dir: Path | str | None) -> list[str]:
    kind = real.get("kind")
    ref = real.get("ref")
    problems: list[str] = []
    if kind == "nai_card":
        if line != "nai":
            problems.append("nai_card realization outside the nai line")
        card = library.get(ref)
        if card is None:
            problems.append(f"unknown NovelAI card {ref!r}")
        else:
            if card.get("status") == "rejected" and real.get("tier") != "blocked":
                problems.append(f"NovelAI card {ref!r} is rejected in the library")
            has_probe = bool((real.get("evidence") or {}).get("ref"))
            if real.get("tier") == "verified" and card.get("status") != "verified" and not has_probe:
                problems.append(f"tier verified but library card {ref!r} is {card.get('status')} "
                                "and no probe evidence is recorded")
    elif kind == "lora":
        rec = loras.get(ref)
        if rec is None:
            problems.append(f"unknown LoRA catalog id {ref!r}")
        else:
            if rec.get("content_rating") == "nsfw":
                problems.append(f"LoRA {ref!r} is rated nsfw")
            base = str(rec.get("base_model") or "").lower()
            if line == "anima" and "anima" not in base:
                problems.append(f"LoRA {ref!r} base {rec.get('base_model')!r} on the anima line")
            if line == "illustrious" and "anima" in base:
                problems.append(f"LoRA {ref!r} is an Anima LoRA on the illustrious line")
    elif kind == "t2i_item":
        item = t2i_item(ref, t2i_dir)
        if item is None:
            problems.append(f"unknown t2i item {ref!r}")
        elif line not in (item.get("dialects") or {}):
            problems.append(f"t2i item {ref!r} has no {line} dialect")
    elif kind == "civitai_air":
        match = _AIR_RE.match(str(real.get("air", "")))
        if not match:
            problems.append("air does not look like a Civitai LoRA AIR")
        elif (int(match.group(1)), int(match.group(2))) != (real.get("model_id"), real.get("version_id")):
            problems.append("air ids differ from model_id/version_id")
        if line != "civitai":
            problems.append("civitai_air realization outside the civitai line")
    elif kind == "caption" and line != "krea2":
        problems.append("caption realization outside the krea2 line")
    return problems


# --------------------------------------------------------------------------- resolving


@dataclass
class RenderFragment:
    style_id: str
    line: str
    realization_id: str
    kind: str
    tier: str
    pos_prefix: str = ""
    pos_suffix: str = ""
    neg_add: str = ""
    loras: list[dict] = field(default_factory=list)
    lora_specs: list[str] = field(default_factory=list)
    base_pin: str | None = None
    sampler: dict = field(default_factory=dict)
    post_fx: str | None = None
    caption: str = ""
    model: str | None = None
    air: str | None = None
    base_model: str | None = None
    trigger: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Artist LoRA import (2026-09-05): one candidate card per SFW artist LoRA on the BOXA box.
# ---------------------------------------------------------------------------

ARTIST_GAP_TABLE = ROOT / "outputs" / "_analysis" / "artist_gap_2026-09-05.json"
_LORA_LINE_BY_BASE = {"illustrious": "illustrious", "noobai": "illustrious", "anima": "anima"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def artist_slug(row: dict) -> str:
    """Card id stem: the Danbooru tag when known, else the LoRA handle (ascii, lowercase)."""
    stem = str(row.get("danbooru_tag") or row.get("handle") or row.get("lora_name") or "").lower()
    for suffix in ("_anima", "_illustrious", "_il", "_style"):
        if stem.endswith(suffix) and not row.get("danbooru_tag"):
            stem = stem[: -len(suffix)]
    stem = _SLUG_RE.sub("_", stem).strip("_")
    if len(stem) < 3 or stem in ("anima", "illustrious", "il", "style", "styles", "lora"):
        # CJK-only handles collapse to nothing useful; fall back to the catalog id.
        lora_id = str(row.get("lora_id") or "")[:8]
        return f"artist_lora_{lora_id}" if lora_id else ""
    return f"artist_{stem}"


def _line_for(rec: dict) -> str | None:
    base = str(rec.get("base_model") or "").lower()
    for key, line in _LORA_LINE_BY_BASE.items():
        if key in base:
            return line
    return None


def referenced_loras(cards: dict[str, dict]) -> set[str]:
    refs: set[str] = set()
    for card in cards.values():
        for reals in (card.get("realizations") or {}).values():
            for real in reals or []:
                if real.get("kind") == "lora" and real.get("ref"):
                    refs.add(str(real["ref"]))
    return refs


def artist_lora_cards(rows: list[dict], loras: dict[str, dict], existing: dict[str, dict],
                      date: str, min_posts: int = 0) -> tuple[dict[str, dict], dict[str, int]]:
    """Build candidate cards for SFW artist LoRAs that no card references yet.

    Returns (cards by id, skip counts). NSFW-rated LoRAs, bases outside the Illustrious /
    NoobAI / Anima families and LoRAs already referenced by a card are skipped, never rewritten.
    """
    out: dict[str, dict] = {}
    skipped = {"nsfw": 0, "base": 0, "referenced": 0, "unknown": 0, "duplicate_id": 0, "posts": 0}
    taken = set(existing) | set(out)
    known_refs = referenced_loras(existing)
    for row in rows:
        lora_id = str(row.get("lora_id") or "")
        rec = loras.get(lora_id)
        if rec is None:
            skipped["unknown"] += 1
            continue
        if rec.get("content_rating") != "sfw":
            skipped["nsfw"] += 1
            continue
        line = _line_for(rec)
        if line is None:
            skipped["base"] += 1
            continue
        if lora_id in known_refs:
            skipped["referenced"] += 1
            continue
        posts = int(row.get("danbooru_posts") or 0)
        if row.get("danbooru_tag") and posts < min_posts:
            skipped["posts"] += 1
            continue
        card_id = artist_slug(row)
        if not card_id or not _ID_RE.match(card_id):
            skipped["unknown"] += 1
            continue
        if card_id in taken:
            # Same slug, different LoRA (e.g. an IL and an Anima build of one artist): keep both.
            card_id = f"{card_id}_{lora_id[:4]}"
            skipped["duplicate_id"] += 1
            if card_id in taken:
                continue
        taken.add(card_id)
        handle = str(row.get("handle") or row.get("lora_name") or lora_id)
        tag = row.get("danbooru_tag")
        provenance = (f"Danbooru {tag}（{posts} 帖）" if tag else "未对上 Danbooru（Twitter / Pixiv 圈画师）")
        triggers = [t for t in (rec.get("triggers") or []) if isinstance(t, str) and t.strip()]
        out[card_id] = {
            "schema": SCHEMA_ID,
            "id": card_id,
            "revision": 1,
            "family": "anime",
            "display_name": f"画师 {tag or handle}",
            "description": f"BOXA 本地画师 LoRA（{rec.get('base_model')}），{provenance}。",
            "priority": "P1" if posts >= 300 else "P2",
            "evidence": [
                {"source": "lora_catalog/boxa_index.json", "date": date, "ref": str(rec.get("file") or lora_id)},
                {"source": "artist_gap_2026-09-05", "date": date, "ref": provenance},
            ],
            "exemplars": [],
            "failure_modes": [],
            "lines": [line],
            "realizations": {line: [{
                "id": f"{line}:lora:{lora_id}",
                "kind": "lora",
                "tier": "candidate",
                "ref": lora_id,
                "weight": float(rec.get("strength_default") or 0.8),
                "prompt": ", ".join(triggers),
                "note": f"BOXA 画师 LoRA {rec.get('name')}; {date} 自动登记，待 CSD 配对与探针",
            }]},
            "gaps": [],
        }
    return out, skipped


def pick_realization(card: dict, line: str, prefer: str | None = None) -> dict:
    reals = [r for r in (card.get("realizations") or {}).get(line, []) if r.get("tier") != "blocked"]
    if prefer:
        for real in reals:
            if real.get("id") == prefer:
                return real
        raise StyleUnavailable(f"{card['id']}: no usable realization {prefer!r} on {line}")
    if not reals:
        paths = [g["path"] for g in card.get("gaps") or [] if g.get("line") == line]
        hint = f"; gap path: {paths[0]}" if paths else ""
        raise StyleUnavailable(f"{card['id']}: no realization on {line}{hint}")
    return sorted(reals, key=lambda r: _TIER_RANK[r["tier"]])[0]


def resolve(style_id: str, line: str, *, cards: dict[str, dict] | None = None,
            box: str | None = None, prefer: str | None = None, library: dict | None = None,
            loras: dict | None = None, t2i_dir: Path | str | None = None) -> RenderFragment:
    """Turn a style id into the prompt, LoRA and model fragment one line needs."""
    if line not in LINES:
        raise StyleRegistryError(f"unknown line {line!r}; expected one of {LINES}")
    cards = cards if cards is not None else load_cards()
    card = cards.get(style_id)
    if card is None:
        raise StyleRegistryError(f"unknown style {style_id!r}")
    if line not in (card.get("lines") or []):
        raise StyleUnavailable(f"{style_id}: line {line} is not a target of this style")
    real = pick_realization(card, line, prefer)
    frag = RenderFragment(style_id=style_id, line=line, realization_id=real["id"],
                          kind=real["kind"], tier=real["tier"], note=real.get("note", ""))
    kind = real["kind"]
    if kind == "nai_card":
        library = library if library is not None else nai_library()
        nai_card = library.get(real["ref"])
        if nai_card is None:
            raise StyleRegistryError(f"{style_id}: unknown NovelAI card {real['ref']!r}")
        frag.pos_prefix = compose_nai_block(nai_card.get("components", []))
        frag.neg_add = nai_card.get("neg_add", "") or ""
        frag.model = real.get("model") or (nai_card.get("verified") or {}).get("model") or NAI_DEFAULT_MODEL
    elif kind in ("lora", "lora_stack"):
        loras = loras if loras is not None else lora_index()
        entries = [{"ref": real["ref"], "weight": real["weight"]}] if kind == "lora" else real["loras"]
        for entry in entries:
            rec = loras.get(entry.get("ref", "")) if entry.get("ref") else None
            file = entry.get("file") or (rec or {}).get("file")
            if file is None:
                raise StyleRegistryError(f"{style_id}: LoRA {entry.get('ref')!r} not in the catalog")
            if box == "boxb" and rec and rec.get("boxb_file"):
                file = rec["boxb_file"]
            elif box == "boxb" and rec and not rec.get("boxb_file"):
                frag.note = (frag.note + "; " if frag.note else "") + f"{file} not registered on boxb"
            weight = float(entry.get("weight", 0.8))
            frag.loras.append({"file": file, "weight": weight, "ref": entry.get("ref"),
                               "trigger": entry.get("trigger", "")})
            frag.lora_specs.append(f"{file}:{weight:g}")
        frag.pos_prefix = real.get("prompt", "")
        frag.base_pin = real.get("base_pin")
    elif kind == "t2i_item":
        item = t2i_item(real["ref"], t2i_dir)
        if item is None:
            raise StyleRegistryError(f"{style_id}: unknown t2i item {real['ref']!r}")
        dialects = item.get("dialects") or {}
        if not dialects.get(line):
            raise StyleUnavailable(f"{style_id}: t2i item {real['ref']!r} has no {line} dialect")
        frag.pos_prefix = dialects[line]
    elif kind == "prompt":
        frag.pos_prefix = real.get("prompt", "")
        frag.base_pin = real.get("base_pin")
    elif kind == "civitai_air":
        frag.air = real["air"]
        frag.base_model = real.get("base_model")
        frag.trigger = real.get("trigger", "")
        frag.pos_prefix = real.get("prompt") or real.get("trigger", "")
        frag.loras.append({"file": real["air"], "weight": float(real["weight"]), "ref": None,
                           "trigger": real.get("trigger", "")})
        frag.lora_specs.append(f"{real['air']}:{float(real['weight']):g}")
    elif kind == "caption":
        frag.caption = real["caption"]
        frag.pos_prefix = real["caption"]
    frag.neg_add = real.get("negative", frag.neg_add) or frag.neg_add
    frag.post_fx = real.get("post_fx", frag.post_fx)
    frag.sampler = dict(real.get("sampler") or {})
    frag.pos_suffix = real.get("suffix", "")
    return frag


# --------------------------------------------------------------------------- coverage


def line_status(card: dict, line: str) -> tuple[str, str | None]:
    """(status, best realization id) for one line: verified, candidate, gap or na."""
    if line not in (card.get("lines") or []):
        return "na", None
    reals = [r for r in (card.get("realizations") or {}).get(line, []) if r.get("tier") != "blocked"]
    if not reals:
        return "gap", None
    best = sorted(reals, key=lambda r: _TIER_RANK[r["tier"]])[0]
    return best["tier"], best["id"]


def matrix(cards: dict[str, dict]) -> list[dict]:
    rows = []
    for card_id, card in cards.items():
        row = {"id": card_id, "family": card.get("family"), "priority": card.get("priority"),
               "display_name": card.get("display_name"), "lines": {}, "gaps": {}}
        for line in LINES:
            status, best = line_status(card, line)
            row["lines"][line] = {"status": status, "best": best}
        for gap in card.get("gaps") or []:
            row["gaps"][gap["line"]] = gap["path"]
        rows.append(row)
    return rows


def matrix_summary(rows: list[dict]) -> dict:
    counts = {line: {s: 0 for s in ("verified", "candidate", "gap", "na")} for line in LINES}
    any_verified = 0
    for row in rows:
        if any(cell["status"] == "verified" for cell in row["lines"].values()):
            any_verified += 1
        for line in LINES:
            counts[line][row["lines"][line]["status"]] += 1
    return {"cards": len(rows), "any_verified": any_verified, "per_line": counts}


def matrix_markdown(rows: list[dict]) -> str:
    header = "| id | " + " | ".join(LINES) + " | 缺口路径 |"
    sep = "|---|" + "---|" * (len(LINES) + 1)
    out = [header, sep]
    for row in rows:
        cells = []
        for line in LINES:
            cell = row["lines"][line]
            glyph = STATUS_GLYPH[cell["status"]]
            cells.append(f"{glyph} {cell['best']}" if cell["best"] else glyph)
        gaps = "; ".join(f"{line}: {path}" for line, path in row["gaps"].items())
        out.append(f"| {row['id']} | " + " | ".join(cells) + f" | {gaps} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Aesthetic registers (2026-09-16): the layer users order by. A register names a finish
# and lists the cards that realize it; density maps the three user levels to each line's
# own control. Registers only reference cards.
# ---------------------------------------------------------------------------

REGISTERS_PATH = CARDS_DIR / "registers.json"
REGISTER_SCHEMA_ID = "loramake.style-register/v1"
DENSITY_LEVELS = ("light", "medium", "strong")


def load_registers(path: Path | str | None = None) -> dict:
    data = _load_json(Path(path) if path else REGISTERS_PATH)
    if data.get("schema") != REGISTER_SCHEMA_ID:
        raise StyleRegistryError(f"registers schema must be {REGISTER_SCHEMA_ID}")
    return data


def register_problems(registers: dict, cards: dict[str, dict]) -> list[str]:
    """Reference and shape checks for registers.json (no GPU, no network)."""
    problems: list[str] = []
    density = registers.get("density") or {}
    if tuple(density.get("levels") or ()) != DENSITY_LEVELS:
        problems.append(f"registers: density.levels must be {list(DENSITY_LEVELS)}")
    for line in LINES:
        table = density.get(line)
        if table is None:
            problems.append(f"registers: density table missing for line {line}")
        elif table and set(table) != set(DENSITY_LEVELS):
            problems.append(f"registers: density.{line} must map exactly {list(DENSITY_LEVELS)}")
    seen: dict[str, str] = {}
    for reg_id, reg in (registers.get("registers") or {}).items():
        if not _ID_RE.match(reg_id):
            problems.append(f"registers: bad register id {reg_id!r}")
        if not reg.get("name_cn"):
            problems.append(f"{reg_id}: missing name_cn")
        card_ids = reg.get("cards") or []
        if not card_ids:
            problems.append(f"{reg_id}: no cards")
        for line in reg.get("hold") or {}:
            if line not in LINES:
                problems.append(f"{reg_id}: hold names unknown line {line!r}")
        for card_id in card_ids:
            if card_id not in cards:
                problems.append(f"{reg_id}: unknown card {card_id!r}")
            elif card_id in seen:
                problems.append(f"{reg_id}: card {card_id!r} already belongs to {seen[card_id]}")
            seen[card_id] = reg_id
    return problems


def normalize_density(value: str | None, registers: dict | None = None) -> str:
    if value is None:
        return "medium"
    aliases = ((registers or {}).get("density") or {}).get("aliases") or {}
    level = aliases.get(value, value)
    if level not in DENSITY_LEVELS:
        raise StyleRegistryError(f"unknown density {value!r}; expected one of {DENSITY_LEVELS}")
    return level


def register_best_card(reg: dict, line: str, cards: dict[str, dict]) -> tuple[str | None, str]:
    """(card_id, status) of the register's best realization on ``line``: verified beats candidate."""
    best_id, best_status, best_rank = None, "na", len(TIERS)
    for card_id in reg.get("cards") or []:
        card = cards.get(card_id)
        if card is None:
            continue
        status, _ = line_status(card, line)
        rank = _TIER_RANK.get(status, len(TIERS))
        if status in ("verified", "candidate") and rank < best_rank:
            best_id, best_status, best_rank = card_id, status, rank
    return best_id, best_status


def register_matrix(registers: dict, cards: dict[str, dict]) -> list[dict]:
    rows = []
    for reg_id, reg in (registers.get("registers") or {}).items():
        row = {"id": reg_id, "name_cn": reg.get("name_cn"), "cards": list(reg.get("cards") or []),
               "lines": {}}
        for line in LINES:
            best_id, status = register_best_card(reg, line, cards)
            row["lines"][line] = {"status": status, "card": best_id}
        rows.append(row)
    return rows


def register_matrix_markdown(rows: list[dict]) -> str:
    header = "| register | 中文 | " + " | ".join(LINES) + " | cards |"
    sep = "|---|---|" + "---|" * (len(LINES) + 1)
    out = [header, sep]
    for row in rows:
        cells = []
        for line in LINES:
            cell = row["lines"][line]
            glyph = STATUS_GLYPH[cell["status"]]
            cells.append(f"{glyph} {cell['card']}" if cell["card"] else glyph)
        out.append(f"| {row['id']} | {row['name_cn']} | " + " | ".join(cells)
                   + f" | {', '.join(row['cards'])} |")
    return "\n".join(out)


def _apply_density(frag: RenderFragment, level: str, registers: dict, library: dict | None) -> RenderFragment:
    table = ((registers.get("density") or {}).get(frag.line) or {})
    value = table.get(level)
    if value is None:
        return frag
    if frag.kind in ("lora", "lora_stack", "civitai_air") and frag.loras:
        scale = value / max(float(frag.loras[0]["weight"]), 1e-6)
        frag.loras = [{**entry, "weight": round(float(entry["weight"]) * scale, 3)} for entry in frag.loras]
        frag.lora_specs = [f"{entry['file']}:{entry['weight']:g}" for entry in frag.loras]
    elif frag.kind == "nai_card":
        library = library if library is not None else nai_library()
        cards = load_cards()
        card = cards[frag.style_id]
        real = next(r for r in card["realizations"][frag.line] if r["id"] == frag.realization_id)
        comps = [dict(c) for c in library[real["ref"]].get("components", [])]
        artists = [c for c in comps if c.get("kind") == "artist" and c.get("weight") is not None]
        if artists:
            primary = max(float(c["weight"]) for c in artists)
            scale = value / max(primary, 1e-6)
            for comp in artists:
                comp["weight"] = round(float(comp["weight"]) * scale, 2)
            frag.pos_prefix = compose_nai_block(comps)
    frag.note = (frag.note + "; " if frag.note else "") + f"density={level}"
    return frag


def resolve_aesthetic(register_id: str, line: str, *, density: str | None = None,
                      registers: dict | None = None, cards: dict[str, dict] | None = None,
                      box: str | None = None, library: dict | None = None) -> RenderFragment:
    """Turn a register (or a bare card id) plus density into one line's render fragment.

    Picks the register's best-tier card on ``line`` (verified over candidate), resolves it,
    then rescales the line's own strength control to the density level. A card id is
    accepted in place of a register id so callers need one entry point."""
    registers = registers if registers is not None else load_registers()
    cards = cards if cards is not None else load_cards()
    level = normalize_density(density, registers)
    reg = (registers.get("registers") or {}).get(register_id)
    if reg is None:
        if register_id in cards:
            card_id = register_id
        else:
            raise StyleRegistryError(f"unknown aesthetic register or style {register_id!r}")
    else:
        card_id, status = register_best_card(reg, line, cards)
        if card_id is None:
            raise StyleUnavailable(f"{register_id}: no usable card on {line} "
                                   f"(cards: {', '.join(reg.get('cards') or [])})")
    frag = resolve(card_id, line, cards=cards, box=box, library=library)
    return _apply_density(frag, level, registers, library)


_ROUND_RE = re.compile(r"\b(love|yes|no) (\d+)")
RANDOM_POOL_MIN_N = 6  # user 2026-09-16: one full probe round (6 seeds) with no rejection is enough for the random pool


def realization_clean_round(real: dict) -> bool:
    """True when the realization's most recent probe round rejected nothing.

    The note's ``verdicts`` tag accumulates over rounds, so once a round rejects an
    image the tag can never read zero again and a later clean round could never admit
    the register. The round is therefore read off ``evidence``, which every writeback
    overwrites and which so describes the latest round alone; evidence that does not
    parse fails closed. The card tier still follows the love-based promotion rule;
    this is the weaker "the last round rejected nothing" floor the aesthetic random
    pool needs (user decision 2026-09-16)."""
    counts = {k: int(v) for k, v in _ROUND_RE.findall((real.get("evidence") or {}).get("note") or "")}
    return bool(counts) and counts.get("no", 0) == 0 and sum(counts.values()) >= RANDOM_POOL_MIN_N


def verified_registers(line: str, registers: dict | None = None,
                       cards: dict[str, dict] | None = None) -> list[str]:
    """Register ids usable by random draws on ``line``: the best card is verified, or its
    best realization has a full probe round with zero rejections (see realization_clean_round)."""
    registers = registers if registers is not None else load_registers()
    cards = cards if cards is not None else load_cards()
    out = []
    for reg_id, reg in (registers.get("registers") or {}).items():
        if line in (reg.get("hold") or {}):  # user judged the line's realization off-register
            continue
        card_id, status = register_best_card(reg, line, cards)
        if status == "verified":
            out.append(reg_id)
        elif status == "candidate" and card_id:
            # judge the realization a draw would actually render, not any sibling on the card
            try:
                picked = pick_realization(cards[card_id], line)
            except StyleUnavailable:
                continue
            if realization_clean_round(picked):
                out.append(reg_id)
    return sorted(out)
