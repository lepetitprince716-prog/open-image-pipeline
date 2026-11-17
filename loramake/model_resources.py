"""Shared Model Resource Catalog — fast, stdlib-only, deterministic LoRA/index operations.

This module is the single home for lookups, role classification, trigger deduplication,
family compatibility, and weighted lexical search over the LoRA catalog.  It must not
import any script under ``scripts/`` (especially ``lora_catalog``) to avoid cycles.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "lora_catalog" / "boxa_index.json"

ROLE_ORDER = ("character", "style", "concept", "pose", "detail")

# --------------------------------------------------------------------------- constants

# civitai nsfw levels: 1=PG 2=PG13 4=R 8=X 16=XXX. >=4 (or an explicit tag) -> nsfw.
NSFW_TAGS = {"nsfw", "explicit", "adult"}

CATEGORY_ROLE = {
    "character": "character",
    "style": "style",
    "concept": "concept",
    "clothing": "concept",
    "objects": "concept",
    "background": "concept",
    "poses": "pose",
    "action": "pose",
}

TAG_ROLE = {
    "character": "character",
    "poses": "pose", "pose": "pose", "action": "pose",
    "style": "style",
    "concept": "concept", "clothing": "concept", "background": "concept", "objects": "concept",
    "full body": "concept",
}

TAG_PRECEDENCE = ("character", "pose", "style", "concept")

# Unambiguous detail-enhancer tokens only (matched as substrings of name+tags+triggers).
DETAIL_SIGNAL = ("enhancer", "detailer", "booster", "tweaker")

ROLE_KEYWORDS = {
    "style": ("style", "artstyle", "aesthetic", "shading", "lineart", "watercolor", "oil painting",
              "painterly", "flat color", "sketch"),
    "concept": ("concept", "clothing", "outfit", "see-through", "see through", "costume", "effect"),
    "pose": ("pose", "posing", "position", "kneeling", "sitting", "lying", "angle"),
    "detail": ("detail", "slider", "enhancer", "enhance", "texture", "tweaker", "sharp",
               "more_detail", "add_detail", "quality"),
}

# Families that cannot drive an SDXL-family graph.  Substrings are matched
# case-insensitively against the catalog's base_model strings.
SDXL_ALIEN_TOKENS = ("anima", "sd 1.5", "sd1.5", "sd15", "flux", "video", "zimage",
                     "chroma", "klein", "kontext")

# Families the relay draft checkpoint (iLustMix / SDXL) is known to accept.
# Unknown / empty falls back to file-path token screening.
DRAFT_BASE_FAMILIES = ("illustrious", "sdxl", "noobai", "pony")

# Markers used when base_model is Unknown / Other / empty and we must infer from
# the record itself.  These are NOT the permissive family_compatible token table;
# they are deliberately conservative so search/auto-selection does not silently
# suggest a wrong-architecture LoRA.
SDXL_MARKERS = ("illustrious", "ilust", "noobai", "noob", "pony", "sdxl", "sd_xl", "sd-xl")
SDXL_REJECT_MARKERS = SDXL_ALIEN_TOKENS + ("krea", "wan", "i2v", "t2v", "cogvideo",
                                            "hunyuan", "mochi", "lumina", "aura", "ltx",
                                            "svd", "pika", "runway", "stablevideo", "sana")
ANIMA_MARKERS = ("anima",)
ANIMA_REJECT_MARKERS = SDXL_MARKERS + ("sd 1.5", "sd1.5", "sd15", "flux", "video",
                                       "zimage", "chroma", "klein", "kontext", "krea")
SD15_MARKERS = ("sd 1.5", "sd1.5", "sd15")
SD15_REJECT_MARKERS = ("anima", "flux", "illustrious", "ilust", "noobai", "noob",
                       "pony", "sdxl", "sd_xl", "sd-xl", "zimage", "chroma", "kontext",
                       "krea", "video", "wan", "i2v", "t2v")


# --------------------------------------------------------------------------- errors

class AmbiguousRequestError(ValueError):
    """A requested name maps to more than one catalog record."""


class UnknownFamilyError(ValueError):
    """A --family value is not one of the supported arch families."""


# --------------------------------------------------------------------------- tokenization

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+")
_SEPARATOR_RE = re.compile(r"[_\-.]+")


def tokenize_query(text: str) -> list[str]:
    """ASCII words plus CJK runs (Hiragana + Katakana + Han) — CJK has no spaces to split on."""
    return _TOKEN_RE.findall(text.lower())


def _bigrams(s: str) -> set[str]:
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _field_hit(tok: str, field: str) -> bool:
    """True when ``tok`` matches inside ``field``.

    ASCII tokens match on word boundaries; file separators are treated as spaces,
    so ``raiden`` hits ``raiden_il.safetensors`` but a short token no longer hits
    the inside of an unrelated word.  A CJK run matches when it is a substring of
    a CJK chunk, or when it shares at least half of the smaller side's character
    bigrams with a CJK chunk in the field.
    """
    if tok.isascii():
        return re.search(rf"\b{re.escape(tok)}\b", _SEPARATOR_RE.sub(" ", field)) is not None
    if tok in field:
        return True
    grams = _bigrams(tok)
    if not grams:
        return False
    for chunk in _TOKEN_RE.findall(field):
        if chunk.isascii():
            continue
        chunk_grams = _bigrams(chunk)
        if not chunk_grams:
            continue
        shared = len(grams & chunk_grams)
        if shared and shared / min(len(grams), len(chunk_grams)) >= 0.5:
            return True
    return False



# Public alias: other selectors (nai_style.py) reuse the same CJK-aware matcher.
field_hit = _field_hit

def _norm_name(name: str) -> str:
    return name.replace("\\", "/").lower()


def _basename(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1].lower()


def _stem(name: str) -> str:
    b = _basename(name)
    if "." in b:
        return b.rsplit(".", 1)[0]
    return b


def _exact_forms(record: dict) -> set[str]:
    """All normalized exact-match targets for a record.

    Used to distinguish a true exact request (e.g., the exact display name,
    canonical loader path, boxb alias, basename, or stem) from a substring phrase
    match.  Cross-collisions between these forms are detected in the snapshot.
    """
    file = str(record.get("file", ""))
    name = str(record.get("name", ""))
    zf = str(record.get("boxb_file") or "")
    forms: set[str] = set()
    for raw in (file, name):
        if raw:
            forms.add(_norm_name(raw))
    for raw in (zf,):
        if raw:
            forms.add(_norm_name(raw))
            forms.add(_norm_name(_basename(raw)))
            forms.add(_norm_name(_stem(raw)))
    if file:
        forms.add(_norm_name(_basename(file)))
        forms.add(_norm_name(_stem(file)))
    return forms


def _is_exact_match(query: str, record: dict) -> bool:
    q = _norm_name(query).strip()
    if not q:
        return False
    return q in _exact_forms(record)


# --------------------------------------------------------------------------- catalog snapshot

_SNAPSHOT_LOCK = threading.RLock()
_SNAPSHOTS: dict[Path, tuple[tuple[int, int], Snapshot]] = {}  # noqa: E402


@dataclass
class Snapshot:
    records: list[dict]
    by_canonical: dict[str, dict]
    by_alias: dict[str, dict]
    by_basename: dict[str, list[dict]]
    ambiguous_canonical: set[str]
    ambiguous_alias: set[str]
    key: tuple[int, int]
    path: Path

    def lookup_exact(self, name: str) -> dict | None:
        """Exact canonical lookup.  Raises on duplicate canonical keys."""
        key = _norm_name(name)
        if key in self.ambiguous_canonical:
            raise AmbiguousRequestError(f"canonical name {name!r} is ambiguous in the catalog")
        return self.by_canonical.get(key)

    def lookup_alias(self, name: str) -> dict | None:
        """Exact BOXB alias lookup.  Raises on duplicate alias keys."""
        key = _norm_name(name)
        if key in self.ambiguous_alias:
            raise AmbiguousRequestError(f"boxb alias {name!r} is ambiguous in the catalog")
        return self.by_alias.get(key)

    def lookup_basename(self, name: str) -> dict | None:
        """Unique basename fallback.  Raises if the basename matches more than one record."""
        base = _basename(name)
        cands = self.by_basename.get(base, [])
        if not cands:
            return None
        if len(cands) > 1:
            names = [r["file"] for r in cands]
            raise AmbiguousRequestError(
                f"basename {name!r} is ambiguous: matches {len(names)} records: {names}"
            )
        return cands[0]


def _build_snapshot(records: list[dict], path: Path, key: tuple[int, int]) -> Snapshot:
    by_canonical: dict[str, dict] = {}
    by_alias: dict[str, dict] = {}
    by_basename: dict[str, set[int]] = {}
    id_to_record: dict[int, dict] = {}
    ambiguous_canonical: set[str] = set()
    ambiguous_alias: set[str] = set()

    for rec in records:
        # exact canonical
        ckey = _norm_name(rec["file"])
        if ckey in by_canonical:
            ambiguous_canonical.add(ckey)
        else:
            by_canonical[ckey] = rec
        id_to_record[id(rec)] = rec

        # basename (canonical)
        b = _basename(rec["file"])
        by_basename.setdefault(b, set()).add(id(rec))

        # exact alias and alias basename
        zf = rec.get("boxb_file")
        if isinstance(zf, str):
            akey = _norm_name(zf)
            if akey in by_alias:
                ambiguous_alias.add(akey)
            by_alias.setdefault(akey, rec)
            ab = _basename(zf)
            by_basename.setdefault(ab, set()).add(id(rec))

    # Cross-collisions: a canonical exact key of one record must not silently match
    # the BOXB alias exact key of another.
    for canon_key, rec in by_canonical.items():
        alias_rec = by_alias.get(canon_key)
        if alias_rec is not None and alias_rec is not rec:
            ambiguous_canonical.add(canon_key)
            ambiguous_alias.add(canon_key)

    by_basename_list = {
        b: [id_to_record[iid] for iid in ids] for b, ids in by_basename.items()
    }

    return Snapshot(
        records=records,
        by_canonical=by_canonical,
        by_alias=by_alias,
        by_basename=by_basename_list,
        ambiguous_canonical=ambiguous_canonical,
        ambiguous_alias=ambiguous_alias,
        key=key,
        path=path,
    )


def load_snapshot(index_path: Path | str | None = None) -> Snapshot:
    """Load the LoRA index and build lookup indexes, cached on (mtime_ns, size).

    The read/rebuild/invalidate seam is fully locked so it is safe for concurrent
    callers (e.g., a FastAPI worker pool).  The cache is invalidated automatically
    when the source file changes.
    """
    with _SNAPSHOT_LOCK:
        path = Path(index_path or INDEX_PATH)
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = _SNAPSHOTS.get(path)
        if cached and cached[0] == key:
            return cached[1]
        records = json.loads(path.read_text(encoding="utf-8"))
        snap = _build_snapshot(records, path, key)
        _SNAPSHOTS[path] = (key, snap)
        return snap


def invalidate_cache(path: Path | str | None = None) -> None:
    """Drop a cached snapshot so the next ``load_snapshot`` re-reads the file."""
    with _SNAPSHOT_LOCK:
        if path is None:
            _SNAPSHOTS.clear()
        else:
            _SNAPSHOTS.pop(Path(path), None)


# --------------------------------------------------------------------------- spec parsing

def parse_spec(spec: str, default_strength: float | None = None) -> tuple[str, float | None]:
    """Split a FILE[:STRENGTH] LoRA spec into (file, strength).

    A non-numeric tail after the last colon belongs to the filename — it must never
    crash the caller.  ``None`` strength means the caller applies its own default.
    """
    if ":" in spec:
        fname, _, raw = spec.rpartition(":")
        try:
            return fname, float(raw)
        except ValueError:
            pass
    return spec, default_strength


# --------------------------------------------------------------------------- resolution

@dataclass(frozen=True)
class ResolveResult:
    record: dict
    canonical: str
    loader_name: str | None
    strength: float | None
    available: bool | None
    reason: str


def _non_empty(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def box_available(record: dict, box: str | None) -> bool | None:
    """Availability for a specific box.

    * boxa is always available (``True``).
    * boxb: ``True`` if ``boxb_file`` is a non-empty string, ``False`` if it is
      explicitly ``None`` or empty (known unavailable), ``None`` if the key is
      missing (availability unknown — must not be reported as absent).
    """
    if box in (None, "boxa"):
        return True
    if box != "boxb":
        return None
    if "boxb_file" not in record:
        return None
    zf = record["boxb_file"]
    return _non_empty(zf)


def box_loader(record: dict, box: str | None) -> str | None:
    """The loader-acceptable name for ``record`` on the requested box.

    For boxb with unknown availability (missing ``boxb_file`` key), the canonical
    BOXA name is passed through to live preflight rather than failing closed.
    """
    if box in (None, "boxa"):
        return record.get("file")
    if box != "boxb":
        return None
    if "boxb_file" not in record:
        return record.get("file")
    zf = record["boxb_file"]
    return zf if _non_empty(zf) else None


def apply_box(records: list[dict], box: str | None) -> list[dict]:
    """Rewrite records for the target box.

    boxa / no box returns the original list unchanged.  boxb returns a new list
    with records whose availability is not known-``None``.  Each returned record
    has ``file`` set to the boxb loader name and ``boxa_file`` set to the original
    canonical name.
    """
    if not box or box == "boxa":
        return records
    out: list[dict] = []
    for rec in records:
        loader = box_loader(rec, "boxb")
        if loader is None:
            continue
        new = dict(rec)
        new["boxa_file"] = rec["file"]
        new["file"] = loader
        out.append(new)
    return out


def resolve_lora(request: str, box: str | None = None,
                 default_strength: float | None = None,
                 index_path: Path | str | None = None) -> ResolveResult:
    """Resolve a requested LoRA name/spec to a catalog record and box loader.

    Resolution order: exact BOXA canonical, exact BOXB alias, unique basename.
    Any ambiguous exact key (canonical, alias, or cross-form) or ambiguous
    basename raises ``AmbiguousRequestError`` so the caller never silently picks
    the wrong record.
    """
    if box is not None and box not in ("boxa", "boxb"):
        raise ValueError(f"unknown box {box!r}; use boxa | boxb")

    name, strength = parse_spec(request, default_strength)
    snap = load_snapshot(index_path)

    # 1. exact canonical
    rec = snap.lookup_exact(name)
    if rec is None:
        # 2. exact boxb alias (only meaningful for boxb, but harmless)
        rec = snap.lookup_alias(name)
    if rec is None:
        # 3. unique basename
        rec = snap.lookup_basename(name)

    if rec is None:
        return ResolveResult(
            record={}, canonical=name, loader_name=None, strength=strength,
            available=False, reason="not in catalog",
        )

    canonical = rec["file"]
    loader = box_loader(rec, box)
    avail = box_available(rec, box)
    if avail is False:
        reason = f"unavailable on {box}"
    elif avail is None:
        reason = f"{box} availability unknown"
    else:
        reason = "ok"
    return ResolveResult(
        record=rec, canonical=canonical, loader_name=loader,
        strength=strength, available=avail, reason=reason,
    )


def record_of(lora_file: str, index_path: Path | str | None = None) -> dict | None:
    """Full catalog record for a LoRA file name, or ``None`` if unknown."""
    try:
        result = resolve_lora(lora_file, box="boxa", index_path=index_path)
    except AmbiguousRequestError:
        raise
    if result.available:
        return result.record
    return None


def base_model_of(lora_file: str, index_path: Path | str | None = None) -> str | None:
    """Catalogued base_model for a LoRA file name, or ``None`` if unknown."""
    rec = record_of(lora_file, index_path=index_path)
    return rec.get("base_model") if rec else None


# --------------------------------------------------------------------------- scoring / search

def score_record(query: str, rec: dict) -> float:
    """Weighted lexical overlap in 0..1: name=3x, trigger/file=2x, tags=1x.

    Deterministic and offline — no embedding model involved.
    """
    score, _hits = _match_score_plain(tokenize_query(query), rec)
    return score


def _why_for(query: str, rec: dict, score: float, hits: dict[str, int], exact: bool) -> str:
    if exact:
        return "exact"
    if not tokenize_query(query):
        return "inventory"
    bits: list[str] = []
    if hits.get("name"):
        bits.append("name")
    if hits.get("file"):
        bits.append("file")
    if hits.get("trigger"):
        bits.append("trigger")
    if hits.get("tag"):
        bits.append("tag")
    return "; ".join(bits) if bits else "match"


def _match_score_plain(toks: list[str], rec: dict) -> tuple[float, dict[str, int]]:
    """Weighted overlap for an already-tokenized query. Missing tokens dilute; they do not AND-fail."""
    if not toks:
        return 0.0, {}
    name = _norm_name(str(rec.get("name", "")))
    fname = _norm_name(str(rec.get("file", "")))
    triggers = " ".join(str(t) for t in rec.get("triggers") or []).lower()
    tags = " ".join(str(t) for t in rec.get("tags") or []).lower()
    hits: dict[str, int] = {}
    total = 0.0
    for tok in toks:
        if _field_hit(tok, name):
            total += 3.0
            hits["name"] = hits.get("name", 0) + 1
        elif _field_hit(tok, triggers):
            total += 2.0
            hits["trigger"] = hits.get("trigger", 0) + 1
        elif _field_hit(tok, fname):
            total += 2.0
            hits["file"] = hits.get("file", 0) + 1
        elif _field_hit(tok, tags):
            total += 1.0
            hits["tag"] = hits.get("tag", 0) + 1
    return round(total / (3.0 * len(toks)), 4), hits


def _match_score(query: str, rec: dict) -> tuple[float, dict[str, int], bool]:
    toks = tokenize_query(query)
    if not toks:
        return 0.0, {}, False
    score, hits = _match_score_plain(toks, rec)
    return score, hits, _is_exact_match(query, rec)


# --------------------------------------------------------------------------- family compatibility

def sdxl_family(base: str | None) -> bool:
    """True when ``base`` is compatible with SDXL-family graphs (or unknown)."""
    lowered = (base or "").lower()
    return not any(t in lowered for t in SDXL_ALIEN_TOKENS)


def sdxl_draft_family(base: str | None) -> bool:
    """Strict family check for the relay draft checkpoint.

    Accepts only explicitly known SDXL families (illustrious / sdxl / noobai / pony).
    Unknown or empty returns False so the caller falls back to path-token screening.
    """
    lowered = (base or "").lower()
    if not lowered or lowered == "unknown":
        return False
    return any(f in lowered for f in DRAFT_BASE_FAMILIES)


def anima_family(base: str | None) -> bool:
    """True for Anima / flow-matching base models."""
    return "anima" in (base or "").lower()


def sd15_family(base: str | None) -> bool:
    """True for SD 1.5 base models."""
    lowered = (base or "").lower()
    return "sd 1.5" in lowered or "sd1.5" in lowered


def family_compatible(base: str | None, family: str | None) -> bool:
    """Predicate for arch-family guards.

    * family = ``any``/``None`` → always True
    * family = ``sdxl`` → rejects known non-SDXL families
    * family = ``anima`` → only Anima
    * family = ``sd15``/``sd1.5`` → only SD 1.5
    """
    fam = (family or "any").lower()
    if fam in ("", "any"):
        return True
    if fam == "sdxl":
        return sdxl_family(base)
    if fam == "anima":
        return anima_family(base)
    if fam in ("sd15", "sd1.5"):
        return sd15_family(base)
    raise UnknownFamilyError(f"unknown family {family!r}; use sdxl | anima | sd15 | any")


def family_pred(family: str | None) -> Callable[[str], bool]:
    """Return a callable predicate for ``family`` (usable in filters).

    The family is validated eagerly so callers can catch ``UnknownFamilyError``
    at call time rather than when the predicate is first applied.
    """
    # Validate family eagerly; base argument is irrelevant for validation.
    family_compatible("", family)

    def pred(base: str) -> bool:
        return family_compatible(base, family)
    return pred


def _record_family_hay(record: dict) -> str:
    """All record text that may signal architecture: file, alias, name, category, tags, triggers."""
    parts: list[str] = []
    for key in ("file", "boxb_file", "name", "category"):
        if record.get(key):
            parts.append(str(record[key]))
    for key in ("tags", "triggers"):
        for entry in record.get(key) or []:
            parts.append(str(entry))
    return _norm_name(" ".join(parts))


_ARCH_SUBSTITUTIONS = (("sd 1.5", "sd15"), ("sd1.5", "sd15"),
                       ("sd_xl", "sdxl"), ("sd-xl", "sdxl"))
_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff]+")


def _record_family_tokens(record: dict) -> set[str]:
    """Tokenized architecture clues from the record, with SD* normalizations."""
    hay = _record_family_hay(record)
    for old, new in _ARCH_SUBSTITUTIONS:
        hay = hay.replace(old, new)
    # Split on punctuation/whitespace while keeping whole CJK words and ASCII tokens.
    return set(_TOKEN_SPLIT_RE.split(hay.lower()))


def record_family_status(record: dict, family: str | None) -> bool | None:
    """Tri-state record-family check for explicit compilation.

    * known compatible -> True
    * proven mismatch  -> False (reject)
    * genuinely unclassified (Unknown/Other/no markers) -> None (allow with warning + live preflight)

    ``family`` = any/None/"" always returns True.  Raises ``UnknownFamilyError``
    for unsupported families.
    """
    fam = (family or "any").lower()
    if fam in ("", "any"):
        return True
    if fam not in ("sdxl", "anima", "sd15", "sd1.5"):
        raise UnknownFamilyError(f"unknown family {family!r}; use sdxl | anima | sd15 | any")

    base = str(record.get("base_model") or "").lower()
    # Explicit catalog metadata wins when it is concrete.
    if base and base not in ("unknown", "other"):
        return bool(family_compatible(record.get("base_model"), family))

    tokens = _record_family_tokens(record)
    if fam == "sdxl":
        if tokens & set(SDXL_REJECT_MARKERS):
            return False
        return True if tokens & set(SDXL_MARKERS) else None
    if fam == "anima":
        if tokens & set(ANIMA_REJECT_MARKERS):
            return False
        return True if tokens & set(ANIMA_MARKERS) else None
    if fam in ("sd15", "sd1.5"):
        if tokens & set(SD15_REJECT_MARKERS):
            return False
        return True if tokens & set(SD15_MARKERS) else None
    return None


def record_family_compatible(record: dict, family: str | None) -> bool:
    """Record-aware family predicate for search and auto-selection.

    If ``base_model`` is a concrete value (not Unknown / Other / empty), this falls
    back to ``family_compatible`` so explicit catalog metadata wins.  Otherwise it
    inspects file, boxb_file, name, category, tags and triggers for strong markers.

    * family = ``any``/``None`` → always True
    * family = ``sdxl`` → accepts records with SDXL markers (illustrious/ilust/noobai/pony/sdxl);
      rejects Anima/SD1.5/Flux/video/Krea and other non-SDXL markers.  Records with
      no strong markers are not auto-suggested under ``sdxl``.
    * family = ``anima`` / ``sd15`` → analogous conservative checks.

    Raises ``UnknownFamilyError`` for an unsupported family value.
    """
    status = record_family_status(record, family)
    return False if status is None else bool(status)


# --------------------------------------------------------------------------- LoRA spec / trace helpers


def parse_lora_spec(spec: str, default_strength: float | None = None) -> tuple[str, str | None, float | None]:
    """Split a FILE[:STRENGTH] LoRA spec into (name, raw strength tail, numeric strength).

    A non-numeric tail after the last colon belongs to the filename and is preserved
    so alias resolution does not rewrite exact strength text.
    """
    if ":" in spec:
        name, _, raw = spec.rpartition(":")
        try:
            return name, f":{raw}", float(raw)
        except ValueError:
            pass
    return spec, None, default_strength


def resolve_for_box(spec: str, box: str, family: str | None = None,
                    index_path=None) -> tuple[str, dict]:
    """Resolve one LoRA spec to a loader-acceptable name for ``box`` and return a trace.

    Raises ``ValueError`` on ambiguity, known absence, or family mismatch.
    Unindexed names pass through unchanged.
    """
    name, raw_tail, strength = parse_lora_spec(spec)
    try:
        res = resolve_lora(name, box=box, default_strength=strength, index_path=index_path)
    except AmbiguousRequestError as exc:
        raise ValueError(f"{spec!r} is ambiguous in the catalog: {exc}") from None
    if not res.record:
        resolved = f"{name}{raw_tail}" if raw_tail is not None else name
        return resolved, {
            "requested": spec,
            "canonical": None,
            "resolved": resolved,
            "strength": strength,
            "available": None,
            "family_ok": None,
            "warning": "unindexed; live preflight is authority",
        }
    family_ok = record_family_status(res.record, family)
    if family is not None and family_ok is False:
        raise ValueError(
            f"{spec!r} is not compatible with the {family} route family "
            f"(record base={res.record.get('base_model')!r})"
        )
    if res.available is False:
        raise ValueError(f"{spec!r} is known to be absent on {box}")
    resolved = res.loader_name or res.canonical
    if raw_tail is not None:
        resolved = f"{resolved}{raw_tail}"
    warning = ""
    if family_ok is None:
        warning = "family unclassified"
    elif res.available is None:
        warning = "boxb availability unknown"
    return resolved, {
        "requested": spec,
        "canonical": res.canonical,
        "resolved": resolved,
        "strength": strength,
        "available": res.available,
        "family_ok": family_ok,
        "warning": warning,
    }


def lora_trace_for_box(specs: list[str] | None, box: str, family: str | None = None,
                       index_path=None) -> tuple[list[str], list[dict]]:
    """Normalize a list of LoRA specs for ``box`` and collect per-spec traces.

    Raises ``ValueError`` on the first ambiguity, known absence, or family mismatch.
    """
    out, traces = [], []
    for spec in specs or []:
        resolved, trace = resolve_for_box(spec, box, family, index_path)
        out.append(resolved)
        traces.append(trace)
    return out, traces


def compile_lora_trace(spec: str, family: str | None = None, index_path=None) -> tuple[dict, set[str]]:
    """Compile one LoRA spec for all boxes: trace + eligible boxes.

    Does not raise; returns an empty ``eligible`` set when the spec is impossible.
    Unindexed names are allowed on both boxes with a warning.
    """
    name, raw_tail, strength = parse_lora_spec(spec)
    try:
        res = resolve_lora(name, box="boxa", default_strength=strength, index_path=index_path)
    except AmbiguousRequestError as exc:
        trace = {
            "requested": spec,
            "name": name,
            "canonical": None,
            "resolved_boxa": None,
            "resolved_boxb": None,
            "strength": strength,
            "available_boxa": None,
            "available_boxb": None,
            "family_ok": None,
            "warning": f"ambiguous: {exc}",
        }
        return trace, set()
    if not res.record:
        trace = {
            "requested": spec,
            "name": name,
            "canonical": None,
            "resolved_boxa": None,
            "resolved_boxb": None,
            "strength": strength,
            "available_boxa": None,
            "available_boxb": None,
            "family_ok": None,
            "warning": "unindexed; live preflight is authority",
        }
        return trace, {"boxa", "boxb"}
    record = res.record
    canonical = record["file"]
    resolved_boxa = box_loader(record, "boxa") or canonical
    resolved_boxb = box_loader(record, "boxb") or canonical
    if raw_tail is not None:
        resolved_boxa = f"{resolved_boxa}{raw_tail}"
        resolved_boxb = f"{resolved_boxb}{raw_tail}"
    available_boxa = box_available(record, "boxa")
    available_boxb = box_available(record, "boxb")
    family_ok = record_family_status(record, family)
    eligible: set[str] = {"boxa"}
    if family is None:
        if available_boxb is not False:
            eligible.add("boxb")
    elif family_ok is not False:
        if available_boxb is not False:
            eligible.add("boxb")
    if family_ok is False:
        eligible.discard("boxa")
    if available_boxb is False:
        eligible.discard("boxb")
    warning = ""
    if family_ok is False:
        warning = f"family mismatch (expects {family}, record base={record.get('base_model')!r})"
    elif family_ok is None:
        warning = "family unclassified"
    elif available_boxb is None:
        warning = "boxb availability unknown"
    trace = {
        "requested": spec,
        "name": name,
        "canonical": canonical,
        "resolved_boxa": resolved_boxa,
        "resolved_boxb": resolved_boxb,
        "strength": strength,
        "available_boxa": available_boxa,
        "available_boxb": available_boxb,
        "family_ok": family_ok,
        "warning": warning,
    }
    return trace, eligible


# --------------------------------------------------------------------------- resource plan compilation


@dataclass
class ResourcePlan:
    """Compiled resource constraints for one route/brief."""

    family: str | None
    eligible_boxes: frozenset[str]
    base_trace: dict
    lora_traces: list[dict]
    style_lora_traces: list[dict]
    pose_ref: str | None
    problems: list[str]


def compile_resources(
    route_spec: dict,
    base_card: dict | None,
    loras: list[str] | None,
    style_loras: list[str] | None,
    pose_ref: str | None,
    requested_box: str | None,
    allowed_boxes: set[str] | frozenset[str] | None = None,
    index_path=None,
) -> ResourcePlan:
    """Compile resource constraints and eligible boxes for a route spec.

    ``route_spec`` is the per-intent engine table entry: it may carry
    ``resource_family``, ``boxa_only``, ``boxb_only``, and a non-None ``ckpt`` flag
    when the intent supports router ``--base``.

    ``requested_box`` may be ``"auto"``; it is treated as not-yet-concrete, so
    no-resource auto briefs validate cleanly.
    """
    family = route_spec.get("resource_family")
    eligible: set[str] = set(allowed_boxes) if allowed_boxes else {"boxa", "boxb"}
    if route_spec.get("boxa_only"):
        eligible = {"boxa"}
    if route_spec.get("boxb_only"):
        eligible = {"boxb"}
    problems: list[str] = []

    base_trace: dict = {}
    if base_card and route_spec.get("ckpt") is not None:
        boxes = set(base_card.get("boxes") or [])
        if not boxes:
            card_label = base_card.get("base") or base_card.get("stem") or "?"
            problems.append(f"--base: card {card_label!r} has no per-box registration")
        else:
            if requested_box and requested_box != "auto" and requested_box not in boxes:
                problems.append(f"--base: {base_card.get('base')!r} is not registered on box {requested_box}")
            if family:
                base_family = base_card.get("family") or base_card.get("ckpt")
                try:
                    if not family_compatible(base_family, family):
                        problems.append(f"--base: {base_card.get('base')!r} is not a {family}-family checkpoint")
                except UnknownFamilyError:
                    problems.append(f"--base: unknown family {family!r}")
        eligible &= boxes
        base_trace = {
            "stem": base_card.get("stem", ""),
            "card": base_card.get("base", ""),
            "ckpt": base_card.get("ckpt"),
            "boxes": sorted(boxes),
        }
    elif base_card and route_spec.get("ckpt") is None:
        problems.append("--base does not apply to this intent")

    lora_traces: list[dict] = []
    for spec_text in loras or []:
        trace, l_eligible = compile_lora_trace(spec_text, family, index_path)
        lora_traces.append(trace)
        if not l_eligible:
            problems.append(f"--lora {spec_text!r}: {trace['warning']}")
        else:
            eligible &= l_eligible

    style_lora_traces: list[dict] = []
    for spec_text in style_loras or []:
        trace, l_eligible = compile_lora_trace(spec_text, family, index_path)
        style_lora_traces.append(trace)
        if not l_eligible:
            problems.append(f"--style-lora {spec_text!r}: {trace['warning']}")
        else:
            eligible &= l_eligible

    if pose_ref:
        if requested_box == "boxa":
            problems.append("pose-ref requires box boxb (pose-library line is BOXB-only)")
        eligible &= {"boxb"}

    if not problems and not eligible:
        problems.append("no eligible box satisfies the resource constraints")
    elif requested_box and requested_box != "auto" and requested_box not in eligible and not problems:
        problems.append(f"box {requested_box} is incompatible with the selected resources")

    return ResourcePlan(
        family=family,
        eligible_boxes=frozenset(eligible),
        base_trace=base_trace,
        lora_traces=lora_traces,
        style_lora_traces=style_lora_traces,
        pose_ref=pose_ref,
        problems=problems,
    )


# --------------------------------------------------------------------------- search

def search_records(
    records: list[dict],
    query: str,
    *,
    top_k: int = 8,
    include_nsfw: bool = False,
    base: str | None = None,
    family: str | None = None,
    box: str | None = None,
    role: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministic weighted lexical search over ``records``.

    Returns a list of result dicts sorted by (exact match, score desc,
    canonical file, original index).  Non-empty queries with zero overlap do
    not return arbitrary catalog-order records.

    ``name_exact`` is true only when the normalized query equals the record's
    display name, canonical file, boxb alias, basename, or extensionless stem —
    not when the query merely appears inside a longer name.

    ``family`` is applied with ``record_family_compatible``, which falls back to
    conservative marker inference when ``base_model`` is Unknown / Other / empty.
    """
    qbase = (base or "").lower()

    results: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not include_nsfw and rec.get("content_rating") == "nsfw":
            continue
        if qbase and str(rec.get("base_model", "")).lower() != qbase:
            continue
        if family and not record_family_compatible(rec, family):
            continue
        if box:
            avail = box_available(rec, box)
            if avail is False:
                continue
        if role and classify_role(rec)[0] != role:
            continue

        score, hits, exact = _match_score(query, rec)
        if tokenize_query(query) and score <= 0.0:
            continue
        why = _why_for(query, rec, score, hits, exact)
        results.append({
            "record": rec,
            "score": score,
            "name_exact": exact,
            "why": why,
            "index": idx,
        })

    results.sort(key=lambda r: (not r["name_exact"], -r["score"], r["record"]["file"], r["index"]))
    return results[:top_k]


# --------------------------------------------------------------------------- classification / helpers

def strength_default(usage_tips: Any) -> float:
    """Extract a default LoRA strength from LoRA-Manager usage tips."""
    try:
        tips = json.loads(usage_tips) if isinstance(usage_tips, str) else (usage_tips or {})
        for k in ("strength", "strength_model", "weight", "lora_strength"):
            if k in tips:
                return round(float(tips[k]), 3)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return 0.8


def content_rating(lm: dict | None, tags: list[str], category: str | None = None) -> str:
    """SFW/NSFW/unknown rating from tags and LoRA-Manager preview level.

    Style LoRAs are rated by tags only (user decision 2026-09-16): a painting style is
    content-neutral, and the Civitai preview level reflects the uploader's sample images,
    not what an SFW prompt renders. Every other category keeps the preview-level rule."""
    low = {t.lower() for t in tags}
    if low & NSFW_TAGS:
        return "nsfw"
    if category == "style":
        return "sfw"
    if lm is not None:
        lvl = lm.get("preview_nsfw_level")
        if isinstance(lvl, (int, float)):
            if lvl >= 4:
                return "nsfw"
            if lvl in (1, 2):
                return "sfw"
    return "unknown"


def triggers_for(records: list[dict]) -> str:
    """Comma-joined, de-duplicated trigger tokens across the given records."""
    seen: dict[str, None] = {}
    for r in records:
        for entry in r.get("triggers") or []:
            for t in str(entry).split(","):
                t = t.strip()
                if t and t not in seen:
                    seen[t] = None
    return ", ".join(seen)


def trigger_tokens(record: dict) -> str:
    """Comma-joined, de-duplicated trigger tokens for a single record."""
    return triggers_for([record])


def classify_role(record: dict) -> tuple[str, int]:
    """Role for the three-step line: character / style / concept / pose / detail.

    Signal priority, most reliable first:
      1. catalog ``category`` (LoRA-Manager type label)
      2. type-signal ``tags``
      3. keyword heuristics over name + tags + triggers
      4. character as final fallback

    Returns (role, hits).  ``hits`` is a crude confidence: 2 = metadata,
    1 = keyword hit, 0 = fallback.
    """
    cat = str(record.get("category", "")).strip().lower()
    if cat in CATEGORY_ROLE:
        return (CATEGORY_ROLE[cat], 2)

    hay = " ".join([
        str(record.get("name", "")),
        *[str(t) for t in record.get("tags") or []],
        *[str(t) for t in record.get("triggers") or []],
    ]).lower()

    tags = {str(t).strip().lower() for t in record.get("tags") or []}
    for role in TAG_PRECEDENCE:
        if any(TAG_ROLE.get(t) == role for t in tags):
            if role == "concept" and any(sig in hay for sig in DETAIL_SIGNAL):
                return ("detail", 2)
            return (role, 2)

    scores = {role: sum(hay.count(kw) for kw in kws) for role, kws in ROLE_KEYWORDS.items()}
    best = max(scores, key=lambda role: scores[role])
    return (best, scores[best]) if scores[best] > 0 else ("character", 0)


# --------------------------------------------------------------------------- boxb matching

def match_boxb(files: list[str], boxb_names: set[str]) -> dict[str, str | None]:
    """Map each canonical (BOXA) LoRA name to its BOXB-registered name.

    Same weights register under different names per box, so:
    * exact name first,
    * then basename match only when the basename is unique on BOTH sides,
    * ambiguous or missing basenames map to ``None`` (never guess).
    """
    by_base: dict[str, list[str]] = {}
    for n in boxb_names:
        by_base.setdefault(_basename(n), []).append(n)

    src_base_counts: dict[str, int] = {}
    for f in files:
        b = _basename(f)
        src_base_counts[b] = src_base_counts.get(b, 0) + 1

    out: dict[str, str | None] = {}
    for f in files:
        if f in boxb_names:
            out[f] = f
            continue
        base = _basename(f)
        if src_base_counts[base] > 1:
            out[f] = None
            continue
        cands = by_base.get(base, [])
        out[f] = cands[0] if len(cands) == 1 else None
    return out


# --------------------------------------------------------------------------- embedding doc (for lora_catalog re-use)

def doc(r: dict) -> str:
    """Embedding text for a record: name + base + category + triggers + tags."""
    return " ".join([
        str(r.get("name", "")),
        str(r.get("base_model", "")),
        str(r.get("category", "")),
        *[str(t) for t in r.get("triggers") or []],
        *[str(t) for t in r.get("tags") or []],
    ]).strip()


def doc_hash(doc: str) -> str:
    """Stable short hash of a doc text for vector-cache invalidation."""
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- unified inventory / bases

UNIFIED_INDEX_PATH = REPO_ROOT / "boxa_model_index" / "index.json"
QUIRKS_PATH = REPO_ROOT / "scripts" / "model_quirks.json"

INVENTORY_KINDS = ("lora", "base", "checkpoint", "checkpoints", "vae", "controlnet",
                   "upscale_models", "clip_vision", "diffusion_models", "ipadapter")


class UnknownKindError(ValueError):
    """An inventory kind is not supported by the unified index or base registry."""


# --------------------------------------------------------------------------- quirks / registered bases

_QUIRKS_LOCK = threading.RLock()
_QUIRKS_CACHE: dict[Path, tuple[tuple[int, int], dict]] = {}


def _as_str_list(value: object) -> list[str]:
    """A quirks list field as clean strings; a bare string counts as one entry."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def load_quirks(quirks_path: Path | str | None = None) -> dict:
    """Parse ``model_quirks.json`` and cache on (mtime_ns, size)."""
    path = Path(quirks_path or QUIRKS_PATH)
    try:
        stat = path.stat()
    except FileNotFoundError:
        with _QUIRKS_LOCK:
            _QUIRKS_CACHE.pop(path, None)
        return {}
    key = (stat.st_mtime_ns, stat.st_size)
    with _QUIRKS_LOCK:
        cached = _QUIRKS_CACHE.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        _QUIRKS_CACHE[path] = (key, data)
        return data


def _base_family_match(card: dict, family: str | None) -> bool:
    """Whether a quirks base card is usable with ``family``."""
    fam = (family or "any").lower()
    if fam in ("", "any"):
        return True
    base = str(card.get("family") or card.get("ckpt") or card.get("base") or "").lower()
    if fam == "sdxl":
        return sdxl_family(base)
    if fam == "anima":
        return anima_family(base)
    if fam in ("sd15", "sd1.5"):
        return sd15_family(base)
    raise UnknownFamilyError(f"unknown family {family!r}; use sdxl | anima | sd15 | any")


def registered_bases(*, family: str | None = None, box: str | None = None,
                      base_id: str | None = None, quirks_path: Path | str | None = None,
                      include_unregistered: bool = False) -> list[dict]:
    """Registered, selectable base-model cards from ``model_quirks.json``.

    A card is selectable only when it has both an engine ``ckpt`` path and a
    non-empty ``boxes`` list.  The projection carries the same decision fields
    ``app/meta.py`` surfaces to the console.
    """
    data = load_quirks(quirks_path)
    out: list[dict] = []
    for stem, card in (data.get("bases") or {}).items():
        if base_id is not None and stem != base_id:
            continue
        selectable = bool(card.get("ckpt") and card.get("boxes"))
        if not include_unregistered and not selectable:
            continue
        if family and not _base_family_match({**card, "base": stem}, family):
            continue
        boxes = list(card.get("boxes") or [])
        if box and box != "auto" and box not in boxes:
            continue
        strength_cap = card.get("lora_strength_ok")
        family_value = str(card.get("family") or "illustrious").strip()
        if not family_value or family_value.lower() == "none":
            family_value = "illustrious"
        out.append({
            "id": stem,
            "kind": "base",
            "name": str(card.get("name") or stem),
            "ckpt": str(card.get("ckpt") or ""),
            "boxes": boxes,
            "family": family_value,
            "style": str(card.get("style") or ""),
            "strengths": _as_str_list(card.get("strengths")),
            "best_shots": _as_str_list(card.get("best_shots")),
            "avoid_shots": _as_str_list(card.get("avoid_shots")),
            "neg_add": str(card.get("neg_add") or ""),
            "no_freeu": bool(card.get("no_freeu")),
            "note": str(card.get("note") or card.get("notes") or ""),
            "lora_strength_ok": (float(strength_cap)
                                 if isinstance(strength_cap, (int, float))
                                 and not isinstance(strength_cap, bool) else None),
            "selectable": selectable,
            "tags": _as_str_list(card.get("tags")),
        })
    out.sort(key=lambda item: item["id"].lower())
    return out


# --------------------------------------------------------------------------- unified index

_UNIFIED_LOCK = threading.RLock()
_UNIFIED_CACHE: dict[Path, tuple[tuple[int, int], dict]] = {}


def load_unified_index(index_path: Path | str | None = None) -> dict:
    """Load ``boxa_model_index/index.json`` with the same (mtime_ns, size) cache as LoRA."""
    with _UNIFIED_LOCK:
        path = Path(index_path or UNIFIED_INDEX_PATH)
        try:
            stat = path.stat()
        except FileNotFoundError:
            _UNIFIED_CACHE.pop(path, None)
            return {}
        key = (stat.st_mtime_ns, stat.st_size)
        cached = _UNIFIED_CACHE.get(path)
        if cached and cached[0] == key:
            return cached[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        _UNIFIED_CACHE[path] = (key, data)
        return data


def _repo_relative(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise the absolute string."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def inventory_freshness(index: dict | None = None) -> dict[str, Any]:
    """Staleness summary for the unified asset index."""
    if index is None:
        index = load_unified_index()
    if not index:
        return {
            "present": False,
            "path": _repo_relative(UNIFIED_INDEX_PATH),
            "generated_at": None,
            "status": "missing",
            "stale_after_hours": 24,
            "counts": {},
        }
    gen = index.get("generated_at")
    ts = _parse_iso_ts(gen)
    hours = float(index.get("stale_after_hours", 24))
    status = str(index.get("refresh_status") or "unknown")
    if ts is not None and datetime.now(timezone.utc) - ts > timedelta(hours=hours):
        status = "stale"
    return {
        "present": bool(index),
        "path": _repo_relative(UNIFIED_INDEX_PATH),
        "generated_at": gen,
        "status": status,
        "stale_after_hours": hours,
        "counts": inventory_counts(index),
    }


def inventory_counts(index: dict | None = None) -> dict[str, Any]:
    """Asset counts from the unified index, with the LoRA count filled in."""
    if index is None:
        index = load_unified_index()
    counts: dict[str, Any] = dict(index.get("counts") or {})
    loras_raw = index.get("loras")
    loras = loras_raw if isinstance(loras_raw, dict) else {}
    counts["loras"] = loras.get("count")
    counts["loras_added"] = loras.get("added_count", 0)
    return counts


def lora_index_freshness(index_path: Path | str | None = None) -> dict[str, Any]:
    """Staleness summary for the LoRA catalog snapshot."""
    path = Path(index_path or INDEX_PATH)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"present": False, "path": _repo_relative(path),
                "generated_at": None, "status": "missing", "count": 0}
    generated_at = (datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"))
    try:
        snap = load_snapshot(path)
        count = len(snap.records)
    except (OSError, json.JSONDecodeError):
        count = 0
    return {
        "present": True,
        "path": _repo_relative(path),
        "generated_at": generated_at,
        "status": "ok",
        "count": count,
    }


CATALOG_REFRESH_CMD = ".venv/bin/python scripts/boxa_model_index.py refresh"
CATALOG_STATUS_CMD = "python3 scripts/boxa_model_index.py status"


def catalog_readiness_problems(
    *,
    allow_stale: bool = False,
    index_path: Path | str | None = None,
    unified_index_path: Path | str | None = None,
) -> list[str]:
    """Fail-closed blockers when the on-disk LoRA catalog is unsafe for local gen.

    Offline-only: reads mtime/generated_at fingerprints and LoRA counts; never
    probes BOXA.      When non-empty, callers must abort planning/render and tell the
    operator to run :data:`CATALOG_REFRESH_CMD` (or Studio ``POST /api/loras/refresh``).

    ``allow_stale`` skips only inventory-age staleness; missing catalog and
    catalog/inventory count drift still block.
    """
    lora_path = Path(index_path or INDEX_PATH)
    unified_path = Path(unified_index_path or UNIFIED_INDEX_PATH)
    problems: list[str] = []

    try:
        lora_path.stat()
        lora_present = True
    except FileNotFoundError:
        lora_present = False
        problems.append(
            f"LoRA catalog missing ({_repo_relative(lora_path)}); run: {CATALOG_REFRESH_CMD}"
        )

    unified = load_unified_index(unified_path)
    inv = inventory_freshness(unified)
    if not inv["present"]:
        problems.append(
            f"model inventory index missing ({inv['path']}); run: {CATALOG_REFRESH_CMD}"
        )
    elif inv["status"] == "stale" and not allow_stale:
        problems.append(
            f"model inventory index stale (generated {inv['generated_at']}); "
            f"run: {CATALOG_REFRESH_CMD}"
        )

    if lora_present:
        try:
            catalog_count = len(load_snapshot(lora_path).records)
        except (OSError, json.JSONDecodeError, TypeError, KeyError, UnicodeDecodeError, ValueError) as exc:
            catalog_count = None
            problems.append(
                f"LoRA catalog is unreadable ({_repo_relative(lora_path)}: {exc}); "
                f"run: {CATALOG_REFRESH_CMD}"
            )
        inv_count = (inv.get("counts") or {}).get("loras")
        loras_section = unified.get("loras") if unified else None
        if loras_section is None and inv["present"]:
            problems.append(
                f"model inventory index loras section missing or null "
                f"(catalog has {catalog_count} records); run: {CATALOG_REFRESH_CMD}"
            )
        elif catalog_count is not None and inv_count is None:
            problems.append(
                f"LoRA inventory index missing loras count "
                f"(catalog has {catalog_count} records); run: {CATALOG_REFRESH_CMD}"
            )
        elif (catalog_count is not None and inv_count is not None
                and catalog_count != inv_count):
            problems.append(
                f"LoRA catalog count ({catalog_count}) disagrees with inventory "
                f"index ({inv_count}); run: {CATALOG_REFRESH_CMD}"
            )

    return problems


# --------------------------------------------------------------------------- inventory search

def _asset_key(kind: str, assets: dict) -> str:
    """Map public singular kind names to the plural keys used in the unified index."""
    if kind in assets:
        return kind
    if kind == "checkpoint":
        return "checkpoints"
    return kind


def _inventory_records(index: dict, kind: str) -> list[dict]:
    """Normalize unified-index assets or the LoRA catalog into search records."""
    if kind == "lora":
        return load_snapshot(INDEX_PATH).records
    if kind == "base":
        return [{**b, "kind": "base"} for b in registered_bases() if b.get("selectable")]
    assets = index.get("assets", {})
    key = _asset_key(kind, assets)
    if key not in assets:
        raise UnknownKindError(f"unknown kind {kind!r}; supported: {', '.join(INVENTORY_KINDS)}")
    out: list[dict] = []
    for it in assets[key]:
        out.append({
            "name": str(it.get("name") or ""),
            "file": str(it.get("canonical_name") or it.get("id") or ""),
            "tags": list(it.get("tags") or []),
            "triggers": list(it.get("triggers") or []),
            "base_model": str(it.get("base_model") or ""),
            "content_rating": it.get("content_rating", "sfw"),
            "availability": it.get("availability", "confirmed"),
            "has_meta": bool(it.get("has_meta")),
            "preview": it.get("preview"),
            "kind": it.get("kind", kind),
            "_source_item": it,
        })
    return out


def _base_search_record(base: dict) -> dict:
    """Turn a registered base card into a record ``search_records`` can score."""
    return {
        "name": base.get("name") or base.get("id") or "",
        "file": base.get("id") or "",
        "tags": list(base.get("tags") or []),
        "triggers": [],
        "base_model": base.get("family") or "unknown",
        "content_rating": "sfw",
    }


def search_inventory(*, query: str, kind: str,
                     index: dict | None = None,
                     top_k: int = 8,
                     include_nsfw: bool = False,
                     family: str | None = None,
                     base: str | None = None,
                     box: str | None = None,
                     role: str | None = None,
                     archived: set[str] | None = None) -> list[dict[str, Any]]:
    """Deterministic search across LoRAs, registered bases, and support assets.

    ``kind`` may be ``lora``, ``base``, or any support kind present in the
    unified index (checkpoint, vae, controlnet, ...).  Results are compact,
    contain no absolute paths, and respect the SFW/NSFW zone unless
    ``include_nsfw`` is True.

    ``base`` is an exact ``base_model`` filter (used by the CLI); ``family`` is
    a route-family filter.
    """
    if kind not in INVENTORY_KINDS:
        raise UnknownKindError(f"unknown kind {kind!r}; supported: {', '.join(INVENTORY_KINDS)}")

    if kind in ("lora", "base"):
        search_family = family
        search_box = box if kind == "lora" else None
        search_base = base
    else:
        # Support assets in the unified index come from BOXA; box is not a
        # per-asset availability field, so we only use it to reject boxb requests.
        search_family = family
        search_box = "boxa" if box == "boxa" else None
        search_base = base

    if kind == "lora":
        records = _inventory_records(index or {}, kind)
    elif kind == "base":
        records = [_base_search_record(b) for b in registered_bases(family=family, box=box)]
    else:
        records = _inventory_records(index or load_unified_index(), kind)

    if archived is not None and kind == "lora":
        records = [r for r in records if r.get("file") not in archived]

    raw = search_records(
        records,
        query,
        top_k=top_k,
        include_nsfw=include_nsfw,
        base=search_base,
        family=search_family,
        box=search_box,
        role=role if kind == "lora" else None,
    )

    results: list[dict[str, Any]] = []
    for res in raw:
        rec = res["record"]
        if kind == "lora":
            results.append(compact_lora(rec, why=res["why"], score=res["score"],
                                        exact_match=res.get("name_exact", False), box=box))
        elif kind == "base":
            # recover the original projection by id
            base_id = rec.get("file")
            original = next((b for b in registered_bases(family=family, box=box)
                             if b.get("id") == base_id), None)
            if original is not None:
                results.append(original)
        else:
            item = rec.get("_source_item") or rec
            results.append(compact_inventory_item(item, score=res["score"]))
    return results


# --------------------------------------------------------------------------- compact projection


def _compact_trigger_preview(triggers: object, max_tokens: int = 8,
                             max_chars: int = 160) -> list[str]:
    """Return at most three short trigger strings, capped at 8 tokens / 160 chars total."""
    tokens: list[str] = []
    for phrase in triggers or []:
        for tok in str(phrase).split(","):
            tok = tok.strip()
            if not tok or tok in tokens:
                continue
            tokens.append(tok)
            if len(tokens) >= max_tokens:
                break
        if len(tokens) >= max_tokens:
            break
    if not tokens:
        return []
    n_chunks = min(3, len(tokens))
    k = (len(tokens) + n_chunks - 1) // n_chunks
    chunks: list[str] = []
    for i in range(n_chunks):
        chunks.append(", ".join(tokens[i * k:(i + 1) * k]))
    joined = ", ".join(chunks)
    if len(joined) > max_chars:
        flat = ", ".join(tokens)
        if len(flat) <= max_chars:
            return [flat]
        trimmed = flat[:max_chars - 3]
        if "," in trimmed:
            trimmed = trimmed.rsplit(",", 1)[0]
        return [trimmed + "..."]
    return chunks


def _synthesize_why(record: dict, score: float | None) -> str:
    bits: list[str] = []
    if score is not None:
        bits.append(f"score {score}")
    if record.get("base_model"):
        bits.append(f"base {record['base_model']}")
    if record.get("triggers"):
        bits.append("triggers " + ", ".join(str(t) for t in record["triggers"][:2]))
    return "; ".join(bits) or "match"


def compact_lora(record: dict, *, box: str | None = None, compact: bool = True,
                 why: str | None = None, score: float | None = None,
                 exact_match: bool = False) -> dict[str, Any]:
    """Compact, safe projection of a LoRA record for the public API.

    ``canonical_name`` is always the stable BOXA canonical name.  ``load_name`` is the
    concrete loader name for the requested ``box`` (boxa or boxb); ``box_names`` carries
    both.  ``available_boxb`` is ``None`` when the BOXB key is absent, ``False`` when
    ``boxb_file`` is empty or explicitly null, and ``True`` when it is a non-empty string.
    """
    boxa_name = record.get("file")
    boxb_name = record.get("boxb_file") if "boxb_file" in record else None
    s = record.get("strength_default")
    role, role_conf = classify_role(record)
    why = why or _synthesize_why(record, score)
    if compact and len(why) > 120:
        why = why[:117] + "..."
    out = {
        "canonical_name": boxa_name,
        "load_name": box_loader(record, box),
        "name": record.get("name"),
        "kind": "lora",
        "score": score,
        "exact_match": exact_match,
        "role": role,
        "role_confidence": role_conf,
        "why": why,
        "triggers": _compact_trigger_preview(record.get("triggers"), max_tokens=8, max_chars=160),
        "recommended_strengths": {"model": s, "clip": s} if s is not None else None,
        "base_model": record.get("base_model"),
        "category": record.get("category"),
        "content_rating": record.get("content_rating"),
        "availability": "confirmed" if record.get("has_meta") else "unconfirmed",
        "has_meta": bool(record.get("has_meta")),
        "available_boxa": True,
        "available_boxb": box_available(record, "boxb"),
        "box_names": {"boxa": boxa_name, "boxb": boxb_name},
    }
    return out


def compact_inventory_item(item: dict, score: float | None = None) -> dict[str, Any]:
    """Compact projection of a unified-index support asset."""
    it = item.get("_source_item") or item
    return {
        "canonical_name": it.get("canonical_name"),
        "name": it.get("name"),
        "kind": it.get("kind"),
        "score": score,
        "base_model": it.get("base_model"),
        "content_rating": it.get("content_rating"),
        "availability": it.get("availability", "confirmed"),
        "has_meta": bool(it.get("has_meta")),
        "tags": list(it.get("tags") or [])[:5],
        "triggers": [],
        "recommended_strengths": None,
    }
