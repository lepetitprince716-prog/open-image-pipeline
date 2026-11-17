"""Official Danbooru TagDB: fetch, pin, resolve, validate.

JSON owns the lexicon (SHA-pinned extract + overlay). This module owns behavior:
HTTP walk, SHA/freshness gates, alias hop, implication-add, morphology, dropped_tags.

Algorithm (normalize / morph / extract / category sort / to_prompt) follows the
MIT TagDB in ChrisJohnson89/ComfyUI-NeuralBooru ``tags.py``. Rows do **not**
come from that project's ``danbooru.csv`` (tagcomplete SHA ``7a3cba59``). Rows
come from ``https://danbooru.donmai.us`` JSON API extracts we pin ourselves.

Agents call the CLI; they should not edit this file to add a rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from loramake.config import ROOT

SCHEMA_OVERLAY = "loramake.tagdb-overlay/v1"
SCHEMA_MANIFEST = "loramake.tagdb-extract/v1"
SCHEMA_FETCH_PLAN = "loramake.tagdb-fetch-plan/v1"

DANBOORU = "https://danbooru.donmai.us"
USER_AGENT = os.environ.get(
    "DANBOORU_USER_AGENT",
    "loramake-tagdb/0.1 (+https://github.com/lepetitprince716-prog/open-image-pipeline)",
)
RATE_SECONDS = 1.0
STALE_AFTER_DAYS = 7
PAGE_LIMIT = 1000
ALIAS_HOP_LIMIT = 8
IMPLICATION_HOP_LIMIT = 8
ENV_TAGDB = "LORAMAKE_TAGDB"

DEFAULT_TAGDB_DIR = ROOT / "outputs" / "_cache" / "tagdb"
OVERLAY_PATH = ROOT / "configs" / "tagdb" / "overlay.json"

DUMP_URLS = (
    f"{DANBOORU}/data/tags.csv.gz",
    f"{DANBOORU}/data/tag_aliases.csv.gz",
    f"{DANBOORU}/data/tag_implications.csv.gz",
)

TAG_FIELDS = "id,name,category,post_count,is_deprecated"
ALIAS_FIELDS = "id,antecedent_name,consequent_name,status"
IMPLICATION_FIELDS = "id,antecedent_name,consequent_name,status"

WALKS: tuple[dict[str, Any], ...] = (
    {
        "name": "tags_used",
        "endpoint": "/tags.json",
        "params": {"search[hide_empty]": "yes", "only": TAG_FIELDS},
        "artifact": "tags",
    },
    {
        "name": "tags_deprecated",
        "endpoint": "/tags.json",
        "params": {"search[is_deprecated]": "yes", "only": TAG_FIELDS},
        "artifact": "tags",
    },
    {
        "name": "aliases_active",
        "endpoint": "/tag_aliases.json",
        "params": {"search[status]": "active", "only": ALIAS_FIELDS},
        "artifact": "aliases",
    },
    {
        "name": "implications_active",
        "endpoint": "/tag_implications.json",
        "params": {"search[status]": "active", "only": IMPLICATION_FIELDS},
        "artifact": "implications",
    },
)

CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
CHARACTER_COPYRIGHT = frozenset({3, 4})
_CATEGORY_ORDER = {4: 1, 3: 2, 1: 3, 0: 4, 5: 5}

_WS = re.compile(r"\s+")
_HAS_LETTER = re.compile(r"[a-z]")
_COUNT_TAG = re.compile(
    r"^(?:\d+(?:girls?|boys?|others?)|solo|multiple_girls|multiple_boys|multiple_others)$"
)

# Engine-owned wraps. Authors never write these tokens. Anima must stay tags_only.
WRAP_TEMPLATES = {
    "tags_only": "{prompt}",
    "illustrious": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, {prompt}",
    "pony": "score_9, score_8_up, score_7_up, {prompt}",
}

# Zoe 2026-09-11 call #7: this module must never inject these.
AGE_INJECT_FORBIDDEN = frozenset({
    "mature female",
    "mature_female",
    "adult woman",
    "adult_woman",
    "onee-san",
    "oneesan",
    "onee_san",
})

EXIT_OK, EXIT_INVALID, EXIT_NET = 0, 2, 4

FetchFn = Callable[[str, dict[str, str]], Any]


class TagDBError(ValueError):
    """Extract missing, SHA mismatch, stale, or a refused TagDB operation."""


@dataclass(frozen=True)
class TagRow:
    name: str
    category: int
    post_count: int
    is_deprecated: bool = False
    tag_id: int | None = None


@dataclass(frozen=True)
class ValidateResult:
    prompt: str
    kept: tuple[str, ...]
    dropped_tags: tuple[str, ...]
    rewrites: tuple[dict[str, str], ...]
    canonical: tuple[str, ...]


# --------------------------------------------------------------------------- paths / time


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(moment: datetime | None = None) -> str:
    value = moment or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def tagdb_dir() -> Path:
    raw = os.environ.get(ENV_TAGDB)
    if raw is not None and raw.strip():
        return Path(raw.strip())
    return DEFAULT_TAGDB_DIR


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# --------------------------------------------------------------------------- normalize / surface


def normalize(tag: str) -> str:
    """Canonical form for matching: lowercase, unescaped, underscored."""
    text = tag.strip().lower()
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = _WS.sub(" ", text).strip()
    return text.replace(" ", "_")


def to_prompt(tag: str) -> str:
    """Render a canonical tag for the prompt: spaces, escaped parens.

    Emoticon-style tags (no letters, e.g. ``>_<``) keep underscores.
    """
    text = tag if not _HAS_LETTER.search(tag) else tag.replace("_", " ")
    return text.replace("(", "\\(").replace(")", "\\)")


def morph_variants(norm: str) -> list[str]:
    """Candidate stems: smirking→smirk, posing→pose, cats→cat, eye→eyes."""
    out: list[str] = []
    if "_" in norm:
        head, _, tail = norm.rpartition("_")
        prefix = head + "_"
    else:
        prefix, tail = "", norm
    for suf, repl in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", "")):
        if tail.endswith(suf) and len(tail) - len(suf) >= 3:
            out.append(prefix + tail[: -len(suf)] + repl)
    if not tail.endswith("s") and len(tail) >= 3:
        out.append(prefix + tail + "s")
    return out


def wrap_prompt(prompt: str, wrap: str = "tags_only") -> str:
    template = WRAP_TEMPLATES.get(wrap)
    if template is None:
        raise TagDBError(f"unknown wrap {wrap!r}; expected tags_only|illustrious|pony")
    return template.replace("{prompt}", prompt)


def _sort_group(norm: str, category: int) -> int:
    if category == 0 and _COUNT_TAG.match(norm):
        return 0
    return _CATEGORY_ORDER.get(category, 4)


def _rewrite(src: str, dst: str, kind: str) -> dict[str, str]:
    return {"from": src, "to": dst, "kind": kind}


# --------------------------------------------------------------------------- overlay / extract I/O


def load_overlay(path: Path | None = None) -> dict[str, str]:
    overlay_path = Path(path) if path else OVERLAY_PATH
    try:
        payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TagDBError(f"TagDB overlay missing: {overlay_path}") from exc
    except json.JSONDecodeError as exc:
        raise TagDBError(f"TagDB overlay invalid JSON {overlay_path}: {exc}") from exc
    if payload.get("schema") != SCHEMA_OVERLAY:
        raise TagDBError(f"TagDB overlay schema must be {SCHEMA_OVERLAY}")
    mapping: dict[str, str] = {}
    for item in payload.get("rewrites") or []:
        if not isinstance(item, dict):
            continue
        src = normalize(str(item.get("from") or ""))
        dst = normalize(str(item.get("to") or ""))
        if src and dst:
            mapping[src] = dst
    return mapping


def _dump_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_extract(
    dest: Path,
    tags: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    implications: list[dict[str, Any]],
    *,
    fetched_at: str | None = None,
    walks: Iterable[dict[str, Any]] = WALKS,
    x_git_hash: str | None = None,
    source_host: str = DANBOORU,
) -> dict[str, Any]:
    """Write tags/aliases/implications JSONL plus a SHA-256 manifest. Atomic replace."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    named = (("tags", tags), ("aliases", aliases), ("implications", implications))
    try:
        for name, rows in named:
            rel = f"{name}.jsonl"
            text = _dump_jsonl(rows)
            (tmp / rel).write_text(text, encoding="utf-8")
            artifacts[name] = {
                "path": rel,
                "sha256": sha256_bytes(text.encode("utf-8")),
                "row_count": len(rows),
            }
        manifest = {
            "schema": SCHEMA_MANIFEST,
            "source_host": source_host,
            "method": "api",
            "fetched_at": fetched_at or utc_stamp(),
            "stale_after_days": STALE_AFTER_DAYS,
            "walks": list(walks),
            "x_git_hash": x_git_hash,
            "artifacts": artifacts,
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            old = dest.with_name(dest.name + ".old")
            if old.exists():
                shutil.rmtree(old)
            dest.rename(old)
            try:
                tmp.rename(dest)
            except Exception:
                old.rename(dest)
                raise
            shutil.rmtree(old)
        else:
            tmp.rename(dest)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def _verify_manifest(directory: Path, manifest: dict[str, Any], *, now: datetime, stale_after_days: int | None) -> None:
    if manifest.get("schema") != SCHEMA_MANIFEST:
        raise TagDBError(f"TagDB manifest schema must be {SCHEMA_MANIFEST}")
    if manifest.get("source_host") != DANBOORU:
        raise TagDBError(f"TagDB source_host must be {DANBOORU}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise TagDBError("TagDB manifest missing artifacts")
    for name in ("tags", "aliases", "implications"):
        meta = artifacts.get(name)
        if not isinstance(meta, dict):
            raise TagDBError(f"TagDB manifest missing artifact {name}")
        path = directory / str(meta.get("path") or f"{name}.jsonl")
        if not path.is_file():
            raise TagDBError(f"TagDB artifact missing: {path}")
        digest = sha256_file(path)
        expected = str(meta.get("sha256") or "")
        if digest != expected:
            raise TagDBError(f"TagDB SHA-256 mismatch for {path.name}: got {digest}, manifest {expected}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        count = int(meta.get("row_count") or -1)
        if count != len(lines):
            raise TagDBError(f"TagDB row_count mismatch for {path.name}: file {len(lines)}, manifest {count}")
    fetched_raw = manifest.get("fetched_at")
    if not fetched_raw:
        raise TagDBError("TagDB manifest missing fetched_at")
    fetched = parse_utc(str(fetched_raw))
    if stale_after_days is not None and now - fetched > timedelta(days=stale_after_days):
        raise TagDBError(
            f"TagDB extract stale (fetched_at {fetched_raw}; max {stale_after_days}d). "
            "Re-run: python3 -m loramake.tags fetch --run"
        )


# --------------------------------------------------------------------------- TagDB


@dataclass
class TagDB:
    rows: dict[str, TagRow]
    aliases: dict[str, str]
    implications: dict[str, tuple[str, ...]]
    overlay: dict[str, str]
    _fuzzy_keys: list[str] | None = field(default=None, repr=False)

    @classmethod
    def from_parts(
        cls,
        rows: dict[str, TagRow] | Iterable[TagRow],
        aliases: dict[str, str],
        implications: dict[str, Iterable[str]],
        overlay: dict[str, str],
    ) -> TagDB:
        mapped: dict[str, TagRow] = {}
        if isinstance(rows, dict):
            mapped = {normalize(key): value for key, value in rows.items()}
        else:
            for row in rows:
                mapped[row.name] = row
        alias_map = {normalize(src): normalize(dst) for src, dst in aliases.items() if src and dst}
        impl_map = {
            normalize(src): tuple(dict.fromkeys(normalize(item) for item in dests if item))
            for src, dests in implications.items()
        }
        overlay_map = {normalize(src): normalize(dst) for src, dst in overlay.items() if src and dst}
        return cls(rows=mapped, aliases=alias_map, implications=impl_map, overlay=overlay_map)

    def __len__(self) -> int:
        return len(self.rows)

    def row_for(self, candidate: str) -> TagRow | None:
        return self.rows.get(normalize(candidate))

    def usable(self, name: str) -> bool:
        row = self.rows.get(name)
        return bool(row) and not row.is_deprecated and row.post_count > 0

    def _alias_hop(self, name: str) -> str | None:
        seen: set[str] = set()
        current = name
        for _ in range(ALIAS_HOP_LIMIT):
            nxt = self.aliases.get(current)
            if not nxt or nxt in seen:
                break
            seen.add(nxt)
            current = nxt
            if self.usable(current):
                return current
        return current if self.usable(current) else None

    def _hit(self, norm: str) -> tuple[str | None, str | None]:
        """Return (canonical, first_kind) for an already-normalized token."""
        overlay_to = self.overlay.get(norm)
        if overlay_to and self.usable(overlay_to):
            return overlay_to, "overlay"
        if self.usable(norm):
            return norm, None
        hopped = self._alias_hop(norm)
        if hopped:
            return hopped, "alias"
        return None, None

    def resolve(self, candidate: str, *, fuzzy_cutoff: float = 0.0) -> tuple[str | None, list[dict[str, str]]]:
        """Map one token to a usable canonical name. Default: no fuzzy."""
        norm = normalize(candidate)
        if not norm:
            return None, []
        hit, kind = self._hit(norm)
        if hit:
            rewrites = [_rewrite(norm, hit, kind)] if kind and hit != norm else []
            return hit, rewrites
        for variant in morph_variants(norm):
            hit, kind = self._hit(variant)
            if hit:
                rewrites = [_rewrite(norm, hit, "morphology")]
                if kind and kind != "morphology" and variant != hit:
                    rewrites.append(_rewrite(variant, hit, kind))
                return hit, rewrites
        if fuzzy_cutoff and fuzzy_cutoff > 0:
            import difflib

            if self._fuzzy_keys is None:
                self._fuzzy_keys = [name for name in self.rows if self.usable(name)]
            close = difflib.get_close_matches(norm, self._fuzzy_keys, n=1, cutoff=fuzzy_cutoff)
            if close:
                return close[0], [_rewrite(norm, close[0], "fuzzy")]
        return None, []

    def extract(self, candidate: str, *, fuzzy_cutoff: float = 0.0) -> list[tuple[str, list[dict[str, str]]]]:
        """Greedy longest-match salvage. Single-word candidates return []."""
        tokens = [token for token in re.split(r"\s+", candidate.strip().lower()) if token]
        if len(tokens) < 2:
            return []
        found: list[tuple[str, list[dict[str, str]]]] = []
        index = 0
        while index < len(tokens):
            matched = False
            for end in range(len(tokens), index, -1):
                sub = "_".join(tokens[index:end])
                if end - index > 1:
                    hit, rewrites = self.resolve(sub, fuzzy_cutoff=fuzzy_cutoff)
                else:
                    hit, kind = self._hit(sub)
                    rewrites = [_rewrite(sub, hit, kind)] if hit and kind else []
                if hit:
                    found.append((hit, rewrites + [_rewrite(candidate, hit, "extract")]))
                    index = end
                    matched = True
                    break
            if not matched:
                index += 1
        return found

    def implied(self, name: str) -> list[str]:
        """Walk implication consequents. Hop limit is shortest-path depth, not |seen|."""
        out: list[str] = []
        emitted: set[str] = set()
        best_depth: dict[str, int] = {name: 0}
        queue: deque[tuple[str, int]] = deque(
            (item, 1) for item in self.implications.get(name, ())
        )
        hops = 0
        while queue and hops < 10_000:
            hops += 1
            item, depth = queue.popleft()
            if depth > IMPLICATION_HOP_LIMIT:
                continue
            recorded = best_depth.get(item)
            if recorded is not None and recorded <= depth:
                continue
            best_depth[item] = depth
            if item != name and self.usable(item) and item not in emitted:
                emitted.add(item)
                out.append(item)
            if depth < IMPLICATION_HOP_LIMIT:
                queue.extend((child, depth + 1) for child in self.implications.get(item, ()))
        return out

    def validate(
        self,
        text: str,
        *,
        domain: str = "sdxl",
        wrap: str = "tags_only",
        min_post_count: int = 0,
        fuzzy_cutoff: float = 0.0,
        max_tags: int = 0,
        exclude_categories: Iterable[int] | None = None,
    ) -> ValidateResult:
        """Validate a comma-separated tag string. Always returns dropped_tags."""
        if domain not in {"sdxl", "anima"}:
            raise TagDBError(f"unknown domain {domain!r}; expected sdxl|anima")
        if domain == "anima":
            if wrap != "tags_only":
                raise TagDBError("Anima domain forbids IL/Pony wrap")
            strict = False
            do_extract = False
            sort_tags = False
        else:
            strict = True
            do_extract = True
            sort_tags = True

        exclude = set(exclude_categories or ())
        kept: list[tuple[str, str, int]] = []  # kind, token, category
        dropped: list[str] = []
        rewrites: list[dict[str, str]] = []
        seen: set[str] = set()

        def accept_tag(norm: str) -> bool:
            row = self.rows[norm]
            if row.category in exclude:
                return False
            if min_post_count and row.post_count < min_post_count:
                return False
            if norm in seen:
                return True
            if max_tags and sum(kind == "tag" for kind, _tok, _cat in kept) >= max_tags:
                return False
            seen.add(norm)
            kept.append(("tag", norm, row.category))
            return True

        for raw in (part.strip() for part in text.split(",")):
            if not raw:
                continue
            if max_tags and sum(kind == "tag" for kind, _tok, _cat in kept) >= max_tags and strict:
                break
            hit, hops = self.resolve(raw, fuzzy_cutoff=fuzzy_cutoff)
            if hit is not None:
                if accept_tag(hit):
                    rewrites.extend(hops)
                else:
                    dropped.append(raw)
                continue
            if do_extract:
                salvaged = self.extract(raw, fuzzy_cutoff=fuzzy_cutoff)
                accepted_any = False
                for sub, sub_rewrites in salvaged:
                    if accept_tag(sub):
                        accepted_any = True
                        rewrites.extend(sub_rewrites)
                if accepted_any:
                    continue
            if strict:
                dropped.append(raw)
            else:
                key = normalize(raw) or raw
                if key not in seen:
                    seen.add(key)
                    kept.append(("raw", raw, 0))

        for kind, token, _cat in list(kept):
            if kind != "tag":
                continue
            for consequent in self.implied(token):
                if accept_tag(consequent):
                    rewrites.append(_rewrite(token, consequent, "implication"))

        if sort_tags:
            tags = [(tok, cat) for kind, tok, cat in kept if kind == "tag"]
            tags.sort(key=lambda item: _sort_group(item[0], item[1]))
            kept = [("tag", tok, cat) for tok, cat in tags]

        surfaces: list[str] = []
        canonical: list[str] = []
        for kind, token, _cat in kept:
            if kind == "tag":
                surfaces.append(to_prompt(token))
                canonical.append(token)
            else:
                surfaces.append(token)
                canonical.append(normalize(token) or token)
        prompt = ", ".join(surfaces)
        prompt = wrap_prompt(prompt, wrap)
        return ValidateResult(
            prompt=prompt,
            kept=tuple(surfaces),
            dropped_tags=tuple(dropped),
            rewrites=tuple(rewrites),
            canonical=tuple(canonical),
        )


def _rows_from_jsonl(items: list[dict[str, Any]]) -> dict[str, TagRow]:
    rows: dict[str, TagRow] = {}
    for item in items:
        name = normalize(str(item.get("name") or ""))
        if not name:
            continue
        try:
            category = int(item.get("category") or 0)
        except (TypeError, ValueError):
            category = 0
        try:
            count = int(item.get("post_count") or 0)
        except (TypeError, ValueError):
            count = 0
        deprecated = bool(item.get("is_deprecated"))
        tag_id = item.get("id")
        try:
            tag_id_int = int(tag_id) if tag_id is not None else None
        except (TypeError, ValueError):
            tag_id_int = None
        existing = rows.get(name)
        # Prefer a usable row over a deprecated duplicate from the second walk.
        if existing and not existing.is_deprecated and existing.post_count > 0 and (deprecated or count <= 0):
            continue
        rows[name] = TagRow(
            name=name, category=category, post_count=count, is_deprecated=deprecated, tag_id=tag_id_int
        )
    return rows


def _aliases_from_jsonl(items: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        status = str(item.get("status") or "active")
        if status != "active":
            continue
        src = normalize(str(item.get("antecedent_name") or ""))
        dst = normalize(str(item.get("consequent_name") or ""))
        if src and dst and src not in mapping:
            mapping[src] = dst
    return mapping


def _implications_from_jsonl(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for item in items:
        status = str(item.get("status") or "active")
        if status != "active":
            continue
        src = normalize(str(item.get("antecedent_name") or ""))
        dst = normalize(str(item.get("consequent_name") or ""))
        if src and dst and dst not in mapping.get(src, []):
            mapping.setdefault(src, []).append(dst)
    return mapping


_TAGDB_LOCK = threading.RLock()
_TAGDB_CACHE: dict[tuple[str, str], tuple[tuple[tuple[int, int], ...], TagDB, datetime, tuple[Path, ...]]] = {}


def _file_sig(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def _tagdb_paths(dest: Path, manifest: dict[str, Any], overlay: Path) -> tuple[Path, ...]:
    artifacts = manifest.get("artifacts") or {}
    return (
        dest / "manifest.json",
        dest / str((artifacts.get("tags") or {}).get("path") or "tags.jsonl"),
        dest / str((artifacts.get("aliases") or {}).get("path") or "aliases.jsonl"),
        dest / str((artifacts.get("implications") or {}).get("path") or "implications.jsonl"),
        overlay,
    )


def _raise_if_stale(fetched_at: datetime, *, now: datetime, stale_after_days: int | None) -> None:
    if stale_after_days is None:
        return
    if now - fetched_at > timedelta(days=stale_after_days):
        raise TagDBError(
            f"TagDB extract stale (fetched_at {fetched_at.strftime('%Y-%m-%dT%H:%M:%SZ')}; "
            f"max {stale_after_days}d). Re-run: python3 -m loramake.tags fetch --run"
        )


def invalidate_tagdb(directory: Path | None = None) -> None:
    """Drop cached TagDB objects so the next load re-reads the extract."""
    with _TAGDB_LOCK:
        if directory is None:
            _TAGDB_CACHE.clear()
            return
        dest = str(Path(directory).resolve())
        for key in [k for k in _TAGDB_CACHE if k[0] == dest]:
            _TAGDB_CACHE.pop(key, None)


def load_tagdb(
    directory: Path | None = None,
    *,
    overlay_path: Path | None = None,
    now: datetime | None = None,
    stale_after_days: int | None = STALE_AFTER_DAYS,
) -> TagDB:
    """Load a pinned extract. Fail-closed on missing / SHA mismatch / stale.

    Cached on (directory, overlay) keyed by (mtime_ns, size) of the manifest,
    three JSONL artifacts, and the overlay. A hit re-evaluates staleness with
    the caller's ``now`` / ``stale_after_days`` without re-hashing.
    """
    dest = Path(directory) if directory else tagdb_dir()
    overlay = Path(overlay_path) if overlay_path else OVERLAY_PATH
    clock = now or utc_now()
    slot = (str(dest.resolve()), str(overlay.resolve()))
    with _TAGDB_LOCK:
        cached = _TAGDB_CACHE.get(slot)
        if cached is not None:
            sigs, db, fetched_at, paths = cached
            try:
                current = tuple(_file_sig(path) for path in paths)
            except OSError:
                current = None
            if current == sigs:
                _raise_if_stale(fetched_at, now=clock, stale_after_days=stale_after_days)
                return db
        manifest_path = dest / "manifest.json"
        if not manifest_path.is_file():
            raise TagDBError(f"TagDB extract missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TagDBError(f"TagDB manifest invalid JSON: {exc}") from exc
        _verify_manifest(dest, manifest, now=clock, stale_after_days=stale_after_days)
        tags = _read_jsonl(dest / manifest["artifacts"]["tags"]["path"])
        aliases = _read_jsonl(dest / manifest["artifacts"]["aliases"]["path"])
        implications = _read_jsonl(dest / manifest["artifacts"]["implications"]["path"])
        db = TagDB.from_parts(
            _rows_from_jsonl(tags),
            _aliases_from_jsonl(aliases),
            _implications_from_jsonl(implications),
            load_overlay(overlay),
        )
        fetched_raw = str(manifest.get("fetched_at") or "")
        fetched_at = parse_utc(fetched_raw)
        paths = _tagdb_paths(dest, manifest, overlay)
        sigs = tuple(_file_sig(path) for path in paths)
        _TAGDB_CACHE[slot] = (sigs, db, fetched_at, paths)
        return db


def load_tagdb_if_present(
    directory: Path | None = None,
    *,
    overlay_path: Path | None = None,
    now: datetime | None = None,
    stale_after_days: int | None = STALE_AFTER_DAYS,
) -> TagDB | None:
    """Return None only when no extract is installed. Corrupt/stale still raises."""
    dest = Path(directory) if directory else tagdb_dir()
    if not (dest / "manifest.json").is_file():
        return None
    return load_tagdb(dest, overlay_path=overlay_path, now=now, stale_after_days=stale_after_days)


def validate_prompt(text: str, *, domain: str = "sdxl", wrap: str = "tags_only", tagdb: TagDB | None = None) -> ValidateResult:
    db = tagdb or load_tagdb()
    return db.validate(text, domain=domain, wrap=wrap)


def assist_payload(prompt: str, *, domain: str = "sdxl") -> dict[str, Any]:
    """Thin /api/assist return shape. Does not compose an IR (Phase B)."""
    db = load_tagdb_if_present()
    if db is None:
        return {
            "prompt": prompt,
            "dropped_tags": [],
            "rewrites": [],
            "ir": None,
            "prompt_domain": "sdxl_illustrious" if domain == "sdxl" else "anima_mixed",
            "loras_suggested": [],
            "tagdb_ready": False,
        }
    result = db.validate(prompt, domain=domain, wrap="tags_only")
    return {
        "prompt": result.prompt,
        "dropped_tags": list(result.dropped_tags),
        "rewrites": list(result.rewrites),
        "ir": None,
        "prompt_domain": "sdxl_illustrious" if domain == "sdxl" else "anima_mixed",
        "loras_suggested": [],
        "tagdb_ready": True,
    }


# --------------------------------------------------------------------------- fetch


def _http_get_json(endpoint: str, params: dict[str, str], *, last: list[float], retries: int = 3) -> tuple[Any, dict[str, str]]:
    url = f"{DANBOORU}{endpoint}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(1, retries + 1):
        wait = RATE_SECONDS - (time.monotonic() - last[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers=headers)
        last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                hdrs = {key.lower(): value for key, value in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(max(5, int(exc.headers.get("Retry-After", "5") or 5)))
                continue
            raise TagDBError(f"GET {endpoint} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise TagDBError(f"GET {endpoint} failed: {exc}") from exc
        return json.loads(body.decode("utf-8")), hdrs
    raise TagDBError(f"GET {endpoint} failed after {retries} attempts")


def paginate(fetch: FetchFn, endpoint: str, params: dict[str, str], *, limit: int = PAGE_LIMIT) -> Iterator[dict[str, Any]]:
    """Walk ``page=b<id>`` until a short or empty page. Does not use numeric page=N."""
    page_id: int | None = None
    while True:
        query = dict(params)
        query["limit"] = str(limit)
        if page_id is not None:
            query["page"] = f"b{page_id}"
        batch = fetch(endpoint, query)
        if isinstance(batch, tuple):
            batch = batch[0]
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            if isinstance(item, dict):
                yield item
        last_id = batch[-1].get("id") if isinstance(batch[-1], dict) else None
        try:
            last_id_int = int(last_id)
        except (TypeError, ValueError):
            break
        if len(batch) < limit:
            break
        page_id = last_id_int


def _compact_tag(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    return {
        "id": item.get("id"),
        "name": name,
        "category": int(item.get("category") or 0),
        "post_count": int(item.get("post_count") or 0),
        "is_deprecated": bool(item.get("is_deprecated")),
    }


def _compact_edge(item: dict[str, Any]) -> dict[str, Any] | None:
    src = str(item.get("antecedent_name") or "").strip()
    dst = str(item.get("consequent_name") or "").strip()
    if not src or not dst:
        return None
    return {
        "id": item.get("id"),
        "antecedent_name": src,
        "consequent_name": dst,
        "status": str(item.get("status") or "active"),
    }


def fetch_extract(dest: Path, *, fetch: FetchFn | None = None) -> dict[str, Any]:
    """Walk the official JSON API and write a pinned extract under ``dest``."""
    last = [0.0]
    x_git_hash: str | None = None

    def default_fetch(endpoint: str, params: dict[str, str]) -> Any:
        nonlocal x_git_hash
        data, headers = _http_get_json(endpoint, params, last=last)
        if x_git_hash is None:
            x_git_hash = headers.get("x-git-hash")
        return data

    client = fetch or default_fetch
    buckets: dict[str, dict[str, dict[str, Any]]] = {"tags": {}, "aliases": {}, "implications": {}}
    for walk in WALKS:
        compact = _compact_tag if walk["artifact"] == "tags" else _compact_edge
        for item in paginate(client, walk["endpoint"], dict(walk["params"])):
            row = compact(item)
            if row is None:
                continue
            key = row["name"] if walk["artifact"] == "tags" else row["antecedent_name"] + "\0" + row["consequent_name"]
            if walk["artifact"] == "tags":
                existing = buckets["tags"].get(key)
                if existing and not existing["is_deprecated"] and existing["post_count"] > 0:
                    if row["is_deprecated"] or row["post_count"] <= 0:
                        continue
                buckets["tags"][key] = row
            else:
                buckets[walk["artifact"]].setdefault(key, row)
    tags = sorted(buckets["tags"].values(), key=lambda row: int(row.get("id") or 0))
    aliases = sorted(buckets["aliases"].values(), key=lambda row: int(row.get("id") or 0))
    implications = sorted(buckets["implications"].values(), key=lambda row: int(row.get("id") or 0))
    return write_extract(
        dest,
        tags,
        aliases,
        implications,
        walks=WALKS,
        x_git_hash=x_git_hash,
    )


def fetch_plan(dest: Path | None = None) -> dict[str, Any]:
    target = Path(dest) if dest else tagdb_dir()
    return {
        "schema": SCHEMA_FETCH_PLAN,
        "mode": "PLAN",
        "source_host": DANBOORU,
        "dest": str(target),
        "rate_seconds": RATE_SECONDS,
        "page": "b<id>",
        "limit": PAGE_LIMIT,
        "stale_after_days": STALE_AFTER_DAYS,
        "dumps": {
            "note": "Official CSV dumps returned HTTP 404 as of 2026-09-11; this fetcher walks the JSON API.",
            "urls": list(DUMP_URLS),
        },
        "walks": [dict(walk) for walk in WALKS],
        "overlay": str(OVERLAY_PATH),
        "run": "python3 -m loramake.tags fetch --run",
    }


# --------------------------------------------------------------------------- CLI


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _status_payload(directory: Path | None = None) -> tuple[dict[str, Any], int]:
    dest = Path(directory) if directory else tagdb_dir()
    manifest_path = dest / "manifest.json"
    payload: dict[str, Any] = {
        "schema": SCHEMA_MANIFEST,
        "dir": str(dest),
        "overlay": str(OVERLAY_PATH),
        "ready": False,
    }
    if not manifest_path.is_file():
        payload["reason"] = "missing"
        return payload, EXIT_INVALID
    try:
        db = load_tagdb(dest)
    except TagDBError as exc:
        payload["reason"] = str(exc)
        return payload, EXIT_INVALID
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update({
        "ready": True,
        "fetched_at": manifest.get("fetched_at"),
        "row_counts": {name: meta.get("row_count") for name, meta in manifest.get("artifacts", {}).items()},
        "tag_rows": len(db),
        "x_git_hash": manifest.get("x_git_hash"),
    })
    return payload, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="fail-closed readiness of the pinned extract")
    status.add_argument("--dir", type=Path)

    lookup = sub.add_parser("lookup", help="resolve one or more tokens")
    lookup.add_argument("tags", nargs="+")
    lookup.add_argument("--dir", type=Path)
    lookup.add_argument("--domain", choices=("sdxl", "anima"), default="sdxl")

    validate = sub.add_parser("validate", help="validate a comma-separated prompt")
    validate.add_argument("--prompt", required=True)
    validate.add_argument("--dir", type=Path)
    validate.add_argument("--domain", choices=("sdxl", "anima"), default="sdxl")
    validate.add_argument("--wrap", choices=tuple(WRAP_TEMPLATES), default="tags_only")

    fetch = sub.add_parser("fetch", help="print a fetch PLAN; --run writes the extract")
    fetch.add_argument("--run", action="store_true")
    fetch.add_argument("--dir", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            payload, code = _status_payload(args.dir)
            _print(payload)
            return code
        if args.command == "fetch":
            dest = args.dir or tagdb_dir()
            plan = fetch_plan(dest)
            if not args.run:
                _print(plan)
                return EXIT_OK
            manifest = fetch_extract(dest)
            _print({"mode": "RUN", "dest": str(dest), "manifest": manifest})
            return EXIT_OK
        db = load_tagdb(args.dir)
        if args.command == "lookup":
            items = []
            for tag in args.tags:
                hit, rewrites = db.resolve(tag)
                items.append({
                    "query": tag,
                    "canonical": hit,
                    "dropped": hit is None,
                    "rewrites": rewrites,
                })
            _print({"dropped_tags": [item["query"] for item in items if item["dropped"]], "tags": items})
            return EXIT_OK
        result = db.validate(args.prompt, domain=args.domain, wrap=args.wrap)
        _print({
            "prompt": result.prompt,
            "dropped_tags": list(result.dropped_tags),
            "rewrites": list(result.rewrites),
            "canonical": list(result.canonical),
            "domain": args.domain,
            "wrap": args.wrap,
        })
        return EXIT_OK
    except TagDBError as exc:
        _print({"error": str(exc), "dropped_tags": []})
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
