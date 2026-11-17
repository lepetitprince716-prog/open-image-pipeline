#!/usr/bin/env python3
"""BOXA LoRA catalog — index the LoRA library and select LoRAs by semantic context.

This module is now a thin adapter over ``loramake.model_resources`` for the
shared deterministic operations (lookup, role classification, trigger handling,
family compatibility).  It keeps the network/build/embedding paths that still
live here for tooling reasons.

Subcommands:
  build   Paginate the ComfyUI-LoRA-Manager API on BOXA and write lora_catalog/boxa_index.json.
  embed   Embed each index record and write lora_catalog/boxa_vectors.npz.
  select  Embed a context string and print the top-k LoRAs by cosine similarity.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from loramake import model_resources as _mr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxa_comfy import BOXA_HOST, BOXB_HOST, atomic_write_text  # shared atomic-write helper

CAT_DIR = ROOT / "lora_catalog"
INDEX_PATH = _mr.INDEX_PATH
VECTORS_PATH = CAT_DIR / "boxa_vectors.npz"
EMBED_MODEL = os.environ.get("LORA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Backward-compatible re-exports of the shared catalog helpers.
NSFW_TAGS = _mr.NSFW_TAGS
CATEGORY_ROLE = _mr.CATEGORY_ROLE
TAG_ROLE = _mr.TAG_ROLE
TAG_PRECEDENCE = _mr.TAG_PRECEDENCE
DETAIL_SIGNAL = _mr.DETAIL_SIGNAL
ROLE_KEYWORDS = _mr.ROLE_KEYWORDS

parse_spec = _mr.parse_spec
record_of = _mr.record_of
base_model_of = _mr.base_model_of
triggers_for = _mr.triggers_for
trigger_tokens = _mr.trigger_tokens
classify_role = _mr.classify_role
strength_default = _mr.strength_default
content_rating = _mr.content_rating
match_boxb = _mr.match_boxb

# Backward-compatible underscore names used by tests and sibling scripts.
_strength_default = strength_default
_content_rating = content_rating
_match_boxb = match_boxb


def _curl(path: str, max_time: int = 20, host: str | None = None) -> str:
    """Fetch ``path`` from a ComfyUI / LoRA-Manager host with curl.

    ``--max-time`` caps the transfer; the ``timeout`` argument to ``subprocess.run``
    is a slightly larger backstop so a wedged curl process does not hang this tool.
    Returns the raw body text, or an empty string on timeout so callers can decide
    whether to treat the endpoint as unreachable.
    """
    host = (host or BOXA_HOST).rstrip("/")
    noproxy = urlparse(host).hostname or ""
    cmd = ["curl", "-s", "--noproxy", noproxy, "--max-time", str(max_time), f"{host}{path}"]
    try:
        # timeout= backstops a wedged curl process; --max-time only caps the transfer
        return subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 5).stdout
    except subprocess.TimeoutExpired:
        return ""


def _comfy_lora_names(host: str | None = None, strict: bool = True) -> list[str]:
    """Authoritative list of names LoraLoader accepts on ``host``.

    ComfyUI returns foldered names with backslashes on Windows and forward slashes
    elsewhere.  This function normalizes nothing; it returns the exact strings the
    loader expects, which are also the keys used in the catalog index.
    """
    # (the remainder of this function is shared catalog access)
    host = host or BOXA_HOST
    raw = _curl("/object_info/LoraLoader", host=host)
    try:
        obj = json.loads(raw)
        spec = obj["LoraLoader"]["input"]["required"]["lora_name"]
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            names = spec[0]
        elif (
            isinstance(spec, list)
            and len(spec) > 1
            and spec[0] == "COMBO"
            and isinstance(spec[1], dict)
            and isinstance(spec[1].get("options"), list)
        ):
            names = spec[1]["options"]
        else:
            raise ValueError("unrecognized LoraLoader lora_name input shape")
        if not all(isinstance(name, str) for name in names):
            raise ValueError("LoraLoader lora_name options must all be strings")
        return list(names)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        if not strict:
            return []
        sys.exit(f"could not read LoraLoader names from {host}/object_info/LoraLoader "
                 f"({type(exc).__name__}); is BOXA ComfyUI up? raw={raw[:200]!r}")


def _lm_records(kind: str = "loras") -> dict[str, list[dict]]:
    """Fetch all LoRA-Manager records for a kind, keyed by lowercased file stem.

    Multiple LoRA-Manager records may share the same file stem (e.g., the same
    filename in different folders).  The dict therefore maps stem -> list so the
    build step can later disambiguate by folder path.

    Paging tolerates transient empty/non-JSON pages with a few retries, but stops
    early if a page repeatedly fails so the build does not hang on a wedged API.
    """
    # (the remainder of this function is network paging and retry logic)
    by_stem: dict[str, list[dict]] = {}
    page, page_size, seen, total = 1, 100, 0, None
    while page <= 100:  # hard safety cap against a misbehaving pager
        data = None
        for attempt in range(3):  # tolerate a transient empty/non-JSON page before giving up
            raw = _curl(f"/api/lm/{kind}/list?page={page}&page_size={page_size}")
            try:
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2)
        if data is None:
            print(f"  LoRA-Manager {kind} page {page} returned non-JSON after retries; index may be incomplete.",
                  file=sys.stderr)
            break
        items = data.get("items", [])
        if total is None:
            total = data.get("total")
        if not items:
            break
        for it in items:
            stem = str(it.get("file_name", "")).lower()
            by_stem.setdefault(stem, []).append(it)
        seen += len(items)
        if total is not None and seen >= total:
            break
        page += 1
    return by_stem


def _pick_lm(stem: str, folder_norm: str, by_stem: dict[str, list[dict]]) -> dict | None:
    """Pick the LoRA-Manager record whose folder path matches the ComfyUI name.

    If only one candidate exists for ``stem``, return it.  Otherwise try to
    disambiguate by the ``folder`` field; as a last resort keep the first candidate
    so a build can still produce a (possibly wrong) record rather than dropping data.
    """
    cands = by_stem.get(stem)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    for c in cands:  # disambiguate same-stem collisions by folder
        if str(c.get("folder", "")).replace("\\", "/").lower() == folder_norm:
            return c
    return cands[0]


@lru_cache(maxsize=1)
def _load_index() -> list[dict]:
    """Return the parsed catalog records, cached for the process.

    The on-disk JSON is read through ``loramake.model_resources.load_snapshot``,
    which uses its own ``(mtime_ns, size)`` cache and is safe for concurrent callers.
    """
    # (the actual snapshot and index building logic lives in model_resources)
    if not INDEX_PATH.exists():
        sys.exit(f"missing {INDEX_PATH}; run: python3 scripts/lora_catalog.py build")
    return _mr.load_snapshot(INDEX_PATH).records


def _doc(r: dict) -> str:
    """Embedding text: name + base + category + triggers + tags."""
    return _mr.doc(r)


def _doc_hash(doc: str) -> str:
    """Stable short hash of a doc text."""
    return _mr.doc_hash(doc)


@lru_cache(maxsize=1)
def _embedder():
    """Load the sentence-transformers embedding model once per process.

    Tries local model files first when ``LORA_EMBED_OFFLINE`` is not ``0``; falls
    back to a normal (potentially network) load only if the local copy is missing.
    This keeps the import path from depending on ``torch`` at module load time.
    """
    # (embedding logic remains here because it requires the sentence-transformers dependency)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed; run: pip install sentence-transformers")
    if os.environ.get("LORA_EMBED_OFFLINE", "1") != "0":
        try:
            return SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:
            pass
    return SentenceTransformer(EMBED_MODEL)


def build() -> None:
    """Build the LoRA catalog index from the BOXA / LoRA-Manager APIs.

    Walks the ComfyUI LoraLoader name list, matches each file to the best
    LoRA-Manager record by stem and folder, builds a record, then maps each
    canonical name to its BOXB-registered name.  Writes ``lora_catalog/boxa_index.json``
    and invalidates the in-process snapshot caches so subsequent callers see the
    new index.
    """
    CAT_DIR.mkdir(parents=True, exist_ok=True)
    names = _comfy_lora_names()
    by_stem = _lm_records()
    records, gaps = [], 0
    for name in names:
        parts = name.replace("\\", "/").split("/")
        fname = parts[-1]
        stem = fname.rsplit(".", 1)[0].lower()
        folder_parts = parts[:-1]
        folder_norm = "/".join(folder_parts).lower()
        lm = _pick_lm(stem, folder_norm, by_stem)

        tags = list(lm.get("tags", [])) if lm else []
        civ = (lm or {}).get("civitai") or {}
        triggers = [w for w in (civ.get("trainedWords") or []) if w and w.strip()]
        base_model = (lm.get("base_model") if lm else None) or (folder_parts[0] if folder_parts else "unknown")
        category = (str(lm.get("folder", "")).replace("\\", "/").split("/")[-1] if lm and lm.get("folder")
                    else (folder_parts[-1] if folder_parts else "root"))
        has_meta = bool(lm and lm.get("from_civitai") and (triggers or tags))
        if not has_meta:
            gaps += 1
        records.append({
            "id": (lm.get("sha256", "")[:12] if lm and lm.get("sha256") else stem[:12]),
            "name": (lm.get("model_name") if lm else None) or fname.rsplit(".", 1)[0],
            "file": name,
            "base_model": base_model,
            "category": category or "root",
            "triggers": triggers,
            "tags": tags,
            "strength_default": strength_default(lm.get("usage_tips") if lm else None),
            "content_rating": content_rating(lm, tags, category or "root"),
            "has_meta": has_meta,
            "preview": (lm.get("preview_url") if lm else None),
        })
    boxb_names = set(_comfy_lora_names(BOXB_HOST, strict=False))
    if boxb_names:
        boxb_map = match_boxb([r["file"] for r in records], boxb_names)
        for r in records:
            r["boxb_file"] = boxb_map[r["file"]]
        on_boxb = sum(1 for r in records if r["boxb_file"])
        print(f"  boxb availability: {on_boxb}/{len(records)} ({len(boxb_names)} names on {BOXB_HOST})")
    else:
        # BOXB unreachable: carry availability over from the previous index.
        prev = {}
        if INDEX_PATH.exists():
            try:
                prev = {r.get("file"): r.get("boxb_file") for r in json.loads(INDEX_PATH.read_text())}
            except (OSError, json.JSONDecodeError):
                prev = {}
        for r in records:
            r["boxb_file"] = prev.get(r["file"])
        print(f"  boxb probe unreachable ({BOXB_HOST}); carried boxb_file from previous index")
    atomic_write_text(INDEX_PATH, json.dumps(records, ensure_ascii=False, indent=2))
    _load_index.cache_clear()
    _mr.invalidate_cache(INDEX_PATH)
    rated = {}
    for r in records:
        rated[r["content_rating"]] = rated.get(r["content_rating"], 0) + 1
    print(f"indexed {len(records)} LoRAs -> {INDEX_PATH}")
    print(f"  metadata gaps (has_meta=false): {gaps}")
    print(f"  content_rating: {rated}")


def embed(force: bool = False) -> None:
    """Embed each record's doc text and write ``lora_catalog/boxa_vectors.npz``.

    Uses a stable short hash of the doc text to skip unchanged records.  The saved
    ``.npz`` also stores the model name and hashes so ``select`` can detect
    mismatches and request a re-embed.
    """
    # (the embedding model is loaded lazily through _embedder)
    import numpy as np
    records = _load_index()
    if not records:
        sys.exit("no records to embed; run: python3 scripts/lora_catalog.py build")
    docs = [_doc(r) for r in records]
    ids = [r["id"] for r in records]
    hashes = [_doc_hash(d) for d in docs]

    reuse: dict[tuple[str, str], object] = {}
    if not force and VECTORS_PATH.exists():
        try:
            old = np.load(VECTORS_PATH, allow_pickle=True)
            if str(old["model"][0]) == EMBED_MODEL and "hashes" in old.files:
                ov = old["vectors"]
                for i, (oid, oh) in enumerate(zip(old["ids"], old["hashes"], strict=False)):
                    reuse[(str(oid), str(oh))] = ov[i]
        except Exception:
            reuse = {}

    todo = [i for i in range(len(records)) if (ids[i], hashes[i]) not in reuse]
    encoded: dict[int, object] = {}
    if todo:
        model = _embedder()
        new = np.asarray(model.encode([docs[i] for i in todo], normalize_embeddings=True,
                                      show_progress_bar=True, batch_size=64), dtype="float32")
        encoded = {i: new[j] for j, i in enumerate(todo)}

    dim = (next(iter(encoded.values())) if encoded else next(iter(reuse.values()))).shape[0]
    vectors = np.zeros((len(records), dim), dtype="float32")
    for i in range(len(records)):
        vectors[i] = encoded[i] if i in encoded else reuse[(ids[i], hashes[i])]
    np.savez(VECTORS_PATH, ids=np.array(ids), vectors=vectors,
             hashes=np.array(hashes), model=np.array([EMBED_MODEL]))
    print(f"embedded {len(records)} docs ({len(todo)} new/changed, {len(records) - len(todo)} reused), "
          f"dim={dim} -> {VECTORS_PATH}")


def select(context: str, k: int = 8, base: str | None = None, include_nsfw: bool = False) -> list[dict]:
    """Semantic search: embed ``context`` and return the top-k LoRAs by cosine similarity.

    Loads precomputed vectors from ``boxa_vectors.npz``.  Records are aligned by id
    so the vector cache survives index additions as long as existing ids do not
    change.  Filters for base_model, content rating, and ``include_nsfw`` are
    applied after the vector scan.
    """
    import numpy as np
    records = _load_index()
    if not VECTORS_PATH.exists():
        sys.exit(f"missing {VECTORS_PATH}; run: python3 scripts/lora_catalog.py embed")
    npz = np.load(VECTORS_PATH, allow_pickle=True)
    vectors = npz["vectors"]
    by_id = {r["id"]: r for r in records}
    ordered = [by_id.get(i) for i in npz["ids"]]
    q = _embedder().encode([context], normalize_embeddings=True)[0].astype("float32")
    scores = vectors @ q
    ranked = sorted(zip(scores, ordered, strict=False), key=lambda x: -float(x[0]))
    out = []
    for score, r in ranked:
        if r is None:
            continue
        if not include_nsfw and r["content_rating"] == "nsfw":
            continue
        if base and r["base_model"].lower() != base.lower():
            continue
        out.append({**r, "score": round(float(score), 4)})
        if len(out) >= k:
            break
    return out


def pick(context: str, base: str | None = None, k: int = 1, min_score: float = 0.0,
         include_nsfw: bool = False) -> list[dict]:
    """Semantic ``select`` filtered by a score floor."""
    return [r for r in select(context, k=k, base=base, include_nsfw=include_nsfw)
            if r["score"] >= min_score]


def main() -> None:
    ap = argparse.ArgumentParser(description="BOXA LoRA catalog: build / embed / select")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    se = sub.add_parser("embed")
    se.add_argument("--force", action="store_true", help="re-encode all docs")
    sp = sub.add_parser("select")
    sp.add_argument("context")
    sp.add_argument("--k", type=int, default=8)
    sp.add_argument("--base", default=None)
    sp.add_argument("--include-nsfw", action="store_true")
    sr = sub.add_parser("roles", help="suggest a role for each LoRA file")
    sr.add_argument("loras", nargs="+", help="LoRA file names to classify")
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    elif args.cmd == "embed":
        embed(force=args.force)
    elif args.cmd == "select":
        for r in select(args.context, k=args.k, base=args.base, include_nsfw=args.include_nsfw):
            trig = ", ".join(r["triggers"]) or "-"
            if len(trig) > 100:
                trig = trig[:97] + "..."
            print(f"{r['score']:.3f}  [{r['base_model']}/{r['category']}] {r['name']}")
            print(f"        file={r['file']}  trigger={trig}  str={r['strength_default']}  rating={r['content_rating']}")
    elif args.cmd == "roles":
        for f in args.loras:
            rec = record_of(f)
            if not rec:
                print(f"?          {f}  (not in catalog)")
                continue
            role, hits = classify_role(rec)
            conf = "strong" if hits >= 2 else ("weak" if hits == 1 else "fallback")
            tags = ", ".join(rec.get("tags", [])[:6]) or "-"
            print(f"{role:9} [{conf}]  {rec['file']}")
            print(f"           base={rec['base_model']} category={rec['category']} tags={tags}")


if __name__ == "__main__":
    main()
