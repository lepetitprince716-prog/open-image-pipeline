#!/usr/bin/env python3
"""BOXA model index — one machine-readable inventory of the models on BOXA ComfyUI.

The deterministic LoRA lookup/pack paths now delegate to ``loramake.model_resources``;
lexical lookup is the default; ``--semantic`` restores ``lora_catalog.select`` (MiniLM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_catalog  # delegate the LoRA build/semantic line
import model_quirks
from loramake import model_resources as _mr
from boxa_comfy import BOXA_HOST, atomic_write_text, curl, object_info_enum

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "boxa_model_index"
INDEX_PATH = OUT_DIR / "index.json"
NSFW_ALIAS_PATH = ROOT / "nsfw" / "lora_search_aliases.json"

SCHEMA_VERSION = "boxa-model-index-v1"
DEFAULT_STALE_HOURS = 24
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Support kinds indexed by canonical name only.
NAME_ONLY_KINDS = {
    "vae": ("VAELoader", "vae_name"),
    "controlnet": ("ControlNetLoader", "control_net_name"),
    "upscale_models": ("UpscaleModelLoader", "model_name"),
    "clip_vision": ("CLIPVisionLoader", "clip_name"),
    "diffusion_models": ("UNETLoader", "unet_name"),
    "ipadapter": ("IPAdapterModelLoader", "ipadapter_file"),
}

SOURCE_ENDPOINTS = [
    "/object_info/<NodeClass>",
    "/api/lm/loras/list",
    "/api/lm/checkpoints/list",
]

KIND_ALIASES = {"checkpoint": "checkpoints", "loras": "lora"}


# --------------------------------------------------------------------------- helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(TS_FMT)


def _reachable() -> bool:
    """True only if BOXA answers /system_stats with valid JSON."""
    try:
        json.loads(curl([f"{BOXA_HOST.rstrip('/')}/system_stats"]) or "")
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _folder_base(name: str) -> str:
    """Top-level folder of a foldered ComfyUI name, or 'unknown' for a flat name."""
    norm = name.replace("\\", "/")
    return norm.split("/")[0] if "/" in norm else "unknown"


def _stem(name: str) -> str:
    fname = name.replace("\\", "/").split("/")[-1]
    return fname.rsplit(".", 1)[0]


# --------------------------------------------------------------------------- build
def _build_checkpoints(names: set[str], lm_by_stem: dict[str, list[dict]]) -> list[dict]:
    """Checkpoints loadable on BOXA, enriched with LoRA-Manager metadata where matched."""
    out = []
    for name in sorted(names):
        parts = name.replace("\\", "/").split("/")
        stem = parts[-1].rsplit(".", 1)[0].lower()
        folder_norm = "/".join(parts[:-1]).lower()
        lm = lora_catalog._pick_lm(stem, folder_norm, lm_by_stem)
        tags = list(lm.get("tags", [])) if lm else []
        base = (lm.get("base_model") if lm else None) or _folder_base(name)
        out.append({
            "id": (lm.get("sha256", "")[:12] if lm and lm.get("sha256") else stem[:12]),
            "kind": "checkpoint",
            "canonical_name": name,
            "name": (lm.get("model_name") if lm else None) or _stem(name),
            "base_model": base or "unknown",
            "tags": tags,
            "content_rating": lora_catalog._content_rating(lm, tags),
            "availability": "confirmed",
            "has_meta": bool(lm),
            "preview": (lm.get("preview_url") if lm else None),
        })
    return out


def _name_only_record(kind: str, name: str) -> dict:
    return {
        "id": _stem(name).lower()[:16],
        "kind": kind,
        "canonical_name": name,
        "name": _stem(name),
        "base_model": _folder_base(name),
        "tags": [],
        "content_rating": "unknown",
        "availability": "confirmed",
        "has_meta": False,
        "preview": None,
    }


def _lora_summary() -> dict | None:
    """Counts + pointer to lora_catalog/boxa_index.json."""
    p = lora_catalog.INDEX_PATH
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    by_base: dict[str, int] = {}
    by_rating: dict[str, int] = {}
    for r in recs:
        by_base[r.get("base_model", "unknown")] = by_base.get(r.get("base_model", "unknown"), 0) + 1
        by_rating[r.get("content_rating", "unknown")] = by_rating.get(r.get("content_rating", "unknown"), 0) + 1
    return {
        "count": len(recs),
        "source": str(p.relative_to(ROOT)),
        "by_base": by_base,
        "by_rating": by_rating,
    }


def refresh(out: Path, skip_loras: bool) -> int:
    if not _reachable():
        print(f"ERROR: BOXA not reachable at {BOXA_HOST} (/system_stats failed); index NOT overwritten.",
              file=sys.stderr)
        return 2

    status = "ok"
    gaps: list[str] = []
    assets: dict[str, list[dict]] = {}

    ckpt_names = object_info_enum("CheckpointLoaderSimple", "ckpt_name")
    if not ckpt_names:
        status = "partial"
        gaps.append("checkpoints: /object_info/CheckpointLoaderSimple returned no names")
    lm_ckpt: dict[str, list[dict]] = {}
    try:
        lm_ckpt = lora_catalog._lm_records("checkpoints")
    except Exception as exc:
        status = "partial"
        gaps.append(f"checkpoints: LoRA-Manager enrichment failed ({type(exc).__name__})")
    assets["checkpoints"] = _build_checkpoints(ckpt_names, lm_ckpt)

    for kind, (node, field) in NAME_ONLY_KINDS.items():
        names = object_info_enum(node, field)
        if not names:
            gaps.append(f"{kind}: /object_info/{node}.{field} returned no names")
        assets[kind] = [_name_only_record(kind, n) for n in sorted(names)]

    old_lora_ids: set[str] = set()
    if lora_catalog.INDEX_PATH.exists():
        try:
            old_lora_ids = {r["id"] for r in json.loads(lora_catalog.INDEX_PATH.read_text())}
        except Exception:
            old_lora_ids = set()

    if not skip_loras:
        try:
            lora_catalog.build()
        except SystemExit as exc:
            status = "partial"
            gaps.append(f"loras: rebuild failed ({exc})")
        except Exception as exc:
            status = "partial"
            gaps.append(f"loras: rebuild failed ({type(exc).__name__})")
        else:
            # The catalog is current at this point; only the --semantic vectors can lag.
            try:
                lora_catalog.embed()
            except ImportError as exc:
                status = "partial"
                gaps.append(f"loras: embed failed ({type(exc).__name__}: {exc}; interpreter={sys.executable};"
                            " run refresh with .venv/bin/python)")
            except SystemExit as exc:
                status = "partial"
                gaps.append(f"loras: embed failed ({exc}; interpreter={sys.executable})")
            except Exception as exc:
                status = "partial"
                gaps.append(f"loras: embed failed ({type(exc).__name__}: {exc})")
    lora_summary = _lora_summary()
    if lora_summary is None:
        status = "partial"
        gaps.append("loras: lora_catalog/boxa_index.json missing")
    elif old_lora_ids and lora_catalog.INDEX_PATH.exists():
        try:
            cur = json.loads(lora_catalog.INDEX_PATH.read_text())
            added = [{"canonical_name": r["file"], "name": r.get("name"), "base_model": r.get("base_model")}
                     for r in cur if r["id"] not in old_lora_ids]
        except Exception:
            added = []
        lora_summary["added_count"] = len(added)
        lora_summary["added"] = added

    counts = {k: len(v) for k, v in assets.items()}
    counts["loras"] = (lora_summary or {}).get("count", 0)

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "source_host": BOXA_HOST,
        "source_endpoints": SOURCE_ENDPOINTS,
        "refresh_status": status,
        "stale_after_hours": DEFAULT_STALE_HOURS,
        "counts": counts,
        "gaps": gaps,
        "loras": lora_summary,
        "assets": assets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(index, ensure_ascii=False, indent=2))
    print(f"wrote {out} (status={status})")
    print(f"  counts: {counts}")
    if gaps:
        print(f"  gaps ({len(gaps)}): " + "; ".join(gaps))
    new_n = (lora_summary or {}).get("added_count", 0)
    if new_n:
        names = [a["canonical_name"] for a in lora_summary["added"][:20]]
        print(f"  NEW loras ({new_n}): " + ", ".join(names) + (" ..." if new_n > 20 else ""))

    try:
        import boxa_model_browser
        print(f"  browser: {boxa_model_browser.build()}")
    except Exception as exc:
        print(f"  browser: skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- lookup
def _staleness(index: dict, max_age_hours: float | None = None) -> str:
    gen = index.get("generated_at")
    hours = max_age_hours if max_age_hours is not None else index.get("stale_after_hours", DEFAULT_STALE_HOURS)
    if not gen:
        return "missing"
    try:
        ts = datetime.strptime(gen, TS_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return "unknown"
    return "stale" if datetime.now(timezone.utc) - ts > timedelta(hours=hours) else "fresh"


@lru_cache(maxsize=1)
def _nsfw_alias_payload() -> tuple[dict[str, list[str]], str | None]:
    """Cached (map, error). Keys are tokenizer-normalized. Error is missing|unreadable."""
    if not NSFW_ALIAS_PATH.is_file():
        return {}, "missing"
    try:
        data = json.loads(NSFW_ALIAS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "unreadable"
    raw = data.get("aliases") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}, "unreadable"
    out: dict[str, list[str]] = {}
    for key, vals in raw.items():
        if not isinstance(key, str) or not isinstance(vals, list):
            continue
        tokens = _mr.tokenize_query(key)
        canon = tokens[0] if tokens else key.lower()
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in [canon, *[str(v).strip() for v in vals if str(v).strip()]]:
            if item not in seen:
                seen.add(item)
                cleaned.append(item)
        if cleaned:
            out[canon] = cleaned
    return out, None


_ALIAS_WARNED = False


def _load_nsfw_alias_map(include_nsfw: bool) -> dict[str, list[str]] | None:
    """Load nsfw/ alias JSON only for NSFW lookups. Warn once if the file is unusable."""
    global _ALIAS_WARNED
    if not include_nsfw:
        return None
    mapping, err = _nsfw_alias_payload()
    if err and not _ALIAS_WARNED:
        print(f"[lookup] nsfw alias file {err}: {NSFW_ALIAS_PATH}", file=sys.stderr)
        _ALIAS_WARNED = True
    return mapping or None


def _alias_expanded_queries(ctx: str, alias_map: dict[str, list[str]]) -> list[str]:
    """Replace alias keys with synonym queries; leftover tokens are folded into each synonym.

    ASCII keys match a whole token. CJK keys also match as substrings inside a CJK
    run, so ``重口场景`` / ``极限题材`` expand the same way as the bare keys.
    """
    toks = _mr.tokenize_query(ctx or "")
    if not toks or not alias_map:
        return [ctx or ""]
    leftover: list[str] = []
    syns: list[str] = []
    seen: set[str] = set()
    hit = False
    cjk_keys = sorted((k for k in alias_map if k and not k.isascii()), key=len, reverse=True)

    def _add(values: list[str]) -> None:
        nonlocal hit
        hit = True
        for syn in values:
            if syn not in seen:
                seen.add(syn)
                syns.append(syn)

    for tok in toks:
        values = alias_map.get(tok)
        if values:
            _add(values)
            continue
        if cjk_keys and not tok.isascii():
            rest = tok
            found = False
            progressing = True
            while progressing:
                progressing = False
                for key in cjk_keys:
                    if key in rest:
                        _add(alias_map[key])
                        rest = rest.replace(key, "")
                        found = True
                        progressing = True
                        break
            if found:
                if rest:
                    leftover.append(rest)
                continue
        leftover.append(tok)
    if not hit:
        return [ctx or ""]
    tail = " ".join(leftover)
    if tail:
        return [f"{syn} {tail}" for syn in syns]
    return syns


def _lora_spec(canonical: str, strength: float | None) -> str:
    """Copyable ``--lora`` flag. Always single-quoted so ``\\`` in Comfy names survives bash."""
    s = 0.8 if strength is None else strength
    esc = canonical.replace("'", "'\\''")
    return f"--lora '{esc}:{s}'"


def _why_lora(r: dict) -> str:
    bits = []
    if r.get("score") is not None:
        bits.append(f"semantic score {r['score']}")
    if r.get("base_model"):
        bits.append(f"base {r['base_model']}")
    if r.get("triggers"):
        bits.append("triggers " + ", ".join(str(t) for t in r["triggers"][:3]))
    return "; ".join(bits) or "match"


def _shape_lora(r: dict, compact: bool = False) -> dict:
    s = r.get("strength_default")
    role, role_conf = _mr.classify_role(r)
    triggers = list(r.get("triggers") or [])
    # Deterministic search already produced a concise why; preserve it.  Semantic
    # results do not carry "why" and are synthesized here with capped triggers.
    # Legacy mocked records with a manually supplied why are also respected.
    if r.get("why") is not None:
        why = r["why"]
    else:
        why = _why_lora(r)
    if compact:
        triggers = triggers[:3]
        if len(why) > 120:
            why = why[:117] + "..."
    out = {
        "canonical_name": r["file"],
        "kind": "lora",
        "score": r.get("score"),
        "role": role,
        "role_confidence": role_conf,
        "why": why,
        "triggers": triggers,
        "recommended_strengths": {"model": s, "clip": s} if s is not None else None,
        "base_model": r.get("base_model"),
        "category": r.get("category"),
        "content_rating": r.get("content_rating"),
        "availability": "confirmed",
        "boxb_file": r.get("boxb_file"),
        "has_meta": bool(r.get("has_meta")),
    }
    if r.get("boxa_file"):
        out["boxa_file"] = r["boxa_file"]
    strength = None
    rec_str = out.get("recommended_strengths") or {}
    if rec_str.get("model") is not None:
        strength = rec_str["model"]
    out["lora_spec"] = _lora_spec(out["canonical_name"], strength)
    return out


def _family_pred(family: str | None):
    """Predicate over a record's base_model for --family."""
    try:
        return _mr.family_pred(family)
    except _mr.UnknownFamilyError:
        sys.exit(f"unknown --family {family!r}; use sdxl | anima | sd15 | any")


def _apply_box_loras(recs: list[dict], box: str | None) -> list[dict]:
    """--box boxb keeps only LoRAs mirrored on BOXB and rewrites ``file`` to the BOXB name."""
    if not box or box == "boxa":
        return recs
    return _mr.apply_box(recs, box)


def _lora_substring(ctx: str, base: str | None, top_k: int, include_nsfw: bool,
                    family: str | None = None, box: str | None = None) -> list[dict]:
    """Deterministic fallback: weighted lexical search over the LoRA index.

    Keeps the numeric ``score`` from ``search_records`` so callers like
    ``build_pack`` can enforce ``min_score`` on lexical hits too.
    NSFW alias keys expand here (one ``search_records`` per synonym, merge by file).
    """
    records = lora_catalog._load_index()
    queries = _alias_expanded_queries(ctx or "", _load_nsfw_alias_map(include_nsfw) or {})
    pool = top_k if len(queries) == 1 else max(top_k * 4, 32)
    merged: dict[str, dict] = {}
    for query in queries:
        for res in _mr.search_records(
            records,
            query,
            top_k=pool,
            include_nsfw=include_nsfw,
            base=base,
            family=family,
            box=box,
        ):
            key = res["record"]["file"]
            prev = merged.get(key)
            if prev is None or res["score"] > prev["score"] or (
                res["score"] == prev["score"] and res["name_exact"] and not prev["name_exact"]
            ):
                merged[key] = res
    ranked = sorted(
        merged.values(),
        key=lambda r: (not r["name_exact"], -r["score"], r["record"]["file"], r["index"]),
    )[:top_k]
    out = []
    for res in ranked:
        r = _apply_box_loras([res["record"]], box)[0]
        new = dict(r)
        new["score"] = res["score"]
        new["why"] = res["why"] if res["why"] else "match"
        out.append(new)
    return out


def _validate_family(family: str | None) -> None:
    """Fail fast on an unsupported --family value."""
    try:
        _mr.family_pred(family)
    except _mr.UnknownFamilyError:
        sys.exit(f"unknown --family {family!r}; use sdxl | anima | sd15 | any")


def _lookup_lora(ctx: str, base: str | None, top_k: int, include_nsfw: bool,
                 new_names: set[str] | None = None, family: str | None = None,
                 box: str | None = None, semantic: bool = False,
                 compact: bool = False) -> list[dict]:
    _validate_family(family)
    wide = new_names is not None or family not in (None, "any") or box == "boxb"
    pool_k = max(top_k * 30, 2000) if wide else top_k
    if semantic and lora_catalog.INDEX_PATH.exists() and lora_catalog.VECTORS_PATH.exists() and ctx.strip():
        recs = lora_catalog.select(ctx, k=pool_k, base=base, include_nsfw=include_nsfw)
        recs = _apply_box_loras([r for r in recs if _mr.record_family_compatible(r, family)], box)
    else:
        recs = _lora_substring(ctx, base, pool_k, include_nsfw, family=family, box=box)
    if new_names is not None:
        recs = [r for r in recs if r.get("boxa_file", r["file"]) in new_names]
    return [_shape_lora(r, compact=compact) for r in recs[:top_k]]


def _lookup_inventory(index: dict, ctx: str, kind: str, base: str | None,
                      top_k: int, include_nsfw: bool) -> list[dict]:
    """Delegate support-asset lookup to the core inventory search.

    The CLI envelope (unknown-kind exit, base-model filtering, NSFW filtering,
    empty-query inventory) is preserved; the deterministic scoring and result
    shaping live in ``loramake.model_resources``.
    """
    assets = index.get("assets", {})
    if kind not in assets:
        sys.exit(f"unknown kind {kind!r}; available: lora, " + ", ".join(assets.keys()))
    try:
        return _mr.search_inventory(
            query=ctx or "",
            kind=kind,
            index=index,
            top_k=top_k,
            include_nsfw=include_nsfw,
            base=base,
        )
    except _mr.UnknownKindError as exc:
        sys.exit(f"[index] {exc}")


def _print_envelope(env: dict) -> None:
    f = env["filters_applied"]
    print(f"# query={env['query']!r} kind={f['kind']} base={f['base']} "
          f"index={env['index_status']} ({env['index_generated_at']})")
    if not env["results"]:
        print("  (no matches)")
        return
    for r in env["results"]:
        sc = f"{r['score']:.3f}  " if r.get("score") is not None else ""
        print(f"{sc}[{r.get('base_model')}/{r['kind']}] {r['canonical_name']}  rating={r.get('content_rating')}")
        extra: list[str] = []
        if r.get("why"):
            extra.append(f"why={r['why']}")
        if r.get("lora_spec"):
            extra.append(r["lora_spec"])
        if extra:
            print("        " + "  ".join(extra))
        if r.get("triggers") or r.get("recommended_strengths"):
            trig = ", ".join(str(t) for t in r.get("triggers", [])) or "-"
            trig = trig[:97] + "..." if len(trig) > 100 else trig
            print(f"        triggers={trig}  rec_str={r.get('recommended_strengths')}")


def _lookup_envelope(index: dict, ctx: str, kind: str, base: str | None, top_k: int,
                     include_nsfw: bool, new_only: bool, family: str | None,
                     box: str | None, semantic: bool, compact: bool = False) -> dict:
    new_names: set[str] | None = None
    if new_only:
        if kind == "lora":
            new_names = {a["canonical_name"] for a in (index.get("loras") or {}).get("added", [])}
        else:
            print(f"NOTE: --new tracks newly-added LoRAs only; ignoring for kind={kind}.", file=sys.stderr)
    results = (_lookup_lora(ctx, base, top_k, include_nsfw, new_names=new_names,
                            family=family, box=box, semantic=semantic, compact=compact) if kind == "lora"
               else _lookup_inventory(index, ctx, kind, base, top_k, include_nsfw))
    return {
        "query": ctx,
        "filters_applied": {"kind": kind, "base": base, "top_k": top_k,
                            "include_nsfw": include_nsfw, "new": new_only,
                            "family": family or "any", "box": box or "boxa",
                            "semantic": semantic},
        "index_generated_at": index.get("generated_at"),
        "index_status": _staleness(index),
        "results": results,
    }


# --------------------------------------------------------------------------- pack
ROLE_ORDER = ("character", "style", "concept", "pose", "detail")
PACK_PICK_CAP = 4


def _score_key(r: dict) -> float:
    return r["score"] if r.get("score") is not None else -1.0


def _suggestible(r: dict, min_score: float) -> bool:
    """Semantic hits must clear the floor; deterministic hits (score=None) are suggestible."""
    return r.get("score") is None or r["score"] >= min_score


def build_pack(contexts: list[str], family: str | None, box: str | None, include_nsfw: bool,
               per_role: int, min_score: float, quirks_base: str | None,
               semantic: bool = False) -> dict:
    t0 = time.monotonic()
    index = _load_index()
    pool_k = max(per_role * 10, 50)
    best: dict[str, dict] = {}
    for ctx in contexts:
        for r in _lookup_lora(ctx, None, pool_k, include_nsfw,
                              family=family, box=box, semantic=semantic, compact=False):
            key = r["canonical_name"]
            if key not in best or _score_key(r) > _score_key(best[key]):
                best[key] = r

    buckets: dict[str, list[dict]] = {role: [] for role in ROLE_ORDER}
    for r in sorted(best.values(), key=_score_key, reverse=True):
        role = str(r.get("role") or "character")
        if role not in buckets:
            role = "character"
        if len(buckets[role]) < per_role:
            buckets[role].append(r)

    picked: list[dict] = []
    for role in ROLE_ORDER:
        for r in buckets[role]:
            if _suggestible(r, min_score):
                picked.append(r)
                break
    picked = picked[:PACK_PICK_CAP]

    # Compact per-role candidate previews; keep selected trigger injection separate.
    compacted = []
    for role in ROLE_ORDER:
        for r in buckets[role]:
            cr = dict(r)
            cr["triggers"] = (cr.get("triggers") or [])[:3]
            if isinstance(cr.get("why"), str) and len(cr["why"]) > 120:
                cr["why"] = cr["why"][:117] + "..."
            compacted.append(cr)
    buckets = {role: [r for r in compacted if r.get("role") == role] for role in ROLE_ORDER}

    argv: list[str] = []
    for r in picked:
        strength = (r.get("recommended_strengths") or {}).get("model") or 0.8
        argv += ["--lora", f"{r['canonical_name']}:{strength}"]
    quirks = model_quirks.card(quirks_base) if quirks_base else None
    return {
        "schema_version": "loramake.resource-pack/v1",
        "queries": list(contexts),
        "filters": {"family": family or "any", "box": box or "boxa", "include_nsfw": include_nsfw,
                    "per_role": per_role, "min_score": min_score},
        "index_generated_at": index.get("generated_at"),
        "index_status": _staleness(index),
        "loras_by_role": {role: rs for role, rs in buckets.items() if rs},
        "suggested": {
            "picked": [{"role": r.get("role"), "canonical_name": r["canonical_name"],
                        "score": r.get("score"),
                        "strength": (r.get("recommended_strengths") or {}).get("model") or 0.8}
                       for r in picked],
            "argv": argv,
            "triggers": _mr.triggers_for(picked),
        },
        "quirks": quirks,
        "timing": {"pack_s": round(time.monotonic() - t0, 3)},
    }


def _print_pack(env: dict) -> None:
    f = env["filters"]
    print(f"# pack queries={env['queries']} family={f['family']} box={f['box']} "
          f"nsfw={f['include_nsfw']} index={env['index_status']} ({env['index_generated_at']}) "
          f"took {env['timing']['pack_s']}s")
    if not env["loras_by_role"]:
        print("  (no matches)")
    for role, rs in env["loras_by_role"].items():
        print(f"[{role}]")
        for r in rs:
            sc = f"{r['score']:.3f}  " if r.get("score") is not None else ""
            print(f"  {sc}{r['canonical_name']}  base={r.get('base_model')} "
                  f"rating={r.get('content_rating')}")
    sug = env["suggested"]
    if sug["picked"]:
        print("suggested: " + " ".join(sug["argv"]))
        if sug["triggers"]:
            trig = sug["triggers"]
            print(f"triggers: {trig[:117] + '...' if len(trig) > 120 else trig}")
    if env.get("quirks"):
        q = env["quirks"]
        print(f"quirks[{q.get('base')}]: neg_add={str(q.get('neg_add', ''))[:80]}  "
              f"flags={'--no-freeu' if q.get('no_freeu') else '-'}")


def pack(contexts: list[str], family: str | None, box: str | None, include_nsfw: bool,
         per_role: int, min_score: float, quirks_base: str | None, as_json: bool,
         allow_stale: bool, semantic: bool = False) -> int:
    _validate_family(family)
    env = build_pack(contexts, family, box, include_nsfw, per_role, min_score,
                     quirks_base, semantic=semantic)
    if env["index_status"] == "stale" and not allow_stale:
        print("WARNING: index is stale; run refresh or pass --allow-stale.", file=sys.stderr)
    if quirks_base and env["quirks"] is None:
        print(f"NOTE: no quirks card matches {quirks_base!r} (see: model_quirks.py list).",
              file=sys.stderr)
    if as_json:
        print(json.dumps(env, ensure_ascii=False, indent=2))
    else:
        _print_pack(env)
    return 0


# --------------------------------------------------------------------------- status / load
def _load_index(path: Path = INDEX_PATH) -> dict:
    if not path.exists():
        sys.exit(f"no index at {path}; run: .venv/bin/python scripts/boxa_model_index.py refresh")
    return json.loads(path.read_text())


def status(index_path: Path, max_age_hours: float | None) -> int:
    if not index_path.exists():
        print(f"no index at {index_path}; run refresh.")
        return 1
    index = json.loads(index_path.read_text())
    verdict = _staleness(index, max_age_hours)
    hours = max_age_hours if max_age_hours is not None else index.get("stale_after_hours", DEFAULT_STALE_HOURS)
    print(f"index: {index_path}")
    print(f"  generated_at: {index.get('generated_at')} ({verdict}, threshold {hours}h)")
    print(f"  refresh_status: {index.get('refresh_status')}  counts: {index.get('counts')}")
    if index.get("gaps"):
        print(f"  gaps ({len(index['gaps'])}): " + "; ".join(index["gaps"]))
    problems = _mr.catalog_readiness_problems(unified_index_path=index_path)
    if problems:
        for problem in problems:
            print(f"  BLOCKED: {problem}", file=sys.stderr)
        return 1
    # --max-age-hours can be stricter than the catalog gate's embedded threshold.
    # Automation that passed that flag must not see exit 0 on a stale verdict.
    if verdict == "fresh":
        return 0
    return 3 if verdict == "stale" else 1


# --------------------------------------------------------------------------- CLI
def lookup(contexts: list[str], kind: str, base: str | None, top_k: int,
           include_nsfw: bool, as_json: bool, allow_stale: bool, new_only: bool = False,
           family: str | None = None, box: str | None = None, semantic: bool = False) -> int:
    _validate_family(family)
    index = _load_index()
    kind = KIND_ALIASES.get(kind, kind)
    idx_status = _staleness(index)
    if idx_status == "stale" and not allow_stale:
        print("WARNING: index is stale; run refresh or pass --allow-stale.", file=sys.stderr)
    if box not in (None, "boxa", "boxb"):
        sys.exit(f"unknown --box {box!r}; use boxa | boxb")
    if box == "boxb" and kind != "lora":
        print(f"NOTE: --box boxb availability is tracked for LoRAs only; kind={kind} names are "
              f"BOXA-side.", file=sys.stderr)
    envelopes = [_lookup_envelope(index, ctx, kind, base, top_k, include_nsfw, new_only,
                                  family, box, semantic, compact=True)
                 for ctx in (contexts or [""])]
    if as_json:
        payload = envelopes[0] if len(envelopes) == 1 else envelopes
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for i, env in enumerate(envelopes):
            if i:
                print()
            _print_envelope(env)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="BOXA model index: refresh / lookup / status")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="rebuild the index from BOXA (read-only)")
    r.add_argument("--out", default=str(INDEX_PATH))
    r.add_argument("--skip-loras", action="store_true", help="reuse the existing LoRA index")

    lk = sub.add_parser("lookup", help="query a kind, print a consumer envelope")
    lk.add_argument("contexts", nargs="*", metavar="CONTEXT",
                    help="natural-language context(s); several are searched in one process")
    lk.add_argument("--kind", default="lora",
                    help="lora | checkpoint | vae | controlnet | upscale_models | clip_vision | "
                         "diffusion_models | ipadapter")
    lk.add_argument("--base", default=None)
    lk.add_argument("--top-k", type=int, default=8)
    lk.add_argument("--include-nsfw", action="store_true")
    lk.add_argument("--family", default=None,
                    help="arch-family filter for kind=lora: sdxl | anima | sd15 | any")
    lk.add_argument("--box", default=None, help="boxa (default) | boxb")
    lk_sem = lk.add_mutually_exclusive_group()
    lk_sem.add_argument("--semantic", action="store_true",
                        help="MiniLM cosine search (slow cold start)")
    lk_sem.add_argument("--no-semantic", action="store_true", dest="no_semantic",
                        help="lexical search (default; kept for old recipes)")
    lk.add_argument("--json", action="store_true", dest="as_json")
    lk.add_argument("--allow-stale", action="store_true")
    lk.add_argument("--new", action="store_true", dest="new_only",
                    help="restrict to LoRAs added since the previous refresh")

    pk = sub.add_parser("pack", help="one-call resource pack: LoRA candidates + triggers + argv")
    pk.add_argument("contexts", nargs="+", metavar="CONTEXT",
                    help="one or more natural-language contexts")
    pk.add_argument("--family", default=None, help="sdxl | anima | sd15 | any (default any)")
    pk.add_argument("--box", default=None, help="boxa (default) | boxb")
    pk.add_argument("--include-nsfw", action="store_true")
    pk.add_argument("--per-role", type=int, default=3, help="candidates per role bucket")
    pk.add_argument("--min-score", type=float, default=0.35, help="score floor (default 0.35)")
    pk.add_argument("--quirks-base", default=None, help="attach quirks card for this base")
    pk_sem = pk.add_mutually_exclusive_group()
    pk_sem.add_argument("--semantic", action="store_true",
                        help="MiniLM cosine search (slow cold start)")
    pk_sem.add_argument("--no-semantic", action="store_true", dest="no_semantic",
                        help="lexical search (default; kept for old recipes)")
    pk.add_argument("--json", action="store_true", dest="as_json")
    pk.add_argument("--allow-stale", action="store_true")

    st = sub.add_parser("status", help="report fresh/stale")
    st.add_argument("--index", default=str(INDEX_PATH))
    st.add_argument("--max-age-hours", type=float, default=None)

    args = ap.parse_args()
    if args.cmd == "refresh":
        sys.exit(refresh(Path(args.out), args.skip_loras))
    elif args.cmd == "lookup":
        sys.exit(lookup(args.contexts, args.kind, args.base, args.top_k,
                        args.include_nsfw, args.as_json, args.allow_stale, args.new_only,
                        family=args.family, box=args.box, semantic=args.semantic))
    elif args.cmd == "pack":
        sys.exit(pack(args.contexts, args.family, args.box, args.include_nsfw,
                      args.per_role, args.min_score, args.quirks_base, args.as_json,
                      args.allow_stale, semantic=args.semantic))
    elif args.cmd == "status":
        sys.exit(status(Path(args.index), args.max_age_hours))


if __name__ == "__main__":
    main()
