#!/usr/bin/env python3
"""Anima (flow-matching) checkpoint cards: exact loader name, text-encoder/VAE contract,
ModelSampling/shift, native steps/CFG/sampler/scheduler, Turbo-LoRA compatibility, resolution
and verification date -- so runners stop guessing or hardcoding stale defaults.

Reads scripts/anima_models.json (loader filename stem -> card). Consumed directly by
z3_anima_fast.py / z3_quality_gen.py (unlike model_quirks.py, which only the router gen.py
imports) because the P0 bug this fixes lives in the engines' own default values.

Lookup is fuzzy, mirroring model_quirks.card(): the JSON key is a loader filename stem; a
query matches if the key is a case-insensitive substring of the query (or vice versa), longest
key wins, so a full ComfyUI path like 'Anima\\anime\\miaomiaoHarem_anima8Step10.safetensors'
resolves to its card.

Usage:
  anima_models.py list
  anima_models.py lookup "<unet name or path>"      # full card (human)
  anima_models.py lookup "<unet name or path>" --json
  anima_models.py recipe "<unet name or path>"       # just steps/cfg/sampler/scheduler (JSON)
  anima_models.py turbo "<unet name or path>"        # just the turbo pairing, if any (JSON)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "anima_models.json"


def _load() -> dict:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing anima models data: {DATA}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt anima models data {DATA}: {exc}")


def card(query: str) -> dict | None:
    """Return the model card whose key is a substring of `query` (case-insensitive), else None.

    Bidirectional (key-in-query resolves a full loader path; query-in-key resolves a short
    stem) with longest key wins, mirroring model_quirks.card().
    """
    models = _load().get("models", {})
    q = query.lower()
    for key in sorted(models, key=len, reverse=True):
        k = key.lower()
        if k in q or q in k:
            return {"name": key, **models[key]}
    return None


def _require(query: str) -> dict:
    c = card(query)
    if not c:
        raise SystemExit(f"no anima model card matches {query!r} (see: anima_models.py list)")
    return c


def shared_contract() -> dict:
    """The clip_name/clip_type/vae_name constants shared by every known Anima checkpoint (verified
    across all 10 Controlled-matrix checkpoints, REPORT.md 2026-07-15). Fallback for an --unet that
    doesn't match any registered card (e.g. a brand-new checkpoint not yet added here)."""
    return _load().get("shared_contract", {})


def primary_output() -> dict:
    """The single card flagged role=='primary_output'. SystemExit if not exactly one match --
    fail loudly rather than silently guess which checkpoint is the current production default.
    """
    models = _load().get("models", {})
    matches = [{"name": k, **v} for k, v in models.items() if v.get("role") == "primary_output"]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one anima model with role=primary_output, found {len(matches)} "
            f"(see: anima_models.py list)"
        )
    return matches[0]


def turbo_profile() -> dict | None:
    """The single card flagged turbo_default==true, or None if no card is flagged."""
    models = _load().get("models", {})
    matches = [{"name": k, **v} for k, v in models.items() if v.get("turbo_default")]
    if not matches:
        return None
    if len(matches) > 1:
        raise SystemExit(
            f"expected at most one anima model with turbo_default=true, found {len(matches)} "
            f"(see: anima_models.py list)"
        )
    return matches[0]


def cmd_list(_args: argparse.Namespace) -> None:
    for key, c in _load().get("models", {}).items():
        recipe = c.get("recipe", {})
        recipe_str = f"{recipe.get('steps', '?')}st/cfg{recipe.get('cfg', '?')}/{recipe.get('sampler', '?')}"
        print(f"{key:<32} {c.get('role', '?'):<26} {recipe_str}")


def cmd_lookup(args: argparse.Namespace) -> None:
    c = _require(args.query)
    if args.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return
    print(f"# {c['name']}")
    for k in ("unet", "sha256", "civitai_url", "clip_name", "clip_type", "vae_name",
              "model_sampling", "recipe", "resolution", "turbo", "turbo_default", "role",
              "status", "training_base", "cfg_note", "verified", "evidence"):
        v = c.get(k)
        if v in (None, "", [], False):
            continue
        if isinstance(v, (list, dict)):
            rendered = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            rendered = v
        print(f"  {k:<14} {rendered}")


def cmd_recipe(args: argparse.Namespace) -> None:
    print(json.dumps(_require(args.query).get("recipe", {}), ensure_ascii=False, indent=2))


def cmd_turbo(args: argparse.Namespace) -> None:
    print(json.dumps(_require(args.query).get("turbo"), ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all anima model cards").set_defaults(func=cmd_list)
    lk = sub.add_parser("lookup", help="print a model's full card")
    lk.add_argument("query")
    lk.add_argument("--json", action="store_true")
    lk.set_defaults(func=cmd_lookup)
    rc = sub.add_parser("recipe", help="print just the steps/cfg/sampler/scheduler recipe (JSON)")
    rc.add_argument("query")
    rc.set_defaults(func=cmd_recipe)
    tb = sub.add_parser("turbo", help="print just the turbo pairing, if any (JSON)")
    tb.add_argument("query")
    tb.set_defaults(func=cmd_turbo)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
