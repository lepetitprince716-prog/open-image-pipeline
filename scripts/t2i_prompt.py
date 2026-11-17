#!/usr/bin/env python3
"""Random prompt stacking from the t2i catalog.

Picks one catalog item per layer with weighted random sampling, drops
combinations forbidden by hard rule cards, down-weights the ones soft rules
discourage, and joins the picked phrases into a prompt.

Each layer draws from its own seeded random source, so pinning or rerolling
one layer leaves every other layer's pick unchanged.

    python3 scripts/t2i_prompt.py --count 5
    python3 scripts/t2i_prompt.py --seed 42 --profile natural --json
    python3 scripts/t2i_prompt.py --seed 42 --pin style.prompt=style.prompt.gouache-atdan
    python3 scripts/t2i_prompt.py --seed 42 --reroll scene,light
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "catalogs" / "t2i"

# Prompt layers in the order their phrases are concatenated.
LAYER_ORDER = [
    "character.body",
    "character.face",
    "hair.identity",
    "eyes",
    "expression",
    "outfit.component",
    "action",
    "scene",
    "camera.layout",
    "light",
    "color",
    "effects",
    "style.prompt",
]

# "natural" only draws from hand-reviewed items; "inspiration" also allows
# machine-validated and candidate ones and ignores the soft weight rules.
PROFILE_STATUS = {
    "natural": {"verified"},
    "inspiration": {"verified", "machine_validated", "candidate"},
}

MAX_RESAMPLES = 200


class CatalogError(RuntimeError):
    pass


def _read_json_files(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def load_catalog(catalog_dir: Path = CATALOG_DIR) -> dict[str, list[dict[str, Any]]]:
    """Group catalog items by layer, keeping only the layers we stack."""
    layers: dict[str, list[dict[str, Any]]] = {layer: [] for layer in LAYER_ORDER}
    for item in _read_json_files(catalog_dir / "core"):
        layer = item.get("layer")
        if layer in layers:
            layers[layer].append(item)
    empty = [layer for layer, items in layers.items() if not items]
    if empty:
        raise CatalogError(f"catalog has no items for layers: {', '.join(empty)}")
    return layers


def load_rules(catalog_dir: Path = CATALOG_DIR) -> list[dict[str, Any]]:
    """Load verified rule cards; unverified ones are drafts and never apply."""
    return [r for r in _read_json_files(catalog_dir / "rules") if r.get("status") == "verified"]


def _split_path(path: str, layers: Iterable[str]) -> tuple[str, str]:
    """Split "outfit.component.motion_allow" into ("outfit.component", "motion_allow")."""
    for layer in layers:
        if path.startswith(layer + "."):
            return layer, path[len(layer) + 1 :]
    layer, _, key = path.rpartition(".")
    return layer, key


def _field(item: dict[str, Any], key: str) -> Any:
    """Rule paths address compatibility keys first, then top-level item fields."""
    compat = item.get("compatibility", {})
    return compat[key] if key in compat else item.get(key)


def _clause_holds(clause: dict[str, Any], picked: dict[str, dict[str, Any]]) -> bool:
    layer, key = _split_path(clause["path"], LAYER_ORDER)
    item = picked.get(layer)
    if item is None:
        return False  # layer not chosen yet: the rule cannot fire
    actual = _field(item, key)
    expected = clause.get("value")
    op = clause["op"]
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "contains":
        return isinstance(actual, (list, tuple)) and expected in actual
    if op == "not_contains":
        return isinstance(actual, (list, tuple)) and expected not in actual
    raise CatalogError(f"unsupported rule op: {op}")


def weigh(
    item: dict[str, Any],
    layer: str,
    picked: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
    profile: str,
) -> float:
    """Return the sampling weight of `item`, or 0.0 if a hard rule forbids it.

    Hard rules always apply; soft rules apply under `natural` only.

    A rule biases whichever of its `when` layers comes last in LAYER_ORDER --
    clauses on earlier layers are conditions. A rule naming only already-picked
    layers scales every candidate equally and so does nothing.
    """
    context = {**picked, layer: item}
    weight = float(item.get("weight", 1.0))
    for rule in rules:
        if not all(_clause_holds(clause, context) for clause in rule["when"]):
            continue
        if rule["kind"] == "hard":
            return 0.0
        if profile == "inspiration":
            continue  # soft rules are advisory; inspiration ignores them
        weight *= float(rule["effect"]["value"])
    return weight


def _layer_rng(seed: int, layer: str, generation: int, attempt: int) -> random.Random:
    """A random source private to one layer, so other layers stay put."""
    key = f"{seed}\0{layer}\0{generation}\0{attempt}".encode()
    return random.Random(hashlib.sha256(key).digest())


def sample(
    catalog: dict[str, list[dict[str, Any]]],
    rules: list[dict[str, Any]],
    seed: int,
    profile: str,
    pins: dict[str, str] | None = None,
    rerolls: frozenset[str] = frozenset(),
    status_overrides: dict[str, frozenset[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pick one item per layer. Restarts if a layer paints itself into a corner.

    `status_overrides` widens (or narrows) the status pool for individual layers
    without touching the others -- how t2i_gacha's chaos quota lets a few "wild"
    layers draw from the full pool while the rest stay on the profile's set.
    Rules are unaffected: hard rules still veto, soft rules still follow `profile`.
    """
    allowed_status = PROFILE_STATUS[profile]
    pins = pins or {}
    status_overrides = status_overrides or {}
    for attempt in range(MAX_RESAMPLES):
        picked: dict[str, dict[str, Any]] = {}
        for layer in LAYER_ORDER:
            pinned_id = pins.get(layer)
            if pinned_id is not None:
                picked[layer] = next(i for i in catalog[layer] if i["id"] == pinned_id)
                continue
            allowed = status_overrides.get(layer, allowed_status)
            candidates = [i for i in catalog[layer] if i.get("status") in allowed]
            weights = [weigh(item, layer, picked, rules, profile) for item in candidates]
            if not any(weights):
                break  # dead end, start over with a fresh draw
            rng = _layer_rng(seed, layer, 1 if layer in rerolls else 0, attempt)
            picked[layer] = rng.choices(candidates, weights=weights, k=1)[0]
        if len(picked) == len(LAYER_ORDER):
            return picked
    hint = " (pinned items may conflict)" if pins else ""
    raise CatalogError(f"no rule-satisfying combination found after {MAX_RESAMPLES} attempts{hint}")


def item_tags(item: dict[str, Any]) -> list[str]:
    """Optional IR tokens stored beside dialects.anima. Empty if the item is phrase-only."""
    raw = item.get("tags")
    if not isinstance(raw, list):
        return []
    return [str(tag).strip() for tag in raw if str(tag).strip()]


def compile_tags(picked: dict[str, dict[str, Any]]) -> list[str]:
    """Union optional tags[] from each picked layer. Does not clone dialects.anima."""
    out: list[str] = []
    seen: set[str] = set()
    for layer in LAYER_ORDER:
        item = picked.get(layer) or {}
        for tag in item_tags(item):
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tag)
    return out


def render(picked: dict[str, dict[str, Any]], dialect: str = "anima") -> str:
    """Join the picked phrases, dropping duplicates while keeping layer order."""
    parts: list[str] = []
    seen: set[str] = set()
    for layer in LAYER_ORDER:
        phrase = picked[layer].get("dialects", {}).get(dialect, "").strip()
        for tag in (t.strip() for t in phrase.split(",")):
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                parts.append(tag)
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible prompts")
    parser.add_argument("--count", type=int, default=1, help="how many prompts to emit")
    parser.add_argument("--profile", choices=sorted(PROFILE_STATUS), default="inspiration")
    parser.add_argument("--dialect", default="anima", help="which dialect field to read phrases from")
    parser.add_argument("--json", action="store_true", help="emit prompt plus the seed and picked item ids")
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="LAYER=ITEM_ID",
        help="hold one layer to a fixed item; repeatable",
    )
    parser.add_argument(
        "--reroll",
        default="",
        metavar="LAYER[,LAYER...]",
        help="redraw only these layers, leaving the rest of the seed's picks intact",
    )
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be at least 1")

    try:
        catalog = load_catalog()
        rules = load_rules()
    except (CatalogError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pins: dict[str, str] = {}
    for spec in args.pin:
        layer, _, item_id = spec.partition("=")
        if layer not in catalog:
            parser.error(f"--pin: unknown layer {layer!r}; choose from {', '.join(LAYER_ORDER)}")
        if not any(i["id"] == item_id for i in catalog[layer]):
            parser.error(f"--pin: layer {layer} has no item {item_id!r}")
        pins[layer] = item_id

    rerolls = frozenset(name for name in args.reroll.split(",") if name)
    unknown = sorted(rerolls - set(catalog))
    if unknown:
        parser.error(f"--reroll: unknown layers: {', '.join(unknown)}")

    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    for offset in range(args.count):
        seed = base_seed + offset
        try:
            picked = sample(catalog, rules, seed, args.profile, pins, rerolls)
        except CatalogError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        prompt = render(picked, args.dialect)
        if args.json:
            payload = {
                "seed": seed,
                "prompt": prompt,
                "picks": {k: v["id"] for k, v in picked.items()},
                "tags": compile_tags(picked),
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            if args.seed is None:
                print(f"seed {seed}", file=sys.stderr)
            print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
