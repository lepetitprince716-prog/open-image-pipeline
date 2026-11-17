#!/usr/bin/env python3
"""Validate and land delegated catalog-item drafts into catalogs/t2i/core.

The dispatcher hands an author model the brief template from
docs/runbooks/t2i_catalog_author_runbook.md; the author delivers one draft JSON
(schema below); this script is the only sanctioned way to move that draft into
the catalog. Validation is fail-closed: an item that breaks the layer contract
is reported and skipped, never landed. Landing only ever creates new files, so
--rollback can undo a landing exactly by deleting what it created.

Draft schema (one JSON object):
    {"scene":  [{"stem", "phrase", "time_of_day", "indoor", "weather"?, "display_name_zh"}],
     "light":  [{"stem", "phrase", "time_affinity", "hardness", "indoor_source", "display_name_zh"}],
     "effects":[{"stem", "phrase", "family", "display_name_zh"}]}

Usage:
    python3 scripts/t2i_catalog_land.py draft.json                  # dry-run: validate + report
    python3 scripts/t2i_catalog_land.py draft.json --apply          # land the valid items
    python3 scripts/t2i_catalog_land.py draft.json --apply --skip a-stem,b-stem
    python3 scripts/t2i_catalog_land.py --rollback landing.json     # delete a landing's files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for _p in (str(SCRIPTS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
CORE = ROOT / "catalogs" / "t2i" / "core"
PROVENANCE_REF = "catalogs/t2i/provenance/rotation_review_2026-08-31.json"

TIME_WORDS = ("blue hour", "golden hour", "dawn", "dusk", "midday", "night")  # multiword first
EFFECT_FAMILIES = {"petals", "mist", "dust", "rain", "snow", "night", "magic", "lens", "water"}
HARDNESS = {"hard", "soft", "mixed"}
AFFINITY = {"day", "night", "any"}
BASE_COMPAT = {"finish_color": "full", "finish_text": "none", "medium_family": "2d",
               "subject_presenting": "female", "content_rating": "PG"}
# Words other layers own outright; a delegated phrase containing one is an
# automatic reject regardless of layer (tests/test_t2i_prompt.py is authoritative,
# this is the cheap pre-check).
BANNED_EVERYWHERE = ("masterpiece", "best quality", "absurdres", "highres", "score_", "@", "_")
BANNED_BY_LAYER = {
    "scene": ("light,", "lighting", "backlit", "key light", "shadow", "glow", "close-up",
              "wide shot", "view", "angle", "smile", "expression", "1girl", "girl"),
    "light": ("1girl", "girl", "smile", "close-up", "view", "angle"),
    "effects": ("1girl", "girl", "lighting", "key light", "close-up", "view", "angle"),
}


def _time_words(text: str) -> list[str]:
    found = []
    for word in TIME_WORDS:
        if word in text:
            found.append(word)
            text = text.replace(word, "")
    return found


def _normalize_stem(layer: str, item: dict) -> None:
    """Authors sometimes deliver stems already carrying the layer prefix; strip it so
    landed ids never double up (effects.effects.x)."""
    item["stem"] = item.get("stem", "").strip().removeprefix(f"{layer}.")


def _phrase_problems(layer: str, item: dict) -> list[str]:
    phrase = item["phrase"].lower()
    problems = [f"banned token {tok!r}" for tok in BANNED_EVERYWHERE if tok in phrase]
    problems += [f"foreign-layer token {tok!r}" for tok in BANNED_BY_LAYER[layer] if tok in phrase]
    tw = _time_words(phrase)
    if layer == "scene":
        if tw != [item.get("time_of_day")]:
            problems.append(f"time words {tw} must be exactly [{item.get('time_of_day')!r}]")
        if item.get("time_of_day") not in TIME_WORDS:
            problems.append(f"unknown time_of_day {item.get('time_of_day')!r}")
        if item.get("weather", "clear") not in ("clear", "rain"):
            problems.append(f"unknown weather {item.get('weather')!r}")
        if not isinstance(item.get("indoor"), bool):
            problems.append("indoor must be true/false")
    elif layer == "light":
        if tw:
            problems.append(f"light phrase may not name a time: {tw}")
        if item.get("time_affinity") not in AFFINITY:
            problems.append(f"time_affinity must be one of {sorted(AFFINITY)}")
        if item.get("hardness") not in HARDNESS:
            problems.append(f"hardness must be one of {sorted(HARDNESS)}")
        if not isinstance(item.get("indoor_source"), bool):
            problems.append("indoor_source must be true/false")
    else:
        if item.get("family") not in EFFECT_FAMILIES:
            problems.append(f"family must be one of {sorted(EFFECT_FAMILIES)}")
    if not item.get("display_name_zh", "").strip():
        problems.append("display_name_zh is required")
    return problems


def _build(layer: str, item: dict) -> dict:
    phrase = item["phrase"].strip().rstrip(",")
    if layer == "scene":
        compat = {**BASE_COMPAT, "weather": item.get("weather", "clear"),
                  "time_of_day": item["time_of_day"], "indoor": item["indoor"]}
        phrase += ", 2d, full color, no text"
    elif layer == "light":
        compat = {**BASE_COMPAT, "hardness": item["hardness"],
                  "indoor_source": item["indoor_source"], "motivated": True,
                  "time_affinity": item["time_affinity"]}
        phrase += ", 2d"
    else:
        compat = {**BASE_COMPAT, "family": item["family"], "intensity": "moderate",
                  "attaches_to": "scene"}
        phrase += ", 2d, full color, no text"
    built = {"schema": "loramake.t2i-catalog-item/v1", "namespace": "core",
            "id": f"{layer}.{item['stem']}", "revision": 1, "layer": layer,
            "category": "authored-delegated", "weight": 1.0, "requirements": [],
            "status": "machine_validated", "display_name": item["display_name_zh"],
            "description": f"delegated authoring, landed via t2i_catalog_land.py: {item['phrase']}",
            "source": {"kind": "authored", "ref": PROVENANCE_REF},
            "dialects": {"anima": phrase}, "compatibility": compat}
    tags = item.get("tags")
    if isinstance(tags, list):
        cleaned = [str(t).strip() for t in tags if str(t).strip()]
        if cleaned:
            built["tags"] = cleaned
    return built


def land(draft_path: Path, apply: bool, skip: set[str]) -> int:
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    unknown = set(draft) - {"scene", "light", "effects"}
    if unknown:
        print(f"error: unknown draft sections {sorted(unknown)}", file=sys.stderr)
        return 2
    landed, rejects, skipped = [], [], []
    for layer in ("scene", "light", "effects"):
        for item in draft.get(layer, []):
            _normalize_stem(layer, item)
            stem = item["stem"]
            full_id = f"{layer}.{stem}"
            if not stem:
                rejects.append((full_id, ["missing stem"]))
                continue
            if stem in skip:
                skipped.append(full_id)
                continue
            problems = _phrase_problems(layer, item)
            if (CORE / layer / f"{full_id}.json").exists():
                problems.append("id already exists in the catalog")
            if problems:
                rejects.append((full_id, problems))
                continue
            if apply:
                path = CORE / layer / f"{full_id}.json"
                path.write_text(json.dumps(_build(layer, item), ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            landed.append(full_id)
    mode = "landed" if apply else "valid (dry-run, nothing written)"
    print(f"{mode}: {len(landed)}  rejected: {len(rejects)}  skipped: {len(skipped)}")
    for full_id, problems in rejects:
        print(f"  REJECT {full_id}: " + "; ".join(problems))
    if apply and landed:
        manifest = draft_path.with_name(draft_path.stem + ".landing.json")
        created = [str((CORE / i.split(".", 1)[0] / f"{i}.json").relative_to(ROOT)) for i in landed]
        manifest.write_text(json.dumps({"created": created}, indent=2) + "\n", encoding="utf-8")
        print(f"landing manifest: {manifest}")
        print("next: python3 -m pytest tests/test_t2i_prompt.py tests/test_t2i_gacha.py -q")
        print(f"undo: python3 scripts/t2i_catalog_land.py --rollback {manifest}")
    return 0 if not rejects else 1


def audition(draft_path: Path, base_seed: int, limit: int | None) -> int:
    """Render one image per valid draft item (nothing lands) and register a review
    batch: the user's love/yes there is the landing list, no is the swap list."""
    import t2i_gacha  # sibling; brings the engine + gate plumbing (and sys.path setup)
    import t2i_prompt  # sibling

    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    catalog = t2i_prompt.load_catalog()
    rules = t2i_prompt.load_rules()
    jobs = []
    for layer in ("scene", "light", "effects"):
        for item in draft.get(layer, []):
            _normalize_stem(layer, item)
            if _phrase_problems(layer, item):
                continue  # dry-run reports these; the audition shows only valid items
            jobs.append((layer, item))
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        print("error: no valid items to audition (run the dry-run for reasons)", file=sys.stderr)
        return 1

    import gen  # heavy sibling
    from loramake.comfy import BOXA_HOST, free_vram
    ok, reason = gen.wait_for_idle(BOXA_HOST)
    if not ok:
        print(f"error: BOXA not idle: {reason}", file=sys.stderr)
        return 1
    spec, failures = [], 0
    try:
        for n, (layer, item) in enumerate(jobs, 1):
            seed = base_seed + n
            picked = t2i_prompt.sample(catalog, rules, seed, "natural",
                                       status_overrides=t2i_gacha.status_overrides_for(frozenset()))
            picked = {**picked, layer: _build(layer, item)}  # the draft item takes its layer
            picked, _ = t2i_gacha.ensure_garment(
                picked, catalog, t2i_gacha._derived_rng(seed, "garment"))
            prompt = t2i_prompt.render(picked)
            rec = {"seed": seed, "prompt": prompt, "picks": {}, "wild_layers": [],
                   "blend_artist": None, "style_lora": None}
            path = t2i_gacha.gen_one(rec)
            if path is None:
                failures += 1
                if n == 1:
                    print("error: first render failed; aborting the audition", file=sys.stderr)
                    return 1
                continue
            fid = f"{layer}.{item['stem']}"
            spec.append({"path": path, "caption": f"{fid} · {item['display_name_zh']} · seed {seed}"})
            print(f"  [{n:02d}/{len(jobs)}] {fid} -> {Path(path).name}")
    finally:
        free_vram(BOXA_HOST)

    import os
    import qc_score  # sibling
    import sfw_gate  # sibling
    venv = os.environ.get("IMGUTILS_VENV_PY", qc_score.DEFAULT_VENV_PY)
    ratings = t2i_gacha.wd14_many(venv, [s["path"] for s in spec])
    shown = [s for s in spec if sfw_gate.image_ok(ratings.get(s["path"])) is True]
    dropped = len(spec) - len(shown)
    spec_path = draft_path.with_name(draft_path.stem + ".audition_spec.json")
    spec_path.write_text(json.dumps(shown, ensure_ascii=False, indent=2), encoding="utf-8")
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "review_batch.py"),
                          "--mode", "review", "--spec", str(spec_path), "--json"],
                         capture_output=True, text=True)
    print(f"auditioned {len(shown)} items ({failures} render failures, {dropped} held by the SFW gate)")
    if out.returncode == 0:
        print("review: " + json.loads(out.stdout)["url"])
        print("then: love/yes = keep; collect the 'no' stems and land with "
              f"`t2i_catalog_land.py {draft_path.name} --apply --skip <no-stems>`")
    else:
        print("review_batch registration failed:\n" + (out.stderr or out.stdout), file=sys.stderr)
        return 1
    return 0


def rollback(manifest_path: Path) -> int:
    created = json.loads(manifest_path.read_text(encoding="utf-8"))["created"]
    removed = 0
    for rel in created:
        p = ROOT / rel
        if p.is_file() and p.resolve().is_relative_to(CORE.resolve()):
            p.unlink()
            removed += 1
    print(f"removed {removed}/{len(created)} landed files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("draft", nargs="?", help="draft JSON from the author model")
    parser.add_argument("--apply", action="store_true", help="write the valid items (default: dry-run)")
    parser.add_argument("--skip", default="", metavar="STEM[,STEM...]",
                        help="reviewer-rejected stems to leave out")
    parser.add_argument("--rollback", metavar="LANDING.json",
                        help="delete every file a previous --apply created")
    parser.add_argument("--audition", action="store_true",
                        help="render one image per valid item (nothing lands) and open a review batch")
    parser.add_argument("--seed", type=int, default=None, help="audition base seed")
    parser.add_argument("--limit", type=int, default=None, help="audition at most N items")
    args = parser.parse_args(argv)
    if args.rollback:
        return rollback(Path(args.rollback))
    if not args.draft:
        parser.error("draft JSON path required (or --rollback)")
    if args.audition:
        import random
        base = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
        return audition(Path(args.draft), base, args.limit)
    return land(Path(args.draft), args.apply, {s for s in args.skip.split(",") if s})


if __name__ == "__main__":
    raise SystemExit(main())
