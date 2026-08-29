#!/usr/bin/env python3
"""Bridge the comfyui-xin booru_distill tag assets into the loramake prompt layer.

Three read-only-upstream functions (outputs land in wildcards/_candidates/ for HUMAN review —
this script never touches the curated wildcards/*.txt or art_prompt.py itself):

  candidates   Grok-reviewed SFW prompt candidates (sfw_prompt_candidates.jsonl, g-rated +
               default_prompt_eligible) mapped by prompt_slot onto our wildcard dimensions ->
               wildcards/_candidates/<dim>.candidates.txt. Identity/subject-owned slots
               (subject/hair/skin_eye/outfit) are listed in _excluded_slots.md instead —
               those belong to the character spec, not the variation layer.
  lint         Check every wildcards/*.txt line and art_prompt.NEGATIVE tag against the
               danbooru alias graph (tag_edges) and tag cards (post_count): non-canonical
               aliases -> suggest the canonical name (Illustrious/NoobAI are trained on
               canonical danbooru tags, so canonical = stronger adherence); post_count < 200
               -> weak-adherence warning (same principle as ARTISTIC_LINE's >=200-post artist
               rule). Freeform English phrases that are not danbooru tags are reported as
               'unknown' (informational — natural language is legal, tags just steer harder).
  trigger-diff Cross-check trigger_map.jsonl lora_matches against lora_catalog/z3_index.json:
               tags that reference a Z3-present LoRA whose catalog entry does not carry the
               tag in its trigger text (catalog gap candidates).

Upstream (read-only): comfyui-xin booru_distill data — override root via BOORU_DISTILL_DIR.
Idempotent: candidate/report files are rewritten in full on every run (upstream jsonl grows).

Usage:
  python3 scripts/booru_wildcard_distill.py candidates
  python3 scripts/booru_wildcard_distill.py lint
  python3 scripts/booru_wildcard_distill.py trigger-diff
  python3 scripts/booru_wildcard_distill.py all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WILDCARDS = ROOT / "wildcards"
OUT = WILDCARDS / "_candidates"
BOORU = Path(os.environ.get(
    "BOORU_DISTILL_DIR",
    str(ROOT / "comfyui-z3" / "AI绘图知识库" / "13-出图系统Harness" / "booru_distill" / "data")))
CATALOG = ROOT / "lora_catalog" / "z3_index.json"

# prompt_slot -> our wildcard dimension. Slots owned by the character spec are deliberately
# excluded from the variation layer (see module docstring).
SLOT_TO_DIM = {
    "pose_action": "pose",
    "hand_foot_action": "pose",
    "location": "background",
    "lighting": "lighting",
    "color_palette": "color_mood",
    "face_expression": "expression",
    "body_framing": "camera",
    "composition": "camera",
    "prop": "prop",  # new-dimension proposal: no wildcards/prop.txt yet
}
EXCLUDED_SLOTS = ("subject", "hair", "skin_eye", "outfit")
WEAK_POST_COUNT = 200  # below this a general tag steers weakly (ARTISTIC_LINE artist-rule generalised)


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm(tag: str) -> str:
    """Prompt text -> danbooru lookup form: lowercase, unescape \\( \\), spaces -> underscores."""
    return tag.strip().lower().replace("\\(", "(").replace("\\)", ")").replace(" ", "_")


def _prompt_form(tag: str) -> str:
    """Danbooru tag -> prompt text form used in our wildcards: underscores -> spaces, escape parens."""
    return tag.replace("_", " ").replace("(", "\\(").replace(")", "\\)")


def _load_alias_map() -> dict[str, str]:
    alias: dict[str, str] = {}
    for e in _jsonl(BOORU / "tag_edges.jsonl"):
        if e.get("edge_type") == "alias" and e.get("status") == "active":
            alias[_norm(e["source_tag"])] = e["target_tag"]
    return alias


def _load_post_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in _jsonl(BOORU / "tag_cards.jsonl"):
        try:
            counts[_norm(c["tag"])] = int(c.get("post_count") or 0)
        except (TypeError, ValueError):
            continue
    for e in _jsonl(BOORU / "tag_edges.jsonl"):  # edges sometimes carry target counts
        tgt, n = e.get("target_tag"), e.get("target_post_count")
        if tgt and n and _norm(tgt) not in counts:
            counts[_norm(tgt)] = int(n)
    return counts


def _existing_lines(dim: str) -> set[str]:
    f = WILDCARDS / f"{dim}.txt"
    if not f.exists():
        return set()
    return {_norm(l) for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")}


def cmd_candidates() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_dim: dict[str, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    for c in _jsonl(BOORU / "curated" / "sfw_prompt_candidates.jsonl"):
        if not c.get("default_prompt_eligible") or "g" not in (c.get("ratings") or []):
            continue
        slot = c.get("prompt_slot", "")
        if slot in EXCLUDED_SLOTS:
            excluded.append(c)
        elif slot in SLOT_TO_DIM:
            by_dim[SLOT_TO_DIM[slot]].append(c)
    total = 0
    for dim, rows in sorted(by_dim.items()):
        have = _existing_lines(dim)
        lines = [f"# CANDIDATES for wildcards/{dim}.txt — generated by booru_wildcard_distill.py",
                 "# Human-review before merging; this file is NOT loaded by the wildcard engine.",
                 f"# Source: booru_distill sfw_prompt_candidates (g-rated, default_prompt_eligible).{'' if (WILDCARDS / (dim + '.txt')).exists() else '  NOTE: new dimension proposal — wildcards/' + dim + '.txt does not exist yet.'}",
                 ""]
        n = 0
        for c in sorted(rows, key=lambda r: r["tag"]):
            if _norm(c["tag"]) in have:
                continue  # already curated into the live wildcard
            note = (c.get("usage_note") or "").strip().replace("\n", " ")
            lines.append(f"# note: {note[:160]}")
            lines.append(_prompt_form(c["tag"]))
            lines.append("")
            n += 1
        (OUT / f"{dim}.candidates.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"candidates: {dim:<12} {n} new (of {len(rows)} eligible)")
        total += n
    ex_lines = ["# Slots excluded from the variation layer (identity/subject-owned) — for reference only.", ""]
    for c in sorted(excluded, key=lambda r: (r["prompt_slot"], r["tag"])):
        ex_lines.append(f"- [{c['prompt_slot']}] {c['tag']} — {(c.get('usage_note') or '')[:120]}")
    (OUT / "_excluded_slots.md").write_text("\n".join(ex_lines), encoding="utf-8")
    print(f"candidates: total {total} new lines across {len(by_dim)} dims; "
          f"{len(excluded)} identity-slot entries -> _excluded_slots.md")


def _iter_prompt_tags():
    """Yield (origin, tag_text) for every tag-ish segment we curate by hand."""
    for f in sorted(WILDCARDS.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for seg in line.split(","):
                if seg.strip():
                    yield f"wildcards/{f.name}", seg.strip()
    sys.path.insert(0, str(ROOT / "scripts"))
    import art_prompt  # noqa: E402  (pure string module, no GPU)
    for seg in art_prompt.NEGATIVE.split(","):
        if seg.strip():
            yield "art_prompt.NEGATIVE", seg.strip()


def cmd_lint() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alias, counts = _load_alias_map(), _load_post_counts()
    rows, n_alias, n_weak, n_unknown = [], 0, 0, 0
    for origin, seg in _iter_prompt_tags():
        key = _norm(seg)
        if key in alias:
            rows.append(f"| {origin} | `{seg}` | ALIAS | canonical: `{_prompt_form(alias[key])}` |")
            n_alias += 1
        elif key in counts:
            if counts[key] < WEAK_POST_COUNT:
                rows.append(f"| {origin} | `{seg}` | WEAK | post_count={counts[key]} < {WEAK_POST_COUNT} |")
                n_weak += 1
        else:
            n_unknown += 1  # freeform phrase or outside the local 404-card snapshot — informational
    report = [
        "# Wildcard / negative tag lint — booru_wildcard_distill.py",
        "",
        f"Checked against booru_distill snapshot: {len(alias)} alias edges, {len(counts)} tag cards.",
        f"Actionable: {n_alias} alias hits, {n_weak} weak-tag hits. "
        f"({n_unknown} segments unknown to the local card snapshot — freeform phrases are fine, "
        "tags just steer harder; coverage is only 404 cards so absence is NOT a warning.)",
        "",
        "| origin | segment | issue | detail |",
        "|---|---|---|---|",
        *rows,
        "",
        "Fixes are applied by hand (curated assets stay human-owned).",
    ]
    (OUT / "lint_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"lint: {n_alias} alias, {n_weak} weak, {n_unknown} unknown -> {OUT / 'lint_report.md'}")


def cmd_trigger_diff() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_file = {}
    for e in json.loads(CATALOG.read_text(encoding="utf-8")):
        by_file[Path(e.get("file", "")).name.lower()] = e
    rows = []
    for t in _jsonl(BOORU / "trigger_map.jsonl"):
        tag = t.get("tag", "")
        for m in t.get("lora_matches") or []:
            fname = Path(str(m.get("lora_name", "")).replace("\\", "/")).name.lower()
            entry = by_file.get(fname)
            if not entry:
                continue  # LoRA not on Z3 — out of scope
            trig_text = " ".join(entry.get("triggers") or []).lower()
            if _norm(tag).replace("_", " ") not in trig_text and _norm(tag) not in trig_text:
                rows.append(f"| `{tag}` | {entry['name'][:60]} | {fname} |")
    report = [
        "# trigger_map ↔ lora_catalog cross-check — booru_wildcard_distill.py",
        "",
        "Tags whose booru_distill lora_matches point at a Z3-present LoRA, but the catalog entry's",
        "trigger text does not contain the tag (catalog gap candidates — verify before editing).",
        "",
        "| tag | catalog entry | file |",
        "|---|---|---|",
        *rows,
    ]
    (OUT / "trigger_diff.md").write_text("\n".join(report), encoding="utf-8")
    print(f"trigger-diff: {len(rows)} gap candidates -> {OUT / 'trigger_diff.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("candidates", "lint", "trigger-diff", "all"))
    args = ap.parse_args()
    if not BOORU.exists():
        sys.exit(f"booru_distill data not found: {BOORU} (set BOORU_DISTILL_DIR)")
    if args.cmd in ("candidates", "all"):
        cmd_candidates()
    if args.cmd in ("lint", "all"):
        cmd_lint()
    if args.cmd in ("trigger-diff", "all"):
        cmd_trigger_diff()


if __name__ == "__main__":
    main()
