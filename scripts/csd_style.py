#!/usr/bin/env python3
"""CSD style-embedding stage, report and pre-registration dedup: the Mac side of the S4 gate.

    python3 scripts/csd_style.py stage --batch csd_r1 --review anima_lora_r2 --review illustrious_lora_r1
    # transfer and compute through the managed wrappers, one per Bash call:
    #   claude-scp -r outputs/csd/stage/csd_r1 boxa:'/opt/loramake/csd/in/'
    #   rdbg ps boxa --file <runner.ps1>   ->  python.exe boxa_csd_embed.py --in ... --out ...
    #   claude-scp boxa:'/opt/loramake/csd/out/csd_r1.json' outputs/csd/csd_r1.json
    python3 scripts/csd_style.py report --batch csd_r1 [--top 3] [--json]
    python3 scripts/csd_style.py compare --batch csd_r1 --a anima_lora_r2 --b illustrious_lora_r1

``stage`` copies the images of one or more review batches into one flat directory with an
index (review batch, item index, caption) and drops ``boxa_csd_embed.py`` next to them, so
the BOXA side needs no repository knowledge. ``report`` reads the embeddings back: nearest
neighbours by style cosine, per-review spread (mean pairwise style similarity; lower means
more varied) and near-duplicate pairs. ``compare`` matches every item of one review batch
to its nearest in another, the cross-line check of the style registry.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CSD_ROOT = REPO / "outputs" / "csd"
REVIEW_ROOT = REPO / "outputs" / "_gallery" / "review_batches"
EMBED_SCRIPT = REPO / "scripts" / "boxa_csd_embed.py"
DUPLICATE_THRESHOLD = 0.98


class CsdError(ValueError):
    pass


def stage(batch: str, reviews: list[str], *, csd_root: Path = CSD_ROOT, review_root: Path = REVIEW_ROOT,
          repo: Path = REPO, embed_script: Path = EMBED_SCRIPT) -> dict:
    """Copy the images of the given review batches into outputs/csd/stage/<batch>/ with an index."""
    if not reviews:
        raise CsdError("stage needs at least one --review batch")
    stage_dir = csd_root / "stage" / batch
    if stage_dir.exists():
        raise CsdError(f"stage directory exists: {stage_dir}")
    stage_dir.mkdir(parents=True)
    entries = []
    for review in reviews:
        manifest_path = review_root / review / "manifest.json"
        if not manifest_path.is_file():
            raise CsdError(f"review batch not found: {review}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            if item.get("media", "image") != "image":
                continue
            source = repo / item["path"]
            if not source.is_file():
                raise CsdError(f"missing image for {review} item {item['i']}: {source}")
            name = f"{review}__{item['i']:03d}{source.suffix.lower()}"
            shutil.copyfile(source, stage_dir / name)
            entries.append({"name": name, "review": review, "i": item["i"],
                            "caption": item.get("caption", ""), "source": item["path"]})
    if embed_script.is_file():
        shutil.copyfile(embed_script, stage_dir / embed_script.name)
    index = {"v": 1, "batch": batch, "reviews": reviews, "items": entries}
    (stage_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


def load_batch(batch: str, *, csd_root: Path = CSD_ROOT) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Index entries joined with their embeddings: (entries, style matrix, content matrix)."""
    index_path = csd_root / "stage" / batch / "index.json"
    embed_path = csd_root / f"{batch}.json"
    if not index_path.is_file():
        raise CsdError(f"no stage index for {batch}: {index_path}")
    if not embed_path.is_file():
        raise CsdError(f"no embeddings for {batch}: {embed_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    embeddings = json.loads(embed_path.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in embeddings.get("items", [])}
    entries, styles, contents = [], [], []
    for entry in index["items"]:
        vectors = by_name.get(entry["name"])
        if vectors is None:
            continue
        entries.append(entry)
        styles.append(vectors["style"])
        contents.append(vectors["content"])
    if not entries:
        raise CsdError(f"no embedded items for {batch}")
    return entries, _normalize(np.asarray(styles, dtype=np.float64)), _normalize(np.asarray(contents, dtype=np.float64))


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def similarity(matrix: np.ndarray) -> np.ndarray:
    return matrix @ matrix.T


def spread(sim: np.ndarray) -> dict:
    """Mean, min and max pairwise similarity over the off-diagonal (n >= 2)."""
    n = sim.shape[0]
    if n < 2:
        return {"n": n, "mean": None, "min": None, "max": None, "spread": None}
    off = sim[~np.eye(n, dtype=bool)]
    return {"n": n, "mean": round(float(off.mean()), 4), "min": round(float(off.min()), 4),
            "max": round(float(off.max()), 4), "spread": round(float(1.0 - off.mean()), 4)}


def neighbours(entries: list[dict], sim: np.ndarray, top: int) -> list[dict]:
    rows = []
    for i, entry in enumerate(entries):
        order = [j for j in np.argsort(-sim[i]) if j != i][:top]
        rows.append({"name": entry["name"], "caption": entry["caption"],
                     "neighbours": [{"name": entries[j]["name"], "caption": entries[j]["caption"],
                                     "sim": round(float(sim[i, j]), 4)} for j in order]})
    return rows


def duplicates(entries: list[dict], sim: np.ndarray, threshold: float = DUPLICATE_THRESHOLD) -> list[dict]:
    pairs = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                pairs.append({"a": entries[i]["name"], "b": entries[j]["name"], "sim": round(float(sim[i, j]), 4)})
    return pairs


def report(batch: str, *, top: int = 3, csd_root: Path = CSD_ROOT) -> dict:
    entries, styles, _contents = load_batch(batch, csd_root=csd_root)
    sim = similarity(styles)
    groups = {}
    for review in sorted({entry["review"] for entry in entries}):
        rows = [i for i, entry in enumerate(entries) if entry["review"] == review]
        groups[review] = spread(sim[np.ix_(rows, rows)])
    return {"batch": batch, "count": len(entries), "overall": spread(sim), "groups": groups,
            "duplicates": duplicates(entries, sim), "neighbours": neighbours(entries, sim, top)}


def compare(batch: str, review_a: str, review_b: str, *, csd_root: Path = CSD_ROOT) -> dict:
    entries, styles, _contents = load_batch(batch, csd_root=csd_root)
    rows_a = [i for i, entry in enumerate(entries) if entry["review"] == review_a]
    rows_b = [i for i, entry in enumerate(entries) if entry["review"] == review_b]
    if not rows_a or not rows_b:
        raise CsdError(f"both review batches must be staged in {batch}: {review_a}, {review_b}")
    cross = styles[rows_a] @ styles[rows_b].T
    matches = []
    for a_pos, i in enumerate(rows_a):
        best = int(np.argmax(cross[a_pos]))
        matches.append({"a": entries[i]["name"], "a_caption": entries[i]["caption"],
                        "b": entries[rows_b[best]]["name"], "b_caption": entries[rows_b[best]]["caption"],
                        "sim": round(float(cross[a_pos, best]), 4)})
    return {"batch": batch, "a": review_a, "b": review_b, "mean_best": round(float(np.mean([m["sim"] for m in matches])), 4),
            "matches": matches}


def _print_report(result: dict) -> None:
    print(f"{result['batch']}: {result['count']} items, overall mean sim {result['overall']['mean']} "
          f"(spread {result['overall']['spread']})")
    for review, stats in result["groups"].items():
        print(f"  {review:28s} n={stats['n']:<3d} mean {stats['mean']} min {stats['min']} max {stats['max']} spread {stats['spread']}")
    if result["duplicates"]:
        print("near-duplicates:")
        for pair in result["duplicates"]:
            print(f"  {pair['sim']:.4f}  {pair['a']}  ~  {pair['b']}")
    print("nearest neighbours by style:")
    for row in result["neighbours"]:
        first = row["neighbours"][0] if row["neighbours"] else None
        if first:
            print(f"  {row['name']:36s} -> {first['name']:36s} {first['sim']:.4f}")


def _print_compare(result: dict) -> None:
    print(f"{result['a']} -> {result['b']}: mean best sim {result['mean_best']}")
    for match in result["matches"]:
        print(f"  {match['sim']:.4f}  {match['a']:36s} -> {match['b']}")


def dedup(batch: str, against: list[str], *, threshold: float = DUPLICATE_THRESHOLD,
          csd_root: Path = CSD_ROOT) -> dict:
    """Items of ``batch`` whose nearest style neighbour in ``against`` is at or above threshold.

    The pre-registration gate of the style plan: a candidate that already has a near-twin in
    the corpus adds nothing to a review batch, so it is skipped rather than judged again.
    Fail-closed — a batch without embeddings raises instead of registering unchecked. Passing
    the candidate batch as its own corpus is harmless: only the self-comparison is suppressed,
    so the same image staged into a *different* batch is still reported as the duplicate it is.
    """
    entries, styles, _ = load_batch(batch, csd_root=csd_root)
    corpus_entries: list[dict] = []
    corpus_rows: list[np.ndarray] = []
    for name in against:
        other_entries, other_styles, _ = load_batch(name, csd_root=csd_root)
        corpus_entries.extend({**entry, "batch": name} for entry in other_entries)
        corpus_rows.append(other_styles)
    if not corpus_entries:
        raise CsdError("no corpus batches to compare against")
    sim = styles @ np.vstack(corpus_rows).T
    # An item is not its own duplicate, but the SAME image staged into a different CSD batch
    # is exactly the hit this gate exists to catch, so only suppress the self-comparison.
    if batch in against:
        by_name = {entry["name"]: i for i, entry in enumerate(entries)}
        for j, other in enumerate(corpus_entries):
            if other["batch"] == batch and other["name"] in by_name:
                sim[by_name[other["name"]], j] = -1.0
    skip, keep = [], []
    for i, entry in enumerate(entries):
        j = int(np.argmax(sim[i]))
        best = float(sim[i, j])
        row = {"name": entry["name"], "caption": entry["caption"], "sim": round(best, 4),
               "nearest": corpus_entries[j]["name"], "nearest_batch": corpus_entries[j]["batch"],
               "nearest_caption": corpus_entries[j]["caption"]}
        (skip if best >= threshold else keep).append(row)
    return {"batch": batch, "against": list(against), "threshold": threshold,
            "n": len(entries), "skip": skip, "keep": keep}


def _print_dedup(result: dict) -> None:
    print(f"{result['batch']}: {len(result['keep'])} keep, {len(result['skip'])} skip "
          f"(nearest in {', '.join(result['against'])} >= {result['threshold']})")
    for row in result["skip"]:
        print(f"  skip {row['name']}  sim {row['sim']}  <- {row['nearest_batch']}/{row['nearest']}")
        print(f"       {row['caption']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSD style embeddings: stage, report, compare")
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("stage", help="collect review batch images for the BOXA embed run")
    st.add_argument("--batch", required=True)
    st.add_argument("--review", action="append", required=True, dest="reviews")
    rp = sub.add_parser("report", help="neighbours, spread and duplicates from the embeddings")
    rp.add_argument("--batch", required=True)
    rp.add_argument("--top", type=int, default=3)
    rp.add_argument("--json", action="store_true")
    dd = sub.add_parser(
        "dedup",
        help="skip candidates whose nearest neighbour in a corpus is a near-twin",
        description="Exits 1 when anything is skipped, so a scripted caller can branch on it. "
                    "That is the gate doing its job, not an execution failure; a missing "
                    "embedding is a hard error instead.",
    )
    dd.add_argument("--batch", required=True)
    dd.add_argument("--against", action="append", required=True,
                    help="corpus batch to compare against (repeatable)")
    dd.add_argument("--threshold", type=float, default=DUPLICATE_THRESHOLD)
    dd.add_argument("--json", action="store_true")
    cp = sub.add_parser("compare", help="match items of review batch A to their nearest in B")
    cp.add_argument("--batch", required=True)
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    cp.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "stage":
            index = stage(args.batch, args.reviews)
            print(f"staged {len(index['items'])} images from {len(args.reviews)} review batch(es) "
                  f"into {CSD_ROOT / 'stage' / args.batch}")
            return 0
        if args.command == "dedup":
            result = dedup(args.batch, args.against, threshold=args.threshold)
            print(json.dumps(result, ensure_ascii=False, indent=1)) if args.json else _print_dedup(result)
            return 1 if result["skip"] else 0
        if args.command == "report":
            result = report(args.batch, top=args.top)
            print(json.dumps(result, ensure_ascii=False, indent=1)) if args.json else _print_report(result)
            return 0
        if args.command == "compare":
            result = compare(args.batch, args.a, args.b)
            print(json.dumps(result, ensure_ascii=False, indent=1)) if args.json else _print_compare(result)
            return 0
    except CsdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
