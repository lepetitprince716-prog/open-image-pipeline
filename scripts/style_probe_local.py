#!/usr/bin/env python3
"""Bare style probes for the registry's local LoRA realizations (Anima on BOXA, Illustrious on BOXB).

One neutral bust per realization and seed, same content everywhere, so the renders compare
styles and nothing else. ``plan`` writes gen_job.py briefs, an index and commands.sh and
renders nothing; ``spec`` chunks the finished renders into review_batch.py specs of at most
``--chunk`` images (the user judges in batches of about fifteen).

Usage:
    python3 scripts/style_probe_local.py plan --out DIR [--line anima|illustrious|all] [--seeds 7,8]
                                              [--limit N] [--content TEXT]
    python3 scripts/style_probe_local.py spec --out DIR [--line anima|illustrious] [--chunk 14] [--register]
    python3 scripts/style_probe_local.py writeback --out DIR [--apply]

Exit codes: 0 ok · 1 refused input · 2 usage.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from loramake import style_registry as sr  # noqa: E402

SCHEMA = "loramake.style-probe-local/v1"
LINES = ("anima", "illustrious")
# Neutral bust: the same content for every style so only the style differs.
DEFAULT_CONTENT = "1girl, solo, upper body, looking at viewer, light smile, simple background, white background"
LINE_BRIEF = {
    "anima": {"tier": "draft", "box": "boxa", "res": "square"},
    "illustrious": {"tier": "best", "box": "boxb", "res": "1024x1024"},
}
REVIEW_BATCH = HERE / "review_batch.py"
_SLUG = re.compile(r"[^a-z0-9]+")


def lora_realizations(cards: dict[str, dict], lines: tuple[str, ...]) -> list[tuple[str, str, dict]]:
    """(style_id, line, realization) for every non-blocked lora realization on the wanted lines."""
    out = []
    for style_id, card in sorted(cards.items()):
        for line in lines:
            for real in (card.get("realizations") or {}).get(line, []) or []:
                if real.get("kind") == "lora" and real.get("tier") != "blocked":
                    out.append((style_id, line, real))
    return out


def probe_brief(frag: sr.RenderFragment, seed: int, content: str) -> dict:
    settings = LINE_BRIEF[frag.line]
    prompt = ", ".join(p for p in (frag.pos_prefix.strip(), content) if p)
    brief = {"tier": settings["tier"], "box": settings["box"], "res": settings["res"], "n": 1,
             "seed": seed, "sfw": True, "prompt": prompt}
    if frag.lora_specs:
        brief["lora"] = frag.lora_specs[0]
        if len(frag.lora_specs) > 1:
            brief["loras"] = frag.lora_specs[1:]
    if frag.neg_add:
        brief["neg_add"] = frag.neg_add
    if frag.base_pin:
        brief["base"] = frag.base_pin
    return brief


def slug(text: str) -> str:
    return _SLUG.sub("_", text.lower()).strip("_")


def plan(out_dir: Path, lines: tuple[str, ...], seeds: list[int], content: str,
         limit: int | None = None, cards: dict[str, dict] | None = None,
         loras: dict[str, dict] | None = None) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise sr.StyleRegistryError(f"{out_dir} is not empty; probe rounds never mix, pick a new --out")
    cards = cards if cards is not None else sr.load_cards()
    loras = loras if loras is not None else sr.lora_index()
    reals = lora_realizations(cards, lines)
    if limit:
        reals = reals[:limit]
    items, commands, files = [], [], []
    for style_id, line, real in reals:
        frag = sr.resolve(style_id, line, prefer=real["id"], cards=cards, loras=loras)
        stem = f"{line}__{slug(style_id)}__{slug(real['id'].split(':')[-1])}"
        for seed in seeds:
            brief = probe_brief(frag, seed, content)
            brief_path = out_dir / "briefs" / f"{stem}__s{seed}.brief.json"
            result = out_dir / "results" / f"{stem}__s{seed}.json"
            files.append((brief_path, json.dumps(brief, ensure_ascii=False, indent=2) + "\n"))
            render_dir = out_dir / "renders" / f"{stem}__s{seed}"
            commands.append(" ".join(shlex.quote(a) for a in (
                "python3", "scripts/gen_job.py", str(brief_path), "--json-out", str(result),
                "--keep-into", str(render_dir))))
            items.append({"style_id": style_id, "line": line, "realization_id": real["id"],
                          "display_name": cards[style_id].get("display_name"), "seed": seed,
                          "brief": str(brief_path), "result": str(result), "lora": brief.get("lora")})
    if not items:
        raise sr.StyleRegistryError("no lora realizations on the requested lines")
    index = {"schema": SCHEMA, "lines": list(lines), "seeds": seeds, "content": content, "items": items}
    files.append((out_dir / "probe_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"))
    by_line: dict[str, list[str]] = {}
    for item, command in zip(items, commands, strict=False):
        by_line.setdefault(item["line"], []).append(command)
    for line, cmds in by_line.items():
        files.append((out_dir / f"commands_{line}.sh",
                      "#!/usr/bin/env bash\nset -uo pipefail\n" + "\n".join(cmds) + "\n"))
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = {line: len(cmds) for line, cmds in by_line.items()}
    return index


# Aesthetic-register probes (2026-09-16): one render per register x seed through the real
# artistic route (`gen.py artistic --style`), so prompt-kind realizations and base pins are
# exercised exactly as an order would render them. Every art slot is pinned so only the
# aesthetic key varies; captions keep the [style_id/realization_id] form writeback reads.
AESTHETIC_SLOTS = ["--medium", "none", "--camera", "upper body", "--composition", "centered composition",
                   "--lighting", "soft light", "--pose", "standing", "--color-mood", "none"]


def plan_aesthetic(out_dir: Path, line: str, box: str, seeds: list[int], content: str,
                   density: str = "medium", registers: dict | None = None,
                   cards: dict[str, dict] | None = None, intent: str = "variants",
                   only: set[str] | None = None) -> dict:
    """``intent`` picks the engine: ``variants`` = single base pass (boxa_gen.py, seconds per
    image, no upscale / detailers — the default for probes, style is visible at base res);
    ``artistic`` = the full Illustrious quality stack with every art slot pinned."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise sr.StyleRegistryError(f"{out_dir} is not empty; probe rounds never mix, pick a new --out")
    registers = registers if registers is not None else sr.load_registers()
    cards = cards if cards is not None else sr.load_cards()
    items, commands = [], []
    skipped = []
    for reg_id in registers.get("registers") or {}:
        if only and reg_id not in only:
            continue
        try:
            frag = sr.resolve_aesthetic(reg_id, line, density=density, registers=registers, cards=cards, box=box)
        except sr.StyleUnavailable as exc:
            skipped.append(f"{reg_id}: {exc}")
            continue
        stem = f"{line}__{slug(reg_id)}__{slug(frag.style_id)}"
        for seed in seeds:
            result = out_dir / "results" / f"{stem}__s{seed}.json"
            slots = ["--sfw", *AESTHETIC_SLOTS] if intent == "artistic" else []
            argv = ["python3", "scripts/gen.py", intent, "--prompt", content, "--style", reg_id,
                    "--density", density, "--box", box, "--seed", str(seed), "--n", "1",
                    *slots, "--run", "--json-out", str(result)]
            # Engines name files by LoRA + seed, so prompt-kind registers sharing a seed would
            # overwrite each other; every render is copied into its own folder right away.
            keep = ["python3", "scripts/style_probe_local.py", "keep", "--result", str(result),
                    "--into", str(out_dir / "renders" / f"{stem}__s{seed}")]
            commands.append(" ".join(shlex.quote(a) for a in argv) + " && "
                            + " ".join(shlex.quote(a) for a in keep))
            items.append({"register": reg_id, "style_id": frag.style_id, "line": line,
                          "realization_id": frag.realization_id, "tier": frag.tier,
                          "display_name": cards[frag.style_id].get("display_name"), "seed": seed,
                          "result": str(result), "lora": frag.lora_specs[0] if frag.lora_specs else None,
                          "base_pin": frag.base_pin})
    if not items:
        raise sr.StyleRegistryError(f"no register resolves on {line}")
    index = {"schema": SCHEMA, "mode": "aesthetic", "intent": intent, "lines": [line], "box": box, "seeds": seeds,
             "content": content, "density": density, "items": items, "skipped": skipped}
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    (out_dir / f"commands_{line}.sh").write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
    index["commands"] = {line: len(commands)}
    return index


def rendered_files(item: dict) -> list[str]:
    path = Path(item["result"])
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    files = data.get("new_files") or data.get("files") or []
    # The engines also write a post-fx twin (*_fx.png); the judged image is the plain render.
    # Kept copies are recorded relative to this checkout; absolutize so another checkout's
    # review_batch.py (the one serving :8771) can snapshot them.
    return [str((ROOT / f).resolve() if not Path(f).is_absolute() else Path(f))
            for f in files if isinstance(f, str) and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            and not Path(f).stem.endswith("_fx")]


CAPTION_LIMIT = 200  # review_batch.py refuses longer captions


def caption_for(item: dict, path: str) -> str:
    text = (f"[{item['style_id']}/{item['realization_id']}] {item.get('display_name') or ''} · "
            f"{item['line']} · seed {item['seed']} · {Path(path).name}")
    if len(text) > CAPTION_LIMIT:
        text = text[:CAPTION_LIMIT - 1] + "…"
    return text


def specs(out_dir: Path, chunk: int, line: str | None = None) -> list[Path]:
    index = json.loads((out_dir / "probe_index.json").read_text(encoding="utf-8"))
    entries = [{"path": p, "caption": caption_for(item, p)}
               for item in index["items"] if not line or item["line"] == line
               for p in rendered_files(item)]
    if not entries:
        raise sr.StyleRegistryError("nothing rendered yet; run the commands_*.sh first")
    paths = []
    tag = f"{line}_" if line else ""
    for n, start in enumerate(range(0, len(entries), chunk), start=1):
        path = out_dir / f"spec_{tag}{n:02d}.json"
        path.write_text(json.dumps(entries[start:start + chunk], ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        paths.append(path)
    return paths


REVIEW_ROOT = ROOT / "outputs" / "_gallery" / "review_batches"


def probe_batches(out_dir: Path, review_root: Path = REVIEW_ROOT, stem: str | None = None) -> list[str]:
    """Review batches this round registered: <round>_<line>_NN (and the plain <round>_NN form)."""
    stem = stem or out_dir.name
    if not review_root.is_dir():
        return []
    names = sorted(p.name for p in review_root.iterdir() if p.is_dir() and p.name.startswith(stem + "_"))
    return [n for n in names if (n[len(stem) + 1:].split("_")[-1]).isdigit()]


def writeback(out_dir: Path, *, review_root: Path = REVIEW_ROOT, cards_dir: Path | None = None,
              date: str | None = None, apply: bool = False, batch_stem: str | None = None,
              unjudged_as: str | None = None) -> list[str]:
    """Record every judged probe on its registry realization (civitai_probe conventions).

    Verdicts accumulate in the realization note as ``verdicts love=a yes=b no=c (n=k)``; the
    tier follows catalogs/style/README.md through civitai_probe.tier_for. Unjudged items are
    skipped, so the command can run again after every judged batch.
    """
    import datetime

    import civitai_probe  # sibling script: tier_for / VERDICT_RE

    cards_dir = cards_dir or sr.CARDS_DIR
    date = date or datetime.date.today().isoformat()
    counts: dict[tuple[str, str], dict[str, int]] = {}
    batches_of: dict[tuple[str, str], set[str]] = {}
    for batch in probe_batches(out_dir, review_root, batch_stem):
        batch_dir = review_root / batch
        try:
            result = json.loads((batch_dir / "result.json").read_text(encoding="utf-8"))
            manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Per item, not per caption key: one realization has one item per seed, and
        # civitai_probe.item_verdicts keeps a single label per key.
        label_by_index = {i: label for key, label in (("l", "love"), ("y", "yes"), ("n", "no"))
                          for i in result.get(key, [])}
        for entry in manifest.get("items", []):
            caption = entry.get("caption", "")
            label = label_by_index.get(entry.get("i"))
            if label is None and unjudged_as:
                label = unjudged_as  # user convention: an opened batch's unmarked image counts as this
            if not (caption.startswith("[") and "]" in caption) or label is None:
                continue
            style_id, _, realization_id = caption[1:caption.index("]")].partition("/")
            if not realization_id:
                continue
            c = counts.setdefault((style_id, realization_id), {"love": 0, "yes": 0, "no": 0})
            c[label] += 1
            batches_of.setdefault((style_id, realization_id), set()).add(batch)
    changes: list[str] = []
    touched: dict[str, dict] = {}
    for (style_id, realization_id), new_counts in sorted(counts.items()):
        card = touched.get(style_id)
        if card is None:
            path = cards_dir / f"{style_id}.json"
            if not path.is_file():
                changes.append(f"skip {style_id}/{realization_id}: no card")
                continue
            card = touched[style_id] = json.loads(path.read_text(encoding="utf-8"))
        line = realization_id.split(":")[0]
        real = next((r for r in (card.get("realizations") or {}).get(line, []) if r.get("id") == realization_id), None)
        if real is None:
            changes.append(f"skip {style_id}/{realization_id}: realization missing on the card")
            continue
        note = real.get("note", "")
        match = civitai_probe.VERDICT_RE.search(note)
        total = ({"love": int(match.group(1)), "yes": int(match.group(2)), "no": int(match.group(3))}
                 if match else {"love": 0, "yes": 0, "no": 0})
        for k, v in new_counts.items():
            total[k] += v
        tag = f"verdicts love={total['love']} yes={total['yes']} no={total['no']} (n={sum(total.values())})"
        real["note"] = civitai_probe.VERDICT_RE.sub(tag, note) if match else (f"{note.rstrip('; ')}; {tag}" if note else tag)
        real["evidence"] = {"date": date, "ref": ", ".join(sorted(batches_of[(style_id, realization_id)])),
                            "note": f"local bare probe {out_dir.name}: " + ", ".join(f"{k} {v}" for k, v in new_counts.items() if v)}
        old_tier = real.get("tier", "candidate")
        real["tier"] = civitai_probe.tier_for(total, old_tier)
        changes.append(f"update {style_id} {realization_id}: {new_counts} -> {tag}; tier {old_tier} -> {real['tier']}")
    if apply:
        for style_id, card in touched.items():
            card["revision"] = int(card.get("revision", 1)) + 1
            (cards_dir / f"{style_id}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8")
    return changes


def cmd_keep(args: argparse.Namespace) -> int:
    import gen_job  # sibling: keep_files copies a result's new_files and rewrites the JSON

    kept = gen_job.keep_files(Path(args.result), Path(args.into))
    import shutil
    for path in kept:  # the engine's .meta.json sidecar travels with the render
        meta = Path(path).with_name(Path(path).name + ".meta.json")
        source = None
        for original in json.loads(Path(args.result).read_text(encoding="utf-8")).get("kept_from") or []:
            if Path(original).name == Path(path).name:
                source = Path(original).with_name(Path(original).name + ".meta.json")
        if source and source.is_file() and not meta.exists():
            shutil.copy2(source, meta)
    print(f"[keep] {len(kept)} file(s) -> {args.into}")
    return 0 if kept else 1


def cmd_writeback(args: argparse.Namespace) -> int:
    review_root = Path(args.review_root) if args.review_root else REVIEW_ROOT
    changes = writeback(Path(args.out), apply=args.apply, review_root=review_root,
                        batch_stem=args.batch_stem, unjudged_as=args.unjudged_as)
    for line in changes:
        print(line)
    if not changes:
        print("nothing judged yet")
    elif not args.apply:
        print("dry run; pass --apply to write the cards")
    else:
        problems = sr.lint(sr.load_cards())
        print(f"lint: {len(problems)} problem(s)")
        return 1 if problems else 0
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.by == "aesthetic":
        if args.line == "all":
            raise sr.StyleRegistryError("--by aesthetic needs one --line")
        only = {r.strip() for r in args.only.split(",") if r.strip()} if args.only else None
        index = plan_aesthetic(Path(args.out), args.line, args.box, seeds, args.content, args.density,
                               intent=args.intent, only=only)
        for line in index["skipped"]:
            print(f"  skipped {line}")
    else:
        lines = LINES if args.line == "all" else (args.line,)
        index = plan(Path(args.out), lines, seeds, args.content, args.limit)
    print(f"{len(index['items'])} probe renders planned: " +
          ", ".join(f"{line} {n}" for line, n in index["commands"].items()))
    for line in index["commands"]:
        print(f"  bash {Path(args.out) / f'commands_{line}.sh'}")
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    paths = specs(out_dir, args.chunk, args.line)
    batch_stem = (args.batch_stem or out_dir.name) + (f"_{args.line}" if args.line else "")
    review_batch = Path(args.repo) / "scripts" / "review_batch.py" if args.repo else REVIEW_BATCH
    for n, path in enumerate(paths, start=1):
        batch = f"{batch_stem}_{n:02d}"
        argv = [sys.executable, str(review_batch), "--mode", "review", "--batch", batch,
                "--spec", str(path.resolve()), "--json"]
        if args.register:
            proc = subprocess.run(argv, capture_output=True, text=True, cwd=args.repo or None)
            sys.stdout.write(proc.stdout or proc.stderr)
        else:
            print(" ".join(shlex.quote(a) for a in argv))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="bare style probes for local LoRA realizations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write briefs, index and commands_<line>.sh (renders nothing)")
    p.add_argument("--out", required=True)
    p.add_argument("--line", choices=(*LINES, "all"), default="all")
    p.add_argument("--seeds", default="7,8")
    p.add_argument("--limit", type=int, help="only the first N realizations (dry sizing)")
    p.add_argument("--content", default=DEFAULT_CONTENT)
    p.add_argument("--by", choices=("lora", "aesthetic"), default="lora",
                   help="aesthetic: one render per register x seed via gen.py artistic --style")
    p.add_argument("--box", choices=("boxa", "boxb"), default="boxa", help="--by aesthetic: render box")
    p.add_argument("--density", default="medium", help="--by aesthetic: light|medium|strong or 淡|中|浓")
    p.add_argument("--intent", choices=("variants", "artistic"), default="variants",
                   help="--by aesthetic: variants = base pass only (default, fast); artistic = full quality stack")
    p.add_argument("--only", default=None, help="--by aesthetic: comma list of register ids to plan")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("keep", help="copy one result's renders into their own folder (called by commands.sh)")
    p.add_argument("--result", required=True)
    p.add_argument("--into", required=True)
    p.set_defaults(func=cmd_keep)
    p = sub.add_parser("writeback", help="record judged probes on the registry realizations")
    p.add_argument("--out", required=True)
    p.add_argument("--apply", action="store_true", help="write the cards (dry run otherwise)")
    p.add_argument("--review-root", default=None, help="review_batches root the batches were registered in")
    p.add_argument("--batch-stem", default=None, help="batch name stem used at `spec` time")
    p.add_argument("--unjudged-as", choices=("yes",), default=None,
                   help="count an opened batch's unmarked images as this verdict (user convention: unmarked = accepted)")
    p.set_defaults(func=cmd_writeback)
    p = sub.add_parser("spec", help="chunked review_batch specs for the finished renders")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=14)
    p.add_argument("--line", choices=LINES, help="only this line's renders (batches get the line in their id)")
    p.add_argument("--register", action="store_true", help="also open the review batches")
    p.add_argument("--batch-stem", default=None, help="batch name stem (default: the --out dir name)")
    p.add_argument("--repo", default=None,
                   help="register through this checkout's review_batch.py (e.g. the main checkout that serves :8771)")
    p.set_defaults(func=cmd_spec)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (sr.StyleRegistryError, sr.StyleUnavailable) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
