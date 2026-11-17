#!/usr/bin/env python3
"""Prompt tools for the Qwen-Image-2.1 line: lint a prompt, or rewrite a brief into one.

``lint``     runs the ``qwen21`` line of ``deai_lint`` plus the tier word band and prints rule ids.
``run``      renders an order's arms with qwen21_gen.py: ``--smoke`` first (one image), then the batch.
``spec``     lists a probe order's rendered PNGs as a review_batch spec (caption = probe/arm/seed).
``writeback`` joins a judged review batch back onto the order as ``verdicts.json`` keyed by sha256.
``rewrite``  asks the headless brain (``loramake.designer_brain.ask``) to turn a brief into a prompt
             following ``configs/qwen21/rewrite_instruction.json``; the result must pass the same lint
             or the command exits non-zero without printing a usable prompt. With no brain available it
             returns a clearly marked ``fallback`` draft (the brief plus the closing sentence, nothing
             invented) and still applies the lint.

Rules and their status live in docs/qwen21_line/PROMPTCRAFT.md; until Phase C is judged every rule is
a candidate, so outputs here are candidate prompts, not a frozen recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deai_lint  # noqa: E402
import review_result  # noqa: E402
from config import ROOT  # noqa: E402
from loramake import designer_brain  # noqa: E402
from loramake.prompt_compose import has_cjk  # noqa: E402

INSTRUCTION_PATH = ROOT / "configs" / "qwen21" / "rewrite_instruction.json"
SCHEMA_INSTRUCTION = "loramake.qwen21-rewrite/v1"
LINE = "qwen21"
# Word bands per tier (PROMPTCRAFT.md §3): compact / mid / official long.
# compact is 80 (not the community 60): the four mandatory sentences (style, camera, light, closer)
# already cost ~30 words, and the user-judged baseline is 45 words + closer. PROMPTCRAFT.md §3.
TIERS: dict[str, tuple[int, int]] = {"compact": (1, 80), "mid": (90, 180), "long": (400, 500)}
CLOSER = "No text, watermark or signature appears in the image."
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")

AskFn = Callable[[str], "str | None"]


class RewriteError(RuntimeError):
    """The brain answered but the result is unusable (bad JSON, bad ratio, lint errors).

    ``rejected`` carries the candidate that failed lint (when there is one) so the caller can show it."""

    def __init__(self, message: str, rejected: dict | None = None) -> None:
        super().__init__(message)
        self.rejected = rejected


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def load_instruction(path: Path | None = None) -> dict:
    payload = json.loads(Path(path or INSTRUCTION_PATH).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_INSTRUCTION:
        raise RewriteError(f"rewrite instruction schema must be {SCHEMA_INSTRUCTION}")
    missing = [tier for tier in TIERS if tier not in (payload.get("tiers") or {})]
    if missing:
        raise RewriteError(f"rewrite instruction lacks tiers: {missing}")
    return payload


def lint_prompt_text(prompt: str, tier: str, *, rules: dict | None = None) -> dict:
    """Lint one prompt on the qwen21 line and check the tier word band. Returns rule-id findings."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    rules = rules or deai_lint.load_rules(deai_lint.DEFAULT_RULES)
    findings = deai_lint.lint_prompt(LINE, prompt, "", "", rules)
    words = word_count(prompt)
    low, high = TIERS[tier]
    if words < low or words > high:
        findings.append({"rule": "word_band", "severity": "error",
                         "message": f"tier {tier} expects {low}-{high} words", "evidence": f"{words} words"})
    return {"ok": not deai_lint.has_errors(findings), "tier": tier, "words": words, "findings": findings}


def build_request(brief: str, tier: str, instruction: dict, ratio: str | None = None) -> str:
    parts = [str(instruction["instruction"]).strip(), str(instruction["tiers"][tier]).strip()]
    if ratio:
        parts.append(f"Use wh_ratio {ratio}.")
    parts.append(f"Brief: {brief.strip()}")
    return "\n\n".join(parts)


def fallback_prompt(brief: str) -> str:
    """The brief itself plus the closing sentence. Nothing is invented to satisfy a tier."""
    body = brief.strip().rstrip(".")
    return f"{body}. {CLOSER}" if body else CLOSER


def _parse_answer(text: str) -> dict:
    body = designer_brain._extract_json(text)
    if not body:
        raise RewriteError("brain answer is not a JSON object")
    prompt = str(body.get("prompt") or "").strip()
    ratio = str(body.get("wh_ratio") or "").strip()
    if not prompt:
        raise RewriteError("brain answer has no prompt")
    return {"prompt": prompt, "wh_ratio": ratio}


def rewrite(brief: str, tier: str, *, ratio: str | None = None, ask: AskFn | None = None,
            instruction: dict | None = None, rules: dict | None = None) -> dict:
    """Brief -> {prompt, wh_ratio, tier, fallback, lint}. Raises RewriteError when unusable."""
    if not brief.strip():
        raise RewriteError("brief must not be empty")
    if tier not in TIERS:
        raise RewriteError(f"tier must be one of {', '.join(TIERS)}")
    instruction = instruction or load_instruction()
    ratios = [str(item) for item in instruction.get("ratios") or []]
    if ratio and ratio not in ratios:
        raise RewriteError(f"ratio {ratio!r} not supported; use one of {', '.join(ratios)}")
    ask_fn = ask or designer_brain.ask
    answer = ask_fn(build_request(brief, tier, instruction, ratio))
    if answer is None:
        if has_cjk(brief):
            raise RewriteError("no brain available and the brief is not English; nothing safe to fall back to")
        result = {"prompt": fallback_prompt(brief), "wh_ratio": ratio or "2:3", "fallback": True}
    else:
        parsed = _parse_answer(answer)
        if parsed["wh_ratio"] not in ratios:
            raise RewriteError(f"brain chose wh_ratio {parsed['wh_ratio']!r}; supported: {', '.join(ratios)}")
        if ratio and parsed["wh_ratio"] != ratio:
            raise RewriteError(f"brain chose wh_ratio {parsed['wh_ratio']!r} but {ratio!r} was requested")
        result = {"prompt": parsed["prompt"], "wh_ratio": parsed["wh_ratio"], "fallback": False}
    lint = lint_prompt_text(result["prompt"], tier, rules=rules)
    result.update({"tier": tier, "lint": lint})
    if not lint["ok"]:
        errors = "; ".join(f"{f['rule']}: {f['evidence'] or f['message']}" for f in lint["findings"]
                           if f["severity"] == "error")
        raise RewriteError(f"rewrite failed lint ({'fallback' if result['fallback'] else 'brain'}): {errors}",
                           result)
    return result


def _print_findings(report: dict) -> None:
    print(f"tier={report['tier']} words={report['words']} ok={report['ok']}")
    for finding in report["findings"]:
        evidence = f" [{finding['evidence']}]" if finding.get("evidence") else ""
        print(f"  {finding['severity']:<7} {finding['rule']}{evidence}: {finding['message']}")


def _read_text_arg(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return value


def _cmd_lint(args: argparse.Namespace) -> int:
    report = lint_prompt_text(_read_text_arg(args.prompt), args.tier)
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        _print_findings(report)
    return 0 if report["ok"] else 1


def _cmd_rewrite(args: argparse.Namespace) -> int:
    try:
        result = rewrite(_read_text_arg(args.brief), args.tier, ratio=args.ratio)
    except RewriteError as exc:
        print(f"rewrite: {exc}", file=sys.stderr)
        if exc.rejected:
            print(f"rejected candidate ({exc.rejected['lint']['words']} words):\n{exc.rejected['prompt']}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        tag = " (FALLBACK: brain unavailable, brief passed through)" if result["fallback"] else ""
        print(f"wh_ratio={result['wh_ratio']} tier={result['tier']} words={result['lint']['words']}{tag}")
        print(result["prompt"])
        for finding in result["lint"]["findings"]:
            print(f"  {finding['severity']:<7} {finding['rule']}: {finding['message']}", file=sys.stderr)
    return 0


_SEED_RE = re.compile(r"_(\d+)(?:__v\d+)?\.png$")


def _seed_of(png: Path) -> int | None:
    meta = Path(str(png) + ".meta.json")
    if meta.is_file():
        try:
            seed = json.loads(meta.read_text(encoding="utf-8")).get("seed")
            if isinstance(seed, int):
                return seed
        except (OSError, json.JSONDecodeError):
            pass
    match = _SEED_RE.search(png.name)
    return int(match.group(1)) if match else None


def order_probes(order: Path) -> list[str]:
    """Probe names of an order: from ``arms.json`` when present, else every subdirectory except ``smoke``."""
    arms_path = Path(order) / "arms.json"
    if arms_path.is_file():
        arms = json.loads(arms_path.read_text(encoding="utf-8")).get("arms") or []
        return sorted({str(arm["probe"]) for arm in arms})
    return sorted(p.name for p in Path(order).iterdir() if p.is_dir() and p.name != "smoke")


def order_spec(order: Path, probe: str | None = None) -> dict:
    """Review spec for an order's rendered arms: ``<order>/<probe>/<arm>/*.png`` -> caption probe/arm/seed."""
    order = Path(order)
    if not order.is_dir():
        raise ValueError(f"order directory missing: {order}")
    items: list[dict] = []
    for probe_dir in (order / name for name in order_probes(order)):
        if not probe_dir.is_dir() or (probe and probe_dir.name != probe):
            continue
        for arm_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
            for png in sorted(arm_dir.glob("*.png")):
                seed = _seed_of(png)
                items.append({"path": png.as_posix(),
                              "caption": f"{probe_dir.name}/{arm_dir.name}/{seed if seed is not None else png.stem}"})
    if not items:
        raise ValueError(f"no rendered arms under {order}" + (f" for {probe}" if probe else ""))
    return {"items": items}


def writeback_verdicts(batch: str, order: Path, pending_as: str | None = None) -> dict:
    """Join one judged review batch onto ``<order>/verdicts.json`` (keyed by sha256). Returns the summary.

    ``pending_as="yes"`` records unmarked items as approved (the user's review convention: an image
    left unmarked is accepted; only ★ / ✗ are explicit)."""
    order = Path(order)
    manifest = json.loads((review_result._batch_dir(batch) / "manifest.json").read_text(encoding="utf-8"))
    report = review_result.read_result(batch)
    tier_of: dict[int, str] = {}
    for tier in ("love", "yes", "no"):
        for item in report[tier]:
            tier_of[int(item["i"])] = tier
    reasons = {int(item["i"]): item.get("reasons", []) for item in report["no"]}
    comments = {int(key): value["comment"] for key, value in (report.get("comments") or {}).items()}
    path = order / "verdicts.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema": "loramake.qwen21-verdicts/v1", "items": {}}
    summary: dict[str, dict[str, int]] = {}
    for raw in manifest.get("items") or []:
        index = int(raw["i"])
        caption = str(raw.get("caption") or "")
        parts = caption.split("/")
        probe, arm, seed = (parts + ["", "", ""])[:3]
        verdict = tier_of.get(index, pending_as or "pending")
        existing["items"][raw["sha256"]] = {
            "batch": batch, "index": index, "caption": caption, "probe": probe, "arm": arm, "seed": seed,
            "source_path": raw.get("source_path"), "verdict": verdict,
            "reasons": reasons.get(index, []), "comment": comments.get(index),
        }
        key = f"{probe}/{arm}"
        summary.setdefault(key, {"love": 0, "yes": 0, "no": 0, "pending": 0})[verdict] += 1
    existing["updated_from"] = batch
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "summary": summary, "total": report["total"],
            "pending": 0 if pending_as else len(report["pending"])}


def import_onbox(order: Path, results: Path, tree: Path) -> list[Path]:
    """Move on-box renders (qwen21_probe_onbox.py) into the engine layout with sidecars.

    ``tree/<probe>/<arm>/s<seed>_00001_.png`` -> ``order/<probe>/<arm>/qwen21_<probe>_<arm>_16st_<w>x<h>_00_<seed>.png``
    plus ``.meta.json`` carrying the arm prompt, seed, canvas, elapsed and the sidecar argv. Existing
    targets are never overwritten (``__vN``)."""
    from krea2_gen import no_clobber_path  # sibling helper; same no-clobber rule as the engine
    order, tree = Path(order), Path(tree)
    arms = {(a["probe"], a["arm"], int(a["seed"])): a
            for a in json.loads((order / "arms.json").read_text(encoding="utf-8"))["arms"]}
    payload = json.loads(Path(results).read_text(encoding="utf-8"))
    moved: list[Path] = []
    for result in payload.get("results") or []:
        key = (result["probe"], result["arm"], int(result["seed"]))
        arm = arms.get(key)
        if arm is None:
            raise ValueError(f"result {key} has no arm in arms.json")
        for image in result.get("images") or []:
            src = tree / image.replace("\\", "/").split("/", 1)[-1] if image.startswith("qwen21probe") else tree / image
            src = Path(str(src).replace("\\", "/"))
            if not src.is_file():
                raise FileNotFoundError(f"rendered file missing: {src}")
            dest_dir = order / result["probe"] / result["arm"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            stem = f"qwen21_{result['probe']}_{result['arm']}_16st_{result['width']}x{result['height']}_00_{result['seed']}.png"
            dest, _rev = no_clobber_path(dest_dir / stem)
            src.replace(dest)
            meta = {
                "schema_version": "loramake.qwen21_gen.v1", "engine": "qwen21_probe_onbox",
                "gen_engine": "comfyui:boxa-sidecar/qwen21", "line": "qwen21_gen", "box": "boxa",
                "host": payload.get("endpoint"), "runtime": {"comfyui_version": payload.get("comfyui_version"), "argv": payload.get("argv")},
                "model": "qwen_image_2.1_int8_convrot.safetensors", "clip": "qwen3vl_8b_w4a8.safetensors", "clip_type": "qwen_image",
                "vae": "qwen_image_2.1_vae_bf16.safetensors", "seed": result["seed"], "steps": 16, "cfg": 1.0,
                "sampler": "euler", "scheduler": "simple", "prompt": arm["prompt"], "negative": "",
                "resolution": f"{result['width']}x{result['height']}", "ratio": arm["ratio"], "size": arm["size"],
                "elapsed_s": result.get("elapsed_s"), "prompt_id": result.get("prompt_id"),
                "probe": result["probe"], "arm": result["arm"], "kept_from": image,
            }
            Path(str(dest) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            moved.append(dest)
    return moved


GEN_SCRIPT = Path(__file__).resolve().with_name("qwen21_gen.py")


def _arm_argv(arm: dict, into: Path, json_out: Path, label: str, python: str,
              no_wait: bool = False, no_free: bool = False) -> list[str]:
    argv = [python, str(GEN_SCRIPT), "--run", "--prompt", arm["prompt"], "--ratio", arm["ratio"],
            "--size", arm["size"], "--seed", str(arm["seed"]), "--label", label,
            "--keep-into", str(into), "--json-out", str(json_out)]
    if no_wait:
        # Skips the production :8188 idle probe only; the sidecar queue check stays inside qwen21_gen.
        # Use when :8188 is down and the GPU is verifiably idle from the sidecar's own system_stats.
        argv.append("--no-wait")
    if no_free:
        argv.append("--no-free")  # batch arms keep the models loaded; only the last arm frees VRAM
    return argv


def _result_ok(path: Path) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return result.get("rc") == 0 and bool(result.get("new_files"))


def run_order(order: Path, *, smoke: bool, python: str = sys.executable, no_wait: bool = False,
              call=None, log=print) -> int:
    """Render an order's arms with qwen21_gen.py, one isolated ``<order>/<probe>/<arm>`` per arm.

    ``smoke`` renders arm 0 into ``<order>/smoke`` and stops. The full run refuses to start until that
    smoke result says ``rc == 0`` with a file (batch-render discipline: one image first), prints the
    measured seconds times the arm count, skips arms whose result already says ok, and aborts on the
    first non-zero rc."""
    call = call or subprocess.call
    order = Path(order)
    arms = json.loads((order / "arms.json").read_text(encoding="utf-8"))["arms"]
    if not arms:
        raise ValueError("arms.json has no arms")
    smoke_json = order / "smoke" / "result.json"
    if smoke:
        rc = call(_arm_argv(arms[0], order / "smoke", smoke_json, "smoke", python, no_wait))
        log(f"smoke rc={rc} -> {smoke_json}")
        return rc
    if not _result_ok(smoke_json):
        raise ValueError(f"no passing smoke at {smoke_json}: run `run --order {order} --smoke` first")
    smoke_runs = json.loads(smoke_json.read_text(encoding="utf-8")).get("runs") or []
    per_image = float(smoke_runs[0].get("elapsed_s") or 0) if smoke_runs else 0.0
    log(f"smoke {per_image:.1f}s/image x {len(arms)} arms ~ {per_image * len(arms) / 60:.1f} min")
    for position, arm in enumerate(arms):
        into = order / arm["probe"] / arm["arm"]
        json_out = into / f"result_{arm['seed']}.json"
        if _result_ok(json_out):
            log(f"== {arm['probe']}/{arm['arm']} seed {arm['seed']}: already rendered, skip")
            continue
        log(f"== {arm['probe']}/{arm['arm']} seed {arm['seed']}")
        rc = call(_arm_argv(arm, into, json_out, f"{arm['probe']}_{arm['arm']}", python, no_wait,
                            no_free=position < len(arms) - 1))
        if rc != 0:
            raise ValueError(f"ABORT: {arm['probe']}/{arm['arm']} seed {arm['seed']} rc={rc}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        return run_order(Path(args.order), smoke=args.smoke, python=args.python, no_wait=args.no_wait)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1


def _cmd_import_onbox(args: argparse.Namespace) -> int:
    try:
        moved = import_onbox(Path(args.order), Path(args.results), Path(args.tree))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"import-onbox: {exc}", file=sys.stderr)
        return 1
    print(f"imported {len(moved)} image(s) into {args.order}")
    return 0


def _cmd_spec(args: argparse.Namespace) -> int:
    try:
        print(json.dumps(order_spec(Path(args.order), args.probe), ensure_ascii=False, indent=2))
    except ValueError as exc:
        print(f"spec: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_writeback(args: argparse.Namespace) -> int:
    try:
        result = writeback_verdicts(args.batch, Path(args.order), pending_as="yes" if args.pending_as_yes else None)
    except (review_result.ReviewResultError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"writeback: {exc}", file=sys.stderr)
        return 1
    print(f"verdicts -> {result['path']} ({result['total'] - result['pending']}/{result['total']} judged)")
    for key, counts in sorted(result["summary"].items()):
        print(f"  {key:<28} love={counts['love']} yes={counts['yes']} no={counts['no']} pending={counts['pending']}")
    return 0 if result["pending"] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 prompt lint / rewrite")
    sub = parser.add_subparsers(dest="command", required=True)
    lint = sub.add_parser("lint", help="lint a prompt on the qwen21 line (rule ids + word band)")
    lint.add_argument("prompt", help="prompt text, or - for stdin")
    lint.add_argument("--tier", choices=tuple(TIERS), default="compact")
    lint.add_argument("--json", action="store_true")
    lint.set_defaults(func=_cmd_lint)
    rw = sub.add_parser("rewrite", help="turn a brief into a linted prompt via the headless brain")
    rw.add_argument("brief", help="one-line brief (Chinese or English), or - for stdin")
    rw.add_argument("--tier", choices=tuple(TIERS), default="compact")
    rw.add_argument("--ratio", choices=("1:1", "2:3", "3:2"), default=None)
    rw.add_argument("--json", action="store_true")
    rw.set_defaults(func=_cmd_rewrite)
    sp = sub.add_parser("spec", help="review_batch spec for an order's rendered probe arms")
    sp.add_argument("--order", required=True)
    sp.add_argument("--probe", default=None, help="only this probe (e.g. P1)")
    sp.set_defaults(func=_cmd_spec)
    run = sub.add_parser("run", help="render an order's arms via qwen21_gen.py (smoke first, then the batch)")
    run.add_argument("--order", required=True)
    run.add_argument("--smoke", action="store_true", help="render arm 0 into <order>/smoke and stop")
    run.add_argument("--python", default=sys.executable, help="interpreter for qwen21_gen.py")
    run.add_argument("--no-wait", action="store_true",
                     help="pass --no-wait to qwen21_gen.py (production :8188 down; GPU idle confirmed elsewhere)")
    run.set_defaults(func=_cmd_run)

    io = sub.add_parser("import-onbox", help="move qwen21_probe_onbox.py renders into the order with sidecars")
    io.add_argument("--order", required=True)
    io.add_argument("--results", required=True, help="probe_results.json written on the box")
    io.add_argument("--tree", required=True, help="local copy of <ComfyUI output>/qwen21probe")
    io.set_defaults(func=_cmd_import_onbox)
    wb = sub.add_parser("writeback", help="join a judged review batch onto <order>/verdicts.json")
    wb.add_argument("--batch", required=True)
    wb.add_argument("--order", required=True)
    wb.add_argument("--pending-as-yes", action="store_true",
                    help="record unmarked items as approved (review convention: unmarked = accepted)")
    wb.set_defaults(func=_cmd_writeback)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
