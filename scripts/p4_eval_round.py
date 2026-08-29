#!/usr/bin/env python3
"""P4 quality-line evaluation round: before/after A-B over a fixed brief set.

For each brief the engine runs twice on the SAME seed:
  A (baseline): current best-tier defaults (face,eye detailers via recipe defaults)
  B (quality):  --detailers face,eye,hand --composition  (the P4 additions)

Output tree (SFW, analysis-only — outside the by_model gallery roots):
  outputs/_analysis/p4_eval/<stamp>/A/  B/  contact_A_B.png  manifest.json

Runs sequentially (GPU-bound; concurrent submits on one box abort each other).
Invoke via gen.py's engine passthrough so every production guardrail stays on:
  python3 scripts/p4_eval_round.py --box zzz --run
Without --run it prints the exact command plan and exits (same convention as gen.py).

A timed-out run (box unreachable, hung queue) is recorded as rc=-9 and the round
continues; the manifest is rewritten after every run. Resume the holes of an
interrupted round into the same stamp dir:
  python3 scripts/p4_eval_round.py --box zzz --into <stamp> --offset 9 --run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "gen.py"

# 12 briefs spanning the playbook axes: framing (close/half/full), setting (interior/
# exterior), action vs still, and hand-visibility (the detail-pass target) — seeds fixed
# so A/B pairs are strictly comparable.
BRIEFS: list[tuple[str, int]] = [
    ("library interior, warm lamplight, sitting at a desk, reading a book, hands holding pages", 91001),
    ("street cafe terrace, morning light, holding a coffee cup with both hands, steam", 91002),
    ("rainy bus stop at night, holding a transparent umbrella, neon reflections", 91003),
    ("kitchen, apron, kneading dough on a floured counter, focused expression", 91004),
    ("park in autumn, full body, walking, fallen leaves, hands in coat pockets", 91005),
    ("music room, playing an upright piano, fingers on keys, sheet music", 91006),
    ("balcony garden, watering potted plants with a metal watering can, dusk", 91007),
    ("close-up portrait, resting chin on interlaced fingers, soft window light", 91008),
    ("beach at golden hour, half body, holding a straw hat against the wind", 91009),
    ("workshop, soldering a circuit board, safety glasses, warm task lighting", 91010),
    ("festival evening, yukata, holding a goldfish bag, lantern bokeh", 91011),
    ("gym, stretching with a towel across shoulders, water bottle in hand", 91012),
]

QUALITY_ARGS = ["--detailers", "face,eye,hand", "--composition"]


def gen_cmd(prompt: str, seed: int, box: str, out_tag: str, extra: list[str]) -> list[str]:
    """One gen.py best invocation with engine passthrough after `--`."""
    argv = [sys.executable, str(GEN), "best",
            "--prompt", prompt, "--seed", str(seed), "--box", box, "--run",
            "--json-out", out_tag]
    if extra:
        argv += ["--", *extra]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 before/after eval round (A/B on fixed seeds).")
    ap.add_argument("--box", default="zzz", choices=("zzz", "z3", "auto"),
                    help="target box (default zzz — Z3 stays free for gaming)")
    ap.add_argument("--limit", type=int, default=len(BRIEFS),
                    help="run only N briefs (smoke: --limit 2)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N briefs (resume an interrupted round)")
    ap.add_argument("--into", metavar="STAMP",
                    help="write into an existing stamp dir and merge its manifest "
                         "(resume) instead of opening a new one")
    ap.add_argument("--run", action="store_true", help="execute; default prints the plan")
    args = ap.parse_args()

    stamp = args.into or time.strftime("%Y%m%d_%H%M%S")
    base = ROOT / "outputs" / "_analysis" / "p4_eval" / stamp
    if args.into and not base.is_dir():
        ap.error(f"--into {args.into}: no such stamp dir under {base.parent}")
    offset = max(0, args.offset)
    briefs = BRIEFS[offset: offset + max(1, args.limit)]
    if not briefs:
        ap.error(f"--offset {offset} is past the end of the {len(BRIEFS)}-brief set")

    plan: list[dict] = []
    for i, (prompt, seed) in enumerate(briefs, start=offset):
        for arm, extra in (("A", []), ("B", QUALITY_ARGS)):
            out = base / arm / f"{i:02d}_result.json"
            plan.append({"idx": i, "arm": arm, "seed": seed, "prompt": prompt,
                         "json_out": str(out),
                         "cmd": gen_cmd(prompt, seed, args.box, str(out), extra)})

    if not args.run:
        for p in plan:
            print(f"[{p['arm']}{p['idx']:02d}] {' '.join(p['cmd'])}")
        print(f"\n{len(plan)} runs planned ({len(briefs)} briefs x 2 arms). Add --run to execute.")
        return

    base.mkdir(parents=True, exist_ok=True)
    (base / "A").mkdir(exist_ok=True)
    (base / "B").mkdir(exist_ok=True)
    manifest_path = base / "manifest.json"
    # merge over an earlier partial round: keep entries this invocation doesn't re-run
    results: list[dict] = []
    if manifest_path.is_file():
        results = json.loads(manifest_path.read_text()).get("results", [])
        planned = {(p["idx"], p["arm"]) for p in plan}
        results = [r for r in results if (r["idx"], r["arm"]) not in planned]

    def write_manifest() -> None:
        ordered = sorted(results, key=lambda r: (r["idx"], r["arm"]))
        manifest = {"stamp": stamp, "box": args.box, "briefs": len({r["idx"] for r in ordered}),
                    "quality_args": QUALITY_ARGS, "results": ordered}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    for p in plan:
        print(f"\n=== [{p['arm']}{p['idx']:02d}] seed={p['seed']} ===", flush=True)
        t0 = time.monotonic()
        # a run that times out (box unreachable mid-round, hung queue) must not kill
        # the round: record it and move on — --offset/--into resumes the holes later
        try:
            proc = subprocess.run(p["cmd"], text=True, capture_output=True, timeout=1800)
            rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc = -9
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = f"timed out after {exc.timeout}s"
        wall = round(time.monotonic() - t0, 1)
        entry = {**{k: p[k] for k in ("idx", "arm", "seed", "prompt", "json_out")},
                 "rc": rc, "wall_s": wall,
                 # keep the engine log tail: it carries the [adapt]/[detail]/composition
                 # lines that prove which detailers actually ran on this box
                 "stdout_tail": stdout[-1500:]}
        try:
            entry["new_files"] = json.loads(Path(p["json_out"]).read_text())["new_files"]
        except Exception:
            entry["new_files"] = []
            entry["stderr_tail"] = stderr[-300:]
        results.append(entry)
        write_manifest()  # after every run, so an interrupted round still has its ledger
        print(f"    rc={rc} wall={wall}s files={len(entry['new_files'])}", flush=True)

    ran = {(p["idx"], p["arm"]) for p in plan}
    ok = sum(1 for r in results if (r["idx"], r["arm"]) in ran and r["rc"] == 0)
    print(f"\n[eval] {ok}/{len(plan)} runs ok this invocation — manifest at {manifest_path}")
    print("[eval] next: build the A/B contact sheet from the manifest (imgutils montage)")


if __name__ == "__main__":
    main()
