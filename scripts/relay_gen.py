#!/usr/bin/env python3
"""Two-box relay line: stream fast drafts on the ZZZ worker while Z3 refines winners.

Latency-first design. The main thread renders drafts on ZZZ one image at a
time; a single worker thread gates each draft the moment it lands (SFW
fail-closed, optional CCIP identity, defect flags) and immediately refines
passing drafts on Z3 through the quality engine's refine path. Both boxes run
simultaneously, each strictly serial inside itself (concurrent submits on one
box abort each other); models stay resident between per-image runs and each
lane releases VRAM once at the end.

Stage-1 recipe: with --lora the draft leg is gen.py variants on zzz in fast
mode (16 steps, iLustMix); without a LoRA it is the Krea-2 Turbo line on zzz
(8 steps), falling back to the quality engine with the detailer chain off
when Krea-2 assets are missing on that box.

--character "name or description" picks the best-matching character LoRA
from the local catalog index by weighted lexical match over names, trigger
words, and tags (deterministic and fully offline — the catalog's embedding
path needs model weights this host cannot fetch). Eligible records must
exist on the ZZZ box, carry a base family the draft checkpoint accepts
(locally trained files without metadata are screened by file-path tokens
instead), and not be rated nsfw — unrated local trains stay eligible. The
winner joins the --lora stack, so the fast variants leg applies. No
eligible match above --min-match fails fast with the closest candidates
listed instead of silently drafting without a LoRA.

Gates are OFF by default — the user reviews the output; the first --top
drafts to land are refined as-is (lowest latency). --gate turns the
unattended posture back on (SFW fail-closed + qc defect flags); --ref adds
the CCIP identity check on its own. --rank waits for the whole batch, ranks
by aesthetic percentile (unflagged first), then refines the top --top — it
implies --gate because ranking needs the scores. Without --run the exact
per-lane commands are printed and nothing is spawned.

The relay never re-implements engine guardrails: every render is a gen.py
(or krea2_gen.py) subprocess with --run, and draft paths come only from each
run's machine-readable result, never from scanning the output tree.
"""
from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config import ROOT, resolve_comfy_host  # noqa: E402  (sibling shim to loramake.config)
import qc_score  # noqa: E402

GEN = SCRIPTS / "gen.py"
KREA2 = SCRIPTS / "krea2_gen.py"
SFW_GATE = SCRIPTS / "sfw_gate.py"
CONSISTENCY = SCRIPTS / "consistency_check.py"
RELAY_BASE = ROOT / "outputs" / "_analysis" / "relay"
SCHEMA = "loramake.relay-result/v1"
GEN_RESULT_SCHEMA = "loramake.gen-result/v2"
# Anchor on the per-image COMPLETION line ("... 14.2s -> path"); krea2 also prints a
# pre-submit plan line ("[00 seed=1] -> path") that must not be counted.
KREA2_SAVED_RE = re.compile(r"\]\s+\d+(?:\.\d+)?s\s+->\s+(\S+\.png)\s*$", re.MULTILINE)
GATE_TIMEOUT_S = 120
_SENTINEL = None  # queue end marker

DEFAULT_N = 4
DEFAULT_TOP = 2
DEFAULT_Z3_WAIT = 60.0
DEFAULT_STAGE_TIMEOUT = 900.0

CATALOG_INDEX = ROOT / "lora_catalog" / "z3_index.json"
# Base families the draft checkpoint (iLustMix, SDXL-family) can load. Anything
# else in the catalog (SD 1.5, Anima, video/flux families) would fail the engine's
# arch guard or render garbage, so it never qualifies for --character.
DRAFT_BASE_FAMILIES = ("illustrious", "sdxl", "noobai", "pony")
# Locally trained files carry no catalog metadata (base_model "Unknown"), so their
# family is screened by file-path tokens instead: a token naming a non-SDXL arch
# rejects the record, anything else stays eligible.
UNKNOWN_FAMILY_REJECT_TOKENS = frozenset(
    ("anima", "flux", "chroma", "klein", "kontext", "zimage", "video", "wan", "i2v", "t2v"))
DEFAULT_MIN_MATCH = 0.4
DEFAULT_CHAR_STRENGTH = 0.8


def _venv() -> str:
    return os.environ.get("IMGUTILS_VENV_PY", qc_score.DEFAULT_VENV_PY)


def _guard_env() -> None:
    """An explicit Z3_HOST maps BOTH boxes to one adapter endpoint (see
    loramake/comfy_endpoints.py) — the draft lane and the refine lane would
    hit the same box. Refuse to run until it is unset."""
    if os.environ.get("Z3_HOST"):
        sys.exit(
            "relay_gen: Z3_HOST is set in the environment. That override is a single "
            "opaque endpoint applied to whichever box a run selects, so the zzz draft "
            "lane and the z3 refine lane would target the SAME machine. Unset it and "
            "let per-box resolution route each lane."
        )


def _run(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run one child; rc -9 on timeout (matches the p4_eval_round convention).
    LORAMAKE_ATTEMPT_ID is stripped so children keep the CLI result contract."""
    env = {k: v for k, v in os.environ.items() if k != "LORAMAKE_ATTEMPT_ID"}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return -9, out, err


def _read_gen_result(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != GEN_RESULT_SCHEMA:
        return None
    return data


# ---------------------------------------------------------------- character pick

def _load_catalog(path: Path = CATALOG_INDEX) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"cannot read the LoRA catalog index at {path}: {exc}\n"
                 f"rebuild it with: python3 scripts/lora_catalog.py index")
    if not isinstance(data, list):
        sys.exit(f"unexpected catalog shape at {path} (expected a list of records)")
    return data


_TOKEN_RE = re.compile(r"[a-z0-9]+|[぀-ヿ一-鿿]+")
_SEPARATOR_RE = re.compile(r"[_\-.]+")


def _query_tokens(text: str) -> list[str]:
    """ASCII words plus CJK runs (Han + kana) — CJK has no spaces to split on."""
    return _TOKEN_RE.findall(text.lower())


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _field_hit(tok: str, field: str) -> bool:
    """ASCII tokens match on word boundaries with file separators treated as
    spaces, so 'raiden' still hits 'raiden_il.safetensors' but a short token no
    longer hits the inside of an unrelated word. A CJK run matches when it
    shares at least half of the smaller side's character bigrams with a CJK
    chunk of the field — neither the query nor the catalog can be word-segmented,
    so exact containment would miss e.g. a name embedded mid-phrase."""
    if tok.isascii():
        return re.search(rf"\b{re.escape(tok)}\b", _SEPARATOR_RE.sub(" ", field)) is not None
    if tok in field:
        return True
    grams = _bigrams(tok)
    if not grams:
        return False  # single CJK char: containment above was its only chance
    for chunk in _TOKEN_RE.findall(field):
        if chunk.isascii():
            continue
        chunk_grams = _bigrams(chunk)
        if not chunk_grams:
            continue
        shared = len(grams & chunk_grams)
        if shared and shared / min(len(grams), len(chunk_grams)) >= 0.5:
            return True
    return False


def _record_score(query: str, rec: dict) -> float:
    """Weighted lexical overlap in 0..1: name hits count 3x, trigger/file hits 2x,
    tag hits 1x. Deterministic and offline — no embedding model involved."""
    toks = _query_tokens(query)
    if not toks:
        return 0.0
    name = str(rec.get("name", "")).lower()
    fname = str(rec.get("file", "")).lower()
    triggers = " ".join(str(t) for t in rec.get("triggers") or []).lower()
    tags = " ".join(str(t) for t in rec.get("tags") or []).lower()
    total = 0.0
    for tok in toks:
        if _field_hit(tok, name):
            total += 3.0
        elif _field_hit(tok, triggers) or _field_hit(tok, fname):
            total += 2.0
        elif _field_hit(tok, tags):
            total += 1.0
    return round(total / (3.0 * len(toks)), 4)


def _trigger_tokens(rec: dict) -> str:
    """Comma-joined de-duplicated trigger tokens (catalog entries may themselves
    be comma-joined strings)."""
    seen: dict[str, None] = {}
    for entry in rec.get("triggers") or []:
        for tok in str(entry).split(","):
            tok = tok.strip()
            if tok and tok not in seen:
                seen[tok] = None
    return ", ".join(seen)


def _ineligible_reason(rec: dict) -> str | None:
    """Why this record cannot be auto-picked, or None when it is eligible.

    Unrated records stay eligible — locally trained files have no catalog
    metadata at all, and they are exactly the LoRAs the user reaches for most;
    only an explicit nsfw rating is excluded."""
    if not rec.get("zzz_file"):
        return "file not present on the zzz box"
    family = str(rec.get("base_model") or "").lower()
    if family and family != "unknown":
        if not any(f in family for f in DRAFT_BASE_FAMILIES):
            return f"base family {rec.get('base_model')!r} unusable on the draft checkpoint"
    else:
        path_tokens = set(_query_tokens(str(rec.get("file", ""))))
        marker = path_tokens & UNKNOWN_FAMILY_REJECT_TOKENS
        if marker:
            return (f"family unknown and the file path names a non-SDXL arch "
                    f"({', '.join(sorted(marker))})")
    if rec.get("content_rating") == "nsfw":
        return "nsfw-rated (use an explicit --lora spec)"
    return None


def resolve_character(query: str, min_match: float,
                      catalog: list[dict] | None = None) -> dict:
    """Best ZZZ-ready SFW character LoRA for the query, or a fail-fast exit
    listing the closest candidates and why each one was rejected."""
    records = _load_catalog() if catalog is None else catalog
    scored = sorted(((rec, _record_score(query, rec)) for rec in records),
                    key=lambda pair: -pair[1])
    best = None
    for rec, score in scored:
        if score < min_match:
            break
        if _ineligible_reason(rec) is None:
            best = (rec, score)
            break
    if best is None:
        lines = [f"--character {query!r}: no eligible LoRA scored >= {min_match}"]
        shown = [(rec, score) for rec, score in scored[:5] if score > 0]
        if shown:
            lines[0] += "; closest:"
            for rec, score in shown:
                why = _ineligible_reason(rec) or f"score below --min-match {min_match}"
                lines.append(f"  {score:.2f}  {rec.get('name')}  [{rec.get('base_model')}]  ({why})")
        else:
            lines.append("  (no lexical overlap with any catalog record)")
        lines.append("pass --lora 'FILE:STRENGTH' explicitly, or lower --min-match")
        sys.exit("\n".join(lines))
    rec, score = best
    strength = rec.get("strength_default")
    if strength is None:
        strength = DEFAULT_CHAR_STRENGTH
    return {"query": query, "name": rec.get("name"), "file": rec["file"],
            "zzz_file": rec["zzz_file"], "score": score, "strength": strength,
            "spec": f"{rec['zzz_file']}:{strength}",
            "renamed_on_zzz": rec["zzz_file"] != rec["file"],
            "triggers": _trigger_tokens(rec)}


def apply_character(args: argparse.Namespace,
                    catalog: list[dict] | None = None) -> None:
    """Resolve --character into the --lora stack (mutates args in place).

    The spec carries the ZZZ-side filename because the draft leg renders on
    that box. When that name differs from the Z3-side one, the draft engine's
    own index lookup misses and would skip trigger injection, so the trigger
    words are prepended to the prompt here instead."""
    if not getattr(args, "character", None):
        return
    pick = resolve_character(args.character, args.min_match, catalog=catalog)
    args.character_pick = pick
    args.lora = (args.lora or []) + [pick["spec"]]
    if pick["renamed_on_zzz"] and pick["triggers"]:
        args.prompt = f"{pick['triggers']}, {args.prompt}"


# ---------------------------------------------------------------- stage 1 (zzz)

def pick_recipe(args: argparse.Namespace) -> str:
    return "variants" if args.lora else "krea2"


def stage1_argv(args: argparse.Namespace, recipe: str, index: int, json_out: Path) -> list[str]:
    """One zzz draft. Every draft but the last keeps models resident on the box."""
    last = index == args.n - 1
    seed = args.seed + index if args.seed is not None else None
    if recipe == "variants":
        argv = [sys.executable, str(GEN), "variants", "--box", "zzz",
                "--prompt", args.prompt, "--n", "1"]
        for spec in args.lora or []:
            argv += ["--lora", spec]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if not last:
            argv.append("--no-free")
        argv += ["--run", "--json-out", str(json_out), "--", "--fast"]
        if args.res:
            argv += ["--res", args.res]
        return argv
    if recipe == "krea2":
        argv = [sys.executable, str(KREA2), "--prompt", args.prompt, "--n", "1", "--box", "zzz"]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if args.res:
            argv += ["--res", args.res]
        if not last:
            argv.append("--no-free")
        argv.append("--run")
        return argv
    # "best": slow fallback through the quality engine with the detailer chain off
    argv = [sys.executable, str(GEN), "best", "--box", "zzz",
            "--prompt", args.prompt, "--n", "1"]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if not last:
        argv.append("--no-free")
    argv += ["--run", "--json-out", str(json_out), "--", "--detailers", "none", "--upscale-by", "1.0"]
    if args.res:
        argv += ["--res", args.res]
    return argv


def stage1_files(recipe: str, rc: int, stdout: str, json_out: Path) -> list[str]:
    """Draft paths from the run's own machine-readable output — never a dir scan."""
    if recipe == "krea2":
        return list(dict.fromkeys(KREA2_SAVED_RE.findall(stdout)))
    result = _read_gen_result(json_out)
    if result is None:
        return []
    return list(dict.fromkeys(
        p for p in result.get("new_files") or [] if isinstance(p, str) and p.strip()))


# ------------------------------------------------------------------- the gates

def sfw_pass(path: str, runner=_run) -> tuple[bool, str]:
    """Strict wd14 gate: unverifiable counts as failure (unattended posture)."""
    rc, out, _err = runner([sys.executable, str(SFW_GATE), "--json", "--strict", path],
                           GATE_TIMEOUT_S)
    if rc == 2:
        return False, "unverified (imgutils venv missing — set IMGUTILS_VENV_PY, or use --keep-all)"
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return False, "gate output unreadable"
    return (data.get("ok") is True), ("pass" if data.get("ok") is True else "fail or unverified")


def ccip_pass(ref: str, path: str, runner=_run) -> bool:
    rc, out, _err = runner(
        [sys.executable, str(CONSISTENCY), "--ref", ref, "--keep", path], GATE_TIMEOUT_S)
    kept = {line.strip() for line in out.splitlines() if line.strip()}
    return rc == 0 and path in kept


def evaluate_draft(path: str, args: argparse.Namespace, runner=_run) -> dict:
    """Check one draft against whichever gates are enabled: --gate turns on the
    SFW + qc pair (fail-closed), --ref adds the CCIP identity check. With
    nothing enabled the draft passes untouched — the user reviews output."""
    row: dict = {"path": path, "hard_pass": False, "clean": False,
                 "percentile": None, "flags": [], "reason": "pass"}
    if args.gate:
        ok, why = sfw_pass(path, runner=runner)
        if not ok:
            row["reason"] = f"sfw:{why}"
            return row
    if args.ref and not ccip_pass(args.ref, path, runner=runner):
        row["reason"] = "ccip:identity mismatch"
        return row
    if not args.gate:
        row.update(hard_pass=True, clean=True, reason="ungated")
        return row
    qc = qc_score.qc_one(_venv(), path)
    if qc is None:
        row["reason"] = "qc:unavailable"
        return row
    row["hard_pass"] = True
    row["percentile"] = qc.get("aesthetic_percentile")
    row["flags"] = qc_score.flags_for(qc, args.subjects)
    row["clean"] = not row["flags"]
    if row["flags"]:
        row["reason"] = "qc:" + ",".join(row["flags"])
    return row


def rank_select(rows: list[dict], top: int) -> list[str]:
    """Rank-mode pick: hard-pass rows, unflagged before flagged, percentile desc."""
    eligible = [r for r in rows if r["hard_pass"]]
    ordered = sorted(eligible, key=lambda r: (not r["clean"], -(r["percentile"] or 0.0)))
    return [r["path"] for r in ordered[:top]]


# ---------------------------------------------------------------- stage 2 (z3)

def sidecar_prompt(draft: str) -> tuple[str | None, str]:
    """The draft's final positive from its sidecar. The sidecar sampler key is a
    combined display string and is never forwarded."""
    meta_path = Path(str(draft) + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "sidecar missing or unreadable"
    prompt = meta.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt, "sidecar"
    return None, "sidecar has no prompt"


def stage2_argv(draft: str, args: argparse.Namespace, json_out: Path) -> tuple[list[str], str]:
    """One z3 refine through the quality engine's refine path. Always keeps the
    model resident (--no-free); the lane releases VRAM once at the end. The
    refine checkpoint stays the engine default: --base takes quirks-card names,
    not the file names a sidecar records."""
    prompt, source = sidecar_prompt(draft)
    argv = [sys.executable, str(GEN), "best", "--box", "z3"]
    if prompt:
        argv += ["--prompt", prompt]
    argv += ["--no-free", "--run", "--json-out", str(json_out), "--",
             "--refine-image", str(draft)]
    for flag, value in (("--refine-steps", args.refine_steps),
                        ("--refine-denoise", args.refine_denoise),
                        ("--upscale-by", args.upscale_by),
                        ("--detailers", args.detailers)):
        if value is not None:
            argv += [flag, str(value)]
    if prompt:
        # the sidecar prompt is already final: no scaffold doubling, no re-expansion
        argv += ["--no-scaffold", "--no-wildcard"]
    return argv, source


def probe_z3(max_wait: float) -> tuple[bool, str]:
    import gen  # sibling; lazy so offline tests can stub this function

    host = resolve_comfy_host("z3", explicit=None)
    return gen.wait_for_idle(host, max_wait=max_wait)


def release_box(box: str) -> None:
    """Release one box's VRAM via the router's own standing helper."""
    import gen

    gen._free_vram(resolve_comfy_host(box, explicit=None))


def release_z3() -> None:
    release_box("z3")


def refine_one(draft: str, args: argparse.Namespace, stamp_dir: Path, index: int,
               runner=_run) -> dict:
    json_out = stamp_dir / f"refine_{index:02d}.json"
    argv, prompt_source = stage2_argv(draft, args, json_out)
    started = time.monotonic()
    rc, _out, err = runner(argv, args.stage_timeout)
    result = _read_gen_result(json_out)
    new_files = [p for p in (result or {}).get("new_files") or []
                 if isinstance(p, str) and p.strip()]
    return {"draft": draft, "argv": argv, "rc": rc, "new_files": new_files,
            "prompt_source": prompt_source, "wall_s": round(time.monotonic() - started, 1),
            "stderr_tail": err[-500:] if rc != 0 else ""}


# ------------------------------------------------------------------- the lanes

def z3_lane(q: "queue_mod.Queue", state: dict, args: argparse.Namespace,
            lock: threading.Lock, stamp_dir: Path, runner=_run) -> None:
    """Consumer: gate each landed draft, refine the first --top passers on z3.

    The --top quota counts refine ATTEMPTS, not successes: a failed refine still
    consumes a slot. Deliberate — the box is strictly serial, so retrying into
    freed slots would stretch the run unboundedly on a persistently failing z3."""
    ready, reason = probe_z3(args.z3_wait)
    with lock:
        state["z3_probe"] = {"ready": ready, "reason": reason}
    refined_any = False
    index = 0
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        path = item["path"]
        if not ready:
            with lock:
                state["skipped"].append({"draft": path, "reason": f"z3 not idle: {reason}"})
            continue
        if args.keep_all:
            row = {"path": path, "hard_pass": True, "clean": True,
                   "percentile": None, "flags": [], "reason": "keep-all"}
        else:
            row = evaluate_draft(path, args, runner=runner)
        with lock:
            state["gates"].append(row)
            quota_full = (not args.keep_all) and len(state["stage2"]) >= args.top
        if not (row["hard_pass"] and row["clean"]):
            continue
        if quota_full:
            with lock:
                state["skipped"].append({"draft": path, "reason": "refine quota already full"})
            continue
        run = refine_one(path, args, stamp_dir, index, runner=runner)
        index += 1
        with lock:
            state["stage2"].append(run)
            if run["rc"] == 0 and run["new_files"] and state["first_refined_s"] is None:
                state["first_refined_s"] = round(time.monotonic() - state["t0"], 1)
        refined_any = True
    if refined_any:
        release_z3()


def zzz_lane(args: argparse.Namespace, recipe: str, state: dict, lock: threading.Lock,
             stamp_dir: Path, on_draft, runner=_run) -> str:
    """Producer: render drafts one at a time on zzz; hand each to on_draft(path)."""
    for index in range(args.n):
        attempts = 0
        while True:
            json_out = stamp_dir / f"draft_{index:02d}.json"
            argv = stage1_argv(args, recipe, index, json_out)
            started = time.monotonic()
            rc, out, err = runner(argv, args.stage_timeout)
            files = stage1_files(recipe, rc, out, json_out)
            with lock:
                state["drafts"].append({
                    "index": index, "recipe": recipe, "argv": argv, "rc": rc,
                    "files": files, "wall_s": round(time.monotonic() - started, 1),
                    "stderr_tail": err[-500:] if rc != 0 and not files else ""})
                if files and state["first_draft_s"] is None:
                    state["first_draft_s"] = round(time.monotonic() - state["t0"], 1)
            if recipe == "krea2" and rc != 0 and not files and attempts == 0:
                recipe = "best"
                attempts += 1
                # the failed leg ran with --no-free; drop whatever it left resident
                release_box("zzz")
                with lock:
                    state["notes"].append(
                        "krea2 leg failed on zzz (assets likely missing) — "
                        "switched to the slow quality fallback")
                continue
            break
        for path in files:
            on_draft(path)
    return recipe


# ----------------------------------------------------------------- run + report

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def assemble_result(args: argparse.Namespace, recipe: str, state: dict) -> dict:
    drafts = [p for row in state["drafts"] for p in row["files"]]
    refined = [p for run in state["stage2"] for p in run["new_files"] if run["rc"] == 0]
    attempted = len(state["stage2"])
    probe = state.get("z3_probe") or {}
    if not drafts:
        status, rc = "stage1_failed", 2
    elif refined:
        status, rc = "refined", 0
    elif probe and not probe.get("ready"):
        status, rc = "drafts_only", 0
    elif attempted:
        status, rc = "refine_failed", 5
    else:
        status, rc = "gated_out", 4
    total = round(time.monotonic() - state["t0"], 1)
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status, "rc": rc,
        "mode": "rank" if args.rank else "stream",
        "prompt": args.prompt, "n": args.n, "top": args.top, "seed": args.seed,
        "lora": args.lora or [], "ref": args.ref, "recipe": recipe,
        "character": getattr(args, "character_pick", None),
        "stage1": state["drafts"],
        "gates": state["gates"],
        "skipped": state["skipped"],
        "z3": probe,
        "stage2": state["stage2"],
        "final": {"refined": refined, "drafts": drafts},
        "notes": state["notes"],
        "latency": {"first_draft_s": state["first_draft_s"],
                    "first_refined_s": state["first_refined_s"],
                    "total_s": total},
    }


def print_plan(args: argparse.Namespace, recipe: str) -> None:
    mode = "rank (batch, global top-K)" if args.rank else "stream (first-K pass, lowest latency)"
    fake = Path("<stamp>/draft_00.json")
    print(f"[relay plan] mode={mode}  n={args.n}  top={args.top}")
    print(f"zzz lane — one draft at a time (recipe: {recipe}; models resident until the last):")
    print("  " + shlex.join(stage1_argv(args, recipe, 0, fake)))
    if args.keep_all or not (args.gate or args.ref):
        gates = "off — every landed draft is eligible, user reviews output"
    else:
        parts = []
        if args.gate:
            parts.append("sfw --strict (fail-closed)")
        if args.ref:
            parts.append("ccip vs --ref")
        if args.gate:
            parts.append("qc defect flags")
        gates = " -> ".join(parts)
    print(f"z3 lane — gate then refine (gates: {gates}):")
    argv, _src = stage2_argv("<draft.png>", args, Path("<stamp>/refine_00.json"))
    print("  " + shlex.join(argv))
    print("nothing was spawned — add --run to execute")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="draft count on zzz")
    ap.add_argument("--seed", type=int, default=None, help="base seed; draft i uses seed+i")
    ap.add_argument("--lora", action="append", default=None,
                    help="LoRA spec (repeatable) — selects the fast variants leg")
    ap.add_argument("--character", default=None, metavar="TEXT",
                    help="pick the best-matching character LoRA from the catalog "
                         "(lexical match over names/triggers/tags; ZZZ-ready, "
                         "draft-compatible, not nsfw-rated) and add it to the "
                         "--lora stack")
    ap.add_argument("--min-match", type=float, default=DEFAULT_MIN_MATCH,
                    help=f"score floor (0..1) for --character (default {DEFAULT_MIN_MATCH})")
    ap.add_argument("--res", default=None, help="WxH forwarded to the draft engine")
    ap.add_argument("--ref", default=None, help="identity anchor image for the CCIP gate")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP, help="how many drafts to refine")
    ap.add_argument("--gate", action="store_true",
                    help="enable the unattended SFW + qc gates (default: off, user reviews output)")
    ap.add_argument("--keep-all", action="store_true",
                    help="skip every gate and refine all drafts")
    ap.add_argument("--rank", action="store_true",
                    help="wait for the full batch and refine the global top-K "
                         "instead of streaming (implies --gate: ranking needs scores)")
    ap.add_argument("--subjects", type=int, default=1, help="expected face count for qc flags")
    ap.add_argument("--refine-steps", type=int, default=None)
    ap.add_argument("--refine-denoise", type=float, default=None)
    ap.add_argument("--upscale-by", type=float, default=None)
    ap.add_argument("--detailers", default=None, help="refine detailer set, e.g. face")
    ap.add_argument("--z3-wait", type=float, default=DEFAULT_Z3_WAIT,
                    help="max seconds to wait for an idle z3 before delivering drafts only")
    ap.add_argument("--stage-timeout", type=float, default=DEFAULT_STAGE_TIMEOUT,
                    help="hard per-subprocess timeout")
    ap.add_argument("--run", action="store_true", help="execute (default: print the plan)")
    ap.add_argument("--json-out", default=None, metavar="PATH")
    args = ap.parse_args(argv)
    if args.n < 1 or args.top < 1:
        ap.error("--n and --top must be >= 1")
    if not 0.0 <= args.min_match <= 1.0:
        ap.error("--min-match must be within 0..1")
    if args.rank:
        args.gate = True
    args.character_pick = None
    return args


def run_relay(args: argparse.Namespace, runner=_run) -> dict:
    recipe = pick_recipe(args)
    stamp_dir = RELAY_BASE / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    state: dict = {"drafts": [], "gates": [], "skipped": [], "stage2": [],
                   "notes": [], "z3_probe": None, "t0": time.monotonic(),
                   "first_draft_s": None, "first_refined_s": None}
    lock = threading.Lock()
    if args.rank:
        collected: list[str] = []
        recipe = zzz_lane(args, recipe, state, lock, stamp_dir, collected.append,
                          runner=runner)
        if collected:
            ready, reason = probe_z3(args.z3_wait)
            state["z3_probe"] = {"ready": ready, "reason": reason}
            if ready:
                if args.keep_all:
                    picks = list(collected)
                else:
                    rows = [evaluate_draft(p, args, runner=runner) for p in collected]
                    state["gates"] = rows
                    picks = rank_select(rows, args.top)
                refined_any = False
                for index, draft in enumerate(picks):
                    run = refine_one(draft, args, stamp_dir, index, runner=runner)
                    state["stage2"].append(run)
                    if run["rc"] == 0 and run["new_files"] and state["first_refined_s"] is None:
                        state["first_refined_s"] = round(time.monotonic() - state["t0"], 1)
                    refined_any = True
                if refined_any:
                    release_z3()
    else:
        q: "queue_mod.Queue" = queue_mod.Queue()
        worker = threading.Thread(
            target=z3_lane, args=(q, state, args, lock, stamp_dir),
            kwargs={"runner": runner}, daemon=True)
        worker.start()
        recipe = zzz_lane(args, recipe, state, lock, stamp_dir,
                          lambda p: q.put({"path": p}), runner=runner)
        q.put(_SENTINEL)
        worker.join()
    result = assemble_result(args, recipe, state)
    _write_json(stamp_dir / "relay_result.json", result)
    if args.json_out:
        _write_json(Path(args.json_out), result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _guard_env()
    apply_character(args)
    if args.character_pick:
        pick = args.character_pick
        print(f"[character] {pick['name']}  score={pick['score']}  -> {pick['spec']}")
    if not args.run:
        print_plan(args, pick_recipe(args))
        return 0
    result = run_relay(args)
    lat = result["latency"]
    print(f"[relay] status={result['status']}  refined={len(result['final']['refined'])}  "
          f"drafts={len(result['final']['drafts'])}  first_draft={lat['first_draft_s']}s  "
          f"first_refined={lat['first_refined_s']}s  total={lat['total_s']}s")
    for path in result["final"]["refined"] or result["final"]["drafts"]:
        print(f"  {path}")
    return result["rc"]


if __name__ == "__main__":
    sys.exit(main())
