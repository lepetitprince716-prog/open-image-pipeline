#!/usr/bin/env python3
"""One-JSON-in image generation: run a job brief through the gen.py router.

The brief is the same JSON shape the Studio app queues (app/brief.py owns validation and
the gen.py translation), so a CLI caller and the app share one vocabulary and one guard
layer. This entry adds nothing on top: it validates, prints or executes the exact router
invocation, and hands back the router's machine-readable result JSON.

Usage:
  gen_job.py JOB.json                 # validate + execute; result JSON printed at the end
  echo '{"tier":"best","prompt":"1girl, silver hair"}' | gen_job.py -
  gen_job.py JOB.json --print         # validate + show the router command, touch nothing
  gen_job.py --fields                 # brief vocabulary quick reference
  gen_job.py JOB.json --json-out PATH # keep the result file at PATH (default: temp, echoed)

Exit codes: 0 ok, 2 invalid brief, otherwise the router's exit code.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:5]):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate repo root containing loramake/__init__.py")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import brief as brief_mod  # noqa: E402


def _print_fields() -> None:
    print("tier      :", ", ".join(brief_mod.TIERS))
    print("box       :", ", ".join(brief_mod.BOXES), "(default auto)")
    print("provider  :", ", ".join(brief_mod.PROVIDERS), "(default local)")
    print("required  : tier, prompt (sequence also: ref; pilot: task_spec)")
    print("common    : n (1..%d), seed, base (quirks-card stem), lora, loras[], res WxH," % brief_mod.MAX_N)
    print("            steps, cfg, sampler, scheduler, neg_add, no_freeu, pag, pick_best,")
    print("            keep_resident, sfw, nsfw, group")
    print("artistic  :", ", ".join(brief_mod.ARTISTIC_ONLY_FIELDS))
    print("samplers  :", ", ".join(brief_mod.ENGINE_SAMPLERS))
    print("schedulers:", ", ".join(brief_mod.ENGINE_SCHEDULERS))
    print("aspects   :", ", ".join(brief_mod.ARTISTIC_ASPECTS))
    print("\nExample: {\"tier\": \"best\", \"prompt\": \"1girl, silver hair, gothic dress\", \"n\": 2}")


def _load_brief(source: str) -> object:
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"brief is not valid JSON: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brief", nargs="?", help="brief JSON file, or '-' for stdin")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="validate and print the router command without executing")
    ap.add_argument("--json-out", default=None, metavar="PATH",
                    help="where the router writes its result JSON (default: a temp file)")
    ap.add_argument("--fields", action="store_true", help="print the brief vocabulary and exit")
    args = ap.parse_args()

    if args.fields:
        _print_fields()
        return
    if not args.brief:
        ap.error("pass a brief JSON file, '-' for stdin, or --fields")

    brief = _load_brief(args.brief)
    problems = brief_mod.validate(brief)
    if problems:
        for problem in problems:
            print(f"  invalid: {problem}", file=sys.stderr)
        raise SystemExit(2)

    if args.json_out:
        json_out = Path(args.json_out).expanduser()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        keep_result = True
    else:
        fd, tmp = tempfile.mkstemp(prefix="gen_job_", suffix=".json")
        import os
        os.close(fd)
        json_out = Path(tmp)
        keep_result = False

    argv = brief_mod.to_argv(brief, json_out)
    if args.print_only:
        print(" ".join(shlex.quote(a) for a in argv))
        if not keep_result:
            json_out.unlink(missing_ok=True)
        return

    rc = subprocess.run(argv).returncode
    result_text = ""
    if json_out.is_file():
        result_text = json_out.read_text(encoding="utf-8").strip()
    if result_text:
        print("\n[result]")
        print(result_text)
        if keep_result:
            print(f"[result-file] {json_out}")
    else:
        print("\n[result] router wrote no result JSON", file=sys.stderr)
    if not keep_result:
        json_out.unlink(missing_ok=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
