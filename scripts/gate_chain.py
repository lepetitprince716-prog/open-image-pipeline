#!/usr/bin/env python3
"""Gate chain: run the uniform SFW, defect and structure gates on images and write reports.

Wraps ``loramake/gates.py``. Images come from paths or ``--dir`` (recursive png / jpg / webp); each
image gets a ``<name>.gates.json`` sidecar and the batch gets one report file. Runs on this
machine through the imgutils venv bridge; no GPU box is touched. Exit 1 when any image fails
in automated mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loramake import gates  # noqa: E402


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def collect(paths: Sequence[str], directory: str | None) -> list[str]:
    out = [p for p in paths]
    if directory:
        out.extend(str(p) for p in sorted(Path(directory).rglob("*"))
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return [p for p in out if not p.endswith(".gates.json")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the uniform quality gate chain on images")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--dir", help="also gate every *.png under this directory")
    parser.add_argument("--mode", choices=gates.MODES, default="automated")
    parser.add_argument("--subjects", type=int, default=1)
    parser.add_argument("--floor", type=float, default=gates.AESTHETIC_FLOOR)
    parser.add_argument("--allow-monochrome", action="store_true",
                        help="skip the monochrome rule for grey, ink or monochrome styles")
    parser.add_argument("--style", help="style id from catalogs/style; grey and ink styles imply "
                                        "--allow-monochrome")
    parser.add_argument("--venv", default=None, help="imgutils venv interpreter")
    parser.add_argument("--report", help="write the batch report JSON here")
    parser.add_argument("--no-sidecars", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the batch report to stdout")
    args = parser.parse_args(argv)

    paths = collect(args.images, args.dir)
    if not paths:
        raise SystemExit("no images given")
    allow_monochrome = args.allow_monochrome
    if args.style:
        from loramake import style_registry as sr

        if args.style not in sr.load_cards():
            raise SystemExit(f"unknown style {args.style!r}; run style_cards.py list")
        allow_monochrome = allow_monochrome or gates.monochrome_expected(args.style)
    reports = gates.run_chain(paths, mode=args.mode, subjects=args.subjects, venv=args.venv,
                              floor=args.floor, allow_monochrome=allow_monochrome)
    if not args.no_sidecars:
        gates.write_sidecars(reports)
    summary = gates.summarize(reports)
    payload = {"schema": gates.REPORT_SCHEMA, "mode": args.mode, "subjects": args.subjects,
               "floor": args.floor, "allow_monochrome": allow_monochrome, "style": args.style,
               "summary": summary, "reports": [r.as_dict() for r in reports]}
    if args.report:
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            failing = [f"{n}:{','.join(g.reasons)}" for n, g in report.gates.items() if g.status != "pass"]
            print(f"{report.overall:10s} {report.path}" + (f"  {' '.join(failing)}" if failing else ""))
        print(f"{summary['images']} images: pass {summary['overall']['pass']}, "
              f"fail {summary['overall']['fail']}, unverified {summary['overall']['unverified']}")
    return 1 if args.mode == "automated" and summary["overall"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
