#!/usr/bin/env python3
"""Render a hash-bound ``loramake-video-edit-v1`` timeline with FFmpeg.

Examples:

  python3 scripts/video_edit.py render edit.json
  python3 scripts/video_edit.py render edit.json --profile master --out master.mov --json

The command never replaces an existing render or evidence file.  Relative asset and
output paths are resolved against the edit JSON directory, not the caller's cwd.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake.video_edit import PROFILE_ARGS, VideoEditError, load_and_validate_edit, render_edit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic FFmpeg editor for loramake-video-edit-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="validate, render, ffprobe, and publish no-clobber output")
    render.add_argument("spec", help="path to loramake-video-edit-v1 JSON")
    render.add_argument("--profile", choices=sorted(PROFILE_ARGS), help="override the profile in the spec")
    render.add_argument("--out", help="override output path (relative to the spec directory when not absolute)")
    render.add_argument("--ffmpeg", default="ffmpeg", help="local ffmpeg executable (default: ffmpeg)")
    render.add_argument("--ffprobe", default="ffprobe", help="local ffprobe executable (default: ffprobe)")
    render.add_argument("--timeout", type=float, default=3600.0, help="render timeout seconds (default: 3600)")
    render.add_argument("--json", action="store_true", help="print a stable JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = load_and_validate_edit(
            args.spec,
            profile_override=args.profile,
            output_override=args.out,
        )
        result = render_edit(contract, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, timeout=args.timeout)
    except VideoEditError as exc:
        print(f"video_edit: {exc}", file=sys.stderr)
        return 2
    payload = {
        "ok": True,
        "output": str(result.output),
        "evidence": str(result.evidence),
        "stream_contract": result.stream_contract,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(result.output)
        print(result.evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
