#!/usr/bin/env python3
"""Tiny local adapter used only by video_finish integration tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def rate(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--version" and sys.argv[2] in {"rife", "seedvr2"}:
        print("RIFE 4.7.0" if sys.argv[2] == "rife" else "SeedVR2 3.0.0")
        return 0
    if len(sys.argv) != 5 or sys.argv[1] not in {"rife", "seedvr2"}:
        print("usage: fake_video_finish_adapter.py rife|seedvr2 INPUT OUTPUT FACTOR", file=sys.stderr)
        return 2
    mode, input_text, output_text, factor_text = sys.argv[1:]
    source = Path(input_text)
    output = Path(output_text)
    factor = int(factor_text)
    stream = probe(source)
    if mode == "rife":
        vf = f"fps={rate(stream['avg_frame_rate']) * factor:g}:round=near"
    else:
        vf = f"scale={int(stream['width']) * factor}:{int(stream['height']) * factor}:flags=neighbor"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-vf", vf, "-an", "-map_metadata", "-1", "-c:v", "ffv1", str(output),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
