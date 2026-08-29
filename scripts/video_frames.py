#!/usr/bin/env python3
"""Sample frames out of a video clip and tile them into a contact sheet (film strip).

An .mp4 is opaque to Read and to a git diff — this makes a clip judgeable at a glance:
K sampled frames (always including the first and last, where identity drift is worst)
become one `<clip stem>.sheet.png` next to the clip, via contact_sheet.build_sheet.
Frame labels carry the frame index and timestamp so the sheet doubles as a drift
timeline. Pure ffmpeg/ffprobe + PIL — no GPU, runs on the Mac.

z3_video.py calls this best-effort after every render; it also replaces the manual
"ffmpeg select=eq(n,N)" one-liner for pulling a single frame (e.g. the last frame for
a future re-seed continuation).

Usage:
  video_frames.py CLIP.mp4                     # 9-frame contact sheet -> CLIP.sheet.png
  video_frames.py CLIP.mp4 --frames 17         # denser sampling
  video_frames.py CLIP.mp4 --frame 80          # single frame -> scratch dir (or --out)
Extracted full-res frames stay in outputs/_gallery/video_frames/<stem>-<path-hash>/<kind>/
(outside the gallery scan roots) for close inspection. The path hash prevents probe arms
with the same clip basename from overwriting one another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS  # noqa: E402

SCRATCH = OUTPUTS / "_gallery" / "video_frames"  # NOT a gallery scan root
DEF_FRAMES = 9


class VideoFrameError(RuntimeError):
    """Inspection/extraction failure raised by library calls; CLI converts it to a clean exit."""


def scratch_for(clip: str | Path) -> Path:
    """Collision-proof scratch directory for clips that share a basename in different arms."""
    path = Path(clip).resolve()
    key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return SCRATCH / f"{path.stem}-{key}"


def _rate(value: object) -> float:
    num, _, den = str(value or "0/1").partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def inspect_video(clip: str | Path) -> dict | None:
    """Return authoritative stream/container metrics, or ``None`` for an invalid clip.

    This is the shared post-download and release-audit seam. A file signature alone is
    deliberately insufficient: ffprobe must be able to demux a real video stream.
    """
    path = Path(clip)
    if not path.is_file():
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries",
           "stream=codec_name,width,height,avg_frame_rate,nb_read_packets,nb_frames:"
           "format=format_name,duration,size",
           "-of", "json", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    except (FileNotFoundError, OSError):
        return None
    if res.returncode != 0:
        return None
    try:
        payload = json.loads(res.stdout)
        stream = payload["streams"][0]
        fmt = payload.get("format", {})
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(stream, dict) or not isinstance(fmt, dict):
        return None
    try:
        frames = int(stream.get("nb_read_packets") or stream.get("nb_frames") or 0)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float(fmt.get("duration") or 0.0)
        reported_size = int(fmt.get("size") or 0)
    except (TypeError, ValueError):
        return None
    fps = _rate(stream.get("avg_frame_rate"))
    format_name = str(fmt.get("format_name") or "")
    container = "mp4" if any(name in format_name.split(",") for name in ("mov", "mp4")) else None
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0 or not container:
        return None
    try:
        actual_size = path.stat().st_size
    except OSError:
        return None
    return {
        "container": container,
        "format_name": format_name,
        "codec": str(stream.get("codec_name") or ""),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_s": duration,
        "bytes": actual_size,
        "reported_bytes": reported_size,
    }


def probe(clip: Path) -> tuple[int, float]:
    """(frame_count, fps) via the shared authoritative ffprobe inspection."""
    observed = inspect_video(clip)
    if observed is None:
        raise VideoFrameError(f"ffprobe returned no valid MP4 video stream for {clip}")
    return observed["frames"], observed["fps"]


def sample_indices(n_frames: int, k: int) -> list[int]:
    """K indices spread over the clip, always including frame 0 and the last frame."""
    if n_frames <= 0:
        raise VideoFrameError("clip reports zero frames")
    k = max(2, min(k, n_frames))
    return sorted({round(i * (n_frames - 1) / (k - 1)) for i in range(k)})


def extract(clip: Path, indices: list[int], fps: float, dest: Path) -> list[Path]:
    """One ffmpeg pass pulling the selected frames as PNGs named f<idx>_t<sec>s.png."""
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.png"):
        old.unlink()
    sel = "+".join(f"eq(n\\,{i})" for i in indices)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(clip),
           "-vf", f"select={sel}", "-fps_mode", "vfr", str(dest / "tmp_%03d.png")]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise VideoFrameError(f"ffmpeg timed out while extracting {clip}") from None
    if res.returncode != 0:
        raise VideoFrameError(f"ffmpeg failed: {res.stderr.strip()[:300]}")
    tmp = sorted(dest.glob("tmp_*.png"))  # select emits in frame order == sorted indices
    if len(tmp) != len(indices):
        raise VideoFrameError(f"expected {len(indices)} frames, ffmpeg produced {len(tmp)}")
    out: list[Path] = []
    for path, idx in zip(tmp, indices):
        sec = (idx / fps) if fps else 0.0
        final = dest / f"f{idx:03d}_t{sec:.1f}s.png"
        path.rename(final)
        out.append(final)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Contact sheet / frame pulls for a video clip (ffmpeg + PIL)")
    ap.add_argument("clip", help="video file (.mp4)")
    ap.add_argument("--frames", type=int, default=DEF_FRAMES,
                    help=f"frames to sample for the sheet (default {DEF_FRAMES}; first+last always included)")
    ap.add_argument("--frame", type=int, default=None,
                    help="extract this single frame index instead of building a sheet")
    ap.add_argument("--out", default=None, help="output path (sheet PNG, or the single frame PNG)")
    ap.add_argument("--thumb", type=int, default=480, help="sheet tile max edge px (default 480)")
    ap.add_argument("--scratch-kind", choices=("inspection", "sfw_sample", "sfw_full", "identity"),
                    default="inspection", help="isolated evidence lane under the clip scratch root")
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.is_file():
        sys.exit(f"clip not found: {clip}")
    try:
        n_frames, fps = probe(clip)
    except VideoFrameError as exc:
        sys.exit(str(exc))
    if args.frame is None and args.frames < 2:
        sys.exit("--frames must be >= 2")
    scratch = scratch_for(clip) / args.scratch_kind

    if args.frame is not None:
        if not 0 <= args.frame < n_frames:
            sys.exit(f"--frame {args.frame} out of range (clip has {n_frames} frames)")
        try:
            frame = extract(clip, [args.frame], fps, scratch)[0]
        except VideoFrameError as exc:
            sys.exit(str(exc))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            frame = Path(str(frame.rename(args.out)))
        print(frame)
        return

    indices = sample_indices(n_frames, args.frames)
    try:
        frames = extract(clip, indices, fps, scratch)
    except VideoFrameError as exc:
        sys.exit(str(exc))
    sheet = args.out or str(clip.with_name(clip.stem + ".sheet.png"))
    from contact_sheet import build_sheet

    build_sheet([str(p) for p in frames], sheet, thumb=args.thumb)
    print(f"{sheet}  ({len(frames)} frames of {n_frames} @ {fps:g}fps; full-res in {scratch})")


if __name__ == "__main__":
    main()
