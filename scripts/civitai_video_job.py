#!/usr/bin/env python3
"""Civitai cloud video job runner: shots manifest -> t2i -> i2v -> concat.

Thin deterministic layer over the vendored official CLI (tools/civitai-gen,
github.com/civitai/civitai-gen-skill). The LLM plans shots (prompts, engines,
durations) into a shots JSON; this script executes them reproducibly:

  init    create a job dir from a shots JSON
  plan    free whatif cost per pending shot + total (no Buzz spent)
  run     execute pending shots in order (REQUIRES --confirm-spend)
  frame   extract the last frame of a video (shot chaining helper)
  concat  stitch finished clips into final.mp4 (local ffmpeg, free)
  status  print job manifest summary

Shot fields (shots JSON, list under "shots"):
  id               int, unique, execution order
  prompt           video prompt (required)
  engine           videoGen engine (default from "defaults", e.g. ltx2)
  duration         seconds (engine-specific menu)
  aspect           aspect ratio, e.g. 16:9 (optional)
  version          engine version, e.g. wan v2.6 (optional)
  resolution       engine resolution tier, e.g. 720p (optional)
  image_prompt     optional t2i prompt; generates the source image first
  image_model      optional AIR urn for the t2i checkpoint
  image            optional explicit source image (local path or URL)
  chain_from_prev  bool; use last frame of the previous shot's clip as source

Source-image resolution order: image > chain_from_prev > image_prompt > none
(pure t2v). Spending is gated: only `run --confirm-spend` submits paid work;
everything else is free. Token is read from ~/.config/civitai/token and passed
to the CLI via the environment, never argv.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_CLI = REPO_ROOT / "tools" / "civitai-gen" / "generate.mjs"
JOBS_ROOT = REPO_ROOT / "outputs" / "civitai_video"
TOKEN_PATH = Path(os.environ.get("CIVITAI_TOKEN_PATH", "~/.config/civitai/token")).expanduser()

VIDEO_EXTS = (".mp4", ".webm", ".mov")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _cli_env():
    if not TOKEN_PATH.exists():
        _fail(f"token file not found: {TOKEN_PATH}")
    env = dict(os.environ)
    env["CIVITAI_API_KEY"] = TOKEN_PATH.read_text().strip()
    return env


def _run_cli(args, timeout=900):
    """Run the vendored generate.mjs; returns (rc, stdout, buzz_cost). Token via env only."""
    cmd = ["node", str(GEN_CLI)] + args
    proc = subprocess.run(cmd, env=_cli_env(), capture_output=True, text=True, timeout=timeout)
    noise = ("trust settings", "UNDICI-EHPA", "trace-warnings")
    err = "\n".join(l for l in proc.stderr.splitlines() if not any(n in l for n in noise))
    if err.strip():
        print(err, file=sys.stderr)
    cost = sum(int(m) for m in re.findall(r"Cost:\s+(\d+)\s+buzz", proc.stderr + proc.stdout))
    return proc.returncode, proc.stdout, cost


def _job_dir(name):
    return JOBS_ROOT / name


def _manifest_path(name):
    return _job_dir(name) / "shots.json"


def _load(name):
    p = _manifest_path(name)
    if not p.exists():
        _fail(f"job not found: {name} (expected {p})")
    return json.loads(p.read_text())


def _save(name, manifest):
    _manifest_path(name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _shot_defaults(manifest, shot):
    d = manifest.get("defaults", {})
    return {
        "engine": shot.get("engine") or d.get("engine", "ltx2"),
        "duration": shot.get("duration") or d.get("duration", 5),
        "aspect": shot.get("aspect") or d.get("aspect"),
        "version": shot.get("version") or d.get("version"),
        "resolution": shot.get("resolution") or d.get("resolution"),
        "style": shot.get("style") or d.get("style"),
    }


def _video_args(manifest, shot, prompt):
    s = _shot_defaults(manifest, shot)
    args = ["--engine", s["engine"], "--duration", str(s["duration"]), "--prompt", prompt]
    if s["aspect"]:
        args += ["--video-aspect", s["aspect"]]
    if s["version"]:
        args += ["--version", s["version"]]
    if s["resolution"]:
        args += ["--video-resolution", s["resolution"]]
    if s["style"]:
        args += ["--style", s["style"]]
    return args


def _find_media(directory, exts):
    files = sorted(
        (p for p in Path(directory).iterdir() if p.suffix.lower() in exts),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


# ---------------------------------------------------------------- commands

def cmd_init(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        _fail("job name must be a safe slug (letters, digits, dot, dash, underscore)")
    spec = json.loads(Path(args.shots).read_text())
    shots = spec.get("shots")
    if not shots:
        _fail("shots JSON must contain a non-empty 'shots' list")
    seen = set()
    for shot in shots:
        if "id" not in shot or "prompt" not in shot:
            _fail("every shot needs 'id' and 'prompt'")
        if shot["id"] in seen:
            _fail(f"duplicate shot id {shot['id']}")
        seen.add(shot["id"])
        shot.setdefault("status", "pending")
        shot.setdefault("buzz_spent", 0)
    job = _job_dir(args.name)
    if _manifest_path(args.name).exists() and not args.force:
        _fail(f"job {args.name} already exists (use --force to overwrite the manifest)")
    job.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job": args.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "defaults": spec.get("defaults", {}),
        "shots": sorted(shots, key=lambda s: s["id"]),
        "buzz_total": 0,
    }
    _save(args.name, manifest)
    print(f"initialized {args.name}: {len(shots)} shots at {job}")


def cmd_plan(args):
    manifest = _load(args.name)
    total = 0
    print(f"{'id':>3} {'status':<10} {'engine':<8} {'dur':>4} {'buzz':>6}  prompt")
    for shot in manifest["shots"]:
        cost = "-"
        if shot["status"] != "clip_done":
            rc, out, _ = _run_cli(["cost"] + _video_args(manifest, shot, shot["prompt"]), timeout=120)
            if rc == 0:
                try:
                    data = json.loads(out[out.index("{"):])
                    cost = data["cost"]["total"]
                    total += cost
                except (ValueError, KeyError):
                    cost = "?"
            else:
                cost = "!"
        s = _shot_defaults(manifest, shot)
        print(f"{shot['id']:>3} {shot['status']:<10} {s['engine']:<8} {s['duration']:>4} {cost!s:>6}  {shot['prompt'][:60]}")
    extra = sum(1 for s in manifest["shots"] if s.get("image_prompt") and s["status"] == "pending")
    if extra:
        print(f"+ {extra} t2i source image(s), ~4-12 buzz each (not included above)")
    print(f"estimated pending video total: {total} buzz (whatif, nothing spent)")


def _resolve_source_image(manifest, shot, prev_clip, shot_dir):
    if shot.get("image"):
        return shot["image"]
    if shot.get("chain_from_prev"):
        if not prev_clip:
            _fail(f"shot {shot['id']}: chain_from_prev set but previous clip is missing")
        frame = shot_dir / "chain_frame.png"
        extract_last_frame(prev_clip, frame)
        return str(frame)
    if shot.get("image_prompt"):
        img_dir = shot_dir / "t2i"
        img_dir.mkdir(exist_ok=True)
        t2i = ["wait", "--prompt", shot["image_prompt"], "-n", "1", "--quiet", "-o", str(img_dir)]
        model = shot.get("image_model") or manifest.get("defaults", {}).get("image_model")
        if model:
            t2i += ["--model", model]
        rc, _, cost = _run_cli(t2i)
        image = _find_media(img_dir, IMAGE_EXTS)
        if rc != 0 or not image:
            _fail(f"shot {shot['id']}: t2i source image generation failed")
        shot["image"] = str(image)
        shot["buzz_spent"] = shot.get("buzz_spent", 0) + cost
        manifest["buzz_total"] = manifest.get("buzz_total", 0) + cost
        return str(image)
    return None


def cmd_run(args):
    if not args.confirm_spend:
        _fail("run submits paid work; re-run with --confirm-spend after user approval "
              "(use 'plan' first for a free total)")
    manifest = _load(args.name)
    prev_clip = None
    for shot in manifest["shots"]:
        shot_dir = _job_dir(args.name) / f"shot_{shot['id']:03d}"
        if shot["status"] == "clip_done":
            prev_clip = shot.get("clip") or prev_clip
            continue
        if args.only is not None and shot["id"] != args.only:
            print(f"shot {shot['id']}: skipped (--shot {args.only})")
            continue
        if args.keyframes_only:
            # only pure image_prompt shots — image/chain_from_prev take priority
            # in _resolve_source_image and chaining needs the previous clip
            if (shot.get("image_prompt") and not shot.get("image")
                    and not shot.get("chain_from_prev")):
                shot_dir.mkdir(parents=True, exist_ok=True)
                image = _resolve_source_image(manifest, shot, None, shot_dir)
                _save(args.name, manifest)
                print(f"shot {shot['id']}: keyframe -> {image}")
            else:
                print(f"shot {shot['id']}: no pending keyframe")
            continue
        shot_dir.mkdir(parents=True, exist_ok=True)
        source = _resolve_source_image(manifest, shot, prev_clip, shot_dir)
        _save(args.name, manifest)  # persist t2i result before the paid video step

        vid_args = ["wait", "--quiet", "-o", str(shot_dir)] + _video_args(manifest, shot, shot["prompt"])
        if source:
            vid_args += ["--source-image", source]
        print(f"shot {shot['id']}: submitting ({'i2v' if source else 't2v'}) ...")
        try:
            rc, out, cost = _run_cli(vid_args, timeout=1200)
        except subprocess.TimeoutExpired:
            shot["status"] = "failed"
            _save(args.name, manifest)
            _fail(f"shot {shot['id']}: video generation timed out; manifest saved, "
                  f"re-run to resume (the workflow may still finish server-side — "
                  f"check 'node {GEN_CLI} status' before resubmitting)")
        clip = _find_media(shot_dir, VIDEO_EXTS)
        if rc != 0 or not clip:
            shot["status"] = "failed"
            _save(args.name, manifest)
            _fail(f"shot {shot['id']}: video generation failed (rc={rc}); manifest saved, "
                  f"fix and re-run to resume")
        shot["clip"] = str(clip)
        shot["status"] = "clip_done"
        shot["buzz_spent"] = shot.get("buzz_spent", 0) + cost
        manifest["buzz_total"] = manifest.get("buzz_total", 0) + cost
        prev_clip = str(clip)
        _save(args.name, manifest)
        print(f"shot {shot['id']}: done -> {clip}")
    print("run complete; use 'concat' to stitch")


def extract_last_frame(video, out_png):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", str(video),
           "-frames:v", "1", "-update", "1", str(out_png)]
    subprocess.run(cmd, check=True)
    if not Path(out_png).exists():
        _fail(f"last-frame extraction produced nothing for {video}")
    return out_png


def cmd_frame(args):
    extract_last_frame(args.video, args.out)
    print(args.out)


def _probe_size(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def cmd_concat(args):
    manifest = _load(args.name)
    done = [s for s in manifest["shots"] if s.get("clip") and s["status"] == "clip_done"]
    if not done:
        _fail("no finished clips to concat")
    clips = [s["clip"] for s in done]
    job = _job_dir(args.name)
    width, height = _probe_size(clips[0])
    norm_dir = job / "_norm"
    norm_dir.mkdir(exist_ok=True)
    normed = []
    for i, (shot, clip) in enumerate(zip(done, clips)):
        out = norm_dir / f"{i:03d}.mp4"
        # chained shots start on the previous clip's last frame — drop that
        # duplicate frame so the cut does not stutter
        trim = ["-ss", "0.05"] if (i > 0 and shot.get("chain_from_prev")) else []
        # audio is dropped by default (engines differ; concat needs uniform streams);
        # --keep-audio re-encodes it and requires every clip to carry an audio track
        audio = ["-c:a", "aac"] if args.keep_audio else ["-an"]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *trim, "-i", clip,
             "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=24",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", *audio, str(out)],
            check=True)
        normed.append(out)
    list_file = norm_dir / "list.txt"
    list_file.write_text("".join(f"file '{p}'\n" for p in normed))
    final = job / (args.out or "final.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(final)],
        check=True)
    print(f"final: {final} ({len(clips)} clips, {width}x{height})")


def cmd_status(args):
    manifest = _load(args.name)
    for shot in manifest["shots"]:
        print(f"shot {shot['id']:>3}  {shot['status']:<10} clip={shot.get('clip', '-')}")
    done = sum(1 for s in manifest["shots"] if s["status"] == "clip_done")
    print(f"{done}/{len(manifest['shots'])} shots done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a job from a shots JSON")
    p.add_argument("name")
    p.add_argument("--shots", required=True, help="path to shots JSON")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("plan", help="free whatif cost per pending shot")
    p.add_argument("name")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="execute pending shots (paid; needs --confirm-spend)")
    p.add_argument("name")
    p.add_argument("--confirm-spend", action="store_true")
    p.add_argument("--shot", type=int, default=None, dest="only", help="run only this shot id")
    p.add_argument("--keyframes-only", action="store_true",
                   help="generate pending t2i keyframes only (cheap), stop before videos")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("frame", help="extract last frame of a video")
    p.add_argument("video")
    p.add_argument("out")
    p.set_defaults(fn=cmd_frame)

    p = sub.add_parser("concat", help="stitch finished clips into final.mp4")
    p.add_argument("name")
    p.add_argument("--out", default=None)
    p.add_argument("--keep-audio", action="store_true",
                   help="carry audio through (all clips must have an audio track)")
    p.set_defaults(fn=cmd_concat)

    p = sub.add_parser("status", help="print job summary")
    p.add_argument("name")
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
