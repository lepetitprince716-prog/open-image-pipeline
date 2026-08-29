#!/usr/bin/env python3
"""Anima 3.8B channel: Mac PLAN / Z3 in-process ComfyUI 0.33.1 renderer.

Default invocation prints the recipe and the exact ``--run`` command. Nothing
touches Z3 until ``--run``. Does not use ComfyUI :8188.

Usage:
    python3 scripts/anima38b_gen.py --prompt 'year 2025,\\nDescription:\\nA fox-girl...'
    python3 scripts/anima38b_gen.py --prompt '...' --run
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, by_model_dir  # noqa: E402

WORKER = Path(__file__).resolve().parent / "anima38b_worker.py"
RDBG_MANIFEST = ROOT / "tools" / "rdbg" / "Cargo.toml"
SSH_HOST = "z3"
REMOTE_DIR = r"E:\anima38b"
REMOTE_WORKER = REMOTE_DIR + r"\run.py"
REMOTE_JOB = REMOTE_DIR + r"\job.json"
REMOTE_PNG = REMOTE_DIR + r"\last.png"

DEFAULT_STEPS = 28
DEFAULT_CFG = 7.0
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1216
DEFAULT_NEGATIVE = (
    "worst quality, low quality, bad hands, extra digits, missing fingers, "
    "jpeg artifacts, watermark, signature"
)
HOLD_VRAM_MIB = 4000


def _slug_model(value: str) -> str:
    slug = value.strip().lower()
    if not slug or any(ch in slug for ch in "/\\") or slug in {".", ".."}:
        raise argparse.ArgumentTypeError("model must be a short slug")
    return slug


def _dest_dir(model: str) -> Path:
    return by_model_dir("anima", "anima38b", model=model)


def _unique_png(directory: Path, seed: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"z38b_s{seed}"
    path = directory / f"{stem}.png"
    n = 1
    while path.exists():
        path = directory / f"{stem}__v{n}.png"
        n += 1
    return path


def plan_text(args: argparse.Namespace, dest: Path) -> str:
    run_cmd = (
        f"python3 scripts/anima38b_gen.py --prompt {args.prompt!r} "
        f"--negative {args.negative!r} --seed {args.seed} --steps {args.steps} "
        f"--cfg {args.cfg} --width {args.width} --height {args.height} "
        f"--model {args.model} --run"
    )
    return "\n".join(
        [
            "PLAN anima38b  (no GPU until --run)",
            f"  box          Z3 RTX 5080  (in-process ComfyUI 0.33.1, not :8188)",
            f"  unet dtype   bfloat16   (fp8 is forbidden)",
            f"  sampler      res_multistep + beta  cfg={args.cfg}  steps={args.steps}",
            f"  size         {args.width}x{args.height}  seed={args.seed}",
            f"  adapter      {args.adapter_strength}",
            f"  elapsed      ~50-60s cold (28 steps); sample ~1.5 it/s",
            f"  dest         {dest}",
            f"  prompt       {args.prompt!r}",
            "",
            "Hard stops: do not run while a game owns the 5080; do not pass fp8;",
            "do not submit this to Z3-ComfyUI-8188.",
            "",
            run_cmd,
        ]
    )


def _rdbg(ps1: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(RDBG_MANIFEST),
            "--",
            "--timeout",
            str(timeout),
            "ps",
            SSH_HOST,
            "--file",
            str(ps1),
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def _ps1_write_and_run(worker_b64: str, job_b64: str) -> str:
    return f"""$ErrorActionPreference = 'Continue'
foreach ($v in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')) {{
  Set-Item -Path ("env:$v") -Value '' -ErrorAction SilentlyContinue
}}
$env:NO_PROXY = '*'
$env:PYTHONPATH = 'E:\\pydeps-anima33'
New-Item -ItemType Directory -Force -Path '{REMOTE_DIR}' | Out-Null
[IO.File]::WriteAllBytes('{REMOTE_WORKER}', [Convert]::FromBase64String('{worker_b64}'))
[IO.File]::WriteAllBytes('{REMOTE_JOB}', [Convert]::FromBase64String('{job_b64}'))
$smi = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1
Write-Output ("GPU " + (($smi | Out-String).Trim()))
$py = 'E:\\.venv\\Scripts\\python.exe'
$raw = & $py -u '{REMOTE_WORKER}' --job '{REMOTE_JOB}'
$text = ($raw | Out-String)
Write-Output $text
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
if (-not (Test-Path '{REMOTE_PNG}')) {{ Write-Output 'NO_PNG'; exit 4 }}
$item = Get-Item '{REMOTE_PNG}'
Write-Output ("PNG_BYTES " + $item.Length)
"""


def _probe_vram() -> tuple[int, int]:
    ps = """$ErrorActionPreference = 'Continue'
$smi = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
Write-Output (($smi | Out-String).Trim())
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=True, encoding="utf-8") as handle:
        handle.write(ps)
        handle.flush()
        proc = _rdbg(Path(handle.name), timeout=40)
    line = (proc.stdout or "").strip().splitlines()
    last = line[-1] if line else ""
    parts = [p.strip() for p in last.replace(",", " ").split() if p.strip()]
    used = int(float(parts[0])) if parts else -1
    util = int(float(parts[1])) if len(parts) > 1 else -1
    return used, util


def run(args: argparse.Namespace, dest: Path) -> int:
    if not args.force:
        used, util = _probe_vram()
        print(f"Z3 GPU used={used} MiB util={util}%")
        if used < 0:
            print("HOLD: nvidia-smi unreadable", file=sys.stderr)
            return 3
        if used >= HOLD_VRAM_MIB:
            print(
                f"HOLD: Z3 VRAM {used} MiB >= {HOLD_VRAM_MIB} (game/other job?). "
                "Pass --force to override.",
                file=sys.stderr,
            )
            return 3
    job = {
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "width": args.width,
        "height": args.height,
        "adapter_strength": args.adapter_strength,
        "out": REMOTE_PNG,
    }
    worker_b64 = base64.b64encode(WORKER.read_bytes()).decode("ascii")
    job_b64 = base64.b64encode(
        json.dumps(job, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    body = _ps1_write_and_run(worker_b64, job_b64)
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=True, encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        proc = _rdbg(Path(handle.name), timeout=args.timeout)
    sys.stdout.write(proc.stdout or "")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode or 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    pull = subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", f"{SSH_HOST}:{REMOTE_PNG}", str(dest)],
        check=False,
    )
    if pull.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1024:
        print("HOLD: failed to pull PNG from Z3", file=sys.stderr)
        return 4
    sidecar = dest.with_suffix(".png.meta.json")
    sidecar.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {dest} {dest.stat().st_size}B")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--adapter-strength", type=float, default=1.0)
    parser.add_argument("--model", type=_slug_model, default=os.environ.get("DRIVING_MODEL", "claude"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore Z3 VRAM HOLD")
    parser.add_argument("--timeout", type=int, default=300, help="rdbg seconds")
    args = parser.parse_args()
    dest = _unique_png(_dest_dir(args.model), args.seed)
    print(plan_text(args, dest))
    if not args.run:
        return 0
    return run(args, dest)


if __name__ == "__main__":
    raise SystemExit(main())
