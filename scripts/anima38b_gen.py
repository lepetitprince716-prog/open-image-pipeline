#!/usr/bin/env python3
"""Anima 3.8B channel: Mac PLAN / BOXA in-process ComfyUI 0.33.1 renderer.

Default invocation prints the recipe and the exact ``--run`` command. Nothing
touches BOXA until ``--run``. Does not use ComfyUI :8188.

Usage:
    python3 scripts/anima38b_gen.py --prompt 'year 2025,\\nDescription:\\nA fox-girl...'
    python3 scripts/anima38b_gen.py --prompt '...' --run
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, by_model_dir  # noqa: E402

WORKER = Path(__file__).resolve().parent / "anima38b_worker.py"
FUSED_MODULE = Path(__file__).resolve().parent / "anima38b_fused.py"
RESIDENT_MODULE = Path(__file__).resolve().parent / "anima38b_resident.py"
RDBG_MANIFEST = ROOT / "tools" / "rdbg" / "Cargo.toml"
# Under a managed network policy the ssh/scp binaries are wrappers and the host may be
# a tunnel alias; rdbg's own -o hardening has to stand down for them (tools/rdbg/README).
SSH_HOST = os.environ.get("ANIMA38B_SSH_HOST", "boxa")
RDBG_SSH = os.environ.get("RDBG_SSH")
SCP_BIN = os.environ.get("ANIMA38B_SCP", "scp")
# The managed scp wrapper accepts no -o options, so its BatchMode stands down with rdbg's.
SCP_OPTS = [] if RDBG_SSH else ["-o", "BatchMode=yes"]
REMOTE_DIR = os.environ.get("ANIMA38B_DIR", "/opt/anima38b")
REMOTE_WORKER = REMOTE_DIR + r"\run.py"
REMOTE_JOB = REMOTE_DIR + r"\job.json"
REMOTE_PNG = REMOTE_DIR + r"\last.png"
REMOTE_FUSED = REMOTE_DIR + r"\anima38b_fused.py"
REMOTE_TMP = REMOTE_DIR + r"\tmp"
REMOTE_PYTHON = "/opt/anima38b/.venv/Scripts/python.exe"
# --resident: one long-lived worker per root (anima38b_resident.py); see plan_text for the trade-offs.
REMOTE_RESIDENT = REMOTE_DIR + r"\resident"
REMOTE_RESIDENT_MODULE = REMOTE_DIR + r"\anima38b_resident.py"
RESIDENT_IDLE_S = 180
RESIDENT_READY_S = 300
DEPS_PYTHONPATH = "/opt/pydeps-anima33"
# --fused: triton-windows lives in this overlay, not in the worker's venv. It finds its bundled cuda.lib and
# TinyCC through sysconfig platlib, which misses an overlay, hence CUDA_PATH and CC; its cache and
# temp files would default to C:.
TRITON_OVERLAY = "/opt/pydeps-anima-exp"
FUSED_ENV = {
    "CUDA_PATH": TRITON_OVERLAY + r"\triton\backends\nvidia",
    "CC": TRITON_OVERLAY + r"\triton\runtime\tcc\tcc.exe",
    "TRITON_CACHE_DIR": REMOTE_DIR + r"\triton_cache",
    "TEMP": REMOTE_TMP,
    "TMP": REMOTE_TMP,
}

DEFAULT_STEPS = 28
DEFAULT_CFG = 7.0
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1216
DEFAULT_NEGATIVE = (
    "worst quality, low quality, bad hands, extra digits, missing fingers, "
    "jpeg artifacts, watermark, signature"
)
HOLD_VRAM_MIB = 4000


def ps_quote(value: str) -> str:
    """Quote ``value`` for PowerShell 5.1 paste (single quotes; inner ``'`` doubled)."""
    return "'" + str(value).replace("'", "''") + "'"


def _slug_model(value: str) -> str:
    slug = value.strip().lower()
    if not slug or any(ch in slug for ch in "/\\") or slug in {".", ".."}:
        raise argparse.ArgumentTypeError("model must be a short slug")
    return slug


def _dest_dir(model: str) -> Path:
    return by_model_dir("anima", "anima38b", model=model)


def _unique_png(directory: Path, seed: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"boxa8b_s{seed}"
    path = directory / f"{stem}.png"
    n = 1
    while path.exists():
        path = directory / f"{stem}__v{n}.png"
        n += 1
    return path


# Optional job fields for a non-3.8B checkpoint on the same worker: text="native" encodes with
# the Qwen3 0.6B CLIP alone (28/40-block Anima), sampler/scheduler replace res_multistep/beta,
# shift adds ModelSamplingAuraFlow.
SAMPLING_KEYS = ("text", "sampler", "scheduler", "shift")


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text", choices=["native"], default=None,
        help="native: Qwen3 0.6B CLIP only, for 28/40-block Anima checkpoints (default: 3.8B adapter)",
    )
    parser.add_argument("--sampler", default=None, help="sampler_name (default res_multistep)")
    parser.add_argument("--scheduler", default=None, help="scheduler (default beta)")
    parser.add_argument("--shift", type=float, default=None, help="ModelSamplingAuraFlow shift (default none)")


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora", action="append", default=None,
        help="BOXA path of a LoRA .safetensors to load onto the DiT (52-block remapped); repeatable",
    )
    parser.add_argument(
        "--lora-strength", type=float, action="append", default=None,
        help="strength per --lora, in order; a single value applies to all (default 1.0)",
    )


def build_job(args: argparse.Namespace) -> dict:
    """The job dict shipped to BOXA. base_ckpt lands in the sidecar, which the taste
    ledger already reads, so an A/B keeps its provenance with no ledger changes."""
    unet = getattr(args, "unet", None)
    job = {
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "width": args.width,
        "height": args.height,
        "adapter_strength": args.adapter_strength,
        "unet": unet,
        "base_ckpt": unet or "Anima-3.8B.safetensors",
        "out": REMOTE_PNG,
    }
    if getattr(args, "attention", None):
        job["attention"] = args.attention
    for key in SAMPLING_KEYS:
        if getattr(args, key, None) is not None:
            job[key] = getattr(args, key)
    loras = lora_specs(args)
    if len(loras) == 1:
        job["lora"] = loras[0]
    elif loras:
        job["loras"] = loras
    if getattr(args, "fused", False):
        job["fused"] = True
    return job


def lora_specs(args: argparse.Namespace) -> list[dict]:
    """--lora / --lora-strength pairs in order. One strength applies to every LoRA; otherwise the
    counts must match. A bare string or float (the single-LoRA shape) is accepted as a list of one."""
    paths = getattr(args, "lora", None) or []
    if isinstance(paths, str):
        paths = [paths]
    strengths = getattr(args, "lora_strength", None)
    if strengths is None:
        strengths = [1.0]
    elif not isinstance(strengths, list):
        strengths = [strengths]
    if len(strengths) == 1:
        strengths = strengths * len(paths)
    if paths and len(strengths) != len(paths):
        raise SystemExit(f"--lora-strength: got {len(strengths)} values for {len(paths)} --lora")
    return [{"path": p, "strength": float(s)} for p, s in zip(paths, strengths, strict=False)]


def plan_text(args: argparse.Namespace, dest: Path) -> str:
    unet = getattr(args, "unet", None)
    # --unet must survive into the copy-pasteable command: this channel's contract is
    # "print the plan, then run the printed line", and dropping it here would silently
    # render bf16 weights into the other arm's directory. Values are PowerShell 5.1
    # single-quoted (inner ' -> '') so a prompt like l'artiste pastes onto BOXA.
    run_cmd = (
        f"python3 scripts/anima38b_gen.py --prompt {ps_quote(args.prompt)} "
        f"--negative {ps_quote(args.negative)} --seed {args.seed} --steps {args.steps} "
        f"--cfg {args.cfg} --width {args.width} --height {args.height} "
        + (f"--unet {ps_quote(unet)} " if unet else "")
        + (f"--attention {args.attention} " if getattr(args, "attention", None) else "")
        + "".join(
            f"--{key} {getattr(args, key)} " for key in SAMPLING_KEYS if getattr(args, key, None) is not None
        )
        + "".join(
            f"--lora {ps_quote(spec['path'])} --lora-strength {spec['strength']} "
            for spec in lora_specs(args)
        )
        + ("--fused " if getattr(args, "fused", False) else "")
        + ("--resident " if getattr(args, "resident", False) else "")
        + (
            f"--idle-s {args.idle_s} "
            if getattr(args, "resident", False) and getattr(args, "idle_s", RESIDENT_IDLE_S) != RESIDENT_IDLE_S
            else ""
        )
        + f"--model {ps_quote(args.model)} --run"
    )
    return "\n".join(
        [
            "PLAN anima38b  (no GPU until --run)",
            f"  box          BOXA RTX 5080  (in-process ComfyUI 0.33.1, not :8188)",
            f"  unet         {unet or 'Anima-3.8B.safetensors (bf16)'}",
            f"  unet dtype   bfloat16   (fp8 is forbidden)",
            f"  sampler      res_multistep + beta  cfg={args.cfg}  steps={args.steps}",
            f"  size         {args.width}x{args.height}  seed={args.seed}",
            f"  adapter      {args.adapter_strength}",
            f"  attention    {getattr(args, 'attention', None) or '(default)'}",
            "  fused        on (Triton adaln/GELU -> MXFP8; needs MXFP8 --unet)"
            if getattr(args, "fused", False)
            else "  fused        off",
            (
                f"  resident     on: one worker stays loaded on BOXA (~12 GiB VRAM, text encoders in RAM) "
                f"until {getattr(args, 'idle_s', RESIDENT_IDLE_S)} s idle; while it lives the VRAM HOLD "
                "cannot see other GPU users; stop it with --resident-stop"
            )
            if getattr(args, "resident", False)
            else "  resident     off",
            "  lora         "
            + (", ".join(f"{s['path']} @ {s['strength']}" for s in lora_specs(args)) or "(none)"),
            f"  lora skipped (listed as lora_skipped_blocks after --run)"
            if getattr(args, "lora", None)
            else "  lora skipped (none)",
            f"  elapsed      ~50-60s cold (28 steps); sample ~1.5 it/s",
            f"  dest         {dest}",
            f"  prompt       {args.prompt!r}",
            "",
            "Hard stops: do not run while a game owns the 5080; do not pass fp8;",
            "do not submit this to BOXA-ComfyUI-8188.",
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
        ]
        + (["--ssh", RDBG_SSH, "--no-ssh-opts"] if RDBG_SSH else [])
        + [
            "--timeout",
            str(timeout),
            "ps",
            SSH_HOST,
            "--file",
            str(ps1),
        ],
        check=False,
        text=True,
        # Windows messages on BOXA arrive in the console code page (GBK); strict UTF-8 would
        # raise here and hide the worker's own error.
        errors="replace",
        capture_output=True,
    )


def _ps1_fused_lines(fused_b64: str | None) -> str:
    """With --fused: triton environment plus anima38b_fused.py written next to run.py."""
    if not fused_b64:
        return ""
    env = "".join(f"$env:{key} = '{value}'\n" for key, value in FUSED_ENV.items())
    return (
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_TMP}' | Out-Null\n"
        + env
        + f"[IO.File]::WriteAllBytes('{REMOTE_FUSED}', [Convert]::FromBase64String('{fused_b64}'))\n"
    )


def _ps1_write_and_run(worker_b64: str, job_b64: str, fused_b64: str | None = None) -> str:
    pythonpath = f"{TRITON_OVERLAY};{DEPS_PYTHONPATH}" if fused_b64 else DEPS_PYTHONPATH
    return f"""$ErrorActionPreference = 'Continue'
foreach ($v in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')) {{
  Set-Item -Path ("env:$v") -Value '' -ErrorAction SilentlyContinue
}}
$env:NO_PROXY = '*'
$env:PYTHONPATH = '{pythonpath}'
New-Item -ItemType Directory -Force -Path '{REMOTE_DIR}' | Out-Null
{_ps1_fused_lines(fused_b64)}[IO.File]::WriteAllBytes('{REMOTE_WORKER}', [Convert]::FromBase64String('{worker_b64}'))
[IO.File]::WriteAllBytes('{REMOTE_JOB}', [Convert]::FromBase64String('{job_b64}'))
$smi = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1
Write-Output ("GPU " + (($smi | Out-String).Trim()))
$py = '{REMOTE_PYTHON}'
$raw = & $py -u '{REMOTE_WORKER}' --job '{REMOTE_JOB}'
$text = ($raw | Out-String)
Write-Output $text
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
if (-not (Test-Path '{REMOTE_PNG}')) {{ Write-Output 'NO_PNG'; exit 4 }}
$item = Get-Item '{REMOTE_PNG}'
Write-Output ("PNG_BYTES " + $item.Length)
"""


def resident_identity(job: dict, code: dict[str, bytes]) -> tuple[dict, str]:
    """What a resident server was loaded with. Any difference, including new code, needs a fresh process
    (attention binds at import time; a running server keeps its old code in memory)."""
    identity = {
        "unet": job.get("unet") or "default",
        "attention": job.get("attention"),
        "lora": job.get("lora"),
        "loras": job.get("loras"),
        "text": job.get("text"),
        "fused": bool(job.get("fused")),
        "code": {name: hashlib.sha256(data).hexdigest()[:16] for name, data in sorted(code.items())},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return identity, key


def build_serve_config(job: dict, identity: dict, key: str, idle_s: int) -> dict:
    """serve.json for anima38b_resident.serve. The WMI-started server inherits none of the payload's
    environment, so everything it needs travels here."""
    env = {"TEMP": REMOTE_TMP, "TMP": REMOTE_TMP}
    sys_path = []
    if job.get("fused"):
        env.update(FUSED_ENV)
        sys_path.append(TRITON_OVERLAY)
    return {
        "root": REMOTE_RESIDENT, "identity": identity, "identity_key": key, "unet": job.get("unet"),
        "attention": job.get("attention"), "lora": job.get("lora"), "loras": job.get("loras"), "text": job.get("text"),
        "fused": bool(job.get("fused")), "idle_s": int(idle_s), "env": env, "sys_path": sys_path,
    }


_PS1_RESIDENT_HELPERS = r"""
function Read-Heartbeat([string]$path) {
  for ($i = 0; $i -lt 20; $i++) {
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try { return ([IO.File]::ReadAllText($path, [Text.Encoding]::UTF8) | ConvertFrom-Json) } catch { Start-Sleep -Milliseconds 100 }
  }
  return $null
}
function Test-Server($hb) {
  if ($null -eq $hb) { return $false }
  $proc = Get-CimInstance Win32_Process -Filter ("ProcessId=" + [int]$hb.pid) -ErrorAction SilentlyContinue
  return ($null -ne $proc -and $proc.CommandLine -match 'run\.py' -and $proc.CommandLine -match '--serve')
}
function Show-Log([string]$root) {
  $log = Join-Path $root 'server.log'
  if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 30 }
}
"""

_PS1_RESIDENT = r"""$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$root = '%%ROOT%%'
$queue = Join-Path $root 'queue'
$out = Join-Path $root 'out'
$hbPath = Join-Path $root 'server.json'
$deadline = (Get-Date).AddSeconds(%%DEADLINE_S%%)
New-Item -ItemType Directory -Force -Path $queue, $out, '%%TMP%%' | Out-Null
Get-ChildItem -LiteralPath $out -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } |
  Remove-Item -Force -ErrorAction SilentlyContinue
%%HELPERS%%
function Start-Server {
  $cmd = '"%%PYTHON%%" -u "%%WORKER%%" --serve "' + (Join-Path $root 'serve.json') + '"'
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = '%%DIR%%' }
  return $r.ReturnValue
}
function Wait-Ready([string]$hbPath) {
  while ((Get-Date) -lt $deadline) {
    $hb = Read-Heartbeat $hbPath
    if ($hb -and ($hb.state -eq 'ready' -or $hb.state -eq 'busy') -and $hb.identity_key -eq '%%KEY%%' -and (Test-Server $hb)) { return $true }
    Start-Sleep -Milliseconds 500
  }
  return $false
}
$hb = Read-Heartbeat $hbPath
$alive = Test-Server $hb
if ($alive -and $hb.identity_key -ne '%%KEY%%') {
  Write-Output ('RESIDENT_RESTART identity ' + $hb.identity_key + ' -> %%KEY%%')
  Stop-Process -Id ([int]$hb.pid) -Force
  Start-Sleep -Seconds 2
  $alive = $false
}
$smi = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1
Write-Output ("GPU " + (($smi | Out-String).Trim()))
if ($alive) {
  Write-Output ('RESIDENT_REUSE pid=' + $hb.pid)
} else {
  $used = -1
  try { $used = [int](($smi | Select-Object -First 1).ToString().Split(',')[0].Trim()) } catch { $used = -1 }
  if (%%HOLD_CHECK%% -and ($used -lt 0 -or $used -ge %%HOLD%%)) {
    Write-Output "HOLD: BOXA VRAM $used MiB (limit %%HOLD%%) and no resident server of ours. Pass --force to override."
    exit 3
  }
%%WRITES%%  $rv = Start-Server
  Write-Output "RESIDENT_START return=$rv"
  if (-not (Wait-Ready $hbPath)) { Write-Output 'RESIDENT_START_TIMEOUT'; Show-Log $root; exit 5 }
}
$jobTmp = Join-Path $queue '%%JOB_ID%%.json.tmp'
$jobPath = Join-Path $queue '%%JOB_ID%%.json'
[IO.File]::WriteAllBytes($jobTmp, [Convert]::FromBase64String('%%JOB_B64%%'))
Move-Item -LiteralPath $jobTmp -Destination $jobPath -Force
$resPath = Join-Path $out '%%JOB_ID%%.json'
$restarts = 0
while (-not (Test-Path -LiteralPath $resPath)) {
  if ((Get-Date) -gt $deadline) {
    $hb = Read-Heartbeat $hbPath
    if ($hb -and $hb.current_job -eq '%%JOB_ID%%' -and (Test-Server $hb)) {
      Stop-Process -Id ([int]$hb.pid) -Force
      Write-Output 'RESIDENT_KILLED the server was still on this job'
    }
    Remove-Item -LiteralPath $jobPath -Force -ErrorAction SilentlyContinue
    Write-Output 'RESIDENT_JOB_TIMEOUT'; Show-Log $root; exit 6
  }
  if ((Test-Path -LiteralPath $jobPath) -and -not (Test-Server (Read-Heartbeat $hbPath))) {
    Start-Sleep -Seconds 1
    if ((Test-Path -LiteralPath $jobPath) -and -not (Test-Server (Read-Heartbeat $hbPath))) {
      if ($restarts -ge 2) { Write-Output 'RESIDENT_DIED'; Show-Log $root; exit 7 }
      $restarts++
      $rv = Start-Server
      Write-Output "RESIDENT_START return=$rv (job still queued)"
      if (-not (Wait-Ready $hbPath)) { Write-Output 'RESIDENT_START_TIMEOUT'; Show-Log $root; exit 5 }
    }
  }
  Start-Sleep -Milliseconds 500
}
$res = [IO.File]::ReadAllText($resPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
foreach ($line in $res.lines) { Write-Output $line }
Remove-Item -LiteralPath $resPath -Force -ErrorAction SilentlyContinue
if (-not $res.ok) { Write-Output ('RESIDENT_ERROR ' + $res.error); Show-Log $root; exit 8 }
Write-Output ('RESIDENT_PNG ' + $res.png)
Write-Output ('PNG_BYTES ' + (Get-Item -LiteralPath $res.png).Length)
"""

_PS1_RESIDENT_STOP = r"""$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$root = '%%ROOT%%'
$hbPath = Join-Path $root 'server.json'
%%HELPERS%%
$hb = Read-Heartbeat $hbPath
if (Test-Server $hb) {
  Stop-Process -Id ([int]$hb.pid) -Force
  Write-Output ('RESIDENT_STOPPED pid=' + $hb.pid)
} else {
  Write-Output 'RESIDENT_NOT_RUNNING'
}
Remove-Item -LiteralPath $hbPath -Force -ErrorAction SilentlyContinue
"""


def _fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f"%%{key}%%", value)
    if "%%" in template:
        raise ValueError("unfilled placeholder in the resident PowerShell payload")
    return template


def _ps1_resident(serve: dict, job: dict, job_id: str, deadline_s: int, force: bool) -> str:
    """Reuse or start the resident server, queue one job, wait for its result. The code files are pushed
    by push_files beforehand, so this payload stays a few KB."""
    serve_b64 = base64.b64encode(json.dumps(serve, ensure_ascii=False).encode("utf-8")).decode("ascii")
    writes = f"  [IO.File]::WriteAllBytes((Join-Path $root 'serve.json'), [Convert]::FromBase64String('{serve_b64}'))\n"
    job_b64 = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return _fill(_PS1_RESIDENT, {
        "HELPERS": _PS1_RESIDENT_HELPERS, "WRITES": writes, "ROOT": REMOTE_RESIDENT, "TMP": REMOTE_TMP,
        "PYTHON": REMOTE_PYTHON, "WORKER": REMOTE_WORKER, "DIR": REMOTE_DIR, "KEY": serve["identity_key"],
        "JOB_ID": job_id, "JOB_B64": job_b64, "DEADLINE_S": str(int(deadline_s)), "HOLD": str(HOLD_VRAM_MIB),
        "HOLD_CHECK": "$false" if force else "$true",
    })


def _ps1_resident_stop() -> str:
    return _fill(_PS1_RESIDENT_STOP, {"HELPERS": _PS1_RESIDENT_HELPERS, "ROOT": REMOTE_RESIDENT})


def resident_job_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)


def apply_worker_stdout(job: dict, stdout: str) -> None:
    """Copy timing and LoRA coverage lines from the worker into the job sidecar."""
    timing = re.search(r"SAMPLE_S (\S+) STEPS (\d+) PEAK_GIB (\S+)", stdout)
    if timing:
        job["timing"] = {
            "sample_s": float(timing.group(1)),
            "steps": int(timing.group(2)),
            "peak_gib": float(timing.group(3)),
        }
    fused = re.search(r"FUSED_STATS blocks=(\d+) hits=(\d+) misses=(\d+) fallbacks=(\d+)", stdout)
    if fused:
        # fallbacks > 0 means some block calls ran the original forward (and gained nothing).
        job["fused_stats"] = dict(
            zip(("blocks", "hits", "misses", "fallbacks"), (int(g) for g in fused.groups()), strict=True)
        )
    resident = re.search(
        r"^RESIDENT job=(\S+) total_s=(\S+) encode_s=(\S+) encode_cached=(\d) dit_load_s=(\S+) vae_s=(\S+)",
        stdout, re.M,
    )
    if resident:
        job["resident"] = {
            "job_id": resident.group(1), "total_s": float(resident.group(2)),
            "encode_s": float(resident.group(3)), "encode_cached": resident.group(4) == "1",
            "dit_load_s": float(resident.group(5)), "vae_s": float(resident.group(6)),
        }
    skipped = re.search(r"^LORA_SKIPPED_BLOCKS[ \t]*(.*)$", stdout, re.M)
    if skipped is None:
        return
    text = skipped.group(1).strip()
    job["lora_skipped_blocks"] = (
        [int(tok) for tok in text.split(",") if tok.strip()] if text else []
    )


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


def resident_files(job: dict) -> dict[str, bytes]:
    """Code the resident server runs, keyed by its BOXA path."""
    files = {REMOTE_WORKER: WORKER.read_bytes(), REMOTE_RESIDENT_MODULE: RESIDENT_MODULE.read_bytes()}
    if job.get("fused"):
        files[REMOTE_FUSED] = FUSED_MODULE.read_bytes()
    return files


def push_files(files: dict[str, bytes]) -> int:
    """Copy files (all under REMOTE_DIR) to BOXA in one scp call. rdbg sends its script on ssh stdin, and scripts of ~60 KB and up
    stalled there intermittently on 2026-10-01 (outputs/_experiments/anima38b_speed/FINDINGS.md section 14),
    so large files travel by scp and the PowerShell payload stays small."""
    with tempfile.TemporaryDirectory() as tmp:
        local = []
        for remote, data in files.items():
            path = Path(tmp) / remote.rsplit("\\", 1)[-1]
            path.write_bytes(data)
            local.append(str(path))
        dest = f"{SSH_HOST}:{REMOTE_DIR.replace(chr(92), '/')}/"
        return subprocess.run([SCP_BIN, "-q", *SCP_OPTS, *local, dest], check=False).returncode


def resident_payload(args: argparse.Namespace, job: dict, files: dict[str, bytes]) -> str:
    """PowerShell for --resident; adds identity_key to job (it lands in the sidecar too). `files` is what
    push_files sends; it only enters the identity here."""
    identity, key = resident_identity(job, {path.rsplit("\\", 1)[-1]: data for path, data in files.items()})
    job["identity_key"] = key
    serve = build_serve_config(job, identity, key, getattr(args, "idle_s", RESIDENT_IDLE_S))
    # Leave the rdbg call some slack after the payload's own deadline so its error output still arrives.
    return _ps1_resident(serve, job, resident_job_id(), max(60, args.timeout - 20), args.force)


def _hold_on_busy_gpu() -> int | None:
    used, util = _probe_vram()
    print(f"BOXA GPU used={used} MiB util={util}%")
    if used < 0:
        print("HOLD: nvidia-smi unreadable", file=sys.stderr)
        return 3
    if used >= HOLD_VRAM_MIB:
        print(
            f"HOLD: BOXA VRAM {used} MiB >= {HOLD_VRAM_MIB} (game/other job?). "
            "Pass --force to override.",
            file=sys.stderr,
        )
        return 3
    return None


def run(args: argparse.Namespace, dest: Path) -> int:
    resident = getattr(args, "resident", False)
    # --resident checks VRAM inside its payload: its own live server must not read as "another job".
    if not args.force and not resident:
        held = _hold_on_busy_gpu()
        if held is not None:
            return held
    job = build_job(args)
    if resident:
        files = resident_files(job)
        body = resident_payload(args, job, files)
        if push_files(files) != 0:
            print("HOLD: failed to push the resident worker code to BOXA", file=sys.stderr)
            return 4
    else:
        worker_b64 = base64.b64encode(WORKER.read_bytes()).decode("ascii")
        job_b64 = base64.b64encode(
            json.dumps(job, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        fused_b64 = (
            base64.b64encode(FUSED_MODULE.read_bytes()).decode("ascii") if job.get("fused") else None
        )
        body = _ps1_write_and_run(worker_b64, job_b64, fused_b64)
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=True, encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        proc = _rdbg(Path(handle.name), timeout=args.timeout)
    sys.stdout.write(proc.stdout or "")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode or 1
    stdout = proc.stdout or ""
    apply_worker_stdout(job, stdout)
    remote_png = REMOTE_PNG
    if resident:
        found = re.search(r"^RESIDENT_PNG (.+?)\s*$", stdout, re.M)
        if found is None:
            print("HOLD: the resident worker reported no PNG", file=sys.stderr)
            return 4
        remote_png = found.group(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pull = subprocess.run(
        [SCP_BIN, "-q", *SCP_OPTS, f"{SSH_HOST}:{remote_png.replace(chr(92), '/')}", str(dest)],
        check=False,
    )
    if pull.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1024:
        print("HOLD: failed to pull PNG from BOXA", file=sys.stderr)
        return 4
    sidecar = dest.with_suffix(".png.meta.json")
    sidecar.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {dest} {dest.stat().st_size}B")
    if "lora_skipped_blocks" in job:
        print(f"lora_skipped_blocks {job['lora_skipped_blocks']}")
    return 0


def stop_resident(timeout: int) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=True, encoding="utf-8") as handle:
        handle.write(_ps1_resident_stop())
        handle.flush()
        proc = _rdbg(Path(handle.name), timeout=timeout)
    sys.stdout.write(proc.stdout or "")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="required unless --resident-stop")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--adapter-strength", type=float, default=1.0)
    add_sampling_args(parser)
    parser.add_argument(
        "--unet", default=None,
        help="BOXA path of the DiT checkpoint (default: the bf16 Anima-3.8B)",
    )
    parser.add_argument(
        "--attention", default=None,
        help="attention backend for the worker, e.g. ck_int8 or sage (validated on BOXA)",
    )
    add_lora_args(parser)
    parser.add_argument(
        "--fused", action="store_true",
        help="opt-in fused Triton adaln/GELU -> MXFP8 kernels (needs an MXFP8 --unet; "
        "sampling -18%% measured on BOXA, 2026-09-30)",
    )
    parser.add_argument(
        "--resident", action="store_true",
        help="render through a worker that stays loaded on BOXA between calls (exits after --idle-s idle)",
    )
    parser.add_argument("--idle-s", type=int, default=RESIDENT_IDLE_S, help="resident worker idle exit, seconds")
    parser.add_argument(
        "--resident-stop", action="store_true",
        help="stop the resident worker on BOXA now (acts immediately; no prompt needed)",
    )
    parser.add_argument("--model", type=_slug_model, default=os.environ.get("DRIVING_MODEL", "claude"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore BOXA VRAM HOLD")
    parser.add_argument("--timeout", type=int, default=300, help="rdbg seconds")
    args = parser.parse_args()
    if args.resident_stop:
        return stop_resident(timeout=60)
    if not args.prompt:
        parser.error("--prompt is required")
    dest = _unique_png(_dest_dir(args.model), args.seed)
    print(plan_text(args, dest))
    if not args.run:
        return 0
    return run(args, dest)


if __name__ == "__main__":
    raise SystemExit(main())
