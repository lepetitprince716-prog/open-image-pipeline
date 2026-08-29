#!/usr/bin/env python3
"""Manage the Forge (SD WebUI Forge) API service on Z3 from the Mac.

The app-facing sibling of scripts/z3_lora_manager.py: the Studio console's
/api/forge endpoint subprocesses this CLI, and it works standalone from a
terminal too.

  status    HTTP probe through the 17862 loopback (reachable + loaded checkpoint)
  start     launch the service on Z3 over SSH (Win32_Process.Create — survives
            the SSH session closing; Start-Process children die with the
            session's job object, verified 2026-08-28)
  stop      kill the :7862 listener on Z3 (scoped to the port owner)
  refresh   POST /refresh-checkpoints + /refresh-loras (after junctioning models)
  unload    POST /unload-checkpoint (release VRAM, service stays up)

SSH uses the `z3` alias first, then `z3-ts` (same fallback contract as
z3_lora_manager). The PowerShell payload is scripts/forge_z3_control.ps1 with
the action prefixed, shipped quote-safe via -EncodedCommand. Exit codes:
0 ok · 1 action failed · 2 bad usage.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401  (bootstraps repo root on sys.path)
from loramake import forge  # noqa: E402
from loramake.forge_endpoints import resolve_forge_host  # noqa: E402

CONTROL_PS1 = Path(__file__).resolve().parent / "forge_z3_control.ps1"
SSH_BOXES = os.environ.get("Z3_SSH_BOXES", "z3,z3-ts").split(",")
START_READY_TIMEOUT_S = float(os.environ.get("FORGE_START_TIMEOUT", "180"))
ACTIONS = ("status", "start", "stop", "refresh", "unload")


def ssh_ps(script: str, timeout: int = 90) -> tuple[int, str]:
    """Run a PowerShell script on Z3 via -EncodedCommand (quote-safe), LAN then Tailscale."""
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    remote = f"powershell -NoProfile -NonInteractive -EncodedCommand {enc}"
    last: tuple[int, str] = (255, "")
    for box in SSH_BOXES:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", box.strip(), remote],
            capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        out = (proc.stdout + proc.stderr).replace("\r", "").strip()
        # ssh's own connect failure is 255; any other rc means the box was reached.
        if proc.returncode != 255:
            return proc.returncode, out
        last = (proc.returncode, out)
    return last


def _control_script(action: str) -> str:
    return f'$Action = "{action}"\n' + CONTROL_PS1.read_text(encoding="utf-8")


def status(host: str) -> int:
    ok, detail = forge.ping(host)
    checkpoint = None
    if ok:
        try:
            checkpoint = forge.options(host).get("sd_model_checkpoint")
        except forge.ForgeError:
            pass
    print(json.dumps({"reachable": ok, "url": host, "checkpoint": checkpoint,
                      "detail": detail}, ensure_ascii=False))
    return 0 if ok else 1


def start(host: str) -> int:
    ok, _ = forge.ping(host)
    if ok:
        print(f"already running -> {host}")
        return 0
    rc, out = ssh_ps(_control_script("Start"), timeout=120)
    print(out)
    if rc != 0 and "already listening" not in out:
        return 1
    deadline = time.time() + START_READY_TIMEOUT_S
    while time.time() < deadline:
        ok, detail = forge.ping(host)
        if ok:
            print(f"ready -> {detail}")
            return 0
        time.sleep(5)
    print(f"timed out after {START_READY_TIMEOUT_S:g}s waiting for {host} "
          f"(service may still be warming; check D:\\Forge\\forge\\forge_api.log)",
          file=sys.stderr)
    return 1


def stop(host: str) -> int:
    rc, out = ssh_ps(_control_script("Stop"), timeout=90)
    print(out)
    return 0 if rc == 0 else 1


def refresh(host: str) -> int:
    ok, detail = forge.ping(host)
    if not ok:
        print(detail, file=sys.stderr)
        return 1
    forge.refresh_models(host)
    print(f"refreshed checkpoint + LoRA enums on {host} "
          "(newly junctioned checkpoints additionally need a service restart)")
    return 0


def unload(host: str) -> int:
    ok, detail = forge.ping(host)
    if not ok:
        print(detail, file=sys.stderr)
        return 1
    released = forge.unload_checkpoint(host)
    print("checkpoint unloaded (VRAM released)" if released else "unload-checkpoint failed")
    return 0 if released else 1


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ACTIONS:
        print(f"usage: forge_service.py {{{'|'.join(ACTIONS)}}}", file=sys.stderr)
        return 2
    host = resolve_forge_host("z3")
    return {"status": status, "start": start, "stop": stop,
            "refresh": refresh, "unload": unload}[sys.argv[1]](host)


if __name__ == "__main__":
    raise SystemExit(main())
