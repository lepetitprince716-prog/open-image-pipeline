# Control the Forge (SD WebUI Forge) API service on Z3.
#
# rdbg ships one script file and has no argv channel, so the action is a plain
# variable: run the file as-is for Status, or compose a temporary caller that
# prefixes `$Action = "Start"` (or "Stop") to this file's content and ship that
# (see docs/forge_line.md; contract: .claude/rules/remote-powershell.md):
#   cargo run --quiet --locked --manifest-path tools/rdbg/Cargo.toml -- \
#     --ssh /usr/local/bin/claude-ssh --no-ssh-opts ps z3 --file scripts/forge_z3_control.ps1
# Deployment facts (docs/forge_line.md): D:\Forge\forge, commit dfdcbab6, port 7862,
# --nowebui pure API, --xformers mandatory for repro fidelity.

if (-not (Get-Variable -Name Action -ErrorAction SilentlyContinue)) {
    $Action = "Status"
}
if (@("Status", "Start", "Stop") -notcontains $Action) {
    Write-Output "unknown action: $Action (expected Status, Start, or Stop)"
    exit 64
}

$ErrorActionPreference = "Stop"
$forgeRoot = "D:\Forge\forge"
$python = Join-Path $forgeRoot "venv\Scripts\python.exe"
$port = 7862

function Get-ForgeListener {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
}

switch ($Action) {
    "Status" {
        $listener = Get-ForgeListener
        if (-not $listener) {
            Write-Output "STATUS: not listening on port $port"
            exit 1
        }
        $ownerId = ($listener | Select-Object -First 1).OwningProcess
        try {
            $options = Invoke-RestMethod -Uri "http://127.0.0.1:$port/sdapi/v1/options" -TimeoutSec 10
            $checkpoint = $options.sd_model_checkpoint
            Write-Output "STATUS: listening pid=$ownerId checkpoint=$checkpoint"
        }
        catch {
            Write-Output "STATUS: listening pid=$ownerId but API not answering: $($_.Exception.Message)"
            exit 1
        }
    }
    "Start" {
        if (Get-ForgeListener) {
            Write-Output "START: already listening on port $port"
            exit 0
        }
        if (-not (Test-Path $python)) {
            Write-Output "START: missing $python"
            exit 1
        }
        $log = Join-Path $forgeRoot "forge_api.log"
        # A process spawned by SSH-hosted PowerShell lives inside that session's job
        # object, and Windows OpenSSH kills the whole tree on disconnect — Start-Process
        # children die the instant the ssh session returns (verified 2026-08-28; same
        # lesson as scripts/z3_lora_manager.py). Win32_Process.Create spawns OUTSIDE the
        # caller's job, so the server survives the session closing. cmd.exe /c wraps the
        # command to capture combined output; doubled outer quotes are cmd.exe's rule.
        $cmd = 'cmd.exe /c ""' + $python + '" launch.py --nowebui --port ' + $port +
            ' --skip-python-version-check --skip-prepare-environment --xformers > "' + $log + '" 2>&1"'
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
            -Arguments @{ CommandLine = $cmd; CurrentDirectory = $forgeRoot }
        if ($r.ReturnValue -ne 0) {
            Write-Output ("START: Win32_Process.Create failed rc=" + $r.ReturnValue)
            exit 1
        }
        Write-Output ("START: launched pid=" + $r.ProcessId + "; log -> " + $log + "; warmup roughly 10-60 seconds")
    }
    "Stop" {
        $listener = Get-ForgeListener
        if (-not $listener) {
            Write-Output "STOP: nothing listening on port $port"
            exit 0
        }
        foreach ($connection in $listener) {
            Stop-Process -Id $connection.OwningProcess -Force
        }
        Write-Output "STOP: killed listener process(es) on port $port"
    }
}
