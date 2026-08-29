$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$baseUri = 'http://127.0.0.1:' + [string]$ComfyPort
$reachable = $false
$errorMessage = $null
$version = $null
$devices = @()
$queueRunning = $null
$queuePending = $null

try {
    $stats = Invoke-RestMethod -Uri ($baseUri + '/system_stats') -Method Get -TimeoutSec 6
    $queue = Invoke-RestMethod -Uri ($baseUri + '/queue') -Method Get -TimeoutSec 6
    $reachable = $true
    $version = [string]$stats.system.comfyui_version
    $devices = @(
        $stats.devices | ForEach-Object {
            [ordered]@{
                name = [string]$_.name
                type = [string]$_.type
                vram_total_mb = [math]::Round(([double]$_.vram_total / 1MB), 0)
                vram_free_mb = [math]::Round(([double]$_.vram_free / 1MB), 0)
            }
        }
    )
    $queueRunning = @($queue.queue_running).Count
    $queuePending = @($queue.queue_pending).Count
}
catch {
    $errorMessage = [string]$_.Exception.Message
}

$versionMatches = $true
if ($null -ne $ExpectedComfyVersion) {
    $versionMatches = ($version -eq [string]$ExpectedComfyVersion)
}

$observation = [ordered]@{
    schema = 'loramake.remote.comfyui.v1'
    healthy = ($reachable -and $versionMatches)
    reachable = $reachable
    base_uri = $baseUri
    version = $version
    expected_version = $ExpectedComfyVersion
    version_matches = $versionMatches
    queue_running = $queueRunning
    queue_pending = $queuePending
    devices = $devices
    error = $errorMessage
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 7 -Compress))
