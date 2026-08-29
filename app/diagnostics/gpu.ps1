$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Convert-NullableInt {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text) -or $text -eq '[Not Supported]') {
        return $null
    }
    $number = 0.0
    if ([double]::TryParse(
            $text,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$number
        )) {
        return [int][math]::Round($number, 0)
    }
    return $null
}

function Convert-NullableNumber {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text) -or $text -eq '[Not Supported]') {
        return $null
    }
    $number = 0.0
    if ([double]::TryParse(
            $text,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$number
        )) {
        return [math]::Round($number, 2)
    }
    return $null
}

function Convert-ProcessName {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    # nvidia-smi may return a full executable path. Keep only the leaf name so
    # usernames and machine-local paths never enter the observation contract.
    return (($text -split '[\\/]') | Select-Object -Last 1)
}

function Get-CsvRows {
    param(
        [string]$Text,
        [string[]]$Headers
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @()
    }
    $lines = @(
        $Text -split "`r?`n" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($lines.Count -eq 0) {
        return @()
    }
    return @($lines | ConvertFrom-Csv -Header $Headers)
}

$collectedAt = [DateTime]::UtcNow.ToString('o')
$command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $command) {
    $observation = [ordered]@{
        schema = 'loramake.remote.gpu.v1'
        healthy = $false
        available = $false
        collected_at_utc = $collectedAt
        devices = @()
        compute_processes = @()
        warning = 'nvidia-smi is not available on the target'
        error = 'nvidia-smi command not found'
    }
    [Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 8 -Compress))
    exit 0
}

$warnings = @()
$devices = @()
$processes = @()
$errorMessage = $null
$gpuExitCode = $null
$queryGpu = 'index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,pstate'
$gpuHeaders = @(
    'index', 'name', 'driver_version', 'memory_total', 'memory_used', 'memory_free',
    'utilization_gpu', 'utilization_memory', 'temperature', 'power_draw', 'power_limit', 'pstate'
)

try {
    $gpuOutput = (& $command.Source "--query-gpu=$queryGpu" '--format=csv,noheader,nounits' 2>&1 | Out-String).Trim()
    $gpuExitCode = $LASTEXITCODE
    if ($gpuExitCode -ne 0) {
        throw "nvidia-smi GPU query exited with code ${gpuExitCode}"
    }

    $gpuRows = Get-CsvRows -Text $gpuOutput -Headers $gpuHeaders
    foreach ($row in $gpuRows) {
        $index = Convert-NullableInt $row.index
        if ($null -eq $index -or [string]::IsNullOrWhiteSpace([string]$row.name)) {
            throw 'nvidia-smi GPU query returned a malformed row'
        }
        $devices += [ordered]@{
            index = $index
            name = ([string]$row.name).Trim()
            driver_version = if ([string]::IsNullOrWhiteSpace([string]$row.driver_version)) { $null } else { ([string]$row.driver_version).Trim() }
            memory_total_mb = Convert-NullableInt $row.memory_total
            memory_used_mb = Convert-NullableInt $row.memory_used
            memory_free_mb = Convert-NullableInt $row.memory_free
            utilization_gpu_pct = Convert-NullableNumber $row.utilization_gpu
            utilization_memory_pct = Convert-NullableNumber $row.utilization_memory
            temperature_c = Convert-NullableNumber $row.temperature
            power_draw_w = Convert-NullableNumber $row.power_draw
            power_limit_w = Convert-NullableNumber $row.power_limit
            pstate = if ([string]::IsNullOrWhiteSpace([string]$row.pstate)) { $null } else { ([string]$row.pstate).Trim() }
        }
    }
    if ($devices.Count -eq 0) {
        $errorMessage = 'nvidia-smi GPU query returned no GPU rows'
        $warnings += $errorMessage
    }
}
catch {
    if ($null -ne $gpuExitCode -and $gpuExitCode -ne 0) {
        $errorMessage = "nvidia-smi GPU query exited with code ${gpuExitCode}"
    }
    else {
        $errorMessage = 'nvidia-smi GPU query returned malformed output'
    }
    $warnings += 'nvidia-smi GPU observation unavailable'
}

try {
    $processOutput = (& $command.Source '--query-compute-apps=pid,process_name,used_memory' '--format=csv,noheader,nounits' 2>&1 | Out-String).Trim()
    $processExitCode = $LASTEXITCODE
    $noProcessText = $processOutput -match '(?i)no running processes|no running processes found'
    if ($processExitCode -ne 0 -and -not $noProcessText) {
        $warnings += "nvidia-smi compute-process query exited with code $processExitCode"
    }
    if (-not $noProcessText) {
        $processHeaders = @('pid', 'process_name', 'used_memory')
        $processRows = Get-CsvRows -Text $processOutput -Headers $processHeaders
        foreach ($row in $processRows) {
            $processId = Convert-NullableInt $row.pid
            $processName = Convert-ProcessName $row.process_name
            if ($null -eq $processId -or $null -eq $processName) {
                $warnings += 'nvidia-smi compute-process query returned a malformed row'
                continue
            }
            $processes += [ordered]@{
                pid = $processId
                process_name = $processName
                used_memory_mb = Convert-NullableInt $row.used_memory
            }
        }
    }
}
catch {
    $warnings += 'nvidia-smi compute-process query failed'
}

$warningMessage = $null
if ($warnings.Count -gt 0) {
    $warningMessage = ($warnings -join '; ')
}
$available = ($null -eq $errorMessage -and $devices.Count -gt 0)
$observation = [ordered]@{
    schema = 'loramake.remote.gpu.v1'
    healthy = ($available -and $null -eq $warningMessage)
    available = [bool]$available
    collected_at_utc = $collectedAt
    devices = @($devices)
    compute_processes = @($processes)
    warning = $warningMessage
    error = $errorMessage
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 8 -Compress))
