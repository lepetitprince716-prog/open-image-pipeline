$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-ProcessLineage {
    param(
        [Parameter(Mandatory = $true)]
        [int]$OwnerPid
    )

    $currentPid = $OwnerPid
    $depth = 0
    $seen = @{}
    while ($currentPid -gt 0 -and $depth -lt 64) {
        $pidKey = [string]$currentPid
        if ($seen.ContainsKey($pidKey)) {
            break
        }
        $seen[$pidKey] = $true

        $proc = $null
        $queryError = $null
        try {
            $proc = Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId=" + $pidKey) -OperationTimeoutSec 5 -ErrorAction Stop
        }
        catch {
            $queryError = [string]$_.Exception.Message
        }

        if ($null -eq $proc) {
            [pscustomobject][ordered]@{
                depth = $depth
                pid = [int]$currentPid
                name = $null
                executable = $null
                parent_pid = $null
                found = $false
                query_error = $queryError
            }
            break
        }

        [pscustomobject][ordered]@{
            depth = $depth
            pid = [int]$proc.ProcessId
            name = [string]$proc.Name
            executable = [string]$proc.ExecutablePath
            parent_pid = [int]$proc.ParentProcessId
            found = $true
            query_error = $null
        }

        $currentPid = [int]$proc.ParentProcessId
        $depth += 1
    }
}

$listenerError = $null
$ownerLineages = @()
try {
    # Enumerate the stable result set first, then filter locally. A port with no
    # owner is therefore an empty array, independent of localized CIM messages.
    $listeners = @(
        Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object {
                [string]$_.State -eq 'Listen' -and [int]$_.LocalPort -eq [int]$ComfyPort
            }
    )
    $ownerPids = @($listeners | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
    $ownerLineages = @(
        foreach ($ownerPid in $ownerPids) {
            [pscustomobject][ordered]@{
                owner_pid = [int]$ownerPid
                lineage = @(Get-ProcessLineage -OwnerPid ([int]$ownerPid))
            }
        }
    )
}
catch {
    $listenerError = [string]$_.Exception.Message
}

$service = $null
$servicePid = 0
$serviceCollectionError = $null
if ($null -ne $ExpectedService) {
    try {
        # ExpectedService is a fixed server-owned identifier. The exact CIM
        # filter avoids a slow full Win32_Service enumeration on Z3.
        $serviceFilter = "Name='" + [string]$ExpectedService + "'"
        $services = @(
            Get-CimInstance -ClassName Win32_Service -Filter $serviceFilter -OperationTimeoutSec 5 -ErrorAction Stop
        )
        $svc = $services | Select-Object -First 1
        if ($null -eq $svc) {
            $service = [pscustomobject][ordered]@{
                name = [string]$ExpectedService
                status = 'Missing'
                present = $false
                process_id = 0
            }
        }
        else {
            $servicePid = [int]$svc.ProcessId
            $service = [pscustomobject][ordered]@{
                name = [string]$svc.Name
                status = [string]$svc.State
                present = $true
                process_id = $servicePid
            }
        }
    }
    catch {
        $serviceCollectionError = [string]$_.Exception.Message
        $service = [pscustomobject][ordered]@{
            name = [string]$ExpectedService
            status = 'Unknown'
            present = $false
            process_id = 0
        }
    }
}

$processes = @(
    foreach ($ownerRecord in $ownerLineages) {
        $lineage = @($ownerRecord.lineage)
        $owner = $lineage | Select-Object -First 1
        $serviceBound = $null
        $canonicalInLineage = $null
        if ($null -ne $ExpectedService) {
            $lineagePids = @($lineage | ForEach-Object { [int]$_.pid })
            $serviceBound = ($servicePid -gt 0 -and $lineagePids -contains $servicePid)
            $canonicalInLineage = @(
                $lineage | Where-Object {
                    $null -ne $CanonicalPython -and
                    [string]::Equals(
                        [string]$_.executable,
                        [string]$CanonicalPython,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )
                }
            ).Count -gt 0
        }

        [pscustomobject][ordered]@{
            pid = [int]$ownerRecord.owner_pid
            name = if ($null -ne $owner) { [string]$owner.name } else { $null }
            executable = if ($null -ne $owner) { [string]$owner.executable } else { $null }
            parent_pid = if ($null -ne $owner) { $owner.parent_pid } else { $null }
            service_bound = $serviceBound
            canonical_python_in_lineage = $canonicalInLineage
            lineage = $lineage
        }
    }
)

$listening = ($processes.Count -gt 0)
$allOwnersServiceBound = ($listening -and @($processes | Where-Object { $_.service_bound -ne $true }).Count -eq 0)
$allOwnersCanonical = ($listening -and @($processes | Where-Object { $_.canonical_python_in_lineage -ne $true }).Count -eq 0)
$serviceRunning = (
    $null -ne $ExpectedService -and
    $null -ne $service -and
    $service.present -eq $true -and
    [string]$service.status -eq 'Running'
)

$verdict = 'listener-present'
if (-not $listening) {
    # An empty port is always missing; service state only classifies a real owner.
    $verdict = 'missing'
}
elseif ($null -ne $ExpectedService) {
    if (-not $serviceRunning) {
        # A stopped, missing, or unreadable required service makes any owner rogue.
        $verdict = 'rogue'
    }
    elseif ($allOwnersServiceBound -and $allOwnersCanonical) {
        $verdict = 'canonical'
    }
    else {
        # Every owner must satisfy both checks; mixed canonical/rogue owners stay rogue.
        $verdict = 'rogue'
    }
}

$observation = [ordered]@{
    schema = 'loramake.remote.listener.v1'
    healthy = ($listening -and $verdict -ne 'rogue' -and $null -eq $listenerError -and $null -eq $serviceCollectionError)
    port = [int]$ComfyPort
    listening = $listening
    verdict = $verdict
    service_pid = $servicePid
    all_owners_service_bound = $allOwnersServiceBound
    all_owners_canonical_python = $allOwnersCanonical
    processes = $processes
    expected_service = $ExpectedService
    canonical_python = $CanonicalPython
    service = $service
    collection_error = $listenerError
    service_collection_error = $serviceCollectionError
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 12 -Compress))
