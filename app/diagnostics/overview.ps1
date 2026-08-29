$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$os = Get-CimInstance -ClassName Win32_OperatingSystem -OperationTimeoutSec 8 -ErrorAction Stop
$cpu = Get-CimInstance -ClassName Win32_Processor -OperationTimeoutSec 8 -ErrorAction Stop | Select-Object -First 1
$serviceNames = @('sshd')
if ($null -ne $ExpectedService) {
    $serviceNames += [string]$ExpectedService
}
$serviceNames = @($serviceNames | Select-Object -Unique)

$services = @(
    foreach ($serviceName in $serviceNames) {
        try {
            $service = Get-Service -Name $serviceName -ErrorAction Stop
            [pscustomobject][ordered]@{
                name = [string]$service.Name
                status = [string]$service.Status
                present = $true
                running = ([string]$service.Status -eq 'Running')
            }
        }
        catch {
            [pscustomobject][ordered]@{
                name = [string]$serviceName
                status = 'Missing'
                present = $false
                running = $false
            }
        }
    }
)
$unhealthyServices = @(
    $services | Where-Object { $_.present -ne $true -or $_.running -ne $true }
)

$observation = [ordered]@{
    schema = 'loramake.remote.overview.v1'
    healthy = ($unhealthyServices.Count -eq 0)
    computer_name = [string]$env:COMPUTERNAME
    powershell_version = [string]$PSVersionTable.PSVersion
    os_caption = [string]$os.Caption
    os_version = [string]$os.Version
    os_build = [string]$os.BuildNumber
    last_boot_utc = $os.LastBootUpTime.ToUniversalTime().ToString('o')
    uptime_seconds = [math]::Round(((Get-Date) - $os.LastBootUpTime).TotalSeconds)
    memory_total_mb = [math]::Round(([double]$os.TotalVisibleMemorySize / 1024), 0)
    memory_free_mb = [math]::Round(([double]$os.FreePhysicalMemory / 1024), 0)
    cpu_name = [string]$cpu.Name
    logical_processors = [int]$cpu.NumberOfLogicalProcessors
    services = $services
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 6 -Compress))
