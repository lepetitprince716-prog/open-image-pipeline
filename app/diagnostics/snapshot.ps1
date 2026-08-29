$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-JsonValue {
    param(
        [object]$Object,
        [string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Convert-NullableInt {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
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

function Convert-BytesToMb {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $number = 0.0
    if ([double]::TryParse(
            ([string]$Value),
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$number
        )) {
        return [int][math]::Round(($number / 1MB), 0)
    }
    return $null
}

function Convert-Text {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    return $text
}

function New-FallbackObservation {
    param(
        [string]$Warning,
        [bool]$FastfetchAvailable,
        [string]$FastfetchVersion
    )

    $os = Get-CimInstance -ClassName Win32_OperatingSystem -OperationTimeoutSec 8 -ErrorAction Stop
    $cpu = Get-CimInstance -ClassName Win32_Processor -OperationTimeoutSec 8 -ErrorAction Stop | Select-Object -First 1
    $gpus = @(
        Get-CimInstance -ClassName Win32_VideoController -OperationTimeoutSec 8 -ErrorAction SilentlyContinue |
            ForEach-Object {
                $name = Convert-Text (Get-JsonValue $_ 'Name')
                if ($null -ne $name) {
                    [ordered]@{
                        name = $name
                        vendor = Convert-Text (Get-JsonValue $_ 'AdapterCompatibility')
                        driver = Convert-Text (Get-JsonValue $_ 'DriverVersion')
                        type = $null
                    }
                }
            }
    )
    $disks = @(
        Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType=3" -OperationTimeoutSec 8 -ErrorAction SilentlyContinue |
            ForEach-Object {
                [ordered]@{
                    filesystem = Convert-Text (Get-JsonValue $_ 'FileSystem')
                    total_mb = Convert-BytesToMb (Get-JsonValue $_ 'Size')
                    used_mb = Convert-BytesToMb (([double](Get-JsonValue $_ 'Size')) - ([double](Get-JsonValue $_ 'FreeSpace')))
                    available_mb = Convert-BytesToMb (Get-JsonValue $_ 'FreeSpace')
                }
            }
    )

    return [ordered]@{
        schema = 'loramake.remote.snapshot.v1'
        healthy = $true
        available = $true
        collected_at_utc = [DateTime]::UtcNow.ToString('o')
        collector = 'powershell'
        fastfetch_available = $FastfetchAvailable
        fastfetch_version = $FastfetchVersion
        os_name = Convert-Text (Get-JsonValue $os 'Caption')
        os_version = Convert-Text (Get-JsonValue $os 'Version')
        os_build = Convert-Text (Get-JsonValue $os 'BuildNumber')
        kernel_name = 'Windows NT'
        kernel_release = Convert-Text (Get-JsonValue $os 'Version')
        kernel_architecture = Convert-Text (Get-JsonValue $os 'OSArchitecture')
        cpu_name = Convert-Text (Get-JsonValue $cpu 'Name')
        cpu_vendor = Convert-Text (Get-JsonValue $cpu 'Manufacturer')
        physical_cores = Convert-NullableInt (Get-JsonValue $cpu 'NumberOfCores')
        logical_processors = Convert-NullableInt (Get-JsonValue $cpu 'NumberOfLogicalProcessors')
        cpu_max_frequency_mhz = Convert-NullableInt (Get-JsonValue $cpu 'MaxClockSpeed')
        gpus = @($gpus)
        memory_total_mb = Convert-BytesToMb (([double](Get-JsonValue $os 'TotalVisibleMemorySize')) * 1KB)
        memory_used_mb = Convert-BytesToMb ((([double](Get-JsonValue $os 'TotalVisibleMemorySize')) - ([double](Get-JsonValue $os 'FreePhysicalMemory'))) * 1KB)
        disks = @($disks)
        packages_total = $null
        warning = $Warning
        error = $null
    }
}

$fastfetchCommand = Get-Command fastfetch -ErrorAction SilentlyContinue
if ($null -eq $fastfetchCommand) {
    $fastfetchCommand = Get-Command fastfetch.exe -ErrorAction SilentlyContinue
}
$fastfetchAvailable = ($null -ne $fastfetchCommand)
$fastfetchVersion = $null
$fastfetchWarnings = @()

if ($fastfetchAvailable) {
    try {
        $versionOutput = (& $fastfetchCommand.Source '--version' 2>&1 | Out-String).Trim()
        $versionExitCode = $LASTEXITCODE
        if ($versionExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($versionOutput)) {
            $fastfetchVersion = (($versionOutput -split "`r?`n" | Select-Object -First 1).Trim())
        }
        else {
            $fastfetchWarnings += 'Fastfetch version query failed'
        }

        $jsonOutput = (& $fastfetchCommand.Source '--json' 2>&1 | Out-String).Trim()
        $jsonExitCode = $LASTEXITCODE
        if ($jsonExitCode -ne 0) {
            throw "Fastfetch JSON query exited with code $jsonExitCode"
        }
        if ([string]::IsNullOrWhiteSpace($jsonOutput)) {
            throw 'Fastfetch JSON query returned empty output'
        }
        # InputObject keeps a top-level Fastfetch JSON array as an array. Piping the
        # string through ConvertFrom-Json on Windows PowerShell 5.1 boxes that array
        # as one object, which makes every section disappear from the allowlist map.
        $items = ConvertFrom-Json -InputObject $jsonOutput
        if ($items.Count -eq 0) {
            throw 'Fastfetch JSON query returned no sections'
        }

        $sections = @{}
        foreach ($item in $items) {
            $type = Convert-Text (Get-JsonValue $item 'type')
            if ($null -ne $type) {
                $sections[$type] = Get-JsonValue $item 'result'
            }
        }

        $os = $sections['OS']
        $kernel = $sections['Kernel']
        $cpu = $sections['CPU']
        $memory = $sections['Memory']
        $packages = $sections['Packages']
        $gpuSection = $sections['GPU']
        $diskSection = $sections['Disk']
        if ($null -eq $os -and $null -eq $cpu -and $null -eq $memory) {
            throw 'Fastfetch JSON query did not contain basic system sections'
        }

        $osBuild = Convert-Text (Get-JsonValue $os 'buildID')
        if ($null -eq $osBuild) {
            $osBuild = Convert-Text (Get-JsonValue $kernel 'release')
        }

        $cores = Get-JsonValue $cpu 'cores'
        $frequency = Get-JsonValue $cpu 'frequency'
        $gpus = @(
            foreach ($gpu in @($gpuSection)) {
                $name = Convert-Text (Get-JsonValue $gpu 'name')
                if ($null -ne $name) {
                    [ordered]@{
                        name = $name
                        vendor = Convert-Text (Get-JsonValue $gpu 'vendor')
                        driver = Convert-Text (Get-JsonValue $gpu 'driver')
                        type = Convert-Text (Get-JsonValue $gpu 'type')
                    }
                }
            }
        )
        $disks = @(
            foreach ($disk in @($diskSection)) {
                $bytes = Get-JsonValue $disk 'bytes'
                if ($null -ne $bytes) {
                    [ordered]@{
                        filesystem = Convert-Text (Get-JsonValue $disk 'filesystem')
                        total_mb = Convert-BytesToMb (Get-JsonValue $bytes 'total')
                        used_mb = Convert-BytesToMb (Get-JsonValue $bytes 'used')
                        available_mb = Convert-BytesToMb (Get-JsonValue $bytes 'available')
                    }
                }
            }
        )
        $fastfetchWarning = $null
        if ($fastfetchWarnings.Count -gt 0) {
            $fastfetchWarning = ($fastfetchWarnings -join '; ')
        }
        $observation = [ordered]@{
            schema = 'loramake.remote.snapshot.v1'
            healthy = $true
            available = $true
            collected_at_utc = [DateTime]::UtcNow.ToString('o')
            collector = 'fastfetch'
            fastfetch_available = $true
            fastfetch_version = $fastfetchVersion
            os_name = Convert-Text (Get-JsonValue $os 'prettyName')
            os_version = Convert-Text (Get-JsonValue $os 'version')
            os_build = $osBuild
            kernel_name = Convert-Text (Get-JsonValue $kernel 'name')
            kernel_release = Convert-Text (Get-JsonValue $kernel 'release')
            kernel_architecture = Convert-Text (Get-JsonValue $kernel 'architecture')
            cpu_name = Convert-Text (Get-JsonValue $cpu 'cpu')
            cpu_vendor = Convert-Text (Get-JsonValue $cpu 'vendor')
            physical_cores = Convert-NullableInt (Get-JsonValue $cores 'physical')
            logical_processors = Convert-NullableInt (Get-JsonValue $cores 'logical')
            cpu_max_frequency_mhz = Convert-NullableInt (Get-JsonValue $frequency 'max')
            gpus = @($gpus)
            memory_total_mb = Convert-BytesToMb (Get-JsonValue $memory 'total')
            memory_used_mb = Convert-BytesToMb (Get-JsonValue $memory 'used')
            disks = @($disks)
            packages_total = Convert-NullableInt (Get-JsonValue $packages 'all')
            warning = $fastfetchWarning
            error = $null
        }
        [Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 8 -Compress))
        exit 0
    }
    catch {
        $fastfetchWarnings += 'Fastfetch JSON collection failed; PowerShell/CIM fallback used'
    }
}
else {
    $fastfetchWarnings += 'Fastfetch is not available; PowerShell/CIM fallback used'
}

$fallbackWarning = $fastfetchWarnings -join '; '
try {
    $fallback = New-FallbackObservation -Warning $fallbackWarning -FastfetchAvailable $fastfetchAvailable -FastfetchVersion $fastfetchVersion
    [Console]::Out.WriteLine(($fallback | ConvertTo-Json -Depth 8 -Compress))
}
catch {
    $observation = [ordered]@{
        schema = 'loramake.remote.snapshot.v1'
        healthy = $false
        available = $false
        collected_at_utc = [DateTime]::UtcNow.ToString('o')
        collector = 'powershell'
        fastfetch_available = [bool]$fastfetchAvailable
        fastfetch_version = $fastfetchVersion
        os_name = $null
        os_version = $null
        os_build = $null
        kernel_name = $null
        kernel_release = $null
        kernel_architecture = $null
        cpu_name = $null
        cpu_vendor = $null
        physical_cores = $null
        logical_processors = $null
        cpu_max_frequency_mhz = $null
        gpus = @()
        memory_total_mb = $null
        memory_used_mb = $null
        disks = @()
        packages_total = $null
        warning = $fallbackWarning
        error = 'PowerShell/CIM fallback failed'
    }
    [Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 8 -Compress))
}
