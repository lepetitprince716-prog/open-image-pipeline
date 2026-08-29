$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Test-LocalTcpPort {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [int]$TimeoutMs = 2000
    )

    $client = New-Object System.Net.Sockets.TcpClient
    $waitHandle = $null
    try {
        $connect = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        $waitHandle = $connect.AsyncWaitHandle
        if (-not $waitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $client.EndConnect($connect)
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $waitHandle) {
            $waitHandle.Close()
        }
        $client.Close()
    }
}

$networkError = $null
$adapters = @()
try {
    $adapters = @(
        [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
            Where-Object {
                $_.OperationalStatus -eq [System.Net.NetworkInformation.OperationalStatus]::Up
            } |
            ForEach-Object {
                $properties = $_.GetIPProperties()
                [ordered]@{
                    alias = [string]$_.Name
                    description = [string]$_.Description
                    ipv4 = @(
                        $properties.UnicastAddresses |
                            Where-Object {
                                $_.Address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork
                            } |
                            ForEach-Object { [string]$_.Address }
                    )
                    gateways = @(
                        $properties.GatewayAddresses |
                            Where-Object {
                                $_.Address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork
                            } |
                            ForEach-Object { [string]$_.Address }
                    )
                    dns = @($properties.DnsAddresses | ForEach-Object { [string]$_ })
                }
            }
    )
}
catch {
    $networkError = [string]$_.Exception.Message
    $adapters = @(
        [ordered]@{
            alias = 'dns-fallback'
            ipv4 = @(
                [System.Net.Dns]::GetHostAddresses($env:COMPUTERNAME) |
                    Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } |
                    ForEach-Object { [string]$_ }
            )
            gateways = @()
            dns = @()
        }
    )
}

$sshListening = Test-LocalTcpPort -Port 22
$comfyListening = Test-LocalTcpPort -Port $ComfyPort

$observation = [ordered]@{
    schema = 'loramake.remote.network.v1'
    healthy = (
        $adapters.Count -gt 0 -and
        [bool]$sshListening -and
        [bool]$comfyListening -and
        $null -eq $networkError
    )
    computer_name = [string]$env:COMPUTERNAME
    adapters = $adapters
    ssh_port_22 = [bool]$sshListening
    comfyui_port = [int]$ComfyPort
    comfyui_listening = [bool]$comfyListening
    collection_error = $networkError
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 7 -Compress))
