$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$edition = 'Desktop'
if ($PSVersionTable.PSObject.Properties.Name -contains 'PSEdition') {
    $edition = [string]$PSVersionTable.PSEdition
}

$apostrophe = "O'Reilly"
$dollarLiteral = '$notExpanded'
$jsonLiteral = '{"kind":"transport","value":"$notExpanded","quote":"\""}'
$jsonObject = $jsonLiteral | ConvertFrom-Json
$multiline = @'
line one
第二行
'@

$observation = [ordered]@{
    schema = 'loramake.remote.transport.v1'
    healthy = (
        $apostrophe -eq "O'Reilly" -and
        $dollarLiteral -eq '$notExpanded' -and
        $jsonObject.value -eq '$notExpanded' -and
        $multiline.Contains('第二行')
    )
    computer_name = [string]$env:COMPUTERNAME
    powershell_version = [string]$PSVersionTable.PSVersion
    powershell_edition = $edition
    remote_time_utc = (Get-Date).ToUniversalTime().ToString('o')
    encoding_sentinel = '中文 / テスト / ✓'
    apostrophe_sentinel = $apostrophe
    dollar_sentinel = $dollarLiteral
    json_literal_ok = ($jsonObject.value -eq '$notExpanded')
    multiline_ok = $multiline.Contains('第二行')
}

[Console]::Out.WriteLine(($observation | ConvertTo-Json -Depth 6 -Compress))
