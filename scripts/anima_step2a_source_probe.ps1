$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Test-Head {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url
    )

    $request = $null
    $response = $null
    $result = [ordered]@{
        name = $Name
        url = $Url
        status_code = $null
        final_origin = $null
        content_length = $null
        content_type = $null
        error = $null
    }
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Method = 'HEAD'
        $request.AllowAutoRedirect = $true
        $request.MaximumAutomaticRedirections = 5
        $request.Timeout = 15000
        $request.ReadWriteTimeout = 15000
        $request.UserAgent = 'loramake-step2a-preflight/1.0'
        $response = [System.Net.HttpWebResponse]$request.GetResponse()
    }
    catch {
        $result.error = $_.Exception.Message
        $responseException = $_.Exception
        while (-not $responseException.Response -and $responseException.InnerException) {
            $responseException = $responseException.InnerException
        }
        if ($responseException.Response) {
            $response = [System.Net.HttpWebResponse]$responseException.Response
        }
    }
    finally {
        if ($response) {
            $result.status_code = [int]$response.StatusCode
            $finalUri = $response.ResponseUri
            $result.final_origin = '{0}://{1}{2}' -f $finalUri.Scheme, $finalUri.Host, $finalUri.AbsolutePath
            $result.content_length = [long]$response.ContentLength
            $result.content_type = $response.ContentType
            $response.Dispose()
        }
        if ($request) { $request.Abort() }
    }
    [pscustomobject]$result
}

$network = @(
    (Test-Head -Name 'modelscope_home' -Url 'https://www.modelscope.cn/'),
    (Test-Head -Name 'modelscope_search_api_candidate' -Url 'https://www.modelscope.cn/api/v1/datasets?PageNumber=1&PageSize=20&Search=danbooru2024'),
    (Test-Head -Name 'hf_origin_dataset_api' -Url 'https://huggingface.co/api/datasets/deepghs/danbooru2024-webp-4Mpixel'),
    (Test-Head -Name 'deepghs_readme_public' -Url 'https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel/resolve/main/README.md'),
    (Test-Head -Name 'deepghs_metadata_without_credentials' -Url 'https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel/resolve/main/metadata.parquet'),
    (Test-Head -Name 'p1atdev_metadata_shard_public' -Url 'https://huggingface.co/datasets/p1atdev/danbooru-2024/resolve/main/data/train-00000-of-00035.parquet')
)

$modelscopeReachable = ($network | Where-Object { $_.name -eq 'modelscope_home' }).status_code -eq 200
$hfApiReachable = ($network | Where-Object { $_.name -eq 'hf_origin_dataset_api' }).status_code -eq 200
$gatedStatus = ($network | Where-Object { $_.name -eq 'deepghs_metadata_without_credentials' }).status_code
$publicMetadataStatus = ($network | Where-Object { $_.name -eq 'p1atdev_metadata_shard_public' }).status_code
$sourceRoute = [pscustomobject]@{
    modelscope_site_reachable = $modelscopeReachable
    modelscope_webp_mirror_confirmed = $false
    hf_origin_reachable = $hfApiReachable
    deepghs_repo_is_gated = ($gatedStatus -eq 401)
    public_metadata_fallback_reachable = ($publicMetadataStatus -eq 200)
    status = if ($hfApiReachable -and $publicMetadataStatus -eq 200 -and $gatedStatus -eq 401) {
        'HOLD_GATED_DATASET_ACCESS_AND_USER_COMPLIANCE'
    }
    else {
        'HOLD_SOURCE_ROUTE_UNRESOLVED'
    }
}

[pscustomobject]@{
    schema = 'loramake.anima_step2a.source_probe.v1'
    captured_utc = (Get-Date).ToUniversalTime().ToString('o')
    hostname = $env:COMPUTERNAME
    network = $network
    source_route = $sourceRoute
} | ConvertTo-Json -Depth 8

# The source gate deliberately remains nonzero until the target dataset access
# and user-approved compliance policy are both in place.
exit 2
