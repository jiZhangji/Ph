param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [string]$RepoId = "shimian123/sar-ssl-paper-baseline-weights-v1",

    [string]$ModelScopeCommand = "modelscope",

    [int]$MaxWorkers = 5,

    [int]$RetryDelaySeconds = 60,

    [int]$MaxAttempts = 0
)

$ErrorActionPreference = "Continue"
$package = (Resolve-Path -LiteralPath $PackageDir).Path

& $ModelScopeCommand create $RepoId `
    --repo_type model `
    --visibility private `
    --exist_ok
if ($LASTEXITCODE -ne 0) {
    throw "Unable to create or access ModelScope repository: $RepoId"
}

$attempt = 0
while ($true) {
    $attempt += 1
    $started = (Get-Date).ToString("o")
    Write-Host "MODELSCOPE_UPLOAD_ATTEMPT_START attempt=$attempt time=$started"

    & $ModelScopeCommand upload $RepoId $package `
        --repo-type model `
        --max-workers $MaxWorkers `
        --commit-message "Upload paper baseline weights"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "MODELSCOPE_UPLOAD_COMPLETE attempts=$attempt"
        Write-Host "repo=https://www.modelscope.cn/models/$RepoId"
        exit 0
    }

    if ($MaxAttempts -gt 0 -and $attempt -ge $MaxAttempts) {
        Write-Error "ModelScope upload did not complete after $attempt attempts"
        exit 1
    }

    Write-Host "MODELSCOPE_UPLOAD_RETRY_WAIT attempt=$attempt seconds=$RetryDelaySeconds"
    Start-Sleep -Seconds $RetryDelaySeconds
}
