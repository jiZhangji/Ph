param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [string]$RepoId = "shimiandeshu/sar-ssl-paper-baseline-weights-v1",

    [string]$HfCommand = "hf",

    [int]$RetryDelaySeconds = 60,

    [int]$MaxAttempts = 0
)

$ErrorActionPreference = "Continue"
$uploadScript = Join-Path $PSScriptRoot "upload_sar_ssl_baseline_release.ps1"

if (-not (Test-Path -LiteralPath $uploadScript -PathType Leaf)) {
    throw "Missing upload script: $uploadScript"
}

$attempt = 0
while ($true) {
    $attempt += 1
    $started = (Get-Date).ToString("o")
    Write-Host "UPLOAD_ATTEMPT_START attempt=$attempt time=$started"

    $succeeded = $false
    try {
        & $uploadScript `
            -PackageDir $PackageDir `
            -RepoId $RepoId `
            -HfCommand $HfCommand
        $succeeded = ($LASTEXITCODE -eq 0)
    }
    catch {
        Write-Host "UPLOAD_ATTEMPT_ERROR attempt=$attempt message=$($_.Exception.Message)"
    }

    if ($succeeded) {
        Write-Host "UPLOAD_RETRY_LOOP_COMPLETE attempts=$attempt"
        exit 0
    }

    if ($MaxAttempts -gt 0 -and $attempt -ge $MaxAttempts) {
        Write-Error "Upload did not complete after $attempt attempts"
        exit 1
    }

    Write-Host "UPLOAD_RETRY_WAIT attempt=$attempt seconds=$RetryDelaySeconds"
    Start-Sleep -Seconds $RetryDelaySeconds
}
