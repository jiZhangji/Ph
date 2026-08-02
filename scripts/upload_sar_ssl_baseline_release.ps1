param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [string]$RepoId = "shimiandeshu/sar-ssl-paper-baseline-weights-v1",

    [string]$HfCommand = "hf"
)

$ErrorActionPreference = "Stop"
$env:HF_XET_HIGH_PERFORMANCE = "0"
$env:HF_HUB_DISABLE_XET = "1"

$package = (Resolve-Path -LiteralPath $PackageDir).Path
$required = @(
    "weights/mae/checkpoint-200.pth",
    "weights/lomar/checkpoint-200.pth",
    "weights/fg_mae/checkpoint-200.pth",
    "weights/i_jepa/jepa-ep200.pth.tar",
    "weights/sar_jepa/checkpoint-200.pth",
    "README.md",
    "manifest.json",
    "SHA256SUMS"
)

& $HfCommand auth whoami
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face authentication check failed"
}

foreach ($relative in $required) {
    $local = Join-Path $package $relative
    if (-not (Test-Path -LiteralPath $local -PathType Leaf)) {
        throw "Missing package file: $local"
    }

    $remote = $relative.Replace("\", "/")
    $size = (Get-Item -LiteralPath $local).Length
    Write-Host "UPLOAD_START path=$remote bytes=$size"
    & $HfCommand upload $RepoId --repo-type=model $local $remote
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $remote"
    }
    Write-Host "UPLOAD_DONE path=$remote bytes=$size"
}

Write-Host "UPLOAD_ALL_COMPLETE repo=https://huggingface.co/$RepoId"
