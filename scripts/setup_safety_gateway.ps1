param(
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [Parameter(Mandatory = $true)][string]$AdapterDir,
    [switch]$ForceConfig
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BootstrapPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SafetyPython = Join-Path $ProjectRoot ".venv-safety\Scripts\python.exe"
$ConfigPath = Join-Path $ProjectRoot ".safety-gateway.env"

$ResolvedModel = (Resolve-Path -LiteralPath $ModelDir).Path
$ResolvedAdapter = (Resolve-Path -LiteralPath $AdapterDir).Path
if (
    -not (Test-Path -LiteralPath (Join-Path $ResolvedModel "model.safetensors")) -and
    -not (Test-Path -LiteralPath (Join-Path $ResolvedModel "model.safetensors.index.json"))
) {
    throw "Qwen Safetensors weights were not found in: $ResolvedModel"
}
if (-not (Test-Path -LiteralPath (Join-Path $ResolvedAdapter "adapter_model.safetensors"))) {
    throw "LoRA adapter_model.safetensors was not found in: $ResolvedAdapter"
}
if ((Test-Path -LiteralPath $ConfigPath) -and -not $ForceConfig) {
    throw "Local config already exists. Use -ForceConfig to replace: $ConfigPath"
}
if (-not (Test-Path -LiteralPath $SafetyPython)) {
    if (-not (Test-Path -LiteralPath $BootstrapPython)) {
        throw "Create the main .venv first; bootstrap Python was not found: $BootstrapPython"
    }
    Write-Host "Creating independent .venv-safety (expect roughly 3-5 GB)..." -ForegroundColor Cyan
    & $BootstrapPython -m venv (Join-Path $ProjectRoot ".venv-safety")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-safety" }
}

Write-Host "Installing pinned Safety Gateway dependencies..." -ForegroundColor Cyan
& $SafetyPython -m pip install `
    -r (Join-Path $ProjectRoot "safety_gateway\requirements.txt") `
    -r (Join-Path $ProjectRoot "safety_gateway\requirements-model-cuda.txt")
if ($LASTEXITCODE -ne 0) { throw "Safety Gateway dependency installation failed" }

$ModelForEnv = $ResolvedModel.Replace("\", "/")
$AdapterForEnv = $ResolvedAdapter.Replace("\", "/")
$ConfigLines = @(
    "SAFETY_MODEL_BACKEND=auto",
    "SAFETY_MODEL_DIR=`"$ModelForEnv`"",
    "SAFETY_ADAPTER_DIR=`"$AdapterForEnv`"",
    "SAFETY_MAX_INPUT_TOKENS=1536",
    "SAFETY_MAX_NEW_TOKENS=320",
    "SAFETY_MAPPING_TTL_SECONDS=3600"
)
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllLines($ConfigPath, $ConfigLines, $Utf8NoBom)

Write-Host "Safety Gateway environment is ready." -ForegroundColor Green
Write-Host "Local config: $ConfigPath" -ForegroundColor DarkGray
Write-Host "Start with: .\scripts\start_safety_gateway.ps1" -ForegroundColor Yellow
