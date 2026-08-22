$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-safety\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "EasyTeaching Safety Gateway environment not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Starting EasyTeaching Local Safety Gateway on 127.0.0.1:8010" -ForegroundColor Cyan
Write-Host "Health: http://127.0.0.1:8010/health" -ForegroundColor DarkGray
Write-Host "Readiness becomes 200 only after local Qwen and the LoRA adapter load." -ForegroundColor DarkYellow
& $Python -m uvicorn safety_gateway.runtime:app --host 127.0.0.1 --port 8010 --workers 1
if ($LASTEXITCODE -ne 0) {
    throw "Safety Gateway exited with code $LASTEXITCODE"
}
