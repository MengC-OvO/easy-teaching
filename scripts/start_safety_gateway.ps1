$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "EasyTeaching virtual environment not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Starting EasyTeaching Local Safety Gateway skeleton on 127.0.0.1:8010" -ForegroundColor Cyan
Write-Host "Health: http://127.0.0.1:8010/health" -ForegroundColor DarkGray
Write-Host "Readiness is expected to return 503 until the model pipeline is added." -ForegroundColor DarkYellow
& $Python -m uvicorn services.local_safety_gateway.api:app --host 127.0.0.1 --port 8010
if ($LASTEXITCODE -ne 0) {
    throw "Safety Gateway exited with code $LASTEXITCODE"
}
