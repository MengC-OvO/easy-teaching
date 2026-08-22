$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SafetyPython = Join-Path $ProjectRoot ".venv-safety\Scripts\python.exe"
$AppPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DemoModule = Join-Path $ProjectRoot "scripts\_privacy_flow_demo.py"
$GatewayUrl = "http://127.0.0.1:8010"
$StartedGateway = $false
$GatewayProcess = $null

if (-not (Test-Path -LiteralPath $SafetyPython)) {
    throw "Missing Safety Gateway environment: $SafetyPython"
}
if (-not (Test-Path -LiteralPath $AppPython)) {
    throw "Missing EasyTeaching environment: $AppPython"
}
if (-not (Test-Path -LiteralPath $DemoModule)) {
    throw "Missing privacy-flow demo module: $DemoModule"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".safety-gateway.env"))) {
    throw "Missing .safety-gateway.env. Run scripts\setup_safety_gateway.ps1 first."
}

function Test-GatewayHealth {
    try {
        $null = Invoke-RestMethod -Uri "$GatewayUrl/health" -TimeoutSec 2
        return $true
    }
    catch {
        return $false
    }
}

function Test-GatewayReady {
    try {
        $result = Invoke-RestMethod -Uri "$GatewayUrl/ready" -TimeoutSec 3
        return ($result.ready -eq $true -and $result.model_loaded -eq $true)
    }
    catch {
        return $false
    }
}

try {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " EasyTeaching full local privacy-flow smoke test" -ForegroundColor Cyan
    Write-Host " Real Qwen v11: YES | External LLM: NO | Real data: NO" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    if (Test-GatewayHealth) {
        Write-Host "[START] Safety Gateway is already running; it will be reused." -ForegroundColor Green
    }
    else {
        Write-Host "[START] Launching the local Safety Gateway in the background..." -ForegroundColor Yellow
        $GatewayProcess = Start-Process `
            -FilePath $SafetyPython `
            -ArgumentList @(
                "-m", "uvicorn", "safety_gateway.runtime:app",
                "--host", "127.0.0.1", "--port", "8010", "--workers", "1"
            ) `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -PassThru
        $StartedGateway = $true
    }

    $Deadline = (Get-Date).AddMinutes(3)
    $Attempt = 0
    while (-not (Test-GatewayReady)) {
        $Attempt += 1
        if ($StartedGateway -and $GatewayProcess.HasExited) {
            throw "Safety Gateway exited while loading Qwen (exit code $($GatewayProcess.ExitCode))."
        }
        if ((Get-Date) -ge $Deadline) {
            throw "Timed out after 3 minutes waiting for Qwen and the LoRA adapter."
        }
        Write-Host "[LOAD] Qwen base model + v11 LoRA are loading... check $Attempt" -ForegroundColor DarkYellow
        Start-Sleep -Seconds 5
    }

    Write-Host "[READY] Real local Qwen and v11 adapter are ready." -ForegroundColor Green
    Push-Location -LiteralPath $ProjectRoot
    try {
        & $AppPython -m scripts._privacy_flow_demo --gateway-url $GatewayUrl
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Full-flow smoke test failed with code $LASTEXITCODE"
    }
}
finally {
    if ($StartedGateway -and $null -ne $GatewayProcess -and -not $GatewayProcess.HasExited) {
        Write-Host "[STOP] Stopping the gateway process started by this test." -ForegroundColor DarkGray
        Stop-Process -Id $GatewayProcess.Id -ErrorAction SilentlyContinue
    }
}
