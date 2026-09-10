# Local VS Code launcher. Agent runs inline; PostgreSQL/Redis run in Docker.
[CmdletBinding()]
param([int]$Port = 8000, [switch]$WithSafetyGateway)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$savedPath = $env:Path
$savedMode = $env:TASK_EXECUTION_MODE
$savedRedis = $env:REDIS_URL
$savedAppEnv = $env:APP_ENV
$savedGatewayMode = $env:PRIVACY_GATEWAY_MODE
$gatewayProcess = $null

function Test-Ready([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch { return $false }
}

Push-Location -LiteralPath $projectRoot
try {
    if (!(Test-Path -LiteralPath $python)) { throw 'Missing .venv\Scripts\python.exe. Set up the project Python environment first.' }
    if (!(Test-Path -LiteralPath '.env')) { throw 'Missing .env. Configure project credentials first.' }

    if (Test-Ready "http://127.0.0.1:$Port/ready") {
        throw "Port $Port already has a running server. Run .\stop.ps1 first, then start again so this terminal owns the server."
    }
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $listener.Start() } catch { throw "Port $Port is occupied but /ready is not healthy. Stop the existing server or use -Port 8001." } finally { $listener.Stop() }

    $dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
    $docker = if ($dockerCommand) { $dockerCommand.Source } else { $null }
    if (!$docker) {
        foreach ($candidate in @(
            "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
            "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe"
        )) {
            if (Test-Path -LiteralPath $candidate) { $docker = $candidate; break }
        }
    }
    if (!$docker) { throw 'Docker CLI not found. Install/open Docker Desktop and restart the terminal.' }
    $env:Path = (Split-Path -Parent $docker) + ';' + $env:Path
    & $docker info --format '{{.ServerVersion}}' *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is unavailable. Open Docker Desktop, wait until it is running, then retry.' }

    if (!(Select-String -LiteralPath '.env' -Pattern '^\s*REDIS_PASSWORD\s*=' -Quiet)) {
        $newPassword = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
        Add-Content -LiteralPath '.env' -Value "`nREDIS_PASSWORD=$newPassword" -Encoding UTF8
        Write-Host 'Added missing local Redis password to .env (value hidden).'
    }

    # Explicit opt-in only, regardless of a stale parent terminal setting.
    $env:PRIVACY_GATEWAY_MODE = if ($WithSafetyGateway) { 'enforce' } else { 'disabled' }
    # Capture configuration in memory; never print connection strings or secrets.
    $configScript = @'
import json, os
from dotenv import dotenv_values
from app.config import settings
values = dotenv_values('.env')
print(json.dumps({'redis_password': os.environ.get('REDIS_PASSWORD', values.get('REDIS_PASSWORD', '')), 'gateway_mode': settings.privacy_gateway_mode, 'gateway_url': settings.privacy_gateway_url}))
'@
    $configJson = $configScript | & $python -
    if ($LASTEXITCODE -ne 0) { throw 'Project configuration is invalid. See the validation error above.' }
    $config = $configJson | ConvertFrom-Json
    if (!$config.redis_password) { throw 'REDIS_PASSWORD is empty. Set a nonempty value in .env.' }

    Write-Host '[1/4] Starting PostgreSQL and Redis...' -ForegroundColor Cyan
    & $docker compose --env-file .env up -d --wait --wait-timeout 120 postgres redis
    if ($LASTEXITCODE -ne 0) { throw 'Database/Redis startup failed. See Docker output above.' }

    $env:TASK_EXECUTION_MODE = 'inline'
    $env:APP_ENV = 'local'
    $env:REDIS_URL = 'redis://:' + [uri]::EscapeDataString($config.redis_password) + '@127.0.0.1:6379/2'

    Write-Host '[2/4] Applying database migrations...' -ForegroundColor Cyan
    & $python -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) { throw 'Database migration failed. Check DATABASE_URL and the existing database credentials.' }

    Write-Host '[3/4] Checking local privacy gateway...' -ForegroundColor Cyan
    if ($config.gateway_mode -ne 'disabled') {
        $gatewayUrl = $config.gateway_url.TrimEnd('/')
        if (!(Test-Ready "$gatewayUrl/ready")) {
            $gatewayUri = [uri]$gatewayUrl
            if ($gatewayUri.Host -notin @('127.0.0.1', 'localhost') -or $gatewayUri.Scheme -ne 'http') {
                throw 'The configured gateway is not ready. Start that gateway separately and retry.'
            }
            if (Test-Ready "$gatewayUrl/health") { throw 'An existing gateway is alive but not ready. Check its model-loading logs before retrying.' }
            $gatewayPython = Join-Path $projectRoot '.venv-safety\Scripts\python.exe'
            if (!(Test-Path -LiteralPath $gatewayPython)) { throw 'Privacy enforcement requires .venv-safety. Run scripts/setup_safety_gateway.ps1 first.' }
            $logDir = Join-Path $projectRoot 'data\local\launcher'
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
            $logStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $gatewayLog = Join-Path $logDir "gateway-$logStamp.stderr.log"
            $gatewayProcess = Start-Process -FilePath $gatewayPython -ArgumentList @('-m', 'uvicorn', 'safety_gateway.runtime:app', '--host', '127.0.0.1', '--port', "$($gatewayUri.Port)", '--workers', '1') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "gateway-$logStamp.stdout.log") -RedirectStandardError $gatewayLog
            $deadline = (Get-Date).AddMinutes(4)
            $nextNotice = Get-Date
            while (!(Test-Ready "$gatewayUrl/ready")) {
                if ($gatewayProcess.HasExited) { throw "Gateway exited. Inspect $gatewayLog" }
                if ((Get-Date) -gt $deadline) { throw "Gateway model load timed out. Inspect $gatewayLog" }
                if ((Get-Date) -ge $nextNotice) {
                    Write-Host 'Loading local privacy model; this can take a few minutes...'
                    $nextNotice = (Get-Date).AddSeconds(20)
                }
                Start-Sleep -Seconds 2
            }
        }
        Write-Host 'Privacy gateway ready.' -ForegroundColor Green
    } else { Write-Host 'Privacy gateway is disabled in your configuration.' }

    Write-Host "[4/4] Starting EasyTeaching: http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host 'Local experience mode: inline Agent, existing local RAG/Drive configuration. Wait for Application startup complete; Ctrl+C stops this API.'
    & $python scripts/run_api.py --port $Port
    if ($LASTEXITCODE -notin @(0, 130, -1073741510)) { throw "API exited with code $LASTEXITCODE. See output above." }
} finally {
    if ($gatewayProcess -and !$gatewayProcess.HasExited) {
        Stop-Process -Id $gatewayProcess.Id -ErrorAction SilentlyContinue
    }
    $env:Path = $savedPath
    $env:TASK_EXECUTION_MODE = $savedMode
    $env:REDIS_URL = $savedRedis
    $env:APP_ENV = $savedAppEnv
    $env:PRIVACY_GATEWAY_MODE = $savedGatewayMode
    Write-Host 'API stopped. PostgreSQL/Redis remain running; use .\stop.ps1 -All to stop them too.'
    Pop-Location
}
