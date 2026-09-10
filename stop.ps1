# Stop only Python servers rooted in this project's virtual environments.
[CmdletBinding()]
param([switch]$All)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$processes = @(Get-CimInstance Win32_Process)
$roots = @($processes | Where-Object {
    $_.ExecutablePath -in @(
        (Join-Path $projectRoot '.venv\Scripts\python.exe'),
        (Join-Path $projectRoot '.venv-safety\Scripts\python.exe')
    ) -and $_.CommandLine -match 'scripts[/\\]run_api\.py|safety_gateway\.runtime:app'
})
$ids = [System.Collections.Generic.HashSet[int]]::new()
foreach ($item in $roots) { [void]$ids.Add([int]$item.ProcessId) }
do {
    $added = $false
    foreach ($item in $processes) {
        if ($ids.Contains([int]$item.ParentProcessId) -and $ids.Add([int]$item.ProcessId)) { $added = $true }
    }
} while ($added)
# Children first; no process-name-wide kill and no database data deletion.
foreach ($item in @($processes | Where-Object { $ids.Contains([int]$_.ProcessId) } | Sort-Object CreationDate -Descending)) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($item.ProcessId)"
    if ($current -and $current.CreationDate -eq $item.CreationDate) {
        Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped EasyTeaching process $($item.ProcessId)."
    }
}
if ($ids.Count -eq 0) { Write-Host 'No local EasyTeaching Python server found.' }
if ($All) {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    $docker = if ($command) { $command.Source } else { $null }
    if (!$docker) {
        foreach ($candidate in @("$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe", "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe")) {
            if (Test-Path -LiteralPath $candidate) { $docker = $candidate; break }
        }
    }
    if (!$docker) { throw 'Docker CLI not found.' }
    & $docker compose --project-directory $projectRoot --env-file (Join-Path $projectRoot '.env') stop
    if ($LASTEXITCODE -ne 0) { throw 'Docker service stop failed.' }
    Write-Host 'Project containers stopped. Database volumes preserved.'
}
