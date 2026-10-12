param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'status', 'logs')]
    [string]$Command = 'start'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    switch ($Command) {
        'start' {
            docker compose -f compose.xtdrone.yaml up -d --wait --wait-timeout 300
            if ($LASTEXITCODE -eq 0) {
                Write-Host 'Gazebo is ready in a WSLg desktop window using NVIDIA graphics.'
            }
        }
        'stop' { docker compose -f compose.xtdrone.yaml stop }
        'status' { docker compose -f compose.xtdrone.yaml ps }
        'logs' { docker compose -f compose.xtdrone.yaml logs -f --tail 100 }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
