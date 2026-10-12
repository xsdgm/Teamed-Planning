param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'status', 'logs', 'takeoff', 'land', 'navigate', 'rviz')]
    [string]$Command = 'start',
    [Parameter(Position=1)][double]$X,
    [Parameter(Position=2)][double]$Y
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    switch ($Command) {
        'start' {
            docker image inspect xtdrone_ad:v4 teamedplanning-ad:v1 *> $null
            if ($LASTEXITCODE -ne 0) {
                $archive = Join-Path $PSScriptRoot 'docker/teamedplanning-docker-images-2026-10-10.tar.gz'
                if (-not (Test-Path -LiteralPath $archive)) { throw "Image archive not found: $archive" }
                Write-Host 'Restoring Docker images from the archive. This can take several minutes.'
                docker load --input $archive
                if ($LASTEXITCODE -ne 0) { throw 'Failed to restore Docker images.' }
            }
            docker compose -f compose.xtdrone.yaml --profile planning up -d --wait --wait-timeout 420
            if ($LASTEXITCODE -eq 0) {
                Write-Host 'Simulation and planning ready. Gazebo uses the NVIDIA GPU in a WSLg desktop window.'
            }
        }
        'stop' { docker compose -f compose.xtdrone.yaml --profile planning stop }
        'logs' { docker compose -f compose.xtdrone.yaml logs -f --tail 100 planning }
        'rviz' {
            docker compose -f compose.xtdrone.yaml exec -d planning bash -c 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && rviz -d src/teamed-planning/config/gcs.rviz >/tmp/rviz.log 2>&1'
        }
        'navigate' {
            if (-not $PSBoundParameters.ContainsKey('X') -or -not $PSBoundParameters.ContainsKey('Y')) {
                throw 'Usage: .\run-planning.ps1 navigate 10 5'
            }
            $culture = [Globalization.CultureInfo]::InvariantCulture
            docker compose -f compose.xtdrone.yaml exec planning bash /opt/teamed-planning/control.sh navigate $X.ToString($culture) $Y.ToString($culture)
        }
        default {
            docker compose -f compose.xtdrone.yaml exec planning bash /opt/teamed-planning/control.sh $Command
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
