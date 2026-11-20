param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'status', 'logs', 'takeoff', 'land', 'navigate', 'rviz')]
    [string]$Command = 'start',
    [Parameter(Position=1)][double]$X,
    [Parameter(Position=2)][double]$Y,
    [string]$VTempCheckpoint,
    [string]$VTemp1Checkpoint
)
$ErrorActionPreference = 'Stop'
$priorPlanningMode = $env:PLANNING_MODE
$priorV9Checkpoint = $env:V9_CHECKPOINT
Push-Location $PSScriptRoot
try {
    switch ($Command) {
        'start' {
            $composeArgs = @('-f', 'compose.xtdrone.yaml')
            $env:PLANNING_MODE = 'default'
            if ($VTemp1Checkpoint -and $VTempCheckpoint) {
                throw 'Choose one guidance strategy with -VTempCheckpoint or -VTemp1Checkpoint.'
            }
            if ($VTemp1Checkpoint -or $VTempCheckpoint) {
                if ($VTempCheckpoint) {
                    $checkpoint = Get-Item -LiteralPath $VTempCheckpoint
                    $env:PLANNING_MODE = 'v9-vtemp'
                } else {
                    $checkpoint = Get-Item -LiteralPath $VTemp1Checkpoint
                    $env:PLANNING_MODE = 'v9-vtemp1'
                }
                $expectedHash = '7AC3AEDCC3876902773B2241F60E839EB3E3B5630DDEB3C8BE660421F2C24198'
                if ((Get-FileHash -LiteralPath $checkpoint.FullName).Hash -ne $expectedHash) {
                    throw 'Expected the original v9 best576 best.pt checkpoint.'
                }
                $env:V9_CHECKPOINT = $checkpoint.FullName
                $composeArgs += @('-f', 'docker/planning/v9.compose.yaml')
            }
            docker image inspect xtdrone_ad:v4 teamedplanning-ad:v1 *> $null
            if ($LASTEXITCODE -ne 0) {
                $archive = Join-Path $PSScriptRoot 'docker/teamedplanning-docker-images-2026-10-10.tar.gz'
                if (-not (Test-Path -LiteralPath $archive)) { throw "Image archive not found: $archive" }
                Write-Host 'Restoring Docker images from the archive. This can take several minutes.'
                docker load --input $archive
                if ($LASTEXITCODE -ne 0) { throw 'Failed to restore Docker images.' }
            }
            if ($VTemp1Checkpoint -or $VTempCheckpoint) {
                docker compose @composeArgs --profile planning build planning
                if ($LASTEXITCODE -ne 0) { throw 'Failed to build the v9 planning image.' }
            }
            docker compose @composeArgs --profile planning up -d --wait --wait-timeout 420
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
} finally {
    $env:PLANNING_MODE = $priorPlanningMode
    $env:V9_CHECKPOINT = $priorV9Checkpoint
    Pop-Location
}
