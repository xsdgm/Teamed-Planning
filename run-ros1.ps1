param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'status', 'takeoff', 'land', 'navigate', 'rviz', 'logs', 'shell')]
    [string]$Command = 'start',
    [Parameter(Position=1)][double]$X,
    [Parameter(Position=2)][double]$Y
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    switch ($Command) {
        'start' {
            docker compose up -d --build --wait --wait-timeout 300
            if ($LASTEXITCODE -eq 0) {
                Write-Host 'Gazebo desktop: http://localhost:6080/vnc.html?autoconnect=true&resize=scale'
            }
        }
        'stop' { docker compose down }
        'logs' { docker compose logs -f --tail 100 ros1 }
        'shell' { docker compose exec ros1 bash }
        'navigate' {
            if (-not $PSBoundParameters.ContainsKey('X') -or -not $PSBoundParameters.ContainsKey('Y')) {
                throw 'Usage: .\run-ros1.ps1 navigate 10 5'
            }
            $culture = [Globalization.CultureInfo]::InvariantCulture
            docker compose exec ros1 /opt/teamed-docker/control.sh navigate $X.ToString($culture) $Y.ToString($culture)
        }
        default { docker compose exec ros1 /opt/teamed-docker/control.sh $Command }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
