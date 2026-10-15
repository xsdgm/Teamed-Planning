param(
    [string]$Config = '',
    [string]$Output = 'artifacts/mappo-moving-ground-fixed-20261015',
    [string]$Python = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../../..')).Path
if (-not $Config) { $Config = Join-Path $taskRoot 'src/teamed-planning/config/algorithm/mappo_training.json' }
$taskConfig = (Resolve-Path -LiteralPath $Config).Path
if (-not $Python) { $Python = Join-Path $env:USERPROFILE '.conda/envs/teamedplanning-2d/python.exe' }
$taskPython = (Resolve-Path -LiteralPath $Python).Path
if ([IO.Path]::IsPathRooted($Output)) { $taskOutput = [IO.Path]::GetFullPath($Output) }
else { $taskOutput = [IO.Path]::GetFullPath((Join-Path $taskRoot $Output)) }
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
if ((Test-Path -LiteralPath (Join-Path $taskOutput 'latest.pt')) -and -not $Resume) {
    throw 'This output directory already has a checkpoint. Use -Resume or choose another output.'
}
$taskLaunchPath = Join-Path $taskOutput 'launch.json'
if (Test-Path -LiteralPath $taskLaunchPath) {
    $taskPrevious = Get-Content -LiteralPath $taskLaunchPath -Raw | ConvertFrom-Json
    $taskRunning = Get-Process -Id $taskPrevious.pid -ErrorAction SilentlyContinue
    if ($taskRunning -and $taskRunning.ProcessName -like 'python*') {
        throw "Training process $($taskPrevious.pid) is already running."
    }
}
# Isolated mode ignores inherited PYTHONPATH and user site-packages. All
# dependencies must come from the selected (normally Conda) interpreter.
& $taskPython -I -c "import pathlib,sys,torch,numpy,scipy,skimage,networkx; assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), 'Install torch into the selected Python environment'; print('Python: '+sys.executable+'; torch: '+str(torch.__version__)+'; path: '+torch.__file__)"
if ($LASTEXITCODE -ne 0) {
    throw 'Training dependency check failed. Install requirements into the selected Conda environment.'
}
$taskTrainer = Join-Path $PSScriptRoot 'train_mappo.py'
$taskArguments = @('-I', '-u', ('"'+$taskTrainer+'"'), '--config', ('"'+$taskConfig+'"'), '--output', ('"'+$taskOutput+'"'))
if ($Resume) { $taskArguments += '--resume' }
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $taskOutput 'stdout.log') `
    -RedirectStandardError (Join-Path $taskOutput 'stderr.log')
@{pid=$taskProcess.Id; started_utc=[DateTime]::UtcNow.ToString('o'); python=$taskPython;
  config=$taskConfig; output=$taskOutput; resume=[bool]$Resume} |
    ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $taskLaunchPath
Write-Output "Started training PID=$($taskProcess.Id) output=$taskOutput"
