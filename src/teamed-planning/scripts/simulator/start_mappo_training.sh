#!/usr/bin/env bash
# Linux counterpart of start_mappo_training.ps1: checks the Conda environment,
# then starts train_mappo.py detached, with logs and launch.json in the output.
#   start_mappo_training.sh [--workers N] [--output DIR] [--config FILE] [--python EXE] [--resume]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
config="$root/src/teamed-planning/config/algorithm/mappo_training_1000m.json"
output="artifacts/mappo-1000m-$(date +%Y%m%d)"
python="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
workers=1
resume=()
while (($#)); do
    case "$1" in
        --config) config="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --resume) resume=(--resume); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$python" ]]; then
    python="$(conda info --base)/envs/teamedplanning-2d/bin/python"
fi
config="$(realpath "$config")"
[[ "$output" = /* ]] || output="$root/$output"
mkdir -p "$output"
output="$(realpath "$output")"
if [[ -e "$output/latest.pt" && ${#resume[@]} -eq 0 ]]; then
    echo "This output directory already has a checkpoint. Use --resume or choose another output." >&2
    exit 1
fi
if [[ -f "$output/launch.json" ]]; then
    previous="$("$python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$output/launch.json")"
    if ps -p "$previous" -o args= 2>/dev/null | grep -q train_mappo.py; then
        echo "Training process $previous is already running." >&2
        exit 1
    fi
fi
# Isolated mode ignores PYTHONPATH and user site-packages.
"$python" -I -c "import pathlib,sys,torch,numpy,scipy,skimage,networkx; assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), 'Install torch into the selected Python environment'; print('Python: '+sys.executable+'; torch: '+str(torch.__version__))"
cd "$root"
OPENBLAS_NUM_THREADS=1 setsid nohup "$python" -I -u src/teamed-planning/scripts/simulator/train_mappo.py \
    --config "$config" --output "$output" --workers "$workers" "${resume[@]}" \
    > "$output/stdout.log" 2> "$output/stderr.log" < /dev/null &
pid=$!
"$python" -I -c 'import json,sys,datetime
pid,python,config,output,workers,resume=sys.argv[1:]
json.dump(dict(pid=int(pid),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=python,config=config,output=output,workers=int(workers),resume=resume=="1"),
    open(output+"/launch.json","w"),indent=2)' \
    "$pid" "$python" "$config" "$output" "$workers" "$((${#resume[@]} > 0))"
echo "Started training PID=$pid workers=$workers output=$output"
