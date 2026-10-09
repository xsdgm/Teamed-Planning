#!/usr/bin/env bash
set -eo pipefail
source /opt/teamed-docker/environment.sh
if [[ "${1:-}" == rviz ]]; then
    exec rviz -d /ws/src/teamed-planning/config/gcs.rviz
fi
exec python3 /opt/teamed-docker/control.py "$@"
