#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /home/dcist/Teamed-Planning/devel/setup.bash
exec python3 /opt/teamed-planning/control.py "$@"
