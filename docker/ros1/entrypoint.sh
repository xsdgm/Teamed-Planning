#!/usr/bin/env bash
set -eo pipefail
source /opt/teamed-docker/environment.sh
if [[ "${1:-}" != sim ]]; then
    exec "$@"
fi
cd /ws
catkin_make -j2 -DPYTHON_EXECUTABLE=/usr/bin/python3
source /opt/teamed-docker/environment.sh

pids=()
cleanup() {
    trap - EXIT TERM INT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
}
trap cleanup EXIT TERM INT
Xvfb :1 -screen 0 1440x900x24 -ac +extension GLX +render -noreset &
pids+=("$!")
for attempt in {1..30}; do
    [[ -S /tmp/.X11-unix/X1 ]] && break
    sleep 1
done
fluxbox >/tmp/fluxbox.log 2>&1 &
pids+=("$!")
x11vnc -display :1 -forever -shared -nopw -localhost -rfbport 5900 >/tmp/vnc.log 2>&1 &
pids+=("$!")
websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
pids+=("$!")
roslaunch /opt/teamed-docker/simulation.launch gui:="${GUI:-true}" &
pids+=("$!")
sim_pid=$!
for attempt in {1..90}; do
    kill -0 "$sim_pid" 2>/dev/null || exit 1
    if rosservice list 2>/dev/null | grep -q /gazebo/get_world_properties; then break; fi
    sleep 2
done
python3 /opt/teamed-docker/control.py ready
roslaunch teamed_planning multi_robot_xtdrone.launch seg_model_path:="" &
pids+=("$!")
echo 'Browser desktop: http://localhost:6080/vnc.html?autoconnect=true&resize=scale'
wait -n "${pids[@]}"
