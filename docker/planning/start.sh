#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
cd /home/dcist/Teamed-Planning
catkin_make -j2 -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
python3 /opt/teamed-planning/control.py prepare
MODEL_PATH=/home/dcist/Teamed-Planning/src/teamed-planning/scripts/drone/models/segformer_b0_ade/model.onnx
if [[ ! -s "$MODEL_PATH" ]]; then
    echo 'Semantic model is missing; install it before starting planning.' >&2
    exit 1
fi
args=(num_drones:=3 seg_model_path:="$MODEL_PATH")
if [[ "${PLANNING_MODE:-default}" == "v9-vtemp1" || "${PLANNING_MODE:-default}" == "v9-vtemp" ]]; then
    checkpoint=/opt/v9-checkpoint/best.pt
    echo "7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198  $checkpoint" | sha256sum -c -
    args+=(sitl_v9_checkpoint:="$checkpoint" guidance_strategy:="${PLANNING_MODE#v9-}")
fi
exec roslaunch teamed_planning multi_robot_xtdrone.launch "${args[@]}"
