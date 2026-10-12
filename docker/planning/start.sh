#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
bash /opt/wslg/check-gpu.sh
cd /home/dcist/Teamed-Planning
catkin_make -j2 -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
python3 /opt/teamed-planning/control.py prepare
MODEL_PATH=/home/dcist/Teamed-Planning/src/teamed-planning/scripts/drone/models/segformer_b0_ade/model.onnx
if [[ ! -s "$MODEL_PATH" ]]; then
    echo 'Semantic model is missing; install it before starting planning.' >&2
    exit 1
fi
exec roslaunch teamed_planning multi_robot_xtdrone.launch num_drones:=3 seg_model_path:="$MODEL_PATH"
