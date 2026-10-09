#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
if [[ -f /ws/devel/setup.bash ]]; then
    source /ws/devel/setup.bash
fi
PX4_ROOT=/opt/PX4-Autopilot
source "$PX4_ROOT/Tools/setup_gazebo.bash" "$PX4_ROOT" "$PX4_ROOT/build/px4_sitl_default" >/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_ROOT:$PX4_ROOT/Tools/sitl_gazebo"
export PYTHONPATH="/ws/src/teamed-planning/scripts:${PYTHONPATH:-}"
