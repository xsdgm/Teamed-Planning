#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/PX4_Firmware/Tools/setup_gazebo.bash /root/PX4_Firmware /root/PX4_Firmware/build/px4_sitl_default
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:/root/PX4_Firmware:/root/PX4_Firmware/Tools/sitl_gazebo"
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1
unset XAUTHORITY
bash /opt/wslg/check-gpu.sh
python3 /opt/wslg/prepare_launch.py
echo 'Gazebo opens as a Windows desktop window through WSLg using NVIDIA graphics.'
exec roslaunch /tmp/teamed-gazebo.launch gui:=true
