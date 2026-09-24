cmake_minimum_required(VERSION 3.0.2)
project(teamed_planning)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  rospy
  message_generation
  mavros_msgs
)

add_message_files(
  FILES
  GraphNode.msg
  GraphEdge.msg
  TopologicalGraphMsg.msg
  MissionBroadcast.msg
  RobotCommand.msg
  RobotStatus.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    nav_msgs
    sensor_msgs
    std_msgs
    rospy
    message_runtime
    mavros_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/drone/ros_drone_node.py
  scripts/drone/ros_seg_node.py
  scripts/ground/ros_ground_node.py
  scripts/ground/ros_gcs_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
