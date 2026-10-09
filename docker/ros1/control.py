"""Operator commands for the Docker-only simulation (never real hardware)."""
import argparse
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, ParamValue
from mavros_msgs.srv import ParamSet, ParamPull, SetMode
from nav_msgs.msg import Odometry
from teamed_planning.msg import RobotCommand, RobotStatus


def publish(topic, message_type, message):
    pub = rospy.Publisher(topic, message_type, queue_size=1)
    deadline = time.monotonic() + 10
    while pub.get_num_connections() == 0 and time.monotonic() < deadline:
        rospy.sleep(0.1)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f'No subscriber on {topic}; check docker compose logs')
    message.header.stamp = rospy.Time.now()
    pub.publish(message)
    rospy.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['ready', 'status', 'takeoff', 'land', 'navigate'])
    parser.add_argument('coordinates', nargs='*', type=float)
    args = parser.parse_args()
    if args.command == 'navigate' and len(args.coordinates) != 2:
        parser.error('navigate requires x y, for example: navigate 10 5')
    rospy.init_node('docker_operator', anonymous=True)
    if args.command == 'ready':
        for i in range(3):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    state = rospy.wait_for_message(f'/iris_{i}/mavros/state', State, timeout=3)
                    if state.connected:
                        break
                except rospy.ROSException:
                    pass
            else:
                raise RuntimeError(f'iris_{i}: PX4 connection timed out')
            service = f'/iris_{i}/mavros/param/pull'
            rospy.wait_for_service(service, timeout=15)
            result = rospy.ServiceProxy(service, ParamPull)(force_pull=True)
            if not result.success or result.param_received == 0:
                raise RuntimeError(f'iris_{i}: MAVROS parameter download failed')
            print(f'iris_{i}: connected, {result.param_received} parameters received', flush=True)
        return
    if args.command == 'status':
        for i in range(3):
            state = rospy.wait_for_message(f'/iris_{i}/mavros/state', State, timeout=5)
            status = rospy.wait_for_message(f'/drone_{i}/status', RobotStatus, timeout=5)
            print(f'iris_{i}: connected={state.connected}, armed={state.armed}, mode={state.mode}')
            print(status)
        return
    if args.command == 'takeoff':
        for i in range(3):
            service = f'/iris_{i}/mavros/param/set'
            rospy.wait_for_service(service, timeout=15)
            setter = rospy.ServiceProxy(service, ParamSet)
            for name, value in [('COM_RCL_EXCEPT', 4), ('EKF2_AID_MASK', 1), ('SDLOG_MODE', 0)]:
                result = setter(name, ParamValue(integer=value, real=0.0))
                if not result.success:
                    raise RuntimeError(f'iris_{i}: failed to set {name}')
            rospy.wait_for_message(f'/iris_{i}/mavros/local_position/odom', Odometry, timeout=45)
    if args.command == 'navigate':
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x, goal.pose.position.y = args.coordinates
        goal.pose.orientation.w = 1.0
        publish('/operator/goal', PoseStamped, goal)
    command = RobotCommand()
    command.command = {'takeoff': 'TAKEOFF_ALL', 'land': 'LAND_ALL', 'navigate': 'START_MISSION_ALL'}[args.command]
    command.target_robot_id = 'ALL'
    publish('/operator/command', RobotCommand, command)
    if args.command == 'land':
        stop = RobotCommand()
        stop.command = 'STOP'
        stop.target_robot_id = 'ground_0'
        publish('/operator/command', RobotCommand, stop)
        # Let PX4's land detector complete touchdown and disarming. The project
        # controller can label a vehicle IDLE before PX4 considers it landed.
        for i in range(3):
            service = f'/iris_{i}/mavros/set_mode'
            rospy.wait_for_service(service, timeout=10)
            result = rospy.ServiceProxy(service, SetMode)(custom_mode='AUTO.LAND')
            if not result.mode_sent:
                raise RuntimeError(f'iris_{i}: AUTO.LAND rejected')
    print(f'Sent {command.command}')


if __name__ == '__main__':
    main()
