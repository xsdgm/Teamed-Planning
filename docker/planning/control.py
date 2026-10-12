#!/usr/bin/env python3
import json
import sys
import time

import rospy
import rosnode
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, ParamValue
from mavros_msgs.srv import ParamSet
from nav_msgs.msg import Odometry
from teamed_planning.msg import RobotCommand, RobotStatus


def status():
    for robot in ("drone_0", "drone_1", "drone_2", "ground_0"):
        msg = rospy.wait_for_message("/%s/status" % robot, RobotStatus, timeout=12)
        print(json.dumps({"robot": robot, "state": msg.flight_state,
                          "phase": msg.mission_phase,
                          "position": [msg.position_x, msg.position_y, msg.position_z],
                          "graph_nodes": msg.graph_nodes, "graph_edges": msg.graph_edges,
                          "comm_ok": msg.comm_ok}))


def publish(topic, message):
    pub = rospy.Publisher(topic, type(message), queue_size=1, latch=True)
    deadline = time.monotonic() + 15
    while pub.get_num_connections() == 0 and time.monotonic() < deadline:
        time.sleep(0.1)
    if pub.get_num_connections() == 0:
        raise RuntimeError("No subscriber for " + topic)
    message.header.stamp = rospy.Time.now()
    pub.publish(message)
    time.sleep(1)


def command(value):
    msg = RobotCommand()
    msg.command = value
    msg.target_robot_id = "ALL"
    publish("/operator/command", msg)


def wait_for_hover():
    latest = {}
    subscribers = [rospy.Subscriber("/drone_%d/status" % i, RobotStatus,
                                   lambda msg, index=i: latest.update({index: msg}))
                   for i in range(3)]
    deadline = time.monotonic() + 600
    last_report = 0
    while time.monotonic() < deadline:
        if len(latest) == 3 and all(msg.flight_state in ("HOVERING", "MISSION")
                                    for msg in latest.values()):
            print("All three drones are hovering", flush=True)
            return
        if time.monotonic() - last_report > 10:
            print({i: (msg.flight_state, round(msg.position_z, 1))
                   for i, msg in latest.items()}, flush=True)
            last_report = time.monotonic()
        time.sleep(0.2)
    raise RuntimeError("Not all drones reached hover before the timeout")


def prepare():
    for index in range(3):
        ns = "/iris_%d/mavros" % index
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            state = rospy.wait_for_message(ns + "/state", State, timeout=10)
            if state.connected:
                break
        else:
            raise RuntimeError(ns + " did not connect")
        rospy.wait_for_service(ns + "/param/set", timeout=30)
        setter = rospy.ServiceProxy(ns + "/param/set", ParamSet)
        for name, value in (("EKF2_AID_MASK", 1), ("SDLOG_MODE", 0)):
            reply = setter(name, ParamValue(integer=value, real=0.0))
            if not reply.success:
                print("WARNING: PX4 did not accept optional parameter", name,
                      "- retaining its existing configuration", flush=True)
        rospy.wait_for_message(ns + "/local_position/odom", Odometry, timeout=120)
        print("READY", ns, flush=True)


def main():
    rospy.init_node("planning_operator", anonymous=True, disable_signals=True)
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "health":
        required = {"/gcs_node", "/ground_0/ground_node"}
        required.update("/drone_%d/%s" % (i, node) for i in range(3)
                        for node in ("drone_node", "seg_node"))
        missing = required - set(rosnode.get_node_names())
        if missing:
            raise RuntimeError("Missing nodes: " + str(sorted(missing)))
    elif action == "status":
        status()
    elif action == "takeoff":
        command("TAKEOFF_ALL")
    elif action == "wait-hover":
        wait_for_hover()
    elif action == "land":
        command("LAND_ALL")
    elif action == "navigate":
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y = map(float, sys.argv[2:4])
        msg.pose.orientation.w = 1.0
        publish("/operator/goal", msg)
        command("START_MISSION_ALL")
    else:
        raise ValueError("Unknown action: " + action)


if __name__ == "__main__":
    main()
