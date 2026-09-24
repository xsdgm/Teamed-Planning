#!/usr/bin/env python3
"""
Ground Control Station (GCS) Node
Runs on the ground robot. Handles:
  - Semi-automatic sequential takeoff / landing
  - Heartbeat monitoring for all robots
  - Operator command relay with validation
  - RViz visualization (markers for robots, graphs, comm links)
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from teamed_planning.msg import RobotCommand, RobotStatus, TopologicalGraphMsg

import sys
import os
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


class GCSNode:
    def __init__(self):
        rospy.init_node('gcs_node', anonymous=False)

        # ── Parameters ─────────────────────────────────────────────────
        self.num_drones = rospy.get_param('~num_drones', 3)
        self.num_ground_robots = rospy.get_param('~num_ground_robots', 1)
        self.heartbeat_timeout = rospy.get_param('~heartbeat_timeout', 5.0)
        self.takeoff_interval = rospy.get_param('~takeoff_interval', 5.0)
        self.takeoff_height = rospy.get_param('~takeoff_height', 5.0)

        # ── Robot registry ─────────────────────────────────────────────
        self.robot_status = {}      # {robot_id: RobotStatus msg}
        self.last_heartbeat = {}    # {robot_id: rospy.Time}
        self.robot_alive = {}       # {robot_id: bool}

        # Graph buffers (for visualization)
        self.drone_graphs = {}        # {drone_id: TopologicalGraphMsg}
        self.unified_graph_msg = None # TopologicalGraphMsg from ground robot

        # Per-drone visualization colors: (R, G, B, alpha)  node / edge
        self._drone_colors = {
            'drone_0': (ColorRGBA(0.2, 0.5, 1.0, 0.5), ColorRGBA(0.2, 0.5, 1.0, 0.4)),
            'drone_1': (ColorRGBA(0.0, 1.0, 0.8, 0.5), ColorRGBA(0.0, 1.0, 0.8, 0.4)),
            'drone_2': (ColorRGBA(1.0, 0.9, 0.0, 0.5), ColorRGBA(1.0, 0.9, 0.0, 0.4)),
            'drone_3': (ColorRGBA(1.0, 0.5, 0.0, 0.5), ColorRGBA(1.0, 0.5, 0.0, 0.4)),
        }

        # Sequential takeoff state
        self.takeoff_queue = []
        self.takeoff_in_progress = False
        self.current_takeoff_id = None
        self.takeoff_start_time = None
        self.auto_start_mission = rospy.get_param('~auto_start_mission', False)

        # Current navigation goal (set by /operator/goal, shown as red sphere in RViz)
        self.current_goal = None   # (x, y) tuple or None

        # ── Publishers ─────────────────────────────────────────────────
        self.marker_pub = rospy.Publisher(
            '/visualization_markers', MarkerArray, queue_size=1)

        self.cmd_pubs = {}
        for i in range(self.num_drones):
            did = f"drone_{i}"
            self.cmd_pubs[did] = rospy.Publisher(
                f"/{did}/command", RobotCommand, queue_size=5)
        for i in range(self.num_ground_robots):
            rid = f"ground_{i}"
            self.cmd_pubs[rid] = rospy.Publisher(
                f"/{rid}/command", RobotCommand, queue_size=5)

        self.goal_pub = {}
        for i in range(self.num_ground_robots):
            rid = f"ground_{i}"
            self.goal_pub[rid] = rospy.Publisher(
                f"/{rid}/goal", PoseStamped, queue_size=1)

        # ── Subscribers ────────────────────────────────────────────────
        for i in range(self.num_drones):
            did = f"drone_{i}"
            rospy.Subscriber(f"/{did}/status", RobotStatus,
                             self._status_cb, callback_args=did)
            rospy.Subscriber(f"/{did}/heartbeat", Header,
                             self._heartbeat_cb, callback_args=did)
        for i in range(self.num_ground_robots):
            rid = f"ground_{i}"
            rospy.Subscriber(f"/{rid}/status", RobotStatus,
                             self._status_cb, callback_args=rid)
            rospy.Subscriber(f"/{rid}/heartbeat", Header,
                             self._heartbeat_cb, callback_args=rid)

        rospy.Subscriber('/operator/goal', PoseStamped, self._operator_goal_cb)
        rospy.Subscriber('/operator/command', RobotCommand, self._operator_cmd_cb)

        for i in range(self.num_drones):
            did = f"drone_{i}"
            rospy.Subscriber(f"/{did}/graph_broadcast", TopologicalGraphMsg,
                             self._drone_graph_cb, callback_args=did)

        for i in range(self.num_ground_robots):
            rid = f"ground_{i}"
            rospy.Subscriber(f"/{rid}/unified_graph", TopologicalGraphMsg,
                             self._unified_graph_cb)

        # ── Timers ─────────────────────────────────────────────────────
        self._hb_check = rospy.Timer(rospy.Duration(1.0), self._check_heartbeats)
        self._viz_timer = rospy.Timer(rospy.Duration(0.5), self._publish_viz)
        self._takeoff_timer = rospy.Timer(rospy.Duration(1.0), self._takeoff_step)
        self._log_timer = rospy.Timer(rospy.Duration(5.0), self._log_summary)

        rospy.loginfo("[GCS] GCS node ready — "
                      f"{self.num_drones} drones, {self.num_ground_robots} ground robots")

    # ════════════════════════ Callbacks ═══════════════════════════════

    def _status_cb(self, msg, robot_id):
        self.robot_status[robot_id] = msg

    def _drone_graph_cb(self, msg: TopologicalGraphMsg, drone_id: str):
        self.drone_graphs[drone_id] = msg

    def _unified_graph_cb(self, msg: TopologicalGraphMsg):
        self.unified_graph_msg = msg

    def _heartbeat_cb(self, msg, robot_id):
        self.last_heartbeat[robot_id] = rospy.Time.now()
        self.robot_alive[robot_id] = True

    def _operator_goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.current_goal = (gx, gy)
        rospy.loginfo(f"[GCS] Operator goal -> ground_0: ({gx:.2f}, {gy:.2f})")
        for rid, pub in self.goal_pub.items():
            pub.publish(msg)

    def _operator_cmd_cb(self, msg):
        cmd = msg.command.upper()
        target = msg.target_robot_id

        if cmd == "TAKEOFF_ALL":
            self._start_sequential_takeoff()
            return
        if cmd == "LAND_ALL":
            self._land_all()
            return
        if cmd == "START_MISSION_ALL":
            self._send_all_drones("START_MISSION")
            return

        if target:
            pub = self.cmd_pubs.get(target)
            if pub:
                pub.publish(msg)
                rospy.loginfo(f"[GCS] Relayed {cmd} -> {target}")
            else:
                rospy.logwarn(f"[GCS] Unknown target: {target}")
        else:
            rospy.logwarn(f"[GCS] Command {cmd} has no target")

    # ════════════════════════ Sequential takeoff ═════════════════════

    def _start_sequential_takeoff(self):
        self.takeoff_queue = [f"drone_{i}" for i in range(self.num_drones)]
        self.takeoff_in_progress = True
        self.current_takeoff_id = None
        rospy.loginfo(f"[GCS] Sequential takeoff started: {self.takeoff_queue}")

    def _takeoff_step(self, _event):
        if not self.takeoff_in_progress:
            return
        if not self.takeoff_queue and self.current_takeoff_id is None:
            self.takeoff_in_progress = False
            rospy.loginfo("[GCS] All drones have taken off")
            if self.auto_start_mission:
                rospy.sleep(2.0)
                self._send_all_drones("START_MISSION")
            return

        if self.current_takeoff_id is not None:
            st = self.robot_status.get(self.current_takeoff_id)
            if st and st.flight_state in ("HOVERING", "MISSION"):
                rospy.loginfo(f"[GCS] {self.current_takeoff_id} is hovering — proceeding")
                self.current_takeoff_id = None
            elif (rospy.Time.now() - self.takeoff_start_time).to_sec() > 30.0:
                rospy.logwarn(f"[GCS] {self.current_takeoff_id} takeoff timeout")
                self.current_takeoff_id = None
            else:
                return

        if self.takeoff_queue:
            did = self.takeoff_queue.pop(0)
            rospy.loginfo(f"[GCS] Sending TAKEOFF to {did}")
            cmd = RobotCommand()
            cmd.header.stamp = rospy.Time.now()
            cmd.command = "TAKEOFF"
            cmd.target_robot_id = did
            cmd.param1 = self.takeoff_height
            pub = self.cmd_pubs.get(did)
            if pub:
                pub.publish(cmd)
            self.current_takeoff_id = did
            self.takeoff_start_time = rospy.Time.now()

    def _land_all(self):
        self.takeoff_in_progress = False
        self.takeoff_queue.clear()
        self._send_all_drones("LAND")

    def _send_all_drones(self, command):
        for i in range(self.num_drones):
            did = f"drone_{i}"
            cmd = RobotCommand()
            cmd.header.stamp = rospy.Time.now()
            cmd.command = command
            cmd.target_robot_id = did
            pub = self.cmd_pubs.get(did)
            if pub:
                pub.publish(cmd)
        rospy.loginfo(f"[GCS] Sent {command} to all drones")

    # ════════════════════════ Health monitoring ══════════════════════

    def _check_heartbeats(self, _event):
        now = rospy.Time.now()
        for rid, last in list(self.last_heartbeat.items()):
            alive = (now - last).to_sec() < self.heartbeat_timeout
            was_alive = self.robot_alive.get(rid, True)
            self.robot_alive[rid] = alive
            if was_alive and not alive:
                rospy.logwarn(f"[GCS] HEARTBEAT LOST: {rid}")
            elif not was_alive and alive:
                rospy.loginfo(f"[GCS] Heartbeat restored: {rid}")

    # ════════════════════════ RViz visualization ════════════════════

    def _publish_viz(self, _event):
        ma = MarkerArray()
        idx = 0

        for rid, st in self.robot_status.items():
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "robots"
            m.id = idx
            idx += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = st.position_x
            m.pose.position.y = st.position_y
            m.pose.position.z = st.position_z
            m.scale.x = m.scale.y = m.scale.z = 1.0

            alive = self.robot_alive.get(rid, False)
            if st.robot_type == "drone":
                # Match sphere color to the drone's graph color so they are visually linked
                graph_color, _ = self._drone_colors.get(
                    rid, (ColorRGBA(0.2, 0.6, 1.0, 1.0), None))
                sphere_color = ColorRGBA(graph_color.r, graph_color.g, graph_color.b, 1.0)
                m.color = sphere_color if alive else ColorRGBA(1.0, 0.2, 0.2, 0.6)
            else:
                m.color = ColorRGBA(0.2, 1.0, 0.2, 1.0) if alive else ColorRGBA(1.0, 0.2, 0.2, 0.6)

            m.lifetime = rospy.Duration(2.0)
            ma.markers.append(m)

            # Label
            t = Marker()
            t.header = m.header
            t.ns = "labels"
            t.id = idx
            idx += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = st.position_x
            t.pose.position.y = st.position_y
            t.pose.position.z = st.position_z + 1.5
            t.scale.z = 0.6
            t.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            t.text = f"{rid}\n{st.flight_state}\nG:{st.graph_nodes}"
            t.lifetime = rospy.Duration(2.0)
            ma.markers.append(t)

        # Goal marker — red sphere + "GOAL" label, persists until a new goal is set
        if self.current_goal is not None:
            gx, gy = self.current_goal

            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = rospy.Time.now()
            sphere.ns = "goal"
            sphere.id = 0
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = gx
            sphere.pose.position.y = gy
            sphere.pose.position.z = 0.5
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 1.0
            sphere.color = ColorRGBA(1.0, 0.0, 0.0, 0.85)
            sphere.lifetime = rospy.Duration(0)   # 0 = persist until replaced
            ma.markers.append(sphere)

            label = Marker()
            label.header = sphere.header
            label.ns = "goal"
            label.id = 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = gx
            label.pose.position.y = gy
            label.pose.position.z = 1.8
            label.scale.z = 0.7
            label.color = ColorRGBA(1.0, 0.3, 0.3, 1.0)
            label.text = f"GOAL\n({gx:.1f}, {gy:.1f})"
            label.lifetime = rospy.Duration(0)
            ma.markers.append(label)

        # ── Per-drone topological graphs ─────────────────────────────────
        for drone_id, graph_msg in self.drone_graphs.items():
            node_color, edge_color = self._drone_colors.get(
                drone_id,
                (ColorRGBA(0.8, 0.8, 0.8, 1.0), ColorRGBA(0.8, 0.8, 0.8, 0.6)))

            # Edges — LINE_LIST (pairs of points)
            if graph_msg.edges:
                node_pos = {n.node_id: (n.x, n.y) for n in graph_msg.nodes}
                edges_m = Marker()
                edges_m.header.frame_id = "map"
                edges_m.header.stamp = rospy.Time.now()
                edges_m.ns = "drone_graphs"
                edges_m.id = idx;  idx += 1
                edges_m.type = Marker.LINE_LIST
                edges_m.action = Marker.ADD
                edges_m.scale.x = 0.12          # line width
                edges_m.color = edge_color
                edges_m.lifetime = rospy.Duration(2.0)
                for e in graph_msg.edges:
                    p1 = node_pos.get(e.source)
                    p2 = node_pos.get(e.target)
                    if p1 and p2:
                        pt = Point(); pt.x = p1[0]; pt.y = p1[1]; pt.z = 0.1
                        edges_m.points.append(pt)
                        pt = Point(); pt.x = p2[0]; pt.y = p2[1]; pt.z = 0.1
                        edges_m.points.append(pt)
                ma.markers.append(edges_m)

            # Nodes — SPHERE_LIST
            if graph_msg.nodes:
                nodes_m = Marker()
                nodes_m.header.frame_id = "map"
                nodes_m.header.stamp = rospy.Time.now()
                nodes_m.ns = "drone_graphs"
                nodes_m.id = idx;  idx += 1
                nodes_m.type = Marker.SPHERE_LIST
                nodes_m.action = Marker.ADD
                nodes_m.scale.x = nodes_m.scale.y = nodes_m.scale.z = 0.5
                nodes_m.color = node_color
                nodes_m.lifetime = rospy.Duration(2.0)
                for n in graph_msg.nodes:
                    pt = Point(); pt.x = n.x; pt.y = n.y; pt.z = 0.2
                    nodes_m.points.append(pt)
                ma.markers.append(nodes_m)

        # ── Ground robot unified (merged) graph ──────────────────────────
        if self.unified_graph_msg is not None:
            g = self.unified_graph_msg
            node_color = ColorRGBA(0.1, 1.0, 0.3, 1.0)
            edge_color  = ColorRGBA(0.3, 1.0, 0.5, 1.0)

            if g.edges:
                node_pos = {n.node_id: (n.x, n.y) for n in g.nodes}
                edges_m = Marker()
                edges_m.header.frame_id = "map"
                edges_m.header.stamp = rospy.Time.now()
                edges_m.ns = "ground_graph"
                edges_m.id = idx;  idx += 1
                edges_m.type = Marker.LINE_LIST
                edges_m.action = Marker.ADD
                edges_m.scale.x = 0.14
                edges_m.color = edge_color
                edges_m.lifetime = rospy.Duration(2.0)
                for e in g.edges:
                    p1 = node_pos.get(e.source)
                    p2 = node_pos.get(e.target)
                    if p1 and p2:
                        pt = Point(); pt.x = p1[0]; pt.y = p1[1]; pt.z = 0.1
                        edges_m.points.append(pt)
                        pt = Point(); pt.x = p2[0]; pt.y = p2[1]; pt.z = 0.1
                        edges_m.points.append(pt)
                ma.markers.append(edges_m)

            if g.nodes:
                nodes_m = Marker()
                nodes_m.header.frame_id = "map"
                nodes_m.header.stamp = rospy.Time.now()
                nodes_m.ns = "ground_graph"
                nodes_m.id = idx;  idx += 1
                nodes_m.type = Marker.SPHERE_LIST
                nodes_m.action = Marker.ADD
                nodes_m.scale.x = nodes_m.scale.y = nodes_m.scale.z = 0.55
                nodes_m.color = node_color
                nodes_m.lifetime = rospy.Duration(2.0)
                for n in g.nodes:
                    pt = Point(); pt.x = n.x; pt.y = n.y; pt.z = 0.2
                    nodes_m.points.append(pt)
                ma.markers.append(nodes_m)

        self.marker_pub.publish(ma)

    # ════════════════════════ Periodic log ═══════════════════════════

    def _log_summary(self, _event):
        lines = ["[GCS] --- Status ---"]
        for rid in sorted(self.robot_status.keys()):
            st = self.robot_status[rid]
            alive = self.robot_alive.get(rid, False)
            hb = "OK" if alive else "LOST"
            lines.append(f"  {rid}: state={st.flight_state} "
                         f"phase={st.mission_phase} "
                         f"pos=({st.position_x:.1f},{st.position_y:.1f},{st.position_z:.1f}) "
                         f"graph={st.graph_nodes}n/{st.graph_edges}e hb={hb}")
        rospy.loginfo("\n".join(lines))


if __name__ == '__main__':
    try:
        GCSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
