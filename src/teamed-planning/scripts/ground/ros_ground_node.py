#!/usr/bin/env python3
"""
Ground Robot ROS Node
Wraps GroundRobotController — bridges SLAM odom, LiDAR, cmd_vel, and
drone graphs into a single node.
"""

import rospy
import numpy as np
import traceback

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header
from scipy.ndimage import binary_dilation

from teamed_planning.msg import (
    TopologicalGraphMsg, MissionBroadcast,
    RobotCommand, RobotStatus,
)

import sys
import os
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from ground.ground_main import GroundRobotController
from common.graph_structure import TopologicalGraph

import tf.transformations as tft


class GroundROSNode:
    def __init__(self):
        rospy.init_node('ground_node', anonymous=False)

        # ── Parameters ─────────────────────────────────────────────────
        self.robot_id = rospy.get_param('~robot_id', 'ground_0')
        self.perception_range = rospy.get_param('~perception_range', 8.0)
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 1.0)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)
        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.comm_timeout = rospy.get_param('~comm_timeout', 10.0)
        self.num_drones = rospy.get_param('~num_drones', 3)

        # libgazebo_ros_diff_drive with legacyMode=false initialises its odom
        # frame at the Gazebo world origin, so /ground_0/odom already carries
        # world-frame absolute positions.  spawn_offset is therefore (0, 0).
        # The param is kept for flexibility; the launch file sets it to 0.
        self.spawn_offset = (
            float(rospy.get_param('~spawn_x', 0.0)),
            float(rospy.get_param('~spawn_y', 0.0)),
        )

        # ── State ──────────────────────────────────────────────────────
        self.current_position = None  # raw ground_0/odom frame
        self.current_heading = 0.0
        self.last_odom_time = None
        self.current_pointcloud = None  # sensor_msgs/PointCloud2 from Velodyne

        # Drone graph buffers
        self.drone_graphs = {}         # {drone_id: TopologicalGraph}
        self.last_graph_times = {}

        # ── Algorithm ──────────────────────────────────────────────────
        self.controller = GroundRobotController(
            robot_id=self.robot_id,
            perception_range=self.perception_range,
            max_linear_vel=self.max_linear_vel,
            max_angular_vel=self.max_angular_vel,
        )

        # ── Publishers ─────────────────────────────────────────────────
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.mission_pub = rospy.Publisher(
            'mission_broadcast', MissionBroadcast, queue_size=1)
        self.status_pub = rospy.Publisher('status', RobotStatus, queue_size=1)
        self.heartbeat_pub = rospy.Publisher('heartbeat', Header, queue_size=1)
        # Unified (merged) graph — consumed by GCS for visualization
        self.unified_graph_pub = rospy.Publisher(
            'unified_graph', TopologicalGraphMsg, queue_size=1)
        # Planned path — the route the robot intends to drive (map frame)
        self.planned_path_pub = rospy.Publisher(
            'planned_path', Path, queue_size=1, latch=True)

        # ── Subscribers ────────────────────────────────────────────────
        rospy.Subscriber('odom', Odometry, self._odom_cb)
        rospy.Subscriber('points', PointCloud2, self._pointcloud_cb)
        rospy.Subscriber('goal', PoseStamped, self._goal_cb)
        rospy.Subscriber('command', RobotCommand, self._command_cb)

        for i in range(self.num_drones):
            did = f"drone_{i}"
            rospy.Subscriber(
                f"/{did}/graph_broadcast", TopologicalGraphMsg,
                self._drone_graph_cb, callback_args=did)

        # ── Timers ─────────────────────────────────────────────────────
        self._ctrl_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._control_loop)
        self._hb_timer = rospy.Timer(
            rospy.Duration(1.0), self._heartbeat_loop)
        self._st_timer = rospy.Timer(
            rospy.Duration(0.5), self._status_loop)
        self._mission_timer = rospy.Timer(
            rospy.Duration(1.0), self._mission_loop)
        self._unified_graph_timer = rospy.Timer(
            rospy.Duration(1.0), self._publish_unified_graph)

        rospy.loginfo(f"[{self.robot_id}] Ground node ready")

    # ════════════════════════ Callbacks ═══════════════════════════════

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.current_position = (p.x, p.y)
        o = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.current_heading = yaw
        self.last_odom_time = rospy.Time.now()

    def _pointcloud_cb(self, msg):
        self.current_pointcloud = msg

    def _goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        rospy.loginfo(f"[{self.robot_id}] New goal: ({gx:.2f}, {gy:.2f})")
        self.controller.set_goal((gx, gy))

    def _command_cb(self, msg):
        tgt = msg.target_robot_id
        if tgt and tgt != self.robot_id and tgt != "ALL":
            return
        cmd = msg.command.upper()
        rospy.loginfo(f"[{self.robot_id}] Command: {cmd}")
        if cmd == "STOP":
            self.controller.goal_position = None
        elif cmd == "RESET":
            self.controller.reset()

    def _drone_graph_cb(self, msg, drone_id):
        try:
            self.drone_graphs[drone_id] = TopologicalGraph.from_ros_msg(msg)
            self.last_graph_times[drone_id] = rospy.Time.now()
        except Exception as e:
            rospy.logwarn(f"[{self.robot_id}] Graph rx error from {drone_id}: {e}")

    # ════════════════════════ Coordinate helper ═══════════════════════

    @property
    def current_position_map(self):
        """Ground robot position in the shared map frame (≈ Gazebo world frame).

        libgazebo_ros_diff_drive (legacyMode=false) initialises its odom frame
        at the Gazebo world origin, so odom positions are already world-frame.
        spawn_offset is (0, 0); the addition is kept for API consistency.
        """
        if self.current_position is None:
            return None
        return (
            self.current_position[0] + self.spawn_offset[0],
            self.current_position[1] + self.spawn_offset[1],
        )

    # ════════════════════════ LiDAR processing ═══════════════════════

    def _pointcloud_to_occupancy_map(self) -> np.ndarray:
        """
        Convert a sensor_msgs/PointCloud2 (from libgazebo_ros_block_laser) to a
        2D occupancy grid in the robot's local body frame.

        Points are in ground_0/lidar_link frame (x+ forward, y+ left, z+ up).
        Height filter keeps only obstacle-level returns:
          z < Z_MIN  → ground reflections          (ignore)
          z > Z_MAX  → tree canopy / above robot   (ignore)
          else       → tree trunk / wall / obstacle (mark occupied)

        Coordinate convention (matches LocalPlanner._avoid_obstacles):
          grid_row = half - x / res    x+ forward → row decreases (up in grid)
          grid_col = half + y / res    y+ left    → col increases

        A 1-cell binary dilation converts sparse beam hits into a solid
        safety margin around each obstacle.
        """
        resolution   = self.controller.resolution        # m/pixel (default 0.5)
        perception_m = self.controller.perception_range  # metres  (default 8.0)
        map_size     = int(perception_m / resolution)
        occupancy_map = np.zeros((map_size, map_size), dtype=np.uint8)

        if self.current_pointcloud is None:
            return occupancy_map

        Z_MIN =  0.10   # ignore ground returns  (lidar is ~0.28 m above ground)
        Z_MAX =  1.80   # ignore canopy / sky returns above navigable space

        try:
            raw = list(pc2.read_points(
                self.current_pointcloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
            ))
            if not raw:
                return occupancy_map

            pts = np.array(raw, dtype=np.float32)   # (N, 3)

            # Height filter — keep only obstacle-height returns
            mask = (pts[:, 2] >= Z_MIN) & (pts[:, 2] <= Z_MAX)
            pts  = pts[mask]
            if len(pts) == 0:
                return occupancy_map

            # Vectorised projection onto 2-D grid
            half = map_size // 2
            rows = (half - pts[:, 0] / resolution).astype(int)
            cols = (half + pts[:, 1] / resolution).astype(int)

            in_bounds = (rows >= 0) & (rows < map_size) & \
                        (cols >= 0) & (cols < map_size)
            occupancy_map[rows[in_bounds], cols[in_bounds]] = 1

            # 1-cell dilation: sparse lidar returns become a solid safety margin
            occupancy_map = binary_dilation(occupancy_map).astype(np.uint8)

        except Exception as e:
            rospy.logwarn_throttle(5.0,
                f"[{self.robot_id}] PointCloud2 parse error: {e}")

        return occupancy_map

    # ════════════════════════ Control loop (10 Hz) ═══════════════════

    def _control_loop(self, _event):
        if self.current_position is None:
            rospy.logwarn_throttle(5.0, f"[{self.robot_id}] No odometry yet — waiting")
            return

        try:
            obs = self._build_observation()
            result = self.controller.get_action(obs)

            twist = Twist()
            motion = result.get('motion', (0.0, 0.0))
            twist.linear.x = np.clip(motion[0], -self.max_linear_vel, self.max_linear_vel)
            twist.angular.z = np.clip(motion[1], -self.max_angular_vel, self.max_angular_vel)
            self.cmd_vel_pub.publish(twist)

            # ── Diagnostic logging (throttled) ─────────────────────────
            goal = self.controller.goal_position
            path = self.controller.global_planner.current_path
            wp   = self.controller.global_planner.get_current_waypoint()
            heading_deg = np.degrees(self.current_heading) if self.current_heading is not None else float('nan')
            desired_heading_deg = float('nan')
            if wp is not None and self.current_position is not None:
                dx = wp[0] - self.current_position[0]
                dy = wp[1] - self.current_position[1]
                desired_heading_deg = np.degrees(np.arctan2(dy, dx))
            rospy.loginfo_throttle(3.0,
                f"[{self.robot_id}] "
                f"pos=({self.current_position[0]:.1f},{self.current_position[1]:.1f}) "
                f"hdg={heading_deg:.1f}deg desired={desired_heading_deg:.1f}deg "
                f"goal={'NONE' if goal is None else f'({goal[0]:.1f},{goal[1]:.1f})'} "
                f"path_wps={0 if path is None else len(path)} "
                f"wp={'NONE' if wp is None else f'({wp[0]:.1f},{wp[1]:.1f})'} "
                f"cmd=(lin={twist.linear.x:.2f} ang={twist.angular.z:.2f})"
            )

            # ── Publish planned path for RViz ───────────────────────────
            if path is not None:
                path_msg = Path()
                path_msg.header.stamp = rospy.Time.now()
                path_msg.header.frame_id = 'map'
                for wp_pos in path:
                    ps = PoseStamped()
                    ps.header = path_msg.header
                    ps.pose.position.x = float(wp_pos[0])
                    ps.pose.position.y = float(wp_pos[1])
                    ps.pose.position.z = 1.1
                    ps.pose.orientation.w = 1.0
                    path_msg.poses.append(ps)
                self.planned_path_pub.publish(path_msg)

        except Exception as e:
            rospy.logerr(f"[{self.robot_id}] Control error: {e}\n{traceback.format_exc()}")
            self.cmd_vel_pub.publish(Twist())

    def _build_observation(self):
        """Assemble simulator-compatible dict from ROS topic data.

        All positions are in the shared map frame (≈ Gazebo world frame).
        The occupancy_map is built from the live Velodyne scan in the robot's
        local body frame so LocalPlanner._avoid_obstacles() can use it directly.
        """
        pos_map = self.current_position_map
        neighbors = []
        now = rospy.Time.now()

        for did, graph in self.drone_graphs.items():
            lt = self.last_graph_times.get(did)
            if lt and (now - lt).to_sec() < self.comm_timeout:
                neighbors.append({
                    'type': 'drone',
                    'broadcast': {
                        'drone_id': did,
                        'position': [0.0, 0.0, 0.0],  # drone position unknown here
                        'graph': graph.to_message(),
                    },
                })

        # Build real occupancy grid from PointCloud2 (robot-local frame).
        # Falls back to zeros silently when no scan has arrived yet.
        occupancy_map = self._pointcloud_to_occupancy_map()

        return {
            'self': {
                'position': np.array([pos_map[0], pos_map[1], 0.0]),
                'orientation': self.current_heading,
                'type': 'diff_drive',
            },
            'occupancy_map': occupancy_map,
            'neighbors': neighbors,
        }

    # ════════════════════════ Mission broadcast ══════════════════════

    def _mission_loop(self, _event):
        m = MissionBroadcast()
        m.header.stamp = rospy.Time.now()
        m.robot_id = self.robot_id
        pos_map = self.current_position_map
        if pos_map:
            m.position_x, m.position_y = pos_map  # map frame for drones to use
        if self.controller.goal_position:
            m.target_x, m.target_y = self.controller.goal_position
            m.has_target = True
        else:
            m.has_target = False
        self.mission_pub.publish(m)

    # ════════════════════════ Heartbeat & Status ═════════════════════

    def _heartbeat_loop(self, _event):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.robot_id
        self.heartbeat_pub.publish(h)

    def _status_loop(self, _event):
        s = RobotStatus()
        s.header.stamp = rospy.Time.now()
        s.robot_id = self.robot_id
        s.robot_type = "ground"
        s.flight_state = "N/A"
        s.mission_phase = ("navigating" if self.controller.goal_position
                           else "idle")
        pos_map = self.current_position_map
        if pos_map:
            s.position_x, s.position_y = pos_map  # map frame
            s.position_z = 0.0
        if self.controller.unified_graph:
            s.graph_nodes = len(self.controller.unified_graph)
            s.graph_edges = self.controller.unified_graph.graph.number_of_edges()
        now = rospy.Time.now()
        s.comm_ok = (self.last_odom_time is not None and
                     (now - self.last_odom_time).to_sec() < 2.0)
        self.status_pub.publish(s)

    def _publish_unified_graph(self, _event):
        """Publish the merged unified graph as TopologicalGraphMsg for GCS to visualize."""
        graph = self.controller.unified_graph
        if graph is None or len(graph) == 0:
            return
        try:
            msg = graph.to_ros_msg()
            msg.header.stamp = rospy.Time.now()
            self.unified_graph_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(10.0,
                f"[{self.robot_id}] unified_graph publish error: {e}")


if __name__ == '__main__':
    try:
        GroundROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
