#!/usr/bin/env python3
"""
Drone ROS Node
Wraps DroneController + DroneLocalPlanner with a flight state machine.
Single node per drone — sole MAVROS publisher.
"""

import rospy
import numpy as np
import traceback
import threading
from enum import Enum

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from mavros_msgs.srv import CommandBool, SetMode, ParamSet
from mavros_msgs.msg import ParamValue

from teamed_planning.msg import (
    TopologicalGraphMsg, MissionBroadcast,
    RobotCommand, RobotStatus,
)

import sys
import os
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from drone.drone_main import DroneController
from drone.drone_local_planner import DroneLocalPlanner
from drone.evidence_grid import EvidenceGrid
from common.graph_structure import TopologicalGraph


class FlightState(Enum):
    IDLE = "IDLE"
    ARMING = "ARMING"
    TAKING_OFF = "TAKING_OFF"
    HOVERING = "HOVERING"
    MISSION = "MISSION"
    RETURNING = "RETURNING"
    LANDING = "LANDING"


class DroneROSNode:
    def __init__(self):
        rospy.init_node('drone_node', anonymous=False)

        # ── Parameters ──────────────────────────────────────────────────
        self.drone_id = rospy.get_param('~drone_id', 'drone_0')
        self.communication_range = rospy.get_param('~communication_range', 30.0)
        self.max_speed = rospy.get_param('~max_speed', 5.0)
        self.height_limit = rospy.get_param('~height_limit', 30.0)
        self.comm_timeout = rospy.get_param('~comm_timeout', 10.0)
        self.algorithm_rate = rospy.get_param('~algorithm_rate', 10.0)
        self.control_rate = rospy.get_param('~control_rate', 50.0)
        self.num_drones = rospy.get_param('~num_drones', 3)
        self.num_ground_robots = rospy.get_param('~num_ground_robots', 1)

        # Staggered takeoff height: drone_i flies at base + i * stagger.
        # Edit config/algorithm/drone.yaml → takeoff_height / takeoff_height_stagger.
        drone_num   = int(rospy.get_param('~drone_num', 0))
        base_height = float(rospy.get_param('~takeoff_height', 5.0))
        stagger     = float(rospy.get_param('~takeoff_height_stagger', 3.0))
        self.takeoff_height = base_height + drone_num * stagger

        # Spawn position in Gazebo world (≈ map frame origin per drone).
        # MAVROS local_position/odom has its origin at the drone's home (spawn)
        # position. Adding spawn_offset converts local → map frame (≈ Gazebo world).
        self.spawn_offset = (
            float(rospy.get_param('~spawn_x', 0.0)),
            float(rospy.get_param('~spawn_y', 0.0)),
        )

        # Camera HFOV — still used by _altitude_map_size() to compute footprint bounds.
        self._camera_hfov = rospy.get_param('~camera_hfov', 2.0944)   # 120 deg default

        # ── State ───────────────────────────────────────────────────────
        self.flight_state = FlightState.IDLE
        self.home_position = None        # MAVROS local frame
        self.current_position = None     # MAVROS local frame (used for flight control)
        self.current_orientation = None
        self.current_target = None       # map frame (from algorithm)
        self.hover_position_local = None # local frame, frozen on entering HOVERING
        self.last_odom_time = None
        self.current_pointcloud = None
        # Camera aspect ratio (occ_h/occ_w); updated from first seg_occ message.
        self._seg_aspect = 720.0 / 1280.0   # default 16:9 landscape
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

        # Communication buffers
        self.other_drone_graphs = {}       # {drone_id: TopologicalGraph}
        self.other_drone_positions = {}    # {drone_id: (x, y, z)} map-frame
        self.other_drone_waypoints = {}    # {drone_id: (x, y, z)}
        self.ground_robot_missions = {}    # {robot_id: dict}
        self.ground_robot_paths = {}       # {robot_id: [(x,y), ...]}
        self.last_graph_times = {}
        self.last_mission_times = {}

        # ── Algorithm ───────────────────────────────────────────────────
        self.controller = DroneController(
            drone_id=self.drone_id,
            communication_range=self.communication_range,
            max_speed=self.max_speed,
            total_drones=self.num_drones,
        )
        self.local_planner = DroneLocalPlanner(
            max_speed=self.max_speed,
            step_distance=self.max_speed / self.control_rate,
        )

        # ── Evidence grid — world-frame temporal vote accumulation ───────
        # Accumulates obstacle/free evidence across multiple segmentation
        # frames so transient misclassifications are filtered out before the
        # graph builder sees the map.
        _ev_world_size = float(rospy.get_param('~ev_world_size', 300.0))
        _ev_origin     = -_ev_world_size / 2.0
        self._evidence_grid = EvidenceGrid(
            resolution=self.controller.resolution,
            world_size=_ev_world_size,
            origin=(_ev_origin, _ev_origin),
        )
        # Lock protects evidence_grid between _seg_occ_cb (subscriber thread)
        # and _build_observation (timer thread).
        self._ev_lock = threading.Lock()
        rospy.loginfo(f"[{self.drone_id}] EvidenceGrid: "
                      f"{_ev_world_size}m × {_ev_world_size}m @ "
                      f"{self.controller.resolution}m/cell")

        # ── Publishers ──────────────────────────────────────────────────
        self.setpoint_pub = rospy.Publisher(
            'mavros_setpoint', PoseStamped, queue_size=10)
        self.graph_pub = rospy.Publisher(
            'graph_broadcast', TopologicalGraphMsg, queue_size=1)
        # Debug: smoothed occupancy grid (evidence grid result) in world frame.
        # Compare against ground robot LiDAR in RViz to verify obstacle detection.
        self.occ_grid_pub = rospy.Publisher(
            'occ_grid', OccupancyGrid, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            'status', RobotStatus, queue_size=1)
        self.heartbeat_pub = rospy.Publisher(
            'heartbeat', Header, queue_size=1)
        self.waypoint_pub = rospy.Publisher(
            'proposed_waypoint', PoseStamped, queue_size=1)

        # ── Subscribers ─────────────────────────────────────────────────
        rospy.Subscriber('odom', Odometry, self._odom_cb)
        rospy.Subscriber('points', PointCloud2, self._pointcloud_cb)
        rospy.Subscriber('command', RobotCommand, self._command_cb)
        # seg_occ: raw per-frame occupancy published by seg_node.
        # Remapped in launch file to the correct namespaced topic.
        rospy.Subscriber('seg_occ', OccupancyGrid, self._seg_occ_cb, queue_size=1)

        for i in range(self.num_drones):
            did = f"drone_{i}"
            if did != self.drone_id:
                rospy.Subscriber(
                    f"/{did}/graph_broadcast", TopologicalGraphMsg,
                    self._drone_graph_cb, callback_args=did)
                rospy.Subscriber(
                    f"/{did}/status", RobotStatus,
                    self._drone_status_cb, callback_args=did)
                rospy.Subscriber(
                    f"/{did}/proposed_waypoint", PoseStamped,
                    self._drone_waypoint_cb, callback_args=did)

        for i in range(self.num_ground_robots):
            rid = f"ground_{i}"
            rospy.Subscriber(
                f"/{rid}/mission_broadcast", MissionBroadcast,
                self._mission_cb, callback_args=rid)
            rospy.Subscriber(
                f"/{rid}/planned_path", Path,
                self._planned_path_cb, callback_args=rid)

        # ── MAVROS services ─────────────────────────────────────────────
        arming_srv = rospy.get_param('~arming_service', 'mavros/cmd/arming')
        mode_srv = rospy.get_param('~set_mode_service', 'mavros/set_mode')
        param_srv = rospy.get_param('~param_service', 'mavros/param/set')

        rospy.loginfo(f"[{self.drone_id}] Waiting for MAVROS services...")
        try:
            rospy.wait_for_service(arming_srv, timeout=10.0)
            rospy.wait_for_service(mode_srv, timeout=10.0)
            rospy.wait_for_service(param_srv, timeout=10.0)
            self.arming_client = rospy.ServiceProxy(arming_srv, CommandBool)
            self.set_mode_client = rospy.ServiceProxy(mode_srv, SetMode)
            self._param_client = rospy.ServiceProxy(param_srv, ParamSet)
            rospy.loginfo(f"[{self.drone_id}] MAVROS services connected")

            # Disable RC loss failsafe (required for simulation without RC controller).
            # COM_RCL_EXCEPT bit 2 = OFFBOARD mode exception.
            self._param_client("COM_RCL_EXCEPT", ParamValue(4, 0.0))
            rospy.loginfo(f"[{self.drone_id}] COM_RCL_EXCEPT set (RC failsafe disabled)")

            # Set maximum horizontal velocity so PX4 flies at configured max_speed.
            self._param_client("MPC_XY_VEL_MAX", ParamValue(0, float(self.max_speed)))
            rospy.loginfo(f"[{self.drone_id}] MPC_XY_VEL_MAX set to {self.max_speed} m/s")
        except rospy.ROSException:
            rospy.logwarn(f"[{self.drone_id}] MAVROS services not available — sim-only mode")
            self.arming_client = None
            self.set_mode_client = None
            self._param_client = None
        except Exception as e:
            rospy.logwarn(f"[{self.drone_id}] MAVROS param set failed: {e}")
            self.arming_client = rospy.ServiceProxy(arming_srv, CommandBool)
            self.set_mode_client = rospy.ServiceProxy(mode_srv, SetMode)
            self._param_client = None 

        # ── Timers ──────────────────────────────────────────────────────
        self._ctrl_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._control_loop)
        self._algo_timer = rospy.Timer(
            rospy.Duration(1.0 / self.algorithm_rate), self._algorithm_loop)
        self._hb_timer = rospy.Timer(
            rospy.Duration(1.0), self._heartbeat_loop)
        self._st_timer = rospy.Timer(
            rospy.Duration(0.5), self._status_loop)

        rospy.loginfo(f"[{self.drone_id}] Drone node ready, state={self.flight_state.value}")

    # ════════════════════════ Callbacks ═══════════════════════════════

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.current_position = (p.x, p.y, p.z)
        o = msg.pose.pose.orientation
        self.current_orientation = (o.x, o.y, o.z, o.w)
        self.last_odom_time = rospy.Time.now()
        if self.home_position is None:
            self.home_position = self.current_position
            rospy.loginfo(f"[{self.drone_id}] Home position: {self.home_position}")

    def _pointcloud_cb(self, msg):
        self.current_pointcloud = msg

    def _seg_occ_cb(self, msg: OccupancyGrid):
        """Receive raw per-frame occupancy from seg_node and update evidence grid.

        Encoding (set by seg_node._publish_seg_occ — rows→X, cols→Y, direct row-major):
          msg.info.height = occ_h  (world-X cells = rows)
          msg.info.width  = occ_w  (world-Y cells = cols)
          origin.(x,y) = (min_x, min_y)
          data[row*occ_w + col] = occ[row][col] * 100
        """
        try:
            res   = msg.info.resolution
            occ_h = msg.info.height
            occ_w = msg.info.width
            min_x = msg.info.origin.position.x
            min_y = msg.info.origin.position.y
            bounds = (min_x, min_y,
                      min_x + occ_h * res,
                      min_y + occ_w * res)
            occ = (np.array(msg.data, dtype=np.float32)
                   .reshape(occ_h, occ_w) > 50).astype(np.uint8)
            # Update stored aspect ratio for _altitude_map_size()
            if occ_w > 0:
                self._seg_aspect = occ_h / occ_w
            with self._ev_lock:
                self._evidence_grid.update(occ, bounds)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[{self.drone_id}] seg_occ decode error: {e}")

    def _command_cb(self, msg):
        tgt = msg.target_robot_id
        if tgt and tgt != self.drone_id and tgt != "ALL":
            return
        cmd = msg.command.upper()
        rospy.loginfo(f"[{self.drone_id}] Command: {cmd}")
        if cmd == "TAKEOFF":
            self._handle_takeoff(msg.param1 if msg.param1 > 0 else self.takeoff_height)
        elif cmd == "LAND":
            self._transition(FlightState.LANDING)
        elif cmd == "RTH":
            self._transition(FlightState.RETURNING)
        elif cmd == "START_MISSION":
            if self.flight_state == FlightState.HOVERING:
                self._transition(FlightState.MISSION)
        elif cmd == "HOVER":
            if self.flight_state == FlightState.MISSION:
                self._transition(FlightState.HOVERING)

    def _drone_graph_cb(self, msg, drone_id):
        try:
            self.other_drone_graphs[drone_id] = TopologicalGraph.from_ros_msg(msg)
            self.last_graph_times[drone_id] = rospy.Time.now()
        except Exception as e:
            rospy.logwarn(f"[{self.drone_id}] Graph rx error from {drone_id}: {e}")

    def _mission_cb(self, msg, robot_id):
        self.ground_robot_missions[robot_id] = {
            'position': (msg.position_x, msg.position_y),
            'target': (msg.target_x, msg.target_y) if msg.has_target else None,
        }
        self.last_mission_times[robot_id] = rospy.Time.now()

    def _planned_path_cb(self, msg, robot_id):
        """Store the ground robot's latest planned path as (x, y) waypoints."""
        self.ground_robot_paths[robot_id] = [
            (ps.pose.position.x, ps.pose.position.y)
            for ps in msg.poses
        ]

    def _drone_status_cb(self, msg, drone_id):
        """Store another drone's map-frame position from its status topic."""
        self.other_drone_positions[drone_id] = (
            msg.position_x, msg.position_y, msg.position_z)

    def _drone_waypoint_cb(self, msg, drone_id):
        """Store another drone's proposed waypoint for C-ADMM consensus."""
        self.other_drone_waypoints[drone_id] = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z)

    # ════════════════════════ State machine ══════════════════════════

    def _transition(self, new_state):
        old = self.flight_state
        self.flight_state = new_state
        # Freeze a fixed local-frame setpoint when entering HOVERING so EKF
        # drift in the odom topic doesn't cause the drone to keep climbing.
        if new_state == FlightState.HOVERING and self.current_position is not None:
            self.hover_position_local = self.current_position
            rospy.loginfo(f"[{self.drone_id}] Hover locked at local "
                          f"z={self.hover_position_local[2]:.2f} m")
        rospy.loginfo(f"[{self.drone_id}] {old.value} -> {new_state.value}")

    def _handle_takeoff(self, height):
        if self.flight_state != FlightState.IDLE:
            rospy.logwarn(f"[{self.drone_id}] Cannot takeoff from {self.flight_state.value}")
            return
        if self.current_position is None:
            rospy.logwarn(f"[{self.drone_id}] No odom yet — EKF2 may not have converged. "
                          f"Retry takeoff in a few seconds.")
            return
        self.takeoff_height = height
        self._transition(FlightState.ARMING)

        # PX4 OFFBOARD requirement: stream setpoints for ~2s before switching mode.
        # _control_loop publishes current position in ARMING state to establish the stream.
        rospy.loginfo(f"[{self.drone_id}] Pre-streaming setpoints for OFFBOARD...")
        rospy.sleep(2.0)

        offboard_ok = False
        armed_ok = False

        if self.set_mode_client:
            try:
                resp = self.set_mode_client(custom_mode="OFFBOARD")
                if resp.mode_sent:
                    rospy.loginfo(f"[{self.drone_id}] OFFBOARD mode set")
                    offboard_ok = True
                else:
                    rospy.logwarn(f"[{self.drone_id}] OFFBOARD mode rejected by PX4")
            except Exception as e:
                rospy.logwarn(f"[{self.drone_id}] SetMode service failed: {e}")
        else:
            # No set_mode client (sim bypass) — treat as success
            offboard_ok = True

        if offboard_ok and self.arming_client:
            try:
                resp = self.arming_client(True)
                if resp.success:
                    rospy.loginfo(f"[{self.drone_id}] Armed")
                    armed_ok = True
                else:
                    rospy.logwarn(f"[{self.drone_id}] Arming rejected by PX4")
            except Exception as e:
                rospy.logwarn(f"[{self.drone_id}] Arming service failed: {e}")
        elif not self.arming_client:
            armed_ok = True

        if offboard_ok and armed_ok:
            self._transition(FlightState.TAKING_OFF)
        else:
            rospy.logerr(f"[{self.drone_id}] Takeoff aborted "
                         f"(OFFBOARD={'OK' if offboard_ok else 'FAIL'}, "
                         f"ARM={'OK' if armed_ok else 'FAIL'}) — returning to IDLE")
            self._transition(FlightState.IDLE)

    # ════════════════════════ Control loop (50 Hz) ═══════════════════

    def _control_loop(self, _event):
        if self.current_position is None:
            return

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        # IDLE and ARMING: publish current position so the setpoint stream is
        # established before PX4 accepts the OFFBOARD mode switch.
        if self.flight_state in (FlightState.IDLE, FlightState.ARMING):
            pose.pose.position.x = self.current_position[0]
            pose.pose.position.y = self.current_position[1]
            pose.pose.position.z = self.current_position[2]
            self.setpoint_pub.publish(pose)
            return

        if self.flight_state == FlightState.TAKING_OFF:
            tgt = (self.home_position[0],
                   self.home_position[1],
                   self.home_position[2] + self.takeoff_height)
            pos = self.local_planner.compute(self.current_position, tgt)
            if abs(self.current_position[2] - tgt[2]) < 0.3:
                self._transition(FlightState.HOVERING)

        elif self.flight_state == FlightState.HOVERING:
            # Use the frozen local-frame position (captured at TAKING_OFF→HOVERING
            # transition) so EKF/GPS drift doesn't cause a continuous climb.
            pos = (self.hover_position_local
                   if self.hover_position_local is not None
                   else self.current_position)

        elif self.flight_state == FlightState.MISSION:
            if self.current_target is not None:
                # current_target arrives in map frame from the algorithm.
                # Convert to MAVROS local frame before sending as setpoint.
                target_local = (
                    self.current_target[0] - self.spawn_offset[0],
                    self.current_target[1] - self.spawn_offset[1],
                    self.current_target[2],
                )
                # Convert other drones' map-frame positions to local frame
                neighbours_local = {}
                for did, mpos in self.other_drone_positions.items():
                    neighbours_local[did] = (
                        mpos[0] - self.spawn_offset[0],
                        mpos[1] - self.spawn_offset[1],
                        mpos[2],
                    )
                raw = self.local_planner.compute(
                    self.current_position, target_local,
                    self.current_pointcloud, neighbours_local)
                pos = self._safety_clamp(raw)
            else:
                pos = (self.hover_position_local
                       if self.hover_position_local is not None
                       else self.current_position)

        elif self.flight_state == FlightState.RETURNING:
            if self.home_position:
                tgt = (self.home_position[0], self.home_position[1],
                       self.home_position[2] + self.takeoff_height)
                pos = self.local_planner.compute(self.current_position, tgt)
                dxy = np.sqrt((self.current_position[0] - self.home_position[0])**2 +
                              (self.current_position[1] - self.home_position[1])**2)
                if dxy < 1.0:
                    self._transition(FlightState.LANDING)
            else:
                pos = self.current_position

        elif self.flight_state == FlightState.LANDING:
            if self.home_position:
                tgt = self.home_position
                pos = self.local_planner.compute(self.current_position, tgt)
                if abs(self.current_position[2] - self.home_position[2]) < 0.3:
                    if self.arming_client:
                        try:
                            self.arming_client(False)
                        except Exception:
                            pass
                    self._transition(FlightState.IDLE)
            else:
                pos = self.current_position
        else:
            pos = self.current_position

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        self.setpoint_pub.publish(pose)

    # ════════════════════════ Coordinate helpers ══════════════════════

    @property
    def current_position_map(self):
        """Drone position in the shared map frame (≈ Gazebo world frame).

        MAVROS local_position/odom has its origin at the drone's home position
        (where it spawned).  Adding spawn_offset brings it into the common map
        frame that all robots share so the planning algorithm, graph nodes, and
        RViz markers are all in the same coordinate system.
        """
        if self.current_position is None:
            return None
        return (
            self.current_position[0] + self.spawn_offset[0],
            self.current_position[1] + self.spawn_offset[1],
            self.current_position[2],
        )

    # ════════════════════════ Algorithm loop (10 Hz) ═════════════════

    def _algorithm_loop(self, _event):
        if self.flight_state != FlightState.MISSION:
            return
        if self.current_position is None:
            return
        try:
            obs = self._build_observation()
            result = self.controller.get_action(obs, mode='real')
            self.current_target = result.get('target_waypoint')

            if self.current_target is not None:
                wp_msg = PoseStamped()
                wp_msg.header.stamp = rospy.Time.now()
                wp_msg.header.frame_id = 'map'
                wp_msg.pose.position.x = self.current_target[0]
                wp_msg.pose.position.y = self.current_target[1]
                wp_msg.pose.position.z = self.current_target[2]
                wp_msg.pose.orientation.w = 1.0
                self.waypoint_pub.publish(wp_msg)

            if self.controller.current_graph is not None:
                try:
                    self.graph_pub.publish(
                        self.controller.current_graph.to_ros_msg())
                except Exception as e:
                    rospy.logwarn(f"[{self.drone_id}] Graph publish: {e}")

            self.consecutive_errors = 0
        except Exception as e:
            self.consecutive_errors += 1
            rospy.logerr(f"[{self.drone_id}] Algorithm error "
                         f"({self.consecutive_errors}): {e}\n{traceback.format_exc()}")
            if self.consecutive_errors >= self.max_consecutive_errors:
                rospy.logerr(f"[{self.drone_id}] Max errors — hover")
                self._transition(FlightState.HOVERING)
                self.consecutive_errors = 0

    def _build_observation(self):
        """Assemble simulator-compatible dict from ROS topic data.

        All positions are in the shared map frame (≈ Gazebo world frame).
        """
        pos_map = self.current_position_map
        neighbors = []
        now = rospy.Time.now()

        for did, graph in self.other_drone_graphs.items():
            lt = self.last_graph_times.get(did)
            if lt and (now - lt).to_sec() < self.comm_timeout:
                real_pos = self.other_drone_positions.get(did)
                neighbors.append({
                    'type': 'drone',
                    'broadcast': {
                        'drone_id': did,
                        'position': list(real_pos) if real_pos else list(pos_map),
                        'graph': graph.to_message(),
                    },
                })

        for rid, mission in self.ground_robot_missions.items():
            lt = self.last_mission_times.get(rid)
            if lt and (now - lt).to_sec() < self.comm_timeout:
                neighbors.append({
                    'type': 'diff_drive',
                    'broadcast': {
                        'robot_id': rid,
                        'position': list(mission['position']),
                        'target': list(mission['target']) if mission['target'] else None,
                        # Planned path waypoints for the uncertainty dispatcher
                        'planned_path': self.ground_robot_paths.get(rid, []),
                    },
                })

        # ── Read smoothed occupancy from the evidence grid ───────────────
        # seg_node runs inference and calls evidence_grid.update() via _seg_occ_cb.
        # Here we only read the latest smoothed result — takes < 1 ms.
        fallback_h, fallback_w = self._altitude_map_size()
        res    = self.controller.resolution
        half_x = fallback_h * res / 2.0   # rows → world X span
        half_y = fallback_w * res / 2.0   # cols → world Y span
        # bounds = (min_x, min_y, max_x, max_y) — order required by EvidenceGrid
        _bounds = (pos_map[0] - half_x, pos_map[1] - half_y,
                   pos_map[0] + half_x, pos_map[1] + half_y)
        with self._ev_lock:
            smoothed_occ = self._evidence_grid.get_occ(
                _bounds, (fallback_h, fallback_w), unobserved_as_free=True)

        # ── Publish smoothed occupancy grid for RViz debug ───────────────
        # Shows the accumulated evidence result — compare against LiDAR to
        # verify obstacle detection quality.
        try:
            grid = OccupancyGrid()
            grid.header.stamp = rospy.Time.now()
            grid.header.frame_id = 'map'
            grid.info.resolution = res
            grid.info.width  = fallback_h   # X cells (rows of occ)
            grid.info.height = fallback_w   # Y cells (cols of occ)
            grid.info.origin.position.x = _bounds[0]   # min_x
            grid.info.origin.position.y = _bounds[1]   # min_y
            grid.info.origin.position.z = 0.0
            grid.info.origin.orientation.w = 1.0
            # flipud + T aligns occ axes (rows→X, cols→Y) with OccupancyGrid axes
            occ_disp = np.flipud(smoothed_occ).T
            grid.data = (occ_disp.astype(np.int8) * 100).flatten(order='C').tolist()
            self.occ_grid_pub.publish(grid)
        except Exception:
            pass

        return {
            'self': {
                'position': np.array([pos_map[0], pos_map[1]]),
                'height': pos_map[2],
                'type': 'drone',
            },
            'occupancy_map': smoothed_occ,
            'neighbors': neighbors,
            'neighbor_waypoints': dict(self.other_drone_waypoints),
        }

    def _altitude_map_size(self) -> tuple:
        """Compute occupancy map (height, width) in pixels from altitude and HFOV.

        Returns (map_h, map_w) = (occ_rows, occ_cols), i.e. (X-cells, Y-cells).
        Uses aspect ratio from the latest seg_occ message (updated in _seg_occ_cb)
        so the bounds passed to the evidence grid match seg_node's actual output.
        """
        import math
        alt = self.current_position[2] if self.current_position else self.takeoff_height
        alt = max(alt, 1.0)
        res      = self.controller.resolution
        ground_w = 2.0 * alt * math.tan(self._camera_hfov / 2.0)
        ground_h = ground_w * self._seg_aspect   # updated from seg_occ messages
        map_w = max(4, int(round(ground_w / res)))
        map_h = max(4, int(round(ground_h / res)))
        return map_h, map_w

    # ════════════════════════ Safety ═════════════════════════════════

    def _safety_clamp(self, position):
        x, y, z = position
        # Only clamp altitude — let PX4's own velocity controller (MPC_XY_VEL_MAX)
        # govern XY speed. Micro-stepping XY here kept drones at ~0.2 m/s.
        z = max(0.5, min(z, self.height_limit))
        return (x, y, z)

    # ════════════════════════ Heartbeat & Status ═════════════════════

    def _heartbeat_loop(self, _event):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.drone_id
        self.heartbeat_pub.publish(h)

    def _status_loop(self, _event):
        s = RobotStatus()
        s.header.stamp = rospy.Time.now()
        s.robot_id = self.drone_id
        s.robot_type = "drone"
        s.flight_state = self.flight_state.value
        s.mission_phase = (self.controller.navigator.phase.value
                           if hasattr(self.controller, 'navigator') else "")
        pos_map = self.current_position_map
        if pos_map:
            s.position_x, s.position_y, s.position_z = pos_map
        if self.controller.current_graph:
            s.graph_nodes = len(self.controller.current_graph)
            s.graph_edges = self.controller.current_graph.graph.number_of_edges()
        now = rospy.Time.now()
        s.comm_ok = (self.last_odom_time is not None and
                     (now - self.last_odom_time).to_sec() < 2.0)
        self.status_pub.publish(s)


if __name__ == '__main__':
    try:
        DroneROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
