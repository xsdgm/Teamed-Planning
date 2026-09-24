"""
Drone Local Planner
Waypoint tracker with reactive inter-drone collision avoidance.

Called by drone_node.py at the control rate (default 50 Hz).
It is NOT a separate ROS node — drone_node.py is the sole MAVROS publisher.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class DroneLocalPlanner:
    """
    Local planner that converts algorithm target waypoints into position
    setpoints for PX4, with a repulsive field that deflects the setpoint
    away from nearby drones at every control tick.
    """

    def __init__(self, max_speed: float = 5.0,
                 arrival_threshold: float = 0.5,
                 step_distance: float = 1.0,
                 safety_radius: float = 8.0,
                 repulsion_gain: float = 3.0):
        """
        Args:
            max_speed: Maximum speed in m/s
            arrival_threshold: Distance to consider "arrived" (m)
            step_distance: Max single-step displacement (m)
            safety_radius: Start repulsion when another drone is within
                           this distance (m). Should be >= physical safety
                           buffer.
            repulsion_gain: Strength of the repulsive deflection.
        """
        self.max_speed = max_speed
        self.arrival_threshold = arrival_threshold
        self.step_distance = step_distance
        self.safety_radius = safety_radius
        self.repulsion_gain = repulsion_gain
        self.current_target = None

    def compute(self, current_position: Tuple[float, float, float],
                target: Optional[Tuple[float, float, float]] = None,
                pointcloud=None,
                other_drone_positions: Optional[Dict[str, tuple]] = None,
                ) -> Tuple[float, float, float]:
        """
        Compute next position setpoint with inter-drone avoidance.

        Args:
            current_position: Current (x, y, z) in local frame
            target: Desired target (x, y, z); if None, uses last known
            pointcloud: Reserved for future obstacle avoidance
            other_drone_positions: {drone_id: (x, y, z)} of neighbours,
                in the same frame as current_position

        Returns:
            (x, y, z) position setpoint for PX4
        """
        if target is not None:
            self.current_target = target

        if self.current_target is None:
            return current_position

        setpoint = np.array(self.current_target, dtype=float)

        # Reactive repulsion from nearby drones
        if other_drone_positions:
            cur = np.array(current_position, dtype=float)
            repulsion = np.zeros(3)
            for _did, pos in other_drone_positions.items():
                other = np.array(pos, dtype=float)
                diff = cur - other          # vector pointing away from other
                dist = float(np.linalg.norm(diff))
                if 0.1 < dist < self.safety_radius:
                    # Inverse-distance repulsive force (stronger when closer)
                    strength = self.repulsion_gain * (
                        1.0 / dist - 1.0 / self.safety_radius)
                    repulsion += strength * (diff / dist)
            setpoint += repulsion

        dx = setpoint[0] - current_position[0]
        dy = setpoint[1] - current_position[1]
        dz = setpoint[2] - current_position[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < self.arrival_threshold:
            return tuple(setpoint)

        return (float(setpoint[0]), float(setpoint[1]), float(setpoint[2]))
