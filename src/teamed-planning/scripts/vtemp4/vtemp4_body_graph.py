"""Express the full decision graph in the body axes of the vehicle at the current review."""
import numpy as np
import torch

from vtemp4.vtemp4_body_angle import BodyAngleDistribution
from vtemp4.vtemp4_vehicle_graph import VehicleGraphHistoryLSTMEpisode, VehicleGraphHistoryLSTMPolicy


def body_frame_observation(observation):
    """Rotate measured offsets to forward/left, keeping topology, history order and sampler arcs."""
    own = observation['vehicle']
    sine, cosine = own[4:6]
    # Right-multiply row vectors by this matrix to express them in the current vehicle frame.
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)

    def bearing_distance(target):
        # The observation builder already scales these offsets in units of 350 metres.
        forward, left = target[2:4] @ rotation
        distance = np.hypot(forward, left)
        if distance == 0.:
            return np.zeros(3, dtype=np.float32)
        return np.asarray([left / distance, forward / distance, distance], dtype=np.float32)

    result = dict(observation)
    result['vehicle'] = np.asarray([*(own[2:4] @ rotation), own[6]], dtype=np.float32)
    result['goal'] = bearing_distance(observation['goal'])
    result['local_target'] = bearing_distance(observation['local_target'])
    # Each history row starts with three XY offsets from the *current* car: issue/target/arrival
    # for a command, or start/end/local driving target for a previously driven segment.
    for key, pairs in (('road_nodes', 1), ('command_nodes', 3), ('drive_segments', 3)):
        rows = observation[key].copy()
        offsets = rows[:, :2*pairs].reshape(len(rows), pairs, 2)
        offsets[:] = offsets @ rotation
        result[key] = rows
    return result


class BodyFrameVehicleGraphEpisode(VehicleGraphHistoryLSTMEpisode):
    def _heading_observation(self):
        return body_frame_observation(super()._heading_observation())


class BodyFrameVehicleGraphPolicy(VehicleGraphHistoryLSTMPolicy):
    vehicle_features = 3  # forward and left velocity, angular velocity
    target_features = 3  # sine and cosine of relative bearing, normalized range
    angle_distribution = BodyAngleDistribution

    @staticmethod
    def _goal_geometry(own, goal):
        sine, cosine, distance = goal.unbind(-1)
        displacement = torch.stack((distance * cosine, distance * sine), dim=-1)
        return displacement, goal


class GoalResidualBodyFramePolicy(BodyFrameVehicleGraphPolicy):
    """Steer by a learned signed offset from the final goal's current bearing."""

    def _action_distribution(self, params, body_goal, arcs):
        bearing = torch.atan2(body_goal[:, 0], body_goal[:, 1])
        return self.angle_distribution(params, arcs, reference_angle=bearing)
