"""Fixed-altitude v9 inference and ROS radio geometry for Gazebo SITL."""

import hashlib
from pathlib import Path

import numpy as np

from drone.communication import RelayManager
from drone.cooperative_mappo import CooperativeNavigator
from drone.drone_main import DroneController
from drone.relay_attention_mappo import RelayActorCritic


FLIGHT_HEIGHT = 15.
RADIO_RANGE = 95.
MAX_SPEED = 6.
WORLD_BOUNDS = (-75., -75., 675., 75.)
CHECKPOINT_SHA256 = '7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198'


def make_controller(drone_id, checkpoint):
    import torch

    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Expected the archived v9 best576 checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('training_episodes') != 576
            or saved.get('reward_contract') != 'uav_remaining_route_mapping_v9'):
        raise ValueError('Checkpoint does not match the v9 inference contract')
    model = RelayActorCritic.from_state_dict(saved['model']).eval()
    controller = DroneController(
        drone_id, resolution=2., total_drones=3, communication_range=RADIO_RANGE,
        max_speed=MAX_SPEED, broadcast_interval=1., fusion_interval=5.,
        communication_mode='backbone', safety_filter='cbf', graph_min_clearance=1.,
    )
    controller.navigator = CooperativeNavigator(
        drone_id, total_drones=3, comm_range=RADIO_RANGE, min_separation=15.,
        model=model, max_speed=MAX_SPEED, world_bounds=WORLD_BOUNDS,
    )
    controller.ground_max_speed = 1.
    controller.relay = RelayManager(drone_id)
    return controller


def radio_delta(receiver, sender):
    """Return the direct 3D relative position if it is within radio range."""
    delta = np.asarray(sender, dtype=float) - np.asarray(receiver, dtype=float)
    return delta if np.linalg.norm(delta) <= RADIO_RANGE else None
