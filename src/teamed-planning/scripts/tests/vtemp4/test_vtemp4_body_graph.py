"""Check that a rigid change of map coordinates preserves the graph and physical action."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtemp4.vtemp4_body_graph import BodyFrameVehicleGraphPolicy, body_frame_observation


def map_observation(car, yaw):
    """Place the same road, car motion and command history in a different map frame."""
    car = np.asarray(car)
    c, s = math.cos(yaw), math.sin(yaw)
    to_world = np.array([[c, s], [-s, c]])

    def target(forward_left):
        offset = np.asarray(forward_left) @ to_world
        return np.r_[(car + offset) / 1650., offset / 350.].astype(np.float32)

    def history(body_points, scalars):
        offsets = (np.asarray(body_points) @ to_world).reshape(-1) / 200.
        return np.array([np.r_[offsets, scalars]], dtype=np.float32)

    roads = np.zeros((2, 8), dtype=np.float32)
    roads[:, :2] = np.array([[15., 0.], [22., -8.]]) @ to_world / 95.
    roads[:, 2] = 1.
    roads[:, 7] = np.linalg.norm(roads[:, :2], axis=1)
    return dict(
        vehicle=np.r_[car / 1650., (np.array([3., -1.]) @ to_world) / 6., s, c, .1].astype(np.float32),
        goal=target([300., 400.]), local_target=target([12., -5.]), local_target_valid=True,
        road_nodes=roads, road_neighbors=np.array([[0, 1], [1, 0]]), road_valid=np.ones(2, dtype=bool),
        road_entry_mask=np.array([True, False]), road_traversals=np.ones((2, 2, 4), dtype=np.float32),
        command_nodes=history([[-6., 3.], [140., 60.], [0., 0.]], [3.4, 3.4, .03, .01, 1.]),
        drive_segments=history([[-6., 3.], [0., 0.], [4., 7.]], [1., 3.4, 3.4, .04]),
        command_predecessor=np.array([-1]), drive_command=np.array([0]),
        angle_arcs=np.array([[-1.9, -.6], [-.1, 1.3]]), all_ring_endpoints_blocked=False)


class VehicleFrameTests(unittest.TestCase):
    def test_map_rotation_translation_and_graph_actor_readout(self):
        frames = [([400., 600.], .3), ([680., 450.], .3 + math.pi / 2)]
        observations = [body_frame_observation(map_observation(car, yaw)) for car, yaw in frames]
        first, rotated = observations
        self.assertEqual(set(first), set(rotated))
        for key in first:
            np.testing.assert_allclose(first[key], rotated[key], atol=2.e-6, err_msg=key)
        np.testing.assert_allclose(first['goal'], [.8, .6, 500. / 350.], atol=1.e-6)
        np.testing.assert_allclose(first['local_target'], [-5/13, 12/13, 13/350], atol=1.e-6)
        np.testing.assert_allclose(first['vehicle'], [.5, -1/6, .1], atol=1.e-6)
        np.testing.assert_allclose(first['command_nodes'][0, :6] * 200., [-6, 3, 140, 60, 0, 0], atol=2.e-5)
        np.testing.assert_allclose(first['drive_segments'][0, :6] * 200., [-6, 3, 0, 0, 4, 7], atol=2.e-5)
        inputs = []
        model = BodyFrameVehicleGraphPolicy().eval()
        hook = model.gat1.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
        try:
            with torch.no_grad():
                distribution, values = model(observations)
                _, angles = distribution.action(deterministic=True)
        finally:
            hook.remove()
        np.testing.assert_allclose(inputs[0][0, :, -2:], np.array([[285., 400.], [278., 408.]]) / 350.,
                                   atol=1.e-6)
        for result in (distribution.location, distribution.rho, angles, values):
            torch.testing.assert_close(result[0], result[1], atol=1.e-6, rtol=1.e-6)


if __name__ == '__main__':
    unittest.main()
