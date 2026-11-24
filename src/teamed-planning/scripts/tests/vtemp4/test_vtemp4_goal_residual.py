"""Goal-relative offsets must define one physical action law for acting and PPO replay."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtemp4.vtemp4_policy import update_policy
from vtemp4.vtemp4_body_angle import BodyAngleDistribution
from vtemp4.vtemp4_body_graph import (
    BodyFrameVehicleGraphPolicy, GoalResidualBodyFramePolicy, body_frame_observation,
)
from tests.simulator.test_vtemp4_body_graph import map_observation


def observation(bearing_deg, arcs_deg):
    obs = body_frame_observation(map_observation([400., 600.], .3))
    bearing = math.radians(bearing_deg)
    obs['goal'] = np.asarray([math.sin(bearing), math.cos(bearing), 2.5], dtype=np.float32)
    obs['angle_arcs'] = np.deg2rad(arcs_deg)
    return obs


class GoalResidualTests(unittest.TestCase):
    def test_zero_offset_and_large_signed_detours_in_the_policy(self):
        goals = [-165., -45., 35., 155.]
        observations = [observation(bearing, [[-180., 180.]]) for bearing in goals]
        new = GoalResidualBodyFramePolicy().eval()
        old = BodyFrameVehicleGraphPolicy().eval()
        with torch.no_grad():
            for offset in (0., 120., -140.):
                new.actor.weight.zero_()
                new.actor.bias.copy_(torch.tensor([math.atanh(offset / 180.), 0.]))
                old.load_state_dict(new.state_dict())
                new_policy, new_values = new(observations)
                old_policy, old_values = old(observations)
                new_angles = new_policy.action(True)[1].rad2deg().numpy()
                old_angles = old_policy.action(True)[1].rad2deg().numpy()
                expected = (np.asarray(goals) + offset + 180.) % 360. - 180.
                np.testing.assert_allclose(new_angles, expected, atol=2.e-5)
                np.testing.assert_allclose(old_angles, offset, atol=2.e-5)
                torch.testing.assert_close(new_values, old_values)
                torch.testing.assert_close(new_policy.rho, old_policy.rho)

    def test_physical_center_governs_clipped_mode_samples_probability_and_entropy(self):
        # Offset +35° places the physical center at -177° after wrapping.
        # Of the two disjoint legal arcs, the endpoint +175° is closest.
        arcs = [np.deg2rad([[-168., -160.], [165., 175.]])]
        head = torch.tensor([[math.atanh(35. / 180.), .75]], dtype=torch.float64, requires_grad=True)
        reference = torch.deg2rad(torch.tensor([148.], dtype=torch.float64))
        anchored = BodyAngleDistribution(head, arcs, reference_angle=reference)
        direct = BodyAngleDistribution(
            torch.tensor([[math.atanh(-177. / 180.), .75]], dtype=torch.float64), arcs)
        turns, mode = anchored.action(True)
        self.assertAlmostEqual(mode.rad2deg().item(), 175.)
        self.assertTrue(turns.all())
        torch.testing.assert_close(mode, direct.action(True)[1])
        torch.testing.assert_close(anchored.entropy(), direct.entropy())
        for degrees in (-164., 172.):
            point = torch.deg2rad(torch.tensor([degrees], dtype=torch.float64))
            torch.testing.assert_close(anchored.log_prob(turns, point), direct.log_prob(turns, point))
        torch.testing.assert_close(anchored.action(False, torch.Generator().manual_seed(71))[1],
                                   direct.action(False, torch.Generator().manual_seed(71))[1])

        def replay_loss(parameters):
            distribution = BodyAngleDistribution(parameters, arcs, reference_angle=reference)
            return distribution.log_prob(turns, mode.detach()) + .005 * distribution.entropy()

        self.assertTrue(torch.autograd.gradcheck(replay_loss, (head,), atol=1.e-6, rtol=1.e-4))

    def test_complex_legal_domain_rotates_together_with_the_goal(self):
        # The arcs stay in sorted order so equal uniforms must select the same rotated sample.
        shift = .4
        arcs = np.array([[-1.9, -1.2], [-.4, -.2], [.25, 1.4]])
        n = 128
        params = torch.tensor([[.12, 2.]], dtype=torch.float64).expand(n, -1)
        reference = torch.linspace(-.25, .5, n, dtype=params.dtype)
        base = BodyAngleDistribution(params, [arcs] * n, reference_angle=reference)
        rotated = BodyAngleDistribution(params, [arcs + shift] * n, reference_angle=reference + shift)
        for deterministic in (True, False):
            turns, angles = base.action(deterministic, torch.Generator().manual_seed(123))
            _, other = rotated.action(deterministic, torch.Generator().manual_seed(123))
            torch.testing.assert_close(angles + shift, other)
            torch.testing.assert_close(base.log_prob(turns, angles), rotated.log_prob(turns, other))
        torch.testing.assert_close(base.entropy(), rotated.entropy())

    def test_sampled_act_replays_as_physical_angle_in_ppo(self):
        observations = [observation(-130., [[-178., -100.], [15., 28.]]),
                        observation(80., [[-90., -45.], [37., 120.]]),
                        observation(167., [[-180., -160.], [70., 180.]])]
        generator = torch.Generator().manual_seed(7)
        model = GoalResidualBodyFramePolicy()
        before = model.actor.weight.detach().clone()
        episodes = []
        for obs, reward in zip(observations, (1., -.7, .3)):
            turn, angle, logp, value = model.act(obs, generator=generator)
            self.assertTrue(any(lo <= angle <= hi for lo, hi in obs['angle_arcs']))
            episodes.append([dict(observation=obs, turn=turn, angle_rad=angle, logp=logp, value=value,
                                  next_value=0., reward=reward, duration=30, terminated=True, truncated=False)])
        result = update_policy(model, torch.optim.Adam(model.parameters(), lr=3.e-4), episodes, epochs=1)
        self.assertEqual(result['policy_transitions'], 3)
        self.assertLess(result['behaviour_logp_error'], 2.e-5)
        self.assertFalse(torch.equal(before, model.actor.weight))
        self.assertTrue(torch.isfinite(model.actor.weight).all())


if __name__ == '__main__':
    unittest.main()
