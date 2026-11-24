"""Check the continuous policy with the same frozen runtime as a training worker."""
from pathlib import Path
import subprocess
import sys
import unittest


class HeadingPolicyTest(unittest.TestCase):
    def test_continuous_actions_and_ppo(self):
        result = subprocess.run([sys.executable, '-I', '-B', __file__, 'isolated'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rule_designations_keep_returns_without_actor_updates(self):
        import math
        import torch
        from vtemp4.vtemp4_periodic_history import ArcAngleDistribution
        from vtemp4.vtemp4_policy import update_policy

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Linear(1, 2)
                self.critic = torch.nn.Linear(1, 1)

            def critic_values(self, observations):
                return self.critic(torch.tensor([[obs['x']] for obs in observations])).squeeze(-1)

            def forward(self, observations):
                features = torch.tensor([[obs['x']] for obs in observations])
                return (ArcAngleDistribution(self.actor(features), [obs['arcs'] for obs in observations]),
                        self.critic_values(observations))

        torch.manual_seed(9)
        model = Policy()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        selected = dict(x=1., arcs=[[0., .5]])
        forced = dict(x=.4, arcs=[])
        with torch.no_grad():
            distribution, value = model([selected])
            turn, angle = distribution.action(deterministic=True)
            logp = distribution.log_prob(turn, angle).item()
            near_value = model.critic_values([forced]).item()
        # Historical network-selected samples omit policy_controlled.
        first = dict(observation=selected, turn=True, angle_rad=angle.item(), logp=logp,
                     reward=.2, value=value.item(), next_value=near_value, duration=15,
                     terminated=False, truncated=False)
        last = dict(observation=forced, turn=True, angle_rad=math.pi, logp=0.,
                    policy_controlled=False, reward=4., value=near_value, next_value=0.,
                    duration=2, terminated=True, truncated=False)
        report = update_policy(model, optimizer, [[first, last]], epochs=1)
        self.assertEqual((report['transitions'], report['policy_transitions']), (2, 1))
        self.assertAlmostEqual(report['mean_return_target'], 4.1, places=5)
        self.assertLess(report['behaviour_logp_error'], 1e-6)

        # Adam has nonzero actor momentum now. A fully rule-directed update
        # must leave both actor parameters and their optimizer state untouched.
        actor_before = [p.detach().clone() for p in model.actor.parameters()]
        actor_steps = [optimizer.state[p]['step'].item() for p in model.actor.parameters()]
        self.assertGreater(optimizer.state[model.actor.weight]['exp_avg'].abs().sum().item(), 0.)
        critic_before = model.critic.bias.detach().clone()
        report = update_policy(model, optimizer, [[last]], epochs=2)
        self.assertEqual(report['policy_transitions'], 0)
        for p, original, step in zip(model.actor.parameters(), actor_before, actor_steps):
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, original))
            self.assertEqual(optimizer.state[p]['step'].item(), step)
        self.assertFalse(torch.equal(model.critic.bias, critic_before))


def check_ego_attention(model, observed):
    import copy
    import math
    import numpy as np
    import torch
    from vtemp2_3.ground_policy import ROAD_LIMIT, TASK_BLOCKS, _blocks
    from vtemp4.vtemp4_observation import DIRECTION_COUNT, RELATIVE_ANGLES

    # Real 200 m rays. Even with no roads there are 24 x 8 terrain sections.
    packed = torch.from_numpy(observed['ground_inputs'])[None]
    data = _blocks(packed, TASK_BLOCKS)
    rays = np.column_stack((np.cos(RELATIVE_ANGLES), np.sin(RELATIVE_ANGLES))).astype('f')
    data['candidate_xy'] = data['candidate_xy'][:, :DIRECTION_COUNT]
    data['candidate_xy'][0] = torch.from_numpy(rays * (200. / 95.))
    features = observed['features'][:DIRECTION_COUNT]
    features[:, :2] = rays
    features[:, 2] = 1.
    features[:, 6::3] = 1.  # all sections unknown
    rows = torch.from_numpy(features)[None]
    context = torch.from_numpy(observed['context'])[None]
    ego = context.new_zeros(1, model.encoder.ego[0].out_features)
    roads = context.new_zeros(1, ROAD_LIMIT, 32)

    # Disable only the learned scores to measure the distance prior in isolation.
    prior = copy.deepcopy(model.ego_map)
    with torch.no_grad():
        prior.query.weight.zero_()
        prior.query.bias.zero_()
        data['road_valid'][0, :2] = 1
        data['roads'][0, :2, 0] = torch.tensor([12.5, 62.5]) / 95.
        _, weights = prior(ego, context, roads, data, rows)
        assert math.isclose((weights[0, 0] / weights[0, 1]).item(), math.e, rel_tol=1e-5)
        terrain = weights[0, ROAD_LIMIT:].view(DIRECTION_COUNT, 8)
        assert math.isclose((terrain[0, 0] / terrain[0, 2]).item(), math.e, rel_tol=1e-5)
        assert weights[0, 2:ROAD_LIMIT].count_nonzero() == 0
        data['road_valid'].zero_()
        data['roads'].zero_()
        unknown, unknown_weights = model.ego_map(ego, context, roads, data, rows)
        assert unknown_weights[:, :ROAD_LIMIT].count_nonzero() == 0
        torch.testing.assert_close(unknown_weights.sum(-1), torch.ones(1))
        blocked = rows.clone()
        blocked[0, 0, 4:7] = torch.tensor([0., 1., 0.])
        updated, blocked_weights = model.ego_map(ego, context, roads, data, blocked)
        assert not torch.equal(unknown, updated)
        assert not torch.equal(unknown_weights, blocked_weights)


def check_policy():
    import math
    import numpy as np
    import torch

    scripts = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(scripts))
    from vtemp2_3.frozen_navigation import runtime
    from vtemp2_3.ground_policy import CANDIDATE_LIMIT, TASK_INPUT_WIDTH
    from vtemp4.vtemp4_observation import CONTEXT_SIZE, DIRECTION_COUNT, FEATURE_COUNT
    from vtemp4.vtemp4_policy import HeadingDistribution, HeadingPolicy, update_policy

    runtime(scripts.parents[2])
    torch.manual_seed(31)
    output = torch.tensor([[1., .23, -.8]], requires_grad=True)
    policy = HeadingDistribution(output)
    hold = policy.log_prob(torch.tensor([False]), torch.tensor([0.]))
    hold.backward()
    torch.testing.assert_close(output.grad[0, 1:], torch.zeros(2))
    # Integration over a full revolution plus P(hold) must conserve probability.
    grid = torch.linspace(-math.pi, math.pi, 1001)
    logp = policy.log_prob(torch.ones_like(grid, dtype=torch.bool), grid)
    torch.testing.assert_close(torch.trapezoid(logp.exp(), grid) + hold.exp().squeeze(), torch.tensor(1.))
    torch.testing.assert_close(logp[0], logp[-1])

    observed = dict(features=np.zeros((CANDIDATE_LIMIT, FEATURE_COUNT), np.float32),
                    context=np.zeros(CONTEXT_SIZE, np.float32),
                    ground_inputs=np.zeros(TASK_INPUT_WIDTH, np.float32),
                    present=np.arange(CANDIDATE_LIMIT) < DIRECTION_COUNT)
    observed['mask'] = observed['present'].copy()
    model = HeadingPolicy()
    check_ego_attention(model, observed)
    actor_weights = model.actor.weight.detach().clone()
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.copy_(torch.tensor([1., math.atanh(37. / 180.), -.1]))
    turn, angle, deterministic_logp, value = model.act(observed, deterministic=True)
    assert turn and abs(math.degrees(angle) - 37.) < .001
    assert value == model.value(observed)
    with torch.no_grad():
        model.actor.weight.copy_(actor_weights)
        model.critic.weight.zero_()  # Check that PPO's actor gradient reaches the ego-map branch.
    turn, angle, deterministic_logp, value = model.act(observed, deterministic=True)
    assert turn

    rng = torch.Generator().manual_seed(123)
    actions = [model.act(observed, generator=rng) for _ in range(20)]
    assert any(choice and abs(theta - angle) > .01 for choice, theta, _, _ in actions)
    assert any(not choice and theta == 0 for choice, theta, _, _ in actions)
    with torch.no_grad():
        distribution, _ = model([observed])
        for choice, theta, sampled_logp, _ in actions:
            expected = distribution.log_prob(torch.tensor([choice]), torch.tensor([theta]))
            assert math.isclose(expected.item(), sampled_logp, abs_tol=1e-5)

    turn_row = dict(observation=observed, turn=True, angle_rad=angle, logp=deterministic_logp, value=value,
                    reward=1., next_value=3., duration=7, terminated=False, truncated=True)
    hold_action = next(action for action in actions if not action[0])
    terminal_row = dict(turn_row, turn=False, angle_rad=0., logp=hold_action[2], next_value=123.,
                        reward=-5., duration=0, terminated=True, truncated=False)
    prior_row = dict(turn_row, reward=0., next_value=value, truncated=False, duration=15)
    before = model.actor.bias.detach().clone()
    stats = update_policy(model, torch.optim.Adam(model.parameters(), lr=.0005),
                          [[prior_row, turn_row], [terminal_row]], epochs=1, microbatch_size=1)
    assert stats['transitions'] == 3 and stats['behaviour_logp_error'] < 1e-5
    assert abs(stats['mean_return_target'] - 1.) < 1e-5  # (4 + 4 - 5) / 3
    assert not torch.equal(before, model.actor.bias)
    assert model.ego_map.terrain.weight.grad.norm() > 0.
    assert model.ego_map.query.weight.grad.norm() > 0.


if __name__ == '__main__':
    if sys.argv[1:] == ['isolated']:
        check_policy()
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        unittest.main()
