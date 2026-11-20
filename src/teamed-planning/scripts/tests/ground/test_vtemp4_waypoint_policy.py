"""Verify constrained point actions, the history ablation and their stored PPO likelihoods."""
from pathlib import Path
import subprocess
import sys
import unittest


class WaypointPolicyTest(unittest.TestCase):
    def test_waypoint_actions_and_history(self):
        result = subprocess.run([sys.executable, '-I', '-B', __file__, 'isolated'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check():
    import copy
    import math
    import numpy as np
    import torch
    scripts = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(scripts))
    from world_model.frozen_navigation import runtime
    from ground.ground_policy import CANDIDATE_LIMIT, TASK_INPUT_WIDTH
    from ground.vtemp4_observation import CONTEXT_SIZE, FEATURE_COUNT
    from ground.vtemp4_waypoint_policy import WAYPOINT_COUNT, WaypointDistribution, WaypointPolicy, update_waypoint_policy

    runtime(scripts.parents[2])
    torch.manual_seed(73)
    mask = torch.tensor([[False, False, False], [True, False, True], [False, True, True]])
    dist = WaypointDistribution(torch.zeros(3), torch.zeros(3, 3), mask, torch.tensor([True, False, True]))
    torch.testing.assert_close(dist.entropy(), torch.tensor([0., math.log(2.), 1.5 * math.log(2.)]))
    logp = dist.log_prob(torch.tensor([False, True, False]), torch.tensor([-1, 2, -1]))
    torch.testing.assert_close(logp, torch.tensor([0., -math.log(2.), -math.log(2.)]))
    logp = dist.log_prob(torch.tensor([False, True, True]), torch.tensor([-1, 0, 1]))
    torch.testing.assert_close(logp, torch.tensor([0., -math.log(2.), -math.log(4.)]))
    for _ in range(12):
        change, index = dist.action(False)
        assert not change[0] and index[0] == -1
        assert change[1] and index[1] in (0, 2)

    obs = dict(features=np.zeros((CANDIDATE_LIMIT, FEATURE_COUNT), np.float32),
               ground_inputs=np.zeros(TASK_INPUT_WIDTH, np.float32),
               context=np.zeros(CONTEXT_SIZE, np.float32),
               can_hold=False, active_waypoint_delta=np.zeros(2, np.float32),
               waypoint_mask=np.ones(WAYPOINT_COUNT, bool),
               waypoint_xy=np.random.default_rng(91).uniform(-1., 1., (WAYPOINT_COUNT, 2)).astype('f'),
               history=np.arange(20, dtype=np.float32).reshape(4, 5) / 7.,
               history_mask=np.array([False, True, True, True]))
    off = WaypointPolicy(use_history=False)
    on = WaypointPolicy(use_history=True)
    on.load_state_dict(off.state_dict())
    padded = copy.deepcopy(obs)
    padded['history'][0] = np.nan
    ordered = copy.deepcopy(obs)
    ordered['history'][1:] = obs['history'][1:][::-1]
    ordered['context'][4:7] = [1.5, -.3, .8]
    empty = dict(obs, history_mask=np.zeros(4, bool))
    def outputs(model, sample):
        with torch.no_grad():
            p, v = model([sample])
            return torch.cat([p.point.probs, p.gate.probs[:, None], v[:, None]], dim=-1)
    assert torch.equal(outputs(off, obs), outputs(off, ordered))
    assert torch.equal(outputs(on, empty), outputs(off, obs))
    assert torch.equal(outputs(on, padded), outputs(on, obs))
    assert not torch.equal(outputs(on, obs), outputs(on, ordered))

    hold = dict(obs, can_hold=True, waypoint_mask=np.zeros(WAYPOINT_COUNT, bool))
    a = on.act(hold)
    assert a[0] is False and a[1] == -1 and a[2] == 0.
    act = on.act(obs, generator=torch.Generator().manual_seed(41))
    assert act[0] is True and 0 <= act[1] < WAYPOINT_COUNT
    fields = dict(next_value=0., duration=15, truncated=False, terminated=True)
    rows = [[dict(fields, observation=o, replace=b, point_index=i, logp=lp, value=v, reward=r)]
            for o, (b, i, lp, v), r in ((obs, act, 1.), (hold, a, -2.))]
    report = update_waypoint_policy(on, torch.optim.Adam(on.parameters(), lr=.0003), rows, epochs=1)
    assert report['transitions'] == 2 and report['behaviour_logp_error'] < 2e-5
    assert on.point_score.weight.grad.norm() > 0
    assert on.history_attention.value.weight.grad.norm() > 0


if __name__ == '__main__':
    if sys.argv[1:] == ['isolated']:
        check()
    else:
        unittest.main()
