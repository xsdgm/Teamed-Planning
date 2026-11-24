"""Check the goal geometry, graph readout, and a native vehicle-graph PPO update."""
import copy
from contextlib import redirect_stdout
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def goal_geometry_probe(model, initial, world_size):
    vehicle = np.array([600., 600.])
    # Each pair changes only the goal bearing or only the vehicle heading.
    cases = (([350., 0.], math.pi / 2), ([0., 350.], math.pi / 2),
             ([350., 0.], -math.pi / 2))
    road_offset = np.array([9., -4.])
    inputs = []
    for delta, yaw in cases:
        obs = copy.deepcopy(initial)
        obs['vehicle'][:2] = vehicle / world_size
        obs['vehicle'][4:6] = [math.sin(yaw), math.cos(yaw)]
        obs['goal'][:2] = (vehicle + delta) / world_size
        obs['goal'][2:4] = np.asarray(delta) / 350.
        obs['road_valid'][:] = False
        obs['road_valid'][0] = True
        obs['road_nodes'][:] = 0
        obs['road_nodes'][0, :2] = road_offset / 95.
        obs['road_neighbors'][:] = -1
        obs['road_neighbors'][0, 0] = 0
        obs['road_entry_mask'][:] = False
        obs['road_entry_mask'][0] = True
        obs['angle_arcs'] = np.array([[-2.5, -0.6], [0.2, 2.7]])
        inputs.append(obs)

    road_inputs, actor_inputs = [], []
    road_hook = model.gat1.register_forward_pre_hook(
        lambda module, args: road_inputs.append(args[0].detach().clone()))
    actor_hook = model.actor.register_forward_pre_hook(
        lambda module, args: actor_inputs.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            policy, values = model(inputs)
    finally:
        road_hook.remove()
        actor_hook.remove()

    expected_road = np.array([(np.asarray(delta) - road_offset) / 350. for delta, _ in cases])
    np.testing.assert_allclose(road_inputs[0][:, 0, 8:10], expected_road, atol=1e-6)
    # East goal under north yaw is to the right. North under north yaw is forward.
    np.testing.assert_allclose(actor_inputs[0][:, -3:], [[-1., 0., 1.], [0., 1., 1.], [1., 0., 1.]], atol=1e-6)
    assert torch.isfinite(values).all()
    weights = model.attention_weights(inputs)
    assert 'roads' in weights and 'vehicle' in weights
    assert {'goal', 'arcs', 'directions', 'peers', 'relay', 'history'}.isdisjoint(weights)
    flags, angles = policy.action(deterministic=True)
    assert torch.isfinite(policy.log_prob(flags, angles)).all()
    for obs, angle in zip(inputs, angles):
        lo, hi = obs['angle_arcs'].T
        assert np.any((lo <= float(angle)) & (float(angle) <= hi))

    # Admissible arcs only constrain action execution; changing them alone
    # must preserve the actor parameters and the value of the observation.
    other_arcs = copy.deepcopy(inputs[0])
    other_arcs['angle_arcs'] = np.array([[-0.1, 0.1]])
    if 'all_ring_endpoints_blocked' in other_arcs:
        other_arcs['all_ring_endpoints_blocked'] = not other_arcs['all_ring_endpoints_blocked']
    with torch.no_grad():
        arc_policy, arc_values = model([inputs[0], other_arcs])
        for same in (arc_policy.location, arc_policy.rho, arc_values):
            torch.testing.assert_close(same[0], same[1])

    # A readout using only the explicit body bearing distinguishes the two
    # goals even before there are map nodes or previously issued commands.
    for obs in inputs:
        obs['road_valid'][:] = False
        obs['road_entry_mask'][:] = False
        assert len(obs['command_nodes']) == len(obs['drive_segments']) == 0
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.weight[0, -3] = 1.
        empty_policy, empty_values = model(inputs)
        _, selected = empty_policy.action(deterministic=True)
        assert torch.isfinite(empty_values).all()
        assert selected.unique().numel() == len(inputs)
        near = copy.deepcopy(inputs[0])
        near['goal'][2:4] = 0.
        near['goal'][:2] = near['vehicle'][:2]
        near['angle_arcs'] = np.empty((0, 2))
        assert math.isfinite(model.value(near))


def native_check():
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_policy import update_policy
    from vtemp4.vtemp4_vehicle_graph import VehicleGraphGoalEpisode, VehicleGraphGoalPolicy

    repo = Path(__file__).resolve().parents[5]
    runtime(repo)
    torch.manual_seed(43)
    model = VehicleGraphGoalPolicy().eval()
    case = dict(seed=17, case_id='vehicle-graph', kind='v9_random', start=[1570., 490.],
                goal=[270., 490.], world_size=1650., direct_distance_m=1300., circles=[])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        episode = VehicleGraphGoalEpisode(repo, case, model, fast_seed=24, budget_s=2670,
                                          collector_limit_s=31, rollout=[])
        try:
            while not episode.finished:
                episode.step()
            markers = episode.fleet_summary()['shared_markers']
            rows = episode.rollout
        finally:
            episode.close()

    assert [m['issued_at_s'] for m in markers] == [0, 15, 30]
    assert [m['physics_broadcasts'] for m in markers] == [15, 15, 1]
    assert [len(row['observation']['command_nodes']) for row in rows] == [0, 1, 2]
    for row, marker in zip(rows, markers):
        assert {'ground_inputs', 'features', 'peers', 'relay', 'history', 'history_mask',
                'context'}.isdisjoint(row['observation'])
        assert row['angle_rad'] == marker['relative_angle_rad']
        assert row['turn'] and row['policy_controlled']
    initial, latest = rows[0]['observation'], rows[-1]['observation']
    weights = model.attention_weights([initial, latest])['commands']
    assert torch.count_nonzero(weights[0]) == 0 and (weights[1] > 0).all()
    # A prior command's target and the segment driven under it must both reach
    # the decision state. Compare observations from the same car/goal/road map.
    for key, column in (('command_nodes', 2), ('drive_segments', 2)):
        changed = copy.deepcopy(latest)
        changed[key][0, column] += 1.
        with torch.no_grad():
            p, v = model([latest, changed])
        assert p.location[0] != p.location[1] and v[0] != v[1], key
    before = model.actor.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    report = update_policy(model.train(), optimizer, [rows], epochs=1)
    assert report['policy_transitions'] == report['transitions'] == 3
    assert report['behaviour_logp_error'] < 2e-5
    assert not torch.equal(before, model.actor.weight)
    goal_geometry_probe(model.eval(), initial, case['world_size'])


class VehicleGraphTests(unittest.TestCase):
    def test_goal_features_native_sampling_and_ppo(self):
        result = subprocess.run([sys.executable, '-IB', __file__, 'native'],
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
