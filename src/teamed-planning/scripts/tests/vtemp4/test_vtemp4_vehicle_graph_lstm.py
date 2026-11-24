"""Exercise full-episode command history, causal recurrence and masked temporal attention."""
import copy
from contextlib import redirect_stdout
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@torch.no_grad()
def check_temporal_encoding(model, history):
    """Hold geometry and all recent commands fixed while perturbing the earliest decision."""
    changed_point = copy.deepcopy(history)
    changed_point['command_nodes'][0, 2:4] += [1., -1.]
    changed_drive = copy.deepcopy(history)
    first_drive = changed_drive['drive_command'] == 0
    assert first_drive.any()
    changed_drive['drive_segments'][first_drive, 2:4] += [-1., .5]

    # Exchange two whole command/drive records, leaving their sequence slots
    # intact. Pooling independent records would be invariant to this exchange.
    reordered = copy.deepcopy(history)
    reordered['command_nodes'][[0, 1]] = history['command_nodes'][[1, 0]]
    owners = history['drive_command']
    reordered['drive_command'][owners == 0] = 1
    reordered['drive_command'][owners == 1] = 0
    # The same drive segments in a different order should yield a different
    # command representation even though any per-command sum or mean is identical.
    drive_order = copy.deepcopy(history)
    middle_command = 3
    first, second = np.flatnonzero(owners == middle_command)[:2]
    drive_order['drive_segments'][[first, second]] = history['drive_segments'][[second, first]]
    late = copy.deepcopy(history)
    late['command_nodes'][-1, 2:4] += [1., -1.]
    late_drive = copy.deepcopy(history)
    late_drive['drive_segments'][-1, 2:4] += [-.8, .3]
    assert owners[-1] == len(history['command_nodes']) - 1

    observations = [history, changed_point, changed_drive, reordered, drive_order, late, late_drive]
    states, valid = model._commands(observations)
    assert valid.all() and valid.shape[1] > 16
    # An older decision changes the following hidden state; a later decision
    # cannot change any hidden state before it. The actor can read the old step.
    # Capture the shared state entering the critic. A change in this vector
    # can still round to the same scalar under an untrained value projection.
    shared = []
    capture = model.critic.register_forward_pre_hook(lambda module, args: shared.append(args[0]))
    try:
        policy, _ = model(observations)
    finally:
        capture.remove()
    for i in (1, 2, 3):
        assert not torch.equal(states[0, 1], states[i, 1])
        assert policy.location[0] != policy.location[i]
        assert not torch.equal(shared[0][0], shared[0][i])
    torch.testing.assert_close(states[0, :middle_command], states[4, :middle_command])
    assert not torch.equal(states[0, middle_command], states[4, middle_command])
    assert not torch.equal(shared[0][0], shared[0][4])
    for i in (5, 6):
        torch.testing.assert_close(states[0, :-1], states[i, :-1])
        assert not torch.equal(states[0, -1], states[i, -1])


@torch.no_grad()
def check_batch_masking(model, observations):
    policy, values = model(observations)
    weights = model.attention_weights(observations)['commands']
    for i, obs in enumerate(observations):
        length = len(obs['command_nodes'])
        assert torch.count_nonzero(weights[i, :, length:]) == 0
        if length:
            actual = weights[i, :, :length]
            assert (actual > 0).all()
            torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)))
        alone, value_alone = model([obs])
        torch.testing.assert_close(alone.location[0], policy.location[i])
        torch.testing.assert_close(alone.rho[0], policy.rho[i])
        torch.testing.assert_close(value_alone[0], values[i])

    encode_commands = model._commands

    def fill_unused(batch):
        tokens, valid = encode_commands(batch)
        return tokens.masked_fill(~valid[..., None], 1234.), valid

    with patch.object(model, '_commands', side_effect=fill_unused):
        noisy, noisy_values = model(observations)
    torch.testing.assert_close(noisy.location, policy.location)
    torch.testing.assert_close(noisy.rho, policy.rho)
    torch.testing.assert_close(noisy_values, values)

    # Altering solely the output domain leaves the learned history/goal response unchanged.
    sample = observations[-1]
    left = dict(sample, angle_arcs=np.array([[-1., -.75]]))
    right = dict(sample, angle_arcs=np.array([[.75, 1.]]))
    restricted, values = model([left, right])
    torch.testing.assert_close(restricted.location[0], restricted.location[1])
    torch.testing.assert_close(restricted.rho[0], restricted.rho[1])
    torch.testing.assert_close(values[0], values[1])
    _, chosen = restricted.action(deterministic=True)
    assert -1. <= chosen[0] <= -.75 and .75 <= chosen[1] <= 1.


def check_native_rewards(repo, case, model):
    from vtemp4.vtemp4_parallel import collect_episode
    from vtemp4.vtemp4_ring_contract import (
        PERIODIC_GOAL_RING_CONTRACTS, RING_VEHICLE_GRAPH_LSTM_FORMAT, VEHICLE_GRAPH_LSTM_VARIANT,
    )
    from vtemp4.vtemp4_vehicle_graph import VehicleGraphHistoryLSTMEpisode
    from vtemp2_3.frozen_navigation import CHECKPOINT_SHA
    from vtemp2_3.hierarchical_sources import file_sha256

    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        episode = VehicleGraphHistoryLSTMEpisode(repo, case, model, fast_seed=24, budget_s=30, rollout=[])
        try:
            while not episode.finished:
                episode.step()
            report = episode.fleet_summary()
            assert episode.progress_reward_weight == 10.
            assert [row['duration'] for row in episode.rollout] == [30]
            assert episode.rollout[-1]['terminated'] and report['end_reason'] == 'time_budget'
        finally:
            episode.close()
    direct = case['direct_distance_m']
    expected = 10. * (direct - report['remaining_m']) / direct - 30. / 30.
    np.testing.assert_allclose(report['reward'], expected, atol=1.e-9)

    # Archived LSTM checkpoints omitted the original weight of 2.
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        path = folder / 'old.pt'
        torch.save(dict(model=model.state_dict(), v9_sha256=CHECKPOINT_SHA, format=RING_VEHICLE_GRAPH_LSTM_FORMAT,
                        policy_variant=VEHICLE_GRAPH_LSTM_VARIANT, clock=dict(review_interval_s=15, lookahead_m=200.),
                        action_contract=PERIODIC_GOAL_RING_CONTRACTS[15]), path)
        legacy_case = dict(case, terrain='synthetic', circles_sha256='empty', navigation_task='random_circles',
                           terrain_family_id='reward-test')
        row = collect_episode(repo, path, file_sha256(path), legacy_case, 24, folder / 'episode',
                              collector_seconds=15)
        assert row['decisions'] == 1 and row['end_reason'] == 'collector_limit'
        progress = direct - row['remaining_m']
        assert abs(progress) > 1.e-3
        expected = 2. * progress / direct - 15. / (3 * direct + 120.)
        np.testing.assert_allclose(row['reward'], expected, atol=1.e-9)


def native_check():
    from vtemp4.vtemp4_policy import update_policy
    from vtemp4.vtemp4_vehicle_graph import VehicleGraphHistoryLSTMEpisode, VehicleGraphHistoryLSTMPolicy
    from vtemp2_3.frozen_navigation import runtime

    repo = Path(__file__).resolve().parents[5]
    runtime(repo)
    torch.manual_seed(43)
    model = VehicleGraphHistoryLSTMPolicy().eval()
    case = dict(seed=17, case_id='full-command-history', kind='v9_random', start=[1570., 490.],
                goal=[270., 490.], world_size=1650., direct_distance_m=1300., circles=[])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        episode = VehicleGraphHistoryLSTMEpisode(repo, case, model, fast_seed=24, budget_s=2670,
                                                 review_interval_s=15, collector_limit_s=256, rollout=[])
        try:
            while not episode.finished:
                episode.step()
            markers = episode.fleet_summary()['shared_markers']
            rows = episode.rollout
        finally:
            episode.close()

    assert len(rows) == len(markers) == 18
    for i, row in enumerate(rows):
        obs = row['observation']
        assert len(obs['command_nodes']) == i
        if i:
            # Convert each stored car-relative point back to issue-time world coordinates.
            xy = obs['command_nodes'][:, 2:4] * 200. + markers[i]['origin_xy']
            np.testing.assert_allclose(xy, [cmd['target'] for cmd in markers[:i]], rtol=0., atol=3.e-5)
            np.testing.assert_array_equal(obs['command_nodes'][:, -1], [0.] * (i - 1) + [1.])
    check_temporal_encoding(model, rows[-1]['observation'])
    check_batch_masking(model, [rows[i]['observation'] for i in (1, 0, -1)])

    recurrent_weights = [next(module.weight_ih_l0 for module in model.modules() if isinstance(module, kind))
                         for kind in (torch.nn.LSTM, torch.nn.GRU)]
    before = [weight.detach().clone() for weight in recurrent_weights]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    report = update_policy(model.train(), optimizer, [rows], epochs=1)
    assert report['transitions'] == report['policy_transitions'] == len(rows)
    assert report['behaviour_logp_error'] < 2.e-5
    for previous, updated in zip(before, recurrent_weights):
        assert not torch.equal(previous, updated)
    check_native_rewards(repo, case, model.eval())


class FullHistoryLSTMTest(unittest.TestCase):
    def test_full_sequence_causality_masking_and_ppo(self):
        result = subprocess.run([sys.executable, '-IB', __file__, 'native'],
                                capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
