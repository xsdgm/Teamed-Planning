"""Exercise the combined attention, continuous action, native broadcast and PPO update."""
from contextlib import redirect_stdout
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ContinuousHistoryTests(unittest.TestCase):
    def test_native_rollout_and_ppo(self):
        run = subprocess.run([sys.executable, '-IB', __file__, 'native'], capture_output=True,
                             text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def native_check():
    import numpy as np
    import torch

    scripts = Path(__file__).resolve().parents[2]
    repo = scripts.parents[2]
    sys.path.insert(0, str(scripts))
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_policy import update_policy
    from vtemp4.vtemp4_ring_history import ContinuousHistoryFleetEpisode, ContinuousHistoryPolicy

    runtime(repo)
    torch.manual_seed(43)
    model = ContinuousHistoryPolicy()
    case = dict(seed=17, case_id='continuous-history', kind='v9_random', start=[500., 500.],
                goal=[1350., 500.], world_size=1650., direct_distance_m=850., circles=[])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        episode = ContinuousHistoryFleetEpisode(repo, case, model.eval(), fast_seed=24,
                                                budget_s=2670, collector_limit_s=61, rollout=[])
        try:
            while not episode.finished:
                episode.step()
            record = episode.fleet_summary()
        finally:
            episode.close()

    rows = episode.rollout
    assert [r['observation']['history_mask'].sum() for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]['turn'] and all(not row['turn'] and row['logp'] == 0. for row in rows[1:])
    assert record['travelled_m'] > 0. and len(record['shared_markers']) == 1
    marker = record['shared_markers'][0]
    angle = marker['relative_angle_rad']
    assert not math.isclose(angle / (2 * math.pi / 48), round(angle / (2 * math.pi / 48)), abs_tol=1e-5)
    bearing = angle + marker['vehicle_yaw_rad']
    np.testing.assert_allclose(marker['target'], np.asarray(marker['origin_xy']) + 200.*np.array(
        [math.cos(bearing), math.sin(bearing)]))
    assert marker['physics_broadcasts'] == 61 and marker['heard_by']
    assert all(r['shared_marker'] == marker['target'] for r in record['ground_replan_events'])
    assert all(r['review_reason'] == 'tracking' for r in record['heading_reviews'][1:])

    def output(obs):
        with torch.no_grad():
            policy, value = model([obs])
            return torch.stack((policy.turn.probs, policy.location, policy.rho, value))

    # The same instant can have different past locations. A free decision must
    # respond through the continuous actor and value head, and ignore padding.
    obs = dict(rows[-1]['observation'], can_hold=True, can_designate=True)
    reversed_history = dict(obs, history=obs['history'][::-1].copy())
    assert not torch.equal(output(obs)[1:3], output(reversed_history)[1:3])
    assert not torch.equal(output(obs)[-1], output(reversed_history)[-1])
    empty = dict(obs, history_mask=np.zeros(4, bool))
    assert torch.equal(output(empty), output(dict(empty, history=np.full((4, 5), np.nan, dtype=np.float32))))
    with torch.no_grad():
        forced_hold, _ = model([rows[1]['observation']])
        assert forced_hold.turn.probs.item() == 0. and forced_hold.entropy().item() == 0.
        first, _ = model([rows[0]['observation']])
        assert first.turn.probs.item() == 1.
        selectable, _ = model([obs])
        assert 0. < selectable.turn.probs.item() < 1.

    before = model.history_attention.value.weight.detach().clone()
    metrics = update_policy(model.train(), torch.optim.Adam(model.parameters(), lr=.0003), [rows])
    assert metrics['transitions'] == 5 and metrics['behaviour_logp_error'] < 1e-5
    assert metrics['epochs'] == 3 and math.isfinite(metrics['loss'])
    assert not torch.equal(before, model.history_attention.value.weight)


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
