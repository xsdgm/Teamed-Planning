"""Check periodic broadcast identity, continuous admissible arcs and PPO replay."""
from collections import deque
from contextlib import nullcontext, redirect_stdout
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as Box
import unittest

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ground.vtemp4_observation import ring_points, valid_ring_points
from simulator.vtemp4_periodic_history import ArcAngleDistribution, PeriodicHistoryFleetEpisode, legal_ring_arcs


class PeriodicHistoryTests(unittest.TestCase):
    def test_map_edges_and_occupied_arc(self):
        pos, yaw = np.array([22., 20.]), 2.43
        cells = np.zeros((333, 333), np.int8)
        cells[:140, 100:] = 1
        adapter = Box(resolution=1., cells=cells, ep=Box(case={'world_size': 332.5}))
        arcs, fallback = legal_ring_arcs(adapter, pos, yaw, 200.)
        self.assertFalse(fallback)
        angles = np.linspace(-math.pi+.0001, math.pi-.0001, 625)
        valid = valid_ring_points(adapter, ring_points(pos, yaw, angles, 200.))
        in_arcs = ((angles[:, None] >= arcs[:, 0]) & (angles[:, None] <= arcs[:, 1])).any(axis=1)
        np.testing.assert_array_equal(valid, in_arcs)
        self.assertTrue(valid.any() and not valid.all())
        cells[:] = 1
        arcs, fallback = legal_ring_arcs(adapter, pos, yaw, 200.)
        self.assertTrue(fallback)
        end = ring_points(pos, yaw, arcs.mean(axis=1), 200.)
        self.assertTrue(np.all((end >= 0.) & (end < 332.5)))

    def test_uniform_density_and_entropy_on_disjoint_arcs(self):
        arcs = np.array([[-2., -.6], [1.2, 2.1]])
        angles = torch.tensor([-1., 1.31, 2.])
        policy = ArcAngleDistribution(torch.tensor([[.1,-100.]]*3), [arcs]*3)
        span = np.diff(arcs).sum()
        designate = torch.ones(3, dtype=torch.bool)
        torch.testing.assert_close(policy.log_prob(designate, angles), torch.full((3,), -math.log(span)))
        torch.testing.assert_close(policy.entropy(), torch.full((3,), math.log(span)))
        with self.assertRaises(ValueError):
            policy.log_prob(designate, torch.zeros(3))
        with self.assertRaises(ValueError):
            policy.log_prob(~designate, angles)

    def test_stationary_same_angle_still_issues_in_final_approach(self):
        e = PeriodicHistoryFleetEpisode.__new__(PeriodicHistoryFleetEpisode)
        pos = np.array([500.,500.])
        e.position = lambda: pos.copy()
        e.samples = deque(maxlen=61)
        e.goal = np.array([510.,500.])
        e.adapter = Box(own={'orientation':.8}, last_ground_reason='', observe_receipts=lambda: None)
        e.env = Box(env=Box(time=0))
        e._heading_observation = lambda: {'all_ring_endpoints_blocked':False}
        e.slow = Box(act=lambda *args, **kwargs: (True,.37,-.2,0.))
        e.torch = Box(inference_mode=nullcontext)
        e.generator, e.rollout = None, None
        e.case = {'direct_distance_m':500.}
        e.limit = e.budget = 500
        e.pending = e.last_review = e.current_request = e._last_control_time = None
        e.requests, e.heading_reviews, e.ground_events = [], [], []
        e.final_approach = False
        e.review_interval, e.ground_interval, e.lookahead_m, e.heading_return = 15,5,200.,0.
        e._native_filter = lambda g,s: g
        e._ground_decision = lambda now,guidance: e.ground_events.append(dict(time_s=now,guidance=guidance))
        for t in range(32):
            e.env.env.time = t
            e._after_fusion(None,None)
        self.assertEqual([r['issued_at_s'] for r in e.requests], [0,15,30])
        self.assertTrue(all(r['target'] == e.requests[0]['target'] for r in e.requests))
        self.assertEqual([r['id'] for r in e.requests], [0,1,2])
        self.assertTrue(all(r['changed'] and r['accepted'] for r in e.heading_reviews))
        self.assertTrue(all(r['final_approach'] and r['guidance'] is None for r in e.ground_events))
        self.assertTrue(all(r['waypoint_id'] == r['time_s']//15 for r in e.ground_events))

    def test_native_periodic_rollout_and_ppo(self):
        run = subprocess.run([sys.executable,'-IB',__file__,'native'], capture_output=True,text=True,timeout=180)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)


def native_check():
    from world_model.frozen_navigation import runtime
    from simulator.vtemp4_periodic_history import PeriodicHistoryPolicy
    from ground.vtemp4_policy import update_policy
    repo = Path(__file__).resolve().parents[5]
    runtime(repo)
    torch.manual_seed(43)
    model = PeriodicHistoryPolicy()
    case = dict(seed=17,case_id='periodic',kind='v9_random',start=[1570.,490.],goal=[270.,490.],
                world_size=1650.,direct_distance_m=1300.,circles=[])
    with open(os.devnull,'w') as sink, redirect_stdout(sink):
        ep = PeriodicHistoryFleetEpisode(repo,case,model.eval(),fast_seed=24,budget_s=2670,
                                        collector_limit_s=61,rollout=[])
        try:
            while not ep.finished:
                ep.step()
            record = ep.fleet_summary()
        finally:
            ep.close()
    markers = record['shared_markers']
    assert [r['issued_at_s'] for r in markers] == list(range(0,61,15))
    assert [r['physics_broadcasts'] for r in markers] == [15,15,15,15,1]
    for marker in markers:
        point = np.asarray(marker['target'])
        assert ((point >= 0.) & (point < 1650.)).all()
        np.testing.assert_allclose(np.linalg.norm(point-marker['origin_xy']),200.)
    assert all(r['shared_marker'] == markers[r['time_s']//15]['target'] for r in record['ground_replan_events'])
    rows = ep.rollout
    assert [r['observation']['history_mask'].sum() for r in rows] == [0,1,2,3,4]
    assert all(r['turn'] and r['angle_rad'] == m['relative_angle_rad'] for r,m in zip(rows,markers))
    before = model.history_attention.value.weight.detach().clone()
    stats = update_policy(model.train(),torch.optim.Adam(model.parameters()),[rows])
    assert stats['transitions'] == 5 and stats['behaviour_logp_error'] < 2e-5
    assert not torch.equal(before,model.history_attention.value.weight)


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
