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
from unittest.mock import Mock, patch

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp4.vtemp4_observation import ring_points, valid_ring_points
from vtemp4.vtemp4_periodic_history import ArcAngleDistribution, PeriodicHistoryFleetEpisode, legal_ring_arcs


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

    def test_vtemp1_reissues_only_changed_coordinates_to_controller(self):
        episode = PeriodicHistoryFleetEpisode.__new__(PeriodicHistoryFleetEpisode)
        controller = Box(_drive_to=Mock(), waypoint_threshold=2.)
        episode.env = Box(ground_controller=controller)
        episode.manager = Box(selected=0)
        episode.adapter = Box()
        episode.goal = np.array([600., 500.])
        episode.ground_events = []
        episode.final_approach = False
        position = np.array([400., 500.])
        for index, target in enumerate(([550., 500.], [550., 500.], [600., 500.])):
            episode.current_request = dict(target=target, id=index)
            episode._issue_native_waypoint(30 * index, position)
        self.assertEqual([call.args[0] for call in controller._drive_to.call_args_list],
                         [(550., 500.), (600., 500.)])
        self.assertEqual([event['waypoint_id'] for event in episode.ground_events], [0, 1, 2])
        self.assertTrue(episode.manager.final)

    def test_vtemp1_rejects_raster_before_starting_simulation(self):
        case = dict(terrain='wildscenes_raster', kind='v9_random')
        with self.assertRaisesRegex(ValueError, 'random-circle'):
            PeriodicHistoryFleetEpisode(None, case, None, ground_control='vtemp1')
        with self.assertRaisesRegex(ValueError, 'Unknown ground control'):
            PeriodicHistoryFleetEpisode(None, {}, None, ground_control='unexpected')

    def test_vtemp1_wait_clears_the_previous_local_target(self):
        from survey.survey_fleet_runtime import SurveyFleetEpisode

        episode = PeriodicHistoryFleetEpisode.__new__(PeriodicHistoryFleetEpisode)
        planner = Box(compute_velocity_command=Mock(return_value=(1., 0.)))
        episode.ground_control = 'vtemp1'
        episode._native_drive_events = []
        episode.env = Box(ground_controller=Box(local_planner=planner), env=Box(time=0))
        episode.adapter = Box(ground_target=None)

        def drive_or_wait(neighbors):
            if episode.clock_s == 0:
                return planner.compute_velocity_command((0., 0.), 0., (4., 2.))
            return (0., 0.)

        with patch.object(SurveyFleetEpisode, '_ground_action', side_effect=drive_or_wait):
            self.assertEqual(episode._ground_action(), (1., 0.))
            np.testing.assert_array_equal(episode.adapter.ground_target, [4., 2.])
            episode.env.env.time = 1
            self.assertEqual(episode._ground_action(), (0., 0.))
            self.assertIsNone(episode.adapter.ground_target)
            episode.env.env.time = 2
            episode._ground_action()
        self.assertEqual(episode._native_drive_events, [
            dict(time_s=0, target=[4., 2.]), dict(time_s=1, target=None)])

    def test_vtemp1_uses_archived_motion_and_shared_broadcast(self):
        run = subprocess.run([sys.executable, '-IB', __file__, 'vtemp1'],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def vtemp1_check():
    from vtemp2_3.frozen_navigation import runtime
    from survey.survey_fleet_drive import GroundMotionGuard
    from survey.survey_fleet_observation import FleetSurveyAdapter

    repo = Path(__file__).resolve().parents[5]
    runtime(repo)
    from shared.ground.ground_main import GroundRobotController
    from shared.ground.local_planner import LocalPlanner
    from vtemp4.vtemp4_vehicle_graph import VehicleGraphGoalEpisode

    tracking = {}

    class GoalEpisode(VehicleGraphGoalEpisode):
        def _heading_observation(self):
            observation = super()._heading_observation()
            point = tracking.get(self.clock_s - 1)
            assert observation['local_target_valid'] == (point is not None)
            if point is not None:
                np.testing.assert_allclose(observation['local_target'][:2] * self.case['world_size'],
                                           point, atol=1.e-4)
            for row in observation['drive_segments']:
                start = self.clock_s - round(np.expm1(row[7]))
                point = tracking[start]
                assert bool(row[6]) == (point is not None)
                if point is not None:
                    np.testing.assert_allclose(self.position() + row[4:6] * self.lookahead_m,
                                               point, atol=1.e-4)
            return observation

    original = GroundRobotController.get_action
    compute = LocalPlanner.compute_velocity_command
    native = {}

    def capture_native(controller, observation):
        native['time'] = int(observation['full_state']['time'])
        tracking[native['time']] = None
        result = original(controller, observation)
        native.update(motion=result['motion'], neighbors=observation['neighbors'])
        return result

    def capture_velocity(planner, **arguments):
        tracking[native['time']] = np.asarray(arguments['target_waypoint']).copy()
        return compute(planner, **arguments)

    case = dict(seed=17, case_id='native-circles', kind='v9_random', start=[500., 490.],
                goal=[630., 490.], world_size=1000., direct_distance_m=130., circles=[])
    with (
        open(os.devnull, 'w') as sink, redirect_stdout(sink),
        patch.object(GroundRobotController, 'get_action', capture_native),
        patch.object(LocalPlanner, 'compute_velocity_command', capture_velocity),
        patch.object(FleetSurveyAdapter, 'plan_ground', side_effect=AssertionError('extra drive target')),
        patch.object(GroundMotionGuard, 'motion', side_effect=AssertionError('modified native motion')),
        patch.object(GoalEpisode, '_native_filter', side_effect=AssertionError('modified native graph')),
    ):
        ep = GoalEpisode(repo, case, Box(value=lambda obs: 0.), fast_seed=24, budget_s=2670,
                         collector_limit_s=61, review_interval_s=30, ground_control='vtemp1')
        try:
            while not ep.finished:
                assert ep.env.ground_action['motion'] == native['motion']
                assert ep.received_neighbors is native['neighbors']
                assert ep.env.ground_action['broadcast']['target'] == case['goal']
                np.testing.assert_array_equal(ep.adapter.ground_target, tracking[ep.clock_s])
                ep.step()
            record = ep.fleet_summary()
        finally:
            ep.close()
    assert record['ground_control'] == 'vtemp1'
    assert record['ground_received_packets'] > 0
    assert any(p is not None and not np.allclose(p, case['goal']) for p in tracking.values())
    assert [event['time_s'] for event in record['ground_replan_events']] == [0, 30, 60]
    assert [cmd['physics_broadcasts'] for cmd in record['shared_markers']] == [30, 30, 1]


def native_check():
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_periodic_history import PeriodicHistoryPolicy
    from vtemp4.vtemp4_policy import update_policy
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
    elif sys.argv[1:] == ['vtemp1']:
        vtemp1_check()
    else:
        unittest.main()
