"""Check real 3-UAV execution on km geometry and frozen reward-mask equivalence."""
from pathlib import Path
import subprocess
import sys
import unittest


class NativeFleetTests(unittest.TestCase):
    def test_native_km_physics_and_reward_mask(self):
        proc = subprocess.run([sys.executable, '-I', '-B', __file__, 'native'],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


def native_check():
    from contextlib import redirect_stdout
    import os
    import tempfile

    import numpy as np
    import torch

    scripts = Path(__file__).resolve().parents[2]
    repo = scripts.parents[2]
    sys.path.insert(0, str(scripts))
    from vtemp2_3.frozen_navigation import runtime
    from shared.simulator.ground_evidence import GroundPolicyEpisode
    from vtemp2_3.hierarchical_sources import file_sha256
    from survey.survey_fleet_metrics import task_mask, use_bounded_task_mask
    from survey.survey_fleet_runtime import SurveyFleetEpisode
    from vtemp2_3.train_slow_policy import SCHEMA, slow_model

    me, _, model, _ = runtime(repo)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    ledger = me.DeliveredEvidence(world=72., resolution=2.)
    rng = np.random.default_rng(40)
    # Include world boundaries, exact tangent/equality, degenerate and tiny
    # corridors, exterior endpoints, and non-grid-aligned radii.
    geometry = [([0., 0.], [72., 72.]), ([1., 1.], [1., 1.]), ([1., 1.], [1., 1.000001]),
                ([23., 23.], [65., 23.]), ([71., 71.], [-250., 200.]), ([-90., 26.], [-70., 91.])]
    geometry.extend((rng.uniform(-40, 110, 2), rng.uniform(-40, 110, 2)) for _ in range(20))
    for a, b in geometry:
        for radii in [(18., 18., 12.), (0., 2., 0.), (1.5, 8.5, 4.)]:
            a, b = np.array(a), np.array(b)
            np.testing.assert_array_equal(task_mask(ledger, a, b, *radii), ledger.task_mask(a, b, *radii))

    with tempfile.TemporaryDirectory() as folder, open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
        circles = np.array([[222., 228., 6.], [890., 782., 4.], [390., 1472., 3.]], dtype=np.float64)
        path = Path(folder) / 'circles.npy'
        np.save(path, circles)
        start, goal = [250., 250.], [1250., 550.]
        distance = float(np.linalg.norm(np.subtract(goal, start)))
        case = dict(seed=17, case_id='physical-smoke', kind='v9_random', terrain='synthetic',
                    world_size=1650., start=start, goal=goal, direct_distance_m=distance,
                    reference_distance_m=distance, circles_path=str(path),
                    circle_count=len(circles), circles_sha256=file_sha256(path))

        def trace(optimized):
            episode = GroundPolicyEpisode(repo, case, fast_seed=24, rollout_limit=4)
            try:
                exp = episode.env
                if optimized:
                    use_bounded_task_mask(exp)
                assert exp.world_size == 1650. and len(exp.ids) == 3
                np.testing.assert_array_equal(exp.start, start)
                np.testing.assert_array_equal(exp.goal, goal)
                physical = np.asarray([[*o.position, o.radius] for o in exp.env.obstacles])
                np.testing.assert_array_equal(physical, circles)
                initial = np.stack([exp.obs[rid]['self']['position'].copy() for rid in exp.ids])
                positions, velocities, rewards = [], [], []
                with torch.inference_mode():
                    for frame in range(4):
                        exp.step(frame)
                        positions.append([exp.obs[rid]['self']['position'].copy() for rid in exp.ids])
                        velocities.append(exp.last_executed_velocities.copy())
                        rewards.append(exp.trace[-1]['reward'])
                assert np.max(np.linalg.norm(np.asarray(positions[-1]) - initial, axis=1)) > .01
                return np.asarray(positions), np.asarray(velocities), np.asarray(rewards), exp.ledger.seen.copy()
            finally:
                episode.close()

        for left, right in zip(trace(False), trace(True)):
            np.testing.assert_array_equal(left, right)

        torch.manual_seed(25)
        survey_model = slow_model(SCHEMA, 16).eval()
        episode = SurveyFleetEpisode(repo, case, survey_model, fast_seed=24, budget_s=3300,
                                     collector_limit_s=16)
        try:
            fast_weights = {name: p.clone() for name, p in episode.fast_model.named_parameters()}
            assert not episode.fast_model.training and not any(
                p.requires_grad for p in episode.fast_model.parameters())
            assert episode.adapter.stats()['unique_received_tiles'] == 0
            while not episode.finished:
                episode.step()
            summary = episode.fleet_summary()
            assert summary['end_reason'] == 'collector_limit' and summary['sim_time_s'] == 16
            assert [e['time_s'] for e in summary['ground_replan_events']] == [0, 5, 10, 15]
            assert any(e['previous_target_reached'] is False and e['target'] != e['previous_target']
                       for e in summary['ground_replan_events'])
            decision = summary['survey_evaluations'][0]
            assert decision['survey_target'] is not None and decision['survey_target'] != decision['ground_target']
            request = summary['request_log'][0]
            assert request['target'] == decision['survey_target']
            assert request['first_physics_broadcast_s'] == request['issued_at_s'] == 0
            assert request['physics_broadcasts'] == 16
            assert set(request['heard_by']) == {c.drone_id for c in episode.env.controllers}
            assert all(0 <= hearing['source_time_s'] <= hearing['actor_time_s']
                       for hearing in request['heard_by'].values())
            assert all(d > 0. for d in summary['uav_travelled_m'].values())
            assert summary['travelled_m'] > 0. and summary['ground_received_packets'] > 0
            assert summary['adapter']['unique_received_tiles'] > 0 and summary['fusion_count'] > 0
            assert summary['adapter']['received_terrain_sources'] == list(episode.env.ids)
            assert sum(r['duration_s'] for r in summary['survey_evaluations']) == summary['sim_time_s']
            assert all(torch.equal(p, fast_weights[name]) for name, p in episode.fast_model.named_parameters())
        finally:
            episode.close()


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'native':
        native_check()
    else:
        unittest.main()
