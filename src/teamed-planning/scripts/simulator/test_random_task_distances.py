"""Random distances must stay bounded, reproducible and geometrically reachable."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic, GLOBAL_FEATURES
from drone.velocity_mappo import VelocityActorCritic
from simulator.large_world_scenarios import (generate_training_scene, generate_large_scene,
    distance_key, reference_geometry_clear)
from simulator.train_mappo import validate_config, train


class RandomTaskTests(unittest.TestCase):
    def test_random_pose_is_reproducible_and_has_continuous_distance(self):
        a = generate_training_scene(50010, (), distance_range=(100.,350.))
        b = generate_training_scene(50010, (), distance_range=(100.,350.))
        self.assertEqual(a, b)
        self.assertTrue(100. <= a['distance'] <= 350.)
        self.assertNotEqual(a['distance'], round(a['distance']))
        self.assertAlmostEqual(np.linalg.norm(np.asarray(a['goal'])-a['start']), a['distance'])
        self.assertTrue(a['reference_geometry_validated'])
        self.assertTrue(reference_geometry_clear(a['circles'], a['start'], a['goal'],
            a['references'][distance_key(a['distance'])]['points']))

    def test_sampler_spans_range_without_integer_rounding(self):
        def map_stub(seed, distances, world, start, heading):
            distance, = distances
            key = distance_key(distance)
            goal=start+distance*np.array([np.cos(heading),np.sin(heading)])
            return dict(start=start.tolist(),goals={key:goal.tolist()},
                        references={key:dict(length=distance,points=[start.tolist(),goal.tolist()])})
        with patch('simulator.large_world_scenarios.generate_large_scene', side_effect=map_stub):
            ds = [generate_training_scene(seed, (), distance_range=(100.,350.))['distance']
                  for seed in range(4000)]
        self.assertGreaterEqual(min(ds), 100.)
        self.assertLessEqual(max(ds), 350.)
        self.assertLess(min(ds), 102.)
        self.assertGreater(max(ds), 348.)
        self.assertAlmostEqual(np.mean(ds), 225., delta=5.)
        self.assertEqual(len(set(ds)), len(ds))

    def test_clearance_audit_rejects_obstacle_crossing_and_bad_access(self):
        circles = [([50.,50.], 5.)]
        self.assertFalse(reference_geometry_clear(circles,[20.,50.],[80.,50.],[[30.,50.],[70.,50.]],100.))
        # A free grid point cannot rescue an actual endpoint inside an obstacle.
        self.assertFalse(reference_geometry_clear(circles,[50.,50.],[80.,50.],[[70.,50.]],100.))
        self.assertTrue(reference_geometry_clear(circles,[20.,30.],[80.,30.],[[40.,30.],[60.,30.]],100.))

    def test_failed_geometry_sample_is_regenerated(self):
        with patch('simulator.large_world_scenarios.reference_geometry_clear', side_effect=[False,True]) as audit:
            scene = generate_large_scene(7, (40.5,), world=120., density=0.,start=[30.,60.])
        self.assertEqual(scene['attempts'], 2)
        self.assertEqual(audit.call_count, 2)

    def test_float_distance_keys_do_not_collapse(self):
        scene = generate_large_scene(7, (40.,40.1,40.9), world=120.,density=0.,start=[30.,60.])
        self.assertEqual(set(scene['goals']), {'40','40.1','40.9'})
        self.assertEqual(len(scene['references']), 3)

    def test_config_range_bounds_and_no_ambiguous_distribution(self):
        base = dict(action_mode='velocity',exploration_candidates=0,world_size=1000.,
                    task_distances=[],task_distance_range=[100.,350.])
        validate_config(base)
        for changes in [dict(task_distance_range=[100.,351.]),dict(task_distance_range=[0.,350.]),
                        dict(task_distance_range=[350.,100.]),dict(task_distance_range=[100.]),
                        dict(task_distance_range=[100.,float('nan')]),dict(task_distance_range='random'),
                        dict(task_distances=[250]),dict(world_size=120.)]:
            with self.assertRaises(ValueError):
                validate_config(dict(base, **changes))

    def test_velocity_warm_start_retains_weights_but_rejects_candidate_model(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = VelocityActorCritic()
            torch.save(dict(model=original.state_dict(),critic_global_features=GLOBAL_FEATURES,
                feature_version=2,action_mode='velocity',training_episodes=19),root/'init.pt')
            torch.save(dict(model=ActorCritic(GLOBAL_FEATURES).state_dict(),critic_global_features=GLOBAL_FEATURES,
                feature_version=1,action_mode='candidate'),root/'candidate.pt')
            config = validate_config(dict(action_mode='velocity',exploration_candidates=0,
                world_size=1000.,task_distances=[],task_distance_range=[100.,350.],initial_validation=False,
                init_checkpoint=str(root/'init.pt')))
            (root/'new').mkdir();(root/'new/STOP').touch()
            with contextlib.redirect_stdout(io.StringIO()):
                train(config,root/'new',workers=2)
            saved = torch.load(root/'new/latest.pt',map_location='cpu',weights_only=True)
            self.assertTrue(all(torch.equal(v,saved['model'][k]) for k,v in original.state_dict().items()))
            self.assertEqual(saved['training_episodes'], 0)
            self.assertEqual(saved['config']['task_distance_range'], [100.,350.])
            with self.assertRaises(ValueError):
                train(dict(config,init_checkpoint=str(root/'candidate.pt')),root/'bad')


if __name__ == '__main__':
    unittest.main()
