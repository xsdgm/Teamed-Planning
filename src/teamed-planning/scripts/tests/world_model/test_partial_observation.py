import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.frozen_navigation import runtime
from world_model.partial_observation import SurveySwaths, partial_v9_runtime
from world_model.wildscenes_runtime import load_map_backend, raster_runtime, world_sensor_tile

REPO = Path(__file__).resolve().parents[5]


class AerialSurveyTests(unittest.TestCase):
    def test_shared_geographic_footprints_and_visibility_range(self):
        positions = [(x + .25, y + .1) for x in range(45, 700, 33) for y in range(40, 700, 79)]
        first = SurveySwaths.from_seed(113)
        replay = SurveySwaths.from_seed(113)
        other = SurveySwaths.from_seed(114)
        coverage = []
        changed_seeds = 0
        for x, y in positions:
            mask = first.known_mask((x, y), (12, 12), 2.)
            np.testing.assert_array_equal(mask, replay.known_mask((x, y), (12, 12), 2.))
            shifted = first.known_mask((x + 2., y - 4.), (12, 12), 2.)
            np.testing.assert_array_equal(mask[:-2, 1:], shifted[2:, :-1])
            coverage.append(mask.sum())
            changed_seeds += not np.array_equal(mask, other.known_mask((x, y), (12, 12), 2.))
        self.assertIn(0, coverage)
        self.assertIn(144, coverage)
        self.assertTrue(any(0 < count < 144 for count in coverage))
        self.assertGreater(changed_seeds, len(positions) // 4)

    def test_native_sensor_actor_world_evidence_and_wildscenes_isolation(self):
        me, torch, model, options = runtime(REPO)
        options = dict(options, world_size=200., start=(80., 100.), subgoal_radius=None)
        goal = np.array([140., 100.])
        options['scene'] = (goal, [((92., 100.), 4.)], 1)
        constructors = me.MultiRobotEnv, me.DroneController, me.DeliveredEvidence
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            with partial_v9_runtime(me, 113):
                experiment = me.Experiment(113, model, **options)
            self.addCleanup(experiment.env.close)
            self.assertEqual(constructors, (me.MultiRobotEnv, me.DroneController, me.DeliveredEvidence))
            native = experiment.env
            # The measurement overlay does not replace any physical geometry or collision code.
            self.assertNotIn('_handle_collisions', vars(native))
            car = experiment.obs[experiment.ground]
            self.assertNotIn('occupancy_known_mask', car)
            np.testing.assert_array_equal(car['occupancy_map'], native._generate_occupancy_map(
                car['self']['position'], 'ground'))
            encountered = []
            packet_sources = {}
            for frame in range(6):
                before = {rid: experiment.obs[rid] for rid in experiment.ids}
                for rid, obs in before.items():
                    known = obs['occupancy_known_mask']
                    source = native._generate_occupancy_map(obs['self']['position'], 'drone')
                    self.assertTrue(obs['sensor_valid'])
                    np.testing.assert_array_equal(obs['occupancy_map'][known], source[known])
                    self.assertTrue((obs['occupancy_map'][~known] == -1).all())
                    encountered.append(known.mean())
                    packet_sources[rid, frame] = obs['occupancy_map'].copy()
                samples, _, metrics = experiment.step(frame)
                np.testing.assert_array_equal([metrics['reward_route_target_x'], metrics['reward_route_target_y']], goal)
                for rid, sample, ctrl in zip(experiment.ids, samples, experiment.controllers):
                    obs = before[rid]
                    occupied, known = world_sensor_tile(obs)
                    actor = sample[0][34:322].reshape(2, 12, 12)
                    # The native actor uses camera XY; the dynamics model uses world YX.
                    np.testing.assert_array_equal(actor[0].T, occupied)
                    np.testing.assert_array_equal(actor[1].T, ~known)
                    semantic = ctrl.semantic_processor.process_occupancy_map(obs['occupancy_map'])
                    self.assertTrue((semantic[~known] == 0).all())
                    self.assertFalse(ctrl.graph_builder._create_traversability_mask(semantic)[~known].any())
                    for packet in experiment.audit_buffers[experiment.ids.index(rid)].values():
                        np.testing.assert_array_equal(packet['occupancy'], packet_sources[packet['source'], packet['frame']])
            self.assertTrue(any(0 < item < 1 for item in encountered))
            self.assertTrue(experiment.ledger.seen.any())
            isolated = type(experiment.ledger)(world=80.)
            pixels = np.full((12, 12), -1, np.int8)
            pixels[5, 4:6] = [0, 1]
            isolated.receive(3, 0, (32., 32.), pixels, 2.)
            self.assertEqual(int(isolated.seen.sum()), 2)
            self.assertEqual(int((isolated.log_odds < 0).sum()), 1)
            self.assertEqual(int((isolated.log_odds > 0).sum()), 1)
            # A physical obstacle also collides inside a wholly unsurveyed patch.
            swaths = SurveySwaths.from_seed(113)
            center = next((x, y) for x in range(25, 180, 15) for y in range(25, 180, 15)
                          if not swaths.known_mask((x, y), (12, 12), 2.).any())
            native.add_circle_obstacle(*center, 2.)
            car = native.ground_robots[0]
            car.prev_position = car.position.copy()
            car.position[:2] = center
            info = {'collisions': []}
            native._handle_collisions(info)
            self.assertEqual(info['collisions'][0]['obstacle_type'], 'circle')
            np.testing.assert_array_equal(car.position, car.prev_position)
            # Use a strict raster fixture with a known ground corridor; its source
            # support must survive a previous synthetic episode in the same worker.
            backend = load_map_backend(REPO)
            known = np.ones((200, 200), bool)
            known[115:, :] = False
            forest = backend.RasterForestMap(np.zeros_like(known), known, 1., np.zeros(2))
            with raster_runtime(me, forest, backend):
                real = me.Experiment(113, model, **dict(options, scene=(goal, [], 1)))
            self.addCleanup(real.env.close)
            for rid in real.ids:
                obs = real.obs[rid]
                expected = forest.local_observation(obs['self']['position'], (12, 12), 2.)
                np.testing.assert_array_equal(obs['occupancy_known_mask'], expected['known_mask'])
                np.testing.assert_array_equal(world_sensor_tile(obs)[1], expected['known_mask'])


if __name__ == '__main__':
    unittest.main()
