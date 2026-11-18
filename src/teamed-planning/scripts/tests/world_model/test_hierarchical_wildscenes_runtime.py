"""Real raster setup preserves unknown support and swept ground safety."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.hierarchical_runtime import HierarchicalEpisode
from world_model.graph_fast_policy import numpy_state
from world_model.graph_rollouts import collect_graph
from world_model.hierarchical_policy import make_graph_fast, GraphSlowActorCritic
from world_model.graph_dynamics import GraphWorldModel
from world_model.frozen_navigation import runtime
from world_model.wildscenes_runtime import (load_map_backend, load_forest,
    receive_known_evidence, install_observation_support, pool_received_tile, filter_ground_motion)
from world_model.wildscenes_runtime import (sensor_support_at, validate_sensor_route_support,
                                     local_safety_observation, world_sensor_tile)

REPO = Path(__file__).resolve().parents[5]


def case_for(path, **extra):
    result = dict(seed=192, kind='wildscenes', terrain='wildscenes_raster',
        map_path=str(path.resolve()), map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        world_size=200., resolution_m=1., required_clearance_m=1.5,
        source_unknown_space='blocked', map_group='fixture', start=[80., 100.],
        goal=[140., 100.], circles=[], direct_distance_m=60., reference_distance_m=180.)
    result.update(extra)
    return result


class WildScenesRuntimeTests(unittest.TestCase):
    def test_world_ground_tile_keeps_off_center_obstacles_and_unknown_support(self):
        cells = np.zeros((16, 16), np.uint8)
        cells[0, 0] = cells[0, 4] = 1
        known = np.ones_like(cells, bool)
        known[0, 4] = known[1, 1] = False
        occupied, observed = world_sensor_tile(dict(occupancy_map=cells, map_resolution=.5,
                                                   occupancy_known_mask=known))
        self.assertEqual((occupied.shape, occupied.dtype), ((12, 12), np.float32))
        self.assertEqual((observed.shape, observed.dtype), ((12, 12), np.bool_))
        expected_known = np.zeros((12, 12), bool)
        expected_known[4:8, 4:8] = True
        expected_known[4, 5] = False
        np.testing.assert_array_equal(observed, expected_known)
        np.testing.assert_array_equal(np.argwhere(occupied), [[4, 4]])

    def test_world_tile_partial_footprint_preserves_positive_area_obstacle_overlap(self):
        # The sensor spans [-1.5, 1.5] m in each axis. Its centre source cell
        # overlaps four 2 m world cells, each only partly within the footprint.
        cells = np.zeros((3, 3), np.int8)
        clear = dict(occupancy_map=cells, map_resolution=1.)
        occupied, observed = world_sensor_tile(clear)
        self.assertFalse(observed.any())
        cells[1, 1] = 1
        occupied, observed = world_sensor_tile(clear)
        expected = np.zeros((12, 12), bool)
        expected[5:7, 5:7] = True
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(occupied, expected)
        ids, values = pool_received_tile(cells, (0., 0.), 1., 2., (12, 12), target_origin=(-12., -12.))
        self.assertEqual(len(ids), 4)
        self.assertFalse(values.any())

    def test_world_tile_preserves_native_uav_cells_and_disconnected_sensor(self):
        cells = np.zeros((12, 12), np.float32)
        cells[0, 0], cells[0, 1], cells[0, 2] = 1., -1., np.nan
        observation = dict(occupancy_map=cells, map_resolution=2.)
        occupied, known = world_sensor_tile(observation)
        np.testing.assert_array_equal(occupied, cells == 1.)
        np.testing.assert_array_equal(known, cells >= 0.)
        occupied, known = world_sensor_tile(dict(observation, sensor_valid=False))
        self.assertFalse(known.any())
        self.assertFalse(occupied.any())
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            world_sensor_tile(dict(observation, occupancy_known_mask=np.ones((1, 12), bool)))

    def test_simplified_protocol_requires_explicit_provenance_and_keeps_unseen_evidence_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'legacy-scene.json'
            source.write_text('{"world_size":200,"obstacles":[]}')
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            path = directory / 'simplified.npz'
            occupied, known = np.zeros((200, 200), bool), np.ones((200, 200), bool)
            occupied[:, 160] = True
            provenance = dict(terrain_protocol='wildscenes_simplified_raster',
                source_unknown_space='free_by_simulation_assumption', coordinate_scale=1.,
                source_scene_path=str(source.resolve()), source_scene_sha256=source_hash)
            np.savez_compressed(path, occupied=occupied, known=known, resolution_m=1.,
                                origin_xy=np.zeros(2), **provenance)
            case = case_for(path, terrain=provenance['terrain_protocol'],
                **{key: value for key, value in provenance.items() if key != 'terrain_protocol'})
            forest = load_forest(case, REPO)
            self.assertTrue(forest.known.all())
            self.assertFalse(forest.point_clear([160.5, 100.], 0.))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                episode = HierarchicalEpisode(REPO, case, rollout_limit=1)
                try:
                    self.assertTrue(episode.raster_case)
                    self.assertFalse(episode.env.ledger.seen.any())
                    self.assertEqual(episode.belief()['grid'].cells[10, 10], 0)
                    obs = episode.observation()
                    self.assertNotIn('map_path', obs)
                    self.assertNotIn('source_scene_path', obs)
                    episode.execute(obs, obs['kinds'].index('hold'))
                    self.assertEqual(episode.frame, 1)
                    self.assertEqual(episode.summary()['ground_safety_violations'], 0)
                    self.assertEqual(episode.belief()['grid'].cells[10, 10], 0)
                    self.assertLess(episode.env.ledger.seen.sum(), 1000)
                finally:
                    episode.close()
            with self.assertRaisesRegex(ValueError, 'source_unknown_space'):
                load_forest(dict(case, source_unknown_space='blocked'), REPO)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                load_forest(dict(case, terrain='wildscenes_raster', source_unknown_space='blocked'), REPO)
            with self.assertRaisesRegex(ValueError, 'original scale'):
                load_forest(dict(case, coordinate_scale=10.), REPO)
            known[0, 0] = False
            np.savez_compressed(path, occupied=occupied, known=known, resolution_m=1.,
                                origin_xy=np.zeros(2), **provenance)
            changed = dict(case, map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'known everywhere'):
                load_forest(changed, REPO)
            known[:] = True
            np.savez_compressed(path, occupied=occupied, known=known, resolution_m=1.,
                                origin_xy=np.zeros(2), **provenance)
            changed = dict(case, map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            source.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'original scene SHA256'):
                load_forest(changed, REPO)

    def test_unknown_terrain_never_falls_back_to_an_empty_synthetic_map(self):
        with patch('world_model.hierarchical_runtime.runtime') as runtime_loader:
            with self.assertRaisesRegex(ValueError, 'Unsupported terrain'):
                HierarchicalEpisode(REPO, dict(terrain='wildscenes_simplifed_raster'))
            runtime_loader.assert_not_called()

    def test_offline_sensor_support_uses_native_local_geometry_without_pooling_inflation(self):
        backend = load_map_backend(REPO)
        occupied, known = np.zeros((160, 160), bool), np.ones((160, 160), bool)
        # Source unknown starts 1.6m east of the pose. Native .25m geometry
        # leaves 1.5m clearance; moving .5m pooling alone incorrectly blocks it.
        known[:, 80:] = False
        forest = backend.RasterForestMap(occupied, known, .25)
        point = [18.4, 20.]
        result = sensor_support_at(forest, point)
        self.assertTrue(result['source_point_clear'])
        self.assertTrue(result['supported'])
        observed = forest.local_observation(point, 16, .5)
        observation = dict(occupancy_map=np.where(observed['known_mask'],
                observed['occupancy_map'], -1), occupancy_known_mask=observed['known_mask'],
            map_resolution=.5, map_axes='yx', time_step=1.,
            self=dict(position=point, orientation=0.))
        _, legacy = filter_ground_motion({'motion': (1., 0.)}, observation, backend)
        self.assertFalse(legacy['current_capsule_observed_free'])
        observation['ground_safety_map'] = local_safety_observation(forest, point, 16, .5)
        safe, control = filter_ground_motion({'motion': (1., 0.)}, observation, backend)
        self.assertEqual(result['supported'], control['current_capsule_observed_free'])
        self.assertGreater(safe['motion'][0], .09)
        self.assertLess(safe['motion'][0], .1)
        self.assertFalse(sensor_support_at(forest, [18.6, 20.])['supported'])
        self.assertTrue(sensor_support_at(forest, [10., 20.])['supported'])

    def test_native_geometry_preserves_safe_endpoint_after_sensor_recentering(self):
        backend = load_map_backend(REPO)
        occupied, known = np.zeros((160, 160), bool), np.ones((160, 160), bool)
        occupied[:, 80:] = True
        forest = backend.RasterForestMap(occupied, known, .25)

        def observation_at(point, heading=0.):
            coarse = forest.local_observation(point, 16, .5)
            return dict(occupancy_map=coarse['occupancy_map'],
                occupancy_known_mask=coarse['known_mask'], map_resolution=.5,
                map_axes='yx', time_step=1., self=dict(position=point, orientation=heading),
                ground_safety_map=local_safety_observation(forest, point, 16, .5))

        before = np.asarray([18., 20.])
        first, _ = filter_ground_motion({'motion': (.4, 0.)}, observation_at(before), backend)
        after = before + [first['motion'][0], 0.]
        np.testing.assert_allclose(after, [18.4, 20.])
        self.assertTrue(forest.segment_clear(before, after, 1.5))
        current = observation_at(after, np.pi)
        # The previous bug accepted the first step, then permanently stopped
        # even a retreat because the newly pooled current capsule was blocked.
        legacy = {key: value for key, value in current.items() if key != 'ground_safety_map'}
        stalled, old_report = filter_ground_motion({'motion': (.5, 0.)}, legacy, backend)
        self.assertFalse(old_report['current_capsule_observed_free'])
        self.assertEqual(stalled['motion'][0], 0.)
        retreat, report = filter_ground_motion({'motion': (.5, 0.)}, current, backend)
        self.assertTrue(report['current_capsule_observed_free'])
        self.assertEqual(retreat['motion'][0], .5)
        self.assertTrue(forest.segment_clear(after, after - [.5, 0.], 1.5))
        blocked, _ = filter_ground_motion({'motion': (1., 0.)},
            observation_at(after), backend)
        self.assertLess(blocked['motion'][0], .1)
        self.assertTrue(forest.segment_clear(after, after + [blocked['motion'][0], 0.], 1.5))

    def test_native_safety_tile_exposes_only_fully_observed_local_cells(self):
        backend = load_map_backend(REPO)
        occupied, known = np.zeros((100, 100), bool), np.ones((100, 100), bool)
        forest = backend.RasterForestMap(occupied, known, .5, origin_xy=(-7., -11.))
        position = np.asarray([10.13, 9.17])
        tile = local_safety_observation(forest, position, 16, .5)
        self.assertEqual(set(tile), {'occupancy_map', 'known_mask', 'origin_xy',
                                    'resolution_m', 'map_axes'})
        self.assertLessEqual(tile['known_mask'].shape[0], 17)
        self.assertLessEqual(tile['known_mask'].shape[1], 17)
        ys, xs = np.indices(tile['known_mask'].shape)
        lower = np.asarray(tile['origin_xy']) + np.stack([xs, ys], -1) * .5
        inside = ((lower >= position - 4.).all(-1)
                  & (lower + .5 <= position + 4.).all(-1))
        np.testing.assert_array_equal(tile['known_mask'], inside)
        self.assertTrue(np.all(tile['occupancy_map'][~inside] == 1))
        self.assertFalse(np.shares_memory(tile['occupancy_map'], forest.occupied))
        self.assertFalse(np.shares_memory(tile['known_mask'], forest.known))
        # Change everything outside the observed full cells, including partial
        # edge cells. The local safety payload must remain exactly unchanged.
        source_ys, source_xs = np.indices(occupied.shape)
        source_lower = forest.origin_xy + np.stack([source_xs, source_ys], -1) * .5
        observed = ((source_lower >= position - 4.).all(-1)
                    & (source_lower + .5 <= position + 4.).all(-1))
        changed = backend.RasterForestMap(~observed, observed, .5, forest.origin_xy)
        other = local_safety_observation(changed, position, 16, .5)
        np.testing.assert_array_equal(other['occupancy_map'], tile['occupancy_map'])
        np.testing.assert_array_equal(other['known_mask'], tile['known_mask'])
        boundary = local_safety_observation(forest, [-6.8, -10.8], 16, .5)
        self.assertFalse(boundary['known_mask'].all())
        self.assertTrue(np.all(boundary['occupancy_map'][~boundary['known_mask']] == 1))

    def test_native_filter_blocks_unknown_invalid_sensor_and_outside_footprint(self):
        backend = load_map_backend(REPO)
        pixels = np.zeros((16, 16), np.int8)
        observation = dict(occupancy_map=pixels, map_resolution=.5, map_axes='yx',
            time_step=4., self=dict(position=[20., 20.], orientation=0.))
        # Even an incorrectly oversized safety tile must not authorize motion
        # beyond the actual 8m sensor footprint.
        geometry = dict(occupancy_map=np.zeros((100, 100), np.uint8),
            known_mask=np.ones((100, 100), bool), resolution_m=.5,
            origin_xy=[0., 0.], map_axes='yx')
        observation['ground_safety_map'] = geometry
        safe, _ = filter_ground_motion({'motion': (1., .2)}, observation, backend)
        self.assertGreater(safe['motion'][0], .62)
        self.assertLess(safe['motion'][0], .625)
        invalid, _ = filter_ground_motion({'motion': (1., .2)},
            dict(observation, sensor_valid=False), backend)
        self.assertEqual(invalid['motion'], (0., .2))
        unknown = dict(geometry, known_mask=np.zeros((100, 100), bool))
        stopped, _ = filter_ground_motion({'motion': (1., .2)},
            dict(observation, ground_safety_map=unknown), backend)
        self.assertEqual(stopped['motion'], (0., .2))

    def test_observation_installs_local_safety_tile_for_ground_only(self):
        backend = load_map_backend(REPO)
        forest = backend.RasterForestMap(np.zeros((100, 100), bool),
                                        np.ones((100, 100), bool), .5)
        def observations():
            return {name: dict(occupancy_map=np.zeros((16, 16)), map_resolution=.5,
                self=dict(position=[20.13, 20.17], orientation=0.))
                for name in ('ground_0', 'drone_0')}
        env = SimpleNamespace(ground_robots=[SimpleNamespace(robot_id='ground_0')],
            _get_observations=observations, _handle_collisions=lambda info: None)
        install_observation_support(env, forest, 1.5)
        result = env._get_observations()
        self.assertIn('ground_safety_map', result['ground_0'])
        self.assertNotIn('ground_safety_map', result['drone_0'])
        np.testing.assert_array_equal(result['ground_0']['occupancy_map'], np.zeros((16, 16)))

    def test_route_support_reports_sampled_subroutes_without_rejecting_alternatives(self):
        backend = load_map_backend(REPO)
        occupied, known = np.zeros((80, 80), bool), np.ones((80, 80), bool)
        known[39:42, 39:42] = False
        forest = backend.RasterForestMap(occupied, known, .5)
        result = validate_sensor_route_support(forest, [[10., 20.], [30., 20.]], max_spacing=.25)
        self.assertTrue(result['start']['supported'])
        self.assertTrue(result['goal']['supported'])
        self.assertGreater(result['unsupported_points'], 0)
        self.assertEqual(result['sampled_points'], 81)
        self.assertEqual(result['usable_points'] + result['unsupported_points'], 81)
        self.assertEqual(len(result['sample_supported_subroutes']), 2)
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['continuous_clearance_certified'])
        # A different parallel path is supported; failure above cannot be
        # interpreted as proof that the environment has no feasible route.
        other = validate_sensor_route_support(forest, [[10., 10.], [30., 10.]])
        self.assertTrue(other['route_samples_supported'])
        with self.assertRaisesRegex(ValueError, 'spacing'):
            validate_sensor_route_support(forest, [[10., 20.]], max_spacing=0.)

    def test_unknown_packet_pixels_cannot_become_received_free(self):
        ledger = SimpleNamespace(n=4, resolution=2., received=set(),
            log_odds=np.zeros((4, 4), np.float32), seen=np.zeros((4, 4), bool))
        pixels = np.asarray([[0., -1.], [1., -1.]])
        self.assertTrue(receive_known_evidence(ledger, 'a', 0, [2., 2.], pixels, 2.))
        np.testing.assert_array_equal(ledger.seen[:2, :2], [[True, False], [True, False]])
        self.assertLess(ledger.log_odds[0, 0], 0.)
        self.assertGreater(ledger.log_odds[1, 0], 0.)
        self.assertFalse(receive_known_evidence(ledger, 'a', 0, [2., 2.], pixels, 2.))
        # A single coarse cell containing only unknown and free pixels is not
        # promoted to observed free by subpixel averaging.
        other = SimpleNamespace(n=4, resolution=2., received=set(),
            log_odds=np.zeros((4, 4), np.float32), seen=np.zeros((4, 4), bool))
        receive_known_evidence(other, 'a', 0, [1., 1.], [[0., -1.], [0., 0.]], 1.)
        self.assertFalse(other.seen.any())
        # A shifted pixel centre lies in a receiver cell but does not cover
        # that cell completely. Its unobserved border must not become free.
        receive_known_evidence(other, 'a', 1, [1.9, 1.], [[0.]], 2.)
        self.assertFalse(other.seen.any())
        ids, values = pool_received_tile(np.zeros((4, 4)), [4.9, 4.], 1., 2., (8, 8))
        free = set(ids[values == -1])
        self.assertEqual(free, {1 * 8 + 2, 2 * 8 + 2})
        local_ids, local_values = pool_received_tile(np.zeros((4, 4)), [104.9, 204.],
            1., 2., (8, 8), target_origin=(100., 200.))
        np.testing.assert_array_equal(local_ids, ids)
        np.testing.assert_array_equal(local_values, values)

    def test_safety_capsule_blocks_unknown_and_counts_pre_rollback_proposal(self):
        backend = load_map_backend(REPO)
        for physical, unknown in ((False, False), (False, True), (True, True)):
            with self.subTest(physical=physical, unknown=unknown):
                occupied, known = np.zeros((30, 30), bool), np.ones((30, 30), bool)
                if unknown:
                    known[:, 11:] = False
                else:
                    occupied[:, 11] = True
                forest = backend.RasterForestMap(occupied, known, 1.)
                robot = SimpleNamespace(prev_position=np.asarray([8., 10., 0.]),
                    position=np.asarray([11.1 if physical else 9.6, 10., 0.]),
                    radius=.3, velocity=np.ones(3), robot_id='ground_0')
                env = SimpleNamespace(ground_robots=[robot], step_count=0,
                    _handle_collisions=lambda info: None, _get_observations=lambda: {})
                backend.install_raster_environment(env, forest)
                install_observation_support(env, forest, 1.5)
                info = {'collisions': []}
                env._handle_collisions(info)
                np.testing.assert_array_equal(robot.position, robot.prev_position)
                self.assertEqual(robot.radius, .3)
                self.assertEqual(env.ground_safety_violations, 1)
                self.assertEqual(env.raster_safety_unknown_attempts, int(unknown))
                self.assertEqual(env.raster_collision_attempts, int(physical))
                self.assertEqual(len(info['collisions']), 1)
                self.assertEqual(env.raster_safety_events[0]['physical_collision'], physical)

    def test_real_episode_preserves_unknown_and_uses_reference_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.npz'
            occupied, known = np.zeros((200, 200), bool), np.ones((200, 200), bool)
            known[96:104, 83:87] = False
            np.savez_compressed(path, occupied=occupied, known=known,
                                resolution_m=1., origin_xy=np.zeros(2))
            case = case_for(path)
            forest = load_forest(case, REPO)
            self.assertFalse(forest.point_clear([84., 100.], 0.))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_forest(dict(case, map_sha256='0' * 64), REPO)
            with self.assertRaisesRegex(ValueError, 'original scale'):
                load_forest(dict(case, resolution_m=2.), REPO)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                episode = HierarchicalEpisode(REPO, case, rollout_limit=1)
                try:
                    self.assertEqual(episode.mission_budget, 960)
                    local = episode.env.obs[episode.env.ground]
                    self.assertTrue((local['occupancy_map'] < 0).any())
                    np.testing.assert_array_equal(local['occupancy_known_mask'], local['occupancy_map'] >= 0)
                    belief = episode.belief()
                    self.assertEqual(belief['grid'].cells[50, 41], 0)
                    obs = episode.observation()
                    episode.execute(obs, obs['kinds'].index('hold'))
                    self.assertEqual(episode.frame, 1)
                    summary = episode.summary()
                    self.assertEqual(summary['termination'], 'rollout_limit')
                    self.assertEqual(summary['ground_safety_violations'], 0)
                    self.assertEqual(summary['raster_unknown_attempts'], 0)
                    self.assertEqual(len(episode.env.ground_controller._observed_obstacles), 0)
                    self.assertTrue((episode.env.ground_controller._last_semantic_map == 0).any())
                    self.assertNotIn('map_path', obs)
                    self.assertNotIn('reference_distance_m', obs)
                    # The signed UAV pixels travel through the actual audit
                    # packet path; delivered unknown is still unobserved.
                    for source in episode.env.controllers:
                        self.assertFalse(np.isnan(source._last_semantic_map).any())
                finally:
                    episode.close()
                # Bypass the rule planner's safe stop with an explicit unsafe
                # control proposal to exercise hierarchy failure bookkeeping.
                unsafe = HierarchicalEpisode(REPO, case, rollout_limit=20)
                try:
                    original = unsafe._base_ground_action

                    def unsafe_motion(neighbors=None):
                        return dict(original(neighbors), motion=(1., 0.))

                    unsafe._base_ground_action = unsafe_motion
                    obs = unsafe.observation()
                    with patch('world_model.wildscenes_runtime.filter_ground_motion',
                               side_effect=lambda action, *a, **k: (action, {'intervened': False})):
                        unsafe.execute(obs, obs['kinds'].index('final'))
                    summary = unsafe.summary()
                    self.assertTrue(summary['terminated'])
                    self.assertFalse(summary['truncated'])
                    self.assertFalse(summary['goal_reached'])
                    self.assertEqual(summary['termination'], 'safety_failure')
                    self.assertEqual(summary['ground_safety_violations'], 1)
                    self.assertEqual(summary['collisions'], 1)
                    self.assertEqual(summary['raster_collision_attempts'], 0)
                    self.assertEqual(summary['raster_unknown_attempts'], 1)
                    self.assertLess(unsafe.trajectory[-1][0], 82.)
                finally:
                    unsafe.close()

    def test_collector_safety_failure_terminates_fast_and_slow_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'map.npz'
            occupied, known = np.zeros((200, 200), bool), np.ones((200, 200), bool)
            known[96:104, 83:87] = False
            np.savez_compressed(path, occupied=occupied, known=known,
                                resolution_m=1., origin_xy=np.zeros(2))
            case = case_for(path)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            models = [make_graph_fast(REPO), GraphWorldModel(16, 4), GraphSlowActorCritic(hidden=16)]
            config = dict(hidden=16, horizon=4, option_steps=4, steps=8, stride=1,
                          loss=0., jitter=0, complete_missions=False, freeze_fast_world=True)

            def unsafe_episode(*args, **kwargs):
                result = HierarchicalEpisode(*args, **kwargs)
                original = result._base_ground_action

                def motion(neighbors=None):
                    return dict(original(neighbors), motion=(1., 0.))

                result._base_ground_action = motion
                return result

            with patch('world_model.graph_rollouts.HierarchicalEpisode', unsafe_episode), \
                    patch('world_model.wildscenes_runtime.filter_ground_motion',
                          side_effect=lambda action, *a, **k: (action, {'intervened': False})):
                row = collect_graph((str(REPO), case, config, str(output), 'unsafe',
                    *(numpy_state(model) for model in models), 0, 'warmup', False))
            self.assertEqual(row['termination'], 'safety_failure')
            self.assertEqual(row['ground_safety_violations'], 1)
            self.assertFalse(row['navigation_success'])
            with np.load(row['ppo_path'], allow_pickle=False) as fast:
                self.assertTrue(fast['terminals'][-1])
                self.assertEqual(float(fast['bootstrap']), 0.)
            with np.load(row['slow_path'], allow_pickle=False) as slow:
                self.assertTrue(slow['terminated'][-1])
                self.assertFalse(slow['truncated'][-1])
                self.assertEqual(float(slow['next_value'][-1]), 0.)

    def test_local_motion_filter_preserves_turn_and_checks_pre_turn_swept_capsule(self):
        backend = load_map_backend(REPO)
        pixels = np.zeros((16, 16), np.int8)
        # Sensor is [6,14]x[6,14], unknown begins at x=12. The actual old
        # heading is east; a simultaneous 90-degree turn happens AFTER travel.
        pixels[:, 12:] = -1
        observation = dict(occupancy_map=pixels, occupancy_known_mask=pixels >= 0,
            map_resolution=.5, map_axes='yx', time_step=1.,
            self=dict(position=[10., 10., 0.], orientation=0.))
        requested = dict(motion=(1., np.pi / 2), broadcast={'target': [20., 10.]})
        safe, report = filter_ground_motion(requested, observation, backend)
        self.assertTrue(report['intervened'])
        self.assertGreater(safe['motion'][0], .49)
        self.assertLess(safe['motion'][0], .5)
        self.assertEqual(safe['motion'][1], requested['motion'][1])
        self.assertEqual(requested['motion'][0], 1.)
        self.assertTrue(np.any(pixels < 0))
        runtime(REPO)
        from simulator.robots import GroundRobot
        robot = GroundRobot(0, np.asarray([10., 10., 0.]), orientation=0., max_speed=1.)
        robot.step(np.asarray(safe['motion']), 1.)
        np.testing.assert_allclose(robot.position, [10. + safe['motion'][0], 10., 0.])
        self.assertAlmostEqual(robot.orientation, np.pi / 2)
        # Shorter dt makes the original command safe without changing v.
        safe_short, report_short = filter_ground_motion(requested,
            dict(observation, time_step=.25), backend)
        self.assertFalse(report_short['intervened'])
        self.assertEqual(safe_short['motion'], requested['motion'])
        unknown = dict(observation, occupancy_map=-np.ones_like(pixels),
                       occupancy_known_mask=np.zeros_like(pixels, bool))
        stopped, report_unknown = filter_ground_motion(requested, unknown, backend)
        self.assertEqual(stopped['motion'], (0., np.pi / 2))
        self.assertFalse(report_unknown['current_capsule_observed_free'])
        # A long dt may leave the actual sensor footprint; it is also limited.
        free = dict(observation, occupancy_map=np.zeros_like(pixels),
                    occupancy_known_mask=np.ones_like(pixels, bool), time_step=4.)
        bounded, _ = filter_ground_motion(requested, free, backend)
        self.assertLess(bounded['motion'][0], .625)
        self.assertGreater(bounded['motion'][0], .62)


if __name__ == '__main__':
    unittest.main(verbosity=2)
