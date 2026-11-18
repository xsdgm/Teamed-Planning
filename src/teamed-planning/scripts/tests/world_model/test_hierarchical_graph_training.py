"""Staged optimization uses graph predictions and duration-aware slow labels."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.graph_fast_policy import numpy_state, state_hash
from world_model.hierarchical_policy import make_graph_fast, FAST_SCHEMA, SLOW_SCHEMA
from world_model.graph_dynamics import GraphWorldModel, LOCAL_EVIDENCE_WIDTH, MODEL_SCHEMA
from world_model.train_hierarchical import (phase_schedule, world_update, distill_fast,
                                evaluate_world, slow_episode, require_finite_optimizer,
                                slow_behavior_clone, initialize_graph_modules,
                                curriculum_ranges, parse_args, validate_resume_schema, TRAIN_SCHEMA,
                                validate_scene_seeds, build_scene_cases, initialize_slow_module,
                                summarize_missions, snapshot, training_batches, evaluation_due,
                                pending_checkpoint_evaluation, restore_training_state, verify_continuation_parent)
from world_model.scene_manifest import file_sha256, describe_scene_assumptions


def dataset():
    history = np.zeros((4, 4, 8, 14), np.float32)
    history[..., 2] = .1
    history[..., 9] = .2
    history[:, 1, :, 9] = 0.
    ego = np.zeros((4, 4), bool)
    ego[:, 1] = True
    states = np.zeros((4, 4, 5, 4), np.float32)
    states[..., 0] = np.arange(5) * .03
    states[..., 2] = .1
    actions = np.zeros((4, 4, 4, 2), np.float32)
    actions[..., 0] = .1
    targets = states.copy()
    targets[:, 2:, :, 0] += .2
    return dict(history=history, history_mask=np.ones((4, 4, 8), bool),
        context=np.zeros((4, 4, 20), np.float32), baseline_states=states,
        local_evidence=np.zeros((4, LOCAL_EVIDENCE_WIDTH), np.float32),
        baseline_actions=actions, node_mask=np.ones((4, 4), bool),
        node_type=np.tile([1, 0, 0, 0], (4, 1)), ego_mask=ego, state_observed=ego.copy(),
        target_states=targets, target_actions=actions.copy(),
        state_mask=np.ones((4, 4, 5), bool), action_mask=np.ones((4, 4, 4), bool),
        raw_features=np.zeros((4, 1826), np.float32), teacher_mean=np.zeros((4, 2), np.float32),
        teacher_std=np.ones((4, 2), np.float32), actor_valid=np.asarray([True, True, True, False]))


def scene_config():
    return dict(warmup_updates=1, fast_updates=0, slow_updates=1, joint_updates=0,
                warmup_distances_m=[30., 60.], mission_distances_m=[100., 200.],
                episodes_per_update=2, eval_scenes=2, seed=100, world_size=2000.)


def raster_manifest(directory):
    result = dict(train=[], validation=[])
    for split, sequence, group in [('train', 'K-01', 'Karawatha'), ('validation', 'V-01', 'Venman')]:
        path = Path(directory) / (sequence + '.npz')
        occupied = np.zeros((256, 256), np.uint8)
        occupied[200, 200 if split == 'train' else 210] = 1
        np.savez_compressed(path, occupied=occupied, known=np.ones_like(occupied),
                            origin_xy=[0., 0.], resolution_m=1.)
        for index, distance in enumerate([40., 40., 120., 120.] if split == 'train' else [120., 120.]):
            result[split].append(dict(terrain='wildscenes_raster', kind='real_forest',
                map_path=str(path.resolve()), map_sha256=file_sha256(path), map_group=group, sequence=sequence,
                world_size=256., start=[60., 60.], goal=[60. + distance, 60.], circles=[],
                direct_distance_m=distance, reference_distance_m=distance + 10.,
                seed=index + (0 if split == 'train' else 20), resolution_m=1.,
                required_clearance_m=1.5, source_unknown_space='blocked'))
    return result


def simplified_raster_manifest(directory):
    result = raster_manifest(directory)
    for cases in result.values():
        path = Path(cases[0]['map_path'])
        source = path.with_suffix('.source.json')
        source.write_text('{}\n', encoding='utf-8')
        provenance = dict(terrain_protocol='wildscenes_simplified_raster',
            source_unknown_space='free_by_simulation_assumption', coordinate_scale=1.,
            source_scene_path=str(source), source_scene_sha256=file_sha256(source))
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        np.savez_compressed(path, **arrays, **provenance)
        for case in cases:
            case.update({key: value for key, value in provenance.items() if key != 'terrain_protocol'})
            case.update(terrain=provenance['terrain_protocol'], map_sha256=file_sha256(path))
    return result


class GraphTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_stages_keep_declared_order_and_allow_skipping_empty_stage(self):
        self.assertEqual(phase_schedule(dict(warmup_updates=2, fast_updates=1, slow_updates=0,
                                             joint_updates=1)), ['warmup', 'warmup', 'fast', 'joint'])

    def test_variable_batches_preserve_global_versions_and_original_evaluation_events(self):
        from world_model.training_schedule import create_continuation_plan
        config = dict(scene_config(), warmup_updates=8, fast_updates=8, slow_updates=64, joint_updates=32,
                      episodes_per_update=4, eval_scenes=9, eval_every=8)
        config['scene_manifest'] = dict(train=[dict(seed=i) for i in range(448)],
                                        validation=[dict(seed=1000+i) for i in range(9)])
        parent = dict(config=config, completed_updates=5, manifest=dict(source_sha256={}))
        continued = create_continuation_plan(parent)['config']
        self.assertEqual(continued['policy_version_offset'], 5)
        self.assertEqual(phase_schedule(continued), ['warmup'] + ['fast'] * 2 + ['slow'] * 16 + ['joint'] * 8)
        batches = training_batches(continued)
        self.assertEqual(batches[:3], [(0, 12), (12, 16), (28, 16)])
        indices = [i for start, count in batches for i in range(start, start + count)]
        self.assertEqual(indices, list(range(428)))
        self.assertEqual([i for i in range(1, 28) if evaluation_due(continued, i)], list(range(1, 28, 2)))
        self.assertEqual(curriculum_ranges(continued), [row['distance_range_m'] for row in continued['update_plan']])
        pending = pending_checkpoint_evaluation(dict(config=continued, completed_updates=1, records=[]))
        self.assertEqual(pending, dict(version=6, phase='warmup'))

    def test_legacy_and_explicit_pending_evaluations_survive_stop_boundaries(self):
        config = dict(scene_config(), warmup_updates=8, fast_updates=0, slow_updates=0,
                      eval_every=8, eval_scenes=2)
        saved = dict(config=config, completed_updates=8, records=[])
        self.assertEqual(pending_checkpoint_evaluation(saved), dict(version=8, phase='warmup'))
        saved['records'] = [dict(label='validation-' + str(i), evaluation=True, version=8) for i in range(2)]
        self.assertIsNone(pending_checkpoint_evaluation(saved))
        saved.update(pending_evaluation=dict(version=8, phase='warmup'))
        self.assertEqual(pending_checkpoint_evaluation(saved), saved['pending_evaluation'])
        saved['pending_evaluation'] = None
        self.assertIsNone(pending_checkpoint_evaluation(saved))
        self.assertEqual(training_batches(config), [(i * 2, 2) for i in range(8)])

    def test_full_state_restore_keeps_all_models_adam_moments_and_rng(self):
        def learners():
            models = {name: torch.nn.Linear(2, 2) for name in ('fast', 'world', 'slow')}
            optimizers = dict(actor_optimizer=torch.optim.Adam([models['fast'].weight], lr=.002),
                critic_optimizer=torch.optim.Adam([models['fast'].bias], lr=.003),
                world_optimizer=torch.optim.Adam(models['world'].parameters(), lr=.004),
                slow_optimizer=torch.optim.Adam(models['slow'].parameters(), lr=.005))
            return models, optimizers
        models, optimizers = learners()
        for _ in range(2):
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            sum(model(torch.ones(1, 2)).square().sum() for model in models.values()).backward()
            for optimizer in optimizers.values():
                optimizer.step()
        original_rng = np.random.default_rng(12)
        original_rng.normal(size=7)
        saved = {**{key: copy.deepcopy(model.state_dict()) for key, model in models.items()},
                 **{key: copy.deepcopy(optimizer.state_dict()) for key, optimizer in optimizers.items()},
                 'torch_rng': torch.get_rng_state().clone(), 'numpy_rng': original_rng.bit_generator.state,
                 'cuda_rng': None}
        expected_torch = torch.rand(5)
        expected_numpy = original_rng.random(5)
        restored_models, restored_optimizers = learners()
        restored_rng = np.random.default_rng(999)
        restore_training_state(saved, restored_models, restored_optimizers, restored_rng, torch.device('cpu'))
        for key, model in restored_models.items():
            self.assertEqual(state_hash(numpy_state(model)), state_hash(numpy_state(models[key])))
        for key, optimizer in restored_optimizers.items():
            state = optimizer.state_dict()
            self.assertEqual(state['param_groups'], saved[key]['param_groups'])
            for index, values in state['state'].items():
                for field, value in values.items():
                    self.assertTrue(torch.equal(value, saved[key]['state'][index][field]))
        self.assertTrue(torch.equal(torch.rand(5), expected_torch))
        np.testing.assert_array_equal(restored_rng.random(5), expected_numpy)

    def test_continuation_cli_requires_an_explicit_plan_and_rejects_resetting_initializers(self):
        base = ['train_hierarchical.py', '--output', '/tmp/new-run', '--continue-from', '/tmp/parent/latest.pt']
        with patch.object(sys, 'argv', base), patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args()
        args = base + ['--continuation-plan', '/tmp/rebatch.json', '--workers', '16']
        with patch.object(sys, 'argv', args):
            parsed = parse_args()
            self.assertEqual(parsed.continue_from, Path('/tmp/parent/latest.pt'))
            self.assertEqual(parsed.workers, 16)
        for extra in (['--resume'], ['--graph-init', '/tmp/graph.pt'], ['--slow-init', '/tmp/slow.pt'],
                      ['--scene-manifest', '/tmp/cases.json']):
            with patch.object(sys, 'argv', args + extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args()

    def test_continuation_verifies_parent_source_checkpoint_and_replay_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'checkpoints' / 'latest.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'parent checkpoint identity')
            source = run / 'runtime_sources' / 'vtemp2'
            source.mkdir(parents=True)
            (source / 'train_graph.py').write_text('# frozen parent\n')
            hashes = {'train_graph.py': file_sha256(source / 'train_graph.py')}
            replay = run / 'world.npz'
            replay.write_bytes(b'persisted replay')
            saved = dict(schema=TRAIN_SCHEMA, world_schema=MODEL_SCHEMA, completed_updates=5,
                manifest=dict(fast_schema=FAST_SCHEMA, slow_schema=SLOW_SCHEMA, source_sha256=hashes),
                replay_paths=[str(replay)])
            plan = dict(parent=dict(checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                        completed_updates=5, source_sha256=hashes), audit={})
            verify_continuation_parent(saved, checkpoint, plan)
            replay.unlink()
            with self.assertRaisesRegex(ValueError, 'replay dataset is missing'):
                verify_continuation_parent(saved, checkpoint, plan)
            replay.write_bytes(b'persisted replay')
            (source / 'train_graph.py').write_text('# mutated parent\n')
            with self.assertRaisesRegex(ValueError, 'source snapshot changed'):
                verify_continuation_parent(saved, checkpoint, plan)
            (source / 'train_graph.py').write_text('# frozen parent\n')
            backend = source / '_wildscenes_map_backend.py'
            backend.write_text('# unchanged geometry backend\n')
            hashes[backend.name] = file_sha256(backend)
            development = run / 'development' / 'world_model'
            development.mkdir(parents=True)
            standalone = development.parent / 'simulator' / 'wildscenes_map.py'
            standalone.parent.mkdir()
            standalone.write_bytes(backend.read_bytes())
            with patch('world_model.train_hierarchical.SOURCE', development):
                verify_continuation_parent(saved, checkpoint, plan)
            archived = run / 'child' / 'runtime_sources' / 'world_model'
            archived.mkdir(parents=True)
            with patch('world_model.train_hierarchical.SOURCE', archived), self.assertRaisesRegex(ValueError, 'omit'):
                verify_continuation_parent(saved, checkpoint, plan)

    def test_world_updates_then_distillation_freezes_world_and_updates_fast(self):
        data = dataset()
        world = GraphWorldModel(16, 4)
        optimizer = torch.optim.Adam(world.parameters(), lr=.001)
        metrics = world_update(world, optimizer, data, 'cpu', np.random.default_rng(1),
                               epochs=2, batch_size=2)
        self.assertEqual(metrics['optimizer_steps'], 4)
        self.assertGreater(metrics['parameter_delta'], 0.)
        fast = make_graph_fast(Path(__file__).resolve().parents[5])
        with torch.no_grad():
            teacher = fast.base.distribution(torch.from_numpy(data['raw_features']))
            data['teacher_mean'] = teacher.mean.numpy() + .1
            data['teacher_std'] = teacher.stddev.numpy()
        world_before = state_hash(numpy_state(world))
        actor = torch.optim.Adam(fast.actor_parameters(), lr=.0001)
        metrics = distill_fast(fast, world, actor, data, 'cpu', np.random.default_rng(2),
                              epochs=2, batch_size=2)
        self.assertEqual(metrics['samples'], 3)
        self.assertGreater(metrics['parameter_delta'], 0.)
        self.assertEqual(world_before, state_hash(numpy_state(world)))
        report = evaluate_world(world, data, 'cpu', batch_size=2)
        self.assertGreater(report['age_5_to_15s']['valid_state_targets'], 0)
        self.assertEqual(report['age_at_least_15s']['valid_state_targets'], 0)
        self.assertTrue(np.isfinite(report['objective']))

    def test_slow_rollout_restores_python_duration_and_terminal_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slow.npz'
            np.savez_compressed(path, action=np.asarray([1, 2]), duration=np.asarray([4, 7]),
                terminated=np.asarray([False, True]), truncated=np.asarray([False, False]),
                reward=np.asarray([.2, 1.]), features=np.zeros((2, 3, 4), np.float32))
            rows = slow_episode(path)
        self.assertIs(type(rows[0]['duration']), int)
        self.assertIs(type(rows[0]['terminated']), bool)
        self.assertEqual(rows[1]['duration'], 7)
        self.assertEqual(rows[0]['features'].shape, (3, 4))

    def test_ragged_environment_graphs_round_trip_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slow.npz'
            np.savez_compressed(path, action=np.asarray([1, 0]), teacher_action=np.asarray([0, 1]),
                duration=np.asarray([4, 7]), features=np.zeros((2, 2, 3), np.float32))
            graphs = {}
            for i, count in enumerate((3, 9)):
                graphs.update({f'g{i}_nodes': np.ones((count, 4), np.float32),
                    f'g{i}_edges': np.asarray([[0, 1], [1, 0]], np.int64),
                    f'g{i}_edge_types': np.asarray([0, 0], np.int64),
                    f'g{i}_candidate_xy': np.zeros((2, 2), np.float32)})
            np.savez_compressed(path.with_suffix('.graphs.npz'), **graphs)
            rows = slow_episode(path)
            self.assertEqual([row['environment_graph']['nodes'].shape[0] for row in rows], [3, 9])
            self.assertEqual(rows[0]['teacher_action'], 0)
            graphs.pop('g1_nodes')
            np.savez_compressed(path.with_suffix('.graphs.npz'), **graphs)
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                slow_episode(path)

    def test_slow_imitation_increases_teacher_probability_without_updating_critic(self):
        class GraphActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor_encoder = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh())
                self.actor_head = torch.nn.Linear(8, 1)
                self.critic = torch.nn.Linear(3, 1)
                self.graph_calls = 0

            def forward(self, x, mask, present, graphs):
                self.graph_calls += 1
                self.assertion = len(graphs) == len(x)
                context = torch.tensor([graph['nodes'][0, 0] for graph in graphs])
                z = x + context[:, None, None] * .01
                logits = self.actor_head(self.actor_encoder(z)).squeeze(-1).masked_fill(~mask, -torch.inf)
                return torch.distributions.Categorical(logits=logits), self.critic(z.mean(1)).squeeze(-1)

        model = GraphActor()
        rows = [dict(features=np.asarray([[1., 0., 0.], [0., 1., 0.]], np.float32),
                     mask=np.ones(2, bool), present=np.ones(2, bool), teacher_action=1,
                     environment_graph=dict(nodes=np.ones((i + 1, 1), np.float32))) for i in range(2)]
        x = torch.tensor(np.stack([row['features'] for row in rows]))
        mask = torch.ones(2, 2, dtype=torch.bool)
        graphs = [row['environment_graph'] for row in rows]
        with torch.no_grad():
            before = model(x, mask, mask, graphs)[0].probs[:, 1].mean().item()
        critic_before = state_hash(numpy_state(model.critic))
        result = slow_behavior_clone(model, torch.optim.Adam(model.parameters(), lr=.03), [rows],
                                     epochs=12, batch_size=2)
        with torch.no_grad():
            after = model(x, mask, mask, graphs)[0].probs[:, 1].mean().item()
        self.assertGreater(after, before + .1)
        self.assertEqual(critic_before, state_hash(numpy_state(model.critic)))
        self.assertEqual(result['optimizer_steps'], 12)
        self.assertEqual(result['informative_samples'], 2)
        self.assertGreater(result['parameter_delta'], 0.)
        for row in rows:
            row['mask'][0] = False
        model_before = state_hash(numpy_state(model))
        singleton = slow_behavior_clone(model, torch.optim.Adam(model.parameters()), [rows])
        self.assertEqual(singleton['optimizer_steps'], 0)
        self.assertEqual(singleton['informative_samples'], 0)
        self.assertEqual(singleton['singleton_samples'], 2)
        self.assertEqual(model_before, state_hash(numpy_state(model)))
        rows[0]['mask'][1] = False
        with self.assertRaisesRegex(ValueError, 'masked action'):
            slow_behavior_clone(model, torch.optim.Adam(model.parameters()), [rows])

    def test_missions_default_to_complete_and_curriculum_reaches_requested_range(self):
        with patch.object(sys, 'argv', ['train_hierarchical.py', '--output', '/tmp/unused-graph-test']):
            args = parse_args()
        self.assertTrue(args.complete_missions)
        self.assertEqual(args.option_steps, 80)
        config = dict(warmup_updates=1, fast_updates=1, slow_updates=2, joint_updates=1,
                      warmup_distances_m=[30., 60.], mission_distances_m=[100., 200.])
        self.assertEqual(curriculum_ranges(config), [[30., 60.], [30., 60.], [65., 130.],
                                                     [100., 200.], [100., 200.]])
        with patch.object(sys, 'argv', ['train_hierarchical.py', '--output', '/tmp/unused-graph-test',
                                       '--no-complete-missions']):
            self.assertFalse(parse_args().complete_missions)

    def test_scene_seed_manifest_requires_exact_lengths_and_structure(self):
        config = scene_config()
        valid = dict(train=[10, 11, 12, 13], validation=[20, 21])
        self.assertEqual(validate_scene_seeds(valid, config), valid)
        for invalid in [[], {'train': valid['train']}, dict(valid, unexpected=[]),
                        dict(valid, train=[10, 11, 12]), dict(valid, train=[10, 11, 12, 13, 14]),
                        dict(valid, validation=[20]), dict(valid, validation=(20, 21))]:
            with self.subTest(manifest=invalid), self.assertRaises(ValueError):
                validate_scene_seeds(invalid, config)

    def test_scene_seed_manifest_rejects_invalid_types_and_any_duplicate(self):
        config = scene_config()
        for value in (True, False, -1, 1., '1', None):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, 'nonnegative integers'):
                validate_scene_seeds(dict(train=[value, 11, 12, 13], validation=[20, 21]), config)
        for seeds in [dict(train=[10, 10, 12, 13], validation=[20, 21]),
                      dict(train=[10, 11, 12, 13], validation=[20, 20]),
                      dict(train=[10, 11, 12, 13], validation=[20, 10])]:
            with self.subTest(manifest=seeds), self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_scene_seeds(seeds, config)
        self.assertEqual(validate_scene_seeds(
            dict(train=[0, 11, 12, 13], validation=[20, 21]), config)['train'][0], 0)

    def test_scene_seed_lists_preserve_curriculum_ranges_and_default_sequence(self):
        config = scene_config()
        with patch('world_model.train_hierarchical.make_scene', side_effect=lambda *args: args) as make:
            cases = build_scene_cases(config)
        self.assertEqual(cases['train'], [(100, [30., 60.], 2000.), (101, [30., 60.], 2000.),
                                         (102, [100., 200.], 2000.), (103, [100., 200.], 2000.)])
        self.assertEqual(cases['validation'], [(1000100, [100., 200.], 2000.),
                                              (1000101, [100., 200.], 2000.)])
        self.assertEqual(make.call_count, 6)
        config['scene_seeds'] = dict(train=[7, 4, 19, 2], validation=[80, 50])
        with patch('world_model.train_hierarchical.make_scene', side_effect=lambda *args: args):
            custom = build_scene_cases(config)
        for split in ('train', 'validation'):
            self.assertEqual([case[0] for case in custom[split]], config['scene_seeds'][split])
            self.assertEqual([case[1:] for case in custom[split]], [case[1:] for case in cases[split]])

    def test_saved_scene_seed_config_reproduces_cases_without_manifest_file(self):
        config = scene_config()
        seeds = dict(train=[300, 303, 306, 309], validation=[312, 315])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene-seeds.json'
            path.write_text(json.dumps(seeds), encoding='utf-8')
            config['scene_seeds'] = validate_scene_seeds(json.loads(path.read_text()), config)
            original = build_scene_cases(config)
            saved_config = json.loads(json.dumps(config))
            path.unlink()
            self.assertEqual(build_scene_cases(saved_config), original)
        self.assertEqual([case['seed'] for case in original['train']], seeds['train'])
        self.assertTrue(all(30 <= case['direct_distance_m'] <= 60 for case in original['train'][:2]))
        self.assertTrue(all(100 <= case['direct_distance_m'] <= 200
                            for case in original['train'][2:] + original['validation']))

    def test_explicit_scene_seeds_keep_geometry_failures_and_cannot_override_resume(self):
        config = scene_config()
        config.update(slow_updates=0, episodes_per_update=1, eval_scenes=1,
                      scene_seeds=dict(train=[26111749], validation=[300]))
        with self.assertRaisesRegex(ValueError, '26111749 failed offline scene feasibility'):
            build_scene_cases(config)
        argv = ['train_hierarchical.py', '--output', '/tmp/unused-graph-test', '--scene-seeds', '/tmp/seeds.json']
        with patch.object(sys, 'argv', argv):
            self.assertEqual(parse_args().scene_seeds, Path('/tmp/seeds.json'))
        with patch.object(sys, 'argv', argv + ['--resume']), patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args()

    def test_raster_manifest_reproduces_metric_cases_and_detects_changed_map(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = raster_manifest(directory)
            config = scene_config()
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            config['scene_manifest'] = json.loads(path.read_text())
            cases = build_scene_cases(config)
            saved_config = json.loads(json.dumps(dict(config, scene_manifest=cases)))
            path.unlink()
            self.assertEqual(build_scene_cases(saved_config), cases)
            self.assertEqual(cases['train'][0]['direct_distance_m'], 40.)
            self.assertEqual(cases['train'][0]['reference_distance_m'], 50.)
            map_path = Path(manifest['train'][0]['map_path'])
            map_path.write_bytes(map_path.read_bytes() + b'changed raster')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                build_scene_cases(saved_config)

    def test_v9_train_strict_validation_split_and_geometry_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = raster_manifest(directory)
            geometry = Path(directory) / 'circles.npy'
            np.save(geometry, np.asarray([[200., 200., 2.]], np.float64))
            manifest['train'] = [
                dict(terrain='synthetic', kind='v9_random', case_id=f'v9-{i}', seed=i,
                     world_size=256., start=[35., 35.], goal=[35.+d, 35.],
                     direct_distance_m=d, reference_distance_m=d, circle_count=1,
                     circles_path=str(geometry), circles_sha256=file_sha256(geometry))
                for i, d in enumerate([40., 40., 120., 120.])]
            for row in manifest['validation']:
                row.update(goal=[81., 60.], direct_distance_m=21., reference_distance_m=24.)
            config = dict(scene_config(), validation_split='synthetic_to_wildscenes',
                          validation_distances_m=[20., 40.], scene_manifest=manifest)
            cases = build_scene_cases(config)
            self.assertEqual(cases, manifest)
            self.assertNotIn('circles', cases['train'][0])
            with self.assertRaises(ValueError):
                build_scene_cases(dict(config, validation_split='physical_map_group'))
            with self.assertRaisesRegex(ValueError, 'curriculum range'):
                build_scene_cases(dict(config, validation_distances_m=None))
            manifest['validation'][0]['source_unknown_space'] = 'free_by_simulation_assumption'
            with self.assertRaisesRegex(ValueError, 'unknown_space'):
                build_scene_cases(config)
            manifest['validation'][0]['source_unknown_space'] = 'blocked'
            np.save(geometry, np.asarray([[201., 200., 2.]], np.float64))
            with self.assertRaisesRegex(ValueError, 'digest'):
                build_scene_cases(config)

    def test_raster_manifest_rejects_false_geometry_ranges_and_split_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            original = raster_manifest(directory)
            changes = [dict(direct_distance_m=140.), dict(reference_distance_m=30.),
                       dict(resolution_m=2.), dict(required_clearance_m=.5),
                       dict(source_unknown_space='free'), dict(seed=True), dict(world_size=100.),
                       dict(map_sha256='0' * 64), dict(sequence='V-02'),
                       dict(start=[1., 60.], goal=[41., 60.]), dict(reference_points=[[60., 60.], [100., 60.]])]
            for values in changes:
                manifest = copy.deepcopy(original)
                manifest['train'][0].update(values)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    build_scene_cases(dict(scene_config(), scene_manifest=manifest))
            manifest = copy.deepcopy(original)
            manifest['train'][0].update(goal=[180., 60.], direct_distance_m=120., reference_distance_m=130.)
            with self.assertRaisesRegex(ValueError, 'curriculum range'):
                build_scene_cases(dict(scene_config(), scene_manifest=manifest))
            for case in manifest['validation']:
                case.update(map_group='Karawatha', sequence='K-03')
            manifest['train'][0] = copy.deepcopy(original['train'][0])
            with self.assertRaisesRegex(ValueError, 'disjoint physical map groups'):
                build_scene_cases(dict(scene_config(), scene_manifest=manifest))
            with self.assertRaisesRegex(ValueError, 'exactly 4'):
                build_scene_cases(dict(scene_config(), scene_manifest=dict(original, train=original['train'][:3])))

    def test_raster_metadata_binds_map_group_and_survives_in_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = raster_manifest(directory)
            path = Path(directory) / 'map_metadata.json'
            metadata = dict(map_sha256=manifest['train'][0]['map_sha256'], sequence='K-03', coordinate_scale=1.)
            path.write_text(json.dumps(metadata), encoding='utf-8')
            for case in manifest['train']:
                case['metadata_path'] = str(path)
            config = dict(scene_config(), scene_manifest=manifest)
            cases = build_scene_cases(config)
            self.assertEqual(cases['train'][0]['metadata_sha256'], file_sha256(path))
            metadata['sequence'] = 'V-01'
            path.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata SHA-256'):
                build_scene_cases(dict(config, scene_manifest=cases))
            with self.assertRaisesRegex(ValueError, 'map_group disagree'):
                build_scene_cases(config)

    def test_simplified_manifest_records_assumption_without_reclassifying_strict_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            original = simplified_raster_manifest(directory)
            cases = build_scene_cases(dict(scene_config(), scene_manifest=original))
            description = describe_scene_assumptions(cases)
            self.assertTrue(description['simplified_geometry'])
            self.assertEqual(description['protocols'], [dict(terrain='wildscenes_simplified_raster',
                             source_unknown_space='free_by_simulation_assumption')])
            self.assertIn('local sensor and received evidence', description['online_knowledge'])
            self.assertIn('not measured free-space evidence', description['simplified_source_assumption'])
            # A later case on the same cached raster cannot claim strict real
            # unknown support or alter provenance after the first load.
            for changes in (dict(terrain='wildscenes_raster', source_unknown_space='blocked'),
                            dict(source_scene_sha256='0' * 64), dict(source_unknown_space='blocked'),
                            dict(coordinate_scale=2.)):
                invalid = copy.deepcopy(original)
                invalid['train'][1].update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    build_scene_cases(dict(scene_config(), scene_manifest=invalid))
        with tempfile.TemporaryDirectory() as directory:
            strict = raster_manifest(directory)
            description = describe_scene_assumptions(build_scene_cases(dict(scene_config(), scene_manifest=strict)))
            self.assertFalse(description['simplified_geometry'])
            self.assertIsNone(description['simplified_source_assumption'])
            strict['train'][0]['source_unknown_space'] = 'free_by_simulation_assumption'
            with self.assertRaisesRegex(ValueError, 'assumption disagree'):
                build_scene_cases(dict(scene_config(), scene_manifest=strict))

    def test_simplified_geometry_requires_complete_known_mask_and_preserves_mission_metrics_label(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = simplified_raster_manifest(directory)
            path = Path(manifest['train'][0]['map_path'])
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays['known'][230, 230] = 0
            np.savez_compressed(path, **arrays)
            for case in manifest['train']:
                case['map_sha256'] = file_sha256(path)
            with self.assertRaises(ValueError):
                build_scene_cases(dict(scene_config(), scene_manifest=manifest))
        summary = summarize_missions([dict(goal_reached=True, terrain='wildscenes_simplified_raster',
            source_unknown_space='free_by_simulation_assumption', map_group='Karawatha',
            complete_missions=True, kilometer_task=True, kilometer_success=True)], True)
        self.assertEqual(summary['terrain_protocols'], ['wildscenes_simplified_raster'])
        self.assertEqual(summary['source_unknown_space_modes'], ['free_by_simulation_assumption'])
        self.assertEqual(summary['kilometer_success_rate'], 1.)

    def test_seen_map_validation_requires_explicit_task_split_and_distinct_endpoint_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = raster_manifest(directory)
            source = manifest['train'][0]
            for case in manifest['validation']:
                case.update(map_path=source['map_path'], map_sha256=source['map_sha256'],
                            map_group='Karawatha', sequence='K-01', start=[60., 100.], goal=[180., 100.])
            config = dict(scene_config(), scene_manifest=manifest)
            with self.assertRaisesRegex(ValueError, 'disjoint physical map groups'):
                build_scene_cases(config)
            config['validation_split'] = 'heldout_tasks'
            self.assertEqual(build_scene_cases(config), manifest)
            for start, goal in [([60., 60.], [180., 60.]), ([61., 61.], [181., 61.]),
                                ([180., 60.], [60., 60.]), ([180., 62.], [60., 62.])]:
                bad = copy.deepcopy(manifest)
                bad['validation'][0].update(start=start, goal=goal)
                with self.subTest(start=start, goal=goal), self.assertRaisesRegex(ValueError, 'endpoint pair within 2m'):
                    build_scene_cases(dict(config, scene_manifest=bad))
            bad = copy.deepcopy(manifest)
            bad['validation'][0]['seed'] = bad['train'][0]['seed']
            with self.assertRaisesRegex(ValueError, 'unique nonnegative integers'):
                build_scene_cases(dict(config, scene_manifest=bad))
            # Explicit task validation also accepts genuinely separate parks.
            self.assertEqual(build_scene_cases(dict(scene_config(), validation_split='heldout_tasks',
                                                   scene_manifest=raster_manifest(directory)))['validation'][0]['map_group'], 'Venman')

    def test_task_pair_validation_accounts_for_different_native_map_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = raster_manifest(directory)
            for split, origin in [('train', [1000., 2000.]), ('validation', [1010., 2020.])]:
                path = Path(manifest[split][0]['map_path'])
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {key: saved[key] for key in saved.files}
                np.savez_compressed(path, **arrays, native_origin_xy=origin)
                for case in manifest[split]:
                    case.update(map_sha256=file_sha256(path), map_group='Karawatha',
                                sequence='K-01' if split == 'train' else 'K-03')
                    if split == 'validation':
                        case.update(start=[50., 40.], goal=[170., 40.])
            with self.assertRaisesRegex(ValueError, 'endpoint pair within 2m'):
                build_scene_cases(dict(scene_config(), scene_manifest=manifest, validation_split='heldout_tasks'))

    def test_raster_cli_requires_declared_distances_and_resume_keeps_saved_inputs(self):
        argv = ['train_hierarchical.py', '--output', '/tmp/unused-graph-test', '--scene-manifest', '/tmp/forest.json']
        explicit = ['--warmup-distances', '30', '60', '--mission-distances', '1000', '2200']
        with patch.object(sys, 'argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args()
        with patch.object(sys, 'argv', argv + explicit + ['--graph-init', '/tmp/graph.pt', '--slow-init', '/tmp/slow.pt']):
            args = parse_args()
            self.assertEqual(args.scene_manifest, Path('/tmp/forest.json'))
            self.assertEqual(args.slow_init, Path('/tmp/slow.pt'))
            self.assertEqual(args.validation_split, 'physical_map_group')
        with patch.object(sys, 'argv', argv + explicit + ['--validation-split', 'heldout_tasks']):
            self.assertEqual(parse_args().validation_split, 'heldout_tasks')
        for overrides in (['--scene-seeds', '/tmp/seeds.json'], ['--resume']):
            with patch.object(sys, 'argv', argv + explicit + overrides), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args()
        with patch.object(sys, 'argv', ['train_hierarchical.py', '--output', '/tmp/unused-graph-test',
                                        '--resume', '--slow-init', '/tmp/slow.pt']), patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args()

    def test_snapshot_archives_the_external_map_backend_without_importing_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source' / 'world_model'
            source.mkdir(parents=True)
            (source / 'train_hierarchical.py').write_text('# archive test\n')
            backend = source.parent / 'simulator' / 'wildscenes_map.py'
            backend.parent.mkdir()
            backend.write_text('# original map backend\n')
            output = root / 'run'
            with patch.object(sys, 'argv', ['train_hierarchical.py', '--repo', str(root), '--output', str(output),
                    '--scene-manifest', str(root / 'scenes.json'), '--warmup-distances', '30', '60',
                    '--mission-distances', '100', '200']):
                args = parse_args()
                with patch('world_model.train_hierarchical.SOURCE', source), patch('os.execv', side_effect=RuntimeError('reexec')):
                    with self.assertRaisesRegex(RuntimeError, 'reexec'):
                        snapshot(args)
            archived = output / 'runtime_sources' / 'world_model' / '_wildscenes_map_backend.py'
            self.assertEqual(file_sha256(archived), file_sha256(backend))
            backend.write_text('# later mutable source\n')
            self.assertNotEqual(file_sha256(archived), file_sha256(backend))

    def test_slow_initialization_checks_current_schema_and_finite_weights(self):
        source, destination = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        saved = dict(manifest=dict(slow_schema=SLOW_SCHEMA), slow=source.state_dict())
        initialize_slow_module(destination, saved)
        self.assertEqual(state_hash(numpy_state(source)), state_hash(numpy_state(destination)))
        with self.assertRaisesRegex(ValueError, 'current slow graph schema'):
            initialize_slow_module(destination, dict(saved, manifest=dict(slow_schema='legacy')))
        with torch.no_grad():
            source.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'initialized slow actor'):
            initialize_slow_module(destination, saved)

    def test_mission_summary_separates_kilometer_tasks_and_requires_safe_success(self):
        short = dict(goal_reached=True, map_group='Karawatha', complete_missions=True,
                     kilometer_task=False, kilometer_success=False, ground_safety_violations=0)
        summary = summarize_missions([short], True)
        self.assertEqual(summary['kilometer_episodes'], 0)
        self.assertIsNone(summary['kilometer_success_rate'])
        safe = dict(short, map_group='Venman', kilometer_task=True, kilometer_success=True)
        unsafe = dict(safe, kilometer_success=False, ground_safety_violations=1, raster_collision_attempts=2)
        summary = summarize_missions([short, safe, unsafe], True)
        self.assertEqual(summary['kilometer_episodes'], 2)
        self.assertEqual(summary['kilometer_success_rate'], .5)
        self.assertEqual(summary['by_map_group']['Venman']['episodes'], 2)
        self.assertEqual(summary['full_mission_episodes'], 3)
        self.assertEqual(summary['ground_safety_violations'], 1)
        self.assertEqual(summary['raster_collision_attempts'], 2)

    def test_graph_initialization_copies_only_fast_world_and_legacy_resume_is_rejected(self):
        fast, world = torch.nn.Linear(2, 2), torch.nn.Linear(3, 2)
        source_fast, source_world = torch.nn.Linear(2, 2), torch.nn.Linear(3, 2)
        saved = dict(schema='vtemp2_received_graph_three_model_training_v1', world_schema=MODEL_SCHEMA,
                     manifest=dict(fast_schema=FAST_SCHEMA, slow_schema='legacy'),
                     fast=source_fast.state_dict(), world=source_world.state_dict(), slow='incompatible')
        initialize_graph_modules(fast, world, saved)
        self.assertEqual(state_hash(numpy_state(fast)), state_hash(numpy_state(source_fast)))
        self.assertEqual(state_hash(numpy_state(world)), state_hash(numpy_state(source_world)))
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            validate_resume_schema(saved)
        saved.update(schema=TRAIN_SCHEMA, manifest=dict(fast_schema=FAST_SCHEMA, slow_schema=SLOW_SCHEMA))
        validate_resume_schema(saved)
        saved['world_schema'] = 'incompatible'
        with self.assertRaisesRegex(ValueError, 'compatible'):
            initialize_graph_modules(fast, world, saved)

    def test_nonfinite_adam_moments_are_rejected_before_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        require_finite_optimizer(optimizer, 'unit')
        next(iter(optimizer.state.values()))['exp_avg'].fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            require_finite_optimizer(optimizer, 'unit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
