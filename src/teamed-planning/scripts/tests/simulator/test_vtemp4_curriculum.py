import json
import math
from itertools import islice
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator import train_vtemp4 as entry
from world_model.hierarchical_sources import file_sha256
from simulator.vtemp4_curriculum import (
    FIELD_TASKS, LAKE_GEOMETRY, SPLITS, STRATA, balanced_cases, geometry_hash, load_curriculum, runtime_case, source_id,
)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.km = self.root / 'km'
        self.field = self.root / 'valleys'
        self.circle_rows = {}
        self.field_rows = {}
        self.field.mkdir()
        (self.km / 'geometry').mkdir(parents=True)
        for i, split in enumerate(SPLITS):
            scene = dict(start=[2., 400.], goal=[1010., 400.], direct_distance_m=1008., world_size=1200, seed=i)
            path = self.km / 'geometry' / f'{split}.npy'
            path.write_bytes(split.encode())
            self.circle_rows[split] = [dict(scene, terrain='synthetic', kind='v9_random', case_id=split,
                                             circles_path=f'/retired/checkout/{path.name}', circles_sha256=file_sha256(path))]
            self.field_rows[split] = []
            for task in sorted(FIELD_TASKS):
                case_id = f'{split}-{task}'
                folder = self.field / case_id
                folder.mkdir()
                path = folder / 'forest_map.npz'
                path.write_bytes(case_id.encode())
                offset = {'direct': 0., 'detour_left': 240., 'detour_right': -240.}[task]
                row = dict(scene, terrain='wildscenes_raster', kind='valley', case_id=case_id, navigation_task=task,
                           terrain_family_id=f'family-{i}', map_path=str(path.relative_to(self.field)),
                           map_sha256=file_sha256(path), required_clearance_m=1.5, detour_offset_m=abs(offset))
                self.field_rows[split].append(row)
                self.write_audit(row, [500., 400. + offset])
        self.write_manifests()

    def write_audit(self, row, bridge):
        audit = dict(row, bridge_closure_disconnects_endpoints=True, certified_clearance_lower_bound_m=1.6,
                     direct_line_hits_water=row['navigation_task'] != 'direct', bridge_center_xy=bridge)
        (self.field / row['map_path']).with_name('audit.json').write_text(json.dumps(audit))

    def write_manifests(self):
        (self.km / 'scenes.json').write_text(json.dumps(self.circle_rows))
        (self.field / 'scenes.json').write_text(json.dumps(self.field_rows))

    def make_lakes(self):
        for rows in self.field_rows.values():
            for row in rows:
                preferred = row['navigation_task'].removeprefix('detour_')
                shores = {'direct': (270., 140.), 'left': (170., -316.), 'right': (316., -170.)}[preferred]
                candidates = (shores[0] + 10., 0. if preferred == 'direct' else shores[1] - 10.)
                row.update(kind='procedural_wooded_lake', detour_offset_m=0. if preferred == 'direct' else 164.)
                audit = dict(row, lake_closed=True, water_component_count=1, bridge_cell_count=0,
                             preferred_side=preferred, certified_clearance_lower_bound_m=3.1,
                             lake_entry_station_m=320., lake_gate_station_m=504., lake_exit_station_m=690.,
                             gate_left_shore_offset_m=shores[0], gate_right_shore_offset_m=shores[1],
                             direct_line_hits_water=preferred != 'direct', direct_line_clearance_lower_bound_m=3.)
                for side, offset in zip(('left', 'right'), candidates):
                    route = [row['start'], [506., 400. + offset], row['goal']]
                    audit[f'{side}_reference_route_xy'] = route
                    audit[f'{side}_route_length_m'] = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                    audit[f'{side}_clearance_lower_bound_m'] = 3.
                (self.field / row['map_path']).with_name('audit.json').write_text(json.dumps(audit))
        self.write_manifests()

    def test_lake_source_flanks_and_direct_control_are_certified(self):
        with self.assertRaisesRegex(ValueError, 'closed lake'):
            load_curriculum(self.km, self.field, field_geometry=LAKE_GEOMETRY)
        self.make_lakes()
        split, hashes = load_curriculum(self.km, self.field, field_geometry=LAKE_GEOMETRY)
        self.assertIn('field_audits_sha256', hashes)
        self.assertEqual(len(split['ppo_train']), 4)
        with self.assertRaisesRegex(ValueError, 'river-bridge'):
            load_curriculum(self.km, self.field)
        row = self.field_rows['test'][0]
        path = (self.field / row['map_path']).with_name('audit.json')
        good = json.loads(path.read_text())
        mutations = [
            dict(bridge_cell_count=1),
            dict(left_clearance_lower_bound_m=.2),
            dict(right_route_length_m=good['left_route_length_m'], right_reference_route_xy=good['left_reference_route_xy']),
        ]
        for edit in mutations:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                path.write_text(json.dumps(dict(good, **edit)))
                load_curriculum(self.km, self.field, field_geometry=LAKE_GEOMETRY)

    def test_default_and_checkpoint_select_lake_data(self):
        self.make_lakes()
        self.assertEqual(entry.training_curriculum(None), 'lakes')
        self.assertEqual(entry.training_curriculum(self.field), 'lakes')
        checkpoint = self.root / 'lake.pt'
        import torch
        torch.save(dict(format=entry.FORMAT, v9_sha256=entry.CHECKPOINT_SHA, training_maps=[], curriculum='lakes',
                        clock={}, model={}, dataset={'km_data': str(self.km), 'field_data': str(self.field)}), checkpoint)
        output = self.root / 'eval'
        argv = ['train_vtemp4.py', 'evaluate', '--output', str(output), '--checkpoint', str(checkpoint)]
        with (patch.object(sys, 'argv', argv), patch.object(entry, 'runtime'),
              patch('ground.vtemp4_policy.HeadingPolicy'), patch.object(entry, 'evaluate') as evaluate):
            entry.main()
        args, _, _, selected, _ = evaluate.call_args.args
        self.assertEqual(args.field_geometry, LAKE_GEOMETRY)
        self.assertEqual({c['kind'] for c in selected}, {'v9_random', 'procedural_wooded_lake'})

    def test_relocated_files_four_strata_and_smaller_batches(self):
        splits, hashes = load_curriculum(self.km, self.field)
        self.assertEqual(hashes['field_scenes_sha256'], file_sha256(self.field / 'scenes.json'))
        cases = splits['ppo_train']
        for case in cases:
            path = case.get('circles_path') or case['map_path']
            self.assertEqual(geometry_hash(case), file_sha256(Path(path)))
        schedule = list(islice(balanced_cases(cases, random.Random(4)), 16))
        self.assertEqual([row['navigation_task'] for row in schedule], list(STRATA) * 4)
        for i in range(len(schedule)-1):
            adjacent = [row['navigation_task'].startswith('detour_') for row in schedule[i:i + 2]]
            self.assertCountEqual(adjacent, [True, False])
        legacy, _ = load_curriculum(self.km)
        self.assertEqual([c['case_id'] for c in legacy['test']], ['test'])
        self.assertEqual(next(balanced_cases(legacy['ppo_train'], random.Random(4)))['case_id'], 'ppo_train')

    def test_tampered_geometry_and_audit(self):
        row = self.field_rows['test'][0]
        self.write_audit(row, [500., 160.])  # This is left-labelled for a mission facing east.
        with self.assertRaisesRegex(ValueError, 'turn direction'):
            load_curriculum(self.km, self.field)
        (self.field / row['map_path']).write_bytes(b'changed raster')
        with self.assertRaisesRegex(ValueError, 'digest differs'):
            load_curriculum(self.km, self.field)

    def test_related_field_sources_cannot_cross_splits(self):
        row = self.field_rows['development'][0]
        row['seed'] = self.field_rows['test'][0]['seed']
        self.write_audit(row, [500., 640.])
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, 'source family overlaps'):
            load_curriculum(self.km, self.field)

    def test_checkpoint_guards_and_truth_is_excluded_from_runtime_case(self):
        cases = load_curriculum(self.km, self.field)[0]['test']
        row = cases[-1]
        for key, identity in (('training_maps', geometry_hash(row)), ('training_sources', source_id(row)),
                              ('training_families', row['terrain_family_id'])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overlap'):
                entry.reject_training_overlap({key: [identity]}, cases)
        entry.reject_training_overlap({'training_maps': ['legacy unseen map']}, cases)
        sim = runtime_case(dict(row, reference_route_xy=['ground truth'], reference_distance_m=1500.))
        self.assertTrue({'map_path', 'map_sha256', 'start', 'goal'} <= sim.keys())
        self.assertFalse({'detour_offset_m', 'navigation_task', 'terrain_family_id', 'reference_route_xy',
                          'reference_distance_m'} & sim.keys())

    def test_default_training_entry_delivers_mixed_batch_without_any_update(self):
        out = self.root / 'prep'
        argv = ['train_vtemp4.py', 'train', '--output', str(out), '--km-data', str(self.km),
                '--field-data', str(self.field)]
        with (patch.object(sys, 'argv', argv), patch.object(entry, 'runtime'),
              patch('ground.vtemp4_policy.HeadingPolicy'), patch.object(entry, 'train') as train):
            entry.main()
        args, _, _, cases, _ = train.call_args.args
        self.assertEqual(args.episodes_per_update, 4)
        counts = json.loads((out / 'manifest.json').read_text())['stratum_counts']
        self.assertEqual(counts, dict.fromkeys(STRATA, 1))
        self.assertTrue(all(c['case_id'].startswith('ppo_train') for c in cases))

    def test_completed_results_and_censored_success(self):
        row = dict(end_reason='arrival', goal_reached=True, decisions=2, turn_count=1, left_turns=1,
                   right_turns=0, remaining_m=1.5)
        cutoff = dict(row, goal_reached=False, end_reason='collector_limit')
        self.assertEqual(entry.metrics([row], 1)['success_rate'], 1.)
        self.assertIsNone(entry.metrics([row], 2)['success_rate'])
        metrics = entry.metrics([row, cutoff], 2)
        self.assertIsNone(metrics['success_rate'])
        self.assertEqual(metrics['completed_missions'], 1)
        self.assertEqual(metrics['turn_rate'], .5)


if __name__ == '__main__':
    unittest.main()
