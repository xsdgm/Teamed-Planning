"""Larger collection batches preserve every remaining fixed navigation task."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.training_schedule import (create_continuation_plan, validate_continuation_plan,
                           validate_update_plan, update_episode_counts, json_sha256)
from world_model.scene_manifest import validate_scene_manifest


def parent(completed=5):
    config = dict(warmup_updates=8, fast_updates=8, slow_updates=64, joint_updates=32,
        warmup_distances_m=[100., 200.], mission_distances_m=[1000., 1000.],
        episodes_per_update=4, eval_scenes=9, eval_every=8, workers=4,
        complete_missions=True, freeze_fast_world=False, wm_epochs=2, ppo_epochs=4,
        slow_epochs=4, slow_bc_epochs=32, replay_episodes=16,
        scene_manifest=dict(train=[dict(seed=1000+i,case_id=f'train-{i}',marker=['unchanged',i]) for i in range(448)],
                            validation=[dict(seed=2000+i,case_id=f'validation-{i}') for i in range(9)]))
    return dict(config=config, completed_updates=completed, manifest=dict(source_sha256={'old.py':'a'*64}))


class RebatchTests(unittest.TestCase):
    def test_formal_fifth_update_leaves_exactly_428_tasks_in_27_batches(self):
        saved = parent()
        original = deepcopy(saved)
        plan = create_continuation_plan(saved)
        config = validate_continuation_plan(plan, saved)
        self.assertEqual(saved, original)
        self.assertEqual(config['scene_manifest']['train'], saved['config']['scene_manifest']['train'][20:])
        self.assertEqual(config['scene_manifest']['validation'], saved['config']['scene_manifest']['validation'])
        self.assertEqual(update_episode_counts(config), [12] + [16] * 26)
        self.assertEqual([config[p+'_updates'] for p in ('warmup','fast','slow','joint')], [1,2,16,8])
        self.assertEqual(config['policy_version_offset'], 5)
        self.assertEqual(config['workers'], 16)
        self.assertEqual(sum(update_episode_counts(config)), 428)
        self.assertEqual(plan['audit']['remaining_training_tasks'], 428)
        self.assertEqual(plan['audit']['completed_training_tasks_total'], 20)
        self.assertEqual([row['original_episode_index'] for row in plan['audit']['tasks']], list(range(20,448)))
        self.assertEqual([row['case_sha256'] for row in plan['audit']['tasks']],
                         [json_sha256(case) for case in saved['config']['scene_manifest']['train'][20:]])

    def test_evaluation_boundaries_and_curriculum_union_are_preserved(self):
        plan = create_continuation_plan(parent())
        updates = plan['config']['update_plan']
        self.assertEqual([i+1 for i,row in enumerate(updates) if row['evaluate_after']], list(range(1,28,2)))
        self.assertEqual(plan['audit']['continuation_evaluation_events'], 14)
        self.assertEqual(plan['audit']['original_remaining_evaluation_events'], 14)
        self.assertEqual(updates[0]['source_update_indices'], [5,6,7])
        first_slow = updates[3]
        self.assertEqual(first_slow['source_update_indices'], [16,17,18,19])
        self.assertEqual(first_slow['distance_range_m'], [100+900/64, 200+800*4/64])
        self.assertEqual(updates[-1]['source_update_indices'], [108,109,110,111])
        self.assertEqual(updates[-1]['distance_range_m'], [1000.,1000.])

    def test_plan_edits_cannot_change_task_order_values_hyperparameters_or_boundaries(self):
        saved = parent(); plan = create_continuation_plan(saved)
        mutations = [lambda p:p['config'].update(wm_epochs=99),
                     lambda p:p['config']['scene_manifest']['train'][0].update(seed=999),
                     lambda p:p['config']['scene_manifest']['validation'].reverse(),
                     lambda p:p['config']['update_plan'][0].update(evaluate_after=False),
                     lambda p:p['config']['update_plan'][3].update(distance_range_m=[1.,2000.]),
                     lambda p:p['audit']['tasks'][0].update(original_episode_index=19)]
        for change in mutations:
            with self.subTest(change=change):
                changed = deepcopy(plan); change(changed)
                with self.assertRaises(ValueError):validate_continuation_plan(changed,saved)
        changed = deepcopy(plan); changed['parent']['config_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'parent checkpoint'):validate_continuation_plan(changed,saved)

    def test_variable_plan_rejects_bad_phase_sources_counts_and_union(self):
        config = create_continuation_plan(parent())['config']
        mutations = [lambda c:c['update_plan'][0].update(episodes=17),
                     lambda c:c['update_plan'][0].update(episodes=True),
                     lambda c:c['update_plan'][0].update(source_update_indices=[5,5,7]),
                     lambda c:c['update_plan'][3].update(phase='warmup'),
                     lambda c:c.update(joint_updates=9),
                     lambda c:c['update_plan'][3].update(distance_range_m=[100.,1000.]),
                     lambda c:c['update_plan'][0].update(evaluate_after=False)]
        for change in mutations:
            bad=deepcopy(config);change(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_update_plan(bad)

    def test_completed_boundary_and_second_continuation_do_not_repeat_tasks(self):
        at_boundary=create_continuation_plan(parent(8))
        self.assertEqual(at_boundary['config']['update_plan'][0]['phase'],'fast')
        self.assertEqual(len(at_boundary['config']['scene_manifest']['train']),416)
        first=create_continuation_plan(parent())
        saved=dict(config=first['config'],completed_updates=1,manifest={'source_sha256':{'next.py':'b'*64}})
        second=create_continuation_plan(saved)
        self.assertEqual(second['audit']['completed_training_tasks_total'],32)
        self.assertEqual(second['config']['policy_version_offset'],6)
        self.assertEqual(len(second['config']['scene_manifest']['train']),416)
        self.assertEqual(second['config']['update_plan'][0]['source_update_indices'],[8,9,10,11])
        with self.assertRaisesRegex(ValueError,'no unfinished'):create_continuation_plan(parent(112))

    def test_checkpoint_and_geometry_proofs_are_bound_without_tensor_loading(self):
        saved=parent()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'latest.pt';checkpoint.write_bytes(b'checkpoint binding fixture')
            proof=root/'geometry.json'
            rows=[dict(case_id=case['case_id'],all_segments_clear=True,required_clearance_m=1.5)
                  for cases in saved['config']['scene_manifest'].values() for case in cases]
            proof.write_text(json.dumps(dict(per_task_geometry=rows)))
            plan=create_continuation_plan(saved,parent_checkpoint=checkpoint,geometry_audit=proof)
            self.assertEqual(plan['parent']['checkpoint_path'],str(checkpoint))
            self.assertEqual(len(plan['parent']['checkpoint_sha256']),64)
            self.assertEqual(plan['audit']['geometry_proof_reference']['task_proofs'],457)
            self.assertEqual(validate_continuation_plan(plan,saved),plan['config'])
            rows.pop();proof.write_text(json.dumps(dict(per_task_geometry=rows)))
            with self.assertRaisesRegex(ValueError,'every original task'):
                create_continuation_plan(saved,geometry_audit=proof)

    def test_scene_manifest_uses_partial_batch_counts_before_loading_geometry(self):
        config=create_continuation_plan(parent())['config']
        ranges=[row['distance_range_m'] for row in config['update_plan']]
        # Correct 428 count proceeds to validating case fields, rather than
        # incorrectly demanding 27 * 16 = 432 cases.
        with self.assertRaisesRegex(ValueError,'lacks required manifest fields'):
            validate_scene_manifest(config['scene_manifest'],config,ranges)
        missing=deepcopy(config['scene_manifest']);missing['train'].pop()
        with self.assertRaisesRegex(ValueError,'exactly 428'):
            validate_scene_manifest(missing,config,ranges)
        wrong=deepcopy(ranges);wrong[0]=[1.,1000.]
        with self.assertRaisesRegex(ValueError,'explicit update_plan'):
            validate_scene_manifest(config['scene_manifest'],config,wrong)

    def test_scene_manifest_assigns_ranges_by_variable_episode_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files=[]
            for site in ('K','V'):
                path=root/(site+'.npz');path.write_bytes(site.encode());files.append(path)
            def case(path,group,distance,seed):
                return dict(terrain='wildscenes_raster',map_path=str(path),
                    map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),map_group=group,
                    world_size=2000.,start=[10.,10.],goal=[10.+distance,10.],circles=[],
                    direct_distance_m=distance,reference_distance_m=distance,seed=seed,kind='unit',
                    resolution_m=1.,required_clearance_m=1.5,source_unknown_space='blocked')
            manifest=dict(train=[case(files[0],'Karawatha',100.,1),
                                 case(files[0],'Karawatha',1000.,2),case(files[0],'Karawatha',1000.,3)],
                          validation=[case(files[1],'Venman',1000.,4)])
            config=dict(warmup_updates=1,fast_updates=0,slow_updates=1,joint_updates=0,
                episodes_per_update=16,eval_scenes=1,mission_distances_m=[1000.,1000.],
                update_plan=[dict(phase='warmup',episodes=1,distance_range_m=[100.,100.],
                                  source_update_indices=[5],evaluate_after=True),
                             dict(phase='slow',episodes=2,distance_range_m=[1000.,1000.],
                                  source_update_indices=[6],evaluate_after=True)])
            forest=SimpleNamespace(metadata={},resolution_m=1.,origin_xy=np.zeros(2),
                upper_xy=np.array([2000.,2000.]),known=np.ones((1,1),bool),point_clear=lambda p,c:True)
            ranges=[[100.,100.],[1000.,1000.]]
            with patch('world_model.wildscenes_runtime.load_forest',return_value=forest):
                self.assertEqual(validate_scene_manifest(manifest,config,ranges),manifest)
                wrong=deepcopy(manifest);wrong['train'][1]=case(files[0],'Karawatha',100.,2)
                with self.assertRaisesRegex(ValueError,'outside its curriculum range'):
                    validate_scene_manifest(wrong,config,ranges)


if __name__=='__main__':
    unittest.main(verbosity=2)
