"""Real-data contract, range integrity, selective prediction and fallback tests."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from world_model import frontier_training as training
from world_model.frontier_data import CLASS_NAMES, FrontierPatches, encode_observation
from world_model.frontier_guard import AdvicePolicy, GuardedSurveyAdvice
from world_model.frontier_model import FrontierWorldModel
from world_model.frontier_selective import fit, predict, selective_metrics
from world_model.goose_data import rasterize
from world_model.goose_download import RemoteZip, read_member, pairs
from world_model.regional_data import FORMAT, load_manifest
from world_model.semantic_data import sha256


def regional_fixture(root):
    records = []
    for i, split in enumerate(('train', 'val', 'calibration', 'test')):
        p = np.zeros((5, 16, 16), np.float32); p[0, :8] = 1; p[1, 8:] = 1
        source_hash = hashlib.sha256(split.encode()).hexdigest()
        path = root/(split+'.npz')
        np.savez_compressed(path, probabilities=p, valid=np.ones((16, 16), bool),
            support=np.ones((16, 16), np.float32), elevation=np.zeros((16, 16), np.float32),
            height_valid=np.zeros((16, 16), bool), origin_xy=np.zeros(2), resolution_m=16.,
            sequence=split, source_sha256=source_hash)
        records.append(dict(split=split, group=split, map=path.name, sha256=sha256(path),
            sequence=split, source_sha256=source_hash, metadata=dict(format=FORMAT)))
    document = dict(format=FORMAT, classes=list(CLASS_NAMES), cell_size_m=16., minimum_support=.125,
        simulation_used=False, dataset='GOOSE test fixture', grouping={'forest':['forest']}, records=records)
    path = root/'manifest.json'; path.write_text(json.dumps(document))
    return path


class GooseDataTests(unittest.TestCase):
    def test_equal_area_columns_and_unmeasured_unknown(self):
        points = np.array([[.5,.5,1,0]]*3+[[2.5,.5,1,0]]*90, np.float32)
        native = np.array([0]*3+[1]*90)
        scene, audit = rasterize(points, native, np.arange(5, dtype=np.int16),
            cell_size=4., fine_size=2., extent=8., minimum_support=.5)
        np.testing.assert_allclose(scene['probabilities'][:,1,1], [.5,.5,0,0,0])
        self.assertEqual(scene['valid'].sum(), 1)
        self.assertEqual(scene['support'][1,1], .5)
        self.assertFalse(scene['height_valid'].any())
        self.assertEqual(scene['probabilities'][:,~scene['valid']].sum(), 0)

    def test_other_and_water_indices_do_not_overflow_16_bit_lookup(self):
        points = np.array([[1,1,1,0]]*3, np.float32)
        for label in (3,4):
            scene, _ = rasterize(points, np.full(3,label), np.arange(5, dtype=np.int16), minimum_support=1/64)
            self.assertAlmostEqual(float(scene['probabilities'][label].sum()), 1.)

    def test_ignored_dynamic_labels_and_nonfinite_points_never_become_free(self):
        scene, _ = rasterize(np.array([[1,1,0,0]]*3+[[np.nan,0,0,0]]*3),
            np.array([0]*3+[1]*3), np.array([-1,1], np.int16))
        self.assertFalse(scene['valid'].any())
        with self.assertRaises(ValueError):
            rasterize(np.zeros((1,4)), np.array([99]), np.arange(5))

    def test_group_leakage_and_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=regional_fixture(root)
            scenes, document=load_manifest(path,cell_size=16.,minimum_support=.125)
            self.assertEqual(set(scenes), {'train','val','calibration','test'})
            document['records'][1]['group']='train'; path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,'leaked'):
                load_manifest(path,cell_size=16.,minimum_support=.125)
            path=regional_fixture(root); (root/'val.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_manifest(path,cell_size=16.,minimum_support=.125)

    def test_single_rare_region_is_not_promoted_to_balanced_sampling_stratum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=regional_fixture(root)
            scenes,_=load_manifest(path,cell_size=16.,minimum_support=.125)
            a,b=scenes['train'][0],scenes['val'][0]
            a['probabilities'][:,0,0]=0;a['probabilities'][3,0,0]=1
            dataset=FrontierPatches([a,b],prior=np.ones(5)/5,minimum_stratum_scenes=2)
            self.assertNotIn(3,dataset.strata)
            self.assertIn(0,dataset.strata)

    def test_generic_training_real_optimizer_steps_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=regional_fixture(root); actor=root/'v9';actor.write_bytes(b'fixed')
            with patch.object(training,'sha256',side_effect=lambda p:training.V9_SHA256 if Path(p)==actor else sha256(p)):
                result=training.train(path,root/'run',v9_checkpoint=actor,epochs=1,hidden=4,members=1,
                    samples_per_epoch=4,validation_masks=1,batch_size=2,device='cpu')
            self.assertEqual(result['optimizer_steps'],2)
            config=json.loads((root/'run/config.json').read_text())
            self.assertEqual(config['dataset'],'GOOSE test fixture')
            self.assertEqual(config['source_groups'],{'forest':['forest']})
            self.assertFalse(config['graph_supervision_available'])
            self.assertEqual(actor.read_bytes(),b'fixed')


class RangeZipTests(unittest.TestCase):
    def test_stored_deflated_and_crc_integrity(self):
        for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',compression=compression) as z:
                z.writestr('sample.bin',b'real bytes'*30)
            payload=stream.getvalue()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                info=z.infolist()[0]
            entry=dict(name=info.filename,compressed=info.compress_size,offset=info.header_offset,
                size=info.file_size,crc=info.CRC,compression=info.compress_type)
            with patch.object(RemoteZip,'fetch',side_effect=lambda o,n:(payload[o:o+n],len(payload))):
                self.assertEqual(read_member(dict(url='fixture',size=len(payload)),entry),b'real bytes'*30)
                with self.assertRaisesRegex(ValueError,'integrity'):
                    read_member(dict(url='fixture',size=len(payload)),{**entry,'crc':1})

    def test_actual_archive_layouts_pair_sensor_suffixes(self):
        for prefix,suffix in (('lidar/train/s','vls128'),('gooseEx_3d_train/lidar/train/s','pcl')):
            paths=[prefix+'/frame_'+suffix+'.bin',prefix.replace('lidar/','labels/')+'/frame_goose.label']
            self.assertEqual(len(pairs(dict(entries=[dict(name=p) for p in paths]))),1)


class SelectiveTests(unittest.TestCase):
    def test_high_confidence_unvalidated_class_is_rejected(self):
        p=np.tile([0,0,.99,0,.01],(10,1));y=p.copy()
        result=selective_metrics(p,y,np.ones(10),.8,allowed_classes=[0,1,4])
        self.assertEqual(result['accepted_cells'],0)
        self.assertEqual(result['coverage'],0)

    def test_always_wrong_model_does_not_pass_by_rejecting_everything(self):
        p=np.tile([.99,.01,0,0,0],(30,1));y=np.tile([0,1,0,0,0],(30,1))
        result=fit(p,y,np.ones(30),minimum_cells=1)
        self.assertFalse(result['operating_point_accepted'])
        self.assertGreater(result['threshold'],1)

    def test_unknown_is_separate_from_observed_other(self):
        torch.set_num_threads(1)
        p=np.zeros((5,16,16),np.float32);p[4]=1
        observed=np.zeros((16,16),bool);observed[:,:8]=True
        maps,_=encode_observation(p,observed,class_prior=np.ones(5)/5)
        output=predict(FrontierWorldModel(hidden=4,members=1),torch.tensor(maps)[None],
            dict(format='vtmp2-selective-frontier-v1',temperature=1.,threshold=1.000001))
        self.assertTrue(output['unknown_mask'][0,:,8:].all())
        self.assertFalse(output['unknown_mask'][0,:,:8].any())
        self.assertTrue((output['semantics'][0,4,:,:8]==1).all())
        self.assertFalse(output['prediction_mask'].any())


class FallbackTests(unittest.TestCase):
    context=dict(mission_id='task',graph_revision=1,vehicle_position=[0,0],final_goal=[0,-100])
    policy=AdvicePolicy(enabled=True,terrain_gate_passed=True,selective_gate_passed=True,
        graph_benefit_validated=True,timeout_s=1.)

    def guard(self,propose,policy=None,clock=time.monotonic):
        return GuardedSurveyAdvice(lambda c:['baseline',c['graph_revision']],propose,
            lambda target,c,r:target[1]<0,policy=policy or self.policy,clock=clock)

    def finish(self,guard):
        for _ in range(100):
            with guard._lock:
                if 'finished' in guard._job:return
            time.sleep(.001)
        self.fail('Worker did not finish')

    def test_default_never_calls_model_and_preserves_baseline(self):
        guard=self.guard(lambda c:self.fail('Model should stay disabled'),AdvicePolicy())
        result=guard.choose(self.context)
        self.assertEqual(result['survey_target'],['baseline',1])
        self.assertEqual(result['vehicle_policy'],'vtmp_on_measured_graph')

    def test_model_exception_returns_live_baseline(self):
        def fail(c):raise RuntimeError('load failure')
        guard=self.guard(fail);guard.choose(self.context);self.finish(guard)
        result=guard.choose(self.context)
        self.assertEqual(result['source'],'baseline')
        self.assertIn('model_error',result['fallback_reason'])

    def test_unknown_no_gain_invalid_flight_each_falls_back(self):
        for result,reason in ((dict(survey_target=[0,-10],unknown_fraction=1.,expected_graph_gain=1.),'prediction_unknown'),
            (dict(survey_target=[0,-10],unknown_fraction=0.,expected_graph_gain=0.),'no_expected_graph_gain_over_baseline'),
            (dict(survey_target=[0,10],unknown_fraction=0.,expected_graph_gain=1.),'survey_not_validated')):
            guard=self.guard(lambda c:result);guard.choose(self.context);self.finish(guard)
            self.assertEqual(guard.choose(self.context)['fallback_reason'],reason)

    def test_new_vehicle_pose_goal_or_graph_discards_stale_result(self):
        for changed in ({'vehicle_position':[1,0]},{'final_goal':[0,100]},{'graph_revision':2}):
            guard=self.guard(lambda c:dict(survey_target=[0,-10],unknown_fraction=0.,expected_graph_gain=1.))
            guard.choose(self.context);self.finish(guard)
            result=guard.choose({**self.context,**changed})
            self.assertEqual(result['fallback_reason'],'stale_mission_pose_or_graph')

    def test_timeout_does_not_wait_and_latches_baseline(self):
        event=threading.Event();now=[0.]
        guard=self.guard(lambda c:event.wait(2.),clock=lambda:now[0])
        guard.choose(self.context);now[0]=2.
        self.assertEqual(guard.choose(self.context)['fallback_reason'],'inference_timeout')
        self.assertEqual(guard.choose(self.context)['source'],'baseline')
        event.set()

    def test_finished_result_can_expire_before_consumption(self):
        now=[0.]
        guard=self.guard(lambda c:dict(survey_target=[0,-10],unknown_fraction=0.,expected_graph_gain=1.),clock=lambda:now[0])
        guard.choose(self.context);self.finish(guard);now[0]=2.
        self.assertEqual(guard.choose(self.context)['fallback_reason'],'model_result_expired')

    def test_accepted_advice_never_contains_vehicle_command(self):
        guard=self.guard(lambda c:dict(survey_target=[0,-10],unknown_fraction=0.,expected_graph_gain=1.))
        guard.choose(self.context);self.finish(guard);result=guard.choose(self.context)
        self.assertEqual(result['source'],'world_model_advice')
        self.assertNotIn('vehicle_target',result)


if __name__=='__main__':unittest.main()
