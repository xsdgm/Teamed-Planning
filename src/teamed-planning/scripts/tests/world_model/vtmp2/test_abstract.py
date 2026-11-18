"""Leakage, task semantics, latent transitions, and offline experiment safeguards."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from simulator.train_world_model import build_command
from world_model import abstract_data as data
from world_model.abstract_model import AbstractWorldModel, advance_state, loss, load_checkpoint, FORMAT, SCHEMA_VERSION
from world_model.abstract_training import fit_ridge, ridge_predict
from world_model.models import make_model
from world_model.abstract_assessment import calibrate_fallback, guarded_choices, guard_metrics


def fixture():
    p=np.zeros((5,16,16),np.float32);p[1]=1
    return dict(probabilities=p,valid=np.ones((16,16),bool),support=np.ones((16,16),np.float32))


def tasks(count=4):
    scene=fixture();items=[]
    for i in range(count):
        task=data.make_task(scene,np.random.default_rng(i));task['scene']=0;items.append(task)
    return scene,{key:np.stack([item[key] for item in items]) for key in items[0]}


class AbstractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_hidden_labels_and_support_never_enter_inputs_or_initial_state(self):
        scene=fixture();known=np.zeros((16,16),bool);known[:,:6]=True
        args=(known,np.array([3,8]),np.array([12,8]))
        first=data.encode(scene['probabilities'],known,scene['support'],*args[1:])
        state=data.evidence_state(scene['probabilities'],*args)
        scene['probabilities'][:,~known]=np.nan;scene['support'][~known]=np.nan
        np.testing.assert_array_equal(first,data.encode(scene['probabilities'],known,scene['support'],*args[1:]))
        np.testing.assert_array_equal(state,data.evidence_state(scene['probabilities'],*args))

    def test_isolated_goal_island_is_not_a_connected_route(self):
        scene=fixture();known=np.zeros((16,16),bool);known[8,2:5]=True
        start=np.array([2,8]);goal=np.array([12,8]);before=data.evidence_state(scene['probabilities'],known,start,goal)
        known[8,11:14]=True
        after=data.evidence_state(scene['probabilities'],known,start,goal)
        self.assertEqual(after[0],0);self.assertEqual(before[1],after[1]);self.assertEqual(after[2],0)
        self.assertEqual(after[6],1)
        known[8,2:13]=True
        complete=data.evidence_state(scene['probabilities'],known,start,goal)
        np.testing.assert_array_equal(complete[:3],[1,1,1])

    def test_goal_changes_corridor_and_unknown_does_not_become_evidence(self):
        start=np.array([8,8]);north=np.array([8,1]);south=np.array([8,15])
        self.assertFalse(np.array_equal(data.corridor(start,north),data.corridor(start,south)))
        scene=fixture();scene['valid'][:,:3]=False
        task=data.make_task(scene,np.random.default_rng(7))
        self.assertFalse(task['after_masks'][:,:,~scene['valid']].any())
        for branch in range(8):
            previous=task['known'].copy()
            for step in range(2):
                previous|=data.query_mask(task['actions_grid'][branch,step])&scene['valid']
                expected=data.evidence_state(scene['probabilities'],previous,task['start'],task['goal'])
                np.testing.assert_array_equal(expected,task['target'][branch,step])

    def test_repeat_window_has_zero_evidence_gain(self):
        scene=fixture();known=np.zeros((16,16),bool);known[8,2]=True
        action=np.array([4,8,2]);known|=data.query_mask(action)
        first=data.evidence_state(scene['probabilities'],known,[2,8],[10,8]);known|=data.query_mask(action)
        np.testing.assert_array_equal(first,data.evidence_state(scene['probabilities'],known,[2,8],[10,8]))

    def test_rotations_keep_task_channels_actions_and_targets_consistent(self):
        scene,arrays=tasks();dataset=data.AbstractTasks([scene],arrays,augment=True,seed=71)
        for epoch in range(8):
            dataset.epoch=epoch;item=dataset[0]
            start,goal=np.rint((item['task'].reshape(2,2)+1)*7.5).astype(int)
            self.assertEqual(item['maps'][7,start[1],start[0]],1)
            self.assertEqual(item['maps'][8,goal[1],goal[0]],1)
            np.testing.assert_array_equal(item['maps'][9],data.corridor(start,goal))
            for branch in range(8):
                window=np.rint((item['actions'][branch,0,:2]+1)*7.5).astype(int)
                action=(*window,int(item['actions'][branch,0,2]*2))
                expected=item['maps'][5].astype(bool)|data.query_mask(action)
                np.testing.assert_array_equal(expected,item['future_maps'][branch,0,5])
                actual=data.evidence_state(item['future_maps'][branch,0,:5],expected,start,goal)
                np.testing.assert_allclose(actual,item['target'][branch,0],atol=1e-6)

    def test_model_masks_hidden_nan_and_validates_observed_nan(self):
        scene,arrays=tasks(2);batch=next(iter(DataLoader(data.AbstractTasks([scene],arrays),batch_size=2)))
        model=AbstractWorldModel(hidden=4,latent=16,members=1)
        args=[batch[key] for key in ('maps','state','task','actions')]
        first=model.predict(*args)['state_mean']
        maps=args[0].clone();unknown=~(maps[:,5:6]>.5)
        maps[:,:5].masked_fill_(unknown,float('nan'));maps[:,6:7].masked_fill_(unknown,float('nan'))
        torch.testing.assert_close(first,model.predict(maps,*args[1:])['state_mean'])
        maps[:,7,0,0]=float('nan')
        with self.assertRaises(ValueError):model.predict(maps,*args[1:])

    def test_state_bounds_and_monotone_updates(self):
        before=torch.zeros(4,8,7)
        for _ in range(5):
            after=advance_state(before,torch.randn_like(before)*5)
            self.assertTrue(((after>=before)&(after<=1)).all())
            self.assertTrue((after[...,2]<=after[...,0]).all())
            self.assertTrue((after[...,4]<=after[...,3]).all())
            self.assertTrue((after[...,1]>=after[...,0]).all());before=after

    def test_training_has_finite_gradients_action_effect_and_checkpoint_roundtrip(self):
        scene,arrays=tasks(3);batch=next(iter(DataLoader(data.AbstractTasks([scene],arrays),batch_size=3)))
        model=AbstractWorldModel(hidden=4,latent=16,members=2)
        objective,_=loss(model,batch,torch.full((7,),.2));objective.backward()
        self.assertTrue(torch.isfinite(objective));self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.predictors[0].transition[0].weight.grad.abs().sum()),0)
        torch.optim.Adam(model.parameters(),lr=.001).step();model.update_teacher()
        inputs=[batch[key] for key in ('maps','state','task','actions')]
        expected=model.predict(*inputs)
        self.assertEqual(expected['state_mean'].shape,(3,8,2,7));self.assertNotIn('maps',expected)
        self.assertFalse(expected['is_vehicle_traversability'])
        self.assertGreater(float(expected['state_mean'][:,1:].sub(expected['state_mean'][:,:1]).abs().max()),0)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            doc=dict(format=FORMAT,schema_version=SCHEMA_VERSION,model_architecture=model.model_architecture,
                states=list(data.STATE_NAMES),architecture=model.configuration(),model=model.state_dict())
            torch.save(doc,path);loaded,_=load_checkpoint(path)
            torch.testing.assert_close(expected['state_mean'],loaded.predict(*inputs)['state_mean'])
            doc['schema_version']=6;torch.save(doc,path)
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_ridge_and_input_features_are_independent_of_evaluation_targets(self):
        _,arrays=tasks(8);scales=np.full(7,.2);coef,_=fit_ridge(arrays,arrays,scales)
        before=ridge_predict(arrays,coef);features=data.baseline_features(arrays,0)
        arrays['target'][:]=np.nan
        np.testing.assert_array_equal(before,ridge_predict(arrays,coef))
        np.testing.assert_array_equal(features,data.baseline_features(arrays,0))

    def test_configuration_routes_to_offline_pipeline(self):
        root=Path(__file__).resolve().parents[4]
        config=root/'config/algorithm/world_model_vtmp2_abstract_goose.json'
        command=build_command(config,algorithm='vtmp2')
        self.assertTrue(command[3].endswith('abstract_training.py'));self.assertIn('--v9-checkpoint',command)
        self.assertIsInstance(make_model(dict(algorithm='vtmp2',model_architecture='abstract_observation_vtmp2')),AbstractWorldModel)

    def test_calibration_uses_worst_supported_place_and_unknown_falls_back(self):
        _,arrays=tasks(8)
        scores=np.tile(np.arange(8),(8,1)).astype(float)
        baseline=-scores;groups=np.array(['A']*4+['B']*4)
        calibration=calibrate_fallback(scores,baseline,arrays,groups,minimum_disagreements=2)
        self.assertTrue(calibration['enabled'])
        self.assertEqual(calibration['overestimation_bound'],max(x['bound'] for x in calibration['groups']))
        disabled=calibrate_fallback(scores,baseline,arrays,groups,minimum_disagreements=5)
        choice,override=guarded_choices(scores,baseline,disabled)
        self.assertFalse(override.any());np.testing.assert_array_equal(choice,baseline.argmax(1))
        scores[0]=np.nan;scores[1]=np.inf
        choice,override=guarded_choices(scores,baseline,calibration)
        self.assertFalse(override[:2].any());np.testing.assert_array_equal(choice[:2],baseline[:2].argmax(1))

    def test_advantage_guard_does_not_count_all_abstention_as_improvement(self):
        _,arrays=tasks(3);scores=np.tile(np.arange(8),(3,1)).astype(float);baseline=-scores
        calibration=dict(format='goose-abstract-fallback-v1',enabled=True,overestimation_bound=8.,
            action_domain='offline_observation_windows',deployment_ready=False)
        result=guard_metrics(scores,baseline,arrays,calibration)
        self.assertTrue(result['all_abstain_is_not_improvement']);self.assertEqual(result['overrides'],0)
        self.assertEqual(result['mean_utility_gain_over_baseline'],0)
        calibration['overestimation_bound']=6.
        choice,override=guarded_choices(scores,baseline,calibration)
        self.assertTrue(override.all());np.testing.assert_array_equal(choice,7)
        calibration['deployment_ready']=True
        self.assertFalse(guarded_choices(scores,baseline,calibration)[1].any())


if __name__=='__main__':unittest.main()
