"""Fixed reward invariants and exact optimizer/RNG checkpoint continuation."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from simulator.train_mappo import DEFAULTS,fixed_schedule,train,validate_config
from drone.mappo import CandidateNavigator


class TrainingTests(unittest.TestCase):
    def test_fixed_weights_do_not_depend_on_training_progress(self):
        config=validate_config({})
        for i in (0,300,1000,1999):
            schedule=fixed_schedule(config,i)
            self.assertEqual(schedule['beta'],1.)
            self.assertEqual(schedule['eta'],.6)
            self.assertEqual(schedule['entropy'],.01)
            self.assertEqual(schedule['focus'],0.)

    def test_historical_reward_modes_are_rejected(self):
        for overrides in ({'reward_schedule':'curriculum'}, {'reward_mode':'legacy'}):
            with self.assertRaises(ValueError):
                validate_config(overrides)

    def test_seed_splits_do_not_overlap(self):
        with self.assertRaises(ValueError):
            validate_config(dict(validation_seeds=[10000]))

    def test_exploration_candidates_are_shared_and_bounded(self):
        obs=dict(time=20.,ground_robots={'g':dict(position=[15,60,0],target=[100,60])})
        variants=[CandidateNavigator('drone_0',algorithm=a,exploration_candidates=5,
                    world_bounds=(0,0,120,120),comm_range=95.)
                  for a in ('mappo','greedy','cadmm_controlled')]
        encoded=[nav.encode((24.,60.,15.),obs) for nav in variants]
        for other in encoded[1:]:
            np.testing.assert_array_equal(encoded[0][0],other[0])
            np.testing.assert_array_equal(encoded[0][1],other[1])
            self.assertEqual(encoded[0][2],other[2])
        all_explore=CandidateNavigator('drone_0',exploration_candidates=20)
        self.assertEqual(len(all_explore.encode((24.,60.,15.),obs)[2]),21)

    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        config=validate_config(dict(episodes=3,steps=3,validation_interval=3,
            snapshot_interval=3,validation_seeds=[20000],test_seeds=[30000]))
        def fake_episode(model,config,seed,schedule,stochastic):
            rollout=[]
            for t in range(config['steps']):
                x=torch.randn(3,21,12)
                mask=torch.ones(3,21,dtype=torch.bool)
                with torch.no_grad():
                    dist=model.distribution(x,mask)
                    a=dist.sample()
                    lp=dist.log_prob(a)
                rollout.append((x.numpy(),mask.numpy(),a.numpy(),lp.numpy(),float((seed+t)%3)))
            return rollout,dict(confidence_area=1.,task_gain=1.,spacing_violation=0.,
                collisions=0.,out_of_bounds=0.,task_share=.5,reward_beta=schedule['beta'],
                reward_eta=schedule['eta'],reward=1.,ground_collisions=0.,chain_link_violation=0.,
                graph_task_connected=1.,graph_first_connection_step=1.,graph_task_path_ratio=1.)
        artifact_root=Path(__file__).resolve().parents[4]/'artifacts'/'training-tests'
        artifact_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            root=Path(directory)
            self.assertTrue(root.resolve().is_relative_to(artifact_root.resolve()))
            with patch('simulator.train_mappo.run_episode',side_effect=fake_episode),contextlib.redirect_stdout(io.StringIO()):
                train(config,root/'full')
                from simulator import train_mappo
                original_append=train_mappo.append_json
                def stop_after_one(path,value):
                    original_append(path,value)
                    if path.name=='training.jsonl' and value['episode']==1:
                        (root/'resumed'/'STOP').touch()
                with patch('simulator.train_mappo.append_json',side_effect=stop_after_one):
                    train(config,root/'resumed')
                self.assertEqual(json.loads((root/'resumed'/'progress.json').read_text())['status'],'stopped')
                (root/'resumed'/'STOP').unlink()
                train(config,root/'resumed',resume=True)
            full=torch.load(root/'full'/'latest.pt',weights_only=True)
            resumed=torch.load(root/'resumed'/'latest.pt',weights_only=True)
            for key in full['model']:
                self.assertTrue(torch.equal(full['model'][key],resumed['model'][key]),key)
            self.assertEqual(full['scene_rng'],resumed['scene_rng'])
            self.assertTrue(torch.equal(full['torch_rng'],resumed['torch_rng']))


if __name__=='__main__':
    unittest.main()
