"""Parallel rollouts keep one PPO update per episode, STOP/resume and validation."""
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
from drone.mappo import GLOBAL_FEATURES
from simulator.train_mappo import train,validate_config


def fake_episode(model,config,seed,schedule,stochastic,progress=None,stop=None):
    rollout=[];draw=float(torch.rand(()))  # Reveals the task's sample seed.
    for t in range(config['steps']):
        if stop is not None and stop() and rollout:
            break
        x=torch.randn(3,21,12)
        mask=torch.ones(3,21,dtype=torch.bool)
        with torch.no_grad():
            dist=model.distribution(x,mask)
            a=dist.sample() if stochastic else dist.logits.argmax(-1)
            lp=dist.log_prob(a)
        rollout.append((x.numpy(),mask.numpy(),a.numpy(),lp.numpy(),float((seed+t)%3),
                       np.full(GLOBAL_FEATURES,t/config['steps'],np.float32)))
        if progress is not None:
            progress(dict(step=t+1))
    return rollout,dict(confidence_area=1.,task_gain=1.,spacing_violation=0.,
        collisions=0.,out_of_bounds=0.,task_share=.5,reward_beta=schedule['beta'],
        reward_eta=schedule['eta'],reward=float(seed%7),ground_collisions=0.,chain_link_violation=0.,
        graph_task_connected=1.,graph_first_connection_step=1.,graph_task_path_ratio=1.,
        user_stopped=float(len(rollout)<config['steps']),sampled=float(stochastic),draw=draw)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class ParallelTrainingTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[4]/'artifacts'/'training-tests'
        root.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=root)
        self.root=Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def config(self,**overrides):
        return validate_config(dict(dict(episodes=6,steps=4,validation_interval=2,
            snapshot_interval=3,validation_seeds=[20000,20001],test_seeds=[30000]),**overrides))

    def test_every_episode_is_updated_once_and_validated_on_saved_weights(self):
        config=self.config(initial_validation=True)
        # Fork shares the patched episode with the rollout processes.
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode), \
             contextlib.redirect_stdout(io.StringIO()):
            train(config,self.root/'run',workers=3,mp_context='fork')
        run=self.root/'run'
        progress=json.loads((run/'progress.json').read_text())
        self.assertEqual(progress['status'],'completed')
        training=rows(run/'training.jsonl')
        self.assertEqual([r['episode'] for r in training],list(range(1,7)))
        self.assertEqual(sorted(r['dispatch'] for r in training),list(range(1,7)))
        self.assertTrue(all(0<=r['policy_lag']<3 for r in training))
        self.assertEqual(progress['joint_steps'],6*config['steps'])
        validation=rows(run/'validation.jsonl')
        self.assertEqual(sorted(r['episode'] for r in validation),[2,4,6])
        # Every validation samples like training: seeds x samples, distinct sample seeds.
        self.assertTrue(all(len(r['scenes'])==2*config['validation_samples'] for r in validation))
        self.assertTrue(all(s['sampled']==1. for r in validation for s in r['scenes']))
        for r in validation:
            self.assertEqual([(s['seed'],s['sample']) for s in r['scenes']],
                             [(20000,0),(20000,1),(20001,0),(20001,1)])
            self.assertEqual(len({s['draw'] for s in r['scenes']}),4)
        # The same sample seeds are reused for every checkpoint.
        self.assertEqual(len({tuple(s['draw'] for s in r['scenes']) for r in validation}),1)
        self.assertTrue((run/'initial_validation.json').exists())
        self.assertEqual(list((run/'pending_validation').iterdir()),[])
        best=max(validation,key=lambda r:r['score'])
        saved=torch.load(run/'best.pt',weights_only=True)
        self.assertEqual(saved['training_episodes'],best['episode'])
        self.assertEqual(saved['best_episode'],best['episode'])
        latest=torch.load(run/'latest.pt',weights_only=True)
        self.assertEqual(latest['training_episodes'],6)
        self.assertEqual(latest['best_episode'],best['episode'])
        self.assertTrue((run/'episode_00003.pt').exists())

    def test_stop_keeps_partial_updates_and_resume_finishes(self):
        config=self.config()
        run=self.root/'run'
        from simulator import train_mappo
        original_append=train_mappo.append_json
        def stop_after_two(path,value):
            original_append(path,value)
            if path.name=='training.jsonl' and value['episode']==2:
                (run/'STOP').touch()
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode), \
             patch('simulator.train_mappo.append_json',side_effect=stop_after_two), \
             contextlib.redirect_stdout(io.StringIO()):
            train(config,run,workers=2,mp_context='fork')
        progress=json.loads((run/'progress.json').read_text())
        self.assertEqual(progress['status'],'stopped')
        stopped=progress['completed_episodes']
        self.assertGreaterEqual(stopped,2)
        self.assertLess(stopped,6)
        self.assertEqual(len(rows(run/'training.jsonl')),stopped)
        (run/'STOP').unlink()
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode), \
             contextlib.redirect_stdout(io.StringIO()):
            train(config,run,resume=True,workers=2,mp_context='fork')
        progress=json.loads((run/'progress.json').read_text())
        self.assertEqual(progress['status'],'completed')
        self.assertEqual([r['episode'] for r in rows(run/'training.jsonl')],list(range(1,7)))
        # The validation pending at STOP is repeated after resume.
        self.assertEqual(sorted(r['episode'] for r in rows(run/'validation.jsonl')),[2,4,6])
        self.assertEqual(list((run/'pending_validation').iterdir()),[])

    def test_spawned_workers_run_real_episodes(self):
        config=self.config(episodes=2,steps=2,validation_interval=2,validation_seeds=[20000])
        with contextlib.redirect_stdout(io.StringIO()):
            train(config,self.root/'real',workers=2)
        run=self.root/'real'
        self.assertEqual(json.loads((run/'progress.json').read_text())['status'],'completed')
        training=rows(run/'training.jsonl')
        self.assertEqual(len(training),2)
        self.assertTrue(all(r['rollout_steps']==2 for r in training))
        self.assertEqual(len(rows(run/'validation.jsonl')),1)
        self.assertEqual(list((run/'running').iterdir()),[])


if __name__=='__main__':
    unittest.main()
