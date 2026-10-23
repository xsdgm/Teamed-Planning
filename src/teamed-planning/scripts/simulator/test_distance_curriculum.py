"""Sampling and stage-transition checks without training or evaluation rollouts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from simulator.large_world_scenarios import generate_training_scene,distance_key
from simulator.train_distance_curriculum import chain_radius,make_plan,stage_config,preserve_checkpoint,run

ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'src/teamed-planning/config/algorithm/mappo_v3_distance_curriculum.json'


class CurriculumTests(unittest.TestCase):
    def plan(self):
        settings=json.loads(CONFIG.read_text())
        base=json.loads((CONFIG.parent/settings.pop('base_config')).read_text())
        return make_plan(base,settings)

    def test_first_ring_is_theoretical_ground_centred_chain_and_five_rings_touch(self):
        plan=self.plan()
        self.assertAlmostEqual(chain_radius(),np.sqrt(95.**2-15.**2)+190.)
        self.assertEqual(plan['stages'][0]['lower_m'],100.)
        self.assertEqual(plan['stages'][0]['upper_m'],chain_radius())
        self.assertLess(plan['cbf_margin_geometry_radius_m'],chain_radius())
        self.assertEqual(len(plan['stages']),5)
        self.assertEqual(plan['stages'][-1]['upper_m'],750.)
        for first,second in zip(plan['stages'],plan['stages'][1:]):
            self.assertEqual(first['upper_m'],second['lower_m'])

    def test_first_stage_fresh_then_restore_weights_and_optimizer(self):
        plan=self.plan()
        first=stage_config(plan,plan['stages'][0])
        self.assertIsNone(first['init_checkpoint'])
        self.assertFalse(first['init_optimizer'])
        for stage in plan['stages'][1:]:
            config=stage_config(plan,stage,Path('/tmp/prior-stage.pt'))
            self.assertTrue(config['init_optimizer'])
            self.assertEqual(config['fleet_sizes'],[3])
            self.assertEqual(config['max_task_distance'],750.)
            self.assertEqual(config['distance_sampling'],'annulus_area')

    def test_annulus_area_and_initial_ground_centre(self):
        def stub(seed,distances,world,start,heading):
            distance,=distances
            key=distance_key(distance)
            goal=start+distance*np.array([np.cos(heading),np.sin(heading)])
            return dict(start=start.tolist(),goals={key:goal.tolist()},
                        references={key:dict(length=distance,points=[start.tolist(),goal.tolist()])})
        radii=[];angles=[]
        with patch('simulator.large_world_scenarios.generate_large_scene',side_effect=stub):
            for seed in range(1000):
                scene=generate_training_scene(seed,(),distance_range=[100.,chain_radius()],distance_sampling='annulus_area')
                vector=np.asarray(scene['goal'])-scene['start']
                radius=np.linalg.norm(vector)
                self.assertGreaterEqual(radius,100.)
                self.assertLessEqual(radius,chain_radius())
                self.assertAlmostEqual(radius,scene['distance'])
                radii.append((radius**2-100.**2)/(chain_radius()**2-100.**2))
                angles.append(vector/radius)
        self.assertAlmostEqual(np.mean(radii),.5,delta=.04)
        self.assertLess(np.linalg.norm(np.mean(angles,axis=0)),.08)

    def test_every_stage_saved_before_next_stage_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run'
            calls=[]
            class FakeTrainer:
                pid=123
                returncode=0
                def __init__(self,command,**kwargs):
                    config=json.loads(Path(command[command.index('--config')+1]).read_text())
                    folder=Path(command[command.index('--output')+1])
                    stage=len(calls)+1
                    if stage>1:
                        previous=output/f'stage_{stage-1:02d}_final.pt'
                        assert previous.exists() and config['init_checkpoint']==str(previous)
                        assert config['init_optimizer']
                    else:
                        assert config['init_checkpoint'] is None
                    calls.append(config)
                    torch.save({'model':{'test':torch.tensor(stage)}},folder/'latest.pt')
                    torch.save({'model':{'test':torch.tensor(-stage)}},folder/'best.pt')
                    (folder/'progress.json').write_text(json.dumps(dict(status='completed')))
                def poll(self):
                    return 0
            with patch('simulator.train_distance_curriculum.subprocess.Popen',FakeTrainer),patch(
                    'simulator.train_distance_curriculum.evaluate_distance_matrix',return_value=True):
                run(CONFIG,output,workers=1)
            self.assertEqual(len(calls),5)
            for stage in range(1,6):
                saved=torch.load(output/f'stage_{stage:02d}_final.pt',weights_only=True)
                self.assertEqual(saved['model']['test'].item(),stage)
                self.assertTrue((output/f'stage_{stage:02d}_best.pt').exists())
            with self.assertRaises(ValueError):
                preserve_checkpoint(output/'stage_02_final.pt',output/'stage_01_final.pt')

    def test_stage_limit_saves_first_stage_and_later_resume_preserves_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run';calls=[]
            class FakeTrainer:
                pid=123
                returncode=0
                def __init__(self,command,**kwargs):
                    config=json.loads(Path(command[command.index('--config')+1]).read_text())
                    folder=Path(command[command.index('--output')+1])
                    calls.append(config)
                    torch.save({'model':{'test':torch.tensor(len(calls))}},folder/'latest.pt')
                    torch.save({'model':{'test':torch.tensor(-len(calls))}},folder/'best.pt')
                    (folder/'progress.json').write_text(json.dumps(dict(status='completed')))
                def poll(self):return 0
            with patch('simulator.train_distance_curriculum.subprocess.Popen',FakeTrainer),patch(
                    'simulator.train_distance_curriculum.evaluate_distance_matrix',return_value=True):
                run(CONFIG,output,workers=1,through_stage=1)
                self.assertEqual(len(calls),1)
                self.assertIsNone(calls[0]['init_checkpoint'])
                self.assertFalse((output/'stage_02').exists())
                first=(output/'stage_01_final.pt').read_bytes()
                self.assertTrue((output/'stage_01_best.pt').exists())
                self.assertEqual(json.loads((output/'progress.json').read_text())['status'],'stage_limit_reached')
                run(CONFIG,output,workers=1,resume=True,through_stage=2)
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[1]['init_checkpoint'],str(output/'stage_01_final.pt'))
                self.assertEqual((output/'stage_01_final.pt').read_bytes(),first)
                self.assertFalse((output/'stage_03').exists())

    def test_curriculum_optimizer_upgrade_keeps_geometry_and_passes_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run';output.mkdir()
            stage=output/'stage_01';stage.mkdir()
            plan=self.plan();old=json.loads(json.dumps(plan))
            old['base_training_config']['rollout_mode']='asynchronous'
            for name in ('critic_epochs','critic_minibatch_size','value_normalization'):
                old['base_training_config'].pop(name)  # predates the separate critic
            (output/'curriculum.json').write_text(json.dumps(old))
            (stage/'progress.json').write_text(json.dumps(dict(status='stopped')))
            torch.save(dict(model={}),stage/'latest.pt')
            commands=[]
            class FakeTrainer:
                pid=123
                returncode=0
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    torch.save(dict(model={}),stage/'best.pt')
                    (stage/'progress.json').write_text(json.dumps(dict(status='completed')))
                def poll(self):return 0
            with self.assertRaises(ValueError):run(CONFIG,output,resume=True,through_stage=1)
            with patch('simulator.train_distance_curriculum.subprocess.Popen',FakeTrainer),patch(
                    'simulator.train_distance_curriculum.evaluate_distance_matrix',return_value=True):
                run(CONFIG,output,resume=True,through_stage=1,upgrade_optimizer=True)
            self.assertIn('--resume',commands[0])
            self.assertIn('--upgrade-optimizer',commands[0])
            self.assertEqual(json.loads((output/'curriculum_before_optimizer_upgrade.json').read_text()),old)
            self.assertEqual(json.loads((output/'curriculum.json').read_text())['stages'],old['stages'])
            self.assertTrue((output/'stage_01_final.pt').exists())
            self.assertFalse((output/'stage_02').exists())


if __name__=='__main__':
    unittest.main()
