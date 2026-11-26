import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[5]
SRC = REPO/'src/teamed-planning/scripts/vtemp5'
sys.path.insert(0,str(SRC))
from v91_trainable_guidance import TrainableGuidance, _options, branch_policy


class PrefixSensor:
    def __init__(self, allow):
        self.allow = allow

    def segment_clear(self,a,b,clearance):
        assert clearance == 1.5
        return self.allow and np.linalg.norm(b-a) <= 3


class BranchTests(unittest.TestCase):
    def test_geometry_offers_measured_gap_and_each_alternative_before_the_draw(self):
        xs,ys = np.meshgrid(np.arange(1,96,2),np.arange(1,96,2))
        xy = np.stack((xs,ys),-1)
        car = np.array([48.,48.])
        theta = math.radians(50.)
        forward = np.array([math.cos(theta),math.sin(theta)])
        lateral = np.array([-forward[1],forward[0]])
        along, across = (xy-car)@forward, abs((xy-car)@lateral)
        banks = (along>=4) & (along<=25) & (across>=4) & (across<=10)
        arguments = (xy,~banks,banks,None,{},1440.)
        choices = _options(car,np.array([110.,48.]),0.,PrefixSensor(True),*arguments)
        self.assertEqual([o['branch'] for o in choices],['forward','opening','left','right'])
        opening = choices[1]
        self.assertLess(abs(opening['degrees']-50.),18.)
        self.assertTrue(all(o['prefix']>=1.5 for o in choices))
        self.assertEqual(len({round(o['angle'],5) for o in choices}),len(choices))
        self.assertEqual(_options(car,np.array([110.,48.]),0.,PrefixSensor(False),*arguments),[])

    def test_no_action_and_single_action_remain_timed_without_actor_credit(self):
        import torch
        guide = TrainableGuidance.__new__(TrainableGuidance)
        guide.commands,guide.transitions = [],[]
        guide.deterministic = True
        car = np.array([42.,42.])
        snapshot = dict(car_xy=car,goal_xy=car+[25.,0.],frame=0,air_cells=np.zeros((80,80)),
                        resolution=2.,execution_feedback={})
        option = dict(branch='forward',target=car+[2.,0.],angle=0.,prefix=2.,final_goal=False)
        obs = dict(branch_angles=[0.],branch_prior=[0.])
        class Angle:
            def log_prob(self,turn,a):
                return torch.zeros_like(a,dtype=torch.float32)
        now = lambda observations:(Angle(),torch.zeros(len(observations)))
        for options in ([],[option]):
            with patch.object(guide,'_inputs',return_value=(obs,options)):
                dec = guide.review(None,snapshot,None,now,'observed')
            row = guide.transitions[-1]
            self.assertFalse(row['policy_controlled'])
            self.assertEqual((row['logp'],row['action']), (0.,0))
            self.assertEqual(dec['candidate_count'],len(options))
            self.assertEqual(dec['target_xy'],(option['target'] if options else car).tolist())
            guide.after_issue(dec,dec['target_xy'],True)
        self.assertEqual(len(guide.transitions),2)

    def test_physical_collection_and_stored_support_replay(self):
        protocol = REPO/'artifacts/v91-guided-km-final-20261126/protocol.json'
        if not protocol.is_file():
            self.skipTest('Archived fleet checkpoint is unavailable')
        subprocess.run([sys.executable,'-IB',str(Path(__file__)),'--probe',str(protocol)],cwd=REPO,
                       timeout=150,check=True)


def physical_probe(protocol):
    import torch
    import run_v91_km_eval as runner
    runner.load_models(json.loads(Path(protocol).read_text())['weights'])
    from v91_bridge_curriculum import bridge_curriculum_scene
    from v91_guided_drive import run_km_actor_arm
    scene = bridge_curriculum_scene('mouth',0,seed=141)
    actor = runner.MODEL_STATE[-1]
    guide = TrainableGuidance(deterministic=False,seed=73)
    roll = run_km_actor_arm(scene,*runner.MODEL_STATE,'predicted',guidance=guide,max_steps=61)
    assert not roll['ground_collisions'] and not roll['ground_safety_violations']
    assert [r['step'] for r in guide.transitions] == [0,30,60]
    assert guide.bootstrap_step == 61 and math.isfinite(guide.bootstrap_value)
    for row, decision in zip(guide.transitions,roll['reviews']):
        if row['policy_controlled']:
            target = decision['candidate_targets'][row['action']]
            assert decision['target_xy'] == target == roll['trace_motion'][row['step']]['vehicle_target']
    actor.requires_grad_(True)
    dist, _ = branch_policy(actor,[r['observation'] for r in guide.transitions])
    actual = dist.log_prob(torch.tensor([r['action'] for r in guide.transitions]))
    expected = torch.tensor([r['logp'] for r in guide.transitions])
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=0)
    assert torch.isfinite(dist.entropy()).all()
    (-actual.mean()).backward()
    assert torch.isfinite(actor.actor.weight.grad).all() and actor.actor.weight.grad.norm()>0
    assert len({r['branch'] for r in guide.transitions}) > 1
    try:
        run_km_actor_arm(scene,*runner.MODEL_STATE,'predicted',max_steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError('The frozen baseline accepted a short course')


if __name__=='__main__':
    if sys.argv[1:2]==['--probe']:
        physical_probe(sys.argv[2])
    else:
        unittest.main()
