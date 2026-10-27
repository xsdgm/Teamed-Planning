"""v6 behavioral tests: reward gates/budget, fixed scoring, learning and restart."""
import copy
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator.surprise_reward import SurpriseReward, transition, MAP_FEATURES
from simulator.train_mappo import validate_config,reward_contract


def config():
    path=Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v6_surprise.json'
    return validate_config(__import__('json').loads(path.read_text()))


def item(gate=1.):
    before=np.zeros(MAP_FEATURES,np.float32)
    after=np.full(MAP_FEATURES,.2,np.float32)
    return dict(before=before,after=after,action=np.zeros(4,np.float32),
                mask=np.ones(MAP_FEATURES,np.float32),gate=gate)


def rollout(items):
    return [(np.zeros((3,2)),np.ones((3,1)),np.zeros((3,2)),np.zeros(3),-1.,np.zeros(10),r)
            for r in items]


class SurpriseTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_gate_excludes_initialization_revisits_connection_and_unsafe_links(self):
        before=np.zeros(MAP_FEATURES,np.float32);after=np.ones(MAP_FEATURES,np.float32)
        m=dict(chain_links_available=1,graph_task_connected=0,cbf_margin_risk=0,task_exploration_area=400)
        def gate(metrics=m,frame=1,post=after):
            return transition(before,post,np.ones((3,3)),metrics,frame)
        self.assertEqual(gate()['gate'],1.)
        self.assertEqual(gate(frame=0)['gate'],0.)
        self.assertEqual(gate(dict(m,chain_links_available=0))['gate'],0.)
        self.assertEqual(gate(dict(m,graph_task_connected=1))['gate'],0.)
        self.assertEqual(gate(dict(m,cbf_margin_risk=1))['gate'],0.)
        self.assertEqual(gate(post=before)['mask'].sum(),0.)
        np.testing.assert_allclose(gate()['action'],[1/6,1/6,0,0])

    def test_budget_preserves_extrinsic_and_fails_closed_on_absorbing_terminal(self):
        c=config();c['surprise_episode_budget']=.2
        reward=SurpriseReward(c);rows=rollout([item() for _ in range(20)]+[None]);summary={'reward':-21.}
        saved=copy.deepcopy(reward.state_dict())
        reward.apply([rows],[summary],0)
        self.assertAlmostEqual(summary['surprise_reward'],.2)
        self.assertAlmostEqual(sum(r[4] for r in rows),-20.8)
        self.assertEqual(summary['reward'],-21.)
        self.assertEqual(rows[-1][4],-1.)
        self.assertTrue(all(len(r)==6 for r in rows))
        for k,v in saved['model'].items():torch.testing.assert_close(v,reward.model.state_dict()[k])
        self.assertEqual(reward.batches,0)

    def test_zero_gate_and_revisit_have_no_bonus(self):
        model=SurpriseReward(config());a=item();a['mask'][:]=0
        rows=rollout([item(0.),a]);s={'reward':-2.}
        model.apply([rows],[s],0)
        self.assertEqual(s['surprise_reward'],0.)

    def test_predictor_learns_and_resume_continues_identically(self):
        c=config();c['surprise_learning_rate']=.005
        model=SurpriseReward(c);data=[[item() for _ in range(16)]]
        x=torch.tensor(np.r_[data[0][0]['before'],data[0][0]['action']])[None,None]
        target=torch.tensor(data[0][0]['after'])
        def loss(m):
            with torch.no_grad():return float((m.model(x)[0][0,0]-target).square().mean())
        initial=loss(model)
        for _ in range(12):model.fit(data)
        self.assertLess(loss(model),initial*.5)
        resumed=SurpriseReward(c);resumed.load_state_dict(model.state_dict())
        for m in (model,resumed):
            rows=rollout(data[0]);s={'reward':-16.};m.apply([rows],[s],100)
            m.fit(data)
        for k,v in model.model.state_dict().items():torch.testing.assert_close(v,resumed.model.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(model.error_square_mean,resumed.error_square_mean)

    def test_real_episode_metadata_and_disabled_reward_are_compatible(self):
        from simulator.train_mappo import run_episode,fixed_schedule
        from drone.velocity_mappo import VelocityActorCritic
        c=config();c['steps']=4
        model=VelocityActorCritic()
        state=torch.get_rng_state().clone()
        rows,summary=run_episode(model,c,50000,fixed_schedule(c,0),True)
        self.assertTrue(rows)
        self.assertEqual(rows[0][6]['action'].shape,(4,))
        self.assertEqual(rows[0][6]['before'].shape,(MAP_FEATURES,))
        torch.set_rng_state(state)
        disabled=dict(c,surprise_weight=0.,surprise_weight_final=0.)
        old,_=run_episode(model,disabled,50000,fixed_schedule(disabled,0),True)
        self.assertEqual(len(rows),len(old))
        for a,b in zip(rows,old):
            for x,y in zip(a[:6],b):np.testing.assert_array_equal(x,y)
        rewards=SurpriseReward(c)
        rewards.apply([rows],[summary],0)
        self.assertAlmostEqual(summary['training_reward'],summary['reward']+summary['surprise_reward'],places=6)

    def test_trainer_stop_resume_matches_uninterrupted_predictor_and_policy(self):
        import contextlib,io,tempfile
        from unittest.mock import patch
        from simulator import train_mappo as trainer
        c=config();c.update(episodes=4,steps=4,validation_interval=4,
                            snapshot_interval=4,validation_seeds=[60000],validation_samples=1,
                            ppo_epochs=1,critic_epochs=1)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            full=Path(folder)/'full';partial=Path(folder)/'partial'
            trainer.train(c,full,workers=2)
            original=trainer.atomic_checkpoint
            def save(path,value):
                original(path,value)
                if path==partial/'latest.pt' and value['training_episodes']==2:
                    (partial/'STOP').touch()
            with patch.object(trainer,'atomic_checkpoint',side_effect=save):
                trainer.train(c,partial,workers=2)
            saved=torch.load(partial/'latest.pt',weights_only=True)
            self.assertEqual(saved['training_episodes'],2)
            self.assertEqual(saved['surprise']['batches'],1)
            (partial/'STOP').unlink()
            trainer.train(c,partial,resume=True,workers=2)
            a=torch.load(full/'latest.pt',weights_only=True)
            b=torch.load(partial/'latest.pt',weights_only=True)
            self.assertEqual(a['surprise']['batches'],2)
            self.assertEqual(a['surprise']['error_square_mean'],b['surprise']['error_square_mean'])
            for group in ('model',):
                for key,value in a[group].items():torch.testing.assert_close(value,b[group][key],rtol=0,atol=0)
            for key,value in a['surprise']['model'].items():
                torch.testing.assert_close(value,b['surprise']['model'][key],rtol=0,atol=0)

    def test_config_contract_and_rng_isolation(self):
        c=config();self.assertIn('v6_surprise',reward_contract(c))
        c['surprise_weight']=c['surprise_weight_final']=0
        self.assertTrue(reward_contract(validate_config(c)).endswith('v5'))
        for changes in [dict(rollout_mode='asynchronous'),dict(exploration_weight=.2),
                        dict(surprise_episode_budget=100),dict(surprise_weight_final=1.),
                        dict(surprise_weight=float('nan'))]:
            with self.assertRaises(ValueError):validate_config(dict(config(),**changes))
        state=torch.get_rng_state().clone();SurpriseReward(config())
        self.assertTrue(torch.equal(state,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
