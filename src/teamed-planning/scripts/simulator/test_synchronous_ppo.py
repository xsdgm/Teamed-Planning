"""Behavioral regressions for frozen-policy batches, KL rollback and migration."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import GLOBAL_FEATURES
from drone.velocity_mappo import VelocityActorCritic,VELOCITY_FEATURES
from simulator.ppo_batch import clip_gradients,episode_advantages,update_batch,update_value_statistics
from simulator.train_mappo import train,validate_config,optimizer_config_changes,optimizer_contract
from simulator.test_parallel_training import fake_episode,rows


def velocity_rollout(model,steps=12,agents=3):
    result=[]
    for t in range(steps):
        x=torch.randn(agents,VELOCITY_FEATURES)*.1
        mask=torch.ones(agents,1,dtype=torch.bool)
        with torch.no_grad():
            dist=model.distribution(x,mask);action=dist.sample();logp=dist.log_prob(action)
        result.append((x.numpy(),mask.numpy(),action.numpy(),logp.numpy(),float(action[:,0].sum()),
                       np.zeros(GLOBAL_FEATURES,np.float32)))
    return result


def assert_state_equal(case,left,right):
    if isinstance(left,torch.Tensor):case.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        case.assertEqual(left.keys(),right.keys())
        for key in left:assert_state_equal(case,left[key],right[key])
    elif isinstance(left,(tuple,list)):
        case.assertEqual(len(left),len(right))
        for a,b in zip(left,right):assert_state_equal(case,a,b)
    else:case.assertEqual(left,right)


class BatchUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(31)

    def test_gae_stops_at_each_episode_boundary(self):
        a,ra=episode_advantages(torch.tensor([1.,2.]),torch.zeros(2),1.,1.)
        b,rb=episode_advantages(torch.tensor([100.]),torch.zeros(1),1.,1.)
        torch.testing.assert_close(a,torch.tensor([3.,2.]))
        torch.testing.assert_close(ra,a);torch.testing.assert_close(rb,b)
        self.assertEqual(float(rb[0]),100.)

    def test_rejected_step_restores_parameters_and_existing_adam_moments(self):
        model=VelocityActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        update_batch(model,optimizer,[velocity_rollout(model)],epochs=1)
        optimizer.param_groups[0]['lr']=100.
        rollout=velocity_rollout(model)
        weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict())
        _,metrics=update_batch(model,optimizer,[rollout],max_backtracks=0)
        self.assertEqual(metrics['epochs_completed'],0)
        self.assertEqual(metrics['kl_rejected_attempts'],1)
        self.assertLessEqual(metrics['exact_kl'],.02)
        assert_state_equal(self,weights,model.state_dict())
        assert_state_equal(self,state,optimizer.state_dict())

    def test_backtracking_accepts_smaller_step_and_preserves_base_learning_rate(self):
        model=VelocityActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=.2)
        _,metrics=update_batch(model,optimizer,[velocity_rollout(model)],epochs=1,max_backtracks=12)
        self.assertEqual(metrics['epochs_completed'],1)
        self.assertGreater(metrics['kl_rejected_attempts'],0)
        self.assertLess(metrics['minimum_lr_scale'],1.)
        self.assertLessEqual(metrics['exact_kl'],.02)
        self.assertEqual(optimizer.param_groups[0]['lr'],.2)
        self.assertTrue(all(int(v['step'])==1 for v in optimizer.state.values()))

    def test_target_kl_stops_remaining_epochs(self):
        model=VelocityActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        _,metrics=update_batch(model,optimizer,[velocity_rollout(model)],epochs=4,target_kl=1e-12,max_kl=1.)
        self.assertEqual(metrics['epochs_completed'],1)
        self.assertEqual(metrics['kl_early_stopped'],1.)

    def test_huge_critic_gradient_does_not_clip_actor_gradient(self):
        model=VelocityActorCritic()
        for name,p in model.named_parameters():
            p.grad=torch.full_like(p,1e3 if name.startswith('critic.') else 1e-4)
        actor=[p.grad.clone() for name,p in model.named_parameters() if not name.startswith('critic.')]
        actor_norm,critic_norm=clip_gradients(model)
        self.assertLess(actor_norm,.5);self.assertGreater(critic_norm,.5)
        for g,(name,p) in zip(actor,[(n,p) for n,p in model.named_parameters() if not n.startswith('critic.')]):
            torch.testing.assert_close(p.grad,g)
        critic=torch.cat([p.grad.flatten() for p in model.critic.parameters()])
        self.assertAlmostEqual(float(critic.norm()),.5,places=4)

    def test_terminal_scale_returns_keep_warm_policy_step(self):
        # Adam's first step is scale-free and momentum hides one shrunken step;
        # warm second moments on small returns, as the real run did, with beta1=0.
        kls=[]
        for scale in (1.,100.):
            torch.manual_seed(7)
            model=VelocityActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(0.,.999))
            for _ in range(3):update_batch(model,optimizer,[velocity_rollout(model)],epochs=1)
            rollout=[r[:4]+(r[4]*scale,)+r[5:] for r in velocity_rollout(model)]
            _,metrics=update_batch(model,optimizer,[rollout],epochs=1)
            kls.append(metrics['exact_kl'])
        self.assertGreater(kls[1],.3*kls[0])

    def test_stale_rollout_is_rejected_before_any_optimizer_step(self):
        model=VelocityActorCritic();rollout=velocity_rollout(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        with torch.no_grad():model.velocity_actor[-1].bias.add_(.5)
        weights=copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'Stale/mismatched'):
            update_batch(model,optimizer,[rollout])
        assert_state_equal(self,weights,model.state_dict())
        self.assertEqual(len(optimizer.state),0)

    def test_same_batch_handles_different_fleet_sizes_without_padding_or_candidates(self):
        model=VelocityActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        rollouts=[velocity_rollout(model,steps=n,agents=n) for n in (3,4,5)]
        _,metrics=update_batch(model,optimizer,rollouts,epochs=2)
        self.assertEqual(metrics['batch_episodes'],3)
        self.assertEqual(metrics['batch_steps'],12)
        self.assertLessEqual(metrics['on_policy_logp_max_error'],1e-4)
        self.assertLessEqual(metrics['exact_kl'],.02)


def predictable_rollout(model,steps=64,scale=1000.,dims=8):
    # The reward is a function of the current state; with discount 0 the
    # return is learnable, at terminal-penalty scale (std ~58). Few active
    # features: 256 samples of 832 noisy ones would only test overfitting.
    result=[]
    for _ in range(steps):
        x=torch.zeros(3,VELOCITY_FEATURES);x[:,:dims]=torch.randn(3,dims)*.1
        mask=torch.ones(3,1,dtype=torch.bool)
        with torch.no_grad():
            dist=model.distribution(x,mask);action=dist.sample();logp=dist.log_prob(action)
        result.append((x.numpy(),mask.numpy(),action.numpy(),logp.numpy(),float(scale*x[:,0].mean()),
                       np.zeros(GLOBAL_FEATURES,np.float32)))
    return result


class SeparateCriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(37)

    def test_popart_keeps_values_and_old_checkpoints_load_as_identity(self):
        model=VelocityActorCritic();x=torch.randn(5,3,VELOCITY_FEATURES);state=torch.zeros(5,GLOBAL_FEATURES)
        with torch.no_grad():before=model.value(x,None,state)
        update_value_statistics(model,torch.tensor([-100.,-5.,0.,20.]))
        self.assertAlmostEqual(float(model.value_mean),-21.25,places=5)
        update_value_statistics(model,torch.tensor([300.,-50.]))
        with torch.no_grad():torch.testing.assert_close(model.value(x,None,state),before,rtol=1e-5,atol=1e-4)
        self.assertGreater(float(model.value_std),10.)
        legacy={k:v for k,v in VelocityActorCritic().state_dict().items() if not k.startswith('value_')}
        old=VelocityActorCritic.from_state_dict(legacy)
        self.assertEqual((float(old.value_mean),float(old.value_std)),(0.,1.))
        with torch.no_grad():
            self.assertTrue(torch.equal(old.value(x,None,state),old.critic(old.critic_input(x,None,state)).squeeze(-1)))

    def test_critic_epochs_leave_the_policy_step_identical_and_are_reproducible(self):
        base=VelocityActorCritic();rollouts=[predictable_rollout(base) for _ in range(2)]
        def run(critic_epochs,seed=5):
            model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            _,metrics=update_batch(model,optimizer,rollouts,discount=0.,gae_lambda=0.,critic_epochs=critic_epochs,
                critic_minibatch_size=32,value_normalization=True,critic_seed=seed)
            return model,metrics
        one,first=run(1);many,metrics=run(8);again,_=run(8);shuffled,_=run(8,seed=6)
        actor=lambda m:{k:v for k,v in m.state_dict().items() if not k.startswith(('critic.','value_'))}
        assert_state_equal(self,actor(one),actor(many))
        assert_state_equal(self,many.state_dict(),again.state_dict())
        self.assertFalse(torch.equal(many.critic[0].weight,shuffled.critic[0].weight))
        self.assertEqual((first['critic_updates'],metrics['critic_updates']),(4,32))
        self.assertLessEqual(metrics['exact_kl'],.02)
        self.assertGreater(metrics['value_std'],10.)

    def test_separate_normalized_critic_learns_terminal_scale_returns(self):
        base=VelocityActorCritic();torch.manual_seed(41);held=predictable_rollout(base,steps=256)
        x=torch.tensor(np.array([r[0] for r in held]));target=torch.tensor([r[4] for r in held])
        def heldout_explained_variance(model):
            with torch.no_grad():value=model.value(x,None,torch.zeros(len(held),GLOBAL_FEATURES))
            return float(1-(target-value).var()/target.var())
        results={}
        for name,options in (('shared',dict()),
                             ('separate',dict(critic_epochs=16,critic_minibatch_size=32,value_normalization=True))):
            model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            for batch in range(3):
                torch.manual_seed(100+batch)
                rollouts=[predictable_rollout(model) for _ in range(4)]
                _,metrics=update_batch(model,optimizer,rollouts,discount=0.,gae_lambda=0.,critic_seed=batch,**options)
            results[name]=heldout_explained_variance(model)
        self.assertLess(results['shared'],.1)
        self.assertGreater(results['separate'],.9)

    def test_config_and_contract_for_separate_critic(self):
        options=dict(rollout_mode='synchronous',action_mode='velocity',exploration_candidates=0,
            task_distances=[250],validation_seeds=[20000],test_seeds=[30000])
        self.assertEqual(optimizer_contract(validate_config(options)),'synchronous_kl_guard_v2')
        config=validate_config(dict(options,critic_epochs=8,value_normalization=True))
        self.assertEqual(optimizer_contract(config),'synchronous_kl_guard_v3')
        for fields in (dict(value_normalization=True),dict(critic_epochs=8,rollout_mode='asynchronous'),
                       dict(critic_epochs=8,action_mode='candidate',exploration_candidates=5),
                       dict(critic_epochs=-1),dict(critic_epochs=2,critic_minibatch_size=0)):
            with self.assertRaises(ValueError):validate_config(dict(options,**fields))
        with self.assertRaises(ValueError):
            update_batch(VelocityActorCritic(),None,[velocity_rollout(VelocityActorCritic())],value_normalization=True)


class SynchronousTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)/'run'

    def tearDown(self):self.temporary.cleanup()

    def config(self,**options):
        return validate_config(dict(dict(episodes=6,steps=4,validation_interval=2,snapshot_interval=3,
            validation_seeds=[20000],test_seeds=[30000],initial_validation=False,rollout_mode='synchronous'),**options))

    def test_batches_have_no_policy_lag_and_keep_validation_save_boundaries(self):
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode),contextlib.redirect_stdout(io.StringIO()):
            train(self.config(),self.root,workers=3,mp_context='fork')
        records=rows(self.root/'training.jsonl')
        self.assertEqual([r['episode'] for r in records],list(range(1,7)))
        self.assertTrue(all(r['policy_lag']==0 for r in records))
        self.assertTrue(all(r['ppo']['on_policy_logp_max_error']<=1e-4 for r in records))
        self.assertEqual([r['episode'] for r in rows(self.root/'validation.jsonl')],[2,4,6])
        self.assertTrue((self.root/'episode_00003.pt').exists())
        batches={}
        for r in records:batches.setdefault(r['optimizer_batch'],[]).append(r)
        self.assertEqual([len(v) for v in batches.values()],[2,1,1,2])
        for group in batches.values():
            self.assertEqual(len({r['behavior_policy_version'] for r in group}),1)
        self.assertEqual(json.loads((self.root/'progress.json').read_text())['status'],'completed')

    def test_stop_discards_interrupted_batch_then_resume_completes(self):
        config=self.config()
        def stop_immediately(*args,**kwargs):
            (self.root/'STOP').touch()
            return fake_episode(*args,**kwargs)
        with patch('simulator.train_mappo.run_episode',side_effect=stop_immediately),contextlib.redirect_stdout(io.StringIO()):
            train(config,self.root,workers=2,mp_context='fork')
        stopped=torch.load(self.root/'latest.pt',weights_only=True)
        self.assertEqual(stopped['training_episodes'],0)
        self.assertEqual(stopped['optimizer_batches'],0)
        self.assertEqual(rows(self.root/'training.jsonl'),[])
        self.assertGreater(len(rows(self.root/'discarded_rollouts.jsonl')),0)
        (self.root/'STOP').unlink()
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode),contextlib.redirect_stdout(io.StringIO()):
            train(config,self.root,resume=True,workers=2,mp_context='fork')
        self.assertEqual(torch.load(self.root/'latest.pt',weights_only=True)['training_episodes'],6)

    def test_optimizer_upgrade_is_explicit_preserves_checkpoint_and_rejects_reward_changes(self):
        old=self.config(rollout_mode='asynchronous')
        from simulator import train_mappo
        original=train_mappo.append_json
        def stop_two(path,row):
            original(path,row)
            if path.name=='training.jsonl' and row['episode']==2:(self.root/'STOP').touch()
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode), \
             patch('simulator.train_mappo.append_json',side_effect=stop_two),contextlib.redirect_stdout(io.StringIO()):
            train(old,self.root,workers=1)
        previous=(self.root/'latest.pt').read_bytes()
        with self.assertRaises(ValueError):train(self.config(),self.root,resume=True,workers=2)
        with self.assertRaises(ValueError):optimizer_config_changes(old,dict(self.config(),arrival_reward=1.),True)
        (self.root/'STOP').unlink()
        with patch('simulator.train_mappo.run_episode',side_effect=fake_episode),contextlib.redirect_stdout(io.StringIO()):
            train(self.config(),self.root,resume=True,workers=2,mp_context='fork',upgrade_optimizer=True)
        self.assertEqual((self.root/'before_optimizer_upgrade_episode_00002.pt').read_bytes(),previous)
        saved=torch.load(self.root/'latest.pt',weights_only=True)
        self.assertEqual(saved['training_episodes'],6)
        self.assertEqual(saved['optimizer_contract'],'synchronous_kl_guard_v2')
        provenance=json.loads((self.root/'resume_provenance.json').read_text())
        self.assertEqual(provenance['optimizer_upgrade']['rollout_mode'],dict(before='asynchronous',after='synchronous'))
        self.assertTrue((self.root/'sources_optimizer_upgrade_00002/simulator/ppo_batch.py').exists())

    def test_real_velocity_simulations_and_spawned_workers_use_guarded_batches(self):
        config=self.config(episodes=2,steps=2,action_mode='velocity',exploration_candidates=0,
            world_size=1000.,task_distances=[250],reward_mode='velocity_mapping',
            map_success_reward=100.,path_quality_weight=20.,map_quality_reward=20.)
        with contextlib.redirect_stdout(io.StringIO()):train(config,self.root,workers=2)
        records=rows(self.root/'training.jsonl')
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['rollout_steps']==2 for r in records))
        self.assertTrue(all(r['ppo']['exact_kl']<=.02 for r in records))
        self.assertEqual(records[0]['optimizer_batch'],records[1]['optimizer_batch'])

    def test_real_velocity_training_with_separate_normalized_critic(self):
        config=self.config(episodes=2,steps=3,action_mode='velocity',exploration_candidates=0,
            world_size=1000.,task_distances=[250],reward_mode='velocity_mapping',
            map_success_reward=100.,path_quality_weight=20.,map_quality_reward=20.,
            critic_epochs=2,critic_minibatch_size=4,value_normalization=True)
        with contextlib.redirect_stdout(io.StringIO()):train(config,self.root,workers=2)
        records=rows(self.root/'training.jsonl')
        self.assertTrue(all(r['optimizer_contract']=='synchronous_kl_guard_v3' for r in records))
        self.assertEqual(records[0]['ppo']['critic_updates'],2*2)  # 6 samples / 4, twice
        self.assertIn('explained_variance_pre',records[0]['ppo'])
        saved=torch.load(self.root/'latest.pt',weights_only=True)
        self.assertEqual(saved['optimizer_contract'],'synchronous_kl_guard_v3')
        self.assertNotEqual(float(saved['model']['value_std']),1.)
        self.assertEqual(json.loads((self.root/'provenance.json').read_text())['optimizer_contract'],'synchronous_kl_guard_v3')


if __name__=='__main__':unittest.main(verbosity=2)
