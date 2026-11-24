"""SMDP learning, evidence masks and hierarchical reward contracts."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from vtemp2_3.navigation_evidence import EvidenceGrid,AttemptMemory
from vtemp2_3.hierarchical_runtime import (HierarchyConfig,HeldTarget,SLOW_FEATURES,MAX_CANDIDATES,build_observation,
                             mission_reward,option_duration_limit,geometry_teacher_action)
from vtemp2_3.slow_policy import SlowActorCritic,smdp_advantages,update
from vtemp2_3.navigation_scenes import make_scene,feasible
from vtemp2_3.train_slow_policy import load_checkpoint,atomic_checkpoint,worker_state


class HierarchyContracts(unittest.TestCase):
    def test_rotated_radius_boundary_keeps_exact_final_target(self):
        case=make_scene(26121602,(200.,200.),2000.,'open')
        b=dict(position=case['start'],goal=case['goal'],
               grid=EvidenceGrid(np.zeros((100,100),np.int8),20.))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        final=obs['kinds'].index('final')
        self.assertTrue(obs['mask'][final])
        np.testing.assert_array_equal(obs['targets'][final],case['goal'])

    def test_unknown_candidates_remain_selectable(self):
        b=dict(position=[250.,250.],goal=[450.,250.],grid=EvidenceGrid(np.zeros((250,250),np.int8)))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        unknown=[i for i,s in enumerate(obs['statuses']) if s=='unknown']
        self.assertTrue(unknown)
        self.assertTrue(obs['mask'][unknown].all())
        self.assertEqual(obs['features'].shape,(MAX_CANDIDATES,len(SLOW_FEATURES)))
        self.assertFalse(obs['mask'][~obs['present']].any())

    def test_surrounded_vehicle_has_hold_without_invented_drive_path(self):
        cells=np.ones((250,250),np.int8);cells[120:131,120:131]=-1
        b=dict(position=[250.,250.],goal=[450.,250.],grid=EvidenceGrid(cells))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        allowed=np.flatnonzero(obs['mask'])
        self.assertEqual(len(allowed),1)
        self.assertEqual(obs['kinds'][allowed[0]],'hold')

    def test_attempt_history_changes_features_but_not_physical_masks(self):
        b=dict(position=[250.,250.],goal=[450.,250.],grid=EvidenceGrid(np.zeros((250,250),np.int8)))
        memory=AttemptMemory();before=build_observation(b,memory,HierarchyConfig())
        memory.record(b['position'],before['targets'][0],0.,'timeout')
        after=build_observation(b,memory,HierarchyConfig(),now=1.)
        np.testing.assert_array_equal(before['mask'],after['mask'])
        self.assertGreater(after['features'][0,SLOW_FEATURES.index('attempt_cost')],0.)
        self.assertFalse(b['grid'].cells.any())

    def test_held_target_never_reverts_to_original_geometric_selector(self):
        manager=HeldTarget([800.,500.],[100.,500.]);manager.target=np.array([220.,540.])
        for now in (0.,180.,600.,1200.):
            target,changed=manager.update([100.,500.],None,now)
            np.testing.assert_array_equal(target,[220.,540.]);self.assertFalse(changed)

    def test_target_switch_cannot_generate_task_reward(self):
        c=HierarchyConfig()
        a,_=mission_reward(500,500,c)
        b,_=mission_reward(500,500,c)
        self.assertEqual(a,b)
        self.assertLess(a,0.)

    def test_discounted_shaping_telescopes_across_detour(self):
        c=replace(HierarchyConfig(),gamma=.9)
        distances=[100.,120.,80.,0.]
        shaping=[]
        for i in range(3):
            _,parts=mission_reward(distances[i],distances[i+1],c,success=i==2)
            shaping.append(parts['shaping'])
        self.assertAlmostEqual(sum(c.gamma**i*v for i,v in enumerate(shaping)),100/c.potential_scale)

    def test_timeout_and_safety_failure_have_different_terminal_rewards(self):
        c=HierarchyConfig()
        ongoing,_=mission_reward(300,300,c)
        failed,_=mission_reward(300,300,c,safety_failure=True)
        self.assertGreater(ongoing,failed+90.)

    def test_candidates_have_side_and_rear_actions(self):
        b=dict(position=[400.,400.],goal=[800.,400.],grid=EvidenceGrid(np.zeros((500,500),np.int8)))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        q=obs['targets'][obs['present']]
        self.assertTrue((q[:,0]<350).any());self.assertTrue((q[:,1]>450).any())

    def test_hidden_fields_do_not_enter_slow_policy_observation(self):
        b=dict(position=[250.,250.],goal=[450.,250.],grid=EvidenceGrid(np.zeros((250,250),np.int8)))
        first=build_observation(b,AttemptMemory(),HierarchyConfig())
        b['hidden_map']=np.ones((250,250));b['future_return']=999.
        second=build_observation(b,AttemptMemory(),HierarchyConfig())
        np.testing.assert_array_equal(first['features'],second['features'])

    def test_kilometer_curriculum_creates_real_scale_task(self):
        case=make_scene(50001,(1100.,1100.),2000.,'wall_gap')
        self.assertAlmostEqual(np.linalg.norm(np.asarray(case['goal'])-case['start']),1100.)
        self.assertTrue(feasible(case))

    def test_near_known_final_is_selected_before_actor_sampling(self):
        b=dict(position=[250.,250.],goal=[270.,250.],grid=EvidenceGrid(-np.ones((250,250),np.int8)))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        chosen=np.flatnonzero(obs['mask'])
        self.assertEqual(len(chosen),1)
        self.assertEqual(obs['kinds'][chosen[0]],'final')
        self.assertEqual(obs['decision_contract'],'known_final_approach')
        model=SlowActorCritic(len(SLOW_FEATURES),16)
        action,logp,_=model.act(obs)
        self.assertEqual(action,chosen[0]);self.assertAlmostEqual(logp,0.)

    def test_unknown_near_final_keeps_exploration_choices(self):
        b=dict(position=[250.,250.],goal=[270.,250.],grid=EvidenceGrid(np.zeros((250,250),np.int8)))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        self.assertGreater(obs['mask'].sum(),1)
        self.assertEqual(obs['decision_contract'],'policy_choice')

    def test_blocked_near_final_does_not_force_a_drive_into_wall(self):
        cells=-np.ones((250,250),np.int8);cells[:,130:134]=1
        b=dict(position=[250.,250.],goal=[280.,250.],grid=EvidenceGrid(cells))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        self.assertFalse(obs['mask'][obs['kinds'].index('final')])
        action=geometry_teacher_action(obs,b['position'],b['goal'])
        self.assertNotEqual(obs['kinds'][action],'final')
        self.assertTrue(obs['mask'][action])

    def test_near_final_known_detour_remains_ground_planners_task(self):
        cells=-np.ones((250,250),np.int8);cells[115:136,130:132]=1
        b=dict(position=[250.,250.],goal=[280.,250.],grid=EvidenceGrid(cells))
        obs=build_observation(b,AttemptMemory(),HierarchyConfig())
        self.assertEqual(obs['statuses'][0],'known_reachable')
        self.assertEqual(obs['decision_contract'],'known_final_approach')
        # The direct line crosses an obstacle; connectivity authorizes the
        # original planner's detour, never an unchecked straight-line motion.
        values,_=b['grid'].sample(np.array([[261.,250.],[263.,250.]]))
        self.assertTrue((values==1).any())

    def test_adaptive_budget_covers_travel_and_preserves_legacy_diagnostics(self):
        config=replace(HierarchyConfig(),option_steps=24)
        self.assertEqual(option_duration_limit(200.,'ring',config,1.),380)
        self.assertEqual(option_duration_limit(200.,'final',config,1.),426)
        self.assertEqual(option_duration_limit(200.,'hold',config,1.),10)
        self.assertEqual(option_duration_limit(200.,'ring',replace(config,adaptive_options=False),1.),24)
        self.assertEqual(option_duration_limit(200.,'ring',config,.1),config.max_option_steps)

    def test_teacher_continues_progress_but_reconsiders_stalled_target(self):
        b=dict(position=[250.,250.],goal=[450.,250.],grid=EvidenceGrid(np.zeros((250,250),np.int8)))
        prev=dict(target=np.array([250.,450.]),duration=80,reason='option_budget')
        obs=build_observation(b,AttemptMemory(),HierarchyConfig(),previous=prev)
        self.assertEqual(obs['kinds'][geometry_teacher_action(obs,b['position'],b['goal'])],'continuation')
        prev=dict(prev,reason='control_stall')
        obs=build_observation(b,AttemptMemory(),HierarchyConfig(),previous=prev)
        self.assertNotEqual(obs['kinds'][geometry_teacher_action(obs,b['position'],b['goal'])],'continuation')

    def test_stalled_final_releases_finishing_contract_for_recovery(self):
        b=dict(position=[250.,250.],goal=[270.,250.],grid=EvidenceGrid(-np.ones((250,250),np.int8)))
        memory=AttemptMemory();memory.record(b['position'],b['goal'],0.,'control_stall')
        previous=dict(target=b['goal'],duration=80,reason='control_stall')
        obs=build_observation(b,memory,HierarchyConfig(),previous=previous,now=1.)
        self.assertGreater(obs['mask'].sum(),1)
        self.assertEqual(obs['statuses'][0],'known_reachable')
        chosen=geometry_teacher_action(obs,b['position'],b['goal'])
        self.assertNotEqual(obs['kinds'][chosen],'final')
        later=build_observation(b,memory,HierarchyConfig(),previous=previous,now=180.)
        self.assertEqual(later['decision_contract'],'known_final_approach')


class LearningContracts(unittest.TestCase):
    def test_variable_duration_and_truncation_bootstrap(self):
        data=[dict(reward=1.,duration=2,value=.5,next_value=2.,terminated=False,truncated=False),
              dict(reward=3.,duration=4,value=2.,next_value=4.,terminated=False,truncated=True)]
        adv,ret=smdp_advantages(data,.9,.8)
        last=3+.9**4*4-2
        self.assertAlmostEqual(float(adv[1]),last,places=5)
        self.assertAlmostEqual(float(adv[0]),1+.9**2*2-.5+.9**2*.8*last,places=5)
        self.assertAlmostEqual(float(ret[0]),float(adv[0])+.5,places=5)

    def test_true_terminal_discards_bootstrap_and_stops_cross_episode_carry(self):
        data=[dict(reward=1.,duration=2,value=.5,next_value=999.,terminated=True,truncated=False),
              dict(reward=1000.,duration=3,value=0.,next_value=0.,terminated=True,truncated=False)]
        adv,_=smdp_advantages(data,.9,.95)
        self.assertEqual(float(adv[0]),.5)

    def test_truncation_bootstraps_without_next_episode_advantage(self):
        data=[dict(reward=1.,duration=2,value=.5,next_value=2.,terminated=False,truncated=True),
              dict(reward=1000.,duration=3,value=0.,next_value=0.,terminated=True,truncated=False)]
        adv,_=smdp_advantages(data,.9,.95)
        self.assertAlmostEqual(float(adv[0]),2.12,places=5)

    def test_policy_mask_and_permutation_invariant_value(self):
        torch.manual_seed(7);model=SlowActorCritic(4,16)
        x=torch.randn(5,4);present=torch.tensor([True,True,True,False,False]);mask=torch.tensor([True,False,True,False,False])
        distribution,value=model(x,mask,present)
        self.assertEqual(float(distribution.probs[1].detach()),0.)
        permutation=torch.tensor([2,4,0,1,3])
        other,other_value=model(x[permutation],mask[permutation],present[permutation])
        torch.testing.assert_close(other_value,value)
        torch.testing.assert_close(other.probs,distribution.probs[permutation])

    def batch(self,model):
        rng=np.random.default_rng(4);data=[]
        for i in range(6):
            x=torch.tensor(rng.normal(size=(3,4)),dtype=torch.float32)
            mask=torch.tensor([True,True,False]);present=torch.tensor([True,True,True])
            with torch.no_grad():dist,value=model(x,mask,present)
            action=i%2
            data.append(dict(features=x.numpy(),mask=mask.numpy(),present=present.numpy(),action=action,
                logp=float(dist.log_prob(torch.tensor(action))),value=float(value),reward=1. if action==0 else -1.,
                duration=i+1,next_value=0.,terminated=i==5,truncated=False))
        return [data]

    def test_on_policy_update_changes_weights(self):
        torch.manual_seed(2);model=SlowActorCritic(4,16)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        batch=self.batch(model);before=copy.deepcopy(model.state_dict())
        result=update(model,optimizer,batch,epochs=2)
        self.assertGreater(result['accepted_epochs'],0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_stale_rollout_is_rejected(self):
        model=SlowActorCritic(4,16);batch=self.batch(model);batch[0][0]['logp']+=.1
        with self.assertRaisesRegex(ValueError,'Stale'):
            update(model,torch.optim.Adam(model.parameters()),batch)

    def test_rejected_kl_step_restores_parameters_and_optimizer(self):
        model=SlowActorCritic(4,16);optimizer=torch.optim.Adam(model.parameters(),lr=.1)
        batch=self.batch(model);before=copy.deepcopy(model.state_dict())
        result=update(model,optimizer,batch,epochs=1,target_kl=1e-14,max_kl=2e-14)
        self.assertEqual(result['rejected_epochs'],1)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertFalse(optimizer.state)

    def test_checkpoint_contract_roundtrip_and_rejection(self):
        model=SlowActorCritic(4,16);expected=dict(schema='test',gamma=.999)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            atomic_checkpoint(path,dict(contract=expected,model=model.state_dict()))
            restored=load_checkpoint(path,expected)
            self.assertIn('actor_head.weight',restored['model'])
            with self.assertRaisesRegex(ValueError,'contract'):
                load_checkpoint(path,dict(schema='test',gamma=.99))

    def test_nonfinite_optimizer_step_restores_model_and_optimizer(self):
        model=SlowActorCritic(4,16);optimizer=torch.optim.Adam(model.parameters())
        batch=self.batch(model);before=copy.deepcopy(model.state_dict())
        original_step=optimizer.step
        def corrupt_step():
            original_step()
            with torch.no_grad():next(model.parameters()).fill_(float('nan'))
        with patch.object(optimizer,'step',side_effect=corrupt_step):
            result=update(model,optimizer,batch,epochs=1)
        self.assertEqual(result['rejected_epochs'],1)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertFalse(optimizer.state)

    def test_worker_policy_snapshot_is_independent_of_later_updates(self):
        model=SlowActorCritic(4,16);snapshot=worker_state(model)
        original={k:v.copy() for k,v in snapshot.items()}
        with torch.no_grad():
            for parameter in model.parameters():parameter.add_(1.)
        for key,value in snapshot.items():
            self.assertIsInstance(value,np.ndarray)
            np.testing.assert_array_equal(value,original[key])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
