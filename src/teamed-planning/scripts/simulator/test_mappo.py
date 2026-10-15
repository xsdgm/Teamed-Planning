"""Information-boundary, evidence-accounting and optimizer regression tests."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic, CandidateNavigator
from common.graph_structure import TopologicalGraph
from simulator.mappo_experiment import DeliveredEvidence, Reward, update


class MAPPOTests(unittest.TestCase):
    def test_duplicate_source_frame_has_no_effect(self):
        ledger = DeliveredEvidence()
        packet = (1, 0, [30,30,15], np.zeros((4,4)), 2.)
        self.assertTrue(ledger.receive(*packet))
        before = ledger.confidence().copy()
        self.assertFalse(ledger.receive(*packet))
        np.testing.assert_array_equal(before, ledger.confidence())

    def test_conflict_can_reduce_confidence(self):
        ledger = DeliveredEvidence()
        ledger.receive(1,0,[30,30,15],np.zeros((4,4)),2.)
        before = ledger.confidence().sum()
        ledger.receive(2,0,[30,30,15],np.ones((4,4)),2.)
        self.assertLess(ledger.confidence().sum(),before)
        self.assertTrue(ledger.seen.any())

    def test_invalid_fallback_is_not_evidence(self):
        ledger = DeliveredEvidence()
        ledger.receive(1,0,[30,30,15],np.zeros((4,4)),2., valid=False)
        self.assertFalse(ledger.seen.any())

    def test_missing_task_holds_and_ignores_privileged_fields(self):
        nav = CandidateNavigator('drone_0', algorithm='greedy')
        p = (20.,30.,15.)
        obs = dict(full_state={'all_robots_positions': np.ones((10,3))},
                   ground_robots={'g': {'broadcast': {'target':[100,100]}, 'position':[10,10]}})
        self.assertEqual(nav.compute_next_target(p, obs),p)

    def test_white_list_and_candidate_permutation(self):
        nav = CandidateNavigator('drone_0', algorithm='greedy', comm_range=100.)
        obs = dict(ground_robots={'g':dict(position=[15,30,0],target=[80,30],planned_path=[])})
        a = nav.encode((20,30,15),obs)
        obs['ground_robots']['g']['broadcast'] = {'secret':np.ones(100), 'target':[999,999]}
        obs['full_state'] = {'obstacles': np.ones(100)}
        b = nav.encode((20,30,15),obs)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
        model = ActorCritic()
        x,mask = torch.from_numpy(a[0]),torch.from_numpy(a[1])
        perm = torch.randperm(len(mask))
        np.testing.assert_allclose(model.distribution(x,mask).probs.detach().numpy()[perm],
            model.distribution(x[perm],mask[perm]).probs.detach().numpy(), atol=1e-7)

    def test_missing_neighbor_forces_hold(self):
        nav = CandidateNavigator('drone_0',total_drones=3,algorithm='greedy')
        _,mask,_,_ = nav.encode((20,30,15),dict(ground_robots={'g':dict(position=[15,30,0],target=[80,30])}))
        self.assertEqual(mask.sum(),1)

    def test_controlled_admm_returns_shared_candidate(self):
        nav=CandidateNavigator('drone_0',algorithm='cadmm_controlled',comm_range=100.)
        graph=TopologicalGraph('drone_0')
        graph.add_node('a',(20.,30.))
        graph.add_node('b',(40.,30.))
        graph.add_edge('a','b')
        obs=dict(graph=graph,ground_robots={'g':dict(position=[15.,30.,0.],target=[80.,30.])})
        _,mask,candidates,_=nav.encode((20.,30.,15.),obs)
        target=nav.compute_next_target((20.,30.,15.),obs)
        self.assertIn(target,[w for i,w in enumerate(candidates) if mask[i]])

    def test_failed_inference_holds(self):
        model=ActorCritic()
        with torch.no_grad():
            model.actor[0].weight.fill_(float('nan'))
        nav=CandidateNavigator('drone_0',model=model)
        self.assertEqual(nav.compute_next_target((20.,30.,15.),{}),(20.,30.,15.))
        self.assertGreater(nav.fallbacks,0)

    def test_negative_gain_and_first_seen_persist(self):
        reward = Reward()
        before = np.array([[.8,.8]])
        after = np.array([[.2,.2]])
        seen = np.ones_like(before,bool)
        r,m = reward.compute(before,after,seen,seen,np.array([[True,False]]),0.)
        self.assertLess(r,0.)
        self.assertEqual(m['outside_new'],0.)

    def test_ppo_updates_actor_and_critic(self):
        torch.manual_seed(1)
        model = ActorCritic()
        initial_actor = model.actor[0].weight.detach().clone()
        initial_critic = model.critic[0].weight.detach().clone()
        rollout=[]
        for t in range(8):
            x = torch.randn(3,21,12)
            mask = torch.ones(3,21,dtype=torch.bool)
            with torch.no_grad():
                dist=model.distribution(x,mask)
                actions=dist.sample()
                logp=dist.log_prob(actions)
            rollout.append((x.numpy(),mask.numpy(),actions.numpy(),logp.numpy(),float(t%3)))
        loss=update(model,torch.optim.Adam(model.parameters(),lr=3e-4),rollout)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(initial_actor,model.actor[0].weight))
        self.assertFalse(torch.equal(initial_critic,model.critic[0].weight))


if __name__ == '__main__':
    unittest.main()
