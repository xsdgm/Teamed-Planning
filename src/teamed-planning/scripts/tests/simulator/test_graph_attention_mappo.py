"""GAT observation, topology, PPO and checkpoint regressions."""
import copy, sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.graph_structure import TopologicalGraph
from drone.mappo import GLOBAL_FEATURES
from drone.graph_attention_mappo import (GraphAttentionActorCritic,GraphEncoder,encode_graph,
    PREFIX,MAX_NODES,NEIGHBORS,NODE_FEATURES,GAT_FEATURES)
from simulator.ppo_batch import update_batch,update_value_statistics


def graph(order=(0,1,2,3)):
    g=TopologicalGraph('test')
    for i in order:
        g.add_node(str(i),(float(i*10),float(i%2*5)),dict(terrain_type='TRAVERSABLE',observation_count=10))
    for i in range(3):g.add_edge(str(i),str(i+1))
    return g


def features(g=None):
    base=np.zeros(PREFIX,np.float32);base[4]=.2;base[10]=1.
    return encode_graph(base,(0.,0.,15.),dict(graph=g))


class GATTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(72);torch.set_num_threads(1)

    def test_encoding_preserves_prefix_and_uses_only_existing_edges(self):
        g=graph();x=features(g);self.assertEqual(len(x),GAT_FEATURES)
        idx=x[PREFIX+MAX_NODES*NODE_FEATURES:-MAX_NODES].reshape(MAX_NODES,NEIGHBORS)
        self.assertEqual(set(idx[0][idx[0]>=0]),{0.,1.})
        self.assertNotIn(3.,idx[0]);self.assertTrue((idx[4:]==-1).all())
        np.testing.assert_array_equal(x[:PREFIX],np.r_[np.zeros(4),.2,np.zeros(5),1.,np.zeros(PREFIX-11)].astype(np.float32))
        np.testing.assert_array_equal(x,features(graph((3,1,0,2))))

    def test_caps_and_disconnected_components_do_not_create_edges(self):
        g=TopologicalGraph('test')
        for i in range(100):g.add_node(str(i),(float(i%10*2),float(i//10*2)))
        x=features(g);self.assertEqual(int(x[-MAX_NODES:].sum()),MAX_NODES)
        idx=x[PREFIX+MAX_NODES*NODE_FEATURES:-MAX_NODES].reshape(MAX_NODES,NEIGHBORS)
        self.assertTrue((idx[:,1:]==-1).all())
        np.testing.assert_array_equal(idx[:,0],np.arange(MAX_NODES))

    def test_graph_encoder_is_permutation_invariant(self):
        x=features(graph());encoder=GraphEncoder().eval()
        permutation=np.arange(MAX_NODES);permutation[:4]=[3,1,0,2]
        inverse=np.argsort(permutation)
        y=x.copy();nodes=x[PREFIX:PREFIX+MAX_NODES*NODE_FEATURES].reshape(MAX_NODES,NODE_FEATURES)
        idx=x[PREFIX+MAX_NODES*NODE_FEATURES:-MAX_NODES].reshape(MAX_NODES,NEIGHBORS)
        y[PREFIX:PREFIX+MAX_NODES*NODE_FEATURES]=nodes[permutation].ravel()
        permidx=idx[permutation].astype(int);valid=permidx>=0
        permidx[valid]=inverse[permidx[valid]]
        y[PREFIX+MAX_NODES*NODE_FEATURES:-MAX_NODES]=permidx.ravel()
        y[-MAX_NODES:]=x[-MAX_NODES:][permutation]
        with torch.no_grad():torch.testing.assert_close(encoder(torch.tensor(x)),encoder(torch.tensor(y)),rtol=1e-5,atol=1e-6)

    def test_attention_responds_to_edges_and_empty_graph_is_finite(self):
        model=GraphAttentionActorCritic();x=torch.tensor(features(graph()))
        changed=x.clone();offset=PREFIX+MAX_NODES*NODE_FEATURES
        idx=changed[offset:-MAX_NODES].reshape(MAX_NODES,NEIGHBORS);idx[:,1:]=-1
        with torch.no_grad():
            a=model.distribution(x).mean;b=model.distribution(changed).mean
            self.assertGreater(float((a-b).abs().max()),1e-8)
            self.assertTrue(torch.isfinite(model.distribution(torch.tensor(features())).mean).all())
            value=model.value(torch.tensor(np.array([features(),features()])),state=torch.zeros(GLOBAL_FEATURES))
            self.assertTrue(torch.isfinite(value))

    def test_ppo_updates_actor_and_private_critic_with_popart(self):
        model=GraphAttentionActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        rollouts=[]
        for ep in range(2):
            rows=[]
            for t in range(5):
                x=np.array([features(graph()) for _ in range(ep+2)]);x[:,:4]=t*.01
                with torch.no_grad():d=model.distribution(torch.tensor(x));a=d.sample();lp=d.log_prob(a)
                rows.append((x,np.ones((ep+2,1),bool),a.numpy(),lp.numpy(),float(t),np.zeros(GLOBAL_FEATURES,np.float32)))
            rollouts.append(rows)
        before=copy.deepcopy(model.state_dict())
        _,stats=update_batch(model,optimizer,rollouts,epochs=2,critic_epochs=2,
                             critic_minibatch_size=4,value_normalization=True)
        self.assertLessEqual(stats['exact_kl'],.02)
        self.assertGreater(stats['critic_updates'],0)
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('gat_actor.')))
        rebuilt=GraphAttentionActorCritic()
        rebuilt.load_state_dict(model.state_dict())
        with torch.no_grad():
            x=torch.tensor(rollouts[0][0][0]);s=torch.zeros(GLOBAL_FEATURES)
            value=model.value(x,state=s).clone();update_value_statistics(model,torch.tensor([100.,200.,300.]))
            torch.testing.assert_close(value,model.value(x,state=s),atol=1e-4,rtol=1e-4)


if __name__=='__main__':unittest.main()
