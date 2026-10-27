"""GAT observation, topology, PPO, checkpoint and live-baseline regressions."""
import copy,contextlib,io,json,sys,tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.graph_structure import TopologicalGraph
from drone.mappo import ActorCritic,GLOBAL_FEATURES
from drone.velocity_mappo import VelocityActorCritic
from drone.graph_attention_mappo import (GraphAttentionActorCritic,GraphEncoder,encode_graph,
    PREFIX,MAX_NODES,NEIGHBORS,NODE_FEATURES,GAT_FEATURES)
from simulator.ppo_batch import update_batch,update_value_statistics
from simulator.train_mappo import train,validate_config,run_episode,fixed_schedule


def config():
    p=Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v6_gat_surprise.json'
    return validate_config(json.loads(p.read_text()))


def graph(order=(0,1,2,3)):
    g=TopologicalGraph('test')
    for i in order:
        g.add_node(str(i),(float(i*10),float(i%2*5)),dict(terrain_type='TRAVERSABLE',observation_count=10))
    for i in range(3):g.add_edge(str(i),str(i+1))
    return g


def features(g=None):
    base=np.zeros(832,np.float32);base[4]=.2;base[10]=1.
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
        rebuilt=ActorCritic.from_state_dict(model.state_dict())
        self.assertIsInstance(rebuilt,GraphAttentionActorCritic)
        with torch.no_grad():
            x=torch.tensor(rollouts[0][0][0]);s=torch.zeros(GLOBAL_FEATURES)
            value=model.value(x,state=s).clone();update_value_statistics(model,torch.tensor([100.,200.,300.]))
            torch.testing.assert_close(value,model.value(x,state=s),atol=1e-4,rtol=1e-4)

    def test_real_rollout_training_checkpoint_and_validation(self):
        c=config();c.update(episodes=2,steps=5,validation_interval=2,snapshot_interval=2,
                           validation_seeds=[60000],validation_samples=1,ppo_epochs=1,critic_epochs=1)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            train(c,directory,workers=2)
            p=Path(directory);saved=torch.load(p/'latest.pt',weights_only=True)
            self.assertEqual(saved['feature_version'],4);self.assertEqual(saved['policy_architecture'],'gat_map_v1')
            self.assertEqual(saved['surprise']['batches'],1)
            rows=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
            for r in rows:self.assertAlmostEqual(r['reward']+r['surprise_reward'],r['training_reward'],places=5)
            v=json.loads((p/'validation.jsonl').read_text().splitlines()[-1])
            self.assertNotIn('surprise_reward',v['means'])
            model=ActorCritic.from_state_dict(saved['model'])
            episode,_=run_episode(model,c,50000,fixed_schedule(c,0),True)
            self.assertEqual(episode[0][0].shape,(3,GAT_FEATURES))
            train(c,directory,resume=True,workers=2)

    def test_existing_v6_mlp_encoding_matches_frozen_running_sources(self):
        import importlib.util
        frozen=Path('/home/bdi/team-planning-zql/Teamed-Planning/artifacts/mappo-v6-stage1-20261026-120411/stage_01/sources/drone/velocity_mappo.py')
        if not frozen.exists():self.skipTest('Original v6 source snapshot is not on this host')
        spec=importlib.util.spec_from_file_location('frozen_velocity_v6',frozen)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        from drone.velocity_mappo import VelocityNavigator
        model=VelocityActorCritic()
        kwargs=dict(drone_id='drone_0',total_drones=3,model=model)
        before=old.VelocityNavigator(**kwargs);after=VelocityNavigator(**kwargs)
        obs=dict(graph=graph(),time=1.,ground_robots={},nearby_drones={},neighbor_waypoints={},
                 occupancy_map=np.zeros((12,12)),sensor_valid=True)
        a,_=before.encode((0.,0.,15.),obs);b,_=after.encode((0.,0.,15.),obs)
        np.testing.assert_array_equal(a,b)
        rebuilt=ActorCritic.from_state_dict(model.state_dict())
        with torch.no_grad():torch.testing.assert_close(model.distribution(torch.tensor(a)).mean,
                                                       rebuilt.distribution(torch.tensor(b)).mean,rtol=0,atol=0)

    def test_invalid_architecture_is_rejected(self):
        with self.assertRaises(ValueError):validate_config(dict(config(),policy_architecture='unknown'))
        with self.assertRaises(ValueError):validate_config(dict(config(),rollout_mode='asynchronous'))


if __name__=='__main__':unittest.main()
