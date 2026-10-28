"""v7 invariance, local connectivity, CBF and actual PPO/checkpoint regressions."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone.local_backbone import relative_neighborhood, received_geometry, backbone_metrics
from drone.cbf_safety import CBFSafetyFilter
from drone.cooperative_mappo import (CooperativeActorCritic, CooperativeNavigator,
    RAW_WIDTH, MAP_WIDTH, MAX_FLEET, PEER_FEATURES)
from drone.mappo import ActorCritic
from simulator.train_mappo import validate_config, train, run_episode, fixed_schedule, make_model, feature_version
from simulator.mappo_experiment import Experiment


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v7.json').read_text())


class BackboneTests(unittest.TestCase):
    def test_local_rule_is_symmetric_connected_and_id_independent(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            points = rng.uniform(10., 160., (12, 3)); points[:, 2] = 15.
            adjacency = []
            for i, p in enumerate(points):
                neighbors = {j: q for j,q in enumerate(points) if j!=i and np.linalg.norm(q-p)<=95.}
                edges = set(relative_neighborhood(p, neighbors, 95.))
                renamed = {f'name_{11-j}':q for j,q in reversed(list(neighbors.items()))}
                self.assertEqual({f'name_{11-j}' for j in edges}, set(relative_neighborhood(p, renamed, 95.)))
                adjacency.append(edges)
            for i, neighbors in enumerate(adjacency):
                for j in neighbors:self.assertIn(i, adjacency[j])
            # Compare components, rather than assuming every random sample connected.
            def reachable(edges):
                reached, stack = set(), [0]
                while stack:
                    j=stack.pop()
                    if j not in reached:reached.add(j);stack.extend(edges[j])
                return reached
            full = [{j for j,q in enumerate(points) if j!=i and np.linalg.norm(q-p)<=95.}
                    for i,p in enumerate(points)]
            self.assertEqual(reachable(adjacency), reachable(full))

    def test_geometry_changes_roles_and_ties_do_not_disconnect(self):
        p=np.array([0.,0.,15.])
        self.assertEqual(set(relative_neighborhood(p, {'a':[30,0,15], 'b':[60,0,15]},95)), {'a'})
        self.assertEqual(set(relative_neighborhood(p, {'a':[60,0,15], 'b':[30,0,15]},95)), {'b'})
        tied={'a':[30,0,15], 'b':[15,30*np.sqrt(3)/2,15]}
        self.assertEqual(set(relative_neighborhood(p,tied,95)), {'a','b'})

    def test_forwarded_positions_never_become_physical_neighbors(self):
        peers,_,ground=received_geometry(np.zeros(3),dict(neighbors=[dict(id=7,type='drone',
            relative_position=[5,6,15],broadcast=dict(drone_id='peer',position=[999,999,15],
            ground_missions={'g':dict(position=[0,0,0])}))],full_state=dict(drones=[[1,2,3]])))
        np.testing.assert_array_equal(peers['peer'],[5,6,15]);self.assertIsNone(ground)

    def test_cbf_preserves_selected_links_during_motion_and_reselection(self):
        rng=np.random.default_rng(17)
        ground=np.array([30.,100.,0.])
        points={'zeta':np.array([65.,100.,15.]),'alpha':np.array([135.,100.,15.]),
                'middle':np.array([185.,130.,15.])}
        cbf=CBFSafetyFilter(15.,95.,6.)
        for _ in range(20):
            velocities={}
            for key,p in points.items():
                visible={j:q for j,q in points.items() if j!=key and np.linalg.norm(q-p)<=95.}
                local=dict(visible)
                g=ground if np.linalg.norm(ground-p)<=95. else None
                if g is not None:local['__ground__']=g
                required=relative_neighborhood(p,local,95.)
                v,stats=cbf.filter(np.r_[rng.uniform(-6,6,2),0.],p,visible,1.,required,g,
                                   drone_id=key,ground_max_speed=0.)
                self.assertFalse(stats['infeasible']);velocities[key]=v
            for fraction in np.linspace(0,1,21):
                moved={key:p+fraction*velocities[key] for key,p in points.items()}
                self.assertTrue(backbone_metrics(moved,ground)[0])
                self.assertGreaterEqual(min(np.linalg.norm(moved[a]-moved[b]) for a in moved for b in moved if a!=b),15.)
            points={key:p+velocities[key] for key,p in points.items()}


class CooperativePolicyTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(12);torch.set_num_threads(1)

    def observation(self):
        peers={'peer':[20.,0.,15.],'other':[0.,30.,15.]}
        return dict(time=0.,graph=None,nearby_drones={},ground_robots={},
            occupancy_map=np.zeros((12,12)),cooperation=dict(peers=peers,
                velocities={k:np.zeros(3) for k in peers},ground=None,required=('peer','other')))

    def features(self, model, name='arbitrary_device'):
        nav=CooperativeNavigator(name,total_drones=3,model=model)
        nav.cbf_margin_source=lambda:2.
        return nav.encode([0.,0.,15.],self.observation())[0]

    def test_policy_ignores_ids_peer_order_and_masked_padding(self):
        model=CooperativeActorCritic().eval()
        x=self.features(model);np.testing.assert_array_equal(x,self.features(model,'different_name'))
        y=x.copy()
        peers=y[MAP_WIDTH:MAP_WIDTH+MAX_FLEET*PEER_FEATURES].reshape(MAX_FLEET,PEER_FEATURES)
        peers[[0,1]]=peers[[1,0]];peers[2:]=123.
        with torch.no_grad():
            torch.testing.assert_close(model.distribution(torch.tensor(x)).mean,
                                       model.distribution(torch.tensor(y)).mean)
            empty=x.copy();empty[-MAX_FLEET:]=0.
            self.assertTrue(torch.isfinite(model.distribution(torch.tensor(empty)).mean).all())
        changed=x.copy();changed[MAP_WIDTH]+=1.
        self.assertGreater(float((model.distribution(torch.tensor(x)).mean-model.distribution(torch.tensor(changed)).mean).detach().abs().max()),1e-8)

    def test_actual_multihop_delivery_with_reordered_physical_roles(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,CooperativeActorCritic(),communication_mode='backbone',safety_filter='cbf')
            # Device 1 is closest to the vehicle; device 0 is the distant explorer.
            for drone,p in zip(env.env.drones,([110,110,15],[40,60,15],[105,60,15])):
                drone.position=np.array(p,float)
            env.obs=env.env._get_observations()
            source=env.ids[0]
            self.assertNotIn(source,{n['id'] for n in env.obs[env.ground]['neighbors']})
            for frame in range(12):
                _,_,metrics=env.step(frame)
                self.assertEqual(metrics['backbone_connected'],1.)
                self.assertEqual(metrics['chain_link_violation'],0.)
        self.assertTrue(any(key[0]==source for key in env.ledger.received))
        self.assertTrue(any(data['position'][1]>90. for _,data in
                            env.ground_controller.unified_graph.graph.nodes(data=True)))
        self.assertNotIn('__ground__',env.controllers[0].navigator.required_links)
        self.assertIn('__ground__',env.controllers[1].navigator.required_links)

    def test_version_contract_and_unsupported_geometry_are_explicit(self):
        c=validate_config(config());self.assertEqual(feature_version(c),7)
        self.assertIsInstance(make_model(c),CooperativeActorCritic)
        for changes in (dict(drop=.1),dict(communication_mode='chain'),dict(actor_input_version=1)):
            with self.assertRaises(ValueError):validate_config(dict(c,**changes))

    def test_real_training_checkpoint_validation_and_resume(self):
        c=config();c.update(episodes=2,steps=5,episodes_per_batch=2,validation_interval=2,
            snapshot_interval=2,validation_seeds=[60000],validation_samples=1,ppo_epochs=1,critic_epochs=1)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            train(c,directory,workers=1)
            path=Path(directory);saved=torch.load(path/'latest.pt',weights_only=True)
            self.assertEqual(saved['feature_version'],7)
            self.assertEqual(saved['communication_mode'],'backbone')
            model=ActorCritic.from_state_dict(saved['model'])
            self.assertIsInstance(model,CooperativeActorCritic)
            episode,summary=run_episode(model,validate_config(c),50000,fixed_schedule(c,0),True)
            self.assertEqual(episode[0][0].shape,(3,RAW_WIDTH))
            self.assertTrue(all(np.isfinite(row[0]).all() for row in episode))
            self.assertEqual(summary['chain_link_violation'],0.)
            validation=json.loads((path/'validation.jsonl').read_text().splitlines()[-1])
            self.assertNotIn('surprise_reward',validation['means'])
            train(c,directory,resume=True,workers=1)


if __name__=='__main__':unittest.main()
