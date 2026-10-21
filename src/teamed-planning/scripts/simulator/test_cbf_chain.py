"""Swept separation, chain connectivity, recovery and actual packet relay tests."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone.cbf_safety import CBFSafetyFilter
from drone.chain_state import ChainState
from drone.mappo import ActorCritic,CandidateNavigator
from drone.drone_main import DroneController
from simulator.mappo_experiment import Experiment


class SafetyTests(unittest.TestCase):
    def test_head_on_swept_separation(self):
        filt=CBFSafetyFilter()
        a=np.array([20.,40.,15.]);b=np.array([40.,40.,15.])
        va,sa=filt.filter([6,0,0],a,{'b':b},1.,drone_id='a')
        vb,sb=filt.filter([-6,0,0],b,{'a':a},1.,drone_id='b')
        self.assertFalse(sa['infeasible'] or sb['infeasible'])
        for t in np.linspace(0,1,101):
            self.assertGreaterEqual(np.linalg.norm(a+va*t-b-vb*t),15.25-1e-6)

    def test_holding_inside_margin_recovers(self):
        filt=CBFSafetyFilter()
        a=np.array([20.,40.,15.]);b=np.array([34.94,40.,15.])
        va,sa=filt.filter([0,0,0],a,{'b':b},1.,drone_id='a')
        vb,sb=filt.filter([0,0,0],b,{'a':a},1.,drone_id='b')
        self.assertFalse(sa['infeasible'] or sb['infeasible'])
        self.assertGreaterEqual(np.linalg.norm(a+va-b-vb),15.25-1e-6)

    def test_chain_upper_distance_and_speed(self):
        filt=CBFSafetyFilter()
        a=np.array([10.,40.,15.]);b=np.array([104.,40.,15.])
        va,sa=filt.filter([-6,0,0],a,{'b':b},1.,required_links=['b'],drone_id='a')
        vb,sb=filt.filter([6,0,0],b,{'a':a},1.,required_links=['a'],drone_id='b')
        self.assertFalse(sa['infeasible'] or sb['infeasible'])
        for t in np.linspace(0,1,101):
            self.assertLessEqual(np.linalg.norm(a+va*t-b-vb*t),95.)
        self.assertLessEqual(np.linalg.norm(va),6.+1e-6)

    def test_ground_radio_bound_is_three_dimensional(self):
        filt=CBFSafetyFilter()
        p=np.array([93.,0.,15.])
        v,s=filt.filter([6,0,0],p,{},1.,required_links=['__ground__'],ground=[0,0,0])
        self.assertFalse(s['infeasible'])
        self.assertLessEqual(np.linalg.norm(p+v),95.)

    def test_infeasibility_is_explicit(self):
        filt=CBFSafetyFilter()
        _,s=filt.filter([0,0,0],[30,30,15],{'a':[29,30,15],'b':[31,30,15]},1.)
        self.assertTrue(s['infeasible'])

    def test_vertical_motion_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            CBFSafetyFilter().filter([0,0,1],[20,20,15],{},1.)

    def test_checkpoint_preserves_chain_and_cbf_execution_settings(self):
        artifact_root=Path(__file__).resolve().parents[4]/'artifacts'/'cbf-tests'
        artifact_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(artifact_root.resolve()))
            path=Path(directory)/'model.pt'
            torch.save(dict(model=ActorCritic().state_dict(),feature_version=1,
                            communication_mode='chain',safety_filter='cbf'),path)
            controller=DroneController('drone_0',total_drones=3,
                navigation_algorithm='mappo',mappo_checkpoint=str(path))
            self.assertEqual(controller.communication_mode,'chain')
            self.assertEqual(controller.safety_filter,'cbf')
            with self.assertRaises(NotImplementedError):
                controller.get_action({},mode='real')

    def test_random_joint_steps_preserve_received_chain_and_spacing(self):
        rng=np.random.default_rng(123)
        filt=CBFSafetyFilter()
        ground=np.array([15.,60.,0.])
        for _ in range(12):
            while True:
                positions=np.c_[rng.uniform(20,100,(3,2)),np.full(3,15.)]
                if (min(np.linalg.norm(positions[i]-positions[j]) for i in range(3) for j in range(i))>18.
                    and max(np.linalg.norm(positions[0]-positions[1]),np.linalg.norm(positions[1]-positions[2]))<85.
                    and np.linalg.norm(positions[0]-ground)<85.):
                    break
            for _ in range(8):
                commands=[]
                for i in range(3):
                    peers={str(j):positions[j] for j in range(3) if j!=i and
                           np.linalg.norm(positions[i]-positions[j])<=95.}
                    required=(['__ground__','1'] if i==0 else ['0','2'] if i==1 else ['1'])
                    nominal=np.r_[rng.uniform(-6,6,2),0.]
                    v,stats=filt.filter(nominal,positions[i],peers,1.,required_links=required,
                        ground=ground if i==0 else None,world_bounds=(0,0,120,120),drone_id=str(i))
                    self.assertFalse(stats['infeasible'])
                    commands.append(v)
                for t in np.linspace(0,1,21):
                    current=positions+t*np.asarray(commands)
                    self.assertGreaterEqual(min(np.linalg.norm(current[i]-current[j])
                        for i in range(3) for j in range(i)),15.-1e-5)
                    self.assertLessEqual(max(np.linalg.norm(current[0]-current[1]),
                        np.linalg.norm(current[1]-current[2]),np.linalg.norm(current[0]-ground)),95.+1e-5)
                positions+=np.asarray(commands)

    def test_nonadjacent_peer_is_not_required(self):
        nav=CandidateNavigator('drone_2',total_drones=3,communication_mode='chain',algorithm='greedy',comm_range=95.)
        obs=dict(time=1.,nearby_drones={'drone_1':dict(position=[40.,60.,15.])},
            ground_robots={'g':dict(position=[15.,60.,0.],target=[110.,60.],direct=False,source_time=1.)})
        _,mask,_,_=nav.encode((60.,60.,15.),obs)
        self.assertEqual(nav.chain.required,('drone_1',))
        self.assertGreater(mask.sum(),1)

    def test_chain_slack_masks_lengthening_links_beyond_the_slack_range(self):
        def link_ends(position,link_range):
            nav=CandidateNavigator('drone_2',total_drones=3,communication_mode='chain',algorithm='greedy',
                comm_range=95.,exploration_candidates=8,chain_link_range=link_range)
            obs=dict(time=1.,nearby_drones={'drone_1':dict(position=[400.,500.,15.])},
                ground_robots={'g':dict(position=[330.,500.,0.],target=[700.,500.],direct=False,source_time=1.)})
            _,mask,candidates,_=nav.encode(position,obs)
            p=np.asarray(position,float)
            ends=[p+(np.asarray(w)-p)*min(1.,6./max(np.linalg.norm(np.asarray(w)-p),1e-6)) for w in candidates]
            return [float(np.linalg.norm(e-[400.,500.,15.])) for e in ends],mask
        lengths,mask=link_ends((488.,500.,15.),95.)
        self.assertTrue(any(m and d>90. for d,m in zip(lengths[1:],mask[1:])))
        lengths,mask=link_ends((488.,500.,15.),90.)
        self.assertTrue(all(d<=90. for d,m in zip(lengths[1:],mask[1:]) if m))
        self.assertGreater(mask.sum(),1)
        # Already beyond the slack range: only motions that shorten the link remain.
        lengths,mask=link_ends((492.,500.,15.),90.)
        self.assertTrue(all(d<92. for d,m in zip(lengths[1:],mask[1:]) if m))
        self.assertGreater(mask.sum(),1)
        with self.assertRaises(ValueError):
            CandidateNavigator('drone_2',total_drones=3,communication_mode='chain',comm_range=95.,
                               chain_link_range=96.)

    def test_chain_never_rewires_around_missing_role(self):
        state=ChainState('drone_2',3)
        state.update(dict(time=0,nearby_drones={'drone_0':dict(position=[10,30,15])}))
        self.assertEqual(state.required,('drone_1',))
        self.assertEqual(state.missing([80,30,15],95.),('drone_1',))

    def test_recovery_uses_only_recent_received_history(self):
        state=ChainState('drone_2',3)
        state.update(dict(time=0,nearby_drones={'drone_1':dict(position=[10.,30.,15.])}))
        state.update(dict(time=1,nearby_drones={}))
        self.assertEqual(state.recovery([100.,30.,15.],95.),(10.,30.,15.))
        state.update(dict(time=20,nearby_drones={}))
        self.assertEqual(state.recovery([100.,30.,15.],95.),(100.,30.,15.))

    def test_hold_command_is_not_zeroed_after_cbf(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),communication_mode='chain',safety_filter='cbf')
            env.env.drones[1].position=np.array([24.,56.94,15.])
            env.obs=env.env._get_observations()
            for c in env.controllers:
                def hold(p,o,controller=c):
                    controller.navigator._current_target=tuple(p)
                    return tuple(p)
                c.navigator.compute_next_target=hold
            old=np.array([d.position.copy() for d in env.env.drones])
            env.step(0)
            current=np.array([d.position.copy() for d in env.env.drones])
        self.assertGreater(np.linalg.norm(current-old),0.)
        self.assertGreaterEqual(np.linalg.norm(current[0]-current[1]),15.25-1e-6)

    def test_evidence_reaches_ground_through_received_peer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),communication_mode='chain',safety_filter='cbf')
            for drone,p in zip(env.env.drones,([40,60,15],[105,60,15],[110,110,15])):
                drone.position=np.array(p,float)
            env.obs=env.env._get_observations()
            source=env.ids[2]
            self.assertNotIn(source,{n['id'] for n in env.obs[env.ground]['neighbors']})
            for frame in range(4):
                env.step(frame)
        self.assertTrue(any(key[0]==source for key in env.ledger.received))
        self.assertFalse(env.controllers[2].navigator.chain.rank==0)


if __name__=='__main__':
    unittest.main()
