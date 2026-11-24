"""Archived CBF conflicts and continuous-step safety under received geometry."""
import json
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.drone.cbf_safety import CBFSafetyFilter, filter_joint
from shared.drone.communication import relative_neighborhood


class CBFRecoveryTests(unittest.TestCase):
    def test_all_four_archived_conflicts_admit_safe_joint_commands(self):
        fixtures = json.loads((Path(__file__).parent/'fixtures/wildscenes_cbf_states.json').read_text())
        rng = np.random.default_rng(118)
        for case in fixtures:
            positions = {str(i): np.array(p) for i, p in enumerate(case['positions'])}
            ground = np.array(case['ground'])
            for attempt in range(5):
                requests, required_links = {}, {}
                independent_failures=0
                for name, pos in positions.items():
                    peers = {other: q for other, q in positions.items()
                             if other != name and np.linalg.norm(pos-q) <= 95.}
                    seen_ground = ground if np.linalg.norm(pos-ground) <= 95. else None
                    visible = dict(peers)
                    if seen_ground is not None:
                        visible['__ground__'] = ground
                    required = relative_neighborhood(pos, visible, 95.)
                    nominal = np.zeros(3) if attempt == 0 else np.r_[rng.uniform(-6., 6., 2), 0.]
                    velocity, stats = CBFSafetyFilter().filter(nominal, pos, peers, 1.,
                        required_links=required, ground=seen_ground, ground_max_speed=1.,
                        world_bounds=(0., 0., case['world_size_m'], case['world_size_m']), drone_id=name)
                    independent_failures+=int(stats['infeasible'])
                    requests[name]=dict(position=pos,nominal=nominal,peers=peers,dt=1.,
                        required_links=required,ground=seen_ground,ground_max_speed=1.,
                        world_bounds=(0.,0.,case['world_size_m'],case['world_size_m']),
                        max_speed=6.,min_separation=15.,comm_range=95.,margin=.25,sample_time=0.)
                    required_links[name]=required
                self.assertGreater(independent_failures,0)
                commands,diagnostic=filter_joint(requests)
                with self.subTest(case=case['case'],attempt=attempt):
                    self.assertIsNotNone(commands,diagnostic)
                    self.assertFalse(diagnostic['infeasible'])
                for velocity in commands.values():
                    self.assertLessEqual(np.linalg.norm(velocity),6.+1e-7)
                for time in np.linspace(0., 1., 21):
                    current = {name: p+time*commands[name] for name, p in positions.items()}
                    for name, p in current.items():
                        for other, q in current.items():
                            if name != other:
                                self.assertGreaterEqual(np.linalg.norm(p-q), 15.-1e-6)
                            if other in required_links[name]:
                                self.assertLessEqual(np.linalg.norm(p-q), 95.+1e-6)
                        if '__ground__' in required_links[name]:
                            # Exact worst horizontal ground displacement at
                            # this time, including the fixed 15m flight height.
                            worst = np.hypot(np.linalg.norm(p[:2]-ground[:2])+time, p[2])
                            self.assertLessEqual(worst, 95.+1e-6)

    def test_missing_or_stale_participants_are_not_given_commands(self):
        request=dict(position=np.array([30.,30.,15.]),nominal=np.zeros(3),
            peers={'b':np.array([50.,30.,15.])},dt=1.,required_links=('b',),
            ground=None,ground_max_speed=0.,world_bounds=(0.,0.,100.,100.),
            max_speed=6.,min_separation=15.,comm_range=95.,margin=.25,sample_time=0.)
        self.assertIsNone(filter_joint({'a':request})[0])
        second=dict(request,position=np.array([50.,30.,15.]),
                    peers={'a':request['position']},required_links=('a',),sample_time=1.)
        self.assertIsNone(filter_joint({'a':request,'b':second})[0])
        second['sample_time']=0.
        second['peers']={'a':np.array([31.,30.,15.])}
        self.assertIsNone(filter_joint({'a':request,'b':second})[0])

    def test_impossible_joint_constraints_remain_explicit(self):
        request=dict(position=np.array([30.,30.,15.]),nominal=np.zeros(3),
            peers={},dt=1.,required_links=('__ground__',),ground=np.array([500.,500.,0.]),
            ground_max_speed=1.,world_bounds=(0.,0.,100.,100.),
            max_speed=6.,min_separation=15.,comm_range=95.,margin=.25,sample_time=0.)
        commands,diagnostic=filter_joint({'a':request})
        self.assertIsNone(commands)
        self.assertTrue(diagnostic['infeasible'])

    def test_experiment_applies_and_records_joint_recovery_before_physics(self):
        import torch
        from mappo.relay_attention_mappo import RelayActorCritic
        from mappo.mappo_experiment import Experiment
        case=json.loads((Path(__file__).parent/'fixtures/wildscenes_cbf_states.json').read_text())[0]
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode():
            env=Experiment(19,RelayActorCritic().eval(),communication_mode='backbone',
                safety_filter='cbf',coordinated_safety=True,moving_ground=True,
                world_size=case['world_size_m'],start=case['ground'][:2],
                scene=([1000.,400.],[],1))
            self.addCleanup(env.env.close)
            for robot,controller,position in zip(env.env.drones,env.controllers,case['positions']):
                robot.position=np.array(position)
                robot.set_broadcast(dict(drone_id=controller.drone_id))
            env.obs=env.env._get_observations()
            env.step(0)
        self.assertEqual(len(env.trace),1)
        self.assertEqual(len(env.coordinated_safety_events),1)
        self.assertTrue(all(c.safety_stats.get('coordinated_recovery') for c in env.controllers))
        self.assertTrue(all(c.safety_stats['max_constraint_residual']<=1e-7 for c in env.controllers))
        self.assertTrue(any(c.safety_stats['independent_cbf_infeasible'] for c in env.controllers))
        np.testing.assert_allclose(env.last_executed_velocities,
            [robot.velocity for robot in env.env.drones])


if __name__ == '__main__':
    unittest.main()
