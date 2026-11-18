"""Regression checks for paired evaluation and independent safety auditing."""
import sys
from pathlib import Path
import unittest
import importlib.util
import json
import os
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_controlled_support import (action_stream, coupled_relay_class, fleet_positions,
    minimum_constraint_speed, swept_clearance, loop_statistics)

ARCHIVE = Path(__file__).resolve().parents[3]/'artifacts/v9-fleet4-wait30-20261111/runtime_sources'
sys.path.insert(0, str(ARCHIVE))
from simulator.relay_channel import RelayChannel
from simulator.env import EnvConfig, MultiRobotEnv
from validate_v9_fleet_controlled import live_status


class FleetControlsTests(unittest.TestCase):
    def test_dead_or_reused_process_is_not_reported_as_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'data'/'case';out.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(status='running',supervisor_pid=os.getpid(),
                supervisor_started='not_this_process',runs=[dict(key='case',status='running')])))
            (out/'process.json').write_text(json.dumps(dict(pid=os.getpid(),started='not_this_process')))
            state=live_status(root)
            self.assertEqual(state['status'],'interrupted')
            self.assertEqual(state['runs'][0]['status'],'interrupted')
            self.assertFalse(state['runs'][0]['process_alive'])
            (out/'result.json').write_text(json.dumps(dict(status='complete',termination='vehicle_arrival')))
            (root/'manifest.json').write_text(json.dumps(dict(jobs=[dict(key='case',runtime='repaired',fleet=4)])))
            state=live_status(root)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['completed'],1)
            self.assertEqual(state['runs'][0]['runtime'],'repaired')

    def test_ground_sensor_overlay_preserves_physics_and_all_uav_sensors(self):
        path=ARCHIVE.parents[1]/'v9-navigation-repair-20261111-r22-multirun/runtime_sources/simulator/env.py'
        spec=importlib.util.spec_from_file_location('simulator.controlled_env_check',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        self.addCleanup(sys.modules.pop,spec.name,None)
        spec.loader.exec_module(module)
        options=dict(world_size=(200.,200.),dt=1.,perception_radius=24.,
            communication_radius=95.,occupancy_map_resolution=2.,occupancy_map_size=12,
            ground_occupancy_map_size=16,ground_occupancy_map_resolution=.5)
        old,new=MultiRobotEnv(EnvConfig(**options)),module.MultiRobotEnv(module.EnvConfig(**options))
        for env in (old,new):
            self.addCleanup(env.close)
            env.add_ground_robot(50.,50.,max_speed=1.)
            for p in fleet_positions((50.,50.),4):
                env.add_drone(*p,15.,max_speed=6.)
            env.add_circle_obstacle(54.,52.,1.2)
            env.add_circle_obstacle(65.,48.,3.)
        a,b=old.reset(),new.reset()
        for _ in range(12):
            self.assertEqual(set(a),set(b))
            for rid in a:
                np.testing.assert_array_equal(a[rid]['self']['position'],b[rid]['self']['position'])
                np.testing.assert_array_equal(a[rid]['occupancy_map'],b[rid]['occupancy_map'])
                self.assertEqual(a[rid]['map_resolution'],b[rid]['map_resolution'])
                self.assertEqual(len(a[rid]['neighbors']),len(b[rid]['neighbors']))
                if a[rid]['self']['type']=='drone':
                    self.assertNotIn('navigation_map',b[rid])
                else:
                    self.assertEqual(b[rid]['navigation_map'].shape,(80,80))
                    self.assertEqual(b[rid]['navigation_resolution'],.1)
            actions={old.ground_robots[0].robot_id:dict(motion=(.3,.1))}
            actions.update({d.robot_id:dict(motion=(.1,.2,0.)) for d in old.drones})
            a,info_a=old.step(actions);b,info_b=new.step(actions)
            self.assertEqual(info_a,info_b)

    def test_right_half_layout_keeps_all_three_existing_aircraft(self):
        start = (742.1731339121583, 373.7332018636892)
        three, four = fleet_positions(start,3), fleet_positions(start,4)
        np.testing.assert_array_equal(three, four[:3])
        np.testing.assert_allclose(three-np.array(start), [(9,-18),(9,0),(9,18)])
        distances=np.linalg.norm(four[:,None]-four[None,:],axis=2)+np.eye(4)*1000
        self.assertGreaterEqual(distances.min(),15.25)

    def test_out_of_bounds_does_not_silently_reposition_the_baseline(self):
        with self.assertRaises(ValueError):
            fleet_positions((990.,500.),4)

    def test_added_agent_and_call_order_do_not_shift_existing_action_noise(self):
        def draw(order):
            values={}
            for frame in range(5):
                for agent in order:
                    with action_stream(919251328,agent,frame):
                        values[agent,frame]=torch.randn(2)
            return values
        three=draw(['drone_0','drone_1','drone_2'])
        four=draw(['drone_3','drone_2','drone_0','drone_1'])
        for key in three:
            torch.testing.assert_close(three[key],four[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(three['drone_0',0],three['drone_1',0]))
        self.assertFalse(torch.equal(three['drone_0',0],three['drone_0',1]))

    def test_action_stream_restores_outer_rng_on_error(self):
        before=torch.random.get_rng_state()
        with self.assertRaises(RuntimeError),action_stream(53,'drone_0',0):
            torch.randn(100)
            raise RuntimeError('test')
        torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)

    def test_relay_extra_senders_do_not_shift_common_loss_delay_or_order(self):
        cls=coupled_relay_class(RelayChannel)
        a,b=cls(53,.1,2,1),cls(53,.1,2,1)
        packet=[dict(state=dict(x=1))]
        for frame in range(50):
            self.assertTrue(a.due('drone_0',frame))
            a.transmit(packet,[1,2],frame)
            self.assertTrue(b.due('drone_3',frame))
            b.transmit(packet,[1,2],frame)
            self.assertTrue(b.due('drone_0',frame))
            b.transmit(packet,[3,2,1],frame)
        for receiver in (1,2):
            common=[p for p in b.queues[receiver] if p[1][1]=='drone_0']
            self.assertEqual(a.queues[receiver],common)

    def test_relay_preserves_one_step_delay_and_packet_copy(self):
        channel=coupled_relay_class(RelayChannel)(53,0.,0,1)
        packet=[dict(state=dict(x=1))]
        channel.due('drone_0',0);channel.transmit(packet,[1],0)
        packet[0]['state']['x']=9
        self.assertEqual(channel.deliver(1,0),[])
        self.assertEqual(channel.deliver(1,1),[[dict(state=dict(x=1))]])
        self.assertEqual(channel.deliver(1,2),[])

    def test_swept_audit_catches_tunneling_even_when_both_endpoints_are_clear(self):
        circles=[[0.,0.,1.]]
        self.assertGreater(swept_clearance([-2,0],[-2,0],circles),.3)
        self.assertGreater(swept_clearance([2,0],[2,0],circles),.3)
        self.assertLess(swept_clearance([-2,0],[2,0],circles),0.)
        self.assertAlmostEqual(swept_clearance([-2,1.5],[2,1.5],circles),.5)

    def test_constraint_audit_distinguishes_speed_limit_and_inconsistent_halfplanes(self):
        self.assertAlmostEqual(minimum_constraint_speed([[-1,0],[0,-1]],[-3,-4]),5.)
        self.assertAlmostEqual(minimum_constraint_speed([[-1,0]],[ -7]),7.)
        self.assertIsNone(minimum_constraint_speed([[1,0],[-1,0]],[-1,-1]))

    def test_loop_audit_detects_motion_but_not_stationary_waiting(self):
        path=[dict(positions=[[np.cos(i*2*np.pi/30)*3,np.sin(i*2*np.pi/30)*3]]) for i in range(180)]
        self.assertTrue(loop_statistics(path)['repeating'])
        self.assertIn(30,loop_statistics(path)['periods'])
        stationary=[dict(positions=[[0.,0.]]) for _ in range(180)]
        self.assertFalse(loop_statistics(stationary)['repeating'])
        forward=[dict(positions=[[i*.5,0.]]) for i in range(180)]
        self.assertFalse(loop_statistics(forward)['repeating'])


if __name__=='__main__':
    unittest.main()
