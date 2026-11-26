import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS/'vtemp5'))
from v91_rejoin_guidance import RejoinGuidance, _can_rejoin, _local_sensor, _narrow_arcs


class MidArcGuidance(RejoinGuidance):
    """Keep state, sensing and candidate code live; control the sampled PPO angle."""
    def _actor(self, env, snapshot, actor, yaw, arcs, candidates):
        choice=max(arcs,key=lambda a:a[1]-a[0])
        return float(np.mean(choice))

    def _ring(self, snapshot, car, yaw):
        return np.array([[-math.pi+.00001,math.pi-.00001]])


def mock_env(car, geometry=None, yaw=-math.pi/2, valid=True):
    obs=dict(self=dict(position=car,orientation=yaw),sensor_valid=valid,map_resolution=.5,
             occupancy_map=np.zeros((16,16)))
    if geometry is not None:
        obs['ground_safety_map']=geometry
    return SimpleNamespace(ground='ugv',obs={'ugv':obs})


def snapshot(car, goal, seed=42, frame=600):
    return dict(frame=frame,car_xy=car,goal_xy=goal,fleet_seed=seed,presence_threshold=.7,
                mask_threshold=.5,air_cells=np.zeros((720,720),np.int8),resolution=2.)


def fields(car, goal, blocked=False, walls=False, probability=True):
    f=(goal-car)/np.linalg.norm(goal-car)
    right=np.array([f[1],-f[0]])
    row,col=np.indices((48,48))
    along=-24+(48-row-.5)*2
    across=(col+.5-24)*2
    xy=car+along[:,:,None]*f+across[:,:,None]*right
    occupied = ((along>4)&(along<32)&(np.abs(across)<8)) if blocked else np.zeros((48,48),bool)
    if walls: occupied |= (np.abs(across)>=4) & (np.abs(along)<42)
    x=np.stack((~occupied,occupied))
    return x, xy, (float(occupied.any()) if probability else None), (occupied.astype(float) if probability else None)


class TestRejoinGuidance(unittest.TestCase):
    def test_central_bridge_deck_and_open_start(self):
        car=np.array([724.,1100.]); goal=car+[0,-950.]
        g=MidArcGuidance()
        d=g.review(mock_env(car),snapshot(car,goal),fields(car,goal,walls=True,probability=False),None,'observed')
        self.assertEqual(d['mode'],'open_policy')
        self.assertFalse(d['direct_blocked'])
        self.assertEqual(d['candidate_count'],0)
        self.assertIsNone(d['survey_target_xy'])
        self.assertAlmostEqual(np.linalg.norm(car-d['target_xy']),200.)

    def test_wait_and_retry_without_issuing_drive_into_water(self):
        car=np.array([724.,1100.]); goal=car+[0,-950.]
        g=MidArcGuidance()
        directions=[]
        for frame in (600,630,660,690,720,750):
            d=g.review(mock_env(car,valid=False),snapshot(car,goal,frame=frame),
                       fields(car,goal,blocked=True,probability=False),None,'observed')
            self.assertEqual(d['mode'],'wait_for_survey')
            np.testing.assert_allclose(d['target_xy'],car)
            self.assertFalse(d['actor_called'])
            directions.append(d['survey_target_xy'])
        self.assertFalse(np.allclose(directions[0],directions[-1]))
        d=g.review(mock_env(car),snapshot(car,goal,frame=780),fields(car,goal,blocked=True),None,'predicted')
        self.assertIn(d['mode'],('follow_observed_branch','local_bank_recovery'))

    def test_rejoin_known_prefix_with_independent_extent_and_presence(self):
        unseen=np.ones((2,2),bool)
        extent=np.full((2,2),.1)
        self.assertTrue(_can_rejoin(False,2.,.7,unseen,'predicted',.1,extent,.7,.5))
        self.assertTrue(_can_rejoin(False,2.,.55,unseen,'predicted',.85,extent,.7,.5))
        self.assertFalse(_can_rejoin(False,0.,1.,unseen,'predicted',.85,extent,.7,.5))
        self.assertFalse(_can_rejoin(False,2.,.7,unseen,'observed',None,None,.7,.5))
        extent[0,1]=.85
        self.assertFalse(_can_rejoin(False,2.,.7,unseen,'predicted',.85,extent,.7,.5))
        self.assertFalse(_can_rejoin(False,2.,.9,unseen,'predicted',.85,extent,.7,.5))

    def test_arcs_cross_body_frame_wraparound(self):
        arcs=_narrow_arcs(np.array([[-math.pi,math.pi]]),math.radians(179),math.radians(6))
        self.assertEqual(len(arcs),2)
        self.assertAlmostEqual(np.diff(arcs).sum(),math.radians(12))

    def test_persistent_stall_unlocks_side(self):
        car=np.array([0.,0.]);g=MidArcGuidance()
        options={s:dict(side=s,angle=s*1.5,score=.7 if s>0 else .3) for s in (1,-1)}
        for now in (0,30,60):
            self.assertEqual(g._decide_side(options,car,now,0.,1,{})['side'],1)
        self.assertEqual(g._decide_side(options,car,90,0.,1,{})['side'],-1)
        self.assertEqual(g._decide_side(options,car,120,0.,1,{})['side'],-1)

    def test_local_refresh_waits_until_near_and_has_a_twenty_metre_limit(self):
        g=MidArcGuidance()
        decision=dict(origin_xy=[100.,100.],course_world_rad=0.)
        car=np.array([100.,100.])
        env=mock_env(car)
        self.assertIsNone(g.refresh_target(env,decision,car,car+[2.,0.]))
        next_target=g.refresh_target(env,decision,car,car+[1.,0.])
        self.assertGreater(next_target[0],car[0]+1.)
        self.assertLessEqual(np.linalg.norm(next_target-car),2.5)
        far=car+[19.,0.]
        self.assertIsNone(g.refresh_target(mock_env(far),decision,far,far))

    def test_original_stopped_sensor_and_mirror(self):
        root=SCRIPTS.parents[2]/'artifacts/v91-frozen-km-paired-20261126'
        sample=root/'diagnostics/km_prediction_samples/lake-dev-00-detour_left.npz'
        if not sample.is_file():
            self.skipTest('Archived kilometre sensor probe unavailable')
        from shared.simulator.wildscenes.wildscenes_map import RasterForestMap
        from vtemp2_3.wildscenes_runtime import local_safety_observation
        case=next(r for r in json.loads((root/'protocol.json').read_text())['cases']
                  if r['case_id']=='lake-dev-00-detour_left')
        # Use original map only to *generate the local sensor* for the old
        # parking pose. Policy code receives only this tile and delivered fields.
        forest=RasterForestMap.from_npz(case['case']['map_path'])
        with np.load(sample) as data:
            i=data['frames'].tolist().index(600)
            car=data['car_xy'][i]; xy=data['world_xy'][i]; goal=np.array(case['goal'])
            x=np.stack([data['observed_free'][i],data['observed_occupied'][i]])
            pres=float(data['presence_probability'][i]);extent=data['extent_probability'][i]
        sensor=local_safety_observation(forest,car)
        original=MidArcGuidance().review(mock_env(car,sensor),snapshot(car,goal),(x,xy,pres,extent),None,'predicted')
        self.assertEqual(original['mode'],'local_bank_recovery')
        self.assertGreaterEqual(original['local_safe_prefix_m'],1.5)
        self.assertGreater(original['target_xy'][0],car[0]+1.)
        self.assertTrue(_local_sensor(mock_env(car,sensor).obs['ugv']).segment_clear(
            car,np.array(original['target_xy']),1.5))
        reflected=lambda a: np.asarray(a)*[-1,1]+[1440,0]
        reflected_tile=dict(sensor)
        reflected_tile['occupancy_map']=np.fliplr(sensor['occupancy_map'])
        reflected_tile['known_mask']=np.fliplr(sensor['known_mask'])
        reflected_tile['origin_xy']=reflected(sensor['origin_xy'])
        reflected_tile['origin_xy'][0]-=sensor['occupancy_map'].shape[1]*sensor['resolution_m']
        mirror=MidArcGuidance().review(mock_env(reflected(car),reflected_tile),
            snapshot(reflected(car),reflected(goal)),(x,reflected(xy),pres,extent),None,'predicted')
        self.assertEqual(original['locked_side'],-mirror['locked_side'])
        np.testing.assert_allclose(reflected(original['target_xy']),mirror['target_xy'],atol=1e-5)
        self.assertNotEqual(original['survey_target_xy'],original['target_xy'])
