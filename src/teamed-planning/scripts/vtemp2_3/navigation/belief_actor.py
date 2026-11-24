"""Frozen actor prior reconstructed solely from one receiver's evidence.

Remote sensor pixels are unknown unless covered by this receiver's own tile.
Receiver graph/telemetry are proxies for the remote private view, not claims
about what that drone truly knows. The learned residual handles that mismatch.
"""
from __future__ import annotations
import numpy as np


def recentered_sensor(sensor, target):
    tile=np.asarray(sensor['cells']);r=float(sensor['resolution'])
    center=np.asarray(sensor['center'][:2]);target=np.asarray(target[:2])
    yy,xx=np.indices((12,12))
    world=target+np.stack([xx+.5-6,yy+.5-6],-1)*2.
    xy=np.floor((world-center)/r+np.array([tile.shape[1],tile.shape[0]])/2).astype(int)
    valid=(xy[...,0]>=0)&(xy[...,0]<tile.shape[1])&(xy[...,1]>=0)&(xy[...,1]<tile.shape[0])
    result=np.full((12,12),-1.,np.float32)
    result[valid]=tile[xy[...,1][valid],xy[...,0][valid]]
    return result


class ActorPrior:
    def __init__(self, model, world_size, horizon=4):
        if horizon<1:raise ValueError('Positive horizon required')
        import torch
        from mappo.cooperative_mappo import CooperativeNavigator
        from shared.drone.local_backbone import relative_neighborhood
        self.model=model;self.torch=torch;self.Navigator=CooperativeNavigator
        self.backbone=relative_neighborhood;self.world_size=world_size;self.horizon=horizon

    def predict(self, own, last, rows, now, sensor, graph):
        """No environment/target controller argument: those cannot leak here."""
        target=last['source'];age=now-last['stamp']
        origin=np.asarray(last['position'][:2],float)
        position=np.asarray(last['position'],float)+age*np.asarray(last['velocity'],float)
        position[:2]=np.clip(position[:2],0.,self.world_size);position[2]=15.
        velocity=np.asarray(last['velocity'],float)
        navigator=self.Navigator(target,model=self.model,deterministic=True,total_drones=3,
            comm_range=95.,min_separation=15.,max_speed=6.,
            world_bounds=(0.,0.,self.world_size,self.world_size))
        navigator.previous_position=position-velocity;navigator.previous_time=now-1.
        navigator.nominal_velocity=np.asarray(last['nominal'],float)
        # The observer cannot know the remote CBF margin. This neutral proxy is
        # explicit in the experiment contract; never query the remote filter.
        navigator.cbf_margin_source=lambda:0.
        states=[np.r_[(position[:2]-origin)/20.,velocity[:2]/6.]];actions=[]
        known=float(np.mean(recentered_sensor(sensor,position)>=0))
        available={**rows,own['source']:own}
        for step in range(self.horizon):
            moment=now+step;peers={};velocities={};ground=None;relay=[]
            for key,row in available.items():
                if key==target:continue
                elapsed=moment-row['stamp']
                if elapsed<0 or elapsed>30.:continue
                q=np.asarray(row['position'])+elapsed*np.asarray(row['velocity'])
                if row['is_ground']:ground=q
                elif np.linalg.norm(q-position)<=95.:
                    peers[key]=q;velocities[key]=np.asarray(row['velocity'])-velocity
                relay.append(dict(origin_id=key,payload_type='vehicle_state' if row['is_ground'] else 'drone_state',
                    position=q,velocity=np.asarray(row['velocity']),goal=row['goal'],age=elapsed,
                    hop=row['hop'],sigma=.5+elapsed,time_unknown=False))
            visible=dict(peers)
            if ground is not None and np.linalg.norm(ground-position)<=95.:visible['__ground__']=ground
            task={}
            if last['goal'] is not None and last['task_ground'] is not None:
                task['ground_0']=dict(position=last['task_ground'],target=last['goal'],planned_path=[],
                    source_time=last['goal_stamp'],direct='__ground__' in visible)
            observation=dict(time=moment,graph=graph,ground_robots=task,
                nearby_drones={key:dict(position=value) for key,value in peers.items()},
                occupancy_map=recentered_sensor(sensor,position),sensor_valid=True,
                cooperation=dict(peers=peers,velocities=velocities,ground=visible.get('__ground__'),
                                 required=self.backbone(position,visible,95.)),relay=relay)
            features,_=navigator.encode(position,observation)
            with self.torch.inference_mode():
                distribution=self.model.distribution(self.torch.from_numpy(features))
                raw=distribution.mean.numpy();std=distribution.stddev.numpy()
            action=np.r_[self.model.velocity(raw,6.),0.]
            navigator.previous_position=position.copy();navigator.previous_time=moment
            navigator.nominal_velocity=action.copy()
            position=position+action;position[:2]=np.clip(position[:2],0.,self.world_size)
            velocity=action
            actions.append(action[:2]/6.)
            states.append(np.r_[(position[:2]-origin)/20.,velocity[:2]/6.])
        return np.asarray(states,np.float32),np.asarray(actions,np.float32),known,np.asarray(std,np.float32)
