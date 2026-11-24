"""Causal experimental telemetry: original timestamps survive every hop.

This is an explicit additional, bounded telemetry channel. It is not simulator
truth exposed as an observation, nor the archived ROS transport implementation.
The simulator supplies physical send links; predictors only see delivered rows.
"""
from __future__ import annotations

from collections import deque
import copy
import numpy as np

SCHEMA = 'vtemp2_received_robot_telemetry_v1'
HISTORY = 8
HISTORY_FEATURES = 14
CONTEXT_FEATURES = 20
POSITION_SCALE = 20.
SPEED_SCALE = 6.


def packet(source, stamp, position, velocity, nominal, *, goal=None,
           goal_stamp=None, task_ground=None, is_ground=False):
    """Called at the source using its own state and its own received task."""
    def vector(value, width):
        result=np.asarray(value,float)[:width]
        if result.shape!=(width,) or not np.isfinite(result).all():
            raise ValueError('Invalid telemetry vector')
        return result.tolist()
    if not np.isfinite(stamp) or stamp<0:raise ValueError('Invalid source time')
    if goal is not None and (goal_stamp is None or not 0<=goal_stamp<=stamp):
        raise ValueError('Task evidence needs its original, nonfuture timestamp')
    return dict(source=str(source),stamp=float(stamp),sequence=int(round(stamp)),
                position=vector(position,3),velocity=vector(velocity,3),nominal=vector(nominal,3),
                goal=None if goal is None else vector(goal,2),
                goal_stamp=None if goal is None else float(goal_stamp),
                task_ground=None if task_ground is None else vector(task_ground,2),
                is_ground=bool(is_ground),hop=0)


class ReceivedHistory:
    def __init__(self, owner, max_age=30.):
        self.owner=str(owner);self.max_age=float(max_age)
        self.histories={};self.latest={}

    def receive(self, rows, now):
        for incoming in rows:
            row=copy.deepcopy(incoming);source=row['source']
            if source==self.owner:continue
            age=now-row['stamp']
            if age<0 or age>self.max_age:continue
            previous=self.latest.get(source)
            if previous is not None and row['sequence']<=previous['sequence']:
                continue
            row['received_at']=float(now)
            self.latest[source]=row
            self.histories.setdefault(source,deque(maxlen=HISTORY)).append(row)

    def rows(self, now):
        return {source:copy.deepcopy(row) for source,row in self.latest.items()
                if 0<=now-row['stamp']<=self.max_age}

    def encode(self, target, now):
        rows=self.rows(now)
        if target not in rows:raise KeyError('No received evidence for this target')
        last=rows[target];origin=np.asarray(last['position'][:2],float)
        x=np.zeros((HISTORY,HISTORY_FEATURES),np.float32);valid=np.zeros(HISTORY,bool)
        history=list(self.histories[target])
        for i,row in enumerate(history,HISTORY-len(history)):
            p=np.asarray(row['position'][:2]);goal=row['goal'];age=now-row['stamp']
            x[i]=np.r_[(p-origin)/95.,np.asarray(row['velocity'][:2])/6.,
                np.asarray(row['nominal'][:2])/6.,
                np.zeros(2) if goal is None else (np.asarray(goal)-p)/200.,
                goal is not None,age/30.,0. if goal is None else (now-row['goal_stamp'])/30.,
                row['hop']/8.,(last['stamp']-row['stamp'])/30.,row['is_ground']]
            valid[i]=True
        return x,valid,origin,last


class TelemetryNetwork:
    """One source packet plus cached evidence; no same-step multi-hop delivery."""
    def __init__(self, owners, seed, *, loss=.1,jitter=3,capacity=4,period=2):
        if not 0<=loss<1 or jitter<0 or capacity<1 or period<1:
            raise ValueError('Invalid telemetry transport configuration')
        self.receivers={str(owner):ReceivedHistory(owner) for owner in owners}
        self.rng=np.random.default_rng(seed);self.loss=loss;self.jitter=int(jitter)
        self.capacity=int(capacity);self.period=int(period);self.pending=[];self.last_frame=-1
        self.stats=dict(sent=0,lost=0,delivered=0)

    def advance(self, own_packets, links, frame):
        if frame<=self.last_frame:raise ValueError('Transport time must increase')
        self.last_frame=frame
        ready=[item for item in self.pending if item[0]<=frame]
        self.pending=[item for item in self.pending if item[0]>frame]
        for _,receiver,rows in sorted(ready,key=lambda item:item[0]):
            self.receivers[receiver].receive(rows,float(frame));self.stats['delivered']+=1
        if frame%self.period:return
        for source,own in own_packets.items():
            cached=sorted(self.receivers[source].rows(frame).values(),key=lambda r:-r['stamp'])
            envelope=[own]+cached[:self.capacity-1]
            for receiver in links.get(source,()):
                if receiver not in self.receivers:raise ValueError('Unknown physical receiver')
                self.stats['sent']+=1
                if self.rng.random()<self.loss:self.stats['lost']+=1;continue
                payload=copy.deepcopy(envelope)
                for row in payload:row['hop']+=1
                arrival=frame+1+int(self.rng.integers(self.jitter+1))
                self.pending.append((arrival,receiver,payload))


def context_features(own, last, rows, now, sensor_known, graph_nodes, actor_std):
    origin=np.asarray(last['position'][:2]);goal=own.get('goal');ack=last.get('goal')
    x=np.r_[(np.asarray(own['position'][:2])-origin)/95.,np.asarray(own['velocity'][:2])/6.,
        own['is_ground'],len(rows)/4.,sensor_known,np.log1p(graph_nodes)/10.,
        (now-last['stamp'])/30.,np.zeros(2) if goal is None else (np.asarray(goal)-origin)/200.,
        goal is not None,0. if goal is None else (now-own['goal_stamp'])/30.,
        np.zeros(2) if ack is None else (np.asarray(ack)-origin)/200.,ack is not None,
        0. if ack is None else (now-last['goal_stamp'])/30.,
        0. if goal is None or ack is None else np.linalg.norm(np.asarray(goal)-ack)/200.,
        actor_std]
    x=np.asarray(x,np.float32)
    if x.shape!=(CONTEXT_FEATURES,) or not np.isfinite(x).all():raise ValueError('Invalid causal context')
    return x
