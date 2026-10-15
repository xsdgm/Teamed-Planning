"""Fixed chain roles and reception-only recovery history (no fleet truth)."""
import numpy as np


class ChainState:
    def __init__(self, drone_id, total_drones, timeout=10., fleet_ids=None):
        self.drone_id=str(drone_id)
        self.fleet_ids=tuple(fleet_ids or ([self.drone_id] if total_drones==1 else
                                        [f'drone_{i}' for i in range(total_drones)]))
        if len(set(self.fleet_ids))!=len(self.fleet_ids) or self.drone_id not in self.fleet_ids:
            raise ValueError('Chain requires unique configured fleet IDs including this drone')
        self.rank=self.fleet_ids.index(self.drone_id)
        self.predecessor='__ground__' if self.rank==0 else self.fleet_ids[self.rank-1]
        self.successor=self.fleet_ids[self.rank+1] if self.rank+1<len(self.fleet_ids) else None
        self.required=tuple(k for k in (self.predecessor,self.successor) if k)
        self.timeout=float(timeout)
        self.history={}
        self.visible={}
        self.mission=None
        self.mission_time=None
        self.now=0.

    def update(self, observation):
        self.now=float(observation.get('time',0.))
        self.visible={}
        for did,info in observation.get('nearby_drones',{}).items():
            if info.get('position') is not None:
                self.visible[str(did)]=np.asarray(info['position'],float)
        for task in observation.get('ground_robots',{}).values():
            if task.get('position') is not None and task.get('target') is not None:
                source_time=float(task.get('source_time',self.now))
                if -.01 <= self.now-source_time <= self.timeout:
                    self.mission={k:task[k] for k in ('position','target','planned_path','source_time') if k in task}
                    self.mission_time=source_time
                    # Relay can deliver a ground position, but only rank 0 must
                    # keep a physical link to this endpoint. Distance is checked.
                    if task.get('direct',True):
                        self.visible['__ground__']=np.r_[task['position'][:2],0.]
                    break
        for did,position in self.visible.items():
            self.history[did]=(position.copy(),self.now)

    def task(self):
        if self.mission_time is not None and self.now-self.mission_time<=self.timeout:
            return self.mission
        return None

    def missing(self, position, comm_range):
        p=np.asarray(position,float)
        return tuple(did for did in self.required if did not in self.visible or
                     np.linalg.norm(self.visible[did]-p)>comm_range)

    def recovery(self,position,comm_range,min_separation=15.,max_speed=6.):
        """Approach a recent last-received missing endpoint; never invent one."""
        p=np.asarray(position,float)
        missing=self.missing(p,comm_range)
        targets=[self.history[did][0] for did in missing if did in self.history and
                 self.now-self.history[did][1]<=self.timeout and
                 np.linalg.norm(self.history[did][0]-p)>
                 min_separation+2*max_speed*(self.now-self.history[did][1]+1.)]
        if not targets:
            return tuple(p)
        target=np.mean(targets,axis=0)
        target[2]=p[2]
        return tuple(float(v) for v in target)
