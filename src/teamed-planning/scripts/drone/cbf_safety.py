"""Decentralized sampled-data CBF velocity filter for the kinematic simulator.

Minimum-distance barriers apply to all received peers; upper-distance barriers
apply only to configured chain links (or ground in direct mode). For fresh,
common position samples and cooperating bounded-speed controllers, the radial
lower bound protects the entire linear step. The communication upper bound
includes a worst-case relative-motion curvature term for zero-order hold.

This is a simulation velocity filter, not a certification of PX4 dynamics or
delayed/missing measurements. Infeasibility and stale links are explicit.
"""
import numpy as np
from scipy.optimize import minimize


class CBFSafetyFilter:
    def __init__(self, min_separation=15., comm_range=95., max_speed=6.,
                 margin=.25, gain=1.):
        if min_separation<=0 or comm_range<=min_separation+2*margin or max_speed<=0 or gain<=0:
            raise ValueError('Invalid CBF distances, speed or gain')
        self.min_separation=float(min_separation)
        self.comm_range=float(comm_range)
        self.max_speed=float(max_speed)
        self.margin=float(margin)
        self.gain=float(gain)

    def filter(self, nominal, position, peers, dt, required_links=(), ground=None,
               world_bounds=None, drone_id='', ground_max_speed=0.):
        if dt<=0 or not np.isfinite(dt):
            raise ValueError('CBF requires a positive finite control step')
        if not np.isfinite(ground_max_speed) or ground_max_speed < 0:
            raise ValueError('Ground speed bound must be finite and nonnegative')
        p=np.asarray(position,float)
        nominal=np.asarray(nominal,float)
        if p.shape!=(3,) or nominal.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(nominal).all():
            raise ValueError('CBF requires finite 3D position and nominal velocity')
        if abs(nominal[2])>1e-8:
            raise ValueError('This CBF implementation requires fixed flight height')
        # The current simulator's waypoint selector keeps flight height fixed.
        # Lock z for this first implementation; real altitude tracking is separate.
        desired=nominal[:2].copy()
        speed=self.max_speed
        desired*=min(1.,speed/max(np.linalg.norm(desired),1e-12))
        rows=[];limits=[]
        minimum=np.inf
        required=set(required_links)
        def upper(normal,limit):
            rows.append(np.asarray(normal,float));limits.append(float(limit))
        for did,other in peers.items():
            r=p-np.asarray(other,float)
            distance=float(np.linalg.norm(r))
            minimum=min(minimum,distance)
            horizontal=float(np.linalg.norm(r[:2]))
            # Fixed z means the required horizontal clearance is derived from
            # the true received 3D relative geometry, rather than ignoring height.
            clearance=np.sqrt(max(0.,(self.min_separation+self.margin)**2-r[2]**2))
            if horizontal>1e-9:
                normal=r[:2]/horizontal
            else:
                normal=np.array([1. if str(drone_id)>str(did) else -1.,0.])
            if clearance>0:
                lower=min(self.gain,1.)*(clearance-horizontal)/(2*dt)
                upper(-normal,-lower)
            if did in required:
                radius=self.comm_range-self.margin
                h=radius**2-distance**2
                # Sum the two endpoint inequalities to obtain a joint CBF.
                bound=(min(self.gain,1.)*h/dt-4*speed**2*dt)/4
                upper(r[:2],bound)
        if '__ground__' in required and ground is not None:
            r=p-np.r_[np.asarray(ground,float)[:2],0.]
            h=(self.comm_range-self.margin)**2-float(np.dot(r,r))
            # Ground need not cooperate with the aerial CBF. Protect against
            # every ground displacement within its configured speed bound.
            upper(r[:2],(min(self.gain,1.)*h/dt-(speed+ground_max_speed)**2*dt)/2
                  - np.linalg.norm(r[:2])*ground_max_speed)
        if world_bounds is not None:
            xmin,ymin,xmax,ymax=world_bounds
            upper([-1,0],(p[0]-xmin)/dt);upper([1,0],(xmax-p[0])/dt)
            upper([0,-1],(p[1]-ymin)/dt);upper([0,1],(ymax-p[1])/dt)
        matrix=np.asarray(rows,float).reshape(-1,2)
        bound=np.asarray(limits,float)
        def feasible(v):
            return (np.isfinite(v).all() and np.linalg.norm(v)<=speed+1e-7 and
                    (not len(bound) or np.all(matrix@v<=bound+1e-7)))
        if feasible(desired):
            result=desired
        else:
            constraints=[dict(type='ineq',fun=lambda v:bound-matrix@v,
                              jac=lambda v:-matrix),
                         dict(type='ineq',fun=lambda v:speed**2-np.dot(v,v),jac=lambda v:-2*v)]
            solved=minimize(lambda v:.5*np.dot(v-desired,v-desired),np.zeros(2),
                jac=lambda v:v-desired,constraints=constraints,method='SLSQP',
                options=dict(ftol=1e-10,maxiter=100))
            result=solved.x if feasible(solved.x) else np.zeros(2)
        infeasible=not feasible(result)
        return np.r_[result,0.],dict(infeasible=infeasible,
            intervention=float(np.linalg.norm(result-desired)),
            min_received_separation=None if not np.isfinite(minimum) else minimum,
            max_constraint_residual=float(max(0.,np.max(matrix@result-bound))) if len(bound) else 0.,
            received_peers=len(peers),missing_links=sorted(required-set(peers)-({'__ground__'} if ground is not None else set())))
