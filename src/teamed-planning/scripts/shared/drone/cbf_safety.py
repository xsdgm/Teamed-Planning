"""Sampled-data CBF velocity filters for the kinematic simulator.

Minimum-distance barriers apply to all received peers; upper-distance barriers
apply only to configured chain links (or ground in direct mode). For fresh,
common position samples and cooperating bounded-speed controllers, the radial
lower bound protects the entire linear step. The communication upper bound
includes a worst-case relative-motion curvature term for zero-order hold.

This is a simulation velocity filter, not a certification of PX4 dynamics or
delayed/missing measurements. Infeasibility and stale links are explicit. The synchronous filter_joint
fallback below requires an additional coordination round.
"""
import numpy as np
from scipy.optimize import linprog, minimize


def feasibility_margin(matrix, bound, speed, sides=32):
    """Largest common slack (m/s) of the velocity constraints within the speed
    limit: max t subject to a_i.v/|a_i| + t <= b_i/|a_i|. Negative exactly when
    no admissible velocity exists; None without constraints. The disk is the
    inscribed regular polygon, so the margin is slightly conservative."""
    norms=np.linalg.norm(matrix,axis=1)
    live=norms>1e-12
    if np.any(~live & (bound<0)):
        return float('-inf')
    if not live.any():
        return None
    angles=np.arange(sides)*2*np.pi/sides
    polygon=np.c_[np.cos(angles),np.sin(angles)]
    rows=np.r_[np.c_[matrix[live]/norms[live,None],np.ones(live.sum())],np.c_[polygon,np.zeros(sides)]]
    limits=np.r_[bound[live]/norms[live],np.full(sides,speed*np.cos(np.pi/sides))]
    result=linprog([0.,0.,-1.],A_ub=rows,b_ub=limits,bounds=[(None,None)]*3,method='highs')
    if result.status!=0:
        raise RuntimeError(f'CBF margin program failed: {result.message}')
    return float(-result.fun)


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
               world_bounds=None, drone_id='', ground_max_speed=0., grounds=None):
        """grounds: optional {vehicle_id: position}; each required vehicle link
        gets the same non-cooperative constraint as the single '__ground__'."""
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
        link_distances={};link_outward_speeds={}
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
                link_distances[did]=distance
                link_outward_speeds[did]=float(np.dot(r[:2],desired)/max(distance,1e-12))
                radius=self.comm_range-self.margin
                h=radius**2-distance**2
                # Sum the two endpoint inequalities to obtain a joint CBF.
                bound=(min(self.gain,1.)*h/dt-4*speed**2*dt)/4
                upper(r[:2],bound)
        vehicles={'__ground__':ground} if '__ground__' in required and ground is not None else {}
        vehicles.update({k:q for k,q in (grounds or {}).items() if k in required})
        for key,vehicle in vehicles.items():
            r=p-np.r_[np.asarray(vehicle,float)[:2],0.]
            distance=float(np.linalg.norm(r))
            link_distances[key]=distance
            link_outward_speeds[key]=float(np.dot(r[:2],desired)/max(distance,1e-12))
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
            feasibility_margin=feasibility_margin(matrix,bound,speed),
            intervention=float(np.linalg.norm(result-desired)),
            min_received_separation=None if not np.isfinite(minimum) else minimum,
            max_constraint_residual=float(max(0.,np.max(matrix@result-bound))) if len(bound) else 0.,
            received_peers=len(peers),
            required_link_distances_m=link_distances,
            required_link_nominal_outward_m_s=link_outward_speeds,
            missing_links=sorted(required-set(peers)-({'__ground__'} if ground is not None else set())
                                 -set(grounds or ())))


# Synchronous simulation fallback when independent CBF programs conflict.
#
# The coordinator uses this frame's explicit safety requests, not world truth or
# terrain. All participants must agree on fresh direct-neighbour geometry and
# apply the returned velocities together. This is an additional coordination
# round, NOT a decentralized controller for delayed or missing messages.

def filter_joint(requests):
    """Return (commands, diagnostics); commands is None on invalid/infeasible input.

    Pair separation uses a radial lower bound throughout the linear step.
    Radio constraints bound the norm of the actual joint displacement. Ground
    links reserve every horizontal ground motion within its declared speed
    bound. Convexity protects the interior of the step as well as its endpoint.
    """
    def reject(reason):
        return None, dict(infeasible=True, reason=reason)

    if not requests:
        return reject('no_participants')
    names=sorted(requests)
    indices={name:i for i,name in enumerate(names)}
    first=requests[names[0]]
    dt=float(first['dt'])
    time=first['sample_time']
    if not np.isfinite(dt) or dt<=0 or time is None or not np.isfinite(time):
        return reject('invalid_sample_time')
    for name,r in requests.items():
        if r['sample_time']!=time or r['dt']!=dt:
            return reject('samples_not_synchronous')
        p=np.asarray(r['position'],float)
        nominal=np.asarray(r['nominal'],float)
        if (p.shape!=(3,) or nominal.shape!=(3,) or not np.isfinite(p).all()
                or not np.isfinite(nominal).all() or abs(nominal[2])>1e-8):
            return reject('invalid_fixed_height_state')
        for peer,q in r['peers'].items():
            if peer not in requests or name not in requests[peer]['peers']:
                return reject('missing_reciprocal_participant')
            if (not np.allclose(q,requests[peer]['position'],atol=1e-7,rtol=0.)
                    or not np.allclose(requests[peer]['peers'][name],p,atol=1e-7,rtol=0.)):
                return reject('inconsistent_received_geometry')
        vehicles=dict(r.get('grounds') or {})
        if r.get('ground') is not None:
            vehicles['__ground__']=r['ground']
        if set(r['required_links'])-set(r['peers'])-set(vehicles):
            return reject('missing_required_link')

    positions=np.asarray([requests[n]['position'] for n in names],float)
    desired=np.asarray([requests[n]['nominal'][:2] for n in names],float)
    speeds=np.asarray([requests[n]['max_speed'] for n in names],float)
    if not np.isfinite(speeds).all() or np.any(speeds<=0):
        return reject('invalid_speed_bound')
    desired*=np.minimum(1.,speeds/np.maximum(np.linalg.norm(desired,axis=1),1e-12))[:,None]
    pairs=[];links=[];vehicles=[];bounds=[]
    for name in names:
        i=indices[name];r=requests[name]
        for other in r['peers']:
            j=indices[other]
            if j<=i:
                continue
            partner=requests[other];delta=positions[i]-positions[j]
            horizontal=np.linalg.norm(delta[:2])
            minimum=max(r['min_separation']+r['margin'],partner['min_separation']+partner['margin'])
            clearance=np.sqrt(max(0.,minimum**2-delta[2]**2))
            normal=delta[:2]/horizontal if horizontal>1e-9 else np.array([-1.,0.])
            if clearance>0:
                pairs.append((i,j,normal,(clearance-horizontal)/dt))
            if other in r['required_links'] or name in partner['required_links']:
                radius=min(r['comm_range']-r['margin'],partner['comm_range']-partner['margin'])
                squared=radius**2-delta[2]**2
                if squared<0:
                    return reject('vertical_link_out_of_range')
                links.append((i,j,delta[:2],np.sqrt(squared)))
        seen=dict(r.get('grounds') or {})
        if r.get('ground') is not None:
            seen['__ground__']=r['ground']
        for key,p in seen.items():
            if key not in r['required_links']:
                continue
            ground_speed=float(r['ground_max_speed'])
            if not np.isfinite(ground_speed) or ground_speed<0:
                return reject('invalid_ground_speed_bound')
            radius=r['comm_range']-r['margin']
            squared=radius**2-positions[i,2]**2
            if squared<0:
                return reject('vertical_ground_link_out_of_range')
            remaining=np.sqrt(squared)-ground_speed*dt
            if remaining<0:
                return reject('ground_displacement_exceeds_radio_radius')
            vehicles.append((i,positions[i,:2]-np.asarray(p)[:2],remaining))
        if r.get('world_bounds') is not None:
            bounds.append((i,r['world_bounds']))

    def slack(flat):
        velocity=flat.reshape(-1,2)
        values=list(speeds-np.linalg.norm(velocity,axis=1))
        values.extend(np.dot(normal,velocity[i]-velocity[j])-lower for i,j,normal,lower in pairs)
        values.extend((radius-np.linalg.norm(delta+dt*(velocity[i]-velocity[j])))/dt
                      for i,j,delta,radius in links)
        values.extend((radius-np.linalg.norm(delta+dt*velocity[i]))/dt
                      for i,delta,radius in vehicles)
        for i,(xmin,ymin,xmax,ymax) in bounds:
            endpoint=positions[i,:2]+dt*velocity[i]
            values.extend((endpoint-np.array([xmin,ymin]))/dt)
            values.extend((np.array([xmax,ymax])-endpoint)/dt)
        return np.asarray(values)

    desired=desired.ravel()
    solved=minimize(lambda x:.5*np.dot(x-desired,x-desired),np.zeros_like(desired),
        jac=lambda x:x-desired,constraints=[dict(type='ineq',fun=slack)],
        method='SLSQP',options=dict(ftol=1e-9,maxiter=200))
    residual=slack(solved.x)
    if not np.isfinite(solved.x).all() or not np.isfinite(residual).all() or residual.min() < -1e-7:
        return reject('joint_constraints_infeasible_or_solver_failed')
    # Solver status alone never authorizes an action: every returned command
    # passes all original physical constraints above, including ground motion.
    commands={n:np.r_[solved.x.reshape(-1,2)[i],0.] for i,n in enumerate(names)}
    return commands,dict(infeasible=False,reason='joint_safe_step',sample_time=time,
        participants=names,min_slack_m_s=float(residual.min()),
        max_constraint_residual=float(max(0.,-residual.min())),solver_success=bool(solved.success))
