"""Vehicle guidance from received occupancy, frozen predictions and the onboard sensor."""
import math
from types import SimpleNamespace

import numpy as np

from v91_frozen_actor_inputs import EpisodeActorInputs, _distance_to_segment
from v91_route_evidence import append_route_evidence, route_candidates


CLEARANCE = 1.5
RING = 200.
CELL_RADIUS = math.sqrt(2.)  # enclosing radius of an output cell
OFFSETS = (20, 35, 50, 65, 80, 90, 100, 115, 130, 145, 160, 175)


def _unit(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def _bearing(direction):
    return math.atan2(direction[1], direction[0])


def _wrap(angle):
    return math.remainder(angle, math.tau)


def _narrow_arcs(arcs, center, width):
    intervals = []
    for lo, hi in arcs:
        for middle in (center-math.tau, center, center+math.tau):
            a, b = max(lo, middle-width), min(hi, middle+width)
            if b-a > 1e-5:
                intervals.append((a,b))
    return np.asarray(sorted(intervals),float).reshape(-1,2)


def _local_sensor(obs):
    from shared.simulator.wildscenes import wildscenes_map
    from v91_rolling_clearance import local_geometry
    return local_geometry(obs, wildscenes_map)


def _local_prefix(local, car, direction):
    # Try longest first. The safety sensor's known footprint limits the length.
    for distance in (3., 2.5, 2., 1.5, 1., .5):
        if local.segment_clear(car, car+distance*direction, CLEARANCE):
            return distance
    return 0.


def _corridor(xy, car, direction, length):
    return _distance_to_segment(xy, car, car+length*direction) <= CLEARANCE+CELL_RADIUS


def _ray_option(side, angle, car, local, xy, free, occupied, prob, route, yaw):
    direction = _unit(angle)
    prefix = _local_prefix(local, car, direction)
    if prefix < 1.5:
        return None
    # The 2 m receiver may mark the starting footprint occupied even while the
    # native safety capsule beside a bank can move. Replace only the beginning
    # by the finer sensor certificate, inspect the entire remaining swept ray.
    delta = xy-car
    along = delta @ direction
    normal = np.abs(delta[...,0]*direction[1]-delta[...,1]*direction[0])
    strip = (along > prefix) & (along <= 26.) & (normal <= CLEARANCE+CELL_RADIUS)
    unknown = ~(occupied|free)
    # Extend the vehicle target only through continuously reported free cells.
    # All longer proposals remain on the separate survey channel.
    target_m = prefix
    for end in (4., 6., 8., 12., 16., 20., 24.):
        if end <= prefix:
            continue
        tail = _distance_to_segment(xy, car+prefix*direction, car+end*direction) <= CLEARANCE+CELL_RADIUS
        tail &= along > prefix
        # A complete end cap must remain inside the car-relative output grid.
        center = car+end*direction
        nearest=np.unravel_index(np.square(xy-center).sum(axis=-1).argmin(), free.shape)
        if (min(nearest) < 1 or nearest[0] >= free.shape[0]-1 or nearest[1] >= free.shape[1]-1
            or not tail.any() or not free[tail].all()):
            break
        target_m=end
    # A cloud forecast ranks unknown pixels only and cannot certify motion.
    unseen = strip & unknown
    risk = float(prob[unseen].mean()) if prob is not None and unseen.any() else .5
    unknown_rate=float(unseen.sum())/max(1,int(strip.sum()))
    obstruction=float((strip & occupied).sum())/max(1,int(strip.sum()))
    # Prefer a usable observed corridor, then forward progress / the selected
    # side proposal. Disfavour headings pointing away from the mission.
    if route is not None:
        towards = _bearing(route.points[0]-car)
        align = math.cos(_wrap(angle-towards))*route.confidence
    else:
        align = 0.
    score = (1.35*math.tanh(target_m/6.) - 1.7*obstruction - .8*risk*unknown_rate
             + .8*align + .12*math.cos(_wrap(angle-yaw)))
    return dict(side=side, angle=angle, target_m=target_m, prefix=prefix, risk=risk,
                score=score, reported_obstruction=obstruction)


def _side_options(car, mission, yaw, local, xy, free, occupied, prob, candidates):
    bearing=_bearing(mission)
    options={}
    for side in (1,-1):
        route=candidates.get(side)
        angles = [bearing+side*math.radians(k) for k in OFFSETS]
        if route is not None:
            proposal=_bearing(route.points[0]-car)
            if _wrap(proposal-bearing)*side>0:
                angles.append(proposal)
        scored=[]
        for a in angles:
            opt=_ray_option(side,a,car,local,xy,free,occupied,prob,route,yaw)
            if opt is not None:
                offset=abs(_wrap(a-bearing))
                # When a bank obstructs the forward tube, a tangential or
                # oblique heading can advance; straight backtracking cannot.
                opt['score'] += .8*math.cos(offset) + .32*math.sin(offset) - 2.5*max(0.,offset-math.pi/2)
                scored.append(opt)
        if scored:
            options[side]=max(scored,key=lambda o:o['score'])
    return options


def _can_rejoin(blocked, local_prefix, measured_ratio, unseen, arm, presence, extent, presence_cut, mask_cut):
    if blocked or local_prefix < 1.5:
        return False
    if arm == 'predicted' and np.any(extent[unseen] >= mask_cut):
        return False
    if measured_ratio >= .85:
        return True
    if arm != 'predicted' or not unseen.any():
        return False
    # On an active obstacle, a clear projected opening can supplement a
    # partially mapped corridor. On a low-presence frame require more measured
    # ground as a cross-check before leaving the selected shore.
    return measured_ratio >= (.5 if presence >= presence_cut else .65)


class RejoinGuidance(EpisodeActorInputs):
    def __init__(self):
        super().__init__()
        self.side=0
        self.stall_reviews=0
        self.stall_origin=None
        self.clear_reviews=0
        self.encounter_forward=None
        self.last_switch=-10**6
        self.wait_reviews=0

    def _decide_side(self, options, car, now, yaw, sensor_side, feedback):
        feedback_stall = self.stall_origin is not None and np.linalg.norm(car-self.stall_origin)<1.
        if feedback_stall:
            self.stall_reviews+=1
        else:
            self.stall_reviews=0
            self.stall_origin=car.copy()
        if (feedback.get('stalled_seconds',0.) >= 60. and feedback.get('veto_frames',0) > 0
                and feedback.get('interval_displacement_m',0.) < .5):
            self.stall_reviews=max(self.stall_reviews,2)
        if not options:
            return None
        if self.side and self.side in options:
            if self.stall_reviews<3 or -self.side not in options or now-self.last_switch<90:
                return options[self.side]
            self.side=-self.side
            self.last_switch=now
            self.stall_reviews=0
            self.stall_origin=car.copy()
        else:
            # Geometry / received risk comes first, yaw breaks metric ties. In
            # exact physical symmetry use the issuing episode seed, not a fixed side.
            chosen=max(options.values(),key=lambda o:(o['score'],math.cos(o['angle']-yaw),o['side']==sensor_side))
            if self.side != chosen['side']:
                self.last_switch=now
            self.side=chosen['side']
        return options[self.side]

    def _actor(self, env, snapshot, actor, yaw, arcs, candidates):
        from vtemp4.vtemp4_body_graph import body_frame_observation
        obs=self._base(env,snapshot,yaw,env.obs[env.ground]['self'])
        obs['angle_arcs']=arcs
        append_route_evidence(obs,candidates,snapshot['car_xy'],snapshot['goal_xy'],snapshot['frame'])
        return float(actor.act(body_frame_observation(obs),deterministic=True)[1])

    def _ring(self, snapshot, car, yaw):
        from vtemp4.vtemp4_periodic_history import legal_ring_arcs
        air,ground=snapshot['air_cells'],snapshot['ground_cells']
        fused=np.where((air>0)|(ground>0),1,np.minimum(air,ground)).astype(np.int8)
        if float(snapshot['resolution'])!=2.:
            raise ValueError('Expected 2 m receiver cells')
        grid=SimpleNamespace(cells=fused.repeat(2,0).repeat(2,1),resolution=1.,
                             ep=SimpleNamespace(case={'world_size':fused.shape[0]*2.}))
        return legal_ring_arcs(grid,car,yaw,RING)[0]

    def review(self, env, snapshot, fields, actor, arm):
        if arm not in ('predicted','observed'):
            raise ValueError('Expected a frozen observed or predicted arm')
        car=np.asarray(snapshot['car_xy'], float)
        goal=np.asarray(snapshot['goal_xy'], float)
        now=int(snapshot['frame'])
        own=env.obs[env.ground]['self']
        yaw=float(own['orientation'])
        x,xy,presence,extent=fields
        free,occupied=x[0].astype(bool),x[1].astype(bool)
        distance=np.linalg.norm(goal-car)
        mission=(goal-car)/max(distance,1.e-9)
        size=float(snapshot['air_cells'].shape[0]*snapshot['resolution'])
        local=_local_sensor(env.obs[env.ground])
        direct=_corridor(xy,car,mission,min(distance,48.))
        near=_corridor(xy,car,mission,min(distance,24.))
        station=(xy-car)@mission
        # Exclude the common rear cap: a bank immediately behind or alongside
        # the vehicle must not indefinitely re-trigger a goalward encounter.
        direct &= station>0
        near &= station>0
        local_direct = _local_prefix(local,car,mission)
        lateral=np.abs((xy-car)@np.array([-mission[1],mission[0]]))
        # Banks outside a clear central deck must not initiate a detour.
        blocked=bool(occupied[direct & (lateral<=CLEARANCE)].any()
                     or (local_direct<1.5 and occupied[direct].any()))
        prob=None
        if arm=='predicted':
            # Rank uncertain unseen cells by the extent head. A weak presence
            # score shrinks extent confidence towards the neutral unknown prior.
            weight=min(1.,float(presence)/snapshot['presence_threshold'])
            prob=.5 + weight*(extent-.5)
        if blocked and not self.encounter:
            self.encounter=True
            self.encounter_forward=mission.copy()
            self.stall_origin=car.copy()
            self.stall_reviews=0
        unseen=near & ~(free|occupied)
        known_ratio=float(free[near].mean()) if near.any() else 0.
        constrained=self.encounter
        if self.encounter:
            return_ok=_can_rejoin(blocked,local_direct,known_ratio,unseen,arm,
                                  presence,extent,snapshot['presence_threshold'],snapshot['mask_threshold'])
            self.clear_reviews=self.clear_reviews+1 if return_ok else 0
            if self.clear_reviews>=2:
                self.encounter=False
                self.side=0
                self.stall_origin=None
                self.stall_reviews=0
        rejoined=bool(not self.encounter and self.clear_reviews>=2)
        if rejoined:
            self.clear_reviews=0
        raw=None
        candidates={}
        prefix=None
        survey=None
        score=None
        course=None
        options={}
        if not self.encounter:
            self.wait_reviews=0
            if distance<=RING:
                mode='final_goal'
                target=goal.copy()
            else:
                mode='open_policy'
                arcs=self._ring(snapshot,car,yaw)
                raw=self._actor(env,snapshot,actor,yaw,arcs,{})
                target=car+RING*_unit(yaw+raw)
        else:
            candidates=route_candidates(xy,car,goal,free,occupied,world_size=size,probabilities=prob)
            options=_side_options(car,self.encounter_forward,yaw,local,xy,free,occupied,prob,candidates)
            seed_side=1 if int(snapshot['fleet_seed'])%2 else -1
            selected=self._decide_side(options,car,now,yaw,seed_side,snapshot.get('execution_feedback',{}))
            if selected is None:
                mode='wait_for_survey'
                self.wait_reviews+=1
                # Rotate requests if progress remains blocked. No rotation or
                # speculative translation is imposed on the ground vehicle.
                sweep_side=self.side or seed_side
                if ((self.wait_reviews-1)//3)%2: sweep_side=-sweep_side
                target=car.copy()
                survey=car+40.*_unit(_bearing(self.encounter_forward)+sweep_side*math.pi/2)
                prefix=0.
            else:
                self.wait_reviews=0
                side=selected['side']
                mode='follow_observed_branch' if side in candidates else 'local_bank_recovery'
                angle=selected['angle']
                prefix=selected['prefix']
                score=selected['score']
                # This constrained action directs a finite, observable vehicle
                # target; the aircraft may look beyond it. Validate the decoded
                # angle again because narrowing an arc is not a clearance proof.
                arcs=_narrow_arcs(self._ring(snapshot,car,yaw),_wrap(angle-yaw),math.radians(6.))
                if len(arcs):
                    raw=self._actor(env,snapshot,actor,yaw,arcs,candidates)
                    proposed=_ray_option(side,yaw+raw,car,local,xy,free,occupied,prob,candidates.get(side),yaw)
                    if proposed is not None and proposed['target_m']>=selected['target_m']-.25:
                        selected=proposed
                        angle=selected['angle']
                        prefix=selected['prefix']
                course=float(angle)
                target=car+selected['target_m']*_unit(angle)
                survey=car+min(58.,max(32.,distance))*_unit(angle)
        if survey is not None:
            survey=np.clip(survey,1.5,size-1.5)
        target=np.clip(target,0.,np.nextafter(size,0.))
        self.commands.append(dict(issued_at_s=now,origin_xy=car.copy(),target=target.copy()))
        angle=_wrap(_bearing(target-car)-yaw) if np.linalg.norm(target-car)>1e-6 else 0.
        return dict(origin_xy=car.tolist(),course_world_rad=course,target_xy=target.tolist(),survey_target_xy=None if survey is None else survey.tolist(),
                    source=mode,mode=mode,intent_key=f'{mode}/{self.side}',actor_called=raw is not None,
                    raw_policy_angle_rad=raw,angle_rad=angle,other_angle_rad=None,changed_angle_rad=None,
                    candidate_count=len(candidates),candidate_input_changed=None,
                    unknown_fraction=float((~(free|occupied))[direct].mean()) if direct.any() else None,
                    locked_side=self.side,direct_blocked=blocked,rejoin_approved=rejoined,
                    stall_reviews=self.stall_reviews,wait_reviews=self.wait_reviews,
                    presence_used=bool(prob is not None and constrained),
                    presence_probability=(float(presence) if arm=='predicted' else None),
                    measured_fraction_24m=known_ratio,rejoin_clear_reviews=self.clear_reviews,
                    predicted_blocked_unknown_cells_24m=(int((extent[unseen]>=snapshot['mask_threshold']).sum())
                                                        if arm=='predicted' else None),
                    local_safe_prefix_m=prefix,side_score=score,
                    side_options={str(s):dict(world_angle_rad=o['angle'],local_safe_prefix_m=o['prefix'],
                        vehicle_target_distance_m=o['target_m'],unseen_risk=o['risk'],score=o['score'])
                        for s,o in options.items()})

    def refresh_target(self, env, decision, car, accepted_target):
        """Advance along the same high-level course only behind a fresh local certificate."""
        course=decision.get('course_world_rad')
        car=np.asarray(car,float)
        if course is None or accepted_target is None or np.linalg.norm(car-accepted_target)>=1.2:
            return None
        direction=_unit(course)
        origin=np.asarray(decision['origin_xy'])
        # Bound both radial displacement and along-track advance, including
        # native detours from the original course. Delay another large choice
        # until the next true PPO review.
        travelled=np.linalg.norm(car-origin)
        limit=20.-travelled
        if limit < 1.5:
            return None
        sensor=_local_sensor(env.obs[env.ground])
        prefix=min(limit,2.5,_local_prefix(sensor,car,direction))
        if prefix<1.5:
            return None
        target=car+prefix*direction
        if np.linalg.norm(target-accepted_target)<.5:
            return None
        return target

    def after_issue(self,decision,accepted_target,changed):
        self.commands[-1]['target']=np.asarray(accepted_target,float).copy()
