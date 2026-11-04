"""CoGraph v2 vehicle-side motion filter: backbone link keeping and vehicle-vehicle yielding.

Runs on every vehicle after its own claim decision, on that vehicle's own
information only: its direct radio neighbours (drones and vehicles, relative
position and velocity, as received), its own route waypoint and its own local
ground map. There is no leader and no shared plan; two vehicles that meet apply
the same rules to the same received numbers.

Link keeping. A vehicle's backbone links are the relative-neighbourhood links of
its received geometry, the rule the drones apply on their side, so a link a
drone must keep to this vehicle is one this vehicle keeps as well. The drones'
CBF treats a vehicle as non-cooperative (it may move outward at its top speed);
a vehicle that drove away regardless pulled its drones past feasibility (all
seven safety terminations of the first CoGraph v2 batch). The vehicle now caps
its own step so that no backbone link grows beyond keep_range (inside the
drones' CBF edge); past keep_range it may only close in. It therefore waits where
its coverage ends until the drones extend it.

Vehicle-vehicle. No step ends within `contact` of another vehicle's current
position (plus that vehicle's own possible step while it moves). A vehicle with
another one in its lane ahead passes it on the right, or on the left when its
local map blocks the right. A stopped vehicle in an approaching vehicle's lane
(its broadcast route, or its motion) steps aside to that vehicle's left. A vehicle
driving on ahead in the same lane is followed; one that has stood still for
parked_after steps is passed. When neither side is open (narrow passage),
the higher rank number yields: it holds, and head-on it backs up after
retreat_after blocked steps.
"""
import math

import numpy as np

from drone.local_backbone import relative_neighborhood

EVENTS = ('link_limited', 'link_hold', 'contact_limited', 'passing', 'make_way', 'yielding', 'retreat')


def link_step(offset, direction, limit_sq):
    """Largest s >= 0 with |offset + s*direction|^2 <= max(limit_sq, |offset|^2).

    offset: partner -> vehicle (horizontal); direction: unit step direction.
    Inside the limit the step may reach it; beyond it only non-outward steps pass."""
    b = float(np.dot(offset, direction))
    current = float(np.dot(offset, offset))
    return max(0., -b+math.sqrt(max(b*b+max(limit_sq, current)-current, 0.)))


def contact_step(offset, direction, distance):
    """Largest s >= 0 (inf if unbounded) keeping |offset + t*direction| >= distance for all t <= s.

    Already closer than `distance`: only steps that do not close in are allowed."""
    b = float(np.dot(offset, direction))
    if b >= 0.:
        return math.inf
    current = float(np.dot(offset, offset))
    if current < distance*distance:
        return 0.
    disc = b*b-(current-distance*distance)
    return math.inf if disc < 0. else max(0., -b-math.sqrt(disc))


def vehicle_rank(neighbor):
    """Rank of a received vehicle: the number in its broadcast id ground_<k>, else its radio id."""
    name = str((neighbor.get('broadcast') or {}).get('robot_id', ''))
    tail = name.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else 1000+int(neighbor['id'])


def intended_course(broadcast, position, velocity):
    """Where a received vehicle means to go: toward the next point of its broadcast route
    while it has a target (also while it holds), else along its motion; None if idle."""
    if broadcast.get('target') is not None:
        for point in broadcast.get('planned_path') or ():
            d = np.asarray(point, float)[:2]-position
            norm = float(np.linalg.norm(d))
            if norm > .5:
                return d/norm
    speed = float(np.linalg.norm(velocity))
    return velocity/speed if speed > .1 else None


class LocalMap:
    """The vehicle's own inflated local obstacle grid, centred on the vehicle (row = y)."""

    def __init__(self, blocked, resolution, center):
        self.blocked = np.asarray(blocked, bool)
        self.resolution = float(resolution)
        self.center = np.asarray(center, float)[:2]

    def free(self, point):
        height, width = self.blocked.shape
        x = int(math.floor((point[0]-self.center[0])/self.resolution+width/2.))
        y = int(math.floor((point[1]-self.center[1])/self.resolution+height/2.))
        # Outside the sensed square is unknown: the check is repeated as the vehicle closes in.
        return not (0 <= x < width and 0 <= y < height and self.blocked[y, x])

    def segment_free(self, first, second):
        first, second = np.asarray(first, float), np.asarray(second, float)
        count = max(2, int(np.linalg.norm(second-first)/(self.resolution/2.))+1)
        return all(self.free(q) for q in np.linspace(first, second, count))


class VehicleMotionFilter:
    def __init__(self, robot_id, rank, comm_range=95., keep_range=88., max_speed=1., max_turn=1., dt=1.,
                 contact=1., lookahead=6., lane=1.6, pass_offset=1.6, retreat_after=10, parked_after=5):
        if not (0. < keep_range < comm_range and contact > 0. and lane > 0. and pass_offset >= contact
                and lookahead > contact and max_speed > 0. and dt > 0. and retreat_after >= 1 and parked_after >= 1):
            raise ValueError('Invalid vehicle motion filter parameters')
        self.robot_id, self.rank = str(robot_id), int(rank)
        self.comm_range, self.keep_range = float(comm_range), float(keep_range)
        self.max_speed, self.max_turn, self.dt = float(max_speed), float(max_turn), float(dt)
        self.contact, self.lookahead, self.lane = float(contact), float(lookahead), float(lane)
        self.pass_offset, self.retreat_after, self.parked_after = float(pass_offset), int(retreat_after), int(parked_after)
        self.reset()

    def reset(self):
        self.blocked_steps = 0
        self.stopped = {}   # received vehicle -> consecutive steps seen standing still
        self.counts = dict.fromkeys(EVENTS, 0)
        self.last_events = ()

    # Helpers -------------------------------------------------------------------
    def _steer(self, theta, direction, speed):
        """Diff-drive command toward a direction: turn (not past it), translate once aligned."""
        error = math.atan2(math.sin(math.atan2(direction[1], direction[0])-theta),
                           math.cos(math.atan2(direction[1], direction[0])-theta))
        turn = float(np.clip(error/self.dt, -self.max_turn, self.max_turn))
        return (speed if abs(error) < .25 else 0.), turn

    def _side_point(self, p, anchor, normal, along, local_map, vehicles, skip):
        """A pass/aside point anchor + offset*normal, if its approach is free of obstacles and vehicles."""
        point = anchor+normal*self.pass_offset
        if local_map is not None and not (local_map.segment_free(p, point)
                                          and local_map.segment_free(point, point+along*1.5)):
            return None
        if any(np.linalg.norm(point-q) < self.pass_offset for key, (q, _, _, _) in vehicles.items() if key != skip):
            return None
        return point

    # Filter --------------------------------------------------------------------
    def filter(self, motion, observation, waypoint=None, local_map=None):
        """(linear, angular) command after link keeping and vehicle yielding.

        motion: the route controller's diff-drive command; observation: this
        vehicle's own simulator observation; waypoint: its current route
        waypoint while it intends to drive (None when holding)."""
        own = observation['self']
        p3 = np.asarray(own['position'], float)
        p = p3[:2]
        theta = float(own.get('orientation', 0.))
        heading = np.array([math.cos(theta), math.sin(theta)])
        own_velocity = np.asarray(own.get('velocity', (0., 0., 0.)), float)
        geometry, vehicles = {}, {}
        for neighbor in observation.get('neighbors', []):
            q = p3+np.asarray(neighbor['relative_position'], float)
            broadcast = neighbor.get('broadcast') or {}
            if neighbor['type'] == 'drone':
                geometry[broadcast.get('drone_id', f"robot_{neighbor['id']}")] = q
            elif neighbor['type'] in ('diff_drive', 'holonomic'):
                key = broadcast.get('robot_id', f"robot_{neighbor['id']}")
                geometry[key] = q
                velocity = (np.asarray(neighbor.get('relative_velocity', (0., 0., 0.)), float)+own_velocity)[:2]
                vehicles[key] = (q[:2], velocity, vehicle_rank(neighbor), intended_course(broadcast, q[:2], velocity))
        self.stopped = {key: (self.stopped.get(key, 0)+1 if float(np.linalg.norm(velocity)) <= .05 else 0)
                        for key, (_, velocity, _, _) in vehicles.items()}
        linear, angular = float(motion[0]), float(motion[1])
        events = []
        head_on = False

        intends = waypoint is not None and (abs(linear) > 1e-9 or abs(angular) > 1e-9)
        route = None
        if intends:
            route = np.asarray(waypoint, float)[:2]-p
            norm = float(np.linalg.norm(route))
            route = route/norm if norm > 1e-6 else heading
            left = np.array([-route[1], route[0]])
            ahead = []
            for key, (q, velocity, rank, course) in vehicles.items():
                d = q-p
                along, lateral = float(np.dot(d, route)), float(np.dot(d, left))
                if 0. < along <= self.lookahead and abs(lateral) < self.lane:
                    ahead.append((along, key))
            if ahead:
                _, key = min(ahead)
                q, velocity, rank, course = vehicles[key]
            if ahead and (float(np.dot(velocity, route)) >= .5*self.max_speed
                          or ((course is None or float(np.dot(course, route)) > .5)
                              and self.stopped[key] < self.parked_after)):
                ahead = []   # it drives (or is about to drive) on ahead of us: follow, do not overtake
            if ahead:
                point = None
                for normal in (-left, left):   # keep right, else left
                    point = self._side_point(p, q, normal, route, local_map, vehicles, key)
                    if point is not None:
                        break
                if point is not None:
                    events.append('passing')
                    linear, angular = self._steer(theta, point-p, self.max_speed)
                else:
                    head_on = course is not None and float(np.dot(course, p-q)) > 0.
                    stopped = float(np.linalg.norm(velocity)) <= .05
                    # Head-on with priority, or following a moving one: keep the
                    # route command; the contact cap below holds the distance.
                    if (head_on and rank < self.rank) or (not head_on and stopped):
                        # Narrow passage: give way to the higher-priority vehicle (or wait
                        # behind a stopped one); head-on, back up after a while.
                        events.append('yielding')
                        self.blocked_steps += 1
                        linear, angular = 0., 0.
                        if head_on and self.blocked_steps >= self.retreat_after:
                            behind = p-heading*self.max_speed*self.dt
                            if local_map is None or local_map.segment_free(p, behind):
                                events.append('retreat')
                                linear = -self.max_speed
        else:
            # Holding: step aside for a moving vehicle whose lane this vehicle blocks.
            for key, (q, velocity, rank, course) in sorted(vehicles.items(), key=lambda item: np.linalg.norm(item[1][0]-p)):
                if course is None:
                    continue
                d = p-q
                along, lateral = float(np.dot(d, course)), float(np.dot(d, np.array([-course[1], course[0]])))
                if 0. < along <= self.lookahead and abs(lateral) < self.lane:
                    left = np.array([-course[1], course[0]])
                    for normal in (left, -left):   # the passer keeps to its right
                        point = self._side_point(p, p, normal, course, local_map, vehicles, key)
                        if point is not None:
                            events.append('make_way')
                            linear, angular = self._steer(theta, point-p, self.max_speed)
                            break
                    break
        if not head_on:
            self.blocked_steps = 0

        # Hard caps on the translation of this step (along the current heading).
        step = abs(linear)*self.dt
        if step > 0.:
            direction = heading*(1. if linear > 0. else -1.)
            required = relative_neighborhood(p3, geometry, self.comm_range)
            limited = step
            for key in required:
                q = geometry[key]
                limit_sq = self.keep_range**2-float(q[2]-p3[2])**2
                limited = min(limited, link_step(p-q[:2], direction, limit_sq))
            if limited < step-1e-9:
                events.append('link_limited')
                if limited <= 1e-9:
                    events.append('link_hold')
            contact = limited
            for q, velocity, _, _ in vehicles.values():
                moving = float(np.linalg.norm(velocity))
                reach = self.contact+(moving*self.dt if moving > .05 else 0.)
                contact = min(contact, contact_step(p-q, direction, reach))
            if contact < limited-1e-9:
                events.append('contact_limited')
            linear = math.copysign(contact/self.dt, linear)
        for event in events:
            self.counts[event] += 1
        self.last_events = tuple(events)
        return (float(linear), float(angular)), dict(events=tuple(events), blocked_steps=self.blocked_steps)
