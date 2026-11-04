"""CoGraph v2 vehicle motion filter: backbone link keeping and vehicle-vehicle yielding."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground.vehicle_motion import VehicleMotionFilter, LocalMap, link_step, contact_step
from simulator.env import MultiRobotEnv, EnvConfig
from simulator.train_mappo import validate_config, reward_contract

CONFIG = Path(__file__).resolve().parents[2]/'config/algorithm/cograph_v2.json'
FLIGHT = 15.


def observation(position, theta=0., drones=(), vehicles=()):
    """A vehicle's own observation: drones are (key, xy) at flight height, vehicles
    are (key, xy, velocity, broadcast extras)."""
    p = np.r_[np.asarray(position, float), 0.]
    neighbors = []
    for i, (key, q) in enumerate(drones):
        neighbors.append(dict(id=100+i, type='drone', relative_position=np.r_[q, FLIGHT]-p,
                              relative_velocity=np.zeros(3), broadcast=dict(drone_id=key)))
    for i, (key, q, velocity, extra) in enumerate(vehicles):
        neighbors.append(dict(id=10+i, type='diff_drive', relative_position=np.r_[q, 0.]-p,
                              relative_velocity=np.r_[velocity, 0.], broadcast=dict(robot_id=key, **extra)))
    return dict(self=dict(position=p, orientation=theta, velocity=np.zeros(3)), neighbors=neighbors)


def distance3(xy, q):
    return float(np.linalg.norm(np.r_[np.asarray(xy, float)-np.asarray(q, float), FLIGHT]))


class StepMathTests(unittest.TestCase):
    def test_link_step(self):
        u = np.array([1., 0.])
        self.assertAlmostEqual(link_step(np.array([5., 0.]), u, 100.), 5.)
        self.assertEqual(link_step(np.array([12., 0.]), u, 100.), 0.)          # beyond: no outward step
        self.assertAlmostEqual(link_step(np.array([-12., 0.]), u, 100.), 24.)  # beyond: closing in is fine
        self.assertTrue(math.isinf(link_step(np.zeros(2), u, math.inf)))

    def test_contact_step(self):
        u = np.array([1., 0.])
        self.assertAlmostEqual(contact_step(np.array([-5., 0.]), u, 1.), 4.)
        self.assertTrue(math.isinf(contact_step(np.array([-5., 3.]), u, 1.)))   # passes beside
        self.assertTrue(math.isinf(contact_step(np.array([.5, 0.]), u, 1.)))    # too close, moving away
        self.assertEqual(contact_step(np.array([-.5, 0.]), u, 1.), 0.)          # too close, closing in


class LinkKeepingTests(unittest.TestCase):
    def setUp(self):
        self.filter = VehicleMotionFilter('ground_0', 0, keep_range=88.)

    def test_lone_link_caps_outward_step_at_keep_range(self):
        for start in (85., 86.5, 87.4):
            x = -math.sqrt(start**2-FLIGHT**2)
            (v, _), info = self.filter.filter((1., 0.), observation((x, 0.), math.pi, drones=[('drone_0', (0., 0.))]), (x-50., 0.))
            self.assertLessEqual(distance3((x-v, 0.), (0., 0.)), 88.+1e-6)
        x = -math.sqrt(88.5**2-FLIGHT**2)
        (v, _), info = self.filter.filter((1., 0.), observation((x, 0.), math.pi, drones=[('drone_0', (0., 0.))]), (x-50., 0.))
        self.assertEqual(v, 0.)
        self.assertIn('link_hold', info['events'])
        # Closing in is never limited.
        (v, _), info = self.filter.filter((1., 0.), observation((x, 0.), 0., drones=[('drone_0', (0., 0.))]), (0., 0.))
        self.assertEqual(v, 1.)
        self.assertEqual(info['events'], ())

    def test_witnessed_link_is_not_kept(self):
        # drone_0 behind at the edge, drone_1 between: the relative neighbourhood drops
        # the vehicle-drone_0 link (the drones keep it through drone_1), as the drones do.
        x = -math.sqrt(88.5**2-FLIGHT**2)
        obs = observation((x, 0.), math.pi, drones=[('drone_0', (0., 0.)), ('drone_1', (x/2, 0.))])
        (v, _), info = self.filter.filter((1., 0.), obs, (x-50., 0.))
        self.assertEqual(v, 1.)
        self.assertNotIn('link_limited', info['events'])

    def test_vehicle_links_are_kept_too(self):
        (v, _), info = self.filter.filter((1., 0.), observation((-87.6, 0.), math.pi,
            vehicles=[('ground_1', (0., 0.), (0., 0.), {})]), (-150., 0.))
        self.assertAlmostEqual(v, .4)
        self.assertIn('link_limited', info['events'])


class Fleet:
    """Two or three diff-drive vehicles in an open (or corridor) world, each with its own
    straight route controller and its own filter; only received geometry is used."""

    def __init__(self, starts, goals, blocked=None):
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = MultiRobotEnv(EnvConfig(world_size=(200., 200.), dt=1., communication_radius=95.))
        self.ids = [self.env.add_ground_robot(x, y, theta, max_speed=1., max_angular_speed=1.)
                    for x, y, theta in starts]
        self.goals = [None if g is None else np.asarray(g, float) for g in goals]
        self.filters = [VehicleMotionFilter(f'ground_{k}', k) for k in range(len(starts))]
        self.blocked = blocked
        self.obs = self.env.reset()
        self.broadcast = [self._broadcast(k) for k in range(len(starts))]
        self.contacts = 0
        self.min_gap = math.inf
        self.events = [[] for _ in starts]

    def _position(self, k):
        return np.asarray(self.obs[self.ids[k]]['self']['position'], float)[:2]

    def _broadcast(self, k):
        goal = self.goals[k]
        return dict(robot_id=f'ground_{k}', target=None if goal is None else goal.tolist(),
                    planned_path=[] if goal is None else [goal.tolist()])

    def _route(self, k):
        goal, p = self.goals[k], self._position(k)
        if goal is None or np.linalg.norm(goal-p) < .5:
            return (0., 0.), None
        theta = float(self.obs[self.ids[k]]['self']['orientation'])
        error = math.atan2(goal[1]-p[1], goal[0]-p[0])-theta
        error = math.atan2(math.sin(error), math.cos(error))
        return ((1. if abs(error) < .25 else 0.), float(np.clip(error, -1., 1.))), goal

    def run(self, steps):
        for _ in range(steps):
            actions = {}
            for k, rid in enumerate(self.ids):
                motion, waypoint = self._route(k)
                local = None
                if self.blocked is not None:
                    local = LocalMap(self.blocked(self._position(k)), .5, self._position(k))
                motion, info = self.filters[k].filter(motion, self.obs[rid], waypoint, local)
                self.events[k].extend(info['events'])
                actions[rid] = dict(motion=np.asarray(motion), broadcast=self._broadcast(k))
            self.obs, info = self.env.step(actions)
            self.contacts += sum(c['object'] == 'robot' for c in info['collisions'])
            ps = [self._position(k) for k in range(len(self.ids))]
            self.min_gap = min([self.min_gap]+[float(np.linalg.norm(a-b)) for i, a in enumerate(ps) for b in ps[:i]])
        return [self._position(k) for k in range(len(self.ids))]


def corridor(half_width):
    """Local 16x16 (0.5 m) inflated grid: free only within |y-100| < half_width."""
    def blocked(center):
        ys = center[1]+(np.arange(16)-8+.5)*.5
        return np.repeat((np.abs(ys-100.) >= half_width)[:, None], 16, axis=1)
    return blocked


class VehicleVehicleTests(unittest.TestCase):
    def test_head_on_in_the_open_pass_on_the_right(self):
        fleet = Fleet([(80., 100., 0.), (92., 100., math.pi)], [(110., 100.), (62., 100.)])
        a, b = fleet.run(60)
        self.assertEqual(fleet.contacts, 0)
        self.assertGreater(fleet.min_gap, 1.)
        self.assertLess(np.linalg.norm(a-(110., 100.)), 1.)
        self.assertLess(np.linalg.norm(b-(62., 100.)), 1.)
        self.assertIn('passing', fleet.events[0])

    def test_parked_vehicle_on_the_route_is_passed(self):
        # ep18: a vehicle stopped on another's route held it for ~300 steps.
        fleet = Fleet([(80., 100., 0.), (88., 100., 0.)], [(100., 100.), None])
        a, b = fleet.run(40)
        self.assertEqual(fleet.contacts, 0)
        self.assertLess(np.linalg.norm(a-(100., 100.)), 1.)

    def test_follower_does_not_overtake_or_touch(self):
        fleet = Fleet([(80., 100., 0.), (83., 100., 0.)], [(120., 100.), (125., 100.)])
        fleet.run(30)
        self.assertEqual(fleet.contacts, 0)
        self.assertNotIn('passing', fleet.events[0])

    def test_holding_vehicle_makes_way(self):
        # Corridor wide enough to step aside, too narrow for a pass beside the holder.
        fleet = Fleet([(80., 100., 0.), (90., 100., math.pi/2)], [(110., 100.), None], blocked=corridor(2.4))
        a, _ = fleet.run(60)
        self.assertEqual(fleet.contacts, 0)
        self.assertIn('make_way', fleet.events[1])
        self.assertLess(np.linalg.norm(a-(110., 100.)), 1.)

    def test_narrow_head_on_the_higher_rank_backs_up(self):
        fleet = Fleet([(80., 100., 0.), (86., 100., math.pi)], [(100., 100.), (60., 100.)], blocked=corridor(.6))
        before = fleet._position(1).copy()
        fleet.run(25)
        self.assertEqual(fleet.contacts, 0)
        self.assertIn('retreat', fleet.events[1])
        self.assertNotIn('retreat', fleet.events[0])
        self.assertGreater(fleet._position(1)[0], before[0])   # ground_1 backed away
        self.assertGreater(fleet._position(0)[0], 80.)          # ground_0 advanced


class ConfigTests(unittest.TestCase):
    def test_contract_and_validation(self):
        config = validate_config(json.loads(CONFIG.read_text()))
        self.assertEqual(config['vehicle_motion'], 'link_keeping')
        self.assertTrue(reward_contract(config).endswith('_vehicle_link_keeping_bridged'))
        free = dict(json.loads(CONFIG.read_text()), vehicle_motion='free')
        free.pop('vehicle_keep_range', None)
        self.assertEqual(reward_contract(validate_config(free)),
                         'cograph_v2_task_option_space_single_task_decentralized_vehicles_tree_budget_bridged')
        for bad in (dict(vehicle_motion='free', vehicle_keep_range=80.), dict(vehicle_keep_range=95.),
                    dict(vehicle_motion='link_keeping', vehicle_planner='parked', ground_stationary=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(dict(json.loads(CONFIG.read_text()), **bad))


if __name__ == '__main__':
    unittest.main()
