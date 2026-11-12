"""A/B survey contracts, mapping backends and received-map navigation.

Survey, receive, navigate, repeat. Only the mapping backend and physical audit
read terrain truth; policy observations and navigation use delivered evidence.
"""
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from simulator.large_world_scenarios import CircleGeometry
from simulator.survey_reward_v10 import SurveyTask

ALGORITHM = 'vtmp3'
ACTION_CONTRACT = 'ab_survey_then_received_map_navigation_v1'

SIDE, CHANNELS, FEATURES, ACTIONS = 32, 4, 8, 4
HALF_EXTENT = 200.
DELTA_SCALE = np.array([.03, .03, 0., 0., .1, .2, 1., 1.], np.float32)


@dataclass(frozen=True)
class MappingConfig:
    world: float = 1000.
    resolution: float = 2.
    half_width: float = 12.
    clearance: float = 1.
    max_endpoint_radius: float = 200.
    min_length: float = 20.
    max_length: float = 200.
    fleet_size: int = 3
    radio_distance: float = 95.
    flight_height: float = 15.
    # Uncalibrated proxy assumptions, not measured v11-2 performance.
    setup_seconds: float = 8.
    survey_speed: float = 2.
    delay_jitter: float = .5
    failure_probability: float = .15
    distance_failure_probability: float = .15
    communication_failure_probability: float = .05
    timeout_seconds: float = 180.
    navigation_seconds: float = 30.
    ground_speed: float = 1.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'Non-finite mapping setting: {key}')
        for key in ('world', 'resolution', 'half_width', 'clearance', 'max_endpoint_radius',
                    'min_length', 'max_length', 'radio_distance', 'flight_height',
                    'survey_speed', 'timeout_seconds', 'navigation_seconds', 'ground_speed'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.setup_seconds < 0 or self.fleet_size != 3:
            raise ValueError('This v11-2 interface requires three aircraft and nonnegative setup time')
        if not 0 <= self.delay_jitter < 1:
            raise ValueError('delay_jitter must lie in [0, 1)')
        for key in ('failure_probability', 'distance_failure_probability', 'communication_failure_probability'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must lie in [0, 1]')
        if self.min_length > self.max_length or self.max_length > 200 or self.max_endpoint_radius > 200:
            raise ValueError('Tasks must fit the v11-2 20–200m bands and the model observation window')
        if not math.isclose(self.world/self.resolution, round(self.world/self.resolution)):
            raise ValueError('world must be divisible by resolution')
        if .8*(self.radio_distance-.25) <= self.flight_height:
            raise ValueError('No usable ground/air radio budget')


@dataclass(frozen=True)
class MappingResult:
    task_id: int
    cell_ids: np.ndarray
    occupied: np.ndarray
    elapsed_seconds: float
    status: str
    backend: str


def task_valid(task, vehicle_position, config):
    """Geometry-only candidate admission; neither truth nor route existence."""
    a, b, c = np.asarray(task.a), np.asarray(task.b), np.asarray(vehicle_position)
    length = float(np.linalg.norm(b-a))
    if not config.min_length-1e-6 <= length <= config.max_length+1e-6:
        return False
    if max(np.linalg.norm(a-c), np.linalg.norm(b-c)) > config.max_endpoint_radius+1e-6:
        return False
    # Sequential observation with a relay chain, matching v11-2's stationary-C
    # task interpretation. A and B need not be occupied simultaneously. This
    # geometric envelope is NOT a dynamic CBF or learned-policy certificate.
    link = .8*(config.radio_distance-.25)
    radius = math.sqrt(link**2-config.flight_height**2)+(config.fleet_size-1)*link
    return max(np.linalg.norm(a-c), np.linalg.norm(b-c)) <= radius


def make_task(a, b, config):
    return SurveyTask(tuple(a), tuple(b), config.half_width,
                      world_size=config.world, resolution=config.resolution)


class ProxyMapper:
    """Stochastic mapping proxy; only delivered occupancy enters navigation.

    This is not a simulation of the trained v11-2 policy. Requests use its
    survey message fields; results never fabricate a traversable A/B link.
    """

    name = 'stochastic_ab_mapping_proxy_v1'

    def __init__(self, circles, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._centers = np.asarray([p for p, r in circles], dtype=float).reshape(-1, 2)
        self._radii = np.asarray([r for p, r in circles], dtype=float)

    def _occupancy(self, centers):
        # Mark an entire pixel blocked if its rectangle intersects an obstacle.
        # This conservatism makes the downstream received-cell graph safe.
        occupied = np.zeros(len(centers), dtype=bool)
        for center, radius in zip(self._centers, self._radii):
            gap = np.maximum(np.abs(centers-center)-self.config.resolution/2, 0.)
            occupied |= (gap*gap).sum(-1) <= radius**2
        return occupied

    def local_observation(self, position, radius=4.):
        task = SurveyTask(tuple(position), tuple(position), radius,
                          world_size=self.config.world, resolution=self.config.resolution)
        return MappingResult(-1, task.cell_ids.copy(), self._occupancy(task.cells_xy),
                             0., 'local', self.name)

    def survey(self, task_id, task, vehicle_position):
        if not task_valid(task, vehicle_position, self.config):
            raise ValueError('Survey violates the declared task envelope')
        cfg, rng = self.config, self.rng
        length = float(np.linalg.norm(np.asarray(task.b)-task.a))
        deployment = min(np.linalg.norm(np.asarray(task.a)-vehicle_position),
                         np.linalg.norm(np.asarray(task.b)-vehicle_position))
        elapsed = (cfg.setup_seconds+(length+deployment)/cfg.survey_speed)
        elapsed *= rng.uniform(1-cfg.delay_jitter, 1+cfg.delay_jitter)
        failed = rng.random() < min(1., cfg.failure_probability+
                                   cfg.distance_failure_probability*(length/cfg.max_length)**2)
        lost = rng.random() < cfg.communication_failure_probability
        timeout = elapsed > cfg.timeout_seconds
        elapsed = min(elapsed, cfg.timeout_seconds)
        status = 'communication_failure' if lost else 'timeout' if timeout else 'partial_failure' if failed else 'complete'
        selected = np.ones(task.cell_count, dtype=bool)
        if lost:
            selected[:] = False
        elif failed or timeout:
            # A partial contiguous swath, rather than independent missing
            # pixels that would unrealistically destroy every navigation edge.
            fraction = rng.uniform(.2, .75)
            direction = (np.asarray(task.b)-task.a)/length
            selected = ((task.cells_xy-task.a)@direction <= fraction*length)
        return MappingResult(int(task_id), task.cell_ids[selected].copy(),
                             self._occupancy(task.cells_xy[selected]), float(elapsed),
                             status, self.name)


class ReceivedMap:
    def __init__(self, config):
        self.config = config
        self.n = round(config.world/config.resolution)
        self.cells = np.full((self.n, self.n), -1, dtype=np.int8)
        self._received = set()

    def ingest(self, packet):
        ids = np.asarray(packet.cell_ids)
        occupied = np.asarray(packet.occupied)
        if ids.ndim != 1 or ids.shape != occupied.shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Invalid delivered occupancy packet')
        if (ids < 0).any() or (ids >= self.n**2).any() or not np.isin(occupied, (0, 1)).all():
            raise ValueError('Delivered occupancy is outside the map contract')
        key = (packet.backend, packet.task_id)
        if packet.task_id >= 0 and key in self._received:
            return 0
        new = int(np.count_nonzero(self.cells.ravel()[np.unique(ids)] < 0))
        self.cells.ravel()[ids] = occupied.astype(np.int8)
        if packet.task_id >= 0:
            self._received.add(key)
        return new

    def safe_nodes(self):
        free = np.pad(self.cells == 0, 1, constant_values=False)
        clearance = distance_transform_edt(free)[1:-1, 1:-1]*self.config.resolution
        # Covers cell rectangles, between-node segments and endpoint access.
        return clearance > self.config.clearance+self.config.resolution*math.sqrt(2)

    def cell(self, point):
        xy = np.floor(np.asarray(point)/self.config.resolution).astype(int)
        return int(xy[1]), int(xy[0])

    def center(self, cell):
        y, x = cell
        return (np.array([x, y])+.5)*self.config.resolution

    def search(self, position, goal):
        """Dijkstra gives routes to the goal or the closest reachable frontier.

        Unknown cells have no graph nodes. There is no simulator/mapper handle
        here, and unsuccessful A/B surveys need not create connected routes.
        """
        safe = self.safe_nodes()
        source = self.cell(position)
        reachable = np.zeros_like(safe)
        if not (0 <= source[0] < self.n and 0 <= source[1] < self.n and safe[source]):
            return [np.asarray(position).copy()], reachable
        costs, previous, queue = {source: 0.}, {}, [(0., source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            reachable[node] = True
            y, x = node
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = y+dy, x+dx
                if not (0 <= nxt[0] < self.n and 0 <= nxt[1] < self.n and safe[nxt]):
                    continue
                if dx and dy and not (safe[y, x+dx] and safe[y+dy, x]):
                    continue
                candidate = cost+math.hypot(dy, dx)*self.config.resolution
                if candidate < costs.get(nxt, math.inf):
                    costs[nxt], previous[nxt] = candidate, node
                    heapq.heappush(queue, (candidate, nxt))
        goal_cell = self.cell(goal)
        if goal_cell in costs:
            target = goal_cell
        else:
            target = min(costs, key=lambda node: (float(np.linalg.norm(self.center(node)-goal)), costs[node]))
            if np.linalg.norm(self.center(target)-goal) >= np.linalg.norm(np.asarray(position)-goal):
                return [np.asarray(position).copy()], reachable
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        points = [np.asarray(position).copy(), *(self.center(node) for node in reversed(path))]
        if target == goal_cell:
            points.append(np.asarray(goal).copy())
        return points, reachable


def follow_path(points, budget_m):
    """A bounded kinematic graph follower; no instantaneous endpoint jumps."""
    result, traveled = [np.asarray(points[0]).copy()], 0.
    for point in points[1:]:
        delta = np.asarray(point)-result[-1]
        length = float(np.linalg.norm(delta))
        if length <= 1e-10:
            continue
        remaining = budget_m-traveled
        if length >= remaining:
            result.append(result[-1]+delta*(remaining/length))
            traveled = float(budget_m)
            break
        result.append(np.asarray(point).copy())
        traveled += length
    return result, traveled


def encode(received, position, goal, elapsed, last_duration, last_failed):
    _, reachable = received.search(position, goal)
    cfg = received.config
    center = np.floor(np.asarray(position)/cfg.resolution).astype(int)
    half = int(HALF_EXTENT/cfg.resolution)
    xs, ys = np.arange(center[0]-half, center[0]+half), np.arange(center[1]-half, center[1]+half)
    xx, yy = np.meshgrid(xs, ys)
    valid = (xx >= 0) & (xx < received.n) & (yy >= 0) & (yy < received.n)
    cells = np.ones_like(xx, dtype=np.int8)
    cells[valid] = received.cells[yy[valid], xx[valid]]
    connected = np.zeros_like(valid)
    connected[valid] = reachable[yy[valid], xx[valid]]
    channels = np.stack((cells >= 0, cells == 0, cells == 1, connected))
    # Area averages preserve small obstacles instead of point-sampling them.
    bins = np.minimum(SIDE-1, np.arange(2*half)*SIDE//(2*half))
    ids = (bins[:, None]*SIDE+bins[None, :]).ravel()
    counts = np.bincount(ids, minlength=SIDE*SIDE)
    maps = np.stack([np.bincount(ids, weights=c.ravel(), minlength=SIDE*SIDE)/counts
                     for c in channels]).reshape(CHANNELS, SIDE, SIDE).astype(np.float32)
    state = np.array([*(np.asarray(position)/1000.), *(np.asarray(goal)/1000.),
                      np.count_nonzero(received.cells >= 0)/10000., elapsed/1000.,
                      last_duration/200., float(last_failed)], np.float32)
    return maps, state


class SurveyNavigationEnv:
    def __init__(self, scene, config, *, backend=None, random_seed=None):
        self.config = config
        self.position = np.asarray(scene['start'], float).copy()
        self.goal = np.asarray(scene['goal'], float).copy()
        seed = int(scene['seed']) ^ 0x33AB if random_seed is None else random_seed
        if scene.get('terrain') == 'wildscenes_simplified':
            from .ab_wildscenes import WildScenesMapper, WildScenesAudit, geometry_for
            geometry, occupied = geometry_for(scene, config)
            self.mapper = backend if backend is not None else WildScenesMapper(occupied, config, seed)
            self._audit_geometry = WildScenesAudit(geometry, config.clearance)
        else:
            self.mapper = backend if backend is not None else ProxyMapper(scene['circles'], config, seed)
            self._audit_geometry = CircleGeometry(scene['circles'], config.world, config.clearance)
        self.received = ReceivedMap(config)
        self.received.ingest(self.mapper.local_observation(self.position))
        # Physical truth is used only for post-execution audit, never selection.
        self.elapsed = self.last_duration = 0.
        self.last_failed = False
        self.steps = 0
        self.done = False

    def observe(self):
        return encode(self.received, self.position, self.goal, self.elapsed,
                      self.last_duration, self.last_failed)

    def step(self, endpoints):
        if self.done:
            raise ValueError('Episode is already complete')
        points = np.asarray(endpoints, float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError('Action must contain A and B, each with x/y coordinates')
        task = make_task(points[0], points[1], self.config)
        before = self.position.copy()
        initial_distance = float(np.linalg.norm(before-self.goal))
        packet = self.mapper.survey(self.steps, task, before.copy())
        if not np.isfinite(packet.elapsed_seconds) or packet.elapsed_seconds < 0:
            raise ValueError('Invalid mapping duration')
        # Mapping is blocking in simulated time; the vehicle holds at real C.
        self.elapsed += packet.elapsed_seconds
        new_cells = self.received.ingest(packet)
        route, _ = self.received.search(self.position, self.goal)
        executed, distance = follow_path(route, self.config.ground_speed*self.config.navigation_seconds)
        if not all(self._audit_geometry.clear(a, b) for a, b in zip(executed, executed[1:])):
            raise RuntimeError('Received-map navigation violated continuous physical clearance')
        self.position = executed[-1].copy()
        self.elapsed += distance/self.config.ground_speed
        self.received.ingest(self.mapper.local_observation(self.position))
        self.last_duration = packet.elapsed_seconds
        self.last_failed = packet.status != 'complete'
        self.steps += 1
        self.done = bool(np.linalg.norm(self.position-self.goal) <= 1.)
        progress = initial_distance-float(np.linalg.norm(self.position-self.goal))
        # Task utility rewards eventual vehicle progress and useful NEW evidence.
        # Waiting and repeated surveys cost time; blocked terrain is valid data.
        reward = progress+.002*new_cells-.02*packet.elapsed_seconds-.01*distance+50.*self.done
        outcome = np.array([progress/30., new_cells/1000., packet.elapsed_seconds/200.,
                            float(self.last_failed), float(self.done)], np.float32)
        info = dict(task_id=self.steps-1, survey=task.to_message(), backend=packet.backend,
                    status=packet.status, survey_seconds=packet.elapsed_seconds,
                    survey_ground_displacement_m=0., delivered_cells=len(packet.cell_ids),
                    new_cells=new_cells, vehicle_distance_m=distance,
                    goal_progress_m=progress, goal_distance_m=float(np.linalg.norm(self.position-self.goal)),
                    vehicle_position=self.position.tolist(), elapsed_seconds=self.elapsed,
                    reward=float(reward), reached=self.done)
        return self.observe(), outcome, info
