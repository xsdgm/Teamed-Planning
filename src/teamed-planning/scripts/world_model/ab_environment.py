"""Survey, receive, navigate, repeat. Policy observations never contain truth."""
import numpy as np

from simulator.large_world_scenarios import CircleGeometry
from .ab_mapping import ProxyMapper, make_task
from .ab_navigation import ReceivedMap, follow_path

SIDE, CHANNELS, FEATURES, ACTIONS = 32, 4, 8, 4
HALF_EXTENT = 200.
DELTA_SCALE = np.array([.03, .03, 0., 0., .1, .2, 1., 1.], np.float32)


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
        self.mapper = backend or ProxyMapper(scene['circles'], config,
                                             int(scene['seed']) ^ 0x33AB if random_seed is None else random_seed)
        self.received = ReceivedMap(config)
        self.received.ingest(self.mapper.local_observation(self.position))
        # Physical truth is used only for post-execution audit, never selection.
        self._audit_geometry = CircleGeometry(scene['circles'], config.world, config.clearance)
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
