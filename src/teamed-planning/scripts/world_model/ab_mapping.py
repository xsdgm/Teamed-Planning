"""The only online component permitted to read unobserved terrain.

This backend is a stochastic proxy, not a simulation of the trained v11-2
policy. Requests use its survey message fields. Results contain only delivered
occupancy evidence, never a fabricated traversable A/B connection.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from simulator.survey_reward_v10 import SurveyTask


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
