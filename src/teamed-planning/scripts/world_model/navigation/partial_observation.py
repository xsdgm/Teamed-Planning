"""Spatially incomplete aerial surveys for synthetic V9 obstacle fields."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np

from .wildscenes_runtime import _known_semantics, receive_known_evidence


PARTIAL_OBSERVATION_PROTOCOL = 'v9_seeded_aerial_survey_swaths_v1'


@dataclass(frozen=True)
class SurveySwaths:
    """Parallel curved flight strips. Only the scene seed sets their geometry."""

    heading: float
    spacing: float
    width: float
    offset: float
    amplitude: float
    wavelength: float
    phase: float

    @classmethod
    def from_seed(cls, scene_seed):
        # Separate the support generator from V9's circle and action generators.
        rng = np.random.default_rng(np.random.SeedSequence([scene_seed, 0x53575259]))
        heading = rng.uniform(0., pi)
        spacing = rng.uniform(125., 230.)
        return cls(heading, spacing, spacing * rng.uniform(.18, .65),
                   rng.uniform(0., spacing), rng.uniform(4., 18.),
                   rng.uniform(120., 260.), rng.uniform(0., 2. * pi))

    def known_mask(self, position, shape, resolution):
        """Return fully supported pixels in simulator row-Y/column-X coordinates."""
        height, width = shape
        resolution = float(resolution)
        if height < 1 or width < 1 or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('An aerial sensor needs positive dimensions and resolution')
        x = position[0] + (np.arange(width) + .5 - width / 2.) * resolution
        y = position[1] + (np.arange(height)[:, None] + .5 - height / 2.) * resolution
        c, s = cos(self.heading), sin(self.heading)
        along = c * x + s * y
        across = -s * x + c * y
        angular_rate = 2. * pi / self.wavelength
        center_offset = self.offset + self.amplitude * np.sin(angular_rate * along + self.phase)
        delta = (across - center_offset + self.spacing / 2.) % self.spacing - self.spacing / 2.
        # The projection of a pixel's half extent onto either strip axis is
        # at most this radius. Account for the maximum slope of the curved
        # centreline too, so a partly surveyed pixel can never count as known.
        radius = .5 * resolution * (abs(c) + abs(s))
        return np.abs(delta) + radius * (1. + self.amplitude * angular_rate) <= self.width / 2.


@contextmanager
def partial_v9_runtime(me, scene_seed):
    """Install before Experiment construction so the first reset is covered.

    Only aerial occupancy observations are masked. Ground local sensing,
    communication, circle geometry and the simulator collision handler keep
    their native implementations. The V9 reward reads the actually received
    maps and its original terminal events.
    """
    support = SurveySwaths.from_seed(scene_seed)
    original_env, original_drone, original_ledger = me.MultiRobotEnv, me.DroneController, me.DeliveredEvidence

    def environment(config):
        env = original_env(config)
        get_observations = env._get_observations

        def observations():
            received = get_observations()
            for drone in env.drones:
                obs = received[drone.robot_id]
                if obs['map_axes'] != 'yx':
                    raise ValueError('Expected simulator aerial sensor grid with row Y, column X')
                pixels = np.asarray(obs['occupancy_map'], dtype=np.int8)
                sensor_valid = bool(obs.get('sensor_valid', True))
                known = support.known_mask(obs['self']['position'], pixels.shape, obs['map_resolution'])
                known &= (pixels >= 0) & sensor_valid
                obs['occupancy_map'] = np.where(known, pixels, np.int8(-1))
                obs['occupancy_known_mask'] = known
                # A functioning sensor may report a completely unsurveyed tile.
                obs['sensor_valid'] = sensor_valid
            return received

        env._get_observations = observations
        return env

    def drone(*args, **kwargs):
        controller = original_drone(*args, **kwargs)
        _known_semantics(controller.semantic_processor)
        return controller

    class ReceivedSurveyEvidence(original_ledger):
        receive = receive_known_evidence

    try:
        me.MultiRobotEnv, me.DroneController, me.DeliveredEvidence = environment, drone, ReceivedSurveyEvidence
        yield
    finally:
        me.MultiRobotEnv, me.DroneController, me.DeliveredEvidence = original_env, original_drone, original_ledger
