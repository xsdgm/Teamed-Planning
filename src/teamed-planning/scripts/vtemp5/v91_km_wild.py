"""Use the formal lake and valley raster sensor and collision contract with V9.1."""
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch


def _raster_case(scene):
    case = scene.get('case', {})
    if case.get('terrain') == 'wildscenes_raster':
        return case
    if scene.get('kind') in ('shore', 'bridge'):
        raise ValueError('A kilometre shore or bridge mission requires its original raster case')
    return None


@contextmanager
def wild_runtime(scene):
    """Enter before importing DeliveredEvidence or constructing the fleet Experiment."""
    case = _raster_case(scene)
    if case is None:
        yield
        return

    # Import after load_frozen binds this process to the episode snapshot. The
    # UAV snapshot stores the raster backend under shared/simulator/wildscenes,
    # while the older stand-alone vehicle trainer uses a flat copy of that file.
    from mappo import mappo_experiment
    from shared.simulator.wildscenes import wildscenes_map
    from vtemp2_3 import wildscenes_runtime

    with patch.object(wildscenes_runtime, 'load_map_backend', return_value=wildscenes_map):
        forest = wildscenes_runtime.load_forest(case, repo=None)
    # The auxiliary fine navigation image is rendered from analytic obstacles.
    # Raster lakes and rivers exist in the measured occupancy tile. Selecting
    # that tile also selects its matching resolution in the native planner.
    raster_config = partial(mappo_experiment.EnvConfig, ground_navigation_resolution=None)
    with patch.object(mappo_experiment, 'EnvConfig', raster_config), \
            wildscenes_runtime.raster_runtime(mappo_experiment, forest, wildscenes_map,
                                              case['required_clearance_m']):
        yield


def install_wild_ground_filter(env, scene):
    """Guard the prepared first command and all following commands using local sensors."""
    case = _raster_case(scene)
    if case is None:
        return
    from shared.simulator.wildscenes import wildscenes_map
    from vtemp2_3.wildscenes_runtime import filter_ground_motion

    if not getattr(env.env, '_forest_adapter_installed', False):
        raise RuntimeError('Raster support must be active before fleet initialization')
    robot = next(r for r in env.env.ground_robots if r.robot_id == env.ground)
    clearance = case['required_clearance_m']
    original_action = env._ground_action
    env.ground_motion_filter_interventions = 0

    def filter_action(action):
        observation = env.obs[env.ground]
        if observation['time_step'] != env.env.config.dt:
            raise ValueError('Ground sensor timestep must match the simulator')
        result, event = filter_ground_motion(action, observation, wildscenes_map,
                                            clearance=clearance, max_speed=robot.max_speed)
        env.ground_motion_filter_interventions += int(event['intervened'])
        return result

    def grounded_action(neighbors=None):
        return filter_action(original_action(neighbors))

    env.ground_action = filter_action(env.ground_action)
    env._ground_action = grounded_action
