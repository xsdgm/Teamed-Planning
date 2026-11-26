"""Check the native planner's selected sensor at the four recorded water stalls."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[5]
BASELINE = REPO / 'artifacts/v91-frozen-km-paired-20261126'


class RasterGroundSensorTest(unittest.TestCase):
    def test_native_planner_receives_lake_and_river_observations(self):
        if not (BASELINE / 'diagnostics/water_stall_audit.json').is_file():
            self.skipTest('Archived kilometre scenarios are unavailable')
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--probe'],
                       cwd=REPO, check=True, timeout=90)


def probe():
    from contextlib import redirect_stdout
    import os
    from unittest.mock import patch
    import numpy as np
    import torch

    sys.path.insert(0, str(REPO / 'src/teamed-planning/scripts/vtemp5'))
    from v91_extent_collect import load_frozen
    from v91_km_cases import km_cases
    from v91_km_wild import wild_runtime
    from v91_lake_drive import _experiment

    weight = Path(json.loads((BASELINE / 'protocol.json').read_text())['weights']['uav']['path'])
    uav, config = load_frozen(weight.parent, weight)
    from mappo import mappo_experiment as me

    audit = json.loads((BASELINE / 'diagnostics/water_stall_audit.json').read_text())
    poses = {r['case_id']: r['stationary_position_xy'] for r in audit if r['arm'] == 'predicted'}
    expected = {'lake-dev-00-detour_left': 80, 'lake-dev-01-detour_right': 76,
                'valley-dev-00-detour_right': 92, 'valley-dev-01-detour_left': 120}
    assert poses.keys() == expected.keys()
    circle = 'vtemp5-circle-development-1545411713'
    scenes = {s['case_id']: s for s in km_cases(REPO)}
    for case_id in [*poses, circle]:
        scene = scenes[case_id]
        with wild_runtime(scene), torch.inference_mode(), open(os.devnull, 'w') as sink, redirect_stdout(sink):
            with patch.object(me.Experiment, '_ground_action', return_value=None):
                env = _experiment(scene, scene['seed'], uav, dict(config, world_size=scene['world_size']))
            try:
                if case_id != circle:
                    env.env.ground_robots[0].position[:2] = poses[case_id]
                env.obs = env.env._get_observations()
                obs = env.obs[env.ground]
                selected = env.ground_controller._parse_simulator_observation(obs)['occupancy_map']
                env._ground_action([])
                if case_id == circle:
                    assert selected is obs['navigation_map'] and env.ground_controller.resolution == .1
                else:
                    assert selected is obs['occupancy_map'] and env.ground_controller.resolution == .5
                    assert selected.shape == (16, 16) and np.count_nonzero(selected > 0) == expected[case_id]
                    assert np.any(obs['ground_safety_map']['known_mask'] & (obs['ground_safety_map']['occupancy_map'] > 0))
                assert all(env.obs[drone]['map_resolution'] == 2. for drone in env.ids)
            finally:
                env.env.close()


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe']:
        probe()
    else:
        unittest.main()
