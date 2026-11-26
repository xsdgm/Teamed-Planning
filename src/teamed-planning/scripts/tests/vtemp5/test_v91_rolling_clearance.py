import math
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS/'vtemp5'))
from shared.simulator.wildscenes import wildscenes_map
from vtemp2_3.wildscenes_runtime import filter_ground_motion
from v91_rolling_clearance import guard_motion, local_geometry, _cell_shifts


def observed_corner(heading=math.pi/2, fixed=False):
    tile = np.zeros((16,16), bool)
    tile[9,11] = True
    obs = dict(self=dict(position=[4.,4.], orientation=heading), time_step=1.,
               occupancy_map=tile, map_resolution=.5)
    if fixed:
        obs['ground_safety_map'] = dict(occupancy_map=tile, known_mask=np.ones_like(tile),
                                         resolution_m=.5, origin_xy=[0.,0.], map_axes='yx')
    return obs


class TestRollingClearance(unittest.TestCase):
    def test_phase_change_can_deadlock_after_ordinary_certified_motion(self):
        # The occupied square starts 1.581 m from the car. A small forward
        # motion is safe in this frame, but shifting the next raster by a
        # whole pixel would put the starting capsule exactly on its boundary.
        obs = observed_corner()
        action = dict(motion=(.12,0.))
        regular, _ = filter_ground_motion(action,obs,wildscenes_map)
        self.assertEqual(regular['motion'][0], .12)
        guarded, event = guard_motion(action,obs,wildscenes_map)
        self.assertEqual(guarded['motion'][0], 0.)
        self.assertTrue(event['rolling_phase_intervention'])
        geometry = local_geometry(obs, wildscenes_map)
        self.assertTrue(geometry.point_clear([4.,4.],1.5))
        self.assertFalse(geometry.segment_clear([4.,4.],[4.,4.12],1.5))
        self.assertTrue(geometry.segment_clear([4.,4.],[3.5,4.],1.5))

    def test_fixed_world_aligned_tile_preserves_native_guard(self):
        obs = observed_corner(fixed=True)
        for speed in (-.5,0.,.12,.9):
            a = dict(motion=(speed,.73))
            self.assertEqual(guard_motion(a,obs,wildscenes_map),
                             filter_ground_motion(a,obs,wildscenes_map))

    def test_never_licenses_unsupported_start_or_unknown_footprint(self):
        obs = observed_corner()
        obs['occupancy_map'][8,11] = True
        geom = local_geometry(obs,wildscenes_map)
        self.assertFalse(geom.point_clear([4.,4.],1.5))
        self.assertFalse(geom.segment_clear([4.,4.],[3.,4.],1.5))
        obs['sensor_valid'] = False
        result, _ = guard_motion(dict(motion=(.6,.4)),obs,wildscenes_map)
        self.assertEqual(result['motion'],(0.,.4))

    def test_negative_and_intermediate_grid_shifts(self):
        states = _cell_shifts(np.array([-2.25,.75]))
        self.assertEqual(states, {(i,j) for i in (0,-1,-2,-3) for j in (0,1)})
        self.assertEqual(_cell_shifts(np.array([0.,1.])),{(0,0),(0,1)})
