"""Ground controller selection of vtemp rolling waypoints and vtemp1 lookahead."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ground.ground_main import GroundRobotController


class GuidanceStrategyTests(unittest.TestCase):
    def test_strategy_controls_the_live_vehicle_broadcast(self):
        def broadcast(car, x, now):
            observation = dict(self=dict(position=[x, 500., 0.], orientation=0.),
                               neighbors=[], occupancy_map=np.zeros((16, 16), np.uint8),
                               map_resolution=.5, full_state=dict(time=now))
            return car.get_action(observation)['broadcast']['target']

        for strategy, expected_after_review in [('vtemp', 300.), ('vtemp1', 310.)]:
            with self.subTest(strategy=strategy), contextlib.redirect_stdout(io.StringIO()):
                car = GroundRobotController('test', subgoal_radius=200., guidance_strategy=strategy)
                car.set_goal((800., 500.))
                np.testing.assert_allclose(broadcast(car, 100., 0.), (300., 500.))
                np.testing.assert_allclose(broadcast(car, 110., 15.), (expected_after_review, 500.))
                if strategy == 'vtemp':
                    np.testing.assert_allclose(broadcast(car, 275., 16.), (475., 500.))
                self.assertEqual(car.final_goal, (800., 500.))

    def test_unrecognized_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'guidance_strategy'):
            GroundRobotController('test', subgoal_radius=200., guidance_strategy='unused')


if __name__ == '__main__':
    unittest.main()
