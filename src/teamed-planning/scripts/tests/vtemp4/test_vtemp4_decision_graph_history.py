"""Check timing and identity of past commands and the path driven under each."""
from pathlib import Path
import sys
import unittest

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp4.vtemp4_decision_graph_history import COMMAND_FEATURES, DRIVE_FEATURES, CommandHistory, FullCommandHistory


def issued(time, origin, target):
    return dict(issued_at_s=time, origin_xy=origin, target=target)


class DecisionHistoryTest(unittest.TestCase):
    def test_same_tick_outputs_are_invisible_and_local_point_can_cross_upper_switch(self):
        for history_type in (CommandHistory, FullCommandHistory):
            with self.subTest(history_type=history_type):
                self.check_review_timing(history_type())

    def check_review_timing(self, history):
        poses = [[x, 0.] for x in [0., 1., 2., 1., 0., 1., 2., 3., 4.]]
        # Reissuing the very same world point makes a distinct command.
        commands = [issued(0, poses[0], [10., 0.]), issued(5, poses[5], [10., 0.]),
                    issued(8, poses[8], [500., 500.])]
        ground = [dict(time_s=1, target=[2., 0.]), dict(time_s=6, target=[8., 0.]),
                  dict(time_s=8, target=[500., 500.])]

        def observe(t):
            # Mimic the post-fusion callback where step() has not appended pose(t).
            return history.observation(t, poses[t], [20., 0.], 10., poses[:t], commands, ground)

        empty = observe(0)
        self.assertEqual(empty['command_nodes'].shape, (0, COMMAND_FEATURES))
        self.assertEqual(empty['drive_segments'].shape, (0, DRIVE_FEATURES))
        at_five = observe(5)
        self.assertEqual(len(at_five['command_nodes']), 1)
        np.testing.assert_allclose(at_five['command_nodes'][0], [-.1, 0., .9, 0., 0., 0.,
                                                                np.log1p(5), np.log1p(5), .1, .1, 1.])
        # The first command drove 5 m despite ending only 1 m nearer its target.
        np.testing.assert_allclose(at_five['drive_segments'][:, 9], [.1, .4])
        np.testing.assert_equal(at_five['drive_segments'][:, 6], [0., 1.])

        final = observe(8)
        np.testing.assert_equal(final['command_predecessor'], [-1, 0])
        prior, active = final['command_nodes']
        np.testing.assert_equal([prior[-1], active[-1]], [0., 1.])
        np.testing.assert_allclose(prior[2:4], active[2:4])
        np.testing.assert_allclose(prior[8:10], [.1, .1])
        np.testing.assert_allclose(active[8:10], [.3, .3])
        # The final move into t=5 belongs to the first command. The earlier
        # ground target stays effective for a further second under the second command.
        np.testing.assert_equal(final['drive_command'], [0, 0, 1, 1])
        np.testing.assert_allclose(final['drive_segments'][2, :7], [-.3, 0., -.2, 0., -.2, 0., 1.])
        np.testing.assert_allclose(final['drive_segments'][3, :7], [-.2, 0., 0., 0., .4, 0., 1.])
        repeated = observe(8)
        for key in final:
            np.testing.assert_array_equal(final[key], repeated[key])

    def test_revisiting_a_place_recalls_earlier_commands_beyond_recent_window(self):
        poses = [[float(x), 0.] for x in [0, 200, *range(400, 1900, 100), 0]]
        now = len(poses) - 1
        commands = [issued(t, position, [position[0] + 10., 0.]) for t, position in enumerate(poses[:-1])]
        result = CommandHistory().observation(now, poses[-1], [100., 0.], 1., poses, commands, [])
        recalled = result['command_nodes'][:, 0].tolist()
        self.assertIn(0., recalled)
        self.assertNotIn(400., recalled)
        self.assertEqual(recalled[-8:], [position[0] for position in poses[-9:-1]])
        index = len(recalled) - 8
        self.assertEqual(result['command_predecessor'][index], -1)
        np.testing.assert_equal(result['command_predecessor'][index + 1:], np.arange(index, len(recalled) - 1))

    def test_complete_history_keeps_every_issue_and_its_drive_segments_on_revisits(self):
        history = FullCommandHistory()
        # Pass many distinct locations before returning to the starting point.
        # Every command points to the same coordinates but has its own issue time.
        poses = [[100. * t, 8.] for t in range(48)] + [[0., 8.]]
        commands = [issued(t, poses[t], [40., 55.]) for t in range(0, 49, 2)]
        ground = [dict(time_s=t, target=[x, 0.]) for t, x in ((0, 1.), (17, 2.), (48, 999.))]
        for t in (0, 5, 17, 24, 48):
            result = history.observation(t, poses[t], np.array([1000., 0.]), 200., poses[:t], commands, ground)
            nodes = result['command_nodes']
            count = sum(c['issued_at_s'] < t for c in commands)
            self.assertEqual(len(nodes), count)
            np.testing.assert_array_equal(result['command_predecessor'], np.arange(-1, count - 1))
            if t == 17:
                saved = nodes
                snapshot = nodes.copy()
        self.assertEqual(count, 24)
        np.testing.assert_array_equal(saved, snapshot)  # Later reviews cannot mutate a saved PPO observation.
        np.testing.assert_allclose(nodes[:, :2], (np.array(poses[:48:2]) - poses[-1]) / 200.)
        np.testing.assert_allclose(nodes[:, 6], np.log1p(48 - np.arange(0, 48, 2)), atol=5e-7)
        np.testing.assert_equal(nodes[:, -1], np.arange(24) == 23)
        owners = result['drive_command']
        np.testing.assert_array_equal(owners, np.sort(np.r_[np.arange(24), 8]))
        # The replan at t=17 splits the t=16 command, not the next one.
        np.testing.assert_allclose(result['drive_segments'][owners == 8, 4], [1. / 200., 2. / 200.])
        np.testing.assert_allclose(result['drive_segments'][-1, 4], 2. / 200.)


if __name__ == '__main__':
    unittest.main()
