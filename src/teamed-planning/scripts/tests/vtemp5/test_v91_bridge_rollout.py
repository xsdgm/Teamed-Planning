import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vtemp5'))
from v91_bridge_rollout import _decisions, _settle


def sample(step, count, duration):
    options = max(1, count)
    action = int(count > 1)
    branch = 'hold' if not count else ('left' if action else 'forward')
    obs = dict(branch_angles=np.arange(options) * .2, branch_prior=np.zeros(options))
    row = dict(step=step, action=action, branch=branch, turn=True, angle_rad=obs['branch_angles'][action],
               policy_controlled=count > 1, value=step + .5, logp=-math.log(options), observation=obs)
    review = dict(step=step, candidate_count=count, selected_branch=branch,
                  action_index=action, applied_target_frames=duration)
    return row, review


class BridgeCollectionTests(unittest.TestCase):
    def test_force_hold_and_singleton_keep_elapsed_time_and_terminal_rewards(self):
        samples, reviews = zip(sample(0, 2, 30), sample(30, 0, 30), sample(60, 1, 5))
        r = dict(reviews=reviews, termination='vehicle_arrival', last_step=65)
        rows = _decisions(r, samples)
        route = dict(first_deck_step=30, first_cross_step=31)
        _settle(rows, np.array([-100., -70., -60., -60.]), route, r['termination'], 120)
        np.testing.assert_allclose([s['reward'] for s in rows],
                                   [.6 - .25 + 1, .2 - .25 + 5, 10 - 5/120])
        self.assertEqual([s['policy_controlled'] for s in rows], [True, False, False])
        self.assertEqual([s['next_value'] for s in rows], [30.5, 60.5, 0.])
        self.assertEqual([s['terminated'] for s in rows], [False, False, True])
        self.assertFalse(any(s['truncated'] for s in rows))

    def test_mission_timeout_is_terminal_and_safety_excludes_unexecuted_proposal(self):
        for reason, terminal in [('frame_limit', -2.), ('safety_termination', -6.),
                                 ('ground_safety_violation', -6.)]:
            pair = [sample(0, 2, 30)]
            if reason != 'frame_limit':
                pair.append(sample(30, 2, 0))
            samples, reviews = zip(*pair)
            rows = _decisions(dict(reviews=reviews, termination=reason, last_step=30), samples)
            self.assertEqual(len(rows), 1)
            _settle(rows, np.zeros(2), {}, reason, 30)
            self.assertEqual(rows[0]['reward'], terminal - 1.)
            self.assertEqual(rows[0]['next_value'], 0.)

    def test_reject_mismatched_issued_branch_and_unattributed_frames(self):
        row, review = sample(0, 2, 30)
        for r in [dict(review, applied_target_frames=29), dict(review, action_index=0)]:
            with self.assertRaises(ValueError):
                _decisions(dict(reviews=[r], termination='frame_limit', last_step=30), [row])
        row, review = sample(0, 0, 2)
        with self.assertRaises(ValueError):
            _decisions(dict(reviews=[review], termination='vehicle_arrival', last_step=2), [dict(row, logp=-1.)])


if __name__ == '__main__':
    unittest.main()
