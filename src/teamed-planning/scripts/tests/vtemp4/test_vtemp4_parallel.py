"""Native smoke coverage for parallel VTemp4 rollouts and their resume contract."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, runtime
from vtemp2_3.hierarchical_sources import file_sha256
from vtemp4.train_vtemp4 import FORMAT
from vtemp4.vtemp4_parallel import collect_episode, episode_pool, load_rollout
from vtemp4.vtemp4_policy import HeadingPolicy


class HeadingParallelTests(unittest.TestCase):
    def test_spawn_reuse_and_repair_with_the_same_behaviour_policy(self):
        result = subprocess.run([sys.executable, '-I', '-B', __file__, 'isolated'], capture_output=True,
                                text=True, timeout=200)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_native(self):
        repo = Path(__file__).resolve().parents[5]
        runtime(repo)
        case = dict(seed=19, case_id='parallel-km', kind='v9_random', terrain='synthetic',
                    start=[250., 250.], goal=[1350., 250.], world_size=1650.,
                    direct_distance_m=1100., circles=[], circles_sha256='inline-test-map',
                    navigation_task='random_circles', terrain_family_id='parallel-test')
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            weights = output / 'policy.pt'
            torch.save(dict(model=HeadingPolicy().state_dict(), format=FORMAT, v9_sha256=CHECKPOINT_SHA,
                            clock=dict(review_interval_s=15, lookahead_m=200.)), weights)
            digest = file_sha256(weights)
            common = (repo, weights, digest, case, 529)
            with episode_pool(2) as pool:
                train = pool.submit(collect_episode, *common, output / 'train', 'train', 2)
                hold = pool.submit(collect_episode, *common, output / 'hold', 'always_hold', 2)
                row = train.result(timeout=180)
                self.assertEqual(row['end_reason'], 'collector_limit')
                self.assertEqual(hold.result(timeout=180)['turn_count'], 0)
                record_path = output / 'train' / 'episode.json'
                mtime = record_path.stat().st_mtime_ns
                record = json.loads(record_path.read_text())
                self.assertEqual(record['trajectory_xy'][0], case['start'])
                self.assertEqual(len(record['trajectory_xy']), 3)
                self.assertTrue(load_rollout(output / 'train')[-1]['truncated'])

                cached = pool.submit(collect_episode, *common, output / 'train', 'train', 2).result(timeout=30)
                self.assertEqual(cached, row)
                self.assertEqual(record_path.stat().st_mtime_ns, mtime)
                original = record_path.read_bytes()
                record_path.write_text(json.dumps(dict(record, progress_distance='traversable')))
                with self.assertRaisesRegex(ValueError, 'different policy, seed or mission'):
                    pool.submit(collect_episode, *common, output / 'train', 'train', 2).result()
                record_path.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'different policy, seed or mission'):
                    pool.submit(collect_episode, repo, weights, digest, case, 530, output / 'train', 'train', 2).result()
                with self.assertRaisesRegex(ValueError, 'Heading checkpoint digest'):
                    pool.submit(collect_episode, repo, weights, 'wrong', case, 529, output / 'train', 'train', 2).result()

                # A lost or damaged transition file leaves this batch recoverable.
                (output / 'train' / 'transitions.pt').write_bytes(b'incomplete')
                with self.assertRaisesRegex(ValueError, 'Rollout digest'):
                    load_rollout(output / 'train')
                repaired = pool.submit(collect_episode, *common, output / 'train', 'train', 2).result(timeout=180)
                self.assertEqual(repaired, row)
                self.assertEqual(len(load_rollout(output / 'train')), 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['isolated']:
        HeadingParallelTests().run_native()
    else:
        unittest.main()
