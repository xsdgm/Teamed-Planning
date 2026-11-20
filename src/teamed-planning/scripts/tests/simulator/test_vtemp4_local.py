"""Check the online obstacle stage's episode count, seed schedule and restart boundary."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulator import train_vtemp4_local as local


class OnlineTrainingPlanTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        archive = root / local.V9_RUN
        (archive / 'runtime_sources').mkdir(parents=True)
        (archive / 'runtime_sources/simulator.py').write_text('# archived runtime\n')
        (archive / 'config.json').write_text('{}')
        (archive / 'best.pt').write_bytes(b'archived policy')
        expected = local.file_sha256(archive / 'best.pt')
        self.enterContext(patch.object(local, 'CHECKPOINT_SHA', expected))
        output = root / 'online'
        output.mkdir()
        self.args = SimpleNamespace(repo=root, output=output, seed=43, updates=25,
                                    episodes_per_update=16, workers=6, learning_rate=.0003, resume=False)

    def test_four_hundred_distinct_online_missions(self):
        first = local.manifest_for(self.args)
        self.assertEqual(first['training_stage'], 1)
        self.assertEqual(first['total_episodes'], 400)
        self.assertEqual(len(first['plan']), 400)
        scene_seeds = {entry['scene_seed'] for entry in first['plan']}
        fast_seeds = {entry['fast_seed'] for entry in first['plan']}
        self.assertEqual(len(scene_seeds), 400)
        self.assertEqual(len(fast_seeds), 400)
        self.assertTrue(scene_seeds.isdisjoint(fast_seeds))
        self.args.workers = 2
        self.assertEqual(local.manifest_for(self.args), first)

    def test_resume_rejects_another_seed_before_touching_training(self):
        path = self.args.output / 'manifest.json'
        local.atomic_json(path, local.manifest_for(self.args))
        original = path.read_bytes()
        self.args.seed += 1
        self.args.resume = True
        with patch.object(local, 'runtime') as simulator:
            with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                local.run(self.args)
            simulator.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.args.output / 'checkpoints').exists())


if __name__ == '__main__':
    unittest.main()
