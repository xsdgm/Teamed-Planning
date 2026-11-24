"""The VTemp2 trainer freezes relocated extensions and resumes old layouts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vtemp2_3 import train_slow_policy as trainer
from vtemp2_3.hierarchical_sources import source_hashes

REPO = Path(__file__).resolve().parents[5]
ARCHIVE = REPO / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/runtime_sources'


def options(*, resume=False):
    return SimpleNamespace(frozen=False, resume=resume, evaluate=None, baseline=False)


def capture_launch(args, output):
    with patch.object(trainer.os, 'execv', side_effect=RuntimeError('launch captured')) as execute:
        with unittest.TestCase().assertRaisesRegex(RuntimeError, 'launch captured'):
            trainer.prepare_snapshot(args, trainer.DEFAULTS, REPO, output)
    return execute.call_args.args[1]


class SlowPolicySnapshotTests(unittest.TestCase):
    def test_new_snapshot_starts_with_relocated_sources_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            launch = capture_launch(options(), out)
            root = out / 'runtime_sources'
            source = root / 'world_model'
            self.assertEqual(launch[4], str(source / 'train_slow_policy.py'))
            self.assertTrue((root / 'ground/ground_policy.py').is_file())
            self.assertTrue((root / 'simulator/ground_evidence.py').is_file())
            self.assertTrue((root / 'simulator/ground_relay.py').is_file())
            self.assertFalse((source / 'ground_policy.py').exists())
            if ARCHIVE.is_dir():
                program = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from vtemp2_3.train_slow_policy import SCHEMA, slow_model
from vtemp2_3.frozen_navigation import runtime
runtime(Path(sys.argv[2]))
slow_model(SCHEMA, 16)
import vtemp2_3.ground_policy, simulator.ground_evidence
assert Path(ground.ground_policy.__file__).is_relative_to(Path(sys.argv[1]))
assert Path(simulator.ground_evidence.__file__).is_relative_to(Path(sys.argv[1]))
'''
                result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(root), str(REPO)],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (out / 'manifest.json').write_text(json.dumps({'source_sha256': source_hashes(source)}))
            resumed = capture_launch(options(resume=True), out)
            self.assertEqual(resumed[4], str(source / 'train_slow_policy.py'))
            self.assertIn('--resume', resumed)

    def test_resume_historical_flat_snapshot_uses_its_original_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            source = out / 'runtime_sources/world_model'
            source.mkdir(parents=True)
            for name in ('__init__.py', 'train_slow_policy.py', 'ground_policy.py'):
                (source / name).write_text('# historical ' + name)
            (out / 'manifest.json').write_text(json.dumps({'source_sha256': source_hashes(source)}))
            resumed = capture_launch(options(resume=True), out)
            self.assertEqual(resumed[4], str(source / 'train_slow_policy.py'))
            self.assertIn('--resume', resumed)
            self.assertEqual((source / 'ground_policy.py').read_text(), '# historical ground_policy.py')
            self.assertFalse((source.parent / 'ground').exists())


if __name__ == '__main__':
    unittest.main()
