"""Compatibility of relocated VTemp4 modules and immutable experiment snapshots."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vtemp4 import evaluate_vtemp4_waypoints as evaluation
from vtemp4.train_vtemp4_formal import source_tree_digest
from vtemp2_3.hierarchical_sources import file_sha256

SCRIPTS = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[5]
FAMILIES = ('common', 'drone', 'ground', 'simulator', 'world_model')


def supervisor():
    file = SCRIPTS / 'vtemp4/run_vtemp4_formal.py'
    spec = importlib.util.spec_from_file_location('vtemp4_formal_supervisor_test', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MigrationContracts(unittest.TestCase):
    def test_legacy_and_new_imports_share_module_identity_in_both_orders(self):
        script = '''
import importlib, sys
sys.path.insert(0, sys.argv[1])
import world_model
pairs = [('world_model.' + n, 'ground.' + n) for n in world_model._VTEMP4_GROUND_MODULES]
pairs += [('world_model.' + n, 'simulator.' + n) for n in world_model._VTEMP4_SIMULATOR_MODULES]
pairs += [('world_model.' + n, 'ground.' + n) for n in world_model._VTEMP2_GROUND_MODULES]
pairs += [('world_model.' + n, 'simulator.' + n) for n in world_model._VTEMP2_SIMULATOR_MODULES]
for old, new in sorted(pairs):
    first, second = (old, new) if sys.argv[2] == 'old' else (new, old)
    assert importlib.import_module(first) is importlib.import_module(second), (old, new)
'''
        for order in ('old', 'new'):
            with self.subTest(order=order):
                result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(SCRIPTS), order],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_freeze_has_five_packages_and_digest_covers_each(self):
        run_formal = supervisor()
        self.assertEqual(run_formal.REPO, REPO)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            run_formal.freeze(REPO, out)
            sources = out / 'sources'
            self.assertTrue(all((sources / family).is_dir() for family in FAMILIES))
            self.assertEqual(run_formal.snapshot_package(sources), sources / 'simulator')
            # New snapshots contain their own supervisors and sibling imports;
            # resuming must not depend on the experiments documentation tree.
            for name in ('run_vtemp4_formal.py', 'run_vtemp4_continuous_history.py',
                         'run_vtemp4_waypoint_comparison.py'):
                result = subprocess.run([sys.executable, '-I', '-B',
                                         str(sources / 'simulator' / name), '--help'],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            baseline = source_tree_digest(sources)
            for family in FAMILIES:
                with self.subTest(family=family):
                    path = sources / family / '__init__.py'
                    original = path.read_bytes()
                    path.write_bytes(original + b'\n# digest mutation\n')
                    self.assertNotEqual(source_tree_digest(sources), baseline)
                    path.write_bytes(original)
            self.assertEqual(source_tree_digest(sources), baseline)
            (sources / 'simulator/train_vtemp4_formal.py').unlink()
            self.assertEqual(run_formal.snapshot_package(sources), sources / 'world_model')

    def test_checkpoint_info_selects_historical_and_new_snapshot_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            snapshot = run / 'sources'
            (snapshot / 'world_model').mkdir(parents=True)
            ckpt = run / 'checkpoints/model.pt'
            ckpt.parent.mkdir()
            ckpt.write_bytes(b'checkpoint')
            manifest = run / 'manifest.json'
            manifest.write_text(json.dumps({'world_model_sha256': 'source-hash', 'episodes_per_update': 16}))
            state = {'v9_sha256': evaluation.CHECKPOINT_SHA, 'manifest_sha256': file_sha256(manifest),
                     'completed_updates': 0, 'clock': {}, 'format': evaluation.FORMAT}
            with patch.object(evaluation.torch, 'load', return_value=state), \
                 patch.object(evaluation, 'source_digest', return_value='source-hash') as legacy, \
                 patch.object(evaluation, 'source_tree_digest', return_value='source-hash') as modern, \
                 patch.object(evaluation, 'checkpoint_contract', return_value=None), \
                 patch.object(evaluation, 'reject_training_overlap'):
                evaluation.checkpoint_info(ckpt, 'heading', [], True)
                legacy.assert_called_once_with(snapshot / 'world_model')
                modern.assert_not_called()
                (snapshot / 'simulator').mkdir()
                (snapshot / 'simulator/vtemp4_parallel.py').touch()
                evaluation.checkpoint_info(ckpt, 'heading', [], True)
                modern.assert_called_once_with(snapshot)


if __name__ == '__main__':
    unittest.main()
