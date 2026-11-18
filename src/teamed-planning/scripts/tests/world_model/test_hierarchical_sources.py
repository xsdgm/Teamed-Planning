"""Current feature snapshots and historical resume select explicit layouts."""
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import train_hierarchical as trainer
from world_model.hierarchical_sources import (copy_source_files, package_python_files, runtime_source,
                                              snapshot_sources, source_hashes)


class SourceLayoutTests(unittest.TestCase):
    def test_current_snapshot_copies_feature_sources_and_keeps_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'scripts' / 'world_model'
            source.mkdir(parents=True)
            for name in ('__init__.py', 'train_hierarchical.py',
                         'hierarchical_sources.py', 'run_io.py', 'unrelated_model.py'):
                (source / name).write_text('# ' + name)
            (source / 'navigation').mkdir()
            (source / 'navigation' / 'hierarchical_runtime.py').write_text('# grouped runtime')
            (source / 'navigation' / 'scene_manifest.py').write_text('# grouped scene manifest')
            backend = source.parent / 'simulator' / 'wildscenes_map.py'
            backend.parent.mkdir()
            backend.write_text('# pinned geometry')
            before = source_hashes(source)
            target = root / 'run' / 'runtime_sources' / 'world_model'
            snapshot_sources(source, target, include_backend=True)
            self.assertEqual(source_hashes(source), before)
            self.assertFalse((target / 'unrelated_model.py').exists())
            self.assertEqual((target / '_wildscenes_map_backend.py').read_text(), '# pinned geometry')
            self.assertEqual((target / 'train_hierarchical.py').read_text(), '# train_hierarchical.py')
            self.assertEqual((target / 'hierarchical_runtime.py').read_text(), '# grouped runtime')
            self.assertEqual((target / 'scene_manifest.py').read_text(), '# grouped scene manifest')
            self.assertEqual((target / 'run_io.py').read_text(), '# run_io.py')
            self.assertEqual(source_hashes(target)['hierarchical_runtime.py'], before['hierarchical_runtime.py'])

    def test_complete_snapshot_preserves_grouped_files_and_public_import_names(self):
        source = Path(trainer.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'world_model'
            hashes = copy_source_files(source, target, package_python_files(source))
            # Survey snapshots also carry their simulator and navigation config.
            for name in ('simulator', 'common'):
                package = source.parent / name
                copy_source_files(package, target.parent / name, package_python_files(package))
            expected = ['terrain/frontier_model.py', 'survey/ab_planner.py', 'navigation/planning_graph.py',
                        'rssm/model.py', 'terrain/decoupled_runtime.py', 'navigation/scene_manifest.py']
            for name in expected:
                self.assertIn(name, hashes)
                self.assertEqual((target / name).read_bytes(), (source / name).read_bytes())
            # Exercise the historical public module names in a process with only the snapshot on its path.
            program = """
import importlib, sys
from pathlib import Path

root = Path(sys.argv[1]) / 'world_model'
sys.path.insert(0, str(root.parent))
for relative in sys.argv[2:]:
    module = importlib.import_module('world_model.' + Path(relative).stem)
    assert Path(module.__file__) == root / relative, module.__file__

# Exercise both import orders in the isolated package, without live sources on sys.path.
aliases = {
    'ab_training': 'ab_model', 'decoupled_training': 'decoupled_model',
    'navigation_scenes': 'scene_manifest', 'joint_rollouts': 'belief_rollouts',
    'subgoal_value': 'navigation_evidence', 'regional_data': 'goose_data',
    'sequence': 'models', 'run_io': 'hierarchical_sources',
}
importlib.invalidate_caches()
for name, canonical in aliases.items():
    old_module = importlib.import_module('world_model.' + name)
    new_module = importlib.import_module('world_model.' + canonical)
    assert old_module is new_module, name
    assert Path(old_module.__file__).is_relative_to(root), name
    assert new_module.__spec__.name == 'world_model.' + canonical
"""
            subprocess.run([sys.executable, '-I', '-B', '-c', program, directory, *expected], check=True)

    def test_ambiguous_layout_requires_manifest_and_declared_layout_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = [root / 'runtime_sources' / name for name in ('vtemp2', 'world_model')]
            old.mkdir(parents=True)
            self.assertEqual(runtime_source(root), old)
            new.mkdir()
            with self.assertRaises(ValueError):
                runtime_source(root)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(source_package='world_model')))
            self.assertEqual(runtime_source(root), new)
            new.rmdir()
            with self.assertRaisesRegex(ValueError, 'missing'):
                runtime_source(root)

    def test_resume_reenters_original_old_or_new_entrypoint(self):
        for package, entrypoint in [('vtemp2', 'train_graph.py'), ('world_model', 'train_hierarchical.py')]:
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                source = run / 'runtime_sources' / package
                source.mkdir(parents=True)
                entry = source / entrypoint
                entry.write_text('# historical source must remain untouched')
                checkpoint = run / 'checkpoints' / 'latest.pt'
                checkpoint.parent.mkdir()
                torch.save(dict(schema=trainer.TRAIN_SCHEMA, world_schema=trainer.MODEL_SCHEMA,
                    manifest=dict(fast_schema=trainer.FAST_SCHEMA, slow_schema=trainer.SLOW_SCHEMA)), checkpoint)
                arguments = ['train_hierarchical.py', '--output', str(run), '--resume']
                with patch.object(sys, 'argv', arguments):
                    args = trainer.parse_args()
                    with patch.object(trainer.os, 'execv', side_effect=RuntimeError('reexec captured')) as execute:
                        with self.assertRaisesRegex(RuntimeError, 'reexec captured'):
                            trainer.snapshot(args)
                self.assertEqual(execute.call_args.args[1][4], str(entry))
                self.assertIn('--frozen', execute.call_args.args[1])
                self.assertEqual(entry.read_text(), '# historical source must remain untouched')
                self.assertEqual([path.name for path in (run / 'runtime_sources').iterdir()], [package])


if __name__ == '__main__':
    unittest.main(verbosity=2)
