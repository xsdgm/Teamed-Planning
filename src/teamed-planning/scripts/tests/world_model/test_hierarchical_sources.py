"""Current feature snapshots and historical resume select explicit layouts."""
from importlib import import_module
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
                                              atomic_checkpoint, snapshot_sources, source_hashes)


class SourceLayoutTests(unittest.TestCase):
    def test_checkpoint_adapters_keep_historical_keyword_arguments(self):
        from world_model import abstract_training, semantic_training, train, train_slow_policy, ab_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writers = ((train.atomic_checkpoint, 'value'), (train_slow_policy.atomic_checkpoint, 'data'),
                       (abstract_training.checkpoint_save, 'document'),
                       (semantic_training.save_checkpoint, 'document'))
            for index, (writer, keyword) in enumerate(writers):
                with self.subTest(writer=writer, keyword=keyword):
                    path = root / f'{index}.pt'
                    writer(path=path, **{keyword: {'completed_updates': 1}})
                    self.assertEqual(torch.load(path, weights_only=True), {'completed_updates': 1})
            ab_model.atomic_json(path=root / 'status.json', document={'state': 'saved'})
            self.assertEqual(json.loads((root / 'status.json').read_text()), {'state': 'saved'})

    def test_shared_checkpoint_writer_preserves_previous_state_after_failed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            atomic_checkpoint(path, {'completed_updates': 1, 'state': torch.tensor([2.])})
            previous = path.read_bytes()

            def fail_mid_write(value, temporary):
                Path(temporary).write_bytes(b'incomplete checkpoint')
                raise OSError('injected disk write failure')

            with patch.object(torch, 'save', side_effect=fail_mid_write):
                with self.assertRaisesRegex(OSError, 'injected disk write failure'):
                    atomic_checkpoint(path, {'completed_updates': 2})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(torch.load(path, weights_only=True)['completed_updates'], 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_shared_checkpoint_writer_preserves_historical_torch_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            value = {'completed_updates': 1, 'state': torch.tensor([2.])}
            for append_suffix, stream in ((False, False), (True, False), (True, True)):
                with self.subTest(append_suffix=append_suffix, stream=stream):
                    temporary = path.with_suffix(path.suffix + '.tmp' if append_suffix else '.tmp')
                    if stream:
                        with temporary.open('wb') as handle:
                            torch.save(value, handle)
                    else:
                        torch.save(value, temporary)
                    expected = temporary.read_bytes()
                    temporary.unlink()
                    atomic_checkpoint(path, value, append_suffix=append_suffix, stream=stream)
                    self.assertEqual(path.read_bytes(), expected)

    def test_relocated_ground_extensions_are_frozen_with_matching_hashes(self):
        source = trainer.SOURCE
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'world_model'
            snapshot_sources(source, target)
            self.assertEqual(source_hashes(source), source_hashes(target))
            for family, name in (('ground', 'ground_policy.py'),
                                 ('simulator', 'ground_evidence.py'),
                                 ('simulator', 'ground_relay.py')):
                self.assertEqual((target.parent / family / name).read_bytes(),
                                 (source.parent / family / name).read_bytes())
                self.assertTrue((target.parent / family / '__init__.py').is_file())
            program = ("import importlib, sys; sys.path.insert(0, sys.argv[1]); "
                       "assert importlib.import_module('world_model.ground_policy') is "
                       "importlib.import_module('ground.ground_policy')")
            subprocess.run([sys.executable, '-I', '-B', '-c', program, directory], check=True)

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
            (source / 'train_vtemp3.py').write_text('# VTemp3 trainer')
            backend = source.parent / 'simulator' / 'wildscenes_map.py'
            backend.parent.mkdir()
            backend.write_text('# pinned geometry')
            before = source_hashes(source)
            target = root / 'run' / 'runtime_sources' / 'world_model'
            snapshot_sources(source, target, include_backend=True)
            self.assertEqual(source_hashes(source), before)
            self.assertFalse((target / 'unrelated_model.py').exists())
            self.assertFalse((target / 'train_vtemp3.py').exists())
            self.assertEqual((target / '_wildscenes_map_backend.py').read_text(), '# pinned geometry')
            self.assertEqual((target / 'train_hierarchical.py').read_text(), '# VTemp3 trainer')
            self.assertEqual((target / 'hierarchical_runtime.py').read_text(), '# grouped runtime')
            self.assertEqual((target / 'scene_manifest.py').read_text(), '# grouped scene manifest')
            self.assertEqual((target / 'run_io.py').read_text(), '# run_io.py')
            after = source_hashes(target)
            after.pop('_wildscenes_map_backend.py')
            self.assertEqual(after, before)

    def test_complete_snapshot_preserves_grouped_files_and_public_import_names(self):
        source = trainer.SOURCE
        self.assertIs(trainer, import_module('world_model.train_vtemp3'))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'world_model'
            hashes = copy_source_files(source, target, package_python_files(source))
            # Survey snapshots also carry their simulator and navigation config.
            for name in ('simulator', 'common'):
                package = source.parent / name
                copy_source_files(package, target.parent / name, package_python_files(package))
            expected = ['terrain/frontier_model.py', 'survey/ab_planner.py', 'navigation/planning_graph.py',
                        'rssm/model.py', 'terrain/decoupled_runtime.py', 'navigation/scene_manifest.py',
                        'navigation/graph_rollouts.py', 'train_vtemp3.py']
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
    'train_hierarchical': 'train_vtemp3',
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
