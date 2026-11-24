"""Frozen evaluation loads one explicit package without mixing live modules."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[2] / 'world_model' / 'hierarchical_sources.py'
BOOTSTRAP = '''
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('source_helper', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
source = pathlib.Path(sys.argv[2])
modules = helper.load_runtime_modules(source, ('hierarchy', 'graph_policy'))
assert {key: value.MARKER for key, value in modules.items()} == {'hierarchy': 'runtime', 'graph_policy': 'policy'}
assert all(pathlib.Path(value.__file__).is_relative_to(source) for value in modules.values())
if len(sys.argv) > 3:
    try:
        helper.load_runtime_modules(pathlib.Path(sys.argv[3]), ('hierarchy',))
    except RuntimeError as error:
        assert 'imported before frozen evaluation' in str(error)
    else:
        raise AssertionError('Mixed source packages were accepted')
'''


class SourceLoaderTests(unittest.TestCase):
    def test_semantic_keys_load_both_layouts_and_reject_mixed_sources(self):
        for package, runtime_name, policy_name in (
                ('vtemp2', 'hierarchy', 'graph_policy'),
                ('world_model', 'hierarchical_runtime', 'hierarchical_policy')):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources = [root / name / package for name in ('original', 'other')]
                for source in sources:
                    source.mkdir(parents=True)
                    (source / '__init__.py').write_text('')
                    (source / (runtime_name + '.py')).write_text("MARKER = 'runtime'\n")
                    (source / (policy_name + '.py')).write_text("MARKER = 'policy'\n")
                result = subprocess.run([sys.executable, '-I', '-B', '-c', BOOTSTRAP,
                    str(HELPER), *(str(source) for source in sources)],
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_merged_module_keys_preserve_source_identity_in_new_and_earlier_snapshots(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('inventory', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for merged in (False, True):
                with self.subTest(merged=merged):
                    layout = root / ('merged' if merged else 'earlier')
                    source = layout / 'src' / 'world_model'
                    source.mkdir(parents=True)
                    (source / '__init__.py').write_bytes(HELPER.with_name('__init__.py').read_bytes()
                                                        if merged else b'')
                    previous = {'joint_data': 'joint_rollouts', 'value': 'subgoal_value',
                                'scenes': 'navigation_scenes'}
                    current = {'joint_data': 'belief_rollouts', 'value': 'navigation_evidence',
                               'scenes': 'scene_manifest'}
                    names = set(current.values())
                    if not merged:
                        names.update(previous.values())
                    for name in names:
                        (source / (name + '.py')).write_text(f'MARKER = {name!r}\n')
                    frozen = layout / 'runtime_sources' / 'world_model'
                    helper.snapshot_sources(source, frozen)
                    hashes = helper.source_hashes(frozen)
                    self.assertEqual(hashes, helper.source_hashes(source))
                    self.assertEqual(set(hashes), {name + '.py' for name in names} | {'__init__.py'})
                    expected = current if merged else previous
                    bootstrap = """
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('inventory', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
source = pathlib.Path(sys.argv[2])
expected = json.loads(sys.argv[3])
modules = helper.load_runtime_modules(source, expected)
assert {key: module.MARKER for key, module in modules.items()} == expected
assert all(pathlib.Path(module.__file__).parent == source for module in modules.values())
"""
                    subprocess.run([sys.executable, '-I', '-B', '-c', bootstrap, str(HELPER),
                                    str(frozen), json.dumps(expected)], check=True)

    def test_snapshot_keeps_all_historical_source_names_and_bytes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('source_helper_snapshot', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'old' / 'vtemp2'
            source.mkdir(parents=True)
            for name, content in {'__init__.py': '', 'hierarchy.py': 'from .validate import runtime\n',
                                  'validate.py': 'runtime = 1\n', 'unlisted.py': '# historical helper\n'}.items():
                (source / name).write_text(content)
            expected = helper.source_hashes(source)
            target = root / 'copy' / 'vtemp2'
            helper.snapshot_sources(source, target)
            self.assertEqual(helper.source_hashes(target), expected)
            self.assertEqual(helper.source_hashes(source), expected)
            self.assertFalse((target / 'hierarchical_runtime.py').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
