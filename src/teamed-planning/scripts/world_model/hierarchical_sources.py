"""Source inventories and frozen-layout compatibility for hierarchical navigation.

New runs use world_model. Historical vtemp2 snapshots remain byte-for-byte
unchanged and are imported under their original package and module names.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

PACKAGE_DIR = Path(__file__).resolve().parent
# Both src/teamed-planning/scripts/world_model and artifacts/<run>/runtime_sources/world_model
# have four directory components below the repository.
DEFAULT_REPO = (PACKAGE_DIR / '../../../..').resolve()

# The semantic keys retain their historical import names. Current packages alias
# merged modules, and older snapshots still need their original filenames hashed.
MODULE_MAP = {'analyze': 'hierarchical_analysis', 'cbf_safety': 'hierarchical_cbf', 'control_safety': 'hierarchical_safety', 'evaluate_fixed_target': 'evaluate_fixed_target', 'evaluate_repairs': 'evaluate_safety_repairs', 'evaluate_slow_control': 'evaluate_slow_policy', 'evidence': 'navigation_evidence', 'fast': 'graph_fast_policy', 'graph_context': 'safety_context', 'graph_data': 'graph_rollouts', 'graph_policy': 'hierarchical_policy', 'graph_world': 'graph_dynamics', 'hierarchy': 'hierarchical_runtime', 'joint_data': 'joint_rollouts', 'policy': 'slow_policy', 'rebatch': 'training_schedule', 'scene_manifest': 'scene_manifest', 'scenes': 'navigation_scenes', 'simplified_wildscenes_tasks': 'simplified_wildscenes_tasks', 'slow_graph': 'planning_graph', 'train': 'train_slow_policy', 'train_graph': 'train_hierarchical', 'train_joint': 'train_joint_policy', 'train_world': 'train_residual_world', 'validate': 'frozen_navigation', 'value': 'subgoal_value', 'wildscenes_runtime': 'wildscenes_runtime', 'wildscenes_tasks': 'wildscenes_tasks', 'wildscenes_workflow': 'wildscenes_workflow', 'wm_actor': 'belief_actor', 'wm_data': 'belief_rollouts', 'wm_model': 'residual_world', 'wm_protocol': 'telemetry'}
BACKEND_NAME = '_wildscenes_map_backend.py'


def _navigation_source(source, filename):
    # Hierarchical snapshots store public module filenames at package root.
    flat = source / filename
    return flat if flat.is_file() else source / 'navigation' / filename


def source_package(source):
    source = Path(source)
    if source.name == 'vtemp2' or ((source / 'hierarchy.py').is_file()
                                  and not _navigation_source(source, 'hierarchical_runtime.py').is_file()):
        return 'vtemp2'
    return 'world_model'


def source_files(source):
    """Select this feature's sources without copying unrelated world models."""
    source = Path(source)
    if source_package(source) == 'vtemp2':
        return sorted(source.glob('*.py'))
    names = {name + '.py' for name in MODULE_MAP.values()}
    # run_io.py is part of the hash inventory in earlier flat snapshots.
    names.update(('__init__.py', 'hierarchical_sources.py', 'run_io.py', BACKEND_NAME))
    paths = (_navigation_source(source, name) for name in sorted(names))
    return [path for path in paths if path.is_file()]


def atomic_json(path, value, *, ensure_ascii=False):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    stream = temporary.open('x', encoding='utf-8')
    try:
        with stream:
            json.dump(value, stream, ensure_ascii=ensure_ascii, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def package_python_files(root):
    """Relative names in a complete world_model package, including implementation families."""
    return sorted(path.relative_to(root) for path in root.rglob('*.py'))


def copy_source_files(root, destination, relative_paths):
    """Preserve the import layout and return hashes of the actual frozen copies."""
    hashes = {}
    for relative in relative_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
        hashes[str(relative)] = file_sha256(target)
    return hashes


def source_hashes(source):
    return {path.name: file_sha256(path) for path in source_files(source)}


def snapshot_sources(source, target, include_backend=False):
    """Copy selected current sources, or an unchanged historical package."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Snapshot target must differ from source')
    target.mkdir(parents=True, exist_ok=True)
    for path in source_files(source):
        shutil.copy2(path, target / path.name)
    if include_backend:
        backend = source / BACKEND_NAME
        if not backend.is_file():
            backend = source.parent / 'simulator' / 'wildscenes_map.py'
        if not backend.is_file():
            raise ValueError('Missing immutable raster backend for source snapshot')
        shutil.copy2(backend, target / BACKEND_NAME)
    return target


def runtime_source(run):
    """Resolve an explicit manifest layout, or a single historical layout."""
    run = Path(run).resolve()
    for filename in ('manifest.json', 'input-manifest.json'):
        manifest = run / filename
        if manifest.is_file():
            package = json.loads(manifest.read_text()).get('source_package')
            if package is not None:
                if package not in ('world_model', 'vtemp2'):
                    raise ValueError('Unsupported frozen source package: ' + str(package))
                source = run / 'runtime_sources' / package
                if not source.is_dir():
                    raise ValueError('Declared frozen source package is missing: ' + str(source))
                return source
    candidates = [run / 'runtime_sources' / name for name in ('world_model', 'vtemp2')]
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError('Run needs one frozen source package or an explicit source_package manifest')
    return candidates[0]


def load_runtime_modules(source, names=None):
    """Return stable semantic keys while selecting historical/current imports."""
    source = Path(source).resolve()
    package = source_package(source)
    for name, module in list(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError('Another ' + package + ' source was imported before frozen evaluation')
    sys.path.insert(0, str(source.parent))
    if names is None:
        names = ['hierarchy', 'graph_data', 'graph_policy', 'graph_world', 'fast', 'validate', 'scenes']
        safety = MODULE_MAP['control_safety'] if package == 'world_model' else 'control_safety'
        if _navigation_source(source, safety + '.py').is_file():
            names.append('control_safety')
    result = {}
    for key in names:
        name = MODULE_MAP.get(key, key) if package == 'world_model' else key
        module = importlib.import_module(package + '.' + name)
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Frozen module resolved outside its source snapshot: ' + key)
        result[key] = module
    return result
