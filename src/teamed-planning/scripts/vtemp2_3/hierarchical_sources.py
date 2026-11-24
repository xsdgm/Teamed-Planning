"""Source inventories and frozen-layout compatibility for hierarchical navigation.

VTemp3 source modules keep the flat world_model package in run snapshots.
Historical vtemp2 snapshots are imported under their original package and
module names.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
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
    flat = source / filename
    return flat if flat.is_file() else source / 'navigation' / filename


def source_package(source):
    source = Path(source)
    if source.name == 'vtemp2' or ((source / 'hierarchy.py').is_file()
                                  and not _navigation_source(source, 'hierarchical_runtime.py').is_file()):
        return 'vtemp2'
    return 'world_model'


def source_paths(source):
    """Map frozen basenames to the files supplying their source."""
    source = Path(source)
    if source_package(source) == 'vtemp2':
        return {path.name: path for path in sorted(source.glob('*.py'))}
    names = {name + '.py' for name in MODULE_MAP.values()}
    # run_io.py is part of the hash inventory in earlier flat snapshots.
    names.update(('__init__.py', 'hierarchical_sources.py', 'run_io.py', 'partial_observation.py',
                  'ground_evidence.py', 'ground_policy.py', 'ground_relay.py', BACKEND_NAME))
    paths = {name: _navigation_source(source, name) for name in sorted(names)}
    # New source trees keep vehicle/simulation extensions in their own packages.
    # Flat snapshots retain their original files and keep taking precedence.
    for name, family in (('ground_policy.py', 'ground'),
                         ('ground_evidence.py', 'simulator'), ('ground_relay.py', 'simulator')):
        if not paths[name].is_file():
            paths[name] = source.parent / family / name
    for family in ('ground', 'simulator'):
        saved_init = source / family / '__init__.py'
        current_init = source.parent / family / '__init__.py'
        if saved_init.is_file():
            paths[family + '/__init__.py'] = saved_init
        elif current_init.is_file() and source.name == 'world_model' and (source / 'navigation').is_dir():
            paths[family + '/__init__.py'] = current_init
    trainer = source / 'train_vtemp3.py'
    if trainer.is_file():
        paths['train_hierarchical.py'] = trainer
    return {name: path for name, path in paths.items() if path.is_file()}


def source_files(source):
    """Select this feature's physical sources without unrelated world models."""
    return list(source_paths(source).values())


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


def atomic_checkpoint(path, value, *, append_suffix=False, stream=False):
    """Publish a complete, durable Torch checkpoint without exposing partial writes."""
    import torch

    path = Path(path)
    # Torch uses the filename stem for the ZIP archive's internal root. Keep
    # both historical temporary suffix conventions and optional file-object
    # writing so consolidation does not change serialized checkpoint bytes.
    temporary = path.with_suffix(path.suffix + '.tmp' if append_suffix else '.tmp')
    if temporary == path:
        temporary = path.with_name(path.name + '.tmp')
    try:
        if stream:
            with temporary.open('wb') as handle:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            torch.save(value, temporary)
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
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
    return {name: file_sha256(path) for name, path in source_paths(source).items()}


def snapshot_sources(source, target, include_backend=False):
    """Copy selected current sources, or an unchanged historical package."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Snapshot target must differ from source')
    target.mkdir(parents=True, exist_ok=True)
    paths = source_paths(source)
    for name, path in paths.items():
        # New layouts resolve these aliases to the sibling package.  Keeping a
        # second flat copy would create a distinct Python module in snapshots.
        relocated = {'ground_policy.py': 'ground', 'ground_evidence.py': 'simulator',
                     'ground_relay.py': 'simulator'}
        if name in relocated and path.parent.name == relocated[name]:
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    # Relocated modules use their public ground/simulator names. Preserve those
    # imports alongside the flat compatibility copies in a newly frozen run.
    for name, family in (('ground_policy.py', 'ground'),
                         ('ground_evidence.py', 'simulator'), ('ground_relay.py', 'simulator')):
        path = paths.get(name)
        extension = source.parent / family / name
        if path is not None and (path.parent.name == family or (family + '/__init__.py' in paths and extension.is_file())):
            path = extension if extension.is_file() else path
            destination = target.parent / family
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / name)
            shutil.copy2(paths[family + '/__init__.py'], destination / '__init__.py')
    if include_backend:
        backend = source / BACKEND_NAME
        if not backend.is_file():
            backend = source.parent / 'shared' / 'simulator' / 'wildscenes' / 'wildscenes_map.py'
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
