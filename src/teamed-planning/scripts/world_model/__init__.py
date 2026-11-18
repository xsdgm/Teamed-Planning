"""World-model navigation with shared dynamics and task-specific extensions."""

from importlib import import_module
from importlib.util import spec_from_loader
from pathlib import Path
import sys

# Public module names are also stored in historical model pickles. Keep the
# existing world_model.* imports across the source directories.
_root = Path(__file__).resolve().parent
__path__.extend(str(_root / family) for family in ("navigation", "rssm", "survey", "terrain"))

# Old imports in scripts and pickles must resolve to the very same module object.
# Real modules in the selected source tree always take precedence.
_MERGED_MODULES = {
    'ab_training': 'ab_model',
    'decoupled_training': 'decoupled_model',
    'joint_rollouts': 'belief_rollouts',
    'navigation_scenes': 'scene_manifest',
    'regional_data': 'goose_data',
    'run_io': 'hierarchical_sources',
    'sequence': 'models',
    'subgoal_value': 'navigation_evidence',
}


class _MergedModules:
    def find_spec(self, fullname, target=None):
        parent, _, name = fullname.rpartition('.')
        if parent == __name__ and name in _MERGED_MODULES:
            return spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        canonical = _MERGED_MODULES[name.rsplit('.', 1)[1]]
        sys.modules[name] = import_module('.' + canonical, __name__)


# A virtual final entry limits the finder to this package and source snapshot.
_alias_path = str(_root / '.merged-imports')
sys.path_importer_cache[_alias_path] = _MergedModules()
__path__.append(_alias_path)
