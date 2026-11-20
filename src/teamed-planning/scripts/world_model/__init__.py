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

# Historical vtemp4 checkpoints may refer to the former world_model module
# names. Resolve those names to the relocated implementations without loading
# a second copy of the module or moving navigation policy code back here.
_VTEMP4_GROUND_MODULES = {'vtemp4_observation', 'vtemp4_policy', 'vtemp4_waypoint_policy'}
_VTEMP4_SIMULATOR_MODULES = {
    'vtemp4_curriculum', 'vtemp4_lake_metrics', 'vtemp4_local_curriculum',
    'vtemp4_parallel', 'vtemp4_periodic_history', 'vtemp4_ring_contract',
    'vtemp4_ring_heading', 'vtemp4_ring_history', 'vtemp4_runtime',
    'vtemp4_stage_eval', 'vtemp4_unified_attention', 'vtemp4_waypoint_runtime',
    'train_vtemp4', 'train_vtemp4_formal', 'train_vtemp4_local',
    'evaluate_vtemp4_continuous_history', 'evaluate_vtemp4_waypoints',
}
_VTEMP2_GROUND_MODULES = {'ground_policy', 'survey_fleet_clearance'}
_VTEMP2_SIMULATOR_MODULES = {
    'ground_evidence', 'ground_relay', 'survey_fleet_drive', 'survey_fleet_metrics',
    'survey_fleet_observation', 'survey_fleet_phase', 'survey_fleet_runtime',
    'survey_km_maps', 'survey_pretrain_env', 'survey_pretrain_maps', 'survey_pretrain_policy',
    'train_survey_fleet', 'train_survey_pretrain', 'evaluate_survey_fleet',
}


class _MergedModules:
    def find_spec(self, fullname, target=None):
        parent, _, name = fullname.rpartition('.')
        if parent == __name__ and name in (_MERGED_MODULES.keys() | _VTEMP4_GROUND_MODULES | _VTEMP4_SIMULATOR_MODULES | _VTEMP2_GROUND_MODULES | _VTEMP2_SIMULATOR_MODULES):
            return spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        short_name = name.rsplit('.', 1)[1]
        if short_name in (_VTEMP4_GROUND_MODULES | _VTEMP2_GROUND_MODULES):
            canonical = 'ground.' + short_name
        elif short_name in (_VTEMP4_SIMULATOR_MODULES | _VTEMP2_SIMULATOR_MODULES):
            canonical = 'simulator.' + short_name
        else:
            canonical = __name__ + '.' + _MERGED_MODULES[short_name]
        sys.modules[name] = import_module(canonical)


# A virtual final entry limits the finder to this package and source snapshot.
_alias_path = str(_root / '.merged-imports')
sys.path_importer_cache[_alias_path] = _MergedModules()
__path__.append(_alias_path)
