"""Ground robot control and planning module.

Load public classes on demand so importing an experimental policy does not
preload mutable navigation code before a frozen runtime is selected.
"""
from importlib import import_module

_EXPORTS = {
    'GroundRobotController': ('.ground_main', 'GroundRobotController'),
    'LocalPerception': ('.local_planner', 'LocalPerception'),
    'GraphFusion': ('.graph_fusion', 'GraphFusion'),
    'GlobalPlanner': ('.global_planner', 'GlobalPlanner'),
    'LocalPlanner': ('.local_planner', 'LocalPlanner'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
