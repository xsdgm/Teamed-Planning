"""Simulator package for multi-robot navigation.

Load public classes on demand so importing an experiment does not preload
mutable simulator code before a frozen runtime is selected.
"""
from importlib import import_module

_EXPORTS = {
    'MultiRobotEnv': ('.env', 'MultiRobotEnv'),
    'EnvConfig': ('.env', 'EnvConfig'),
    'GroundRobot': ('.robots', 'GroundRobot'),
    'Drone': ('.robots', 'Drone'),
    'CircleObstacle': ('.obstacles', 'CircleObstacle'),
    'RectObstacle': ('.obstacles', 'RectObstacle'),
    'create_forest_scenario': ('.obstacles', 'create_forest_scenario'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
