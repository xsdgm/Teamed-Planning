"""Shared world-model training entry, selected by algorithm configuration.

Example: python -I -B simulator/train_world_model.py --algorithm vtmp3
         --output /path/to/new/run --device cuda
Additional options override the selected task pipeline's configuration.
"""
import argparse
import json
import os
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from world_model.models import ARCHITECTURES

PIPELINES = {'vtmp2': 'mission_pipeline.py', 'vtmp3': 'ab_pipeline.py'}


def build_command(config_path, overrides=(), algorithm=None):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    selected = config.get('algorithm')
    if selected not in PIPELINES or config.get('model_architecture') != ARCHITECTURES[selected]:
        raise ValueError('Configuration must declare a supported algorithm and matching model_architecture')
    if algorithm is not None and algorithm != selected:
        raise ValueError('--algorithm disagrees with the selected configuration')
    parameters = config.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('Configuration parameters must be an object')
    arguments = []
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.replace('_', '').isalnum():
            raise ValueError('Invalid pipeline option name')
        if value is None or value is False:
            continue
        option = '--'+key.replace('_', '-')
        if value is True:
            arguments.append(option)
            continue
        if not isinstance(value, (str, int, float)):
            raise ValueError('Pipeline option values must be scalars')
        if key in ('v11_config', 'wildscenes_manifest'):
            value = str((path.parent/value).resolve())
        arguments.extend((option, str(value)))
    return [sys.executable, '-I', '-B', str(SCRIPTS/'world_model'/PIPELINES[selected]),
            *arguments, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=tuple(PIPELINES))
    parser.add_argument('--config', type=Path)
    args, remaining = parser.parse_known_args()
    selected = args.algorithm or 'vtmp3'
    config = args.config or SCRIPTS.parent/'config/algorithm'/f'world_model_{selected}.json'
    command = build_command(config, remaining, args.algorithm)
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
