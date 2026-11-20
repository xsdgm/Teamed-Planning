"""Compatibility entry point for VTemp3 hierarchical training."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import train_vtemp3

if __name__ == '__main__':
    train_vtemp3.main()
else:
    sys.modules[__name__] = train_vtemp3
