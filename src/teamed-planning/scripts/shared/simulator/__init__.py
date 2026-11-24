"""Core 2D multi-robot simulation, scenarios, and WildScenes map backend."""
from pathlib import Path

_root = Path(__file__).resolve().parent
__path__.append(str(_root / "wildscenes"))
