"""Core package for the FunGrid training environment."""

from __future__ import annotations

from .cell_types import CellType, ObstacleSetupChoice
from .config import Config
from .entities import Agent
from .grid import GridEnvironment
from .main import main
from .rl import CoordPointerGRU, MLP, PPO, PPOAgent, RMSNorm1d, TransitionStorage
from .simulation import EnvironmentSimulation

try:  # Optional dependency; pygame might not be installed in headless setups.
    from .visualizer import GridVisualizer
except ModuleNotFoundError:  # pragma: no cover - depends on optional pygame install
    GridVisualizer = None  # type: ignore[assignment]

try:  # Preserve compatibility with legacy API if available.
    from .game import Game  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - legacy module may not exist
    Game = None  # type: ignore[assignment]

__all__ = [
    "Agent",
    "CellType",
    "Config",
    "CoordPointerGRU",
    "EnvironmentSimulation",
    "GridEnvironment",
    "MLP",
    "ObstacleSetupChoice",
    "PPO",
    "PPOAgent",
    "RMSNorm1d",
    "TransitionStorage",
    "main",
]

if GridVisualizer is not None:
    __all__.append("GridVisualizer")

if Game is not None:
    __all__.append("Game")
