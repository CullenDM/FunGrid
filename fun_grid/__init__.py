"""Core package for the FunGrid training environment."""

from __future__ import annotations

from .cell_types import CellType, ObstacleSetupChoice
from .config import Config
from .entities import Agent, GameAgent, GameEntity, GameFood, GameObstacle
from .grid import GridEnvironment, SpatialGrid
from .main import main
from .rl import CoordPointerGRU, MLP, PPO, PPOAgent, RMSNorm1d, TransitionStorage
from .simulation import EnvironmentSimulation
from .game import Game

try:  # Optional dependency; pygame might not be installed in headless setups.
    from .visualizer import GridVisualizer
except ModuleNotFoundError:  # pragma: no cover - depends on optional pygame install
    GridVisualizer = None  # type: ignore[assignment]

__all__ = [
    "Agent",
    "CellType",
    "Config",
    "CoordPointerGRU",
    "EnvironmentSimulation",
    "Game",
    "GameAgent",
    "GameEntity",
    "GameFood",
    "GameObstacle",
    "GridEnvironment",
    "MLP",
    "ObstacleSetupChoice",
    "PPO",
    "PPOAgent",
    "SpatialGrid",
    "RMSNorm1d",
    "TransitionStorage",
    "main",
]

if GridVisualizer is not None:
    __all__.append("GridVisualizer")
