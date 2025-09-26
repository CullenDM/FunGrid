"""Configuration defaults and user override handling for FunGrid."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

try:
    import torch
except Exception:  # pragma: no cover - torch is an optional runtime dep for config inspection
    torch = None

_DEFAULT_DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


@dataclass(slots=True)
class Config:
    """Default hyperparameters and feature toggles for the training stack."""

    # --- Environment settings ---
    NUM_ENVS: int = 1
    CELL_SIZE: int = 10
    ENVIRONMENT_SIZE: int = 99
    MIN_EMPTY_PERCENTAGE: float = 0.5
    VARIABLE_OBSTACLE_COUNT: bool = False
    SET_FOOD_COUNT: int = 10
    VARIABLE_FOOD_COUNT: bool = False
    SCALE_FOOD_COUNT: bool = True
    FOOD_TICK: bool = True
    USE_RANDOM_FOOD_TICK: bool = False
    FOOD_TICK_SPEED: int = 4
    MAX_FOOD_TICK_SPEED: int = 4
    USE_FOOD_MULTIPLIER: bool = True
    USE_STEP_MULTIPLIER: bool = False
    USE_SINCE_LAST_FOOD_MULTIPLIER: bool = False
    OBSTACLE_CHOICE: int = 5
    USE_FPS: bool = False
    FPS: int = 60
    SAVE_TRANSITIONS: bool = False
    SAVE_FREQUENCY: int = 5
    TRANSITIONS_FILE_PATH: str = "transitions.pkl"
    USE_PANEL: bool = False
    DRAW_AGENT_PATH: bool = False
    HIGHLIGHT_MOVED_OBSTACLES: bool = True
    HIGHLIGHT_PLACED_OBSTACLES: bool = True
    VISUALIZE: bool = True

    # --- Training settings ---
    GAMMA: float = 0.99
    TAU: float = 0.95
    TARGET_KL: float = 0.01
    PPO_EPOCHS: int = 4
    LEARNING_RATE: float = 3e-4
    ENTROPY_COEFFICIENT: float = 0.02
    CLIP_PARAM: float = 0.2
    MODEL_GRAD_CLIP_NORM: float = 1.0
    DEVICE: str = _DEFAULT_DEVICE

    # PPO behaviour toggles
    MINIBATCH_SIZE: int = 50
    UPDATE_FREQUENCY: int = 250
    USE_PPO_UPDATE: bool = True
    PPO_TOGETHER: bool = True
    USE_EARLY_STOPPING: bool = False
    PPO_HALTING_COEF: float = 0.0
    PPO_PONDER_PENALTY: float = 0.0
    PPO_SELECTOR_ENT_COEF: float = 0.0
    MODEL_MAX_SEGMENTS: int = 1

    # Agent and simulation
    NUM_AGENTS: int = 1
    USE_GLOBAL_STATE: bool = False
    MAX_INVENTORY: int = 5
    DEBUG: bool = False

    # View/state dimensions
    STATE_DIM: int = 13
    GRID_SIZE: int = 11
    SEQUENCE_LENGTH: int = 20

    # Action space
    NUM_ACTIONS: int = 3
    NUM_DIRECTIONS: int = 4

    # Model persistence
    LOAD_MODEL: bool = True
    MODEL_NAME: str = "ms_pointer_001"
    MODEL_PATH: str = "./models/ppo_model_ms_pointer_001.pth"

    # Optional external model hook
    USER_MODEL_MODULE: str = "fun_grid.user_model"
    USER_MODEL_CLASS: str | None = None

    # Model hyperparameters (default implementation)
    MODEL_CELL_EMB_DIM: int = 32
    MODEL_STEP_DIM: int = 64
    MODEL_GRU_HIDDEN: int = 96
    MODEL_HEAD_HIDDEN: int = 96
    MODEL_DROPOUT: float = 0.0
    MODEL_DIR_BIAS_SCALE: float = 1.0
    MODEL_AFFORDANCE_SCALE: float = 0.5
    MODEL_ACTION_BIAS_SCALE: float = 0.5
    MODEL_MAX_CELL_ID: int = 31
    MODEL_FOOD_ID: int = 10
    MODEL_AGENT_ID: int = 11
    MODEL_EMPTY_ID: int = 9
    MODEL_MOVEABLE_ID: int = 7
    MODEL_GRABBABLE_ID: int = 8
    MODEL_VMAX_VALUE: float = 1000.0
    MODEL_SELECTOR_TAU: float = 0.6
    MODEL_SELECTOR_RECENCY: float = 0.3
    MODEL_RAY_MAX_STEPS: int = 6
    MODEL_RAY_ALPHA: float = 1.0
    MODEL_RAY_BETA: float = 0.5

    # Metadata
    DESCRIPTION: str = "Default FunGrid configuration"


def apply_user_overrides() -> None:
    """Import ``user_config`` if present and promote uppercase attributes to Config."""

    module_name = "fun_grid.user_config"
    try:
        user_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return

    overrides = {
        name: getattr(user_module, name)
        for name in dir(user_module)
        if name.isupper()
    }
    for name, value in overrides.items():
        setattr(Config, name, value)


apply_user_overrides()
print(f"Using device: {Config.DEVICE}")


__all__ = ["Config", "apply_user_overrides"]
