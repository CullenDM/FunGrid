"""Configuration defaults for the FunGrid reinforcement learning setup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    """Default hyperparameters and feature toggles."""

    # Device and optimization
    DEVICE: str = "cpu"
    ENTROPY_COEFFICIENT: float = 0.01
    MODEL_GRAD_CLIP_NORM: float = 1.0

    # PPO training parameters
    CLIP_PARAM: float = 0.2
    PPO_EPOCHS: int = 1
    TARGET_KL: float = 0.01
    GAMMA: float = 0.99
    TAU: float = 0.95
    MINIBATCH_SIZE: int = 64
    UPDATE_FREQUENCY: int = 128
    USE_PPO_UPDATE: bool = True
    USE_EARLY_STOPPING: bool = False

    # Optional halting auxiliaries
    PPO_HALTING_COEF: float = 0.0
    PPO_PONDER_PENALTY: float = 0.0
    PPO_SELECTOR_ENT_COEF: float = 0.0
    MODEL_MAX_SEGMENTS: int = 1

    # Persistence
    SAVE_TRANSITIONS: bool = False
    TRANSITIONS_FILE_PATH: str = "artifacts/transitions.pkl"

    # Environment layout and counts
    ENVIRONMENT_SIZE: int = 16
    GRID_SIZE: int = 16
    MIN_EMPTY_PERCENTAGE: float = 0.35
    VARIABLE_FOOD_COUNT: bool = False
    SCALE_FOOD_COUNT: bool = False
    SET_FOOD_COUNT: int = 10
    VARIABLE_OBSTACLE_COUNT: bool = False
    OBSTACLE_CHOICE: int = 17
    MAX_INVENTORY: int = 3

    # Food tick behaviour
    FOOD_TICK: bool = False
    FOOD_TICK_SPEED: int = 8
    USE_RANDOM_FOOD_TICK: bool = False
    MAX_FOOD_TICK_SPEED: int = 10

    # Simulation toggles
    NUM_ENVS: int = 1
    NUM_AGENTS: int = 1
    VISUALIZE: bool = False
    TARGET_FPS: int = 60
    CELL_SIZE: int = 32
    SEQUENCE_LENGTH: int = 4
    DRAW_AGENT_PATH: bool = False
    HIGHLIGHT_MOVED_OBSTACLES: bool = False
    HIGHLIGHT_PLACED_OBSTACLES: bool = False
    PPO_TOGETHER: bool = False
    SAVE_FREQUENCY: int = 10
    DEBUG: bool = False
    USE_GLOBAL_STATE: bool = False

    # Reward multipliers
    USE_STEP_MULTIPLIER: bool = False
    USE_SINCE_LAST_FOOD_MULTIPLIER: bool = False
    USE_FOOD_MULTIPLIER: bool = False

    # Agent/state dimensions
    STATE_DIM: int = 13

    # Model persistence and optimization
    LOAD_MODEL: bool = False
    MODEL_PATH: str = "artifacts/model.pt"
    LEARNING_RATE: float = 3e-4

    # Miscellaneous values that may be extended by future configs
    DESCRIPTION: str = "Default FunGrid configuration"


__all__ = ["Config"]
