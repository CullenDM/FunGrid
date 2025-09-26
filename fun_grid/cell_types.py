"""Enumerations describing grid cell contents and obstacle presets."""

from enum import IntEnum


class CellType(IntEnum):
    """Discrete labels used throughout the FunGrid environment."""

    BOUNDARY = 1
    WANDERING_OBSTACLE = 2
    COUNTER_LOOPING_OBSTACLE = 3
    LOOPING_OBSTACLE = 4
    BOUNCING_OBSTACLE = 5
    OBSTACLE = 6
    MOVEABLE_OBSTACLE = 7
    GRABBABLE_OBSTACLE = 8
    EMPTY = 9
    FOOD = 10
    AGENT = 11
    OTHER_AGENT = 12


class ObstacleSetupChoice(IntEnum):
    """Preset obstacle distributions used when generating environments."""

    ONLY_STATIC_LARGE = 0
    ONLY_MOVEABLE_LARGE = 1
    ONLY_GRABBABLE_LARGE = 2
    MIXED_SMALL_NO_BOUNCE = 3
    EMPTY_NO_OBSTACLES = 4
    MIXED_RANDOM_SMALL_NO_BOUNCE = 5
    STATIC_MOVEABLE_HALF = 6
    MOVEABLE_GRABBABLE_HALF = 7
    STATIC_GRABBABLE_HALF = 8
    MIXED_SMALL_BOUNCE_LOW = 9
    MIXED_SMALL_BOUNCE_MED = 10
    MOVEABLE_GRABBABLE_BOUNCE_MED = 11
    STATIC_GRABBABLE_BOUNCE_MED = 12
    STATIC_MOVEABLE_BOUNCE_MED = 13
    ONLY_STATIC_BOUNCE_MED = 14
    ONLY_MOVEABLE_BOUNCE_MED = 15
    ONLY_GRABBABLE_BOUNCE_MED = 16
    MIXED_ALL_TYPES = 17


__all__ = ["CellType", "ObstacleSetupChoice"]
