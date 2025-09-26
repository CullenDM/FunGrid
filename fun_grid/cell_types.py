"""Enumerations describing grid cell contents."""

from enum import IntEnum


class CellType(IntEnum):
    """Discrete labels used throughout the FunGrid environment."""

    EMPTY = 0
    AGENT = 1
    FOOD = 2
    OBSTACLE = 3
    MOVEABLE_OBSTACLE = 4
    GRABBABLE_OBSTACLE = 5
    BOUNCING_OBSTACLE = 6
    LOOPING_OBSTACLE = 7
    COUNTER_LOOPING_OBSTACLE = 8
    WANDERING_OBSTACLE = 9
    BOUNDARY = 10


__all__ = ["CellType"]
