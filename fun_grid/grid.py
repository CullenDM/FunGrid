"""Spatial grid utilities for the FunGrid game."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

from .constants import CELL_SIZE


class SpatialGrid:
    """Represents a spatial grid environment for the game."""

    def __init__(self, width: int, height: int, cell_size: int = CELL_SIZE) -> None:
        """Initialize the spatial grid with given dimensions and cell size."""
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid: List[List[Set[object]]] = [
            [set() for _ in range(self.rows)] for _ in range(self.cols)
        ]

    def _get_cell_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert screen coordinates to grid cell coordinates."""
        col = x // self.cell_size
        row = y // self.cell_size
        return col, row

    def insert(self, item: object, x: int, y: int) -> None:
        """Insert item at the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        self.grid[col][row].add(item)

    def remove(self, item: object, x: int, y: int) -> None:
        """Remove item from the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        self.grid[col][row].discard(item)

    def move(self, item: object, x_from: int, y_from: int, x_to: int, y_to: int) -> None:
        """Move item from one set of screen coordinates to another."""
        self.remove(item, x_from, y_from)
        self.insert(item, x_to, y_to)

    def get_items(self, x: int, y: int) -> Set[object]:
        """Retrieve all items from the cell at the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        return self.grid[col][row]

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Return a list of coordinates for all empty cells."""
        empty_cells = []
        for col in range(self.cols):
            for row in range(self.rows):
                if not self.grid[col][row]:
                    empty_cells.append((col * self.cell_size, row * self.cell_size))
        return empty_cells

    def get_view(self, agent_x: int, agent_y: int, size: int = 7) -> List[List[Sequence[int]]]:
        """Retrieve a view of the grid around a given agent."""
        half_size = size // 2
        view: List[List[Sequence[int]]] = []

        for y in range(
            agent_y - half_size * CELL_SIZE,
            agent_y + (half_size + 1) * CELL_SIZE,
            CELL_SIZE,
        ):
            row_view = []
            for x in range(
                agent_x - half_size * CELL_SIZE,
                agent_x + (half_size + 1) * CELL_SIZE,
                CELL_SIZE,
            ):
                items = self.get_items(x, y)
                entity_types = {getattr(item, "entity_type", None) for item in items}
                if "agent" in entity_types:
                    row_view.append([0, 0, 0, 1])
                elif "obstacle" in entity_types:
                    row_view.append([0, 1, 0, 0])
                elif "food" in entity_types:
                    row_view.append([0, 0, 1, 0])
                else:
                    row_view.append([1, 0, 0, 0])
            view.append(row_view)

        return view
