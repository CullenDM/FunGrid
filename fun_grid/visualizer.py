"""Pygame-based visualization utilities for FunGrid."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pygame

from .cell_types import CellType
from .config import Config


class GridVisualizer:
    """Render one or more grid environments and agent traces via pygame."""

    def __init__(self, envs: Sequence, cell_size: int | None = None) -> None:
        self.envs = list(envs)
        self.cell_size = cell_size or Config.CELL_SIZE

        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.target_fps = getattr(Config, "TARGET_FPS", 60)

        grid_width = len(self.envs) * self.envs[0].size * self.cell_size
        grid_height = self.envs[0].size * self.cell_size

        self.view_scale = 4
        self.memory_scale = 3

        self.width = grid_width
        self.height = grid_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World Training Environment")

        self.buffers = [np.full((env.size, env.size), None) for env in self.envs]

        self.colors = {
            CellType.EMPTY: (240, 240, 240),
            CellType.AGENT: (255, 0, 0),
            CellType.OTHER_AGENT: (200, 0, 200),
            CellType.FOOD: (0, 255, 0),
            CellType.OBSTACLE: (40, 40, 40),
            CellType.MOVEABLE_OBSTACLE: (90, 90, 90),
            CellType.GRABBABLE_OBSTACLE: (125, 125, 125),
            CellType.BOUNCING_OBSTACLE: (190, 190, 0),
            CellType.LOOPING_OBSTACLE: (0, 134, 134),
            CellType.COUNTER_LOOPING_OBSTACLE: (34, 53, 129),
            CellType.WANDERING_OBSTACLE: (0, 223, 186),
            CellType.BOUNDARY: (0, 0, 0),
        }

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_p:
                    pass

        self.clock.tick(self.target_fps)
        return True

    def update(self, agents: Iterable) -> None:
        if Config.DRAW_AGENT_PATH:
            visited_all = self._draw_agent_paths(agents)
        else:
            visited_all = set()

        for idx, env in enumerate(self.envs):
            for x in range(env.size):
                for y in range(env.size):
                    if env.grid[x, y] != self.buffers[idx][x, y]:
                        self.buffers[idx][x, y] = env.grid[x, y]
                        if (
                            Config.DRAW_AGENT_PATH
                            and (x, y) in visited_all
                            and CellType(env.grid[x, y]) == CellType.EMPTY
                        ):
                            self._draw_cell(idx, x, y, (0, 0, 0))
                        elif (
                            Config.HIGHLIGHT_MOVED_OBSTACLES
                            and (x, y) in env.moved_obstacles
                            and CellType(env.grid[x, y]) == CellType.MOVEABLE_OBSTACLE
                        ):
                            self._draw_cell(idx, x, y, (255, 20, 90))
                        elif (
                            Config.HIGHLIGHT_PLACED_OBSTACLES
                            and (x, y) in env.placed_obstacles
                            and CellType(env.grid[x, y]) == CellType.GRABBABLE_OBSTACLE
                        ):
                            self._draw_cell(idx, x, y, (0, 255, 255))
                        else:
                            try:
                                color = self.colors[CellType(self.buffers[idx][x, y])]
                            except (ValueError, KeyError):
                                color = (128, 128, 128)
                            self._draw_cell(idx, x, y, color)

        pygame.display.flip()

    def _draw_agent_paths(self, agents: Iterable) -> set[tuple[int, int]]:
        visited = set()
        for agent in agents:
            visited.update(getattr(agent, "visited_cells", []))
        return visited

    def _draw_cell(self, env_index: int, x: int, y: int, color: tuple[int, int, int]) -> None:
        rect = (
            env_index * self.envs[0].size * self.cell_size + x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, color, rect)

    def force_update_cell(self, env_index: int, x: int, y: int) -> None:
        self.buffers[env_index][x, y] = self.envs[env_index].grid[x, y]
        try:
            color = self.colors[CellType(self.buffers[env_index][x, y])]
        except (ValueError, KeyError):
            color = (128, 128, 128)
        self._draw_cell(env_index, x, y, color)
        pygame.display.update()

    def reset(self) -> None:
        for buffer in self.buffers:
            buffer.fill(None)
        self.screen.fill(self.colors[CellType.EMPTY])
        pygame.display.update()

    def close(self) -> None:
        try:
            pygame.display.quit()
        finally:
            pygame.quit()


__all__ = ["GridVisualizer"]
