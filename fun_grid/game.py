"""Lightweight pygame-powered FunGrid game loop."""

from __future__ import annotations

from typing import List

from .constants import (
    AGENT_COLOR,
    BACKGROUND_COLOR,
    CELL_SIZE,
    FOOD_COLOR,
    FPS,
    GRID_COLOR,
    NUM_FOOD,
    NUM_OBSTACLES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from .entities import GameAgent, GameFood, GameObstacle
from .grid import SpatialGrid

try:  # pragma: no cover - optional dependency
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - headless environments
    pygame = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class Game:
    """Simple interactive pygame loop showcasing the grid world."""

    def __init__(self, screen: "pygame.Surface" | None = None) -> None:
        if pygame is None:  # pragma: no cover - requires pygame
            raise RuntimeError(
                "pygame is not installed. Install pygame to use the FunGrid game interface."
            ) from _IMPORT_ERROR

        self.screen = screen or pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("FunGrid")
        self.clock = pygame.time.Clock()

        self.grid = SpatialGrid(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.obstacles: List[GameObstacle] = []
        self.foods: List[GameFood] = []
        self.agent = self._spawn_agent()

    def _spawn_agent(self) -> GameAgent:
        spawn = self.grid.random_empty_cell()
        if spawn is None:
            spawn = (0, 0)
        agent = GameAgent(*spawn, color=AGENT_COLOR)
        self.grid.insert(agent, agent.x, agent.y)
        self._populate_level()
        return agent

    def _populate_level(self) -> None:
        for _ in range(NUM_OBSTACLES):
            self._place_obstacle()
        for _ in range(NUM_FOOD):
            self._place_food()

    def _place_obstacle(self) -> None:
        loc = self.grid.random_empty_cell()
        if loc is None:
            return
        obstacle = GameObstacle(*loc)
        self.obstacles.append(obstacle)
        self.grid.insert(obstacle, obstacle.x, obstacle.y)

    def _place_food(self) -> None:
        loc = self.grid.random_empty_cell()
        if loc is None:
            return
        food = GameFood(*loc)
        self.foods.append(food)
        self.grid.insert(food, food.x, food.y)

    def run(self) -> None:  # pragma: no cover - real-time loop
        running = True
        while running:
            running = self._process_events()
            self._draw()
            self.clock.tick(FPS)

    def _process_events(self) -> bool:
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                self._handle_move(event.key)
        return True

    def _handle_move(self, key: int) -> None:
        assert pygame is not None
        direction = {
            pygame.K_UP: "up",
            pygame.K_DOWN: "down",
            pygame.K_LEFT: "left",
            pygame.K_RIGHT: "right",
        }.get(key)
        if not direction:
            return
        reward = self.agent.move(direction, self.grid, self.foods)
        if reward > 0:
            self._place_food()

    def _draw(self) -> None:
        assert pygame is not None
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_grid()
        for obstacle in self.obstacles:
            obstacle.render(self.screen)
        for food in self.foods:
            food.render(self.screen)
        self.agent.render(self.screen)
        pygame.display.flip()

    def _draw_grid(self) -> None:
        assert pygame is not None
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))


__all__ = ["Game"]
