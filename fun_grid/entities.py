"""Entity definitions for the FunGrid game world."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pygame

from .constants import BATCH_SIZE, CELL_SIZE, FOOD_COLOR, FOOD_RADIUS, OBSTACLE_COLOR
from .dqn import DQNAgent, DQNConfig


class Entity:
    """Base class for all entities in the game."""

    entity_type = "entity"

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def render(self, surface: pygame.Surface) -> None:
        """Render the entity on the game surface."""


class Obstacle(Entity):
    """Represents an obstacle in the game."""

    entity_type = "obstacle"

    def render(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, OBSTACLE_COLOR, (self.x, self.y, CELL_SIZE, CELL_SIZE))


class Food(Entity):
    """Represents a food item in the game."""

    entity_type = "food"

    def render(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(
            surface,
            FOOD_COLOR,
            (self.x + CELL_SIZE // 2, self.y + CELL_SIZE // 2),
            FOOD_RADIUS,
        )


class Agent(Entity):
    """Represents an agent in the game."""

    entity_type = "agent"

    def __init__(self, x: int, y: int, spatial_grid, color=(255, 0, 0)) -> None:
        super().__init__(x, y)
        self.color = color
        self.spatial_grid = spatial_grid
        self.food_eaten = 0
        self.steps_since_food = 0
        self.score = 0
        state_size = 7 * 7 * 4 + 6
        self.dqn_agent = DQNAgent(DQNConfig(state_size=state_size, action_size=4))

    def render(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, (self.x, self.y, CELL_SIZE, CELL_SIZE))

    def get_reward(self, items_in_target_cell: Sequence[Entity]) -> int:
        entity_types = {getattr(item, "entity_type", None) for item in items_in_target_cell}
        if {"obstacle", "agent"} & entity_types:
            return -10
        if "food" in entity_types:
            return 100
        return -1

    def move(self, action: int, food_list: List[Food]) -> Tuple[int, int]:
        move_map = {
            0: (0, -CELL_SIZE),
            1: (0, CELL_SIZE),
            2: (-CELL_SIZE, 0),
            3: (CELL_SIZE, 0),
        }

        dx, dy = move_map.get(action, (0, 0))
        new_x, new_y = self.x + dx, self.y + dy

        items_in_target_cell = self.spatial_grid.get_items(new_x, new_y)
        reward = self.get_reward(items_in_target_cell)

        if reward != -10:
            self.spatial_grid.move(self, self.x, self.y, new_x, new_y)
            self.x, self.y = new_x, new_y
            if reward == 100:
                self.food_eaten += 1
                self.steps_since_food = 0
                self.score += 100
                for food in list(food_list):
                    if food.x == self.x and food.y == self.y:
                        food_list.remove(food)
                        self.spatial_grid.remove(food, food.x, food.y)
                        break

        return reward, action

    def choose_action(self, state) -> int:
        return self.dqn_agent.act(state)

    def update(self, food_list: List[Food], state, game) -> Tuple[int, int]:
        choice = self.choose_action(state)
        reward, action = self.move(choice, food_list)
        self.score += reward
        next_state = game.get_state(self)
        done = False
        self.dqn_agent.remember(state, action, reward, next_state, done)
        if (
            len(self.dqn_agent.memory) > BATCH_SIZE
            and (game.steps + 1) % BATCH_SIZE == 0
            and game.steps > 0
        ):
            self.dqn_agent.replay(BATCH_SIZE)

        return reward, action
