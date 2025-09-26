"""Game loop and orchestration for FunGrid."""

from __future__ import annotations

import random
import sys
from typing import List

import pygame
import numpy as np

from .constants import (
    BACKGROUND_COLOR,
    BORDER_WIDTH,
    CELL_SIZE,
    FPS,
    GRID_COLOR,
    NUM_AGENTS,
    NUM_FOOD,
    NUM_OBSTACLES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    AGENT_COLORS,
)
from .entities import Agent, Food, Obstacle
from .grid import SpatialGrid


class Game:
    """Main game class that handles game logic and rendering."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.episode = 0
        self.steps = 0
        self.score = 0
        self.initialize_game()

    def add_border_obstacles(self, border_width: int) -> None:
        """Add obstacles around the border of the game screen."""
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            for y in range(0, border_width * CELL_SIZE, CELL_SIZE):
                top_obstacle = Obstacle(x, y)
                bottom_obstacle = Obstacle(x, SCREEN_HEIGHT - CELL_SIZE - y)

                self.spatial_grid.insert(top_obstacle, x, y)
                self.spatial_grid.insert(bottom_obstacle, x, SCREEN_HEIGHT - CELL_SIZE - y)

                self.obstacles.append(top_obstacle)
                self.obstacles.append(bottom_obstacle)

        for y in range(border_width * CELL_SIZE, SCREEN_HEIGHT - border_width * CELL_SIZE, CELL_SIZE):
            for x in range(0, border_width * CELL_SIZE, CELL_SIZE):
                left_obstacle = Obstacle(x, y)
                right_obstacle = Obstacle(SCREEN_WIDTH - CELL_SIZE - x, y)

                self.spatial_grid.insert(left_obstacle, x, y)
                self.spatial_grid.insert(right_obstacle, SCREEN_WIDTH - CELL_SIZE - x, y)

                self.obstacles.append(left_obstacle)
                self.obstacles.append(right_obstacle)

    def add_random_food(self, num_food: int) -> None:
        """Add a random number of food items to the game."""
        empty_cells = self.spatial_grid.get_empty_cells()
        if not empty_cells:
            return
        food_cells = random.sample(empty_cells, min(num_food, len(empty_cells)))
        for cell in food_cells:
            food = Food(cell[0], cell[1])
            self.food.append(food)
            self.spatial_grid.insert(food, food.x, food.y)

    def add_random_obstacles(self, num_obstacles: int) -> None:
        """Add a random number of obstacles to the game."""
        empty_cells = self.spatial_grid.get_empty_cells()
        if not empty_cells:
            return
        obstacle_cells = random.sample(empty_cells, min(num_obstacles, len(empty_cells)))
        for cell in obstacle_cells:
            obstacle = Obstacle(cell[0], cell[1])
            self.obstacles.append(obstacle)
            self.spatial_grid.insert(obstacle, obstacle.x, obstacle.y)

    def initialize_game(self) -> None:
        """Initialize the game state and entities."""
        self.spatial_grid = SpatialGrid(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.obstacles: List[Obstacle] = []
        self.food: List[Food] = []
        self.add_border_obstacles(BORDER_WIDTH)
        self.add_random_obstacles(NUM_OBSTACLES)
        self.add_random_food(NUM_FOOD)

        empty_cells = self.spatial_grid.get_empty_cells()
        self.agents: List[Agent] = []
        for agent_index in range(NUM_AGENTS):
            empty_cells = self.spatial_grid.get_empty_cells()
            if not empty_cells:
                break
            agent_start_x, agent_start_y = random.choice(empty_cells)
            color = AGENT_COLORS[agent_index % len(AGENT_COLORS)]
            agent = Agent(agent_start_x, agent_start_y, self.spatial_grid, color=color)
            self.spatial_grid.insert(agent, agent_start_x, agent_start_y)
            self.agents.append(agent)

    def reset(self) -> None:
        """Reset the game state and entities."""
        self.initialize_game()
        self.steps = 0
        self.score = 0
        self.episode += 1
        for agent in self.agents:
            agent.score = 0
            agent.food_eaten = 0
            agent.steps_since_food = 0

    def update_food(self) -> None:
        """Update the food items in the game."""
        if len(self.food) < 10:
            self.add_random_food(1)

    def draw_grid(self) -> None:
        """Draw the grid lines on the game screen."""
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

    def get_state(self, agent: Agent):
        """Retrieve the current state of the agent."""
        grid_state = self.spatial_grid.get_view(agent.x, agent.y)
        food_eaten = agent.food_eaten
        steps_since_food = agent.steps_since_food
        score = agent.score
        time_step = self.steps
        agent_col = agent.x // CELL_SIZE
        agent_row = agent.y // CELL_SIZE

        flattened_grid_state = np.array(grid_state).reshape(-1)
        combined_state = np.concatenate(
            [
                flattened_grid_state,
                [food_eaten, steps_since_food, score, time_step, agent_col, agent_row],
            ]
        )
        return combined_state.reshape(1, -1)

    def draw_entities(self, entity_list) -> None:
        """Draw all entities on the game screen."""
        for entity in entity_list:
            entity.render(self.screen)

    def run(self) -> None:
        """Main game loop."""
        clock = pygame.time.Clock()
        self.episode += 1

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(BACKGROUND_COLOR)
            for agent in self.agents:
                state = self.get_state(agent)
                reward, action = agent.update(food_list=self.food, state=state, game=self)
                print(
                    f"Episode: {self.episode}, Step: {self.steps}, Reward: {reward}, "
                    f"Action: {action}, Score: {agent.score}"
                )

            self.draw_entities(self.agents + self.obstacles + self.food)
            self.update_food()
            self.draw_grid()

            pygame.display.flip()
            clock.tick(FPS)
            self.steps += 1

            if self.steps % 1000 == 0 and self.steps > 0:
                self.reset()

