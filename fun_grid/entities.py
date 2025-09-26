"""Agent logic and interaction handling for FunGrid."""

from __future__ import annotations

from collections import deque
import random
from enum import Enum
from typing import Deque, List, Set, Tuple

import torch
from torch.distributions import Categorical

from .cell_types import CellType
from .config import Config
from .rl import TransitionStorage


class Action(Enum):
    """Available high-level actions."""

    MOVE = 0
    GRAB = 1
    PLACE = 2


class Direction(Enum):
    """Cardinal movement directions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Agent:
    """Drop-in replacement Agent with enhanced reward shaping."""

    FOOD_REWARD = 1.0
    COMPLETION_REWARD = 100.0

    UNVISITED_EMPTY_REWARD = 0.010
    EMPTY_REWARD = 0.002

    ADJACENT_FOOD_NOT_EATEN_REWARD = -0.0010
    STEPS_SINCE_FOOD_REWARD = -0.000008
    DIRECTION_MAINTAINED_REWARD = 0.020
    STEP_REWARD = -0.00005
    REDUNDANT_MOVEMENT_REWARD = 0.0
    NO_FOOD_LEFT_REWARD = 0.0

    BOUNCING_COLLISION_REWARD = -0.020
    OTHER_AGENT_REWARD = -0.020
    OBSTACLE_REWARD = 0.0
    LOOPING_OBSTACLE_REWARD = 0.0
    COUNTER_LOOPING_OBSTACLE_REWARD = 0.0
    WANDERING_OBSTACLE_REWARD = 0.0

    INVALID_STATIC_OBSTACLE_PENALTY = -0.030
    INVALID_GRABBABLE_OBSTACLE_PENALTY = -0.020
    INVALID_OTHER_AGENT_PENALTY = -0.030
    INVALID_DYNAMIC_OBSTACLE_PENALTY = -0.015
    BOUNDARY_REWARD = -0.030

    MOVEABLE_OBSTACLE_REWARD = 0.020
    MOVEABLE_OBSTACLE_FAIL_REWARD = -0.008

    GRABBABLE_OBSTACLE_REWARD = 0.010
    PLACEABLE_OBSTACLE_REWARD = 0.010
    GRABBABLE_OBSTACLE_FAIL_REWARD = -0.005

    MOVEMENT_FAILURE_REWARD = -0.010

    NO_FOOD_EATEN_TOTAL_REWARD = -0.200

    INITIAL_ENERGY = 1000
    ENERGY_CONSUMPTION_RATE = 1
    ENERGY_FOR_FOOD = 10

    ENERGY_EXTRA_FOR_PUSH_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_PUSH_FAIL = 1.0

    ENERGY_EXTRA_FOR_GRAB_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_GRAB_FAIL = 1.0
    ENERGY_EXTRA_FOR_PLACE_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_PLACE_FAIL = 1.0

    ENERGY_PENALTY_BOUNCING_COLLISION = 10

    VIEW_SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH

    def __init__(self, model, environment, env_idx: int, agent_idx: int, visualizer=None, initial_energy: int = INITIAL_ENERGY) -> None:
        self.model = model
        self.env = environment
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.visualizer = visualizer
        self.energy = initial_energy
        self.start_x: int | None = None
        self.start_y: int | None = None
        self.transitions = TransitionStorage()
        self.reset()

    def reset(self) -> None:
        if self.env.new_grid_generated or self.start_x is None:
            spawn_cells = list(self.env.get_empty_cells())
            if not spawn_cells:
                raise RuntimeError("No empty cells available for agent spawn.")
            self.x, self.y = random.choice(spawn_cells)
            self.start_x, self.start_y = self.x, self.y
        else:
            self.x, self.y = self.start_x, self.start_y

        self.env.grid[self.x, self.y] = CellType.AGENT
        self.env.empty_cells.discard((self.x, self.y))
        self.energy = self.INITIAL_ENERGY
        self.x_prev, self.y_prev = self.x, self.y

        self.num_grabbable_obstacle_interactions = 0
        self.num_moveable_obstacle_interactions = 0
        self.num_placeable_obstacle_interactions = 0

        self.steps_since_last_food = 0
        self.total_reward = 0.0
        self.food_eaten = 0
        self.done = False
        self.direction: int | None = None
        self.last_direction = 0
        self.direction_counter = 0
        self.last_action = 0
        self.inventory: List[CellType] = []
        self.steps = 0
        self.state_vector = None
        self.last_view = None
        self.init_sequences()
        self.total_predicted_rewards = 0.0
        self.last_reward = 0.0

        self.value_loss: List[float] = []
        self.action_loss: List[float] = []
        self.direction_loss: List[float] = []
        self.mean_action_loss = 0.0
        self.mean_direction_loss = 0.0
        self.mean_value_loss = 0.0
        self.compute_times: List[float] = []
        self.mean_compute_time = 0.0

        self.visited_cells: Set[Tuple[int, int]] = set()
        self.visited_cells.add((self.x, self.y))
        self.collided_with_bouncing_obstacle = False
        self.env.agents.add((self.x, self.y))

    def get_state_vector(self) -> torch.Tensor:
        state = [
            self.energy / self.INITIAL_ENERGY,
            self.steps_since_last_food,
            len(self.inventory) / Config.MAX_INVENTORY,
            self.food_eaten / (self.env.initial_food_count + 1e-6),
            self.x / Config.ENVIRONMENT_SIZE,
            self.y / Config.ENVIRONMENT_SIZE,
            self.steps / ((self.INITIAL_ENERGY * self.ENERGY_CONSUMPTION_RATE) + (self.env.num_food * self.ENERGY_FOR_FOOD * self.ENERGY_CONSUMPTION_RATE) + 1e-6),
            self.last_reward,
            self.total_reward / (self.steps + 1),
            float(self.collided_with_bouncing_obstacle),
            float(self.last_action),
            float(self.last_direction),
            float(self.env.food_tick_speed),
        ]
        state_vector = torch.tensor(state, dtype=torch.float32, device=Config.DEVICE)
        self.state_vector = state_vector
        self.state_vector_sequence.append(state_vector)
        self.collided_with_bouncing_obstacle = False
        return state_vector

    def init_sequences(self) -> None:
        initial_view_tensor = torch.full(
            (Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=torch.float32, device=Config.DEVICE
        )
        self.view_sequence: Deque[torch.Tensor] = deque(
            [initial_view_tensor for _ in range(Config.SEQUENCE_LENGTH)],
            maxlen=self.VIEW_SEQUENCE_LENGTH,
        )
        placeholder_state = torch.ones((Config.STATE_DIM,), dtype=torch.float32, device=Config.DEVICE) * -1
        self.state_vector = placeholder_state
        self.state_vector_sequence: Deque[torch.Tensor] = deque(
            [placeholder_state for _ in range(Config.SEQUENCE_LENGTH)],
            maxlen=self.VIEW_SEQUENCE_LENGTH,
        )

    def prepare_view_tensor(self, update_view: bool = True) -> torch.Tensor:
        if update_view:
            center_x, center_y = (
                (self.env.size // 2, self.env.size // 2)
                if Config.USE_GLOBAL_STATE
                else (self.x, self.y)
            )
            current_view = self.env.get_view(center_x, center_y, self.x, self.y, Config.GRID_SIZE)
            current_view_tensor = torch.from_numpy(current_view).float().to(Config.DEVICE)
            self.view_sequence.append(current_view_tensor)
            self.last_view = current_view
        return torch.stack(list(self.view_sequence), dim=0)

    def prepare_state_vector_sequence(self) -> torch.Tensor:
        self.get_state_vector()
        return torch.stack(list(self.state_vector_sequence), dim=0)

    def update(self, action_probs, direction_probs, view_tensor_sequence, state_vector_sequence, value):
        action, direction, action_log_prob, direction_log_prob, reward, done = self.take_action(
            action_probs, direction_probs
        )
        return (
            (view_tensor_sequence, state_vector_sequence),
            action,
            direction,
            action_log_prob,
            direction_log_prob,
            value,
            reward,
            done,
        )

    def take_action(self, action_probs, direction_probs):
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        direction_dist = Categorical(direction_probs)
        direction = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction)

        self.last_direction = self.direction if self.direction is not None else direction.item()
        self.direction = direction.item()
        self.steps_since_last_food += 1
        self.steps += 1

        action_result = self.perform_action(action, direction)
        reward = self.calculate_rewards(action_result)
        self.last_reward = reward
        self.total_reward += reward

        if self.energy <= 0:
            self.done = True
            if self.food_eaten == 0:
                reward += self.NO_FOOD_EATEN_TOTAL_REWARD
                self.total_reward += self.NO_FOOD_EATEN_TOTAL_REWARD

        return action.item(), direction.item(), action_log_prob.item(), direction_log_prob.item(), reward, self.done

    def perform_action(self, action, direction):
        a = int(action.item())
        d = int(direction.item())
        self.last_action = a

        if a == Action.MOVE.value:
            return self.handle_movement(d)
        return self.handle_obstacle_interaction(a, d)

    def calculate_rewards(self, action_result):
        reward = action_result["reward"]
        reward += self.STEP_REWARD * self.steps if Config.USE_STEP_MULTIPLIER else self.STEP_REWARD

        if action_result["ate"]:
            self.steps_since_last_food = 0

        if Config.USE_SINCE_LAST_FOOD_MULTIPLIER:
            reward += self.STEPS_SINCE_FOOD_REWARD * self.steps_since_last_food
        else:
            if self.steps_since_last_food > 0:
                reward += self.STEPS_SINCE_FOOD_REWARD

        reward += (
            self.REDUNDANT_MOVEMENT_REWARD if self.x == self.x_prev and self.y == self.y_prev else 0.0
        )

        if (
            self.last_direction == self.direction
            and self.last_action == Action.MOVE.value
            and self.direction_counter > 0
        ):
            reward += self.DIRECTION_MAINTAINED_REWARD / self.direction_counter

        reward += self.check_bouncing_obstacle_collisions()
        reward += self.check_adjacent_food_not_eaten_last_position(action_result["ate"])

        self.energy -= self.ENERGY_CONSUMPTION_RATE
        return reward

    def check_adjacent_food_not_eaten_last_position(self, ate):
        return (
            self.ADJACENT_FOOD_NOT_EATEN_REWARD
            if self.env.get_adjacent_and_diagonal_food(self.x_prev, self.y_prev) and not ate
            else 0.0
        )

    def check_bouncing_obstacle_collisions(self):
        for obstacle in self.env.bouncing_obstacle_objects:
            if (self.x, self.y) in obstacle.bounced_agents:
                obstacle.bounced_agents.remove((self.x, self.y))
                self.collided_with_bouncing_obstacle = True
                self.energy -= self.ENERGY_PENALTY_BOUNCING_COLLISION
                return self.BOUNCING_COLLISION_REWARD
        return 0.0

    def handle_movement(self, direction):
        dx, dy = self.get_movement_delta(direction)
        new_x, new_y = self.x + dx, self.y + dy

        if not (0 <= new_x < self.env.size and 0 <= new_y < self.env.size):
            return self.handle_invalid_move(target_cell_value=None, out_of_bounds=True)

        target_cell_value = CellType(self.env.grid[new_x, new_y])

        if target_cell_value == CellType.MOVEABLE_OBSTACLE:
            return self.execute_move(new_x, new_y)

        if self.is_valid_move(new_x, new_y):
            return self.execute_move(new_x, new_y)

        return self.handle_invalid_move(target_cell_value=target_cell_value, out_of_bounds=False)

    def is_valid_move(self, new_x, new_y):
        invalid_states = {
            CellType.OBSTACLE,
            CellType.GRABBABLE_OBSTACLE,
            CellType.AGENT,
            CellType.BOUNCING_OBSTACLE,
            CellType.LOOPING_OBSTACLE,
            CellType.COUNTER_LOOPING_OBSTACLE,
            CellType.WANDERING_OBSTACLE,
        }
        return (
            0 <= new_x < self.env.size
            and 0 <= new_y < self.env.size
            and self.env.grid[new_x, new_y] not in invalid_states
        )

    def execute_move(self, new_x, new_y):
        target_cell_value = CellType(self.env.grid[new_x, new_y])
        moved, ate, moved_obstacle, blocked_by_obstacle = self.env.move_agent(
            self.x, self.y, new_x, new_y
        )
        unvisited = (new_x, new_y) not in self.visited_cells

        if moved:
            self.x_prev, self.y_prev = self.x, self.y
            self.update_agent_position(new_x, new_y, ate)
            self.visited_cells.add((self.x, self.y))
            if self.last_action == Action.MOVE.value and self.direction == self.last_direction:
                self.direction_counter = (self.direction_counter or 0) + 1
            else:
                self.direction_counter = 1
            return {
                "reward": self.calculate_movement_reward(
                    new_x, new_y, ate, unvisited, moved_obstacle, blocked_by_obstacle, target_cell_value
                ),
                "moved": True,
                "ate": ate,
                "moved_obstacle": moved_obstacle,
                "failed_attempt": blocked_by_obstacle,
            }

        return {
            "reward": self.calculate_movement_reward(
                new_x, new_y, ate, unvisited, moved_obstacle, blocked_by_obstacle, target_cell_value
            ),
            "moved": False,
            "ate": False,
            "moved_obstacle": moved_obstacle,
            "failed_attempt": blocked_by_obstacle,
        }

    def calculate_movement_reward(self, new_x, new_y, ate, unvisited, moved_obstacle, failed_attempt, grid_value):
        reward = 0.0

        if ate:
            reward += self.calculate_dynamic_food_reward()
            self.direction_counter = 0
            self.num_grabbable_obstacle_interactions = 0
            self.num_placeable_obstacle_interactions = 0
            if self.env.food_remaining <= 0:
                reward += self.COMPLETION_REWARD
                self.done = True
            return reward

        if not ate and self.env.food_remaining <= 0:
            reward += self.NO_FOOD_LEFT_REWARD
            self.done = True
            return reward

        if grid_value == CellType.MOVEABLE_OBSTACLE:
            reward += self.check_moved_obstacle(moved_obstacle, failed_attempt)
            if moved_obstacle:
                reward += self.UNVISITED_EMPTY_REWARD if unvisited else self.EMPTY_REWARD
            return reward

        if grid_value == CellType.EMPTY:
            reward += self.UNVISITED_EMPTY_REWARD if unvisited else self.EMPTY_REWARD
        elif grid_value == CellType.AGENT:
            reward += self.OTHER_AGENT_REWARD
        elif grid_value == CellType.OBSTACLE:
            reward += self.OBSTACLE_REWARD
        elif grid_value == CellType.LOOPING_OBSTACLE:
            reward += self.LOOPING_OBSTACLE_REWARD
        elif grid_value == CellType.COUNTER_LOOPING_OBSTACLE:
            reward += self.COUNTER_LOOPING_OBSTACLE_REWARD
        elif grid_value == CellType.WANDERING_OBSTACLE:
            reward += self.WANDERING_OBSTACLE_REWARD

        return reward

    def handle_invalid_move(self, target_cell_value=None, out_of_bounds=False):
        if out_of_bounds:
            penalty = self.BOUNDARY_REWARD
        else:
            mapping = {
                CellType.OBSTACLE: self.INVALID_STATIC_OBSTACLE_PENALTY,
                CellType.GRABBABLE_OBSTACLE: self.INVALID_GRABBABLE_OBSTACLE_PENALTY,
                CellType.AGENT: self.INVALID_OTHER_AGENT_PENALTY,
                CellType.BOUNCING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.LOOPING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.COUNTER_LOOPING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.WANDERING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
            }
            penalty = mapping.get(target_cell_value, self.MOVEMENT_FAILURE_REWARD)

        return {
            "reward": penalty,
            "moved": False,
            "ate": False,
            "moved_obstacle": False,
            "failed_attempt": False,
        }

    def check_moved_obstacle(self, moved_obstacle, failed_attempt):
        if moved_obstacle:
            self.num_moveable_obstacle_interactions += 1
            self.energy -= self.ENERGY_EXTRA_FOR_PUSH_SUCCESS
            return self.MOVEABLE_OBSTACLE_REWARD / float(self.num_moveable_obstacle_interactions)

        if failed_attempt:
            self.energy -= self.ENERGY_EXTRA_FOR_PUSH_FAIL
            return self.MOVEABLE_OBSTACLE_FAIL_REWARD

        return 0.0

    def handle_obstacle_interaction(self, action, direction):
        grabbable_obstacles = self.env.get_adjacent_grabbable_obstacles(self.x, self.y)
        reward = 0.0
        directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        act_direction = directions[direction]

        if action == Action.GRAB.value:
            interaction_reward = self.handle_grab_action(grabbable_obstacles, act_direction)
        elif action == Action.PLACE.value:
            interaction_reward = self.handle_place_action(act_direction)
        else:
            interaction_reward = self.GRABBABLE_OBSTACLE_FAIL_REWARD

        reward += interaction_reward
        return {
            "reward": reward,
            "moved": False,
            "ate": False,
            "moved_obstacle": False,
            "failed_attempt": False,
        }

    def handle_grab_action(self, grabbable_obstacles, act_direction):
        if grabbable_obstacles:
            for obstacle in grabbable_obstacles:
                if (
                    obstacle == (self.x + act_direction[0], self.y + act_direction[1])
                    and len(self.inventory) < Config.MAX_INVENTORY
                ):
                    self.inventory.append(CellType.GRABBABLE_OBSTACLE)
                    obstacle_x, obstacle_y = self.x + act_direction[0], self.y + act_direction[1]
                    self.env.grabbable_obstacle_cells.remove((obstacle_x, obstacle_y))
                    self.env.placed_obstacles.discard((obstacle_x, obstacle_y))
                    self.env.grid[obstacle_x, obstacle_y] = CellType.EMPTY
                    self.env.empty_cells.add((obstacle_x, obstacle_y))
                    self.num_grabbable_obstacle_interactions += 1
                    self.energy -= self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_GRAB_SUCCESS
                    return (
                        self.GRABBABLE_OBSTACLE_REWARD
                        if self.num_grabbable_obstacle_interactions == 1
                        else self.GRABBABLE_OBSTACLE_REWARD * (self.num_grabbable_obstacle_interactions + 1) * -1
                    )

        self.energy -= self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_GRAB_FAIL
        return self.GRABBABLE_OBSTACLE_FAIL_REWARD

    def handle_place_action(self, act_direction):
        if self.inventory:
            place_x, place_y = self.x + act_direction[0], self.y + act_direction[1]
            if (
                self.env.is_within_bounds(place_x, place_y)
                and (place_x, place_y) in self.env.empty_cells
                and self.env.grid[place_x, place_y] != CellType.MOVEABLE_OBSTACLE
            ):
                self.env.grid[place_x, place_y] = CellType.GRABBABLE_OBSTACLE
                self.env.grabbable_obstacle_cells.add((place_x, place_y))
                self.env.placed_obstacles.add((place_x, place_y))
                self.env.empty_cells.discard((place_x, place_y))
                self.inventory.pop(0)
                self.num_placeable_obstacle_interactions += 1
                self.energy -= self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_PLACE_SUCCESS
                return (
                    self.PLACEABLE_OBSTACLE_REWARD
                    if self.num_placeable_obstacle_interactions == 1
                    else self.PLACEABLE_OBSTACLE_REWARD * (self.num_placeable_obstacle_interactions + 1) * -1
                )

        self.energy -= self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_PLACE_FAIL
        return self.GRABBABLE_OBSTACLE_FAIL_REWARD

    def update_agent_position(self, new_x, new_y, ate):
        self.x, self.y = new_x, new_y
        if self.visualizer:
            self.visualizer.force_update_cell(self.env_idx, self.x_prev, self.y_prev)
            self.visualizer.force_update_cell(self.env_idx, new_x, new_y)
        if ate:
            self.food_eaten += 1
            self.energy += self.ENERGY_FOR_FOOD

    def get_movement_delta(self, direction):
        movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        return movements.get(direction, (0, 0))

    def calculate_dynamic_food_reward(self):
        if Config.USE_FOOD_MULTIPLIER:
            return self.FOOD_REWARD * (self.food_eaten + 1)
        return self.FOOD_REWARD


__all__ = ["Action", "Agent", "Direction"]
