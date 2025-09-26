"""Environment generation and obstacle dynamics for FunGrid."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .cell_types import CellType, ObstacleSetupChoice
from .config import Config


class GridEnvironment:
    """Represents the grid environment for agents."""

    MIN_EMPTY_PERCENTAGE = Config.MIN_EMPTY_PERCENTAGE

    ADJACENT_DIRECTIONS: Sequence[Tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    DIAGONAL_DIRECTIONS: Sequence[Tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ALL_DIRECTIONS: Sequence[Tuple[int, int]] = tuple(ADJACENT_DIRECTIONS + list(DIAGONAL_DIRECTIONS))

    def __init__(self, idx: int, size: Optional[int] = None) -> None:
        self.idx = idx
        self.size = size if size is not None else Config.ENVIRONMENT_SIZE

        self.initial_food_count = 0

        self.num_food = 0
        self.num_obstacles = 0
        self.num_moveable_obstacles = 0
        self.num_grabbable_obstacles = 0
        self.num_bouncing_obstacles = 0
        self.food_tick_speed = Config.FOOD_TICK_SPEED

        self.world_type = ObstacleSetupChoice.MIXED_ALL_TYPES
        self.agents: Set[Tuple[int, int]] = set()
        self.new_grid_generated = True

        self.grid: np.ndarray | None = None
        self.empty_cells: Set[Tuple[int, int]] = set()
        self.food_cells: Set[Tuple[int, int]] = set()
        self.obstacle_cells: Set[Tuple[int, int]] = set()
        self.moveable_obstacle_cells: Set[Tuple[int, int]] = set()
        self.grabbable_obstacle_cells: Set[Tuple[int, int]] = set()
        self.bouncing_obstacle_objects: Set[BouncingObstacle] = set()
        self.moved_obstacles: Set[Tuple[int, int]] = set()
        self.placed_obstacles: Set[Tuple[int, int]] = set()
        self.food_remaining = 0

        self.reset()

    def reset(self) -> None:
        self._initialize_grid()
        self._determine_item_counts()
        self._distribute_items()

        self.agents.clear()
        self.new_grid_generated = True

    def _initialize_grid(self) -> None:
        self.grid = np.full((self.size, self.size), CellType.EMPTY, dtype=int)
        self.empty_cells = {(r, c) for r in range(self.size) for c in range(self.size)}
        self.food_cells.clear()
        self.obstacle_cells.clear()
        self.moveable_obstacle_cells.clear()
        self.grabbable_obstacle_cells.clear()
        self.bouncing_obstacle_objects.clear()
        self.moved_obstacles.clear()
        self.placed_obstacles.clear()
        self.food_remaining = 0

    def _determine_item_counts(self) -> None:
        total_cells = self.size * self.size
        required_agent_space = getattr(Config, "NUM_AGENTS", 1)
        max_items = math.floor(total_cells * (1.0 - self.MIN_EMPTY_PERCENTAGE))
        max_items = min(max_items, total_cells - required_agent_space)
        max_items = max(0, max_items)

        if Config.VARIABLE_FOOD_COUNT:
            initial_num_food = random.randint(1, max(1, total_cells // 10))
        elif Config.SCALE_FOOD_COUNT:
            initial_num_food = max(1, total_cells // 10)
        else:
            initial_num_food = max(0, Config.SET_FOOD_COUNT)
            if Config.SET_FOOD_COUNT > 0:
                initial_num_food = max(1, initial_num_food)

        initial_num_food = min(initial_num_food, max_items)
        if (
            initial_num_food == 0
            and (Config.VARIABLE_FOOD_COUNT or Config.SCALE_FOOD_COUNT or Config.SET_FOOD_COUNT > 0)
            and max_items >= 1
        ):
            initial_num_food = 1
            print(f"Info: Env {self.idx} - Adjusted initial food count to 1 due to capping.")

        final_num_food = initial_num_food

        if Config.USE_RANDOM_FOOD_TICK:
            self.food_tick_speed = random.randint(0, Config.MAX_FOOD_TICK_SPEED)
        else:
            self.food_tick_speed = Config.FOOD_TICK_SPEED

        total_obstacle_count = total_cells // 3
        break_down_count = max(1, total_obstacle_count // 4)

        if Config.VARIABLE_OBSTACLE_COUNT:
            choice = random.choice(list(ObstacleSetupChoice))
        else:
            try:
                choice = ObstacleSetupChoice(Config.OBSTACLE_CHOICE)
            except ValueError:
                print(
                    f"Warning: Invalid Config.OBSTACLE_CHOICE ({Config.OBSTACLE_CHOICE}). "
                    "Defaulting to MIXED_ALL_TYPES."
                )
                choice = ObstacleSetupChoice.MIXED_ALL_TYPES
        self.world_type = choice

        num_obstacles = 0
        num_moveable = 0
        num_grabbable = 0
        num_bouncing = 0

        if choice in (ObstacleSetupChoice.ONLY_STATIC_LARGE, ObstacleSetupChoice.ONLY_STATIC_BOUNCE_MED):
            num_obstacles = total_obstacle_count
        if choice in (ObstacleSetupChoice.ONLY_MOVEABLE_LARGE, ObstacleSetupChoice.ONLY_MOVEABLE_BOUNCE_MED):
            num_moveable = total_obstacle_count
        if choice in (ObstacleSetupChoice.ONLY_GRABBABLE_LARGE, ObstacleSetupChoice.ONLY_GRABBABLE_BOUNCE_MED):
            num_grabbable = total_obstacle_count
        if choice in (
            ObstacleSetupChoice.MIXED_SMALL_NO_BOUNCE,
            ObstacleSetupChoice.MIXED_SMALL_BOUNCE_LOW,
            ObstacleSetupChoice.MIXED_SMALL_BOUNCE_MED,
        ):
            num_obstacles = num_moveable = num_grabbable = break_down_count
        if choice == ObstacleSetupChoice.EMPTY_NO_OBSTACLES:
            pass
        if choice == ObstacleSetupChoice.MIXED_RANDOM_SMALL_NO_BOUNCE:
            num_obstacles = random.randint(0, break_down_count)
            num_moveable = random.randint(0, break_down_count)
            num_grabbable = random.randint(0, break_down_count)
        if choice in (
            ObstacleSetupChoice.STATIC_MOVEABLE_HALF,
            ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF,
            ObstacleSetupChoice.STATIC_GRABBABLE_HALF,
        ):
            half_count = total_obstacle_count // 2
            if choice == ObstacleSetupChoice.STATIC_MOVEABLE_HALF:
                num_obstacles, num_moveable = half_count, half_count
            elif choice == ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF:
                num_moveable, num_grabbable = half_count, half_count
            else:
                num_obstacles, num_grabbable = half_count, half_count
        if choice in (
            ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED,
            ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED,
            ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED,
        ):
            num_obstacles = break_down_count if choice in (
                ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED,
                ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED,
            ) else 0
            num_moveable = break_down_count if choice in (
                ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED,
                ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED,
            ) else 0
            num_grabbable = break_down_count if choice in (
                ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED,
                ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED,
            ) else 0
            num_bouncing = break_down_count // 3
        if choice == ObstacleSetupChoice.MIXED_ALL_TYPES:
            num_obstacles = random.randint(0, break_down_count)
            num_moveable = random.randint(0, break_down_count)
            num_grabbable = random.randint(0, break_down_count)
            num_bouncing = random.randint(0, max(1, break_down_count // 2))
        if choice in (
            ObstacleSetupChoice.MIXED_SMALL_BOUNCE_MED,
            ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED,
            ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED,
            ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED,
            ObstacleSetupChoice.ONLY_STATIC_BOUNCE_MED,
            ObstacleSetupChoice.ONLY_MOVEABLE_BOUNCE_MED,
            ObstacleSetupChoice.ONLY_GRABBABLE_BOUNCE_MED,
            ObstacleSetupChoice.MIXED_ALL_TYPES,
        ):
            num_bouncing = max(num_bouncing, break_down_count // 4)

        max_allowed_obstacles = max(0, max_items - final_num_food)
        current_obstacles_total = num_obstacles + num_moveable + num_grabbable + num_bouncing

        if current_obstacles_total > max_allowed_obstacles:
            obstacles_to_remove = current_obstacles_total - max_allowed_obstacles
            print(
                f"Warning: Env {self.idx} - Initial obstacle count ({current_obstacles_total}) exceeds "
                f"max allowed ({max_allowed_obstacles}) after placing food. Reducing {obstacles_to_remove} obstacles."
            )

            pool: List[CellType] = []
            pool.extend([CellType.OBSTACLE] * num_obstacles)
            pool.extend([CellType.MOVEABLE_OBSTACLE] * num_moveable)
            pool.extend([CellType.GRABBABLE_OBSTACLE] * num_grabbable)
            pool.extend([CellType.BOUNCING_OBSTACLE] * num_bouncing)
            random.shuffle(pool)

            for _ in range(obstacles_to_remove):
                if not pool:
                    break
                removed = pool.pop()
                if removed == CellType.OBSTACLE:
                    num_obstacles -= 1
                elif removed == CellType.MOVEABLE_OBSTACLE:
                    num_moveable -= 1
                elif removed == CellType.GRABBABLE_OBSTACLE:
                    num_grabbable -= 1
                elif removed == CellType.BOUNCING_OBSTACLE:
                    num_bouncing -= 1

            final_total = num_obstacles + num_moveable + num_grabbable + num_bouncing
            if final_total != max_allowed_obstacles:
                print(
                    f"Warning: Env {self.idx} - Obstacle capping mismatch. Target: {max_allowed_obstacles}, "
                    f"Final: {final_total}"
                )

        self.num_food = final_num_food
        self.initial_food_count = final_num_food
        self.num_obstacles = max(0, num_obstacles)
        self.num_moveable_obstacles = max(0, num_moveable)
        self.num_grabbable_obstacles = max(0, num_grabbable)
        self.num_bouncing_obstacles = max(0, num_bouncing)

        final_total_items = (
            self.num_food
            + self.num_obstacles
            + self.num_moveable_obstacles
            + self.num_grabbable_obstacles
            + self.num_bouncing_obstacles
        )
        if final_total_items > max_items:
            print(
                f"ERROR: Env {self.idx} - Final item count ({final_total_items}) STILL exceeds max ({max_items}) after capping!"
            )

    def _distribute_items(self) -> None:
        self._distribute_randomly(CellType.OBSTACLE, self.num_obstacles)
        self._distribute_randomly(CellType.FOOD, self.num_food)
        self._distribute_randomly(CellType.MOVEABLE_OBSTACLE, self.num_moveable_obstacles)
        self._distribute_randomly(CellType.GRABBABLE_OBSTACLE, self.num_grabbable_obstacles)
        self._distribute_bouncing_obstacles(self.num_bouncing_obstacles)

        self.empty_cells = {
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid is not None and self.grid[r, c] == CellType.EMPTY
        }
        self.food_remaining = self.num_food
        self._perform_sanity_checks()

    def _distribute_randomly(self, item_type: CellType, count: int) -> None:
        if count <= 0 or self.grid is None:
            return
        actual_count = min(count, len(self.empty_cells))
        if actual_count < count:
            print(
                f"Warning: Requested {count} of {item_type.name}, but only {actual_count} empty cells available."
            )
        if actual_count == 0:
            return

        cells_to_place = random.sample(list(self.empty_cells), actual_count)
        for cell in cells_to_place:
            self.grid[cell] = item_type
            self.empty_cells.remove(cell)
            self._add_to_specific_set(item_type, cell)

    def _distribute_bouncing_obstacles(self, count: int) -> None:
        if count <= 0 or self.grid is None:
            return
        actual_count = min(count, len(self.empty_cells))
        if actual_count < count:
            print(
                f"Warning: Requested {count} bouncing obstacles, but only {actual_count} empty cells available."
            )
        if actual_count == 0:
            return

        directions = ["up", "down", "left", "right"]
        cells_to_place = random.sample(list(self.empty_cells), actual_count)

        for cell in cells_to_place:
            direction = random.choice(directions)
            try:
                obstacle = BouncingObstacle(cell[0], cell[1], direction, self.size)
            except NameError:
                print("Error: BouncingObstacle class not defined. Skipping bouncing obstacle placement.")
                break
            self.bouncing_obstacle_objects.add(obstacle)
            if self.grid is not None:
                self.grid[cell] = obstacle.grid_value
            self.empty_cells.remove(cell)

    def _add_to_specific_set(self, item_type: CellType, cell: Tuple[int, int]) -> None:
        if item_type == CellType.FOOD:
            self.food_cells.add(cell)
        elif item_type == CellType.OBSTACLE:
            self.obstacle_cells.add(cell)
        elif item_type == CellType.MOVEABLE_OBSTACLE:
            self.moveable_obstacle_cells.add(cell)
        elif item_type == CellType.GRABBABLE_OBSTACLE:
            self.grabbable_obstacle_cells.add(cell)

    def _perform_sanity_checks(self) -> None:
        if self.grid is None:
            return

        obstacle_types = {
            CellType.OBSTACLE,
            CellType.MOVEABLE_OBSTACLE,
            CellType.GRABBABLE_OBSTACLE,
        }
        min_accessible_sides = 2

        food_cells_to_move: List[Tuple[int, int]] = []
        for r, c in list(self.food_cells):
            adjacent_obstacle_count = 0
            for dr, dc in self.ADJACENT_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if self.is_within_bounds(nr, nc):
                    if self.grid[nr, nc] in obstacle_types:
                        adjacent_obstacle_count += 1
                else:
                    adjacent_obstacle_count += 1

            if adjacent_obstacle_count >= 4 - min_accessible_sides + 1:
                food_cells_to_move.append((r, c))

        for r, c in food_cells_to_move:
            if not self.empty_cells:
                print(
                    f"Warning: Cannot relocate boxed-in food at ({r},{c}), no empty cells left."
                )
                continue

            new_cell = random.choice(list(self.empty_cells))
            self._move_item(r, c, new_cell[0], new_cell[1], CellType.FOOD)
            if Config.DEBUG:
                print(f"Relocated potentially boxed-in food from ({r},{c}) to {new_cell}")

    def get_view(self, center_r: int, center_c: int, agent_r: int, agent_c: int, view_size: int) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("Grid has not been initialized.")

        if Config.USE_GLOBAL_STATE:
            full_view = self.grid.copy()
            processed_view = self._process_view_for_agents(full_view, agent_r, agent_c, use_global=True)
            if full_view.shape != (view_size, view_size):
                padded_view = np.full((view_size, view_size), CellType.BOUNDARY, dtype=int)
                pad_r = max(0, (view_size - self.size) // 2)
                pad_c = max(0, (view_size - self.size) // 2)
                end_r = min(view_size, pad_r + self.size)
                end_c = min(view_size, pad_c + self.size)
                src_r_end = end_r - pad_r
                src_c_end = end_c - pad_c
                padded_view[pad_r:end_r, pad_c:end_c] = processed_view[:src_r_end, :src_c_end]
                return padded_view
            return processed_view

        half_view = view_size // 2
        view = np.full((view_size, view_size), CellType.BOUNDARY, dtype=int)

        grid_r_min = max(0, center_r - half_view)
        grid_r_max = min(self.size, center_r + half_view + 1)
        grid_c_min = max(0, center_c - half_view)
        grid_c_max = min(self.size, center_c + half_view + 1)

        grid_slice = self.grid[grid_r_min:grid_r_max, grid_c_min:grid_c_max]

        view_r_start = max(0, half_view - center_r + grid_r_min)
        view_c_start = max(0, half_view - center_c + grid_c_min)
        view_r_end = view_r_start + grid_slice.shape[0]
        view_c_end = view_c_start + grid_slice.shape[1]

        if grid_slice.size > 0:
            view[view_r_start:view_r_end, view_c_start:view_c_end] = grid_slice

        return self._process_view_for_agents(view, agent_r, agent_c, use_global=False)

    def _process_view_for_agents(
        self, view_array: np.ndarray, agent_r: int, agent_c: int, use_global: bool
    ) -> np.ndarray:
        processed_view = view_array.copy()

        if use_global:
            for r, c in self.agents:
                if r == agent_r and c == agent_c:
                    if processed_view[r, c] != CellType.AGENT:
                        print(
                            f"Warning: Agent at ({agent_r},{agent_c}) not found as AGENT on grid during global view processing."
                        )
                        processed_view[r, c] = CellType.AGENT
                elif self.is_within_bounds(r, c):
                    processed_view[r, c] = CellType.OTHER_AGENT
        else:
            view_h, view_w = view_array.shape
            center_r_view, center_c_view = view_h // 2, view_w // 2
            agent_mask = processed_view == CellType.AGENT
            agent_coords = np.argwhere(agent_mask)

            for r_view, c_view in agent_coords:
                if r_view != center_r_view or c_view != center_c_view:
                    processed_view[r_view, c_view] = CellType.OTHER_AGENT
                elif processed_view[center_r_view, center_c_view] != CellType.AGENT:
                    processed_view[center_r_view, center_c_view] = CellType.AGENT

        return processed_view

    def move_agent(self, old_r: int, old_c: int, new_r: int, new_c: int) -> Tuple[bool, bool, bool, bool]:
        if not self.is_within_bounds(new_r, new_c):
            return False, False, False, False

        if self.grid is None:
            return False, False, False, False

        target_cell_type = CellType(self.grid[new_r, new_c])
        return self._handle_movement(old_r, old_c, new_r, new_c, target_cell_type)

    def _handle_movement(
        self, old_r: int, old_c: int, new_r: int, new_c: int, target_type: CellType
    ) -> Tuple[bool, bool, bool, bool]:
        if target_type in (CellType.EMPTY, CellType.FOOD):
            ate = target_type == CellType.FOOD
            self._perform_move(old_r, old_c, new_r, new_c, ate)
            return True, ate, False, False
        if target_type == CellType.MOVEABLE_OBSTACLE:
            return self._handle_moveable_obstacle(old_r, old_c, new_r, new_c)
        return False, False, False, True

    def _perform_move(self, old_r: int, old_c: int, new_r: int, new_c: int, ate_food: bool) -> None:
        if self.grid is None:
            return
        self.grid[old_r, old_c] = CellType.EMPTY
        self.grid[new_r, new_c] = CellType.AGENT
        self._update_sets_on_move(old_r, old_c, new_r, new_c, ate_food)

    def _update_sets_on_move(self, old_r: int, old_c: int, new_r: int, new_c: int, ate_food: bool) -> None:
        self.empty_cells.add((old_r, old_c))
        if ate_food:
            if (new_r, new_c) in self.food_cells:
                self.food_cells.remove((new_r, new_c))
                self.food_remaining -= 1
            else:
                print(
                    f"Warning: Agent ate at ({new_r},{new_c}), but no food found in food_cells set."
                )
            self.empty_cells.discard((new_r, new_c))
        else:
            self.empty_cells.discard((new_r, new_c))
        self.agents.discard((old_r, old_c))
        self.agents.add((new_r, new_c))

    def _handle_moveable_obstacle(
        self, agent_r: int, agent_c: int, obstacle_r: int, obstacle_c: int
    ) -> Tuple[bool, bool, bool, bool]:
        dr, dc = obstacle_r - agent_r, obstacle_c - agent_c
        behind_r, behind_c = obstacle_r + dr, obstacle_c + dc

        if self.grid is None:
            return False, False, False, True

        if self.is_within_bounds(behind_r, behind_c) and self.grid[behind_r, behind_c] == CellType.EMPTY:
            self.grid[behind_r, behind_c] = CellType.MOVEABLE_OBSTACLE
            self.grid[obstacle_r, obstacle_c] = CellType.AGENT
            self.grid[agent_r, agent_c] = CellType.EMPTY

            self.moveable_obstacle_cells.remove((obstacle_r, obstacle_c))
            self.moveable_obstacle_cells.add((behind_r, behind_c))
            self.moved_obstacles.add((behind_r, behind_c))

            self.empty_cells.remove((behind_r, behind_c))
            self.empty_cells.add((agent_r, agent_c))

            self._update_sets_on_move(agent_r, agent_c, obstacle_r, obstacle_c, ate_food=False)
            return True, False, True, False
        return False, False, False, True

    def update_food(self, current_tick: int) -> None:
        if not Config.FOOD_TICK or self.food_tick_speed <= 0:
            return
        if current_tick % self.food_tick_speed != 0:
            return

        for r, c in list(self.food_cells):
            empty_adjacent = self._get_adjacent_cells(r, c, {CellType.EMPTY})
            if empty_adjacent:
                new_r, new_c = random.choice(empty_adjacent)
                self._move_item(r, c, new_r, new_c, CellType.FOOD)

    def build_non_empty_cells(self) -> Set[Tuple[int, int]]:
        s: Set[Tuple[int, int]] = set()
        s |= self.food_cells
        s |= self.obstacle_cells
        s |= self.moveable_obstacle_cells
        s |= self.grabbable_obstacle_cells
        s |= self.agents.copy()
        s |= {(o.x, o.y) for o in self.bouncing_obstacle_objects}
        return s

    def update_bouncing_obstacles(
        self, current_tick: int, non_empty_cells: Set[Tuple[int, int]]
    ) -> None:
        if not self.bouncing_obstacle_objects or self.grid is None:
            return

        dead_to_remove: List[BouncingObstacle] = []
        for obstacle in list(self.bouncing_obstacle_objects):
            if obstacle.dead:
                dead_to_remove.append(obstacle)
                continue

            old_pos = (obstacle.x, obstacle.y)
            other_objects = non_empty_cells.copy()
            other_objects.discard(old_pos)

            try:
                obstacle.update(self.grid, other_objects, current_tick)
            except Exception as exc:  # pragma: no cover - safety logging
                print(f"Error updating bouncing obstacle at ({obstacle.x},{obstacle.y}): {exc}")
                obstacle.dead = True

            if obstacle.dead:
                if 0 <= old_pos[0] < self.size and 0 <= old_pos[1] < self.size:
                    if self.grid[old_pos] == obstacle.grid_value:
                        self.grid[old_pos] = CellType.EMPTY
                dead_to_remove.append(obstacle)
                self.empty_cells.add(old_pos)
                non_empty_cells.discard(old_pos)
                continue

            new_pos = (obstacle.x, obstacle.y)
            if new_pos != old_pos:
                self.empty_cells.add(old_pos)
                self.empty_cells.discard(new_pos)
                non_empty_cells.discard(old_pos)
                non_empty_cells.add(new_pos)

        for obstacle in dead_to_remove:
            self.bouncing_obstacle_objects.discard(obstacle)

    def _move_item(self, old_r: int, old_c: int, new_r: int, new_c: int, item_type: CellType) -> None:
        if self.grid is None:
            return
        if self.grid[old_r, old_c] != item_type:
            print(
                f"Warning: Attempted to move item type {item_type.name} from ({old_r},{old_c}), "
                f"but found {CellType(self.grid[old_r, old_c]).name}"
            )
            return
        if self.grid[new_r, new_c] != CellType.EMPTY:
            print(
                f"Warning: Attempted to move item type {item_type.name} to ({new_r},{new_c}), "
                f"but cell is not empty ({CellType(self.grid[new_r, new_c]).name})"
            )
            return

        self.grid[new_r, new_c] = item_type
        self.grid[old_r, old_c] = CellType.EMPTY

        self._remove_from_specific_set(item_type, (old_r, old_c))
        self._add_to_specific_set(item_type, (new_r, new_c))
        self.empty_cells.add((old_r, old_c))
        self.empty_cells.remove((new_r, new_c))

    def _remove_from_specific_set(self, item_type: CellType, cell: Tuple[int, int]) -> None:
        if item_type == CellType.FOOD:
            self.food_cells.discard(cell)
        elif item_type == CellType.OBSTACLE:
            self.obstacle_cells.discard(cell)
        elif item_type == CellType.MOVEABLE_OBSTACLE:
            self.moveable_obstacle_cells.discard(cell)
            self.moved_obstacles.discard(cell)
        elif item_type == CellType.GRABBABLE_OBSTACLE:
            self.grabbable_obstacle_cells.discard(cell)
            self.placed_obstacles.discard(cell)

    def is_within_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _get_adjacent_cells(
        self,
        r: int,
        c: int,
        target_types: Set[CellType],
        directions: Sequence[Tuple[int, int]] | None = None,
    ) -> List[Tuple[int, int]]:
        dirs = directions if directions is not None else self.ADJACENT_DIRECTIONS
        adjacent_cells: List[Tuple[int, int]] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if self.is_within_bounds(nr, nc) and CellType(self.grid[nr, nc]) in target_types:
                adjacent_cells.append((nr, nc))
        return adjacent_cells

    def get_adjacent_empty_cells(self, r: int, c: int) -> List[Tuple[int, int]]:
        return self._get_adjacent_cells(r, c, {CellType.EMPTY})

    def get_adjacent_food(self, r: int, c: int) -> List[Tuple[int, int]]:
        return self._get_adjacent_cells(r, c, {CellType.FOOD})

    def get_adjacent_and_diagonal_food(self, r: int, c: int) -> List[Tuple[int, int]]:
        return self._get_adjacent_cells(r, c, {CellType.FOOD}, directions=self.ALL_DIRECTIONS)

    def get_adjacent_grabbable_obstacles(self, r: int, c: int) -> List[Tuple[int, int]]:
        return self._get_adjacent_cells(r, c, {CellType.GRABBABLE_OBSTACLE})

    def get_empty_cells(self) -> Set[Tuple[int, int]]:
        return self.empty_cells.copy()

    def update_all_sets(self) -> None:
        if Config.DEBUG:
            print("Warning: update_all_sets called. This might indicate inconsistent state updates elsewhere.")
        self.empty_cells = set()
        self.food_cells = set()
        self.obstacle_cells = set()
        self.moveable_obstacle_cells = set()
        self.grabbable_obstacle_cells = set()

        if self.grid is None:
            return

        for r in range(self.size):
            for c in range(self.size):
                cell = (r, c)
                cell_type = CellType(self.grid[r, c])
                if cell_type == CellType.EMPTY:
                    self.empty_cells.add(cell)
                elif cell_type == CellType.FOOD:
                    self.food_cells.add(cell)
                elif cell_type == CellType.OBSTACLE:
                    self.obstacle_cells.add(cell)
                elif cell_type == CellType.MOVEABLE_OBSTACLE:
                    self.moveable_obstacle_cells.add(cell)
                elif cell_type == CellType.GRABBABLE_OBSTACLE:
                    self.grabbable_obstacle_cells.add(cell)
        self.food_remaining = len(self.food_cells)


class BouncingObstacle:
    """Represents an obstacle that moves around the grid with different behaviors."""

    ALL_DIRECTIONS_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    _NEXT_ID = 0

    def __init__(self, initial_x: int, initial_y: int, initial_direction: str, grid_size: int) -> None:
        self._id = BouncingObstacle._NEXT_ID
        BouncingObstacle._NEXT_ID += 1

        self.x = initial_x
        self.y = initial_y
        self.direction = initial_direction
        self.grid_size = grid_size

        self.type = random.choice(["rotating", "inverting", "counter_rotating", "wandering"])
        self.tick, self.grid_value = self.determine_type_properties()

        self.dead = False
        self.bounced_agents: Set[Tuple[int, int]] = set()
        self.max_attempts = 10

    def __hash__(self) -> int:  # pragma: no cover - identity hashing
        return hash(self._id)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - identity equality
        return isinstance(other, BouncingObstacle) and self._id == other._id

    def determine_type_properties(self) -> Tuple[int, CellType]:
        if self.type == "inverting":
            return 1, CellType.BOUNCING_OBSTACLE
        if self.type in ("rotating", "counter_rotating"):
            grid_value = (
                CellType.LOOPING_OBSTACLE
                if self.type == "rotating"
                else CellType.COUNTER_LOOPING_OBSTACLE
            )
            return 1, grid_value
        if self.type == "wandering":
            return 1, CellType.WANDERING_OBSTACLE
        return 1, CellType.BOUNCING_OBSTACLE

    def invert_direction(self) -> None:
        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        self.direction = opposites.get(self.direction, self.direction)

    def rotate_direction(self, clockwise: bool = True) -> None:
        order = ["up", "right", "down", "left"]
        if not clockwise:
            order.reverse()
        try:
            current_index = order.index(self.direction)
            self.direction = order[(current_index + 1) % len(order)]
        except ValueError:
            print(f"Warning: Invalid direction '{self.direction}' encountered during rotation.")

    def handle_collision_behavior(self) -> None:
        if self.type == "inverting":
            self.invert_direction()
        elif self.type == "rotating":
            self.rotate_direction(clockwise=True)
        elif self.type == "counter_rotating":
            self.rotate_direction(clockwise=False)

    def _handle_collision_behavior_wandering(self, other_objects: Set[Tuple[int, int]]) -> bool:
        valid_directions = []
        for potential_dir, (dr, dc) in BouncingObstacle.ALL_DIRECTIONS_DELTAS.items():
            next_x, next_y = self.x + dr, self.y + dc
            if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                if (next_x, next_y) not in other_objects:
                    valid_directions.append(potential_dir)

        if valid_directions:
            self.direction = random.choice(valid_directions)
            return False
        self.dead = True
        return True

    def update(self, grid: np.ndarray, other_objects: Set[Tuple[int, int]], tick: int) -> None:
        if tick % self.tick != 0 or self.dead:
            return

        for attempt in range(self.max_attempts):
            pending_x, pending_y = self.update_position_based_on_direction()

            is_in_bounds = 0 <= pending_x < self.grid_size and 0 <= pending_y < self.grid_size
            is_blocked = (pending_x, pending_y) in other_objects if is_in_bounds else True

            if is_in_bounds and not is_blocked:
                grid[self.x, self.y] = CellType.EMPTY
                other_objects.discard((self.x, self.y))

                self.x, self.y = pending_x, pending_y

                grid[self.x, self.y] = self.grid_value
                other_objects.add((self.x, self.y))
                return

            if is_in_bounds and grid[pending_x, pending_y] == CellType.AGENT:
                self.bounced_agents.add((pending_x, pending_y))

            became_dead = False
            if self.type == "wandering":
                became_dead = self._handle_collision_behavior_wandering(other_objects)
            else:
                self.handle_collision_behavior()

            if became_dead:
                break

            if self.type == "inverting" and attempt == 0:
                continue

        if not self.dead:
            self.dead = True

    def update_position_based_on_direction(self) -> Tuple[int, int]:
        dr, dc = BouncingObstacle.ALL_DIRECTIONS_DELTAS.get(self.direction, (0, 0))
        return self.x + dr, self.y + dc


__all__ = ["BouncingObstacle", "GridEnvironment"]
