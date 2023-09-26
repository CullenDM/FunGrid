"""
Grid Environment Game
Author: Cullen Maglothin
Date: September 25, 2023

This game simulates agents navigating a grid environment, avoiding obstacles, and collecting food.
"""

import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)  # White
GRID_COLOR = (0,0,0)  # Black
OBSTACLE_COLOR = (128, 128, 128)  # Grey
FOOD_COLOR = (0, 255, 0)  # Green
CELL_SIZE = 40  # Size of each grid cell
FOOD_RADIUS = CELL_SIZE // 4  # Radius of the food
BORDER_WIDTH = 3  # Width of the border around the screen
FPS = 60
NUM_AGENTS = 1
NUM_OBSTACLES = 10
NUM_FOOD = 10

# Create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Grid Environment')

class SpatialGrid:
    """Represents a spatial grid environment for the game."""
    
    def __init__(self, width, height, cell_size):
        """Initialize the spatial grid with given dimensions and cell size."""
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid = [[set() for _ in range(self.rows)] for _ in range(self.cols)]

    def _get_cell_coords(self, x, y):
        """Convert screen coordinates to grid cell coordinates."""
        col = x // self.cell_size
        row = y // self.cell_size
        return col, row

    def insert(self, item, x, y):
        """Insert item at the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        self.grid[col][row].add(item)

    def remove(self, item, x, y):
        """Remove item from the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        self.grid[col][row].discard(item)

    def move(self, item, x_from, y_from, x_to, y_to):
        """Move item from one set of screen coordinates to another."""
        self.remove(item, x_from, y_from)
        self.insert(item, x_to, y_to)

    def get_items(self, x, y):
        """Retrieve all items from the cell at the given screen coordinates."""
        col, row = self._get_cell_coords(x, y)
        return self.grid[col][row]
    
    def get_empty_cells(self):
        """Return a list of coordinates for all empty cells."""
        empty_cells = []
        for col in range(self.cols):
            for row in range(self.rows):
                if not self.grid[col][row]:
                    empty_cells.append((col * self.cell_size, row * self.cell_size))
        return empty_cells
    
    def get_view(self, agent_x, agent_y, size=7):
        """Retrieve a view of the grid around a given agent."""
        half_size = size // 2
        view = []

        for y in range(agent_y - half_size * CELL_SIZE, agent_y + (half_size + 1) * CELL_SIZE, CELL_SIZE):
            row = []
            for x in range(agent_x - half_size * CELL_SIZE, agent_x + (half_size + 1) * CELL_SIZE, CELL_SIZE):
                items = self.get_items(x, y)
                if any(isinstance(item, Agent) for item in items):
                    row.append([0, 0, 0, 1])
                elif any(isinstance(item, Obstacle) for item in items):
                    row.append([0, 1, 0, 0])
                elif any(isinstance(item, Food) for item in items):
                    row.append([0, 0, 1, 0])
                else:
                    row.append([1, 0, 0, 0])
            view.append(row)

        return view

class Entity:
    """Base class for all entities in the game."""
    
    def __init__(self, x, y):
        """Initialize the entity with given coordinates."""
        self.x = x
        self.y = y

    def render(self, surface):
        """Render the entity on the game surface."""
        pass

class Obstacle(Entity):
    """Represents an obstacle in the game."""
    
    def render(self, surface):
        """Render the obstacle on the game surface."""
        pygame.draw.rect(surface, OBSTACLE_COLOR, (self.x, self.y, CELL_SIZE, CELL_SIZE))

class Food(Entity):
    """Represents a food item in the game."""
    
    def render(self, surface):
        """Render the food item on the game surface."""
        pygame.draw.circle(surface, FOOD_COLOR, (self.x + CELL_SIZE // 2, self.y + CELL_SIZE // 2), FOOD_RADIUS)

class Agent(Entity):
    """Represents an agent in the game."""
    
    def __init__(self, x, y, spatial_grid, color=(255, 0, 0)):
        """Initialize the agent with given coordinates, spatial grid, and color."""
        super().__init__(x, y)
        self.color = color
        self.spatial_grid = spatial_grid
        self.food_eaten = 0
        self.steps_since_food = 0
        self.score = 0

    def render(self, surface):
        """Render the agent on the game surface."""
        pygame.draw.rect(surface, self.color, (self.x, self.y, CELL_SIZE, CELL_SIZE))

    def get_reward(self, items_in_target_cell):
        """Determine the reward for the agent based on its interaction with other items."""
        if any(isinstance(item, Obstacle) for item in items_in_target_cell) or \
        any(isinstance(item, Agent) for item in items_in_target_cell):
            return -10

        if any(isinstance(item, Food) for item in items_in_target_cell):
            return 100

        return -1

    def move(self, action, food_list):
        """Move the agent based on the chosen action and update its state."""
        move_map = {
            "up": (0, -CELL_SIZE),
            "down": (0, CELL_SIZE),
            "left": (-CELL_SIZE, 0),
            "right": (CELL_SIZE, 0)
        }

        dx, dy = move_map.get(action, (0, 0))
        new_x, new_y = self.x + dx, self.y + dy

        items_in_target_cell = self.spatial_grid.get_items(new_x, new_y)
        reward = self.get_reward(items_in_target_cell)

        if reward != -10:
            self.spatial_grid.move(self, self.x, self.y, new_x, new_y)
            self.x, self.y = new_x, new_y
            self.steps_since_food += 1

            if reward == 100:
                food = next(item for item in items_in_target_cell if isinstance(item, Food))
                self.spatial_grid.remove(food, food.x, food.y)
                if food in food_list:
                    food_list.remove(food)
                self.food_eaten += 1
                self.steps_since_food = 0

        return reward, action

    def choose_action(self, state):
        """Choose an action for the agent based on its current state."""
        action = random.choice(["up", "down", "left", "right"])
        return action
    
    def update(self, food_list, state):
        """Update the agent's state and take an action."""
        choice = self.choose_action(state)
        step_reward, action = self.move(choice, food_list)
        self.score += step_reward
        return step_reward, action

class Game:
    """Main game class that handles game logic and rendering."""
    
    def __init__(self, screen):
        """Initialize the game with a given screen."""
        self.screen = screen
        self.episode = 0
        self.steps = 0
        self.score = 0
        self.initialize_game()

    def add_border_obstacles(self, BORDER_WIDTH):
        """Add obstacles around the border of the game screen."""
        # Top and bottom borders
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            for y in range(0, BORDER_WIDTH * CELL_SIZE, CELL_SIZE):
                top_obstacle = Obstacle(x, y)
                bottom_obstacle = Obstacle(x, SCREEN_HEIGHT - CELL_SIZE - y)

                # Inserting into spatial grid
                self.spatial_grid.insert(top_obstacle, x, y)
                self.spatial_grid.insert(bottom_obstacle, x, SCREEN_HEIGHT - CELL_SIZE - y)

                self.obstacles.append(top_obstacle)
                self.obstacles.append(bottom_obstacle)
        
        # Left and right borders, excluding the corners (already added)
        for y in range(BORDER_WIDTH * CELL_SIZE, SCREEN_HEIGHT - BORDER_WIDTH * CELL_SIZE, CELL_SIZE):
            for x in range(0, BORDER_WIDTH * CELL_SIZE, CELL_SIZE):
                left_obstacle = Obstacle(x, y)
                right_obstacle = Obstacle(SCREEN_WIDTH - CELL_SIZE - x, y)

                # Inserting into spatial grid
                self.spatial_grid.insert(left_obstacle, x, y)
                self.spatial_grid.insert(right_obstacle, SCREEN_WIDTH - CELL_SIZE - x, y)

                self.obstacles.append(left_obstacle)
                self.obstacles.append(right_obstacle)

    def add_random_food(self, num_food):
        """Add a random number of food items to the game."""
        # Get a list of all empty cells
        empty_cells = self.spatial_grid.get_empty_cells()
        # Randomly select num_food cells
        food_cells = random.sample(empty_cells, num_food)
        # Create a food object for each cell
        for cell in food_cells:
            food = Food(cell[0], cell[1])
            self.food.append(food)
            self.spatial_grid.insert(food, food.x, food.y)

    def add_random_obstacles(self, num_obstacles):
        """Add a random number of obstacles to the game."""
        # Get a list of all empty cells
        empty_cells = self.spatial_grid.get_empty_cells()
        # Randomly select num_obstacles cells
        obstacle_cells = random.sample(empty_cells, num_obstacles)
        # Create an obstacle object for each cell
        for cell in obstacle_cells:
            obstacle = Obstacle(cell[0], cell[1])
            self.obstacles.append(obstacle)
            self.spatial_grid.insert(obstacle, obstacle.x, obstacle.y)

    def initialize_game(self):
        """Initialize the game state and entities."""
        self.spatial_grid = SpatialGrid(SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
        self.obstacles = []
        self.food = []
        self.add_border_obstacles(BORDER_WIDTH)
        self.add_random_obstacles(NUM_OBSTACLES)
        self.add_random_food(NUM_FOOD)
        
        empty_cells = self.spatial_grid.get_empty_cells()
        agent_start_x, agent_start_y = random.choice(empty_cells)
        self.agents = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for color in colors:
            empty_cells = self.spatial_grid.get_empty_cells()
            agent_start_x, agent_start_y = random.choice(empty_cells)
            agent = Agent(agent_start_x, agent_start_y, self.spatial_grid, color=color)
            self.spatial_grid.insert(agent, agent_start_x, agent_start_y)
            self.agents.append(agent)

    def reset(self):
        """Reset the game state and entities."""
        self.initialize_game()
        self.steps = 0
        self.score = 0
        self.episode += 1

    def update_food(self):
        """Update the food items in the game."""
        if len(self.food) < 10:
            self.add_random_food(1)

    def draw_grid(self):
        """Draw the grid lines on the game screen."""
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

    def get_state(self, agent):
        """Retrieve the current state of the agent."""
        grid_state = self.spatial_grid.get_view(agent.x, agent.y)
        food_eaten = agent.food_eaten
        steps_since_food = agent.steps_since_food
        score = agent.score
        time_step = self.steps
        agent_col = agent.x // CELL_SIZE
        agent_row = agent.y // CELL_SIZE
        return grid_state, food_eaten, steps_since_food, score, time_step, agent_col, agent_row

    def draw_entities(self, entity_list):
        """Draw all entities on the game screen."""
        for entity in entity_list:
            entity.render(self.screen)

    def run(self):
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
                reward, action = agent.update(food_list=self.food, state=state)
                print(f"Episode: {self.episode}, Step: {self.steps}, Reward: {reward}, Action: {action}, Score: {agent.score}")
                

            self.draw_entities(self.agents + self.obstacles + self.food)  # Update this line to include all agents
            self.update_food()
            self.draw_grid()

            pygame.display.flip()
            clock.tick(0)
            self.steps += 1
            

            if self.steps % 100 == 0 and self.steps > 0:
                self.reset()
            
def main():
    """Main function to start the game."""
    game = Game(screen)
    game.run()

if __name__ == "__main__":
    main()
