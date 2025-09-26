"""Constants and configuration values for the FunGrid project."""

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)  # White
GRID_COLOR = (0, 0, 0)  # Black
OBSTACLE_COLOR = (128, 128, 128)  # Grey
FOOD_COLOR = (0, 255, 0)  # Green
CELL_SIZE = 40  # Size of each grid cell
FOOD_RADIUS = CELL_SIZE // 4  # Radius of the food
BORDER_WIDTH = 3  # Width of the border around the screen
FPS = 10
NUM_AGENTS = 5
NUM_OBSTACLES = 20
NUM_FOOD = 20
BATCH_SIZE = 100

AGENT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]
