"""Entry point for running the FunGrid game."""

import pygame

from .constants import SCREEN_HEIGHT, SCREEN_WIDTH
from .game import Game


def main() -> None:
    """Initialize pygame and start the game loop."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Environment")

    game = Game(screen)
    game.run()


if __name__ == "__main__":
    main()
