"""Command-line entry point for running FunGrid."""

from __future__ import annotations

import argparse
from typing import Sequence

from .simulation import EnvironmentSimulation


def main(argv: Sequence[str] | None = None) -> None:
    """Instantiate the requested runtime: training simulation or pygame game."""

    parser = argparse.ArgumentParser(description="Run the FunGrid project")
    parser.add_argument(
        "--mode",
        choices={"simulation", "game"},
        default="simulation",
        help="Select 'simulation' for PPO training or 'game' for the interactive demo.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "game":
        from .game import Game  # Local import to avoid pygame dependency unless requested

        Game().run()
        return

    simulation = EnvironmentSimulation()
    simulation.run()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
