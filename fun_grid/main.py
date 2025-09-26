"""Command-line entry point for running the FunGrid training simulation."""

from __future__ import annotations

from .simulation import EnvironmentSimulation


def main() -> None:
    """Instantiate the training environment and start the run loop."""

    simulation = EnvironmentSimulation()
    simulation.run()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
