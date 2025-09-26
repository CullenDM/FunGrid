"""Command-line entry point for running the FunGrid simulation."""

from .simulation import EnvironmentSimulation


def main() -> None:
    """Instantiate the training environment and start the run loop."""

    simulation = EnvironmentSimulation()
    simulation.run()
