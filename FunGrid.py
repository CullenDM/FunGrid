"""Convenience launcher for the FunGrid training simulation."""

from fun_grid.simulation import EnvironmentSimulation


def main() -> None:
    """Instantiate and run the environment simulation."""

    simulation = EnvironmentSimulation()
    simulation.run()


if __name__ == "__main__":
    main()
