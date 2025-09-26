# FunGrid

Basic grid environment implemented in Python and pygame.

## Running the training simulation

Install the project dependencies (``pygame`` and ``torch`` are the primary
requirements) and launch the PPO training loop with:

```bash
python FunGrid.py
```

This convenience wrapper instantiates :class:`fun_grid.simulation.EnvironmentSimulation`,
which coordinates the grid environments, agents, PPO optimiser, and the optional
pygame visualiser.

If you prefer to target the package entry point directly (for example when
embedding the project in another codebase), you can run:

```bash
python -m fun_grid.main
```

## Running the interactive pygame demo

To explore the legacy game mode, ensure ``pygame`` is installed and start the
arcade experience via the CLI:

```bash
python -m fun_grid.main --mode game
```

Use the arrow keys to move the agent, collect food, and avoid obstacles.

## Package layout

- ``fun_grid/config.py`` – centralised configuration and optional user overrides.
- ``fun_grid/cell_types.py`` – enumerations for cell and obstacle semantics.
- ``fun_grid/constants.py`` – shared screen/grid dimensions for the pygame game.
- ``fun_grid/grid.py`` – environment generation, obstacle logic, and movement.
- ``fun_grid/entities.py`` – PPO agents plus the lightweight pygame entities.
- ``fun_grid/game.py`` – interactive pygame loop for the legacy demo.
- ``fun_grid/rl/`` – neural policy definition and PPO/transition utilities.
- ``fun_grid/visualizer.py`` – pygame renderer for one or more environments.
- ``fun_grid/simulation.py`` – orchestration of training loops and PPO updates.
