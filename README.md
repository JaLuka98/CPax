# CPax

**CPax** is an open-source, JAX-based library for computational physics.
It separates physical **systems**, **problems**, **solvers** and
**solutions** in reusable workflows. The current implementation supports
continuous-time initial-value problems and classical mechanics; additional
physical domains are planned, not yet implemented.

**[Documentation](https://cpax.readthedocs.io/en/latest/)** ·
**[Source](https://github.com/JaLuka98/CPax)** ·
**[Issues](https://github.com/JaLuka98/CPax/issues)**

## Installation

Clone the repository, then run `uv sync` to create a reproducible
development environment. CPax currently requires Python 3.12 or newer.

## Quickstart

```python
import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState

system = NewtonianSystem(potential=lambda q: q**2 / 2, mass=1.0)
problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 10.0))
solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(500))

print(solution.ys.q)  # Position
print(solution.ys.p)  # Canonical momentum
```

The same workflow supports the conservative Duffing oscillator, the
built-in `cpax.dynamics.LorenzSystem`, and multibody Newtonian gravity.
See the [examples](examples/) and documentation for complete scripts,
plots, and the animated Sun–Earth–Mars–comet showcase.

The default `solver="auto"` currently selects Diffrax Tsit5 for IVPs.
It does not infer stiffness or symplectic structure. A standalone
symplectic leapfrog integrator is preserved in `cpax.ode`.

## Development

CPax uses **uv**, **Ruff**, and **ty**:

```shell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run sphinx-build -W -b html docs/source docs/_build/html
```

Releases use calendar versions `YYYY.M.N`. The manual GitHub Actions
release workflow supports non-publishing previews and publishing from
`main`. See the workflow for its latest controls and validation.
