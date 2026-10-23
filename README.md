# CPax

**CPax** is an open-source, JAX-based library for computational physics.
It separates physical **systems**, **problems**, **solvers** and
**solutions** in reusable workflows. The current implementation supports
continuous-time initial-value problems and classical mechanics; additional
physical domains are planned, not yet implemented.

**[Documentation](https://cpax.readthedocs.io/en/latest/)** ·
**[PyPI](https://pypi.org/project/CPax/)** ·
**[Source](https://github.com/JaLuka98/CPax)** ·
**[Issues](https://github.com/JaLuka98/CPax/issues)**

## Installation

CPax is published on [PyPI](https://pypi.org/project/CPax/) and requires
Python 3.12 or newer. Install it in your chosen Python environment:

```shell
pip install cpax
```

Other PyPI-compatible installers work as well, including `uv pip install cpax`,
`uv add cpax`, or `pixi add --pypi cpax` (in an existing Pixi workspace).
See the [installation guide](https://cpax.readthedocs.io/en/latest/getting-started.html)
for details.

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
plots, and the animated Sun–Earth–Mars–Jupiter–comet showcase.

The default `solver="auto"` currently selects Diffrax Tsit5 for IVPs.
It does not infer stiffness or symplectic structure. A standalone
symplectic leapfrog integrator is preserved in `cpax.ode`.

## Development and releases

CPax uses **uv**, **Ruff**, and **ty** for reproducible development.
See the [development guide](https://cpax.readthedocs.io/en/latest/development/index.html)
for setting up a checkout, running tests and linting, and building the docs.

Releases use calendar versions `YYYY.M.N` and are published to PyPI through
a manually approved GitHub Actions workflow. The
[versioning and release guide](https://cpax.readthedocs.io/en/latest/development/versioning.html)
explains the policy and publication process.
