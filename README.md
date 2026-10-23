# CPax

**CPax** is an open-source, JAX-based library for computational physics.
It separates physical **systems**, **problems**, **solvers** and
**solutions** in reusable workflows. The current implementation supports
continuous-time initial-value problems and classical mechanics; additional
physical domains are planned,  but not yet implemented.

- GitHub: https://github.com/JaLuka98/CPax
- Documentation: https://cpax.readthedocs.io
- PyPI: https://pypi.org/project/CPax/
- Examples: see the [examples](https://github.com/JaLuka98/CPax/tree/main/examples) directory in this repository
- Issues: https://github.com/JaLuka98/CPax/issues

## Installation

CPax is published on [PyPI](https://pypi.org/project/CPax/) and requires
Python 3.12 or newer. The latest release or a specific version can be
installed with `pip`:

```shell
pip install cpax
pip install cpax==2026.10.0  # specific version
```

Other PyPI-compatible installers work as well, including `uv` or `pixie`.
See the [installation guide](https://cpax.readthedocs.io/stable/getting-started.html)
for details.

## Quickstart

CPax provides a simple interface for defining and solving initial-value problems (IVPs) in classical mechanics.
For example, the classical one-dimensional haromonic oscillator can be defined and solved for an initial condition of `q=1.0, p=0.0` over a timespace of `[0,10]` as follows:

```python
import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState

system = NewtonianSystem(potential=lambda q: q**2 / 2, mass=1.0)
problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 10.0))
solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(500))

print(solution.ys.q)  # Position
print(solution.ys.p)  # Momentum
```

The same workflow supports other systems.
Consult the [examples](https://github.com/JaLuka98/CPax/tree/main/examples) and the documentation
for further information and code to get started.

## Development and releases

Contributions are welcome! See the [contributing guide](https://cpax.readthedocs.io/stable/development/contributing.html)

CPax uses **uv**, **Ruff**, and **ty** for reproducible development.
See the [development guide](https://cpax.readthedocs.io/stable/development/index.html)
for setting up a checkout, running tests and linting, and building the docs.
