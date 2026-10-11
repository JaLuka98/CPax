# CPax
Computational Physics with Jax

## Development

CPax uses [uv](https://docs.astral.sh/uv/) for reproducible Python environments
and the Astral tools [Ruff](https://docs.astral.sh/ruff/) and
[ty](https://docs.astral.sh/ty/) for linting, formatting and static type checking.

```shell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Run `uv run ruff check --fix .` to fix safe lint issues and
`uv run ruff format .` to format the code. Optional pre-commit hooks are
available via `uvx pre-commit install`, or can be run without installation
using `uvx pre-commit run --all-files`. Type checking currently covers the
library under `src/`; tests and examples are linted and formatted.

## Continuous-time dynamics

Define equations of motion using `cpax.dynamics.VectorFieldSystem`, or
subclass `cpax.dynamics.DynamicalSystem` to represent physical models with
array-valued parameters. See the
[continuous-time dynamics documentation](docs/source/dynamics.rst) for the
interface, PyTree states and supported JAX transformation conventions.
The existing `cpax.ode` integration utilities remain available.
