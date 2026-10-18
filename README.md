# CPax

**CPax** is an open-source, JAX-based library for computational physics.
It aims to provide reusable physical models, numerical simulation workflows,
and differentiable scientific computing tools. The current physics-first API
supports continuous-time initial-value problems (IVPs) using
[Diffrax](https://docs.kidger.site/diffrax/) for numerical integration.
Other physical models and simulation methods remain under development.

- **Documentation:** https://cpax.readthedocs.io/en/latest/index.html

## Quickstart: harmonic oscillator

Define a system, specify an initial-value problem, choose a solver, and
inspect the resulting trajectory. For a unit-mass harmonic oscillator,
`dq/dt = p` and `dp/dt = -k q`:

```python
import cpax
import jax.numpy as jnp

from cpax.dynamics import VectorFieldSystem


def oscillator(t, state, args):
    del t  # The equations do not depend explicitly on time.
    q, p = state
    return p, -args["k"] * q


system = VectorFieldSystem(oscillator)

problem = cpax.IVP(
    system=system,
    y0=(jnp.array(1.0), jnp.array(0.0)),
    t_span=(0.0, 10.0),
    args={"k": jnp.array(1.0)},
)

solution = cpax.solve(
    problem,
    saveat=cpax.SaveAt.linspace(1000),
)

q, p = solution.ys
print("Integration successful:", bool(solution.is_success))
print("Saved points:", solution.ts.shape[0])
```

The state is a tuple of coordinates and momenta, and the saved trajectory
preserves that structure: both `q` and `p` have shape `(1000,)`.
Use `solution.ts`, `q`, and `p` for time series or a phase-space plot.

The four layers are **System → Problem → Solver → Solution**. A system
describes equations of motion; an `IVP` supplies initial conditions and
an interval; `cpax.solve` handles integration; `Solution` contains saved
states, status, and diagnostics. The default ``solver="auto"`` currently chooses Diffrax's Tsit5 for all
supported IVPs. This is a general-purpose non-stiff default, not a guarantee
of suitability for stiff or long-time structure-preserving simulations.
Use ``solver=diffrax.Tsit5()`` (or another native Diffrax solver) to select
the numerical algorithm explicitly. Inspect ``solution.metadata["solver"]``
and ``solution.metadata["solver_selection"]`` for the actual choice.

## Documentation

- [Continuous-time dynamical systems](docs/source/dynamics.rst): vector fields,
  system parameters and JAX-compatible states.
- [Initial-value problems](docs/source/problems.rst): initial conditions and
  problem validation.
- [Solving IVPs](docs/source/solvers.rst): solver choices, sampling, tolerances,
  failure handling and JAX transformations.
- [Numerical solutions](docs/source/results.rst): saved trajectories, statuses
  and diagnostic information.

The existing `cpax.ode` integrators, including leapfrog, remain available
during the transition to the physics-first interfaces.

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

## Releases

CPax uses calendar versions `YYYY.M.N` (for example, `2026.10.0`), with `N`
starting at zero for each new UTC calendar month. The authoritative version
is `[project].version` in `pyproject.toml`; `cpax.__version__` is read from
installed distribution metadata. The historical `v2025.05.0` tag is preserved.

To prepare a release, navigate to **Actions → Manual CalVer Release → Run workflow**
on GitHub. Dry runs can use `main` or a feature branch: leave **dry_run**
selected (the default) to validate a release in a temporary checkout without
commits, tags, or remote changes. Publishing requires selecting `main`.

To publish, run the same workflow with **dry_run** disabled. It checks that
`main` has not changed, computes the next calendar version, updates
`pyproject.toml` and `uv.lock`, runs tests and static checks, builds a
wheel and source distribution, checks a clean wheel installation, then creates
one release commit and an annotated `vYYYY.M.N` tag. The commit and tag are
pushed atomically. Concurrent workflow runs are serialised.

If preparation fails, the remote repository is unchanged. If publication
fails because `main` has advanced or the tag exists, inspect the error and
rerun from the latest `main`; never force-push or overwrite a release tag.
Pushing a tag does not automatically publish to PyPI or create a GitHub Release.
