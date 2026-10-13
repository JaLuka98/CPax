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
