Development
===========

This guide is for contributors working on CPax itself. Users who only want
to install the published library should follow :doc:`../getting-started`.

Local environment
-----------------

Clone `the CPax repository <https://github.com/JaLuka98/CPax>`_, change
into its root directory and install the locked development environment.
CPax requires Python 3.12 or newer and uses `uv <https://docs.astral.sh/uv/>`_
for managing its own development dependencies:

.. code-block:: console

   git clone https://github.com/JaLuka98/CPax.git
   cd CPax
   uv sync

Run checks locally before opening a pull request:

.. code-block:: console

   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   uv run sphinx-build -W -b html docs/source docs/_build/html

GitHub Actions runs the test suite on supported Python versions and checks
documentation, linting, typing and distribution builds. Pull requests should
be focused and reviewable, with tests for changed behaviour and documentation
for any public API change.

Source layout
-------------

The physics implementation lives under ``src/cpax/``, automated tests in
``tests/``, runnable examples in ``examples/``, and documentation under
``docs/source/``. Prefer changes that maintain the System → Problem →
Solver → Solution separation and preserve JAX transformations where applicable.

Releases and documentation versions
-----------------------------------

Read :doc:`versioning` for the versioning policy and Read the Docs versions.
The step-by-step, security-sensitive publication procedure is in
:doc:`publishing`.

.. toctree::
   :maxdepth: 1

   versioning
   publishing
   force-investigation
