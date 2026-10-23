Getting started
===============

Requirements
------------

CPax is available from `PyPI <https://pypi.org/project/CPax/>`_ and requires
Python 3.12 or newer. Its principal numerical backend uses JAX and Diffrax.
Follow the `JAX installation instructions <https://docs.jax.dev/en/latest/installation.html>`_
if you need a specific GPU or accelerator configuration; compatible JAX
wheels and platform requirements can differ.

Installation
------------

For an ordinary Python environment, install CPax with pip:

.. code-block:: console

   python -m pip install cpax

You can instead use another PyPI-compatible tool. For example:

.. code-block:: console

   uv pip install cpax

Or, when managing a project with uv:

.. code-block:: console

   uv add cpax

In an existing Pixi workspace, the equivalent is:

.. code-block:: console

   pixi add --pypi cpax

These are *alternatives*, not sequential installation steps. Use the
environment manager that suits your project. ``uv sync`` in the CPax
repository is a **contributor workflow**, not an installation requirement
for library users.

First simulation
----------------

A harmonic oscillator can be described by a potential and initial
canonical coordinates:

.. code-block:: python

   import cpax
   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   system = NewtonianSystem(potential=lambda q: 0.5 * q**2, mass=1.0)
   problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 10.0))
   solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(101))

   print(solution.ts)
   print(solution.ys.q)

To work on CPax itself, use the :doc:`development/index` instead.
