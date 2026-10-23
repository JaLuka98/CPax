Getting Started
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

For an ordinary Python environment, you can install CPax with pip:

.. code-block:: console

   pip install cpax
   pip install cpax==2026.10.0  # specific version

You can also use other PyPI-compatible tools as you prefer. For example:

.. code-block:: console

   uv add cpax

In an existing Pixi workspace, the equivalent is:

.. code-block:: console

   pixi add --pypi cpax

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

   print(solution.ts) # Time steps
   print(solution.ys.q) # Positions
   print(solution.ys.p) # Momenta

This simple example demonstrates the basic workflow for defining and solving an initial-value problem (IVP) in classical mechanics using CPax.
CPax follows the physics-first approach: A system is defined by its properties, independent of a concrete problem or the numerical method used to solve it.
The IVP is then defined by specifying the system, initial conditions, and the time span for the simulation.
Finally, the problem is solved using the `cpax.solve` function, which returns a solution object containing the time steps and corresponding positions and momenta.
Please refer to the further pages in the documentation for more details on the available systems, solvers, and options for customising your simulations.
Examples can be found in the :doc:`Simple examples <examples/simple>` and :doc:`Advanced examples <examples/advanced>` pages.
