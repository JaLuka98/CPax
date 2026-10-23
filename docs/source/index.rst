CPax — computational physics with JAX
=====================================

CPax is an open-source, JAX-based library for defining, simulating, and
analysing physical systems. Its physics-first continuous-time workflow
separates **System → Problem → Solver → Solution**.

CPax currently supports initial-value problems for dynamical systems and
conservative classical mechanics. Molecular dynamics, spin models and other
domains are long-term directions, not currently available capabilities.

Installation
------------

CPax is distributed on `PyPI <https://pypi.org/project/CPax/>`_ and requires
Python 3.12 or newer. Install it using your preferred Python package manager:

.. code-block:: console

   pip install cpax

You may alternatively use uv or Pixi. See :doc:`getting-started` for
virtual environment setup, installation options and a first simulation.

Quickstart: harmonic oscillator
-------------------------------

.. code-block:: python

   import cpax
   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   system = NewtonianSystem(potential=lambda q: q**2 / 2, mass=1.0)
   problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 10.0))
   solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(500))

   print(solution.ys.q)  # Position at sampled times
   print(solution.ys.p)  # Canonical momentum

The default solver is currently Diffrax's Tsit5, appropriate as a starting
point for many non-stiff IVPs but not necessarily for long-time Hamiltonian
structure preservation.

A planetary simulation example
------------------------------

This five-body example integrates the Sun, Earth, Mars, Jupiter and an
inbound comet. The view focuses on the inner 10 AU; the comet starts
outside the frame and moves inward.

.. image:: _static/nbody_simulation.gif
   :alt: Animated numerical trajectories of the Sun, Earth, Mars, Jupiter and a comet
   :width: 520px

This example can be found in :doc:`examples/advanced`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Core concepts

   concepts/systems
   concepts/problems
   concepts/solvers
   concepts/solutions

.. toctree::
   :maxdepth: 2
   :caption: Physics domains

   physics/mechanics

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/simple
   examples/advanced

.. toctree::
   :maxdepth: 2
   :caption: Development and releases

   development/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   autoapi/cpax/index
