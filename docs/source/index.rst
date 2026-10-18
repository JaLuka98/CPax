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

CPax is currently distributed from its source repository. Clone the
`CPax GitHub repository <https://github.com/JaLuka98/CPax>`_ and install
its development environment with ``uv sync``.

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

Planetary simulation showcase
-----------------------------

Simulate an interacting Sun, Earth, Mars, and comet using the same
System → Problem → Solver → Solution workflow.

.. image:: _static/nbody_simulation.gif
   :alt: Animated numerical trajectories of the Sun, Earth, Mars, and a comet
   :width: 520px

The complete executable code is in :doc:`examples/advanced`.

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
   :caption: API reference

   autoapi/cpax/index
