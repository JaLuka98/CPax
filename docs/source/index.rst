CPax — computational physics with JAX
=====================================

CPax is an open-source, JAX-based library for defining, simulating, and
analysing physical systems. Its physics-first continuous-time workflow
separates **System → Problem → Solver → Solution**.

CPax currently supports initial-value problems for dynamical systems and
conservative classical mechanics. Molecular dynamics, spin models and other
domains are long-term directions, not currently available capabilities.

As an example, the following animation shows a five-body simulation of the Sun, Earth, Mars, Jupiter and an inbound comet.
The view focuses on the inner 10 AU; the comet starts outside the frame and moves inward.

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
