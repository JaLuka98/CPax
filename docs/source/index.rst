CPax documentation
==================

CPax is a JAX-based computational physics library. Its current
physics-first workflow lets you define continuous-time equations of
motion, construct an initial-value problem (IVP), integrate it using
Diffrax, and inspect a structured numerical solution.

The core separation is **System → Problem → Solver → Solution**.
Physical equations are independent of numerical integration; this
workflow currently supports continuous-time IVPs, not arbitrary PDEs,
boundary-value problems, or Monte Carlo algorithms.

Start with :doc:`dynamics` to define a system, then :doc:`problems`
and :doc:`solvers` for a complete simulation. See :doc:`results`
for trajectory shapes, numerical diagnostics, and failure handling.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   dynamics
   problems
   solvers
   results
   leapfrog

The :doc:`leapfrog` page documents a separate legacy, symplectic
integrator. It remains supported while CPax develops its new
physics-oriented interfaces.
