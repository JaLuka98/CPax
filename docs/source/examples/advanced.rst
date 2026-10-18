Advanced examples
=================

Sun, Earth, Mars and a comet
----------------------------

The four-body gravitational example demonstrates coupled two-dimensional
coordinates, automatic force differentiation, trajectory sampling,
energy/angular momentum diagnostics and animation.

.. image:: ../_static/nbody_simulation.gif
   :alt: Sun, Earth, Mars and comet orbiting under Newtonian gravity
   :width: 520px

.. literalinclude:: ../../../examples/nbody_gravity.py
   :language: python

The example uses an adaptive solver. It is a compact visual demonstration,
not a high-precision ephemeris or an assurance of long-time symplectic
integration. In the unsoftened model a collision between distinct particles
is a physical singularity.

Retained symplectic leapfrog
----------------------------

For separable Hamiltonians, the kick-drift-kick leapfrog integrator
remains available as a specialised algorithm until it is integrated into
the common solver interface.

.. literalinclude:: ../../../examples/leapfrog_two_body.py
   :language: python
