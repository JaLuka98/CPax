Advanced examples
=================

A comet in a planetary system
-----------------------------

This example shows a planetary system with earth, mars, jupiter and a comet.
The comet is initialized with a high eccentricity orbit that crosses the orbits of the planets.

.. image:: ../_static/nbody_simulation.gif
   :alt: Sun, Earth, Mars and comet orbiting under Newtonian gravity
   :width: 520px

This example demonstrates coupled two-dimensional coordinates, automatic force differentiation, trajectory sampling, energy/angular momentum diagnostics and animation of the trajectories.
Compared to a simple central force analytical solution, only considering the influence of the sun, the comet's trajectory is perturbed by the planets due to the close encounters.
If you are curious and want to dive deeper, it can be instructive to compare the numerical solution for the comet's position and its velocity with the corresponding analytical solutions

.. literalinclude:: ../../../examples/nbody_gravity.py
   :language: python

Retained symplectic leapfrog
----------------------------

For separable Hamiltonians, the kick-drift-kick leapfrog integrator
remains available as a specialised algorithm until it is integrated into
the common solver interface.

.. literalinclude:: ../../../examples/leapfrog_two_body.py
   :language: python
