Advanced examples
=================

A comet in a planetary system
-----------------------------

The Sun, Earth, Mars and Jupiter interact gravitationally with a small
comet. We initialise the planets on approximately circular, coplanar
orbits. The comet starts inbound at 15 AU; its velocity is chosen from
an *unperturbed* Sun--comet ellipse with semi-major axis 40 AU and
perihelion 1.5 AU. Once the planets are included, it no longer follows
that ideal Kepler ellipse exactly.

.. image:: ../_static/nbody_simulation.gif
   :alt: Animated Sun, Earth, Mars, Jupiter and a comet under Newtonian gravity
   :width: 520px

The animation is centred on the inner system (within 10 AU of the origin),
so the comet begins outside the frame. The planets perturb its orbit
through their gravitational forces; a particularly close encounter is
not required for a deviation from the Sun-only trajectory.

The example constructs a five-body Newtonian potential, obtains the forces
with JAX automatic differentiation, integrates the coupled equations and
plots the resulting positions. It also calculates total energy and the
out-of-plane angular momentum to check numerical drift. It is a simple
illustration of N-body dynamics, not a precision Solar System ephemeris:
the initial orbits are idealised and the adaptive Tsit5 solver does not
preserve symplectic structure.

For an instructive comparison, integrate the comet in the Sun-only
potential with the same initial conditions and compare the positions
against the full five-body result. A Kepler ellipse is an exact reference
only for the isolated two-body problem.

.. literalinclude:: ../../../examples/nbody_gravity.py
   :language: python

Retained symplectic leapfrog
----------------------------

The following binary-orbit example uses the retained kick-drift-kick
leapfrog integrator. With a fixed time step and a separable Hamiltonian,
the method is symplectic. It is currently separate from ``cpax.solve``.

.. literalinclude:: ../../../examples/leapfrog_two_body.py
   :language: python
