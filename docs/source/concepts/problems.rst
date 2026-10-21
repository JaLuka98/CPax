Problems
========

An initial-value problem specifies a system, initial state, and integration
interval, independently of how the equations will be solved.

.. code-block:: python

   import cpax
   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   system = NewtonianSystem(potential=lambda q: q**2 / 2, mass=1.0)
   problem = cpax.IVP(
       system=system,
       y0=PhaseSpaceState(q=1.0, p=0.0),
       t_span=(0.0, 10.0),
   )

``PhaseSpaceState.p`` denotes the canonical momentum. The
initial state can also be another array-valued JAX PyTree when using a
compatible system. Only increasing forward time intervals are currently
supported. Physical domain constraints are not enforced by ``IVP``.
