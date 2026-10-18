Systems
=======

A system specifies equations of motion independently of initial conditions
and numerical solvers. Continuous-time models implement
``DynamicalSystem.vector_field(t, state, args=None)``. The state and
its derivative share a JAX PyTree structure.

For arbitrary equations, use ``VectorFieldSystem``:

.. code-block:: python

   import jax.numpy as jnp
   from cpax.dynamics import VectorFieldSystem

   def lorenz_rhs(t, state, params):
       del t
       x, y, z = state
       sigma, rho, beta = params
       return jnp.array([
           sigma * (y - x),
           x * (rho - z) - y,
           x * y - beta * z,
       ])

   system = VectorFieldSystem(lorenz_rhs)

For the standard model a reusable shortcut is available:

.. code-block:: python

   from cpax.dynamics import LorenzSystem

   system = LorenzSystem()  # sigma=10, rho=28, beta=8/3
   customised = LorenzSystem(sigma=12.0, rho=30.0)

Mechanics provides ``NewtonianSystem`` and ``HamiltonianSystem`` based
on physical potentials and Hamiltonians rather than hand-written ODEs.

The generic vector field receives run-specific parameters through
``IVP.args``. Physics-first mechanics and Lorenz models store their
parameters on the system and do not accept non-None IVP args.

For supported models and solver configurations, JAX compilation, vectorisation
and differentiation can be used. Solver and floating-point limitations apply.
