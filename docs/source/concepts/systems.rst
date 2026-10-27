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

CPax implements some common systems via convenience classes.
For example, the Lorenz system is a three-dimensional system with
parameters ``sigma``, ``rho`` and ``beta``. Depending on the parameter values,
it can exhibit chaotic behavior. The default parameters are
``sigma=10``, ``rho=28``, and ``beta=8/3``. The system can be instantiated with custom parameters as needed.

.. code-block:: python

   from cpax.dynamics import LorenzSystem

   system = LorenzSystem()  # sigma=10, rho=28, beta=8/3
   customised = LorenzSystem(sigma=12.0, rho=30.0)

Mechanics provides ``NewtonianSystem``, ``HamiltonianSystem`` and
``ForcedNewtonianSystem`` based on physical potentials, Hamiltonians and
applied generalised forces rather than hand-written ODEs.

The generic vector field receives run-specific parameters through
``IVP.args``. Physics-first mechanics and Lorenz models store their
parameters on the system and do not accept non-None IVP args.

For supported models and solver configurations, JAX compilation, vectorisation
and differentiation can be used. Solver and floating-point limitations apply.
