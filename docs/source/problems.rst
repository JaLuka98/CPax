Initial-value problems
======================

An initial-value problem (IVP) combines a continuous-time system with an
initial state and a time interval, without choosing an integration method.
Its equation is ``dy/dt = system.vector_field(t, y, args)`` with
``y(t0) = y0``.

.. code-block:: python

   import jax.numpy as jnp
   from cpax import IVP
   from cpax.dynamics import VectorFieldSystem

   def decay(t, y, args):
       return -args["rate"] * y

   system = VectorFieldSystem(decay)
   problem = IVP(
       system=system,
       y0=jnp.array(1.0),
       t_span=(0.0, 10.0),
       args={"rate": jnp.array(0.5)},
   )

A system may be reused with different initial states, time intervals and
``args``. The optional ``args`` field forwards data for individual
vector-field evaluations; model parameters may instead live on the
Equinox system module.

The initial state may be any non-empty JAX PyTree whose leaves contain
floating-point or complex arrays, or Python numerical scalars. CPax
converts Python numerical scalars to JAX arrays and rejects explicitly
integer-valued or Boolean arrays rather than silently changing their
meaning. Nested tuples and mappings are preserved.

Time endpoints must be scalar, finite and increasing: currently
``t_span=(t0, t1)`` requires ``t1 > t0``. Checks of concrete time
values take place when constructing the problem. For JAX-traced times,
value-dependent checks are deferred to the solver. Model-specific
coordinate domains are not validated by ``IVP``.

Solving a problem
-----------------

Constructing an ``IVP`` does **not** perform integration. Supply it
to ``cpax.solve`` together with a concrete numerical algorithm:

.. code-block:: python

   import cpax
   import diffrax

   solution = cpax.solve(
       problem,
       solver=diffrax.Tsit5(),
       saveat=cpax.SaveAt.linspace(101),
   )
   times, values = solution.ts, solution.ys

The problem and system remain independent of the solver backend.
Read :doc:`solvers` for numerical method selection, step-size control,
output sampling and failure handling, and :doc:`results` for the
returned solution structure.
