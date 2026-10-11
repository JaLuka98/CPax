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
value-dependent checks are deferred to the eventual solver. Model-specific
coordinate domains are not validated by ``IVP``.

An IVP stores a mathematical problem; constructing one **does not**
perform numerical integration. The ``cpax.solve`` API is planned for
issue #37. The physical system is backend-independent.
