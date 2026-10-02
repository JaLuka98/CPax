Leapfrog integration
====================

CPax provides a kick-drift-kick leapfrog integrator for Hamiltonian systems
with positions ``q`` and canonical momenta ``p``. It is symplectic for
separable Hamiltonians of the form ``H(q, p) = T(p) + V(q)``. Dynamics that
depend explicitly on time or momentum can still be called, but do not retain
that symplectic guarantee.

The integration functions accept a dynamics callable with the signature
``f(q, p, t, **kwargs) -> (dqdt, dpdt)``:

.. code-block:: python

   import jax.numpy as jnp

   from cpax.ode import simulate_leapfrog_scan

   def harmonic_oscillator(q, p, t, spring_constant=1.0):
       return p, -spring_constant * q

   q0 = jnp.array([[1.0]])
   p0 = jnp.array([[0.0]])
   ts, qs, ps = simulate_leapfrog_scan(
       q0, p0, t0=0.0, dt=0.01, n_steps=1_000,
       f=harmonic_oscillator, spring_constant=1.0,
   )

``qs`` and ``ps`` contain the state after each step; the initial state is not
included. Thus ``ts[0]`` is ``t0 + dt`` and each time entry corresponds to the
position and momentum at the same array index.

N-body gravity
--------------

``hamiltonian_nd`` creates dynamics for a separable N-body Hamiltonian with
positions and momenta shaped ``(N, D)``. Passing ``potential="gravity"`` uses
Newtonian pairwise gravity with ``G = 4 pi^2`` (the usual AU, solar-mass, and
year unit system):

.. code-block:: python

   import jax.numpy as jnp

   from cpax.ode import simulate_leapfrog_scan
   from cpax.ode.models import hamiltonian_nd

   masses = jnp.array([1.0, 0.5])
   dynamics = hamiltonian_nd(masses, potential="gravity")
   ts, qs, ps = simulate_leapfrog_scan(q0, p0, 0.0, 0.002, 500, dynamics)

Alternatively, supply a callable ``V(q) -> float`` as ``potential`` to use a
custom position-dependent potential. Masses must be nonzero. The gravity
model does not include collision softening, so distinct particles must not
occupy exactly the same position.

See ``examples/leapfrog_two_body.py`` for a complete two-body orbit and plot.
