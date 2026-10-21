Classical mechanics
===================

The ``cpax.mechanics`` package provides ``NewtonianSystem``,
``HamiltonianSystem``, canonical ``PhaseSpaceState``, and an
unsoftened Newtonian gravitational potential.

Newtonian potentials and parameters
-----------------------------------

For a potential energy ``V(q)``, CPax differentiates the supplied
JAX-compatible function to obtain the force ``F(q) = -grad(V)(q)``.
With a mass ``m`` and canonical momentum ``p``, the equations are
``dq/dt=p/m`` and ``dp/dt=F(q)``.

For the conservative Duffing oscillator:

.. code-block:: python

   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   system = NewtonianSystem(
       potential=lambda q: -q**2 / 2 + q**4 / 4,
       mass=1.0,
   )
   initial = PhaseSpaceState(q=-1.0, p=0.2)
   force = system.force(initial.q)
   energy = system.energy(initial)

The potential has two stable minima and a barrier at ``q=0``.
Initial states below the barrier energy stay in one well; trajectories
above it can travel across both.

A parameterised function can receive a parameter PyTree stored in the
system, without requiring a special potential class:

.. code-block:: python

   import jax.numpy as jnp

   def quartic(q, params):
       return params["a"] * q**2 / 2 + params["b"] * q**4 / 4

   model = NewtonianSystem(
       potential=quartic,
       mass=1.0,
       params={"a": jnp.array(-1.0), "b": jnp.array(1.0)},
   )

With ``params=None`` (the default), CPax calls ``potential(q)``.
Otherwise it calls ``potential(q, params)``. In particular,
``params={}`` does not mean the same thing as ``params=None``.
Potential functions must return a *single real floating-point scalar*
even when ``q`` is a vector or matrix; sum particle energies where
appropriate, for example with ``jnp.sum``.

The mass must be positive and finite. Supported combinations are:

* Scalar coordinate ``q``: scalar mass.
* Vector ``q`` of shape ``(D,)``: scalar mass or mass shape ``(D,)``.
* Matrix ``q`` of shape ``(N, D)``: scalar mass, per-particle
  ``(N,)`` mass, or per-coordinate ``(N, D)`` mass.

A ``(N,)`` mass is interpreted as one mass per particle and broadcast
over the ``D`` coordinate components, including when ``N == D``.
``PhaseSpaceState(q, p)`` requires the same shape and dtype for
both arrays, with floating-point values. The ``p`` field contains
momentum, *not velocity*. For independent simulations, use
``jax.vmap`` instead of assuming that an extra leading axis in
``q`` automatically denotes a batch.

The methods ``force(q)``, ``kinetic_energy(state)``,
``potential_energy(state)``, and ``energy(state)`` are available.
For a saved trajectory, ``jax.vmap(system.energy)(solution.ys)``
evaluates its energy at every saved point.

General Hamiltonians
--------------------

``HamiltonianSystem`` accepts a real scalar autonomous Hamiltonian
``H(q,p)`` or ``H(q,p,params)``. It obtains the canonical equations
``dq/dt = dH/dp`` and ``dp/dt = -dH/dq`` by differentiation,
including for nonseparable Hamiltonians with mixed position-momentum
terms.

.. code-block:: python

   from cpax.mechanics import HamiltonianSystem

   system = HamiltonianSystem(
       hamiltonian=lambda q, p: (q**2 + p**2) / 2,
   )

Unlike ``NewtonianSystem``, a general Hamiltonian has no uniquely
defined kinetic/potential decomposition, so ``HamiltonianSystem``
exposes ``energy(state)`` rather than a force method.

Gravitational interactions
--------------------------

``gravitational_potential(masses)`` returns the pairwise Newtonian
potential for ``(N,D)`` positions and ``(N,)`` masses. Its default
gravitational constant is ``4*pi**2``, corresponding to astronomical
units, solar masses and years; another value may be supplied explicitly.

.. code-block:: python

   from cpax.mechanics import gravitational_potential

   masses = jnp.array([1.0, 3e-6])
   gravity = NewtonianSystem(
       potential=gravitational_potential(masses),
       mass=masses,
   )

The diagonal self-interaction terms are excluded before differentiating
the pairwise distances. Two *distinct* bodies at the same coordinates
are still physically singular: there is no automatic collision softening.
The :doc:`../examples/advanced` page illustrates a planetary system.

Scope and numerical accuracy
----------------------------

The Newtonian and Hamiltonian classes currently support autonomous,
conservative models. They do not directly express external time-dependent
forces, damping, general mass matrices, hard constraints or collision
events. Non-None ``IVP.args`` are rejected for these systems; use
``system.params``.

Physical domains are not automatically enforced, and CPax does not
clip positions or regularise singular potentials. An exact autonomous
Hamiltonian conserves energy, but adaptive Runge--Kutta integration
generally does not conserve either energy or symplectic structure
exactly. Inspect drift and convergence; for suitable separable
Hamiltonians, the retained fixed-step leapfrog integrator offers a
symplectic alternative.
