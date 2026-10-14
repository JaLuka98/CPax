Classical mechanics
===================

The `cpax.mechanics` namespace provides canonical phase-space states and
autonomous Newtonian and Hamiltonian models. These are physical systems,
independent of the numerical solver. They integrate with the usual
`cpax.IVP`, `cpax.solve` and `Solution` workflow.

Parameterised Newtonian potential
---------------------------------

Physical parameters are stored on the **system**, not in `IVP.args`.
Define your potential as a normal Python function; no subclass is needed:

.. code-block:: python

   import cpax
   import diffrax
   import jax
   import jax.numpy as jnp

   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   def potential(q, params):
       return 0.5 * params["k"] * q**2 + 0.25 * params["lam"] * q**4

   system = NewtonianSystem(
       potential=potential,
       mass=1.0,
       params={"k": jnp.array(2.0), "lam": 0.5},
   )
   problem = cpax.IVP(
       system=system,
       y0=PhaseSpaceState(q=1.0, p=0.0),
       t_span=(0.0, 5.0),
   )
   solution = cpax.solve(
       problem,
       solver=diffrax.Tsit5(),
       saveat=cpax.SaveAt.linspace(100),
   )

   q = solution.ys.q
   p = solution.ys.p
   energies = jax.vmap(system.energy)(solution.ys)
   forces = jax.vmap(system.force)(q)

A parameter-free potential may instead have the signature `potential(q)`,
with `params` omitted (or `None`). An empty dict is **not** equivalent to
`None`: passing `params={}` uses `potential(q, params)`. All potentials
must return a single real floating-point scalar, including for vector or
matrix-shaped coordinates (use `jnp.sum` as appropriate). `params` may
contain JAX-compatible PyTree values. Avoid relying on hidden global
parameters or mutable closures.

For Newtonian dynamics the canonical momentum satisfies `p = m * dq/dt`;
the force is `-jax.grad(V)(q)`. Public diagnostics include
`system.force(q)`, `system.kinetic_energy(state)`,
`system.potential_energy(state)` and `system.energy(state)`.
All energies are scalar quantities. A mass must be explicitly supplied
and must be finite and positive. Supported shapes are:

* Scalar `q`: scalar mass.
* Vector `q` of shape `(D,)`: scalar or `(D,)` mass.
* Matrix `q` of shape `(N, D)`: scalar, per-particle `(N,)`,
  or per-coordinate `(N, D)` mass.

A `(N,)` mass is **always** interpreted as a per-particle mass and is
broadcast over coordinate components, even when `N == D`.
Other shapes are rejected. `PhaseSpaceState(q, p)` requires matching
floating-point array shapes and dtypes, supports scalars/vectors/matrices,
and is a JAX PyTree. For independent ensembles, use `jax.vmap` rather than
assuming the leading axis is a batch of independent systems.

Parameter sensitivities
-----------------------

Differentiating through a numerical solve requires no extra potential class:

.. code-block:: python

   def final_position(k):
       model = NewtonianSystem(
           potential=potential,
           mass=1.0,
           params={"k": k, "lam": 0.5},
       )
       problem = cpax.IVP(model, PhaseSpaceState(1.0, 0.0), (0.0, 5.0))
       return cpax.solve(problem, solver=diffrax.Tsit5()).ys.q[-1]

   dq_dk = jax.grad(final_position)(jnp.array(2.0))
   q_final_for_k = jax.vmap(final_position)(jnp.array([1.0, 2.0, 3.0]))

Callable Equinox modules with their own differentiable array parameters
are also supported. Ordinary Python numeric parameters should be considered
constants unless explicitly supplied as dynamic values to a transformed
function. Supported differentiation depends on solver and adjoint settings.

General Hamiltonians
--------------------

For an autonomous real scalar Hamiltonian, `HamiltonianSystem` evaluates
`dq/dt = dH/dp` and `dp/dt = -dH/dq`, including mixed `q,p` terms.

.. code-block:: python

   from cpax.mechanics import HamiltonianSystem

   def hamiltonian(q, p, params):
       return 0.5 * jnp.sum(q**2 + p**2) + params["coupling"] * jnp.sum(q * p)

   system = HamiltonianSystem(
       hamiltonian=hamiltonian,
       params={"coupling": 0.3},
   )
   state = PhaseSpaceState(q=jnp.array([1.0, 2.0]), p=jnp.array([0.0, 0.0]))
   field = system.vector_field(0.0, state)
   energy = system.energy(state)

Without `params`, the Hamiltonian signature is `H(q, p)`. Since a
general Hamiltonian need not be separable, it deliberately does not
promise a Newtonian force or unique kinetic/potential-energy split.

Current limitations
-------------------

These models are **autonomous**: they accept time in `vector_field` to
follow the standard interface, but reject non-`None` `IVP.args`.
Explicit time dependence, external forces, non-diagonal Newtonian masses,
coordinate-domain termination and constraints are outside this feature.

Models do not impose coordinate-domain restrictions. For instance,
`V(q) = 1/q + q**3` is only mathematically valid for positive `q`;
use appropriate initial conditions and inspect solver results. No clipping
or automatic regularisation is performed. Positivity checks on concrete
masses do not establish dynamic constraints on JAX-traced values.

The Hamiltonian is conserved along the exact trajectory of an autonomous
system. Generic numerical integration, including adaptive Runge–Kutta
methods, does not generally preserve the energy or symplectic structure
exactly; inspect the energy drift and choose numerical methods accordingly.
