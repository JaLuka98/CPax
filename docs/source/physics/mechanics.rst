Classical mechanics
===================

The ``cpax.mechanics`` namespace provides ``PhaseSpaceState``,
``NewtonianSystem``, ``HamiltonianSystem``, and the reusable
``gravitational_potential`` helper.

Newtonian potential
-------------------

For a mass ``m`` and potential ``V(q)``, CPax obtains forces by
automatic differentiation: ``F = -grad(V)``. The canonical equations
are ``dq/dt=p/m`` and ``dp/dt=F``.

.. code-block:: python

   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   system = NewtonianSystem(
       potential=lambda q: -q**2 / 2 + q**4 / 4,
       mass=1.0,
   )
   initial = PhaseSpaceState(q=-1.0, p=0.2)
   force = system.force(initial.q)  # q - q**3
   energy = system.energy(initial)

This symmetric double-well potential defines an undamped, unforced
Duffing oscillator, not a harmonic perturbation near the unstable origin.
See :doc:`../examples/simple` for trajectories.

General Hamiltonians
--------------------

``HamiltonianSystem`` implements Hamilton's canonical equations
for a real scalar autonomous Hamiltonian ``H(q,p)``, including
nonseparable Hamiltonians:

.. code-block:: python

   from cpax.mechanics import HamiltonianSystem

   system = HamiltonianSystem(
       hamiltonian=lambda q, p: 0.5 * (q**2 + p**2),
   )

Multibody gravity
-----------------

Construct an unsoftened Newtonian pair potential with
``gravitational_potential(masses)``. The default gravitational
constant is ``4*pi**2``, for AU, solar masses and years. The
helper expects an ``(N,D)`` positions array and ``(N,)``
floating-point masses, compatible with ``NewtonianSystem``.
Coincident *distinct* particles are singular: CPax does not
silently introduce softening.

See :doc:`../examples/advanced` for the planetary example.

Numerical structure
-------------------

Conservative autonomous Hamiltonians conserve energy along their exact
trajectories. General adaptive integration does not guarantee exact
energy or symplectic conservation. The separate ``cpax.ode``
leapfrog integrator remains available for suitable separable systems.
