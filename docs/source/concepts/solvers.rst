Solvers
=======

CPax currently integrates initial-value problems using Diffrax. A call to
``cpax.solve(problem)`` uses ``diffrax.Tsit5()``, an adaptive explicit
Runge--Kutta method. This is a useful default for many smooth, non-stiff
equations; CPax does not yet detect stiffness or select a method based on
conservation laws.

Selecting an algorithm
----------------------

Pass a Diffrax solver explicitly when the model demands a particular method:

.. code-block:: python

   import cpax
   import diffrax

   solution = cpax.solve(
       problem,
       solver=diffrax.Dopri8(),
       saveat=cpax.SaveAt.linspace(501),
       rtol=1e-8,
       atol=1e-10,
   )

``diffrax.Dopri8()`` is a high-order explicit method, useful for smooth
non-stiff problems at demanding accuracy. ``diffrax.Kvaerno5()`` is an
implicit option for stiff equations; it has different computational costs
and convergence requirements. ``diffrax.Euler()`` is a basic first-order
method, typically used with a fixed step. Neither ``"auto"`` nor
``cpax.solvers.Auto()`` presently tests for stiffness or retries a
failed integration with another solver.

Time steps and output samples
-----------------------------

An adaptive Runge--Kutta solver takes internal steps according to a local
error estimate. The initial step ``dt0`` is a starting value; subsequent
steps need not have the same duration. The default PID controller uses
``rtol=1e-5`` and ``atol=1e-7``, unless overridden.

For a scalar ``y``, the acceptance criterion involves an error scale
of order ``atol + rtol * abs(y)``. These tolerances constrain *estimated
local error*, not the global trajectory error. With components measured
in different units or spanning different scales, a single absolute
tolerance deserves careful consideration. Check convergence by comparing
solutions at progressively tighter tolerances.

Output sampling is independent of integration. For example,
``SaveAt.linspace(501)`` requests 501 regularly spaced values, including
the endpoints, but the solver may take many more or fewer internal steps.
Between accepted steps Diffrax provides numerical interpolation. A dense
plot is therefore not evidence that the integration itself was accurate.

.. code-block:: python

   cpax.SaveAt.final()                     # Save only the final state
   cpax.SaveAt.linspace(501)               # Include both interval endpoints
   cpax.SaveAt.at([0.0, 0.1, 0.4, 1.0])    # Explicit sample times

Saving only the final state is also the default (``saveat=None``).
``SaveAt.at`` requires finite, strictly increasing times within the
problem interval.

For fixed-step integration, explicitly supply a step-size controller:

.. code-block:: python

   fixed = cpax.solve(
       problem,
       solver=diffrax.Euler(),
       stepsize_controller=diffrax.ConstantStepSize(),
       dt0=0.001,
       saveat=cpax.SaveAt.linspace(501),
   )

A custom ``stepsize_controller`` cannot be combined with explicit
``rtol`` or ``atol`` arguments in the same ``cpax.solve`` call.
``max_steps`` defaults to 4096; a solver reaching that limit before
the end of the interval reports failure.

Precision and conservation
--------------------------

JAX commonly defaults to 32-bit floating-point arithmetic. For longer
integrations or tight error tolerances, enable 64-bit arithmetic *before*
constructing model arrays:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)

This improves the available precision but cannot rescue an inaccurate
numerical scheme. Check the sensitivity of trajectories to solver
tolerances, and monitor quantities such as energy or angular momentum
when the physics predicts their conservation.

For autonomous Hamiltonian models, adaptive Runge--Kutta schemes do not
generally preserve the symplectic form. The retained kick-drift-kick
``cpax.ode.simulate_leapfrog_scan`` is symplectic for suitable separable
Hamiltonians and a fixed step; it is not yet integrated with ``cpax.solve``.
Small energy drift over a short run is not proof of long-term reliability.

Failures
--------

By default, ``throw=True`` raises when numerical integration fails. To
inspect the failure rather than raise, use ``throw=False``:

.. code-block:: python

   attempted = cpax.solve(
       problem,
       solver=diffrax.Tsit5(),
       max_steps=1,
       throw=False,
   )
   print(attempted.is_success)
   print(attempted.stats)
   print(attempted.metadata["backend_result"])

A failed result may contain padded, partial or non-finite saved states.
Check ``is_success`` before analysing them. The meaning of the
``Solution`` fields is described in :doc:`solutions`.

JAX transformations and parameter sensitivities
-------------------------------------------------

An integrated trajectory can depend differentiably on model parameters.
For example, the final position of a unit-mass harmonic oscillator
changes as its spring constant changes:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from cpax.mechanics import NewtonianSystem, PhaseSpaceState

   def final_position(k):
       model = NewtonianSystem(
           potential=lambda q, params: params * q**2 / 2,
           mass=1.0,
           params=k,
       )
       ivp = cpax.IVP(model, PhaseSpaceState(1.0, 0.0), (0.0, 1.0))
       return cpax.solve(ivp, solver=diffrax.Tsit5()).ys.q[-1]

   sensitivity = jax.grad(final_position)(jnp.array(2.0))

The derivative here is through the *numerical solution*, not merely
through the expression for force. Solver algorithms and adjoint choices
limit which transformations are supported. For JAX-compatible systems,
``jax.jit`` and ``jax.vmap`` can compile and batch computations.
When passing an Equinox module containing non-array callable fields
through a transformation, prefer ``equinox.filter_jit`` or
``equinox.filter_grad``. Returning numeric arrays rather than the
entire ``Solution`` from transformed functions avoids tracing its
non-array metadata. See the `Diffrax documentation
<https://docs.kidger.site/diffrax/>`_ for adjoint details.
