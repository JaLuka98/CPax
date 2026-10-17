Solving initial-value problems
==============================

``cpax.solve`` integrates a :doc:`problems` using a native Diffrax
solver and returns a CPax :doc:`results`. CPax keeps physical systems
independent of Diffrax, with the backend integration handled internally.
Only continuous-time ``IVP`` problems are supported at present;
automatic selection is available with a conservative Tsit5 default.

First integration
-----------------

For exponential decay, ``dy/dt = -rate * y``, the analytical
solution is ``y(t) = y0 * exp(-rate * (t - t0))``:

.. code-block:: python

   import cpax
   import jax.numpy as jnp
   from cpax.dynamics import VectorFieldSystem

   def decay(t, y, args):
       del t
       return -args["rate"] * y

   problem = cpax.IVP(
       system=VectorFieldSystem(decay),
       y0=jnp.array(2.0),
       t_span=(0.0, 5.0),
       args={"rate": jnp.array(0.5)},
   )

   solution = cpax.solve(
       problem,
       rtol=1e-6,
       atol=1e-8,
       saveat=cpax.SaveAt.linspace(101),
   )

   times = solution.ts
   values = solution.ys
   successful = solution.is_success

The ``solver`` argument defaults to ``"auto"``, currently selecting
``diffrax.Tsit5()`` for every supported IVP. Equivalent explicit requests
are ``solver="auto"`` and ``solver=cpax.solvers.Auto()``. For a specific
method, pass a native Diffrax solver such as ``diffrax.Dopri8()``.
This default is primarily suitable for non-stiff problems and does not
infer stiffness, autonomy, or geometric structure. ``saveat=None`` saves only
the final state, so use an explicit save policy when you want a
trajectory. Model state leaves may be arrays or a PyTree of arrays;
the saved trajectory preserves the PyTree structure with a leading
sample dimension.

Configuration
-------------

The public ``cpax.solve(problem, *, ...)`` keyword arguments are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Behaviour
   * - ``solver``
     - Default ``"auto"`` (currently Tsit5), ``cpax.solvers.Auto()``,
       or an explicit native ``diffrax.AbstractSolver``.
   * - ``saveat``
     - A ``cpax.SaveAt`` sampling policy; ``None`` saves only the
       final state.
   * - ``dt0``
     - Initial time step for adaptive integration; required positive
       step size when using ``diffrax.ConstantStepSize()``.
   * - ``rtol``, ``atol``
     - Relative and absolute error tolerances for CPax's default
       ``diffrax.PIDController``; effective defaults are ``1e-5``
       and ``1e-7``.
   * - ``stepsize_controller``
     - Optional native Diffrax controller. Supplying one together with
       explicit ``rtol`` or ``atol`` is rejected.
   * - ``max_steps``
     - Maximum number of integration steps; default ``4096``.
   * - ``throw``
     - ``True`` (default) raises if integration fails; ``False``
       returns a failed ``Solution`` with diagnostics.

Only configurations supported by the selected Diffrax solver and step-size
controller are valid. CPax does not infer whether an equation is stiff,
whether it is Hamiltonian, or whether a particular solver is suitable.

Choosing a numerical method
---------------------------

Diffrax provides several algorithms. Representative choices include:

* ``diffrax.Tsit5()``: adaptive explicit Runge--Kutta integration,
  often a practical starting point for non-stiff systems.
* ``diffrax.Dopri8()``: a higher-order explicit method that can be useful
  when stringent accuracy is required for sufficiently smooth non-stiff
  equations.
* ``diffrax.Kvaerno5()``: an implicit method designed for suitable
  stiff problems.
* ``diffrax.Euler()``: a simple first-order method, generally used with
  a fixed time step rather than adaptive PID control.

Automatic selection does not diagnose stiffness and never retries with a
different solver after failure. For long-time Hamiltonian integration, an
explicit geometric method may be preferable where applicable; the current
CPax IVP auto path does not select symplectic integrators.

For fixed-step Euler integration, specify both the controller and step:

.. code-block:: python

   fixed = cpax.solve(
       problem,
       solver=diffrax.Euler(),
       dt0=0.01,
       stepsize_controller=diffrax.ConstantStepSize(),
       saveat=cpax.SaveAt.linspace(101),
   )

An adaptive explicit Runge--Kutta method is not necessarily a good
choice for long-time Hamiltonian simulation: it is not inherently
symplectic. The legacy :doc:`leapfrog` integrator remains available
for appropriate separable Hamiltonians.

Step sizes, error tolerances and precision
------------------------------------------

It is important to distinguish **internal integration steps** from
**saved output samples**.

With ``diffrax.Tsit5()``, ``dt0`` is the initial step size.
The adaptive controller subsequently selects steps using local error
estimates. ``rtol`` and ``atol`` control those estimates; they are
not direct guarantees of global solution error or physical conservation.
Absolute tolerances should be chosen with the units and scales of
each state variable in mind.

For example, ``SaveAt.linspace(101)`` on ``(0.0, 10.0)`` requests
101 saved samples at spacing 0.1. It does **not** prescribe 100
integration steps. Diffrax interpolates the numerical solution at the
requested sample times where supported. Increasing saved-point density
can make plotted trajectories look smoother without improving the
underlying integration accuracy.

JAX normally uses 32-bit floating-point arithmetic by default. For
demanding accuracy or longer-time studies, enable 64-bit precision
before creating the initial arrays:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)

Use tolerance-convergence studies and, where available, compare physical
invariants such as energy. Tighter tolerances alone do not imply exact
conservation or guarantee accuracy beyond floating-point limitations.

Saving the trajectory
---------------------

CPax provides three save policies:

.. code-block:: python

   cpax.SaveAt.final()                  # Only t1; also the default
   cpax.SaveAt.linspace(101)            # Both endpoints and 99 interior points
   cpax.SaveAt.at([0.0, 0.2, 1.0, 5.0]) # Specific, irregular times

``SaveAt.linspace(n)`` requires an integer ``n >= 2``, includes
both interval endpoints, and can be reused for IVPs with different time
intervals. ``SaveAt.at(times)`` requires a nonempty, finite,
strictly increasing sequence within the problem interval. Concrete
invalid values are rejected before integration where possible; JAX-traced
values are subject to backend/runtime constraints.

Failure reporting and diagnostics
---------------------------------

By default (``throw=True``) the solver raises an exception if
integration fails. Use ``throw=False`` to inspect unsuccessful
integrations instead:

.. code-block:: python

   failed = cpax.solve(
       problem,
       solver=diffrax.Tsit5(),
       dt0=0.001,
       max_steps=1,
       throw=False,
   )

   print(failed.is_success)
   print(failed.status)
   print(failed.stats)
   print(failed.metadata["backend_result"])

``Solution.status`` uses ``SolutionStatus.SUCCESS`` or
``SolutionStatus.FAILED``. The ``is_success`` property is a
JAX-compatible Boolean scalar. ``stats`` contains backend statistics
such as ``num_steps``, ``num_accepted_steps`` and
``num_rejected_steps`` where available. ``metadata`` includes
the selected backend, actual solver, selection mode (``auto`` or
``explicit``), controller, initial step, available tolerances, and maximum
steps, as well as the original Diffrax ``RESULTS`` value in
``backend_result``. For a custom controller, unavailable tolerances are
recorded as ``None`` rather than inferred.

Failed solutions can contain padded, partial, or non-finite samples.
A saved final-time entry does **not** prove that the solver reached
that time successfully. Check ``is_success`` before interpreting the
numerical results.

JAX transformations
-------------------

For supported configurations, a complete IVP solve can be compiled,
batched, or differentiated. For example, differentiating the final
state of the decay system with respect to its rate:

.. code-block:: python

   import jax

   def final_value(rate):
       local_problem = cpax.IVP(
           system=VectorFieldSystem(decay),
           y0=jnp.array(2.0),
           t_span=(0.0, 5.0),
           args={"rate": rate},
       )
       return cpax.solve(
           local_problem,
           solver=diffrax.Tsit5(),
       ).ys[-1]

   sensitivity = jax.grad(final_value)(jnp.array(0.5))

This computes a numerical sensitivity of the **integrated solution**,
not merely a derivative of the vector field.

``jax.jit`` and ``jax.vmap`` are supported where the model and
solver configuration are JAX-compatible. When a system is passed as
a function argument and contains non-array Equinox fields, use
``equinox.filter_jit`` or ``equinox.filter_grad`` as appropriate.
For compiled functions, returning numerical fields (for example,
``solution.ys`` or ``solution.status``) is more robust than
returning the entire ``Solution``, whose metadata also contains
non-array values.

Diffrax's default adjoint strategy supports common reverse-mode
differentiation workflows; not every transform, including forward-mode
differentiation, works with every adjoint or solver. Neither derivative
accuracy nor arbitrary branching on traced values is guaranteed. See
the `Diffrax documentation <https://docs.kidger.site/diffrax/>`_
for details on solver and adjoint capabilities.
