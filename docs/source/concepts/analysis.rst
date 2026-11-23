Lyapunov spectra
================

``cpax.analysis.lyapunov_spectrum`` estimates the complete Lyapunov spectrum
of a smooth, real, continuous-time :class:`cpax.IVP`. It uses the Benettin
algorithm: CPax integrates the physical state together with its
autodifferentiated variational equation, then periodically applies QR
re-orthonormalisation to the tangent basis.

The workflow remains physics-first: construct a system and IVP as usual, then
pass the IVP to the analysis routine. The IVP interval is the complete run.
``transient_time`` is discarded from the beginning, and the remaining duration
is divided into the explicitly required ``num_segments`` equal QR intervals.

.. code-block:: python

   import cpax
   import jax.numpy as jnp
   from cpax.analysis import lyapunov_spectrum
   from cpax.dynamics import LorenzSystem

   problem = cpax.IVP(LorenzSystem(), jnp.array([1.0, 1.0, 1.0]), (0.0, 80.0))
   result = lyapunov_spectrum(
       problem,
       transient_time=20.0,
       num_segments=120,
       rtol=1e-7,
       atol=1e-9,
       max_steps=20000,
   )
   assert bool(result.is_success)
   print(result.exponents)             # Descending final spectrum
   print(result.times)                 # QR-interval endpoints
   print(result.running_exponents)     # Convergence history

The QR interval is a numerical parameter: it must be short enough to avoid
ill-conditioned tangent vectors, but long enough to keep repeated solver
restarts affordable. Increase the physical analysis duration and examine
``running_exponents`` before interpreting a finite-time estimate. Tighten
solver tolerances and, where appropriate, enable x64 before constructing all
JAX arrays. A spectrum is an analysis result, not a proof that the trajectory
or numerical integration has converged.

Diagnostics and reports
-----------------------

``result.is_success`` reports whether every Diffrax solve completed. It does
not certify the numerical quality of the tangent calculation: a tangent basis
may lose its contracting directions between two QR boundaries even when the
state solve succeeds. CPax therefore provides JAX-compatible diagnostics and
a short, concrete-result report:

.. code-block:: python

   diagnostics = result.diagnostics
   print(result.report())
   # Lyapunov diagnostics: PASS.
   # Completed QR segments: 120/120.
   # ...

``diagnostics.quality`` is a ``LyapunovQuality`` code. ``PASS`` means no
configured numerical-health threshold was crossed; it does not establish that
the finite-time average has converged. ``WARNING`` asks for a shorter QR
interval or tighter solver tolerances. ``INVALID`` means a segment failed,
the tangent data became non-finite, or the tangent-volume consistency check
was grossly violated.

``diagnostics.qr_log_diagonal_spans`` records the spread of the log QR
diagonal factors in every interval. It is a cheap conditioning proxy, with an
advisory threshold derived from the active floating-point dtype. Its maximum
should remain comfortably below ``diagnostics.precision_budget``; in
particular, do not increase the total interval while leaving
``num_segments`` fixed.

For every smooth ODE, Liouville's identity says that the QR log-volume growth
equals the time integral of the Jacobian trace. CPax records their cumulative
difference in ``diagnostics.liouville_residuals`` and its final rate in
``diagnostics.liouville_rate_residual``. A large residual is a strong warning
that the tangent calculation is not trustworthy. For the standard Lorenz
parameters this also checks that the spectrum sum approaches
``-(10 + 1 + 8 / 3)``.

Finally, ``diagnostics.tail_drift`` gives the absolute difference between the
final spectrum and the estimate halfway through the valid post-transient
history. It is a transparent convergence aid rather than a statistical error
bar, since consecutive chaotic samples are correlated.

Parameters
----------

The analysis routine deliberately introduces no separate parameter argument.
For generic vector fields, supply run-specific parameters in ``IVP.args``:

.. code-block:: python

   from cpax.dynamics import VectorFieldSystem

   def rhs(t, state, params):
       del t
       return params["rate"] * state

   ivp = cpax.IVP(
       VectorFieldSystem(rhs),
       jnp.array(1.0),
       (0.0, 20.0),
       args={"rate": jnp.array(0.15)},
   )
   exponent = lyapunov_spectrum(ivp, num_segments=40).exponents[0]

Physics-first mechanics systems continue to store their callback parameters in
``system.params``. Thus a ``ForcedNewtonianSystem`` with a parameterised
potential or applied force can be analysed without changing its API. Parameter
sweeps can construct or vectorise compatible IVPs; all cases must share the
same state shape and ``num_segments``.

Scope and failures
------------------

The current implementation requires real floating-point state leaves and a
JAX-differentiable vector field. It supports arbitrary state PyTrees, but the
full tangent basis costs ``O(D^2)`` memory for ``D`` scalar state coordinates.
Complex systems, discrete maps, non-smooth maps such as the tent-map draft,
and finite-separation estimators are outside this API's scope.

``throw=True`` (the default) raises a Diffrax error on an unsuccessful
transient or QR segment. With ``throw=False``, inspect ``result.is_success``:
the successful prefix of ``running_exponents`` is retained, later rows and
``exponents`` are ``NaN``, and ``stats`` aggregates the solver work performed.
``VelocityVerlet`` is deliberately unavailable because its specialised
phase-space step is not an integrator for the augmented tangent ODE.
