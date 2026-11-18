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
