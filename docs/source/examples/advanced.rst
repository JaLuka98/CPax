Advanced examples
=================

A comet in a planetary system
-----------------------------

The Sun, Earth, Mars and Jupiter interact gravitationally with a small
comet. We initialise the planets on approximately circular, coplanar
orbits. The comet starts inbound at 15 AU; its velocity is chosen from
an *unperturbed* Sun--comet ellipse with semi-major axis 40 AU and
perihelion 1.5 AU. Once the planets are included, it no longer follows
that ideal Kepler ellipse exactly.

.. image:: ../_static/nbody_simulation.gif
   :alt: Animated Sun, Earth, Mars, Jupiter and a comet under Newtonian gravity
   :width: 520px

The animation is centred on the inner system (within 10 AU of the origin),
so the comet begins outside the frame. The planets perturb its orbit
through their gravitational forces; a particularly close encounter is
not required for a deviation from the Sun-only trajectory.

The example constructs a five-body Newtonian potential, obtains the forces
with JAX automatic differentiation, integrates the coupled equations and
plots the resulting positions. It also calculates total energy and the
out-of-plane angular momentum to check numerical drift. It is a simple
illustration of N-body dynamics, not a precision Solar System ephemeris:
the initial orbits are idealised and the adaptive Tsit5 solver does not
preserve symplectic structure.

For an instructive comparison, integrate the comet in the Sun-only
potential with the same initial conditions and compare the positions
against the full five-body result. A Kepler ellipse is an exact reference
only for the isolated two-body problem.

.. literalinclude:: ../../../examples/nbody_gravity.py
   :language: python

Symplectic two-body orbit with Velocity Verlet
----------------------------------------------

This binary-orbit example uses ``NewtonianSystem`` and CPax's
``VelocityVerlet`` solver with a fixed step. The canonical momenta and
autonomous separable Hamiltonian meet the method's symplectic contract.
The example samples at integration-step endpoints and checks energy drift;
symplecticity does not imply exact energy conservation.

.. literalinclude:: ../../../examples/velocity_verlet_two_body.py
   :language: python


Differentiable mechanics and damping-parameter inference
--------------------------------------------------------

This example illustrates an *inverse problem*: infer an unknown damping
coefficient from positions measured along a trajectory. A
``ForcedNewtonianSystem`` describes a damped harmonic oscillator, while
``IVP``, ``cpax.solve`` and ``SaveAt.at`` generate candidate predictions
at the measurement times. A JAX differentiable Gaussian negative
log-likelihood compares those positions to seeded noisy observations.
A short gradient-descent fit recovers the damping coefficient. First-order
trajectory sensitivities provide an approximate standard error through
expected Fisher information.

**This is a deliberately illustrative choice of model.** For a damped
harmonic oscillator, the exact analytical solution is known. In an actual
parameter-estimation problem for this oscillator, fitting that analytical
solution directly would usually be simpler, faster and more accurate than
numerically solving the equations for every candidate parameter. The
simulation-based method becomes particularly useful for more complex
physical systems without tractable analytical trajectories; other
optimisation and sensitivity techniques can also be appropriate there.

The data model is ``q_observed(t_i) = q_exact(t_i; gamma_true) + epsilon_i``,
with independent zero-mean Gaussian errors of **known**, common standard
deviation ``config.observation_sigma``. The unknown damping is strictly positive
via ``gamma = exp(log_gamma)``. The scalar loss used for inference is the
negative log-likelihood up to a parameter-independent constant:

.. math::

   \mathrm{NLL}(\gamma) = \frac{1}{2\sigma_{\mathrm{obs}}^2}
   \sum_i \left[q_{\mathrm{CPax}}(t_i;\gamma)-q_i^{\mathrm{obs}}\right]^2.

For independent Gaussian measurements with common, known standard
deviation, the expected Fisher information is

.. math::

   \mathcal{I}(\gamma) =
   \frac{1}{\sigma_{\mathrm{obs}}^2}
   \sum_i \left[\frac{\partial q_{\mathrm{CPax}}(t_i;\gamma)}
   {\partial\gamma}\right]^2.

The example obtains the first-order trajectory sensitivity with
``jax.jacrev`` and reports the approximate standard error
``sigma_gamma = 1/sqrt(I(gamma_hat))``. This is the
Gauss--Newton (expected-curvature) approximation, not the observed
Hessian of the negative log-likelihood. It avoids second-order
differentiation through Diffrax's default checkpointing adjoint, which
does not generally support the required higher-order transformations.
The Fisher standard error is a local approximation based on expected
information, rather than the exact curvature of this particular noisy
dataset. The plot also shows the profile-likelihood threshold
``Delta NLL = 0.5``: for one fitted parameter, Wilks' theorem gives
``2 Delta NLL = 1`` for an approximate 68% confidence interval.
The factor of two matters because the plotted quantity is **NLL**, not
``-2 log L``. The Fisher band and profile-likelihood crossings need not
coincide exactly because the likelihood is not perfectly quadratic and
the observed and expected curvatures differ. Both are approximate,
conditional on the measurement model; neither accounts for systematic
or correlated errors.

The example groups physical, measurement and solver settings in an immutable
``InferenceConfig`` instance, passed explicitly through the simulation and
inference functions. No module-level numerical configuration is required.

The example uses an adaptive Tsit5 integration with explicit tolerances,
81 equally spaced observation times over a fixed interval, and the JAX
default floating-point precision (typically float32). Output sample
times are independent of the adaptive integration mesh: values at those
times can use solver interpolation. Tolerances control estimated local
integration errors, not the accuracy of the inferred parameter.
Sensitivity results can depend on numerical tolerances, finite precision,
the interpolation/adjoint method and the smoothness of the model.
Differentiability is not guaranteed for arbitrary parameter-dependent
branching, discontinuous forces or solvers. For precision-sensitive
inference, consider enabling ``jax_enable_x64`` before creating arrays
and verifying convergence under tighter tolerances.

To run from the repository root, use
``uv run python examples/damping_parameter_inference.py``.
The example uses no additional optimisation or inference dependency.
See :doc:`../concepts/solvers` for the general JAX transformation
contract and numerical sensitivity limitations.

.. literalinclude:: ../../../examples/damping_parameter_inference.py
   :language: python
