Numerical solutions
===================

``cpax.results.Solution`` defines a backend-independent representation
for saved IVP trajectories, numerical outcomes and diagnostics.

.. code-block:: python

   import jax.numpy as jnp
   from cpax.results import Solution, SolutionStatus

   trajectory = Solution(
       ts=jnp.array([0.0, 0.5, 1.0]),
       ys=jnp.array([1.0, 0.78, 0.61]),
       status=SolutionStatus.SUCCESS,
       stats={"num_steps": jnp.array(12)},
       metadata={"solver": "example"},
   )

   success = trajectory.is_success

``ts`` is a nonempty one-dimensional array with shape ``(N,)``.
``ys`` is an array or arbitrary non-empty PyTree of arrays; each leaf
must have leading dimension ``N``. For example, a position leaf with
initial shape ``(3,)`` has trajectory shape ``(N, 3)``.

``SolutionStatus.SUCCESS`` and ``SolutionStatus.FAILED`` are
backend-independent integer outcomes; ``is_success`` returns a
JAX-compatible Boolean scalar. ``stats`` stores solver statistics, and
``metadata`` stores descriptive information such as algorithm identity.
Numerical backends will populate these fields in subsequent issues.
Non-array metadata may require ``equinox.filter_jit`` if a Solution
is passed through compiled transformations.

The container checks shapes, numeric types and concrete status codes,
but intentionally does not infer whether the solver actually reached the
final time. Failed integrations may produce incomplete, padded or
non-finite trajectories. The integration backend is responsible for
reporting meaningful failures and diagnostics.

Sampling conventions
--------------------

Whether ``t0`` is included in a returned trajectory depends on the
requested save policy; the initial state is always specified by the
problem, not implicitly inserted into ``Solution``. Save policies will
be introduced in issue #37. The save times and saved states are paired
one-to-one in their leading dimension. Solver-provided valid sampling
order and endpoint coverage will be documented with the numerical solver.

``Solution`` intentionally has no plotting, interpolation or physical
energy-analysis methods at this stage.
