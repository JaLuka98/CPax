Numerical solutions
===================

``cpax.results.Solution`` defines a backend-independent representation
for saved IVP trajectories, numerical outcomes and diagnostics. It is
returned by ``cpax.solve``:

.. code-block:: python

   import cpax
   import diffrax
   import jax.numpy as jnp
   from cpax.dynamics import VectorFieldSystem

   def decay(t, y, args):
       del t
       return -args["rate"] * y

   problem = cpax.IVP(
       system=VectorFieldSystem(decay),
       y0=jnp.array(1.0),
       t_span=(0.0, 2.0),
       args={"rate": jnp.array(0.5)},
   )
   solution = cpax.solve(
       problem,
       solver=diffrax.Tsit5(),
       saveat=cpax.SaveAt.linspace(21),
   )

   times = solution.ts       # Shape (21,)
   values = solution.ys      # Shape (21,)
   success = solution.is_success

Saved states
------------

``ts`` is a nonempty one-dimensional array with shape ``(N,)``.
``ys`` is an array or arbitrary non-empty PyTree of arrays; each
leaf has leading sample dimension ``N`` and retains the initial
state leaf's remaining dimensions.

For example, a state ``(q, p)`` consisting of two scalar leaves
produces a saved trajectory ``(qs, ps)`` in which both arrays have
shape ``(N,)``. A vector leaf initially shaped ``(3,)`` becomes
a trajectory shaped ``(N, 3)``. CPax does not force users to flatten
states into one array.

Sampling is controlled by :doc:`solvers` through ``cpax.SaveAt``.
By default only the final state is saved. ``SaveAt.linspace(n)``
includes both interval endpoints, while ``SaveAt.at(times)`` saves
at explicitly specified times. The initial state is not automatically
included under every policy.

Status and diagnostics
----------------------

The ``status`` is an integer scalar corresponding to
``cpax.results.SolutionStatus.SUCCESS`` or ``SolutionStatus.FAILED``.
``is_success`` provides a JAX-compatible Boolean indicator.

``stats`` is a dictionary of integration diagnostics, typically
including ``num_steps``, ``num_accepted_steps``, and
``num_rejected_steps``. ``metadata`` describes the actual
backend and algorithm. For the current Diffrax adapter, it contains
``backend``, ``solver``, ``stepsize_controller``, and
``backend_result``. The last entry is the original Diffrax
``RESULTS`` value, not a plain numeric array.

.. code-block:: python

   print(bool(solution.is_success))
   print(solution.stats["num_steps"])
   print(solution.metadata["solver"])
   print(solution.metadata["backend_result"])

``throw=True`` makes a failed integration raise by default. With
``throw=False`` a failure is represented in the returned status
and diagnostics. A failed integration can return partial, padded or
non-finite samples; the ``Solution`` class intentionally validates
structure rather than physical accuracy or endpoint coverage.
**Do not treat the saved trajectory as valid without checking the status.**

The container is an Equinox PyTree, but non-array metadata can limit
how a complete ``Solution`` object is passed through raw JAX
transformations. Returning numerical leaves from a compiled function
or using Equinox's filtered transformations is often appropriate.

``Solution`` currently does not implement plotting, continuous
interpolation, or physical energy analysis methods. Consult
:doc:`solvers` for numerical accuracy and diagnostic conventions.
