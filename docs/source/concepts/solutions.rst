Solutions
=========

``cpax.solve`` returns a ``Solution`` with the saved times, states,
success indicator, step statistics and solver metadata:

.. code-block:: python

   times = solution.ts
   positions = solution.ys.q
   momenta = solution.ys.p
   success = solution.is_success
   stats = solution.stats
   metadata = solution.metadata

Trajectory shapes and sampling
------------------------------

The saved times ``ts`` form an array of shape ``(N,)``. Each array
leaf in ``ys`` has a leading sample axis of length ``N``; other
axes match the original state. Thus a ``PhaseSpaceState`` with
``q.shape == (5, 2)`` yields ``solution.ys.q.shape == (N, 5, 2)``.
A generic vector state ``y0.shape == (3,)``, as in the Lorenz
system, yields ``ys.shape == (N, 3)``.

By default, CPax stores the final state only. Use
``cpax.SaveAt.linspace(n)`` for evenly spaced samples (including both
endpoints), or ``cpax.SaveAt.at(times)`` for specified times.
Saved times are *not* necessarily internal solver steps; see
:doc:`solvers` for step-size and interpolation details.

Status and numerical diagnostics
--------------------------------

``solution.is_success`` is a JAX-compatible boolean and ``status``
uses ``cpax.results.SolutionStatus.SUCCESS`` or ``FAILED``.
With the Diffrax backend, ``stats`` normally includes
``num_steps``, ``num_accepted_steps`` and ``num_rejected_steps``.
``metadata["solver"]`` records the algorithm actually used;
``metadata["solver_selection"]`` distinguishes automatic and
explicit selection. Other entries report the backend, step controller,
initial step, configured tolerances, maximum steps and backend result.

.. code-block:: python

   print(bool(solution.is_success))
   print(solution.stats["num_accepted_steps"])
   print(solution.metadata["solver"])
   print(solution.metadata["solver_selection"])

Some details depend on the backend configuration. A custom controller
may not expose scalar tolerances, in which case CPax records ``None``
rather than guessing them.

A failed integration may return partial, padded or non-finite samples.
A saved timestamp labelled with the requested endpoint is *not* proof
that integration reached it successfully. Check the status first.

Physical analysis
-----------------

``Solution`` contains numerical output, not inferred physical
observables. For conservative mechanics, evaluate energy on the saved
states and compare it to the initial value:

.. code-block:: python

   import jax

   energies = jax.vmap(system.energy)(solution.ys)
   energy_drift = energies - energies[0]

The same principle applies to other invariants. Such checks diagnose
numerical fidelity but do not replace convergence tests.
