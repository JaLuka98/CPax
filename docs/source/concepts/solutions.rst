Solutions
=========

``cpax.solve`` returns a backend-independent ``Solution`` containing
saved states, sample times, status and numerical metadata.

.. code-block:: python

   times = solution.ts
   positions = solution.ys.q
   momenta = solution.ys.p
   success = solution.is_success
   metadata = solution.metadata
   stats = solution.stats

The saved state keeps the original PyTree structure and adds a leading
sample dimension. ``SaveAt.linspace(n)`` includes both endpoints.

Always check ``is_success`` before interpreting a trajectory. With
``throw=False``, unsuccessful solves may return partial, padded or
nonfinite data. Solver metadata records the actual algorithm and whether
selection was automatic or explicit.

Energy and other physical diagnostics belong to the physical system,
rather than being silently assumed by the numerical solution container.
