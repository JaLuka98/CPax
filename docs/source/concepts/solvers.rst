Solvers
=======

A solver is an algorithm for integrating the equations of motion.
CPax's public ``solve`` accepts ``IVP`` and delegates numerical
integration to Diffrax.

.. code-block:: python

   import cpax
   import diffrax

   solution = cpax.solve(
       problem,
       solver="auto",  # Currently selects Tsit5
       saveat=cpax.SaveAt.linspace(500),
       rtol=1e-6,
       atol=1e-8,
   )

Use an explicit Diffrax solver such as ``diffrax.Dopri8()`` when
appropriate. The default currently selects Tsit5 without diagnosing
stiffness or geometric structure.

``dt0`` controls the initial step for adaptive methods, while
``SaveAt.linspace(n)`` specifies saved output times, **not** the internal
integration time step. For fixed stepping supply an explicit solver,
``diffrax.ConstantStepSize()`` and a positive ``dt0``.

The existing kick-drift-kick leapfrog algorithm is preserved separately
in ``cpax.ode.simulate_leapfrog_scan``. It is symplectic for appropriate
separable Hamiltonians and is not yet exposed through ``cpax.solve``.
Adaptive Runge-Kutta solvers do not generally preserve symplectic structure.

Use tighter tolerances and 64-bit precision when justified by convergence
studies; neither guarantees exact conservation.
