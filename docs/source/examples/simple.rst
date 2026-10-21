Simple examples
===============

These examples illustrate the CPax workflow for solving initial-value problems
involving a physical system.
The chain **System → Problem → Solver → Solution** represents the sequence of steps.

Harmonic oscillator
-------------------

The potential ``V(q)=q**2/2`` yields ``q''=-q``; for
``q(0)=1`` and ``p(0)=0`` the exact position is ``cos(t)``.

.. literalinclude:: ../../../examples/harmonic_oscillator.py
   :language: python

Conservative Duffing oscillator
-------------------------------

The double-well potential ``V(q)=-q**2/2+q**4/4`` gives
``q''=q-q**3``. Depending on the initial conditions, the system can be trapped in one well or exhibit
oscillations that cross both wells.

.. literalinclude:: ../../../examples/duffing_oscillator.py
   :language: python

Lorenz attractor
----------------

One of the simplest and most famous examples of a chaotic system is the Lorenz attractor, defined by a
system of three coupled first-order ODEs.
The system is often used to illustrate the sensitive dependence on initial conditions that characterizes chaotic dynamics.
CPax provides a convenience class for the Lorenz system, which can be instantiated with
the customary default parameters ``10, 28, 8/3`` or customized as needed.
The resulting trajectories show the characteristic butterfly-shaped attractor.

.. literalinclude:: ../../../examples/lorenz.py
   :language: python
