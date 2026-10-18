Simple examples
===============

These examples illustrate the same CPax workflow: system, initial-value
problem, numerical solver, then numerical solution.

Harmonic oscillator
-------------------

The potential ``V(q)=q**2/2`` yields ``q''=-q``; for
``q(0)=1`` and ``p(0)=0`` the exact position is ``cos(t)``.

.. literalinclude:: ../../../examples/harmonic_oscillator.py
   :language: python

Conservative Duffing oscillator
-------------------------------

The double-well potential ``V(q)=-q**2/2+q**4/4`` gives
``q''=q-q**3``. Compare motion trapped in one well with
higher-energy trajectories crossing both wells.

.. literalinclude:: ../../../examples/duffing_oscillator.py
   :language: python

Lorenz attractor
----------------

``LorenzSystem()`` defaults to parameters ``10, 28, 8/3``.
The three-dimensional phase trajectory exhibits sensitive dependence
on initial conditions.

.. literalinclude:: ../../../examples/lorenz.py
   :language: python
