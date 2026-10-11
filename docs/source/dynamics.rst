Continuous-time dynamical systems
================================

CPax models continuous-time evolution through the shared
``cpax.dynamics.DynamicalSystem`` interface. A system describes the
physical equations of motion; it does **not** choose an integrator, initial
condition, sampling interval, or numerical backend.

Interface
---------

A dynamical system is an Equinox module exposing
``vector_field(t, state, args=None)``. Here, ``state`` is an array-valued
JAX PyTree, and the returned derivative must have the **same PyTree structure
and leaf shapes** as the input state. The time argument comes first even for
autonomous systems; ``args`` can contain per-evaluation external parameters.

For direct equations of motion, wrap a callable in ``VectorFieldSystem``:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from cpax.dynamics import VectorFieldSystem

   def oscillator(t, state, args):
       del t
       q, p = state
       return p, -args["k"] * q

   system = VectorFieldSystem(oscillator)
   state = (jnp.array(1.0), jnp.array(0.0))
   dqdt, dpdt = system.vector_field(
       0.0, state, {"k": jnp.array(2.0)}
   )

The returned derivatives are ``(0.0, -2.0)`` for this initial state.
Tuples, dictionaries and other supported array-valued PyTrees are valid
state representations. This interface does not perform integration; problem,
solver and solution objects are introduced separately.

Custom physical systems can subclass ``DynamicalSystem`` directly:

.. code-block:: python

   import equinox as eqx

   class HarmonicOscillator(DynamicalSystem):
       k: jax.Array

       def vector_field(self, t, state, args=None):
           del t, args
           q, p = state
           return p, -self.k * q

   model = HarmonicOscillator(k=jnp.array(2.0))

Since ``model`` is an Equinox PyTree, its array-valued fields may be used
with JAX transformations, including differentiation.

JAX transformation conventions
------------------------------

Equinox modules may also hold non-array fields, such as plain Python
functions. Such a ``VectorFieldSystem`` works with ordinary
``jax.jit`` when the system is closed over by the compiled function.
When passing a module containing non-array leaves **as a function argument**,
prefer ``equinox.filter_jit``. Likewise, ``equinox.filter_grad``
can differentiate through floating-point array-valued fields in a callable
Equinox module. The ``rhs`` field is deliberately not marked static.

Evaluation assumes a pure, JAX-traceable vector field for supported
transformations. It does not validate the state structure at every call:
the system author is responsible for returning consistent derivatives.
Physical domains, boundary conditions and numerical failures will be
handled by the corresponding model and solver interfaces.

Dependency boundaries
---------------------

The intended public namespaces are ``dynamics``, ``mechanics``,
``problems``, ``solvers`` and ``results``. Only implemented packages
are created. In particular:

* ``dynamics`` contains shared continuous-time abstractions, and may use
  JAX and Equinox. It must not import concrete physical models or solvers.
* Physical-domain packages (e.g. future ``mechanics``) may depend on
  ``dynamics``, but the reverse dependency is forbidden.
* Problems refer to systems and their conditions, without requiring a
  concrete integration backend.
* Solvers may use private backend adapters; physical systems must not depend
  on Diffrax, solver implementations or backend-specific public types.
* Results expose outcomes and diagnostics independently of any one backend.

The legacy ``cpax.ode`` utilities remain available. The continuous-time
interface does not claim to cover discrete dynamics, Monte Carlo algorithms,
or general PDE and boundary-value problem formulations.
