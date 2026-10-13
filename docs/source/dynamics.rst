Continuous-time dynamical systems
=================================

In CPax, a continuous-time dynamical system defines equations of motion,
independently of initial conditions, numerical solvers and output handling.
All such systems share the abstract
``cpax.dynamics.DynamicalSystem`` interface. Its central method is
``vector_field(t, state, args=None)``.

Most users do not need to subclass ``DynamicalSystem`` themselves.
For arbitrary equations of motion, ``VectorFieldSystem`` provides the
simplest starting point. It is just a concrete implementation of the same
interface, **not a different kind of physical system**.

Defining equations of motion
----------------------------

For example, the Lorenz system can be specified as a Python function:

.. code-block:: python

   import jax.numpy as jnp

   from cpax.dynamics import VectorFieldSystem

   def lorenz(t, state, args):
       del t # time is unused for this autonomous system
       x, y, z = state
       sigma, rho, beta = args
       return (
           sigma * (y - x),
           x * (rho - z) - y,
           x * y - beta * z,
       )

   system = VectorFieldSystem(lorenz)
   state = (jnp.array(1.0), jnp.array(1.0), jnp.array(1.0))
   parameters = (10.0, 28.0, 8.0 / 3.0)
   derivative = system.vector_field(0.0, state, parameters)

``VectorFieldSystem`` simply delegates ``vector_field`` to the supplied
callable. No custom class is necessary for this use case.

The method represents ``dy/dt = f(t, y; parameters)``. The ``state``
may be an array or an array-valued JAX PyTree, such as a tuple or dictionary.
Its derivative must have the **same PyTree structure and leaf shapes**.
Time is the first argument even for autonomous systems. The optional
``args`` provides parameters or other data for individual evaluations.

This interface evaluates equations of motion without integrating them.
Use :doc:`problems` to specify initial conditions and a time interval,
:doc:`solvers` to integrate the system, and :doc:`results` to inspect
the saved trajectory and numerical diagnostics.

Specialised physical systems
----------------------------

CPax will also provide ready-made models, such as the planned
``cpax.mechanics.NewtonianSystem`` and ``HamiltonianSystem``.
They will implement ``DynamicalSystem`` while exposing physically
meaningful inputs such as potentials, masses or Hamiltonians, and potentially
quantities such as energy.

Library developers and advanced users can subclass ``DynamicalSystem``
when a specialised model warrants its own methods and parameters. Whether
a system is wrapped by ``VectorFieldSystem`` or implemented as a dedicated
class, numerical solvers will see the **same** ``vector_field`` interface.

Parameters and JAX transformations
----------------------------------

Model parameters may be stored as array-valued Equinox module fields
(e.g. a spring constant ``self.k``) or supplied at evaluation time
through ``args``. This is a choice about where parameters live, not a
distinction between the two system classes. A ``VectorFieldSystem``
can also wrap a callable Equinox module containing differentiable fields.

Pure JAX-traceable vector fields support transformations such as
``jax.jit``, ``jax.vmap`` and differentiation where applicable.
For a system containing ordinary Python callables, ``jax.jit`` works
when the system is captured by a closure. When **passing a module with
non-array leaves as a function argument**, prefer
``equinox.filter_jit``; ``equinox.filter_grad`` can likewise
differentiate its floating-point array-valued parameters. The wrapper's
``rhs`` field is intentionally not static.

Evaluation does not perform per-call validation of state structure.
Model authors must return consistent derivatives. Physical domains,
boundary conditions and numerical failures belong to the relevant model
and solver interfaces.

Dependency boundaries
---------------------

The intended public namespaces are ``dynamics``, ``mechanics``,
``problems``, ``solvers`` and ``results``. The ``mechanics``
package remains planned; the others are implemented for continuous-time
IVPs:

* ``dynamics`` provides shared continuous-time abstractions using JAX
  and Equinox, without importing concrete physical models or solvers.
* Physical-domain packages may depend on ``dynamics``, but not vice versa.
* Problems combine systems with conditions without depending on a specific
  numerical backend.
* Solvers may use private backend adapters; physical models must not depend
  on Diffrax or backend-specific public types.
* Results present numerical outcomes and diagnostics independently of
  the chosen backend.

The legacy ``cpax.ode`` utilities remain available. This interface
is not intended as a universal abstraction for discrete dynamics, Monte
Carlo algorithms, or general PDE and boundary-value problem formulations.
