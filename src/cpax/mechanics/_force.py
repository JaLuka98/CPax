"""Force-calculation strategies and their contract with a reference potential.

A ``NewtonianSystem`` is defined by a reference potential ``V(q)``. A
``ForceCalculation`` decides how the force ``F(q)`` is evaluated and declares
how that force relates to ``-grad(V)(q)``. Solvers and diagnostics rely on
the declaration; CPax cannot certify it for user implementations.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax

from cpax._validation import as_array, require_callable

type BoundPotential = Callable[[jax.Array], jax.Array]
type ConservativeForce = Callable[..., jax.Array]


def _checked_force(value: Any, q: jax.Array, name: str) -> jax.Array:
    """Return a force with exactly the shape and floating dtype of ``q``."""
    result = as_array(value, f"{name} must return a real floating-point array.")
    if result.dtype.kind != "f":
        raise TypeError(f"{name} must return a real floating-point array.")
    if result.shape != q.shape:
        raise ValueError(f"{name} output shape {result.shape} must match coordinates shape {q.shape}.")
    if result.dtype != q.dtype:
        raise TypeError(f"{name} output dtype {result.dtype} must match coordinates dtype {q.dtype}.")
    return result


class ForceCalculation(eqx.Module):
    """Abstract strategy for evaluating the force of a `NewtonianSystem`.

    A calculation must be a deterministic function of `q`, its own fields
    and the system parameters. It must not carry state between evaluations
    (for example a tree reused from an earlier step), so that the equations
    of motion remain an autonomous ODE in `(q, p)`. It returns generalised
    forces conjugate to `q` (not accelerations) with the shape and dtype of `q`.

    Concrete calculation classes declare two contract class attributes:

    `exact_gradient`
        True if `F(q) == -grad(V)(q)` on the physical domain. Only then do
        the conservative and symplectic interpretations hold.
    `momentum_conserving`
        True if the forces sum to zero whenever `V` is translation-invariant
        (Newton's third law for pairwise interactions).

    Subclass `ExactGradientForce` or `ApproximateGradientForce` rather
    than this class; instantiation fails while a declaration is missing.
    """

    exact_gradient: eqx.AbstractClassVar[bool]
    momentum_conserving: eqx.AbstractClassVar[bool]

    @abstractmethod
    def __call__(self, q: jax.Array, potential: BoundPotential, params: Any) -> jax.Array:
        """Return the generalised force at `q`.

        `potential` is the system's reference potential with its parameters
        bound, a real scalar function of `q`. `params` is the system's
        parameter PyTree, or None when the system has none.
        """
        raise NotImplementedError


class ExactGradientForce(ForceCalculation):
    """Calculations that reproduce `-grad(V)(q)` exactly (up to round-off).

    The model keeps its conservative Hamiltonian structure: exact flows
    conserve `energy(state)`, translation-invariant potentials conserve
    total momentum, and `VelocityVerlet` is symplectic.
    """

    exact_gradient: ClassVar[bool] = True
    momentum_conserving: ClassVar[bool] = True


class ApproximateGradientForce(ForceCalculation):
    """Calculations that approximate `-grad(V)(q)` with a controlled error.

    Hierarchical methods such as Barnes-Hut tree codes or fast multipole
    methods belong here. Their forces are generally not curl-free and can be
    discontinuous in `q`, so they are not the gradient of any potential.
    `energy(state)` then reports the reference energy, which the modelled
    dynamics do not conserve exactly, and `VelocityVerlet` is reversible and
    volume-preserving but not symplectic.

    Subclasses must declare `momentum_conserving` (False for standard
    Barnes-Hut, True for symmetric FMM variants). They should document their
    accuracy parameters and reproduce the exact force in their exact limit,
    for example at opening angle zero, so that convergence can be tested.
    """

    exact_gradient: ClassVar[bool] = False


class AutodiffForce(ExactGradientForce):
    """Exact force `-grad(V)(q)` by automatic differentiation (the default)."""

    def __call__(self, q: jax.Array, potential: BoundPotential, params: Any) -> jax.Array:
        del params
        return -jax.grad(potential)(q)


class ExplicitConservativeForce(ExactGradientForce):
    """User-supplied analytical force asserted to equal `-grad(V)(q)`.

    `force_fn(q)` is used when the system has `params=None`; otherwise the
    signature is `force_fn(q, params)`. CPax checks the output shape and
    dtype but cannot certify the gradient identity. A force that only
    approximates `-grad(V)` must be an `ApproximateGradientForce` instead.
    """

    force_fn: ConservativeForce

    def __init__(self, force_fn: ConservativeForce) -> None:
        require_callable("force_fn", force_fn)
        object.__setattr__(self, "force_fn", force_fn)

    def __call__(self, q: jax.Array, potential: BoundPotential, params: Any) -> jax.Array:
        del potential
        value = self.force_fn(q) if params is None else self.force_fn(q, params)
        return _checked_force(value, q, "force_fn")
