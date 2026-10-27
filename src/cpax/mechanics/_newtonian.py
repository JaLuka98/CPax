"""Conservative Newtonian mechanics with diagonal positive masses."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cpax.dynamics import DynamicalSystem

from ._mass import _mass_for, _validate_mass
from ._state import PhaseSpaceState

type Potential = Callable[..., jax.Array]
type ConservativeForce = Callable[..., jax.Array]


def _scalar_energy(value: Any, name: str) -> jax.Array:
    result = jnp.asarray(value)
    if result.shape != () or result.dtype.kind != "f":
        raise ValueError(f"{name} must return a real floating-point scalar.")
    return result


class NewtonianSystem(DynamicalSystem):
    """Autonomous conservative mechanics for diagonal positive masses.

    `potential(q)` is used when `params=None`; otherwise the signature
    is `potential(q, params)`. Parameters are stored in this system, not
    in `IVP.args`. Coordinates and momenta use `PhaseSpaceState`.

    Experimental: `force_fn` may bypass automatic differentiation when
    supplied. It must equal `-grad(potential)(q)` across the physical
    domain. CPax checks its output contract but cannot certify this identity.
    Inconsistent callbacks invalidate energy diagnostics and can invalidate
    the conservative/symplectic interpretation. Omitting `force_fn`
    retains the established automatic-differentiation path.
    """

    potential: Potential
    mass: jax.Array
    params: Any
    force_fn: ConservativeForce | None

    def __init__(
        self,
        *,
        potential: Potential,
        mass: float | jax.Array,
        params: Any = None,
        force_fn: ConservativeForce | None = None,
    ) -> None:
        if not callable(potential):
            raise TypeError("potential must be callable.")
        if force_fn is not None and not callable(force_fn):
            raise TypeError("force_fn must be callable or None.")
        m = _validate_mass(mass)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "force_fn", force_fn)

    def _mass_for(self, q: jax.Array) -> jax.Array:
        return _mass_for(self.mass, q)

    def _potential_value(self, q: jax.Array) -> jax.Array:
        if self.params is None:
            return _scalar_energy(self.potential(q), "potential")
        return _scalar_energy(self.potential(q, self.params), "potential")

    def force(self, q: jax.Array) -> jax.Array:
        """Return the conservative force, from `force_fn` or `-grad(V)`."""
        if self.force_fn is None:
            return -jax.grad(self._potential_value)(q)
        value = self.force_fn(q) if self.params is None else self.force_fn(q, self.params)
        try:
            result = jnp.asarray(value)
        except (ValueError, TypeError) as exc:
            raise TypeError("force_fn must return a real floating-point array.") from exc
        if result.dtype.kind != "f":
            raise TypeError("force_fn must return a real floating-point array.")
        if result.shape != q.shape:
            raise ValueError(f"force_fn output shape {result.shape} must match coordinates shape {q.shape}.")
        if result.dtype != q.dtype:
            raise TypeError(f"force_fn output dtype {result.dtype} must match coordinates dtype {q.dtype}.")
        return result

    def vector_field(self, t: Any, state: PhaseSpaceState, args: Any = None) -> PhaseSpaceState:
        """Return velocity and momentum derivative in canonical coordinates."""
        del t
        if args is not None:
            raise ValueError("NewtonianSystem does not support IVP.args; use system.params.")
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        return PhaseSpaceState(q=state.p / self._mass_for(state.q), p=self.force(state.q))

    def kinetic_energy(self, state: PhaseSpaceState) -> jax.Array:
        """Return scalar kinetic energy of the canonical momenta."""
        return jnp.sum(state.p**2 / (2.0 * self._mass_for(state.q)))

    def potential_energy(self, state: PhaseSpaceState) -> jax.Array:
        """Return scalar potential energy."""
        return self._potential_value(state.q)

    def energy(self, state: PhaseSpaceState) -> jax.Array:
        """Return total mechanical energy."""
        return self.kinetic_energy(state) + self.potential_energy(state)
