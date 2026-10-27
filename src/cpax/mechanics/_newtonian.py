"""Conservative Newtonian mechanics with diagonal positive masses."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cpax.dynamics import DynamicalSystem

from ._mass import _mass_for, _validate_mass
from ._state import PhaseSpaceState

type Potential = Callable[..., jax.Array]


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
    """

    potential: Potential
    mass: jax.Array
    params: Any

    def __init__(
        self,
        *,
        potential: Potential,
        mass: float | jax.Array,
        params: Any = None,
    ) -> None:
        if not callable(potential):
            raise TypeError("potential must be callable.")
        m = _validate_mass(mass)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "params", params)

    def _mass_for(self, q: jax.Array) -> jax.Array:
        return _mass_for(self.mass, q)

    def _potential_value(self, q: jax.Array) -> jax.Array:
        if self.params is None:
            return _scalar_energy(self.potential(q), "potential")
        return _scalar_energy(self.potential(q, self.params), "potential")

    def force(self, q: jax.Array) -> jax.Array:
        """The negative gradient of the scalar potential energy."""
        return -jax.grad(self._potential_value)(q)

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
