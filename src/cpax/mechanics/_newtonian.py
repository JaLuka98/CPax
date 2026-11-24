"""Newtonian mechanics from a reference potential with diagonal positive masses."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cpax._validation import require_callable
from cpax.dynamics import DynamicalSystem

from ._force import AutodiffForce, ConservativeForce, ExplicitConservativeForce, ForceCalculation, _checked_force
from ._mass import _mass_for, _validate_mass
from ._state import PhaseSpaceState

type Potential = Callable[..., jax.Array]


def _scalar_energy(value: Any, name: str) -> jax.Array:
    result = jnp.asarray(value)
    if result.shape != () or result.dtype.kind != "f":
        raise ValueError(f"{name} must return a real floating-point scalar.")
    return result


class NewtonianSystem(DynamicalSystem):
    """Autonomous Newtonian mechanics for diagonal positive masses.

    Equations: q' = p/m, p' = F(q). The reference potential `V` defines
    the model and its energy. `force_calculation` evaluates `F` and declares
    its relation to `-grad(V)`:

    * `ExactGradientForce` (default `AutodiffForce`): `F = -grad(V)`. The
      model is conservative and `VelocityVerlet` is symplectic.
    * `ApproximateGradientForce` (e.g. Barnes-Hut or FMM): `F` approximates
      `-grad(V)`. `energy(state)` is the reference energy, not an exact
      invariant of the modelled dynamics.

    `potential(q)` is used when `params=None`; otherwise the signature
    is `potential(q, params)`. Parameters are stored in this system, not
    in `IVP.args`. Coordinates and momenta use `PhaseSpaceState`.

    `force_fn=f` is shorthand for
    `force_calculation=ExplicitConservativeForce(f)`.
    """

    potential: Potential
    mass: jax.Array
    params: Any
    force_calculation: ForceCalculation

    def __init__(
        self,
        *,
        potential: Potential,
        mass: float | jax.Array,
        params: Any = None,
        force_fn: ConservativeForce | None = None,
        force_calculation: ForceCalculation | None = None,
    ) -> None:
        require_callable("potential", potential)
        if force_fn is not None and force_calculation is not None:
            raise ValueError("Pass either force_fn or force_calculation, not both.")
        if force_calculation is None:
            force_calculation = AutodiffForce() if force_fn is None else ExplicitConservativeForce(force_fn)
        elif not isinstance(force_calculation, ForceCalculation):
            raise TypeError("force_calculation must be a ForceCalculation instance.")
        m = _validate_mass(mass)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "force_calculation", force_calculation)

    @property
    def force_fn(self) -> ConservativeForce | None:
        """The explicit force callback, if configured through `force_fn`."""
        if isinstance(self.force_calculation, ExplicitConservativeForce):
            return self.force_calculation.force_fn
        return None

    def _mass_for(self, q: jax.Array) -> jax.Array:
        return _mass_for(self.mass, q)

    def _potential_value(self, q: jax.Array) -> jax.Array:
        if self.params is None:
            return _scalar_energy(self.potential(q), "potential")
        return _scalar_energy(self.potential(q, self.params), "potential")

    def force(self, q: jax.Array) -> jax.Array:
        """Return the force evaluated by the configured force calculation."""
        q = jnp.asarray(q)
        value = self.force_calculation(q, self._potential_value, self.params)
        return _checked_force(value, q, type(self.force_calculation).__name__)

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
        """Return scalar reference potential energy."""
        return self._potential_value(state.q)

    def energy(self, state: PhaseSpaceState) -> jax.Array:
        """Return total reference mechanical energy.

        It is an invariant of the exact dynamics only for exact-gradient
        force calculations; otherwise its drift also reflects force errors.
        """
        return self.kinetic_energy(state) + self.potential_energy(state)
