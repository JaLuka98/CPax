"""Newtonian mechanics with non-conservative applied generalised forces."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cpax.dynamics import DynamicalSystem

from ._mass import _mass_for, _validate_mass
from ._newtonian import _scalar_energy
from ._state import PhaseSpaceState

type AppliedForce = Callable[..., jax.Array]
type Potential = Callable[..., jax.Array]


class ForcedNewtonianSystem(DynamicalSystem):
    """Constant-diagonal-mass mechanics with externally applied generalised forces.

    Equations: q' = p/m, p' = -grad(V(q)) + F(t, q, p/m).
    With params=None, callback signatures are potential(q) and
    applied_force(t, q, velocity). Otherwise both take params last.
    The optional autonomous potential defaults to zero.

    Coordinates and generalised momenta have identical shapes. The applied
    force must have exactly that shape (no implicit force broadcasting).
    Units must be consistent; for angular coordinates use moment of inertia
    as mass and torque as generalised force.

    Neither energy conservation nor symplectic integration is guaranteed.
    """

    mass: jax.Array
    potential: Potential | None
    applied_force: AppliedForce
    params: Any

    def __init__(
        self,
        *,
        mass: float | jax.Array,
        applied_force: AppliedForce,
        potential: Potential | None = None,
        params: Any = None,
    ) -> None:
        if not callable(applied_force):
            raise TypeError("applied_force must be callable.")
        if potential is not None and not callable(potential):
            raise TypeError("potential must be callable or None.")
        object.__setattr__(self, "mass", _validate_mass(mass))
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "applied_force", applied_force)
        object.__setattr__(self, "params", params)

    def _mass_for(self, q: jax.Array) -> jax.Array:
        return _mass_for(self.mass, q)

    def _potential_value(self, q: jax.Array) -> jax.Array:
        if self.potential is None:
            return jnp.zeros((), dtype=q.dtype)
        if self.params is None:
            value = self.potential(q)
        else:
            value = self.potential(q, self.params)
        return _scalar_energy(value, "potential")

    def conservative_force(self, q: jax.Array) -> jax.Array:
        """Return the negative potential gradient (zero without a potential)."""
        if self.potential is None:
            return jnp.zeros_like(q)
        return -jax.grad(self._potential_value)(q)

    def _applied_force_value(self, t: Any, q: jax.Array, velocity: jax.Array) -> jax.Array:
        if self.params is None:
            value = self.applied_force(t, q, velocity)
        else:
            value = self.applied_force(t, q, velocity, self.params)
        try:
            force = jnp.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("applied_force must return a real floating-point array.") from exc
        if force.dtype.kind != "f":
            raise TypeError("applied_force must return a real floating-point array.")
        if force.shape != q.shape:
            raise ValueError(f"applied_force output shape {force.shape} must match coordinates shape {q.shape}.")
        return force

    def vector_field(self, t: Any, state: PhaseSpaceState, args: Any = None) -> PhaseSpaceState:
        """Evaluate canonical derivatives including velocity-dependent forcing."""
        if args is not None:
            raise ValueError("ForcedNewtonianSystem does not support IVP.args; use system.params.")
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        velocity = state.p / self._mass_for(state.q)
        force = self.conservative_force(state.q) + self._applied_force_value(t, state.q, velocity)
        return PhaseSpaceState(q=velocity, p=force)

    def kinetic_energy(self, state: PhaseSpaceState) -> jax.Array:
        """Instantaneous kinetic energy, not generally conserved."""
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        return jnp.sum(state.p**2 / (2.0 * self._mass_for(state.q)))

    def potential_energy(self, state: PhaseSpaceState) -> jax.Array:
        """Instantaneous autonomous potential energy, zero when omitted."""
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        return self._potential_value(state.q)

    def mechanical_energy(self, state: PhaseSpaceState) -> jax.Array:
        """Instantaneous T + V; generally changes under applied forces."""
        return self.kinetic_energy(state) + self.potential_energy(state)

    def applied_power(self, t: Any, state: PhaseSpaceState) -> jax.Array:
        """Instantaneous energy-transfer rate sum(v * F_applied).

        For smooth autonomous potentials and constant diagonal mass,
        the exact equations obey d(mechanical_energy)/dt = applied_power.
        """
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        velocity = state.p / self._mass_for(state.q)
        return jnp.sum(velocity * self._applied_force_value(t, state.q, velocity))
