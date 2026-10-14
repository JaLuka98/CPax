"""General autonomous canonical Hamiltonian systems."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from cpax.dynamics import DynamicalSystem

from ._newtonian import _scalar_energy
from ._state import PhaseSpaceState

type Hamiltonian = Callable[..., jax.Array]


class HamiltonianSystem(DynamicalSystem):
    """Canonical dynamics from a real scalar Hamiltonian, including nonseparable H.

    Uses `hamiltonian(q, p)` if `params=None`, else
    `hamiltonian(q, p, params)`. These models are autonomous.
    """

    hamiltonian: Hamiltonian
    params: Any

    def __init__(self, *, hamiltonian: Hamiltonian, params: Any = None) -> None:
        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable.")
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "params", params)

    def _hamiltonian_value(self, q: jax.Array, p: jax.Array) -> jax.Array:
        if self.params is None:
            return _scalar_energy(self.hamiltonian(q, p), "hamiltonian")
        return _scalar_energy(self.hamiltonian(q, p, self.params), "hamiltonian")

    def vector_field(self, t: Any, state: PhaseSpaceState, args: Any = None) -> PhaseSpaceState:
        """Evaluate canonical Hamilton equations without assuming separability."""
        del t
        if args is not None:
            raise ValueError("HamiltonianSystem does not support IVP.args; use system.params.")
        if not isinstance(state, PhaseSpaceState):
            raise TypeError("state must be a PhaseSpaceState.")
        dh_dq, dh_dp = jax.grad(self._hamiltonian_value, argnums=(0, 1))(state.q, state.p)
        return PhaseSpaceState(q=dh_dp, p=-dh_dq)

    def energy(self, state: PhaseSpaceState) -> jax.Array:
        """Evaluate the autonomous Hamiltonian on a phase-space state."""
        return self._hamiltonian_value(state.q, state.p)
