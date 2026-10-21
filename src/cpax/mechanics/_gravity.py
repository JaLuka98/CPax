"""Newtonian pairwise gravitational interaction helpers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def gravitational_potential(
    masses: jax.Array, *, gravitational_constant: float | jax.Array = 4.0 * jnp.pi**2
) -> Callable[[jax.Array], jax.Array]:
    """Construct an unsoftened pairwise Newtonian gravitational potential.

    Coordinates have shape (N, D) and masses shape (N,). Default units
    are AU, solar masses, and years. Coincident distinct particles cause
    a physical singularity; no softening is introduced implicitly.
    """
    masses = jnp.asarray(masses)
    constant = jnp.asarray(gravitational_constant)

    if masses.ndim != 1 or masses.shape[0] < 2:
        raise ValueError("masses must have shape (N,) for N >= 2.")
    if masses.dtype.kind != "f":
        raise TypeError("masses must be floating-point.")
    if constant.shape != ():
        raise ValueError("gravitational_constant must be scalar.")

    def potential(q: jax.Array) -> jax.Array:
        if q.ndim != 2 or q.shape[0] != masses.shape[0]:
            raise ValueError("positions must have shape (N, D), matching masses.")
        displacement = q[:, None, :] - q[None, :, :]
        identity = jnp.eye(q.shape[0], dtype=q.dtype)
        # Exclude diagonal *before* differentiating norms at zero.
        distance = jnp.linalg.norm(displacement + identity[:, :, None], axis=-1)
        interactions = (1.0 - identity) * (masses[:, None] * masses[None, :]) / distance
        return -0.5 * constant * jnp.sum(interactions)

    return potential
