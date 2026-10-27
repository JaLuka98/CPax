"""Matched potential and explicit-force workloads for issue #72 (not public API)."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ForceCase:
    name: str
    q: jax.Array
    mass: jax.Array
    potential: Callable[[jax.Array], jax.Array]
    analytical_force: Callable[[jax.Array], jax.Array]


def harmonic(*, dtype: jnp.dtype = jnp.float32) -> ForceCase:
    k = jnp.asarray(3.25, dtype=dtype)
    q = jnp.asarray([0.4, -0.7, 1.2], dtype=dtype)

    def potential(x):
        return 0.5 * k * jnp.sum(x**2)

    def force(x):
        return -k * x

    return ForceCase("harmonic", q, jnp.ones_like(q), potential, force)


def coupled_quartic(*, size: int = 32, dtype: jnp.dtype = jnp.float32) -> ForceCase:
    """Periodic nearest-neighbour coupling and local quartic interaction."""
    q = jnp.linspace(-0.8, 0.9, size, dtype=dtype)
    a, b, coupling = [jnp.asarray(v, dtype=dtype) for v in (1.7, 0.35, 0.8)]

    def potential(x):
        delta = x - jnp.roll(x, 1)
        return 0.5 * a * jnp.sum(x**2) + 0.25 * b * jnp.sum(x**4) + 0.5 * coupling * jnp.sum(delta**2)

    def force(x):
        return -a * x - b * x**3 - coupling * (2 * x - jnp.roll(x, 1) - jnp.roll(x, -1))

    return ForceCase("quartic", q, jnp.ones_like(q), potential, force)


def gravity(*, size: int = 32, dimensions: int = 3, dtype: jnp.dtype = jnp.float32) -> ForceCase:
    """Unsoftened Newtonian gravity; self-distances are shifted before norms."""
    if size < 2 or dimensions not in (2, 3):
        raise ValueError("Gravity needs at least two particles in 2D or 3D.")
    i = jnp.arange(size, dtype=dtype)
    coords = [0.4 * i, jnp.sin(0.7 * i)]
    if dimensions == 3:
        coords.append(jnp.cos(0.4 * i))
    q = jnp.stack(coords, axis=-1)
    masses = jnp.asarray(0.5, dtype=dtype) + (i % 7) / jnp.asarray(7, dtype=dtype)
    constant = jnp.asarray(1.4, dtype=dtype)
    identity = jnp.eye(size, dtype=dtype)

    def distances(x):
        displacement = x[:, None, :] - x[None, :, :]
        distance = jnp.linalg.norm(displacement + identity[:, :, None], axis=-1)
        return displacement, distance

    def potential(x):
        _, distance = distances(x)
        interactions = (1.0 - identity) * (masses[:, None] * masses[None, :]) / distance
        return -0.5 * constant * jnp.sum(interactions)

    def force(x):
        delta, distance = distances(x)
        prefactor = (1.0 - identity) * (masses[:, None] * masses[None, :]) / distance**3
        return -constant * jnp.sum(prefactor[:, :, None] * delta, axis=1)

    return ForceCase("gravity", q, masses, potential, force)


def ad_force(case: ForceCase) -> Callable[[jax.Array], jax.Array]:
    return jax.grad(lambda q: -case.potential(q))
