"""Tests for the retained kick-drift-kick symplectic method."""

import jax
import jax.numpy as jnp
import numpy as np

from cpax.mechanics import gravitational_potential
from cpax.ode import leapfrog_step, simulate_leapfrog_scan


def harmonic_oscillator(q, p, t, spring_constant=1.0):
    del t
    return p, -spring_constant * q


def test_kick_drift_kick_step():
    q, p = leapfrog_step(harmonic_oscillator, jnp.array([[1.0]]), jnp.array([[0.0]]), 0.0, 0.1)
    np.testing.assert_allclose(q, [[0.995]], atol=1e-6)
    np.testing.assert_allclose(p, [[-0.09975]], atol=1e-6)


def test_scan_timestamps_and_energy():
    ts, qs, ps = simulate_leapfrog_scan(jnp.array([[1.0]]), jnp.array([[0.0]]), 0.0, 0.01, 1000, harmonic_oscillator)
    assert ts.shape == (1000,)
    assert qs.shape == (1000, 1, 1)
    np.testing.assert_allclose(ts[0], 0.01, atol=1e-6)
    energies = 0.5 * (qs[:, 0, 0] ** 2 + ps[:, 0, 0] ** 2)
    assert float(jnp.max(jnp.abs(energies - energies[0]))) < 1e-3


def test_leapfrog_gravity_interoperability():
    masses = jnp.array([1.0, 0.5])
    potential = gravitational_potential(masses)
    force = jax.grad(potential)

    def dynamics(q, p, t):
        del t
        return p / masses[:, None], -force(q)

    q0 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    p0 = jnp.zeros_like(q0)
    q1, p1 = leapfrog_step(dynamics, q0, p0, 0.0, 0.001)
    assert bool(jnp.all(jnp.isfinite(q1)))
    assert bool(jnp.all(jnp.isfinite(p1)))
