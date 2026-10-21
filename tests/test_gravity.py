"""Pairwise gravitational potential and force regression tests."""

import jax
import jax.numpy as jnp
import numpy as np

from cpax.mechanics import NewtonianSystem, PhaseSpaceState, gravitational_potential


def test_two_body_gravity_and_zero_net_force():
    masses = jnp.array([1.0, 2.0])
    potential = gravitational_potential(masses)
    q = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    expected = 8.0 * jnp.pi**2
    np.testing.assert_allclose(potential(q), -expected, rtol=1e-6)
    forces = -jax.grad(potential)(q)
    np.testing.assert_allclose(forces, [[expected, 0.0], [-expected, 0.0]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jnp.sum(forces, axis=0), [0, 0], atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(forces)))


def test_gravity_mechanics_integration():
    masses = jnp.array([1.0, 1e-3])
    potential = gravitational_potential(masses)
    system = NewtonianSystem(potential=potential, mass=masses)
    state = PhaseSpaceState(jnp.array([[0.0, 0.0], [1.0, 0.0]]), jnp.zeros((2, 2)))
    result = system.vector_field(0.0, state)
    np.testing.assert_allclose(result.q, 0.0)
    assert bool(jnp.all(jnp.isfinite(result.p)))
