import jax.numpy as jnp
import pytest

from cpax.ode import leapfrog_step, simulate_leapfrog_scan
from cpax.ode.models import hamiltonian_nd, potential_gravitational


def harmonic_oscillator(q, p, t, spring_constant=1.0):
    del t
    return p, -spring_constant * q


def test_leapfrog_step_matches_kick_drift_kick_update():
    q0 = jnp.array([[1.0]])
    p0 = jnp.array([[0.0]])
    dt = 0.1

    q1, p1 = leapfrog_step(harmonic_oscillator, q0, p0, 0.0, dt)

    assert jnp.allclose(q1, jnp.array([[0.995]]), atol=1e-6)
    assert jnp.allclose(p1, jnp.array([[-0.09975]]), atol=1e-6)


def test_leapfrog_scan_returns_post_step_states_and_times():
    q0 = jnp.array([[1.0]])
    p0 = jnp.array([[0.0]])

    ts, qs, ps = simulate_leapfrog_scan(
        q0, p0, 1.0, 0.1, 3, harmonic_oscillator
    )

    assert ts.shape == (3,)
    assert qs.shape == (3, 1, 1)
    assert ps.shape == (3, 1, 1)
    assert jnp.allclose(ts, jnp.array([1.1, 1.2, 1.3]), atol=1e-6)
    assert jnp.allclose(qs[0], jnp.array([[0.995]]), atol=1e-6)


def test_leapfrog_forwards_keyword_arguments_to_dynamics():
    q0 = jnp.array([[1.0]])
    p0 = jnp.array([[0.0]])

    q1, p1 = leapfrog_step(
        harmonic_oscillator, q0, p0, 0.0, 0.1, spring_constant=4.0
    )

    assert jnp.allclose(q1, jnp.array([[0.98]]), atol=1e-6)
    assert jnp.allclose(p1, jnp.array([[-0.396]]), atol=1e-6)


def test_leapfrog_has_bounded_harmonic_oscillator_energy_error():
    q0 = jnp.array([[1.0]])
    p0 = jnp.array([[0.0]])

    _, qs, ps = simulate_leapfrog_scan(
        q0, p0, 0.0, 0.01, 1_000, harmonic_oscillator
    )
    energies = 0.5 * (qs[:, 0, 0] ** 2 + ps[:, 0, 0] ** 2)

    assert jnp.max(jnp.abs(energies - energies[0])) < 1e-3


def test_hamiltonian_nd_supports_custom_potential():
    masses = jnp.array([2.0, 4.0])
    q = jnp.array([[1.0, -1.0], [2.0, 3.0]])
    p = jnp.array([[4.0, -2.0], [8.0, 12.0]])
    dynamics = hamiltonian_nd(masses, potential=lambda positions: 0.5 * jnp.sum(positions**2))

    dqdt, dpdt = dynamics(q, p, 0.0)

    assert jnp.allclose(dqdt, jnp.array([[2.0, -1.0], [2.0, 3.0]]))
    assert jnp.allclose(dpdt, -q)


def test_gravity_potential_and_forces_are_pairwise_symmetric():
    masses = jnp.array([1.0, 2.0])
    q = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    p = jnp.zeros_like(q)
    gravitational_potential = potential_gravitational(masses)
    dynamics = hamiltonian_nd(masses)

    _, dpdt = dynamics(q, p, 0.0)

    expected_force = 8.0 * jnp.pi**2
    assert jnp.allclose(gravitational_potential(q), -expected_force, atol=1e-5)
    assert jnp.allclose(dpdt, jnp.array([[expected_force, 0.0], [-expected_force, 0.0]]), atol=1e-5)
    assert jnp.allclose(jnp.sum(dpdt, axis=0), jnp.zeros(2), atol=1e-5)


def test_hamiltonian_nd_rejects_unknown_potential_alias():
    with pytest.raises(ValueError, match="Unknown potential alias: harmonic"):
        hamiltonian_nd(jnp.array([1.0]), potential="harmonic")
