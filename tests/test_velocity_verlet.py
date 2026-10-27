"""Regression tests for the fixed-step Diffrax Velocity Verlet integration."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.mechanics import HamiltonianSystem, NewtonianSystem, PhaseSpaceState
from cpax.solvers import VelocityVerlet


def oscillator(k=1.0, mass=1.0, q=1.0, p=0.0):
    system = NewtonianSystem(
        potential=lambda x, params: 0.5 * params * jnp.sum(x**2),
        mass=mass,
        params=k,
    )
    return cpax.IVP(system, PhaseSpaceState(q, p), (0.0, 1.0))


def integrate(problem, h=0.01, saveat=None):
    return cpax.solve(
        problem,
        solver=VelocityVerlet(),
        dt0=h,
        saveat=cpax.SaveAt.final() if saveat is None else saveat,
        max_steps=10000,
    )


def test_second_order_convergence():
    errors = []
    for n in (20, 40, 80, 160):
        solution = integrate(oscillator(), 1.0 / n)
        error = jnp.linalg.norm(jnp.stack([solution.ys.q[-1] - jnp.cos(1.0), solution.ys.p[-1] + jnp.sin(1.0)]))
        errors.append(float(error))
        assert bool(solution.is_success)
        assert solution.metadata["solver"] == "VelocityVerlet"
        assert solution.metadata["stepsize_controller"] == "ConstantStepSize"
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.8)


@pytest.mark.parametrize(
    "shape,mass",
    [
        ((), 2.0),
        ((3,), jnp.array([1.0, 2.0, 4.0])),
        ((2, 2), jnp.array([1.0, 2.0])),
        ((2, 2), jnp.array([[1.0, 2.0], [3.0, 4.0]])),
    ],
)
def test_shape_and_mass_broadcast(shape, mass):
    q = jnp.ones(shape)
    p = jnp.zeros(shape)
    problem = oscillator(mass=mass, q=q, p=p)
    solution = integrate(problem, 0.01)
    assert isinstance(solution.ys, PhaseSpaceState)
    assert solution.ys.q.shape == (1, *shape)
    assert solution.ys.p.shape == (1, *shape)
    assert bool(jnp.all(jnp.isfinite(solution.ys.q)))


def test_long_term_energy_and_reversibility():
    system = NewtonianSystem(potential=lambda q: 0.5 * jnp.sum(q**2), mass=1.0)
    y0 = PhaseSpaceState(1.0, 0.0)
    sol = integrate(cpax.IVP(system, y0, (0.0, 100.0)), 0.1, cpax.SaveAt.linspace(1001))
    energy = 0.5 * (sol.ys.q**2 + sol.ys.p**2)
    assert float(jnp.max(jnp.abs(energy - 0.5))) < 0.002
    # Reversibility of the step map, using momentum inversion to remain
    # within CPax's forward-only IVP convention.
    y1 = PhaseSpaceState(sol.ys.q[-1], -sol.ys.p[-1])
    back = integrate(cpax.IVP(system, y1, (0.0, 100.0)), 0.1)
    np.testing.assert_allclose(back.ys.q[-1], y0.q, atol=2e-3, rtol=0)
    np.testing.assert_allclose(back.ys.p[-1], y0.p, atol=2e-3, rtol=0)


def test_kepler_energy_and_angular_momentum():
    def potential(q):
        return -1.0 / jnp.linalg.norm(q)

    system = NewtonianSystem(potential=potential, mass=1.0)
    y0 = PhaseSpaceState(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    sol = integrate(cpax.IVP(system, y0, (0.0, 20.0)), 0.02, cpax.SaveAt.linspace(1001))
    energies = jax.vmap(system.energy)(sol.ys)
    angular = sol.ys.q[:, 0] * sol.ys.p[:, 1] - sol.ys.q[:, 1] * sol.ys.p[:, 0]
    assert float(jnp.max(jnp.abs(energies - energies[0]))) < 0.002
    assert float(jnp.max(jnp.abs(angular - angular[0]))) < 2e-4


def test_jax_transformations_and_parameter_gradient():
    def final_position(k):
        return integrate(oscillator(k=k), 0.02).ys.q[-1]

    k = jnp.asarray(2.0)
    np.testing.assert_allclose(jax.jit(final_position)(k), jnp.cos(jnp.sqrt(k)), atol=2e-4)
    np.testing.assert_allclose(
        jax.vmap(final_position)(jnp.array([1.0, 2.0, 4.0])),
        jnp.cos(jnp.sqrt(jnp.array([1.0, 2.0, 4.0]))),
        atol=4e-4,
    )
    expected = -jnp.sin(jnp.sqrt(k)) / (2 * jnp.sqrt(k))
    np.testing.assert_allclose(jax.grad(final_position)(k), expected, atol=3e-4)


def test_saveat_offgrid_linear_interpolation():
    solution = integrate(oscillator(), 0.1, cpax.SaveAt.at([0.05, 0.15, 0.25]))
    np.testing.assert_allclose(solution.ts, [0.05, 0.15, 0.25], atol=1e-6)
    assert bool(solution.is_success)


def test_bad_configuration_is_rejected():
    problem = oscillator()
    with pytest.raises(ValueError, match="dt0"):
        cpax.solve(problem, solver=VelocityVerlet())
    with pytest.raises(ValueError, match="fixed stepping"):
        cpax.solve(problem, solver=VelocityVerlet(), dt0=0.1, rtol=1e-7)
    with pytest.raises(ValueError, match="ConstantStepSize"):
        cpax.solve(
            problem,
            solver=VelocityVerlet(),
            dt0=0.1,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        )
    general = HamiltonianSystem(hamiltonian=lambda q, p: (q**2 + p**2) / 2)
    with pytest.raises(TypeError, match="NewtonianSystem"):
        cpax.solve(
            cpax.IVP(general, PhaseSpaceState(1.0, 0.0), (0.0, 1.0)),
            solver=VelocityVerlet(),
            dt0=0.1,
        )


def test_two_body_gravity_symplectic_integration():
    """A binary orbit preserves total momentum and bounded energy error."""
    from cpax.mechanics import gravitational_potential

    masses = jnp.array([1.0, 0.5])
    total_mass = jnp.sum(masses)
    omega = 2 * jnp.pi * jnp.sqrt(total_mass)
    q0 = jnp.array([[-masses[1] / total_mass, 0.0], [masses[0] / total_mass, 0.0]])
    velocities = jnp.array([[0.0, -omega * masses[1] / total_mass], [0.0, omega * masses[0] / total_mass]])
    p0 = masses[:, None] * velocities
    system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
    problem = cpax.IVP(system, PhaseSpaceState(q0, p0), (0.0, 1.0))
    result = integrate(problem, 0.002, cpax.SaveAt.linspace(501))

    assert bool(result.is_success)
    assert result.ys.q.shape == (501, 2, 2)
    np.testing.assert_allclose(result.ts[[0, -1]], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.ys.q[0], q0)
    initial_total_momentum = jnp.sum(p0, axis=0)
    total_momenta = jnp.sum(result.ys.p, axis=1)
    np.testing.assert_allclose(
        total_momenta,
        jnp.broadcast_to(initial_total_momentum, total_momenta.shape),
        atol=1e-4,
        rtol=0,
    )
    energies = jax.vmap(system.energy)(result.ys)
    assert float(jnp.max(jnp.abs((energies - energies[0]) / energies[0]))) < 1e-3
