"""Reference and contract tests for non-conservative Newtonian mechanics."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.mechanics import ForcedNewtonianSystem, NewtonianSystem, PhaseSpaceState
from cpax.solvers import VelocityVerlet


def solve(system, y0=None, end=2.0, samples=61):
    problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0) if y0 is None else y0, (0.0, end))
    return cpax.solve(problem, saveat=cpax.SaveAt.linspace(samples), rtol=1e-7, atol=1e-9)


def test_conservative_limit_matches_newtonian():
    def potential(q):
        return 0.5 * jnp.sum(q**2)
    classic = NewtonianSystem(mass=jnp.array([1.0, 2.0]), potential=potential)
    forced = ForcedNewtonianSystem(
        mass=jnp.array([1.0, 2.0]),
        potential=potential,
        applied_force=lambda t, q, v: jnp.zeros_like(q),
    )
    y0 = PhaseSpaceState(jnp.array([1.0, 0.5]), jnp.array([0.0, 0.3]))
    a, b = solve(classic, y0), solve(forced, y0)
    np.testing.assert_allclose(a.ys.q, b.ys.q, atol=2e-5)
    np.testing.assert_allclose(a.ys.p, b.ys.p, atol=2e-5)
    np.testing.assert_allclose(forced.mechanical_energy(y0), classic.energy(y0))
    np.testing.assert_allclose(forced.applied_power(0.0, y0), 0.0)


def test_damped_oscillator_analytic_trajectory_and_energy_balance():
    mass, gamma, k = 2.0, 0.8, 8.0
    system = ForcedNewtonianSystem(
        mass=mass,
        potential=lambda q: 0.5 * k * q**2,
        applied_force=lambda t, q, v: -gamma * v,
    )
    result = solve(system, end=3.0, samples=301)
    beta = gamma / (2 * mass)
    omega = np.sqrt(k / mass - beta**2)
    t = np.asarray(result.ts)
    exact_q = np.exp(-beta * t) * (np.cos(omega * t) + beta / omega * np.sin(omega * t))
    exact_p = -mass * np.exp(-beta * t) * (omega + beta**2 / omega) * np.sin(omega * t)
    np.testing.assert_allclose(result.ys.q, exact_q, atol=3e-4)
    np.testing.assert_allclose(result.ys.p, exact_p, atol=6e-4)
    energy = jax.vmap(system.mechanical_energy)(result.ys)
    power = jax.vmap(system.applied_power)(result.ts, result.ys)
    np.testing.assert_array_less(np.diff(np.asarray(energy)), 2e-5)
    transferred = np.trapezoid(np.asarray(power), t)
    np.testing.assert_allclose(energy[-1] - energy[0], transferred, atol=4e-4)


def test_periodic_force_against_analytic_solution():
    # q'' + omega0**2 q = A cos(Omega t), with zero initial conditions.
    omega0, omega, amplitude = 2.0, 0.75, 0.4
    system = ForcedNewtonianSystem(
        mass=1.0,
        potential=lambda q: 0.5 * omega0**2 * q**2,
        applied_force=lambda t, q, v: amplitude * jnp.cos(omega * t) + jnp.zeros_like(q),
    )
    result = solve(system, PhaseSpaceState(0.0, 0.0), end=5.0)
    exact = amplitude * (np.cos(omega * np.asarray(result.ts)) - np.cos(omega0 * np.asarray(result.ts))) / (
        omega0**2 - omega**2
    )
    np.testing.assert_allclose(result.ys.q, exact, atol=2e-4)


def test_nonlinear_pendulum_and_force_receives_velocity():
    params = {"gravity": 9.81, "mass": 2.0, "length": 1.5, "damping": 0.3}
    inertia = params["mass"] * params["length"] ** 2
    system = ForcedNewtonianSystem(
        mass=inertia,
        potential=lambda q, p: p["mass"] * p["gravity"] * p["length"] * (1 - jnp.cos(q)),
        applied_force=lambda t, q, v, p: -p["damping"] * v + 0.1 * jnp.sin(t),
        params=params,
    )
    state = PhaseSpaceState(0.5, inertia * 0.2)
    field = system.vector_field(0.0, state)
    np.testing.assert_allclose(field.q, 0.2)
    np.testing.assert_allclose(
        field.p, -params["mass"] * params["gravity"] * params["length"] * np.sin(0.5) - 0.06
    )
    assert bool(solve(system, state, end=1.0).is_success)


@pytest.mark.parametrize("shape,mass", [
    ((), 2.0),
    ((3,), jnp.array([1.0, 2.0, 3.0])),
    ((2, 3), jnp.array([1.0, 2.0])),
    ((2, 3), jnp.ones((2, 3))),
])
def test_shapes_and_optional_potential(shape, mass):
    system = ForcedNewtonianSystem(mass=mass, applied_force=lambda t, q, v: -v)
    state = PhaseSpaceState(jnp.ones(shape), jnp.ones(shape))
    field = system.vector_field(0.0, state)
    assert field.q.shape == field.p.shape == shape
    np.testing.assert_allclose(system.potential_energy(state), 0)
    np.testing.assert_allclose(system.applied_power(0.0, state), -jnp.sum(field.q**2))


@pytest.mark.parametrize("mass", [0.0, -1.0, float("inf"), float("nan"), True, jnp.array([1, 2])])
def test_bad_mass(mass):
    with pytest.raises((ValueError, TypeError)):
        ForcedNewtonianSystem(mass=mass, applied_force=lambda t, q, v: -v)


def test_invalid_callbacks_state_and_args():
    with pytest.raises(TypeError, match="applied_force"):
        ForcedNewtonianSystem(mass=1.0, applied_force=None)
    with pytest.raises(TypeError, match="potential"):
        ForcedNewtonianSystem(mass=1.0, applied_force=lambda t, q, v: -v, potential=1)
    scalar = PhaseSpaceState(1.0, 0.0)
    for callback, error in [
        (lambda t, q, v: jnp.ones(2), ValueError),
        (lambda t, q, v: jnp.array(1), TypeError),
        (lambda t, q, v: {"bad": 1.0}, TypeError),
    ]:
        system = ForcedNewtonianSystem(mass=1.0, applied_force=callback)
        with pytest.raises(error):
            system.vector_field(0.0, scalar)
        with pytest.raises(ValueError):
            cpax.IVP(system, scalar, (0.0, 1.0))
    system = ForcedNewtonianSystem(mass=1.0, applied_force=lambda t, q, v: jnp.zeros_like(q))
    with pytest.raises(TypeError, match="PhaseSpaceState"):
        system.vector_field(0.0, jnp.array(1.0))
    with pytest.raises(ValueError, match=r"IVP\.args"):
        system.vector_field(0.0, scalar, {"invalid": 1})
    with pytest.raises(ValueError, match="mass shape"):
        ForcedNewtonianSystem(
            mass=jnp.ones(2), applied_force=lambda t, q, v: jnp.zeros_like(q)
        ).vector_field(0.0, PhaseSpaceState(jnp.ones(3), jnp.zeros(3)))
    with pytest.raises(ValueError, match="scalar"):
        ForcedNewtonianSystem(
            mass=1.0, potential=lambda q: jnp.ones(2),
            applied_force=lambda t, q, v: jnp.zeros_like(q),
        ).vector_field(0.0, scalar)


def test_auto_solver_and_verlet_rejection():
    system = ForcedNewtonianSystem(mass=1.0, applied_force=lambda t, q, v: -0.1 * v)
    problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 1.0))
    result = cpax.solve(problem)
    assert bool(result.is_success)
    assert result.metadata["solver"] == "Tsit5"
    with pytest.raises(TypeError, match="NewtonianSystem"):
        cpax.solve(problem, solver=VelocityVerlet(), dt0=0.01)


def test_jit_vmap_grad_and_equinox_force_parameters():
    def final_position(gamma):
        model = ForcedNewtonianSystem(
            mass=1.0,
            applied_force=lambda t, q, v, p: -p * v,
            params=gamma,
        )
        problem = cpax.IVP(model, PhaseSpaceState(0.0, 1.0), (0.0, 1.0))
        return cpax.solve(problem, solver=diffrax.Tsit5(), rtol=1e-7, atol=1e-9).ys.q[-1]

    gamma = jnp.array(0.4)
    exact = (1 - jnp.exp(-gamma)) / gamma
    derivative = ((gamma + 1) * jnp.exp(-gamma) - 1) / gamma**2
    np.testing.assert_allclose(jax.jit(final_position)(gamma), exact, atol=2e-4)
    gs = jnp.array([0.2, 0.4, 0.7])
    np.testing.assert_allclose(jax.vmap(final_position)(gs), (1 - jnp.exp(-gs)) / gs, atol=3e-4)
    np.testing.assert_allclose(jax.grad(final_position)(gamma), derivative, atol=5e-4)

    class Damping(eqx.Module):
        gamma: jax.Array

        def __call__(self, t, q, v):
            return -self.gamma * v

    model = ForcedNewtonianSystem(mass=1.0, applied_force=Damping(jnp.array(0.4)))

    @eqx.filter_jit
    def power(s):
        return s.applied_power(0.0, PhaseSpaceState(1.0, 1.0))

    np.testing.assert_allclose(power(model), -0.4)
    grad_model = eqx.filter_grad(power)(model)
    np.testing.assert_allclose(grad_model.applied_force.gamma, -1.0)
