"""Physics-first mechanics: forces, energies, numerical solutions and JAX."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.dynamics import DynamicalSystem
from cpax.mechanics import HamiltonianSystem, NewtonianSystem, PhaseSpaceState


def oscillator(q, params):
    return 0.5 * params["k"] * jnp.sum(q**2)


def quartic(q, params):
    return 0.5 * params["k"] * q**2 + 0.25 * params["lam"] * q**4


def nonseparable(q, p, params):
    return 0.5 * jnp.sum(p**2 + q**2) + params["coupling"] * jnp.sum(q * p)


def integrate(system, initial=None, *, duration=1.0, samples=11):
    problem = cpax.IVP(
        system=system,
        y0=PhaseSpaceState(1.0, 0.0) if initial is None else initial,
        t_span=(0.0, duration),
    )
    return cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(samples),
        rtol=1e-7,
        atol=1e-9,
    )


def test_newtonian_force_and_all_energies():
    system = NewtonianSystem(potential=quartic, mass=2.0, params={"k": 3.0, "lam": 4.0})
    state = PhaseSpaceState(2.0, 6.0)
    assert isinstance(system, DynamicalSystem)
    np.testing.assert_allclose(system.force(state.q), -38.0, rtol=1e-6)
    np.testing.assert_allclose(system.kinetic_energy(state), 9.0)
    np.testing.assert_allclose(system.potential_energy(state), 22.0)
    np.testing.assert_allclose(system.energy(state), 31.0)
    field = system.vector_field(0.0, state)
    np.testing.assert_allclose(field.q, 3.0)
    np.testing.assert_allclose(field.p, -38.0)


def test_parameter_free_potential_and_none_vs_empty_params():
    system = NewtonianSystem(potential=lambda q: q**4 - q**2, mass=1.0)
    np.testing.assert_allclose(system.force(jnp.array(2.0)), -28.0)
    with pytest.raises(TypeError):
        NewtonianSystem(potential=lambda q: q**2, mass=1.0, params={}).force(jnp.array(1.0))


def test_multidimensional_mass_broadcast_and_energy():
    q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    p = jnp.ones_like(q)
    for mass, expected_dq, expected_t in [
        (2.0, jnp.full((2, 2), 0.5), 1.0),
        (jnp.array([2.0, 4.0]), jnp.array([[0.5, 0.5], [0.25, 0.25]]), 0.75),
        (jnp.array([[2.0, 4.0], [8.0, 16.0]]), 1 / jnp.array([[2.0, 4.0], [8.0, 16.0]]), 0.46875),
    ]:
        system = NewtonianSystem(potential=lambda x: jnp.sum(x**2), mass=mass)
        field = system.vector_field(0.0, PhaseSpaceState(q, p))
        np.testing.assert_allclose(field.q, expected_dq)
        np.testing.assert_allclose(system.kinetic_energy(PhaseSpaceState(q, p)), expected_t)


def test_vector_mass_and_bad_shapes():
    system = NewtonianSystem(potential=lambda q: jnp.sum(q**2), mass=jnp.array([1.0, 2.0, 4.0]))
    field = system.vector_field(0.0, PhaseSpaceState(jnp.ones(3), jnp.ones(3)))
    np.testing.assert_allclose(field.q, [1.0, 0.5, 0.25])
    for mass in [jnp.ones((2, 1)), jnp.ones(3)]:
        invalid = NewtonianSystem(potential=lambda q: jnp.sum(q**2), mass=mass)
        with pytest.raises(ValueError, match="mass shape"):
            invalid.vector_field(0.0, PhaseSpaceState(jnp.ones((2, 2)), jnp.ones((2, 2))))


@pytest.mark.parametrize("mass", [0.0, -1.0, float("inf"), float("nan"), True, jnp.array([1, 2])])
def test_invalid_mass_is_rejected(mass):
    with pytest.raises((ValueError, TypeError)):
        NewtonianSystem(potential=lambda q: q**2, mass=mass)


def test_bad_potential_and_hamiltonian_outputs_rejected():
    newton = NewtonianSystem(potential=lambda q: q**2, mass=1.0)
    with pytest.raises(ValueError, match="scalar"):
        newton.force(jnp.ones(2))
    hamilton = HamiltonianSystem(hamiltonian=lambda q, p: q + p)
    with pytest.raises(ValueError, match="scalar"):
        hamilton.vector_field(0.0, PhaseSpaceState(jnp.ones(2), jnp.ones(2)))


def test_non_none_ivp_args_are_rejected():
    state = PhaseSpaceState(1.0, 0.0)
    systems = [
        NewtonianSystem(potential=lambda q: q**2, mass=1.0),
        HamiltonianSystem(hamiltonian=lambda q, p: q**2 + p**2),
    ]
    for system in systems:
        with pytest.raises(ValueError, match="IVP.args"):
            system.vector_field(0.0, state, {"k": 1.0})
        with pytest.raises(ValueError, match="IVP.args"):
            cpax.IVP(system, state, (0.0, 1.0), args={"k": 1.0})


def test_general_hamiltonian_canonical_equations():
    system = HamiltonianSystem(hamiltonian=nonseparable, params={"coupling": 0.3})
    state = PhaseSpaceState(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert isinstance(system, DynamicalSystem)
    field = system.vector_field(0.0, state)
    np.testing.assert_allclose(field.q, [3.3, 4.6], rtol=1e-6)
    np.testing.assert_allclose(field.p, [-1.9, -3.2], rtol=1e-6)
    np.testing.assert_allclose(system.energy(state), 15.0 + 3.3, rtol=1e-6)


def test_hamiltonian_without_params():
    system = HamiltonianSystem(hamiltonian=lambda q, p: 0.5 * (q**2 + p**2))
    field = system.vector_field(0.0, PhaseSpaceState(1.0, 0.0))
    np.testing.assert_allclose(field.q, 0.0)
    np.testing.assert_allclose(field.p, -1.0)


def test_oscillator_agrees_with_analytic_solution_and_conserves_energy():
    system = NewtonianSystem(potential=oscillator, mass=2.0, params={"k": 8.0})
    result = integrate(system, duration=3.0, samples=51)
    assert bool(result.is_success)
    assert isinstance(result.ys, PhaseSpaceState)
    np.testing.assert_allclose(result.ys.q, jnp.cos(2.0 * result.ts), atol=1e-4)
    np.testing.assert_allclose(result.ys.p, -4.0 * jnp.sin(2.0 * result.ts), atol=2e-4)
    energies = jax.vmap(system.energy)(result.ys)
    np.testing.assert_allclose(energies, energies[0], rtol=2e-4, atol=2e-4)


def test_newtonian_and_hamiltonian_models_agree():
    mass = 2.0
    newton = NewtonianSystem(potential=oscillator, mass=mass, params={"k": 3.0})
    hamilton = HamiltonianSystem(
        hamiltonian=lambda q, p, params: p**2 / (2 * params["m"]) + 0.5 * params["k"] * q**2,
        params={"m": mass, "k": 3.0},
    )
    n = integrate(newton)
    h = integrate(hamilton)
    np.testing.assert_allclose(n.ys.q, h.ys.q, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(n.ys.p, h.ys.p, rtol=2e-5, atol=2e-5)


def test_singular_potential_valid_positive_trajectory():
    system = NewtonianSystem(potential=lambda q: 1.0 / q + q**3, mass=1.0)
    np.testing.assert_allclose(system.force(jnp.array(2.0)), 0.25 - 12.0, rtol=1e-6)
    result = integrate(system, duration=0.1, samples=5)
    assert bool(result.is_success)
    assert jnp.all(result.ys.q > 0)


def test_jit_vmap_and_differentiation_through_trajectory():
    def final_position(k):
        system = NewtonianSystem(potential=oscillator, mass=1.0, params={"k": k})
        problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 1.0))
        return cpax.solve(problem, solver=diffrax.Tsit5(), rtol=1e-7, atol=1e-9).ys.q[-1]

    k = jnp.array(2.0)
    np.testing.assert_allclose(jax.jit(final_position)(k), jnp.cos(jnp.sqrt(k)), rtol=2e-4)
    values = jax.vmap(final_position)(jnp.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(values, jnp.cos(jnp.sqrt(jnp.array([1.0, 2.0, 4.0]))), atol=2e-4)
    expected_grad = -jnp.sin(jnp.sqrt(k)) / (2 * jnp.sqrt(k))
    np.testing.assert_allclose(jax.grad(final_position)(k), expected_grad, rtol=2e-3, atol=2e-4)


def test_differentiate_mass_and_initial_coordinate():
    def final_position(m, q0):
        system = NewtonianSystem(potential=lambda q: 0.5 * q**2, mass=m)
        problem = cpax.IVP(system, PhaseSpaceState(q0, 0.0), (0.0, 1.0))
        return cpax.solve(problem, solver=diffrax.Tsit5(), rtol=1e-7, atol=1e-9).ys.q[-1]

    m = jnp.array(2.0)
    np.testing.assert_allclose(jax.grad(final_position, argnums=0)(m, jnp.array(1.0)),
                               jnp.sin(jnp.sqrt(1 / m)) / (2 * m**1.5), rtol=3e-3)
    np.testing.assert_allclose(jax.grad(final_position, argnums=1)(m, jnp.array(1.0)),
                               jnp.cos(jnp.sqrt(1 / m)), rtol=3e-3)


def test_equinox_parameter_module_transformations():
    class Potential(eqx.Module):
        k: jax.Array

        def __call__(self, q):
            return 0.5 * self.k * jnp.sum(q**2)

    system = NewtonianSystem(potential=Potential(jnp.array(2.0)), mass=1.0)

    @eqx.filter_jit
    def energy(model):
        return model.energy(PhaseSpaceState(2.0, 0.0))

    np.testing.assert_allclose(energy(system), 4.0)
    derivative = eqx.filter_grad(energy)(system)
    np.testing.assert_allclose(derivative.potential.k, 2.0)
