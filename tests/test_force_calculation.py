"""Contract tests for pluggable exact and approximate force calculations."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.mechanics import (
    ApproximateGradientForce,
    AutodiffForce,
    ExactGradientForce,
    ExplicitConservativeForce,
    ForceCalculation,
    NewtonianSystem,
    PhaseSpaceState,
    gravitational_potential,
)
from cpax.solvers import VelocityVerlet


class PerturbedGradient(ApproximateGradientForce):
    """Per-particle relative errors, like an unsymmetrised tree code (test double).

    The forces are neither curl-free nor pairwise antisymmetric for nonzero
    `epsilon`, and reduce to the exact gradient at `epsilon = 0`.
    """

    epsilon: jax.Array
    momentum_conserving: ClassVar[bool] = False

    def __call__(self, q, potential, params):
        del params
        weights = jnp.linspace(-1.0, 1.0, q.shape[0], dtype=q.dtype)[:, None]
        return (1.0 + self.epsilon * weights) * -jax.grad(potential)(q)


def three_body(force_calculation=None):
    masses = jnp.asarray([1.0, 0.5, 0.3])
    system = NewtonianSystem(
        potential=gravitational_potential(masses),
        mass=masses,
        force_calculation=force_calculation,
    )
    q = jnp.asarray([[0.0, 0.0], [1.0, 0.2], [-0.4, 1.1]])
    return system, q


def binary_orbit(system_masses, force_calculation):
    total_mass = jnp.sum(system_masses)
    omega = 2 * jnp.pi * jnp.sqrt(total_mass)
    q0 = jnp.stack([jnp.array([-system_masses[1], 0.0]), jnp.array([system_masses[0], 0.0])]) / total_mass
    velocities = omega * jnp.stack([jnp.array([0.0, -system_masses[1]]), jnp.array([0.0, system_masses[0]])])
    system = NewtonianSystem(
        potential=gravitational_potential(system_masses),
        mass=system_masses,
        force_calculation=force_calculation,
    )
    return system, PhaseSpaceState(q0, system_masses[:, None] * velocities / total_mass)


def test_default_is_exact_autodiff():
    system, q = three_body()
    assert isinstance(system.force_calculation, AutodiffForce)
    assert system.force_calculation.exact_gradient
    assert system.force_calculation.momentum_conserving
    assert system.force_fn is None
    np.testing.assert_allclose(system.force(q), -jax.grad(system.potential)(q))


def test_force_fn_is_shorthand_for_explicit_conservative_force():
    def force(q):
        return -q

    shorthand = NewtonianSystem(potential=lambda q: 0.5 * jnp.sum(q**2), mass=1.0, force_fn=force)
    explicit = NewtonianSystem(
        potential=lambda q: 0.5 * jnp.sum(q**2),
        mass=1.0,
        force_calculation=ExplicitConservativeForce(force),
    )
    q = jnp.asarray([0.3, -0.7])
    assert isinstance(shorthand.force_calculation, ExplicitConservativeForce)
    assert shorthand.force_fn is force
    np.testing.assert_allclose(shorthand.force(q), explicit.force(q))


def test_reject_ambiguous_or_invalid_force_calculation():
    with pytest.raises(ValueError, match="not both"):
        NewtonianSystem(
            potential=lambda q: jnp.sum(q**2),
            mass=1.0,
            force_fn=lambda q: -2.0 * q,
            force_calculation=AutodiffForce(),
        )
    with pytest.raises(TypeError, match="ForceCalculation"):
        NewtonianSystem(potential=lambda q: jnp.sum(q**2), mass=1.0, force_calculation=lambda q: -2.0 * q)


def test_contract_declarations_are_mandatory():
    class Undeclared(ForceCalculation):
        momentum_conserving: ClassVar[bool] = True

        def __call__(self, q, potential, params):
            return q

    class MissingMomentum(ApproximateGradientForce):
        def __call__(self, q, potential, params):
            return q

    with pytest.raises(TypeError, match="exact_gradient"):
        Undeclared()
    with pytest.raises(TypeError, match="momentum_conserving"):
        MissingMomentum()
    assert ExactGradientForce.exact_gradient
    assert not ApproximateGradientForce.exact_gradient


def test_custom_calculation_output_is_validated():
    class WrongShape(ApproximateGradientForce):
        momentum_conserving: ClassVar[bool] = False

        def __call__(self, q, potential, params):
            return q[0]

    system, q = three_body(WrongShape())
    with pytest.raises(ValueError, match="WrongShape output shape"):
        system.force(q)


def test_approximate_force_converges_and_keeps_reference_energy():
    exact, q = three_body()
    reference = exact.force(q)
    errors = []
    for epsilon in (1e-1, 1e-2, 1e-3):
        approximate, _ = three_body(PerturbedGradient(jnp.asarray(epsilon)))
        errors.append(float(jnp.max(jnp.abs(approximate.force(q) - reference))))
    np.testing.assert_allclose(errors[1:], np.asarray(errors[:-1]) / 10, rtol=1e-3)

    approximate, _ = three_body(PerturbedGradient(jnp.asarray(0.1)))
    state = PhaseSpaceState(q, jnp.zeros_like(q))
    np.testing.assert_allclose(approximate.energy(state), exact.energy(state))
    # Unlike the exact gradient, the approximation violates Newton's third law.
    assert float(jnp.max(jnp.abs(jnp.sum(approximate.force(q), axis=0)))) > 1e-2
    np.testing.assert_allclose(jnp.sum(reference, axis=0), 0.0, atol=1e-4)


def test_approximate_force_is_not_a_gradient():
    system, q = three_body(PerturbedGradient(jnp.asarray(0.1)))
    jac = jax.jacfwd(system.force)(q).reshape(q.size, q.size)
    assert not bool(jnp.allclose(jac, jac.T, atol=1e-3))


def test_verlet_rejects_approximate_force_without_opt_in():
    system, state = binary_orbit(jnp.asarray([1.0, 0.5]), PerturbedGradient(jnp.asarray(0.1)))
    problem = cpax.IVP(system, state, (0.0, 0.1))
    with pytest.raises(ValueError, match="allow_approximate_force"):
        cpax.solve(problem, solver=VelocityVerlet(), dt0=0.01)


def test_opted_in_verlet_is_time_reversible_for_approximate_force():
    with jax.enable_x64(True):
        masses = jnp.asarray([1.0, 0.5], dtype=jnp.float64)
        system, state = binary_orbit(masses, PerturbedGradient(jnp.asarray(0.1, dtype=jnp.float64)))
        solver = VelocityVerlet(allow_approximate_force=True)

        def integrate(initial):
            problem = cpax.IVP(system, initial, (0.0, 0.5))
            return cpax.solve(problem, solver=solver, dt0=0.005, max_steps=200).ys

        forward = integrate(state)
        final = PhaseSpaceState(forward.q[-1], forward.p[-1])
        backward = integrate(PhaseSpaceState(final.q, -final.p))
        np.testing.assert_allclose(backward.q[-1], state.q, atol=1e-10)
        np.testing.assert_allclose(-backward.p[-1], state.p, atol=1e-10)


def test_approximate_calculation_fields_are_differentiable_and_jittable():
    system, q = three_body(PerturbedGradient(jnp.asarray(0.1)))
    np.testing.assert_allclose(eqx.filter_jit(lambda s, x: s.force(x))(system, q), system.force(q), rtol=1e-6)
    derivative = eqx.filter_grad(lambda s: jnp.sum(s.force(q)))(system)
    weights = jnp.linspace(-1.0, 1.0, 3)[:, None]
    np.testing.assert_allclose(
        derivative.force_calculation.epsilon,
        jnp.sum(weights * -jax.grad(system.potential)(q)),
        rtol=1e-5,
    )
