"""Experimental conservative force_fn regression checks for issue #72."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from benchmarks.force_models import ad_force, coupled_quartic, gravity, harmonic
from cpax.mechanics import NewtonianSystem, PhaseSpaceState
from cpax.solvers import VelocityVerlet


@pytest.mark.parametrize("case", [harmonic(), coupled_quartic(size=8), gravity(size=2), gravity(size=8)])
def test_analytical_matches_ad_force_and_field(case):
    ad = NewtonianSystem(potential=case.potential, mass=case.mass)
    direct = NewtonianSystem(potential=case.potential, mass=case.mass, force_fn=case.analytical_force)
    state = PhaseSpaceState(case.q, jnp.zeros_like(case.q))
    np.testing.assert_allclose(direct.force(case.q), ad.force(case.q), rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(direct.force(case.q), ad_force(case)(case.q), rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(
        direct.vector_field(0.0, state).p, ad.vector_field(0.0, state).p, rtol=5e-5, atol=5e-6
    )
    np.testing.assert_allclose(direct.energy(state), ad.energy(state))


def test_parameterised_callback_and_none_semantics():
    def potential(q, params):
        return 0.5 * params["k"] * jnp.sum(q**2)

    def force(q, params):
        return -params["k"] * q

    q = jnp.asarray([0.5, -0.4])
    system = NewtonianSystem(potential=potential, mass=1.0, params={"k": jnp.asarray(3.0)}, force_fn=force)
    np.testing.assert_allclose(system.force(q), [-1.5, 1.2])
    with pytest.raises(TypeError):
        NewtonianSystem(
            potential=lambda x: jnp.sum(x**2), mass=1.0,
            params={}, force_fn=lambda x: -x,
        ).force(q)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_force_shapes(shape):
    q = jnp.ones(shape)
    system = NewtonianSystem(
        potential=lambda x: jnp.sum(x**2), mass=2.0,
        force_fn=lambda x: -2.0 * x,
    )
    assert system.force(q).shape == shape


@pytest.mark.parametrize("bad", [1, object()])
def test_reject_invalid_force_callable(bad):
    with pytest.raises(TypeError, match="force_fn"):
        NewtonianSystem(potential=lambda q: jnp.sum(q**2), mass=1.0, force_fn=bad)


@pytest.mark.parametrize(
    "bad,error",
    [
        (lambda q: jnp.asarray(1.0), ValueError),
        (lambda q: jnp.asarray([1, 2]), TypeError),
        (lambda q: {"unexpected": 1.0}, TypeError),
    ],
)
def test_reject_invalid_force_outputs(bad, error):
    model = NewtonianSystem(potential=lambda x: jnp.sum(x**2), mass=1.0, force_fn=bad)
    with pytest.raises(error, match="force_fn"):
        model.force(jnp.ones(2))


def test_reject_dtype_mismatch():
    with jax.enable_x64(True):
        q = jnp.ones(2, dtype=jnp.float32)
        model = NewtonianSystem(
            potential=lambda x: jnp.sum(x**2), mass=1.0,
            force_fn=lambda x: jnp.ones_like(x, dtype=jnp.float64),
        )
        with pytest.raises(TypeError, match="dtype"):
            model.force(q)


def test_jit_vmap_grad_and_equinox_callback():
    def evaluate(k):
        system = NewtonianSystem(
            potential=lambda q, p: 0.5 * p * jnp.sum(q**2),
            mass=1.0, params=k,
            force_fn=lambda q, p: -p * q,
        )
        return system.force(jnp.asarray([2.0, -1.0]))[0]

    np.testing.assert_allclose(jax.jit(evaluate)(jnp.asarray(3.0)), -6.0)
    np.testing.assert_allclose(jax.vmap(evaluate)(jnp.asarray([1.0, 2.0, 3.0])), [-2.0, -4.0, -6.0])
    np.testing.assert_allclose(jax.grad(evaluate)(jnp.asarray(3.0)), -2.0)

    class Force(eqx.Module):
        k: jax.Array

        def __call__(self, q):
            return -self.k * q

    system = NewtonianSystem(
        potential=lambda q: 0.5 * jnp.sum(q**2), mass=1.0, force_fn=Force(jnp.asarray(3.0))
    )
    derivative = eqx.filter_grad(lambda s: jnp.sum(s.force(jnp.asarray([2.0, 3.0]))))(system)
    np.testing.assert_allclose(derivative.force_fn.k, -5.0)


def test_verlet_and_tsit5_against_ad_and_oscillator_reference():
    k = jnp.asarray(2.0)
    potential = lambda q: 0.5 * k * jnp.sum(q**2)
    initial = PhaseSpaceState(1.0, 0.0)
    ad_system = NewtonianSystem(potential=potential, mass=1.0)
    direct_system = NewtonianSystem(potential=potential, mass=1.0, force_fn=lambda q: -k * q)

    for solver, settings in [
        (VelocityVerlet(), {"dt0": 0.02}),
        (diffrax.Tsit5(), {"rtol": 1e-5, "atol": 1e-7}),
    ]:
        def integrate(system):
            problem = cpax.IVP(system, initial, (0.0, 1.0))
            return cpax.solve(problem, solver=solver, **settings).ys.q[-1]

        np.testing.assert_allclose(integrate(ad_system), integrate(direct_system), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(integrate(direct_system), jnp.cos(jnp.sqrt(k)), atol=2e-4)


def test_inconsistent_force_changes_reported_energy():
    system = NewtonianSystem(
        potential=lambda x: 0.5 * x**2,
        mass=1.0,
        force_fn=lambda x: -2.0 * x,
    )
    state = PhaseSpaceState(jnp.asarray(1.0), jnp.asarray(2.0))
    dynamics = system.vector_field(0.0, state)
    energy_rate = jax.jvp(lambda y: system.energy(y), (state,), (dynamics,))[1]
    # dE/dt = v * (F + grad(V)) = 2 * (-2 + 1).
    np.testing.assert_allclose(energy_rate, -2.0)


def test_position_dependent_force_need_not_be_conservative():
    system = NewtonianSystem(
        potential=lambda q: jnp.asarray(0.0, dtype=q.dtype),
        mass=1.0,
        force_fn=lambda q: jnp.stack((-q[1], q[0])),
    )
    q = jnp.asarray([1.0, 0.0])
    jac = jax.jacfwd(system.force)(q)
    assert not bool(jnp.allclose(jac, jac.T))


def test_benchmark_speedup_convention():
    from benchmarks.force_comparison import _speedup

    assert _speedup(4.0, 2.0) == 2.0  # Analytical is twice as fast.
    assert _speedup(2.0, 4.0) == 0.5  # AD is twice as fast.


def test_benchmark_trajectory_and_parameter_gradient_checks():
    from benchmarks.force_comparison import _integrate, _validate_parameter_gradients
    from benchmarks.force_models import gravity

    with jax.enable_x64(True):
        case = gravity(size=3, dtype=jnp.float64)
        result = _integrate(case, steps=4, dt=0.001, trials=1, repetitions=1)
        assert result["analytical_speedup_factor_vs_ad"] > 0
        np.testing.assert_allclose(
            result["validation"]["final_energy_ad"],
            result["validation"]["final_energy_analytical"],
            rtol=1e-9,
        )
        gradient = _validate_parameter_gradients(case, steps=4, dt=0.001)
        np.testing.assert_allclose(
            gradient["gradient_ad"], gradient["gradient_analytical"], rtol=1e-7, atol=1e-10
        )
