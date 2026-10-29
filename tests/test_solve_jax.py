"""JAX transformation contracts for the CPax IVP/Diffrax integration."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import cpax
from cpax.dynamics import DynamicalSystem, VectorFieldSystem


def _decay(t, y, args):
    del t
    return -args["rate"] * y


class DecaySystem(DynamicalSystem):
    rate: jax.Array

    def vector_field(self, t, state, args=None):
        del t, args
        return -self.rate * state


def test_jit_compiles_end_to_end_solve_and_returns_numeric_fields():
    system = VectorFieldSystem(_decay)

    @jax.jit
    def integrate(rate):
        problem = cpax.IVP(
            system,
            jnp.array(2.0),
            (0.0, 2.0),
            {"rate": rate},
        )
        solution = cpax.solve(
            problem,
            solver=diffrax.Tsit5(),
            saveat=cpax.SaveAt.linspace(9),
        )
        return solution.ts, solution.ys, solution.status

    times, values, status = integrate(jnp.array(0.5))
    assert times.shape == values.shape == (9,)
    assert int(status) == 0
    np.testing.assert_allclose(values[-1], 2.0 * np.exp(-1.0), rtol=1e-4)


def test_jit_resolves_linspace_from_traced_interval_endpoint():
    system = VectorFieldSystem(_decay)

    @jax.jit
    def integrate(t1):
        problem = cpax.IVP(
            system,
            jnp.array(1.0),
            (jnp.array(0.0), t1),
            {"rate": jnp.array(0.5)},
        )
        return cpax.solve(
            problem,
            solver=diffrax.Tsit5(),
            saveat=cpax.SaveAt.linspace(7),
        ).ts

    np.testing.assert_allclose(integrate(jnp.array(2.0)), np.linspace(0.0, 2.0, 7), atol=1e-6)


def test_vmap_batches_independent_initial_conditions():
    system = VectorFieldSystem(_decay)

    def integrate(y0):
        problem = cpax.IVP(system, y0, (0.0, 1.0), {"rate": jnp.array(0.5)})
        result = cpax.solve(problem, solver=diffrax.Tsit5())
        return result.ys[-1], result.status

    initial = jnp.array([1.0, 2.0, 4.0])
    values, statuses = jax.vmap(integrate)(initial)

    np.testing.assert_allclose(values, initial * np.exp(-0.5), rtol=1e-4)
    np.testing.assert_array_equal(statuses, jnp.zeros(3, dtype=statuses.dtype))


def test_grad_of_final_value_matches_analytic_sensitivity():
    system = VectorFieldSystem(_decay)

    def final_value(rate):
        problem = cpax.IVP(system, jnp.array(2.0), (0.0, 2.0), {"rate": rate})
        return cpax.solve(
            problem,
            solver=diffrax.Tsit5(),
            rtol=1e-5,
            atol=1e-7,
        ).ys[-1]

    rate = jnp.array(0.5)
    derivative = jax.grad(final_value)(rate)
    expected = -2.0 * 2.0 * np.exp(-2.0 * 0.5)

    np.testing.assert_allclose(derivative, expected, rtol=3e-4, atol=3e-5)


def test_filter_jit_accepts_equinox_system_as_argument():
    @eqx.filter_jit
    def integrate(model):
        problem = cpax.IVP(model, jnp.array(2.0), (0.0, 1.0))
        return cpax.solve(problem, solver=diffrax.Tsit5()).ys[-1]

    value = integrate(DecaySystem(jnp.array(0.5)))
    np.testing.assert_allclose(value, 2.0 * np.exp(-0.5), rtol=1e-4)


def test_filter_grad_differentiates_array_valued_model_parameters():
    def final_value(model):
        problem = cpax.IVP(model, jnp.array(2.0), (0.0, 1.0))
        return cpax.solve(
            problem,
            solver=diffrax.Tsit5(),
            rtol=1e-5,
            atol=1e-7,
        ).ys[-1]

    gradient = eqx.filter_grad(final_value)(DecaySystem(jnp.array(0.5)))
    np.testing.assert_allclose(gradient.rate, -2.0 * np.exp(-0.5), rtol=3e-4)


def test_vmap_preserves_each_problem_failure_status():
    system = VectorFieldSystem(_decay)

    def integrate(end):
        problem = cpax.IVP(system, jnp.array(1.0), (jnp.array(0.0), end), {"rate": 0.5})
        return cpax.solve(
            problem,
            solver=diffrax.Euler(),
            stepsize_controller=diffrax.ConstantStepSize(),
            dt0=0.1,
            max_steps=2,
            throw=False,
        ).status

    statuses = jax.vmap(integrate)(jnp.array([0.1, 2.0]))
    np.testing.assert_array_equal(statuses, [0, 1])


def test_auto_selection_supports_jit_vmap_and_grad():
    system = VectorFieldSystem(_decay)

    @jax.jit
    def compiled(rate):
        problem = cpax.IVP(system, jnp.array(2.0), (0.0, 1.0), {"rate": rate})
        return cpax.solve(problem).ys[-1]

    rate = jnp.array(0.5)
    np.testing.assert_allclose(compiled(rate), 2 * np.exp(-0.5), rtol=1e-4)
    values = jax.vmap(compiled)(jnp.array([0.2, 0.5, 0.7]))
    np.testing.assert_allclose(values, 2 * np.exp(-np.array([0.2, 0.5, 0.7])), rtol=1e-4)
    np.testing.assert_allclose(jax.grad(compiled)(rate), -2 * np.exp(-0.5), rtol=5e-4)



def test_float32_ivp_jit_vmap_and_grad_with_explicit_dtype():
    """Keep the default-precision differentiable integration path covered."""
    with jax.enable_x64(False):
        system = VectorFieldSystem(_decay)

        @jax.jit
        def final_value(rate):
            problem = cpax.IVP(
                system,
                jnp.asarray(2.0, dtype=jnp.float32),
                (0.0, 1.0),
                {"rate": rate},
            )
            return cpax.solve(problem, solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7).ys[-1]

        rates = jnp.asarray([0.2, 0.5, 0.7], dtype=jnp.float32)
        values = jax.vmap(final_value)(rates)
        derivative = jax.grad(final_value)(rates[1])

        assert values.dtype == jnp.float32
        assert derivative.dtype == jnp.float32
        np.testing.assert_allclose(values, 2.0 * np.exp(-np.asarray(rates)), rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(derivative, -2.0 * np.exp(-0.5), rtol=5e-4, atol=5e-5)
