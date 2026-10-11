"""Tests for solver-independent initial-value problems."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from cpax import IVP
from cpax.dynamics import DynamicalSystem, VectorFieldSystem


def decay(t, y, args):
    del t
    return -args["rate"] * y


class DecaySystem(DynamicalSystem):
    rate: jax.Array

    def vector_field(self, t, state, args=None):
        del t, args
        return -self.rate * state


def test_one_system_supports_multiple_problems():
    system = VectorFieldSystem(decay)
    first = IVP(system, 1.0, (0.0, 1.0), {"rate": jnp.array(2.0)})
    second = IVP(system, 3.0, (1.0, 4.0), {"rate": jnp.array(4.0)})

    assert first.system is second.system
    assert isinstance(first.y0, jax.Array)
    assert first.y0.dtype.kind == "f"
    assert jnp.allclose(first.system.vector_field(0.0, first.y0, first.args), -2.0)
    assert jnp.allclose(second.system.vector_field(1.0, second.y0, second.args), -12.0)
    assert float(first.t_span[0]) == 0.0
    assert float(second.t_span[1]) == 4.0


def test_array_valued_pytree_initial_state_is_preserved():
    y0 = {
        "position": jnp.array([1.0, 2.0]),
        "velocity": (jnp.array([0.0, 0.5]), jnp.array(-1.0)),
    }
    system = VectorFieldSystem(lambda t, state, args: state)
    problem = IVP(system, y0, (0, 1))

    assert jax.tree.structure(problem.y0) == jax.tree.structure(y0)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(problem.y0))
    assert all(leaf.dtype.kind == "f" for leaf in jax.tree.leaves(problem.y0))
    assert jnp.array_equal(problem.y0["position"], y0["position"])


@pytest.mark.parametrize(
    "interval",
    [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf")),
     (0.0,), (0.0, 1.0, 2.0), ([0.0, 1.0], 2.0), (0.0, 1.0j)],
)
def test_invalid_intervals_are_rejected(interval):
    with pytest.raises((ValueError, TypeError)):
        IVP(VectorFieldSystem(decay), jnp.array(1.0), interval)


@pytest.mark.parametrize(
    "state",
    [jnp.array([1, 2]), jnp.array(True), {"label": "not a state"}, {}],
)
def test_noncontinuous_initial_states_are_rejected(state):
    with pytest.raises((ValueError, TypeError)):
        IVP(VectorFieldSystem(decay), state, (0.0, 1.0))


def test_invalid_system_is_rejected():
    with pytest.raises(TypeError, match="DynamicalSystem"):
        IVP(object(), jnp.array(1.0), (0.0, 1.0))


def test_problem_supports_jit_vmap_and_differentiation():
    system = VectorFieldSystem(decay)

    @jax.jit
    def evaluate(rate):
        problem = IVP(system, jnp.array(2.0), (jnp.array(0.0), jnp.array(1.0)), {"rate": rate})
        return problem.system.vector_field(problem.t_span[0], problem.y0, problem.args)

    assert jnp.allclose(evaluate(jnp.array(3.0)), -6.0)

    def evaluate_amplitude(amplitude):
        problem = IVP(system, amplitude, (0.0, 1.0), {"rate": jnp.array(2.0)})
        return problem.system.vector_field(0.0, problem.y0, problem.args)

    amplitudes = jnp.array([1.0, 2.0, 4.0])
    assert jnp.allclose(jax.vmap(evaluate_amplitude)(amplitudes), -2.0 * amplitudes)
    assert jnp.allclose(jax.grad(evaluate)(jnp.array(3.0)), -2.0)


def test_problem_with_model_parameters_supports_equinox_transformations():
    problem = IVP(DecaySystem(jnp.array(2.0)), jnp.array(3.0), (0.0, 1.0))

    @eqx.filter_jit
    def evaluate(p):
        return p.system.vector_field(p.t_span[0], p.y0)

    assert jnp.allclose(evaluate(problem), -6.0)
    derivative = eqx.filter_grad(evaluate)(problem)
    assert jnp.allclose(derivative.system.rate, -3.0)
