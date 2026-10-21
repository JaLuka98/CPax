"""Contract, transformation and interoperability tests for dynamical systems."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from cpax.dynamics import DynamicalSystem, VectorFieldSystem


def oscillator(t, state, args):
    del t
    q, p = state
    return p, -args["k"] * q


class Oscillator(DynamicalSystem):
    k: jax.Array

    def vector_field(self, t, state, args=None):
        del t, args
        q, p = state
        return p, -self.k * q


def test_abstract_class_requires_vector_field():
    with pytest.raises(TypeError):
        DynamicalSystem()


def test_vector_field_supports_args_and_jax():
    system = VectorFieldSystem(oscillator)
    state = (jnp.array(1.0), jnp.array(0.0))
    fn = jax.jit(lambda k: system.vector_field(0.0, state, {"k": k})[1])
    assert fn(jnp.array(2.0)) == -2.0
    assert jax.grad(fn)(jnp.array(2.0)) == -1.0
    assert jnp.allclose(
        jax.vmap(fn)(jnp.array([1.0, 2.0, 3.0])),
        jnp.array([-1.0, -2.0, -3.0]),
    )


def test_equinox_system_pytree():
    system = Oscillator(jnp.array(3.0))
    assert isinstance(system, eqx.Module)
    assert jax.tree.leaves(system)
    derivative = system.vector_field(0.0, (jnp.array(2.0), jnp.array(0.0)))
    assert jnp.allclose(jnp.stack(derivative), jnp.array([0.0, -6.0]))


class TrainableVectorField(eqx.Module):
    k: jax.Array

    def __call__(self, t, state, args):
        del t, args
        q, p = state
        return p, -self.k * q


def test_callable_equinox_module_filter_transformations():
    system = VectorFieldSystem(TrainableVectorField(jnp.array(2.0)))
    state = (jnp.array(3.0), jnp.array(0.0))

    @eqx.filter_jit
    def evaluate(model):
        return model.vector_field(0.0, state)[1]

    assert jnp.allclose(evaluate(system), -6.0)

    gradient = eqx.filter_grad(lambda model: model.vector_field(0.0, state)[1])(system)
    assert jnp.allclose(gradient.rhs.k, -3.0)
