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


class TrainableVectorField(eqx.Module):
    k: jax.Array

    def __call__(self, t, state, args):
        del t, args
        q, p = state
        return p, -self.k * q


def test_abstract_class_requires_vector_field():
    with pytest.raises(TypeError):
        DynamicalSystem()


def test_custom_dynamical_system_is_an_equinox_pytree():
    system = Oscillator(jnp.array(2.0))
    state = (jnp.array(1.0), jnp.array(0.0))
    assert isinstance(system, eqx.Module)
    assert jax.tree.leaves(system)
    derivative = system.vector_field(0.0, state)
    assert jnp.allclose(jnp.stack(derivative), jnp.array([0.0, -2.0]))


def test_vector_field_system_forwards_time_and_args():
    def rhs(t, state, args):
        return {"x": -args["rate"] * state["x"] + t}

    system = VectorFieldSystem(rhs)
    result = system.vector_field(jnp.array(1.0), {"x": jnp.array(2.0)}, {"rate": jnp.array(3.0)})
    assert isinstance(result, dict)
    assert jnp.allclose(result["x"], -5.0)
    assert jax.tree.structure(result) == jax.tree.structure({"x": jnp.array(2.0)})


def test_default_args_are_none():
    system = VectorFieldSystem(lambda t, y, args: y if args is None else t)
    assert system.vector_field(0.0, jnp.array(2.0)) == 2.0


def test_jit_vmap_and_differentiation_of_external_args():
    system = VectorFieldSystem(oscillator)
    state = (jnp.array(1.0), jnp.array(0.0))
    evaluate = jax.jit(lambda s, k: system.vector_field(0.0, s, {"k": k}))
    assert jnp.allclose(jnp.stack(evaluate(state, jnp.array(2.0))), jnp.array([0.0, -2.0]))

    positions = jnp.array([1.0, 2.0, 3.0])
    batched = jax.vmap(lambda q: evaluate((q, jnp.array(0.0)), jnp.array(2.0))[1])(positions)
    assert jnp.allclose(batched, -2.0 * positions)

    sensitivity = jax.grad(lambda k: system.vector_field(0.0, state, {"k": k})[1])(jnp.array(2.0))
    assert jnp.allclose(sensitivity, -1.0)


def test_jit_vmap_and_grad_for_array_parameter_subclass():
    system = Oscillator(jnp.array(3.0))
    state = (jnp.array(2.0), jnp.array(0.0))
    derivative = jax.jit(lambda model, y: model.vector_field(0.0, y))(system, state)
    assert jnp.allclose(derivative[1], -6.0)

    batched = jax.vmap(lambda q: system.vector_field(0.0, (q, jnp.array(0.0)))[1])(jnp.array([1.0, 2.0]))
    assert jnp.allclose(batched, jnp.array([-3.0, -6.0]))
    grad = jax.grad(lambda k: Oscillator(k).vector_field(0.0, state)[1])(jnp.array(3.0))
    assert jnp.allclose(grad, -2.0)


def test_callable_equinox_module_parameters_are_differentiable():
    system = VectorFieldSystem(TrainableVectorField(jnp.array(2.0)))
    state = (jnp.array(3.0), jnp.array(0.0))

    @eqx.filter_jit
    def evaluate(model):
        return model.vector_field(0.0, state)[1]

    assert jnp.allclose(evaluate(system), -6.0)

    def loss(model):
        return model.vector_field(0.0, state)[1]

    gradient = eqx.filter_grad(loss)(system)
    assert jnp.allclose(gradient.rhs.k, -3.0)


def test_public_names_and_legacy_import():
    import cpax

    assert cpax.dynamics.VectorFieldSystem is VectorFieldSystem
    assert callable(cpax.ode.simulate_rk4_scan)
