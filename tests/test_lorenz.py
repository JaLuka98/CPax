"""Public LorenzSystem integration and transformation tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.dynamics import DynamicalSystem, LorenzSystem


def test_defaults_and_parameters():
    system = LorenzSystem()
    assert isinstance(system, DynamicalSystem)
    np.testing.assert_allclose([system.sigma, system.rho, system.beta], [10.0, 28.0, 8.0 / 3.0])
    state = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(system.vector_field(0.0, state), [10.0, 23.0, -6.0])
    np.testing.assert_allclose(LorenzSystem(sigma=2.0).vector_field(0.0, state)[0], 2.0)


def test_rejects_invalid_states_and_args():
    system = LorenzSystem()
    with pytest.raises(ValueError, match="shape"):
        system.vector_field(0.0, jnp.ones(2))
    with pytest.raises(ValueError, match="IVP.args"):
        system.vector_field(0.0, jnp.ones(3), {"rho": 2})
    with pytest.raises(TypeError, match="scalar"):
        LorenzSystem(sigma=jnp.ones(2))


def test_lorenz_jit_vmap_grad():
    state = jnp.array([1.0, 2.0, 3.0])
    field = jax.jit(lambda sigma: LorenzSystem(sigma=sigma).vector_field(0.0, state))
    np.testing.assert_allclose(field(jnp.array(10.0)), [10.0, 23.0, -6.0])
    np.testing.assert_allclose(jax.vmap(lambda s: field(s)[0])(jnp.array([2.0, 4.0])), [2.0, 4.0])
    np.testing.assert_allclose(jax.grad(lambda s: field(s)[0])(jnp.array(10.0)), 1.0)


def test_lorenz_solves_public_ivp():
    problem = cpax.IVP(LorenzSystem(), jnp.array([1.0, 1.0, 1.0]), (0.0, 0.2))
    result = cpax.solve(problem, saveat=cpax.SaveAt.linspace(11))
    assert bool(result.is_success)
    assert result.ys.shape == (11, 3)
    assert bool(jnp.all(jnp.isfinite(result.ys)))
