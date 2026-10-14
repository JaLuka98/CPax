"""Phase-space state contracts and PyTree interoperability."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from cpax.mechanics import PhaseSpaceState


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_state_preserves_coordinates_and_pytree_structure(shape):
    q = jnp.ones(shape)
    p = jnp.zeros(shape)
    state = PhaseSpaceState(q, p)

    assert isinstance(state, eqx.Module)
    assert state.q.shape == state.p.shape == shape
    assert len(jax.tree.leaves(state)) == 2
    assert jax.tree.structure(state) == jax.tree.structure(PhaseSpaceState(q, p))

    rebuilt = jax.tree.unflatten(jax.tree.structure(state), jax.tree.leaves(state))
    assert isinstance(rebuilt, PhaseSpaceState)
    assert jnp.array_equal(rebuilt.q, q)
    assert jnp.array_equal(rebuilt.p, p)


def test_python_numeric_scalars_normalise_to_real_arrays():
    state = PhaseSpaceState(1, 0.0)
    assert state.q.shape == state.p.shape == ()
    assert state.q.dtype.kind == state.p.dtype.kind == "f"


@pytest.mark.parametrize(
    ("q", "p", "error"),
    [
        (jnp.ones(2), jnp.zeros(3), ValueError),
        (jnp.array([1, 2]), jnp.zeros(2), TypeError),
        (jnp.ones(2), jnp.array([0, 1]), TypeError),
        (jnp.array(1.0 + 0.0j), 0.0, TypeError),
        (True, False, TypeError),
        (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2)), ValueError),
        ("invalid", 0.0, (TypeError, ValueError)),
    ],
)
def test_invalid_coordinates_are_rejected(q, p, error):
    with pytest.raises(error):
        PhaseSpaceState(q, p)


def test_jit_and_vmap_work_with_phase_space():
    @jax.jit
    def make_state(q):
        return PhaseSpaceState(q, 0.0)

    state = make_state(jnp.array(2.0))
    assert jnp.allclose(state.q, 2.0)

    states = jax.vmap(make_state)(jnp.array([1.0, 2.0]))
    assert isinstance(states, PhaseSpaceState)
    assert states.q.shape == states.p.shape == (2,)
