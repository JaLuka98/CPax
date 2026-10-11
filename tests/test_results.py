"""Tests for the backend-independent structured solution type."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from cpax.results import Solution, SolutionStatus


def test_sample_times_and_states_have_matching_shapes():
    solution = Solution(
        ts=[0, 1, 2],
        ys=jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        status=SolutionStatus.SUCCESS,
        stats={"num_steps": jnp.array(5)},
        metadata={"solver": "reference"},
    )
    assert solution.ts.shape == (3,)
    assert solution.ts.dtype.kind == "f"
    assert solution.ys.shape == (3, 2)
    assert bool(solution.is_success)
    assert solution.stats["num_steps"] == 5
    assert solution.metadata["solver"] == "reference"


def test_pytree_states_retain_sample_axis():
    ys = {
        "position": jnp.ones((3, 2)),
        "momentum": (jnp.arange(3.0), jnp.zeros((3, 1))),
    }
    solution = Solution(jnp.array([0.0, 0.5, 1.0]), ys, SolutionStatus.SUCCESS)

    assert jax.tree.structure(solution.ys) == jax.tree.structure(ys)
    assert [leaf.shape for leaf in jax.tree.leaves(solution.ys)] == [(3, 1), (3, 2), (3,)]
    assert solution.stats == {}
    assert solution.metadata == {}


def test_failure_preserves_partial_or_nonfinite_trajectory_and_diagnostics():
    solution = Solution(
        ts=jnp.array([0.0, 0.5, jnp.inf]),
        ys=jnp.array([1.0, 0.7, jnp.nan]),
        status=SolutionStatus.FAILED,
        stats={"num_rejected_steps": jnp.array(8)},
        metadata={"reason": "step size underflow"},
    )

    assert not bool(solution.is_success)
    assert jnp.isnan(solution.ys[-1])
    assert jnp.isinf(solution.ts[-1])
    assert solution.metadata["reason"] == "step size underflow"


@pytest.mark.parametrize(
    "ts,ys,status,error",
    [
        ([], jnp.array([]), 0, ValueError),
        (jnp.ones((2, 1)), jnp.ones(2), 0, ValueError),
        (jnp.array([0.0, 1.0]), jnp.array(1.0), 0, ValueError),
        (jnp.array([0.0, 1.0]), jnp.ones(3), 0, ValueError),
        (jnp.array([0.0, 1.0]), {"q": jnp.array([1, 2])}, 0, TypeError),
        (jnp.array([0.0, 1.0]), {}, 0, ValueError),
        (jnp.array([0.0, 1.0]), jnp.ones(2), 1.5, TypeError),
        (jnp.array([0.0, 1.0]), jnp.ones(2), 17, ValueError),
    ],
)
def test_invalid_result_structures_are_rejected(ts, ys, status, error):
    with pytest.raises(error):
        Solution(ts, ys, status)


def test_solution_is_compatible_with_jax_transformations():
    ts = jnp.array([0.0, 1.0])

    @jax.jit
    def successful(code):
        solution = Solution(ts, jnp.array([1.0, 0.5]), code)
        return solution.is_success

    assert successful(jnp.array(0, dtype=jnp.int32))
    assert not successful(jnp.array(1, dtype=jnp.int32))

    def final_value(amplitude):
        solution = Solution(ts, jnp.stack([amplitude, amplitude / 2]), SolutionStatus.SUCCESS)
        return solution.ys[-1]

    amplitudes = jnp.array([2.0, 4.0, 6.0])
    assert jnp.allclose(jax.vmap(final_value)(amplitudes), amplitudes / 2)
    assert jnp.allclose(jax.grad(final_value)(jnp.array(2.0)), 0.5)

    with_metadata = Solution(ts, jnp.array([1.0, 0.5]), 0, metadata={"solver": "reference"})
    assert bool(eqx.filter_jit(lambda s: s.is_success)(with_metadata))
