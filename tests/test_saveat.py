"""Sampling-policy tests independent of numerical integration details."""

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.dynamics import VectorFieldSystem


def _decay(t, y, args):
    del t
    return -args["rate"] * y


def _problem(t_span=(2.0, 7.0)):
    return cpax.IVP(
        system=VectorFieldSystem(_decay),
        y0=jnp.array(2.0),
        t_span=t_span,
        args={"rate": jnp.array(0.3)},
    )


def _solve(problem, saveat=None):
    return cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=saveat,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("n", [2, 3, 11, 101])
def test_linspace_saves_exact_count_and_both_endpoints(n):
    solution = _solve(_problem(), cpax.SaveAt.linspace(n))

    assert solution.ts.shape == (n,)
    assert solution.ys.shape == (n,)
    np.testing.assert_allclose(solution.ts, np.linspace(2.0, 7.0, n), rtol=1e-6, atol=1e-6)
    assert bool(solution.is_success)


@pytest.mark.parametrize("n", [-2, 0, 1, 2.5, True, "3"])
def test_linspace_rejects_invalid_sample_counts(n):
    with pytest.raises(ValueError, match="n >= 2"):
        cpax.SaveAt.linspace(n)


def test_linspace_policy_can_be_reused_across_intervals():
    saveat = cpax.SaveAt.linspace(5)
    first = _solve(_problem((0.0, 1.0)), saveat)
    second = _solve(_problem((3.0, 7.0)), saveat)

    np.testing.assert_allclose(first.ts, np.linspace(0.0, 1.0, 5), atol=1e-6)
    np.testing.assert_allclose(second.ts, np.linspace(3.0, 7.0, 5), atol=1e-6)


def test_default_saves_final_state_only():
    solution = _solve(_problem())

    assert solution.ts.shape == (1,)
    assert solution.ys.shape == (1,)
    np.testing.assert_allclose(solution.ts, [7.0], atol=1e-6)


def test_explicit_final_policy_matches_default():
    problem = _problem()
    default = _solve(problem)
    explicit = _solve(problem, cpax.SaveAt.final())

    np.testing.assert_allclose(explicit.ts, default.ts, rtol=0, atol=0)
    np.testing.assert_allclose(explicit.ys, default.ys, rtol=1e-6, atol=1e-6)


def test_at_saves_requested_irregular_times():
    requested = jnp.array([2.0, 2.1, 2.8, 4.75, 7.0])
    solution = _solve(_problem(), cpax.SaveAt.at(requested))

    np.testing.assert_allclose(solution.ts, requested, rtol=0, atol=1e-6)
    assert solution.ys.shape == (len(requested),)


def test_at_accepts_integer_times_without_keeping_integer_dtype():
    saveat = cpax.SaveAt.at([2, 3, 7])
    solution = _solve(_problem(), saveat)

    assert saveat.times.dtype.kind == "f"
    np.testing.assert_allclose(solution.ts, [2, 3, 7], atol=1e-6)


@pytest.mark.parametrize(
    "times",
    [
        [],
        1.0,
        [[2.0, 3.0]],
        [True, False],
        [2.0, 2.0],
        [4.0, 3.0],
        [2.0, jnp.nan],
        [2.0, jnp.inf],
        ["not", "numeric"],
    ],
)
def test_at_rejects_invalid_times(times):
    with pytest.raises((TypeError, ValueError)):
        cpax.SaveAt.at(times)


@pytest.mark.parametrize("times", [[1.9, 3.0], [3.0, 7.1], [1.0, 8.0]])
def test_at_rejects_times_outside_problem_interval(times):
    with pytest.raises(ValueError, match="within the IVP time interval"):
        _solve(_problem(), cpax.SaveAt.at(times))
