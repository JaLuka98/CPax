"""Regression tests for Diffrax result/status translation."""

import diffrax
import jax.numpy as jnp

import cpax
from cpax.dynamics import VectorFieldSystem
from cpax.results import SolutionStatus


def _decay(t, y, args):
    del t
    return -args["rate"] * y


def test_diffrax_result_enum_converts_to_cpax_status():
    problem = cpax.IVP(
        system=VectorFieldSystem(_decay),
        y0=jnp.array(2.0),
        t_span=(0.0, 1.0),
        args={"rate": jnp.array(0.5)},
    )
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(11),
    )
    assert int(solution.status) == SolutionStatus.SUCCESS
    assert bool(solution.is_success)
    assert solution.ts.shape == (11,)
    assert jnp.allclose(solution.ys[-1], 2.0 * jnp.exp(-0.5), rtol=2e-5)
    assert solution.metadata["backend_result"] == diffrax.RESULTS.successful
