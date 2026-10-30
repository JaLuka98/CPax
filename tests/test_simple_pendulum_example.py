"""Physical reference checks for the runnable nonlinear pendulum example."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "simple_pendulum.py"


def _simulate():
    return runpy.run_path(str(EXAMPLE))["simulate"]


def _quarter_period(times, positions):
    """Interpolate the first positive-to-negative angle crossing."""
    t = np.asarray(times)
    q = np.asarray(positions)
    crossings = np.flatnonzero((q[:-1] > 0) & (q[1:] <= 0))
    assert crossings.size > 0
    i = crossings[0]
    return t[i] - q[i] * (t[i + 1] - t[i]) / (q[i + 1] - q[i])


def test_pendulum_small_angle_and_finite_amplitude_period():
    simulate = _simulate()
    with jax.enable_x64(True):
        _, small = simulate(0.03, t_end=3.0, steps=1200)
        _, large = simulate(1.0, t_end=3.0, steps=1200)
        assert bool(small.is_success) and bool(large.is_success)
        assert small.ys.q.dtype == jnp.float64
        omega0 = np.sqrt(9.81)
        np.testing.assert_allclose(
            small.ys.q,
            0.03 * np.cos(omega0 * np.asarray(small.ts)),
            rtol=0,
            atol=3e-4,
        )
        period_small = 4 * _quarter_period(small.ts, small.ys.q)
        period_large = 4 * _quarter_period(large.ts, large.ys.q)
        period_harmonic = 2 * np.pi / omega0
        assert abs(period_small / period_harmonic - 1) < 0.003
        assert period_large > 1.05 * period_harmonic


def test_pendulum_energy_and_second_order_convergence():
    simulate = _simulate()
    with jax.enable_x64(True):
        model, solution = simulate(1.0, angular_velocity0=0.2, t_end=4.0, steps=800)
        assert bool(solution.is_success)
        energy = jax.vmap(model.energy)(solution.ys)
        relative_drift = jnp.max(jnp.abs((energy - energy[0]) / energy[0]))
        assert float(relative_drift) < 3e-4

        # Compare final canonical states against a much finer fixed-step solve.
        def endpoint(steps):
            _, sol = simulate(0.8, angular_velocity0=0.15, t_end=1.0, steps=steps)
            return np.array([sol.ys.q[-1], sol.ys.p[-1]], dtype=float)

        reference = endpoint(1600)
        errors = [np.linalg.norm(endpoint(n) - reference) for n in (50, 100, 200)]
        assert errors[0] > 3.0 * errors[1] > 9.0 * errors[2]
