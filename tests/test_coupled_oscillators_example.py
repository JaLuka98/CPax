"""Analytical normal-mode checks for the runnable coupled-oscillator example."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "coupled_oscillators.py"


def _functions():
    return runpy.run_path(str(EXAMPLE))


def test_coupled_normal_modes_and_superposition_float64():
    simulate = _functions()["simulate"]
    with jax.enable_x64(True):
        mass, spring, coupling = 1.5, 3.0, 0.8
        low = np.sqrt(spring / mass)
        high = np.sqrt((spring + 2 * coupling) / mass)
        for q0, frequency in (((1.0, 1.0), low), ((1.0, -1.0), high)):
            system, sol = simulate(q0=q0, mass=mass, spring=spring, coupling=coupling, t_end=6.0, steps=1200)
            assert bool(sol.is_success)
            assert sol.ys.q.dtype == jnp.float64
            expected = np.cos(frequency * np.asarray(sol.ts))[:, None] * np.asarray(q0)[None, :]
            np.testing.assert_allclose(sol.ys.q, expected, rtol=0, atol=4e-4)
            energies = jax.vmap(system.energy)(sol.ys)
            assert float(jnp.max(jnp.abs((energies - energies[0]) / energies[0]))) < 3e-4

        _, mixed = simulate(q0=(1.0, 0.0), mass=mass, spring=spring, coupling=coupling, t_end=6.0, steps=1200)
        t = np.asarray(mixed.ts)
        expected = 0.5 * np.stack(
            [np.cos(low * t) + np.cos(high * t), np.cos(low * t) - np.cos(high * t)],
            axis=1,
        )
        np.testing.assert_allclose(mixed.ys.q, expected, rtol=0, atol=4e-4)


def test_coupled_example_jax_transforms_float32():
    simulate = _functions()["simulate"]
    with jax.enable_x64(False):

        def last_position(coupling):
            _, sol = simulate(q0=(1.0, -1.0), coupling=coupling, t_end=1.0, steps=100)
            return sol.ys.q[-1, 0]

        coupling = jnp.asarray(0.6, dtype=jnp.float32)
        omega = jnp.sqrt(1.0 + 2.0 * coupling)
        np.testing.assert_allclose(jax.jit(last_position)(coupling), jnp.cos(omega), atol=4e-4)
        values = jnp.array([0.2, 0.6, 1.0], dtype=jnp.float32)
        np.testing.assert_allclose(jax.vmap(last_position)(values), jnp.cos(jnp.sqrt(1.0 + 2.0 * values)), atol=4e-4)
        expected_derivative = -jnp.sin(omega) / omega
        np.testing.assert_allclose(jax.grad(last_position)(coupling), expected_derivative, atol=5e-4)
