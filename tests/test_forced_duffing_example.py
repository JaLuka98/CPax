"""Physical work, dissipation and reference tests of the driven examples."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    return runpy.run_path(str(EXAMPLES / name))


def test_duffing_conservative_limit_matches_newtonian_float32():
    with jax.enable_x64(False):
        duffing = _load("forced_duffing_oscillator.py")
        system, forced = duffing["simulate"](damping=0.0, drive=0.0, q0=-1.0, velocity0=0.2, t_end=5.0, samples=251)
        conservative = NewtonianSystem(
            mass=1.0,
            potential=lambda q: -0.5 * q**2 + 0.25 * q**4,
        )
        reference = cpax.solve(
            cpax.IVP(conservative, PhaseSpaceState(-1.0, 0.2), (0.0, 5.0)),
            saveat=cpax.SaveAt.linspace(251),
            rtol=1e-5,
            atol=1e-7,
        )
        assert bool(forced.is_success) and bool(reference.is_success)
        assert forced.ys.q.dtype == jnp.float32
        np.testing.assert_allclose(forced.ys.q, reference.ys.q, atol=3e-4, rtol=0)
        np.testing.assert_allclose(forced.ys.p, reference.ys.p, atol=3e-4, rtol=0)
        energies = jax.vmap(system.mechanical_energy)(forced.ys)
        assert float(jnp.max(jnp.abs(energies - energies[0]))) < 4e-4


def test_duffing_damping_and_driven_energy_balance_float64():
    simulate = _load("forced_duffing_oscillator.py")["simulate"]
    with jax.enable_x64(True):
        for damping, drive in ((0.4, 0.0), (0.3, 0.5)):
            model, sol = simulate(
                damping=damping,
                drive=drive,
                frequency=1.7,
                velocity0=0.2,
                t_end=6.0,
                samples=1201,
                rtol=1e-9,
                atol=1e-11,
            )
            assert bool(sol.is_success)
            energy = np.asarray(jax.vmap(model.mechanical_energy)(sol.ys))
            power = np.asarray(jax.vmap(model.applied_power)(sol.ts, sol.ys))
            assert sol.ys.q.dtype == jnp.float64
            np.testing.assert_allclose(
                energy[-1] - energy[0],
                np.trapezoid(power, np.asarray(sol.ts)),
                rtol=0,
                atol=8e-4,
            )
            if drive == 0.0:
                assert np.all(power <= 1e-12)
                assert np.all(np.diff(energy) <= 2e-6)
                assert energy[-1] < energy[0] - 0.01


def test_driven_duffing_reference_refinement():
    simulate = _load("forced_duffing_oscillator.py")["simulate"]
    with jax.enable_x64(True):
        params = {"damping": 0.3, "drive": 0.5, "frequency": 1.7, "velocity0": 0.2, "t_end": 6.0, "samples": 151}
        _, ordinary = simulate(**params, rtol=1e-6, atol=1e-8)
        _, refined = simulate(**params, rtol=1e-10, atol=1e-12)
        assert bool(ordinary.is_success) and bool(refined.is_success)
        np.testing.assert_allclose(ordinary.ys.q, refined.ys.q, rtol=0, atol=5e-4)
        np.testing.assert_allclose(ordinary.ys.p, refined.ys.p, rtol=0, atol=5e-4)


def test_existing_forced_pendulum_damping_limit():
    simulate = _load("forced_damped_pendulum.py")["simulate"]
    with jax.enable_x64(False):
        system, sol = simulate(damping=0.4, drive=0.0, t_end=4.0, samples=401)
        assert bool(sol.is_success)
        energy = np.asarray(jax.vmap(system.mechanical_energy)(sol.ys))
        power = np.asarray(jax.vmap(system.applied_power)(sol.ts, sol.ys))
        assert np.all(power <= 1e-7)
        assert np.all(np.diff(energy) <= 3e-5)
        np.testing.assert_allclose(
            energy[-1] - energy[0],
            np.trapezoid(power, np.asarray(sol.ts)),
            atol=2e-3,
            rtol=0,
        )
