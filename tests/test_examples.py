"""Smoke checks for public example scripts (without GUI rendering)."""

import runpy
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState

ROOT = Path(__file__).resolve().parents[1]


def test_harmonic_example_equations():
    system = NewtonianSystem(potential=lambda q: 0.5 * q**2, mass=1.0)
    problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 1.0))
    result = cpax.solve(problem, saveat=cpax.SaveAt.linspace(21))
    np.testing.assert_allclose(result.ys.q, jnp.cos(result.ts), atol=2e-4)


def test_duffing_example_force():
    namespace = runpy.run_path(str(ROOT / "examples" / "duffing_oscillator.py"))
    system = NewtonianSystem(potential=namespace["potential"], mass=1.0)
    np.testing.assert_allclose(system.force(jnp.array(2.0)), -6.0)


def test_import_all_simple_examples():
    for filename in ("harmonic_oscillator.py", "duffing_oscillator.py", "lorenz.py"):
        assert callable(runpy.run_path(str(ROOT / "examples" / filename))["main"])
