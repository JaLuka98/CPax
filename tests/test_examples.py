"""Numerical tests of the runnable physics-first examples."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState, gravitational_potential

ROOT = Path(__file__).resolve().parents[1]


def test_harmonic_example_equations():
    system = NewtonianSystem(potential=lambda q: 0.5 * q**2, mass=1.0)
    problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 1.0))
    result = cpax.solve(problem, saveat=cpax.SaveAt.linspace(21))
    assert bool(result.is_success)
    np.testing.assert_allclose(result.ys.q, jnp.cos(result.ts), atol=2e-4)


def test_duffing_example_both_energy_regimes():
    namespace = runpy.run_path(str(ROOT / "examples" / "duffing_oscillator.py"))
    system = NewtonianSystem(potential=namespace["potential"], mass=1.0)
    np.testing.assert_allclose(system.force(jnp.array(2.0)), -6.0)

    for p0, crosses_barrier in ((0.2, False), (0.9, True)):
        model, result = namespace["simulate"](-1.0, p0, t1=16.0, samples=401)
        assert bool(result.is_success)
        assert result.ys.q.shape == (401,)
        energies = jax.vmap(model.energy)(result.ys)
        np.testing.assert_allclose(energies, energies[0], rtol=1e-3, atol=2e-4)
        assert bool(jnp.any(result.ys.q > 0.0)) == crosses_barrier
        if not crosses_barrier:
            assert bool(jnp.all(result.ys.q < 0.0))


def test_short_gravitational_ivp_conserves_momentum():
    masses = jnp.array([1.0, 1.0])
    system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
    q0 = jnp.array([[-1.0, 0.0], [1.0, 0.0]])
    v0 = jnp.array([[0.0, -0.5], [0.0, 0.5]])
    initial = PhaseSpaceState(q0, masses[:, None] * v0)
    result = cpax.solve(
        cpax.IVP(system, initial, (0.0, 0.2)),
        saveat=cpax.SaveAt.linspace(21),
        rtol=1e-5,
        atol=1e-7,
    )
    assert bool(result.is_success)
    assert result.ys.q.shape == (21, 2, 2)
    assert result.ys.q.dtype == jnp.float32
    total_momentum = jnp.sum(result.ys.p, axis=1)
    # Exactly symmetric cancellation is a useful regression but a weak general conservation test.
    np.testing.assert_allclose(
        total_momentum, np.broadcast_to(total_momentum[0], total_momentum.shape), rtol=0, atol=1e-6
    )
    energy = jax.vmap(system.energy)(result.ys)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-3, atol=1e-5)


def test_asymmetric_gravitational_ivp_conserves_momentum_float32():
    """Check momentum conservation without exact two-body symmetry."""
    with jax.enable_x64(False):
        masses = jnp.asarray([1.1, 0.8, 0.5], dtype=jnp.float32)
        q0 = jnp.asarray([[-1.2, 0.3], [1.5, -0.4], [0.2, 1.8]], dtype=jnp.float32)
        v0 = jnp.asarray([[0.12, -0.31], [-0.25, 0.18], [0.33, 0.07]], dtype=jnp.float32)
        initial = PhaseSpaceState(q0, masses[:, None] * v0)
        system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
        result = cpax.solve(
            cpax.IVP(system, initial, (0.0, 0.2)),
            saveat=cpax.SaveAt.linspace(21),
            rtol=1e-5,
            atol=1e-7,
        )

        assert bool(result.is_success)
        assert result.ys.p.dtype == jnp.float32
        assert float(jnp.linalg.norm(jnp.sum(initial.p, axis=0))) > 0.1
        # Verify that interacting particles exchanged appreciable momentum.
        assert float(jnp.max(jnp.abs(result.ys.p[-1] - result.ys.p[0]))) > 1e-3

        total_momentum = jnp.sum(result.ys.p, axis=1)
        # Measured float32 component drift <= 3.58e-7 (CPU, JAX 0.12.0).
        # The absolute threshold reflects rounding and integration error,
        # independently of the solver's local rtol/atol.
        np.testing.assert_allclose(
            total_momentum,
            np.broadcast_to(total_momentum[0], total_momentum.shape),
            rtol=0,
            atol=1e-6,
        )


def test_import_all_simple_examples():
    for filename in ("harmonic_oscillator.py", "duffing_oscillator.py", "lorenz.py"):
        assert callable(runpy.run_path(str(ROOT / "examples" / filename))["main"])
