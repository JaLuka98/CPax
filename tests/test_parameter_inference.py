"""Independent numerical/statistical checks for the damping-inference example."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    """Load the runnable example without executing its plotting entry point."""
    return runpy.run_path(str(ROOT / "examples" / "damping_parameter_inference.py"))


@pytest.fixture(scope="module")
def measurements(example):
    times = jnp.linspace(0.0, example["END_TIME"], example["NUM_OBSERVATIONS"])
    observations = example["generate_observations"](times)
    return times, observations


def test_numerical_forward_trajectory_matches_analytical_reference(example, measurements):
    times, _ = measurements
    gamma = jnp.asarray(0.75)
    numerical = example["simulated_position"](gamma, times)
    analytical = example["analytical_position"](gamma, times)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-4, atol=3e-5)


def test_trajectory_loss_gradient_matches_independent_analytical_sensitivity(example, measurements):
    """Check differentiating the *numerical* trajectory, not only the potential."""
    times, observations = measurements
    gamma = 0.75
    mass = example["MASS"]
    stiffness = example["SPRING_CONSTANT"]
    noise_sigma = example["OBSERVATION_SIGMA"]
    t = np.asarray(times)
    beta = gamma / (2 * mass)
    omega_d = np.sqrt(stiffness / mass - beta**2)
    reference = np.exp(-beta * t) * (
        np.cos(omega_d * t) + beta / omega_d * np.sin(omega_d * t)
    )
    # Exact derivative dq/dgamma for q(0)=1 and p(0)=0.
    sensitivity = (
        np.exp(-beta * t)
        * (stiffness / mass) / (2 * mass * omega_d**3)
        * (np.sin(omega_d * t) - omega_d * t * np.cos(omega_d * t))
    )
    expected = np.sum((reference - np.asarray(observations)) * sensitivity / noise_sigma**2)
    measured = jax.jit(jax.grad(example["negative_log_likelihood"]))(
        jnp.asarray(gamma), times, observations
    )
    np.testing.assert_allclose(measured, expected, rtol=3e-3, atol=0.5)


def test_parameter_recovery_and_local_uncertainty(example, measurements):
    times, observations = measurements
    gamma_hat, standard_error, losses = example["fit_damping"](times, observations)

    assert bool(jnp.all(jnp.isfinite(losses)))
    assert float(losses[-1]) < 0.2 * float(losses[0])
    # The fixed noise seed gives a reproducible realisation, not a promise
    # that *every* random realisation falls within one standard error.
    assert abs(float(gamma_hat) - example["TRUE_DAMPING"]) < 0.04
    assert 0.003 < float(standard_error) < 0.03

    # Compare the observed-Hessian uncertainty to the Fisher-information
    # approximation calculated from the independent analytical sensitivity.
    mass = example["MASS"]
    stiffness = example["SPRING_CONSTANT"]
    beta = float(gamma_hat) / (2 * mass)
    omega_d = np.sqrt(stiffness / mass - beta**2)
    t = np.asarray(times)
    dq_dgamma = (
        np.exp(-beta * t)
        * (stiffness / mass) / (2 * mass * omega_d**3)
        * (np.sin(omega_d * t) - omega_d * t * np.cos(omega_d * t))
    )
    fisher_error = example["OBSERVATION_SIGMA"] / np.sqrt(np.sum(dq_dgamma**2))
    np.testing.assert_allclose(standard_error, fisher_error, rtol=0.15)

    gradient = jax.grad(example["mean_nll_log_damping"])(
        jnp.log(gamma_hat), times, observations
    )
    assert abs(float(gradient)) < 0.01


def test_seeded_observations_are_deterministic(example, measurements):
    times, observations = measurements
    np.testing.assert_array_equal(example["generate_observations"](times), observations)
    different_seed = example["generate_observations"](times, seed=example["NOISE_SEED"] + 1)
    assert bool(jnp.any(different_seed != observations))
