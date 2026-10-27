"""Infer viscous damping from noisy oscillator positions using differentiable CPax.

Run with `uv run python examples/damping_parameter_inference.py`.

This deliberately illustrates differentiation *through a numerical simulation*.
For this analytically solvable oscillator, fitting the closed-form trajectory
directly would normally be simpler and more accurate. The numerical approach
becomes useful for physical systems without tractable analytical trajectories.
"""

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState

# Consistent arbitrary mechanical units: mass, stiffness, time and position.
MASS = 1.0
SPRING_CONSTANT = 4.0
INITIAL_POSITION = 1.0
TRUE_DAMPING = 0.55
OBSERVATION_SIGMA = 0.035
END_TIME = 8.0
NUM_OBSERVATIONS = 81
NOISE_SEED = 2026

# These settings are adequate for the example; they do not control the output
# sampling grid. JAX's default precision (usually float32) is left unchanged.
RTOL = 1e-6
ATOL = 1e-8


def potential(q: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Autonomous spring potential V(q) = k q² / 2."""
    return 0.5 * params["k"] * q**2


def applied_force(t: jax.Array, q: jax.Array, velocity: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Viscous force F = -gamma * velocity."""
    del t, q
    return -params["gamma"] * velocity


def analytical_position(gamma: jax.Array, times: jax.Array) -> jax.Array:
    """Reference solution for underdamping, q(0)=1 and p(0)=0.

    Valid for 0 < gamma < 2 * sqrt(m * k). The example stays in this
    regime; we do not clip or otherwise alter the model.
    """
    beta = gamma / (2.0 * MASS)
    omega_d = jnp.sqrt(SPRING_CONSTANT / MASS - beta**2)
    return INITIAL_POSITION * jnp.exp(-beta * times) * (
        jnp.cos(omega_d * times) + beta / omega_d * jnp.sin(omega_d * times)
    )


def generate_observations(times: jax.Array, *, seed: int = NOISE_SEED) -> jax.Array:
    """Add independent Gaussian position errors with known standard deviation."""
    truth = analytical_position(jnp.asarray(TRUE_DAMPING), times)
    noise = jax.random.normal(jax.random.PRNGKey(seed), shape=times.shape)
    return truth + OBSERVATION_SIGMA * noise


def simulated_position(gamma: jax.Array, times: jax.Array) -> jax.Array:
    """Integrate a candidate physical parameter through the public CPax API."""
    system = ForcedNewtonianSystem(
        mass=MASS,
        potential=potential,
        applied_force=applied_force,
        params={"k": jnp.asarray(SPRING_CONSTANT), "gamma": gamma},
    )
    problem = cpax.IVP(
        system,
        PhaseSpaceState(q=INITIAL_POSITION, p=0.0),
        (0.0, END_TIME),
    )
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.at(times),
        rtol=RTOL,
        atol=ATOL,
    )
    # By default cpax.solve raises on numerical failure. Only numerical
    # position arrays enter the differentiated objective (not metadata).
    return solution.ys.q


def negative_log_likelihood(gamma: jax.Array, times: jax.Array, observations: jax.Array) -> jax.Array:
    """Gaussian negative log-likelihood, up to parameter-independent terms.

    Assumes independent, homoscedastic position errors with *known*
    OBSERVATION_SIGMA. It does not model systematic or correlated errors.
    """
    residuals = (simulated_position(gamma, times) - observations) / OBSERVATION_SIGMA
    return 0.5 * jnp.sum(residuals**2)


def mean_nll_log_damping(log_gamma: jax.Array, times: jax.Array, observations: jax.Array) -> jax.Array:
    """Mean NLL for convenient gradient-descent scaling; gamma = exp(log_gamma)."""
    return negative_log_likelihood(jnp.exp(log_gamma), times, observations) / times.size


_value_and_grad = jax.jit(jax.value_and_grad(mean_nll_log_damping))


def fit_damping(
    times: jax.Array,
    observations: jax.Array,
    *,
    initial_gamma: float = 1.0,
    learning_rate: float = 0.025,
    iterations: int = 30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit gamma and return (estimate, approximate standard error, loss history).

    Gradient descent is intentionally tuned for this one-dimensional example,
    not presented as a general-purpose optimiser. Optimising log(gamma)
    enforces positivity without clipping the physical damping parameter.

    Uncertainty is the inverse square root of the curvature of the *full*
    NLL with respect to gamma at the fitted minimum. This local Gaussian
    approximation requires an identifiable interior optimum; it is not a
    general confidence-interval construction.
    """
    log_gamma = jnp.log(jnp.asarray(initial_gamma))
    history = []
    for _ in range(iterations):
        value, gradient = _value_and_grad(log_gamma, times, observations)
        history.append(value)
        log_gamma = log_gamma - learning_rate * gradient

    gamma_hat = jnp.exp(log_gamma)
    history.append(mean_nll_log_damping(log_gamma, times, observations))
    curvature = jax.hessian(negative_log_likelihood)(gamma_hat, times, observations)
    if not bool(jnp.isfinite(curvature) & (curvature > 0)):
        raise RuntimeError("Cannot estimate damping uncertainty: NLL curvature is not finite and positive.")
    standard_error = 1.0 / jnp.sqrt(curvature)
    return gamma_hat, standard_error, jnp.stack(history)


def main() -> None:
    """Perform reproducible recovery and visualise fit and likelihood profile."""
    times = jnp.linspace(0.0, END_TIME, NUM_OBSERVATIONS)
    observations = generate_observations(times)
    gamma_hat, standard_error, loss_history = fit_damping(times, observations)

    # Report optimisation convergence and the ground-truth comparison.
    print(f"True damping:                {TRUE_DAMPING:.6f}")
    print(f"Known position noise sigma:  {OBSERVATION_SIGMA:.6f}")
    print(f"Estimated damping:           {float(gamma_hat):.6f} +/- {float(standard_error):.6f} (approx. 1 SE)")
    print(f"Fit minus truth:             {float(gamma_hat - TRUE_DAMPING):+.6f}")
    print(f"Initial mean NLL:            {float(loss_history[0]):.6f}")
    print(f"Final mean NLL:              {float(loss_history[-1]):.6f}")
    print(f"Final absolute log-gradient: {float(jnp.abs(jax.grad(mean_nll_log_damping)(jnp.log(gamma_hat), times, observations))):.3e}")

    true_position = analytical_position(jnp.asarray(TRUE_DAMPING), times)
    fitted_position = simulated_position(gamma_hat, times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(times, observations, s=12, alpha=0.55, label="Noisy observations")
    axes[0].plot(times, true_position, label="Analytical ground truth")
    axes[0].plot(times, fitted_position, "--", label="CPax fitted trajectory")
    axes[0].set(xlabel="Time", ylabel="Position", title="Damping-parameter recovery")
    axes[0].legend()

    # The profile is obtained by actually re-solving CPax at each candidate
    # damping value; the shaded +/- 1 SE range comes from the *local* Hessian.
    profile_gamma = jnp.linspace(gamma_hat - 5.0 * standard_error, gamma_hat + 5.0 * standard_error, 51)
    profile_nll = jax.vmap(negative_log_likelihood, in_axes=(0, None, None))(
        profile_gamma, times, observations
    )
    axes[1].plot(profile_gamma, profile_nll - jnp.min(profile_nll), label="Numerical NLL profile")
    axes[1].axvline(float(gamma_hat), linestyle="--", label="Fitted damping")
    axes[1].axvline(TRUE_DAMPING, linestyle=":", label="True damping")
    axes[1].axvspan(
        float(gamma_hat - standard_error),
        float(gamma_hat + standard_error),
        alpha=0.15,
        label="Local Hessian +/- 1 SE",
    )
    axes[1].set(xlabel="Damping gamma", ylabel="NLL - min(NLL)", title="Likelihood curvature")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
