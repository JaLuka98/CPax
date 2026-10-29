"""Infer viscous damping from noisy oscillator positions using differentiable CPax.

Run with `uv run python examples/damping_parameter_inference.py`.

This deliberately illustrates differentiation *through a numerical simulation*.
For this analytically solvable oscillator, fitting the closed-form trajectory
directly would normally be simpler and more accurate. The numerical approach
becomes useful for physical systems without tractable analytical trajectories.
"""

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState


@dataclass(frozen=True)
class InferenceConfig:
    """Physical, measurement and solver settings for this example.

    Units are consistent but arbitrary. Output sampling is independent of
    the solver tolerances; JAX's default precision is left unchanged.
    """

    mass: float = 1.0
    spring_constant: float = 4.0
    initial_position: float = 1.0
    true_damping: float = 0.55
    observation_sigma: float = 0.035
    end_time: float = 8.0
    num_observations: int = 81
    noise_seed: int = 2027
    rtol: float = 1e-6
    atol: float = 1e-8


def potential(q: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Autonomous spring potential V(q) = k q² / 2."""
    return 0.5 * params["k"] * q**2


def applied_force(t: jax.Array, q: jax.Array, velocity: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Viscous force F = -gamma * velocity."""
    del t, q
    return -params["gamma"] * velocity


def analytical_position(gamma: jax.Array, times: jax.Array, config: InferenceConfig) -> jax.Array:
    """Reference solution for underdamping, q(0)=config.initial_position and p(0)=0.

    Valid for 0 < gamma < 2 * sqrt(m * k). The example stays in this
    regime; we do not clip or otherwise alter the model.
    """
    beta = gamma / (2.0 * config.mass)
    omega_d = jnp.sqrt(config.spring_constant / config.mass - beta**2)
    return (
        config.initial_position
        * jnp.exp(-beta * times)
        * (jnp.cos(omega_d * times) + beta / omega_d * jnp.sin(omega_d * times))
    )


def generate_observations(times: jax.Array, config: InferenceConfig, *, seed: int | None = None) -> jax.Array:
    """Add independent Gaussian position errors with known standard deviation."""
    truth = analytical_position(jnp.asarray(config.true_damping), times, config)
    rng_seed = config.noise_seed if seed is None else seed
    noise = jax.random.normal(jax.random.PRNGKey(rng_seed), shape=times.shape)
    return truth + config.observation_sigma * noise


def simulated_position(gamma: jax.Array, times: jax.Array, config: InferenceConfig) -> jax.Array:
    """Integrate a candidate physical parameter through the public CPax API."""
    system = ForcedNewtonianSystem(
        mass=config.mass,
        potential=potential,
        applied_force=applied_force,
        params={"k": jnp.asarray(config.spring_constant), "gamma": gamma},
    )
    problem = cpax.IVP(
        system,
        PhaseSpaceState(q=config.initial_position, p=0.0),
        (0.0, config.end_time),
    )
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.at(times),
        rtol=config.rtol,
        atol=config.atol,
    )
    # By default cpax.solve raises on numerical failure. Only numerical
    # position arrays enter the differentiated objective (not metadata).
    return solution.ys.q


def negative_log_likelihood(
    gamma: jax.Array, times: jax.Array, observations: jax.Array, config: InferenceConfig
) -> jax.Array:
    """Gaussian negative log-likelihood, up to parameter-independent terms.

    Assumes independent, homoscedastic position errors with *known*
    config.observation_sigma. It does not model systematic or correlated errors.
    """
    residuals = (simulated_position(gamma, times, config) - observations) / config.observation_sigma
    return 0.5 * jnp.sum(residuals**2)


def mean_nll_log_damping(
    log_gamma: jax.Array, times: jax.Array, observations: jax.Array, config: InferenceConfig
) -> jax.Array:
    """Mean NLL for convenient gradient-descent scaling; gamma = exp(log_gamma)."""
    return negative_log_likelihood(jnp.exp(log_gamma), times, observations, config) / times.size


def fit_damping(
    times: jax.Array,
    observations: jax.Array,
    config: InferenceConfig,
    *,
    initial_gamma: float = 1.0,
    learning_rate: float = 0.025,
    iterations: int = 30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit gamma and return (estimate, approximate standard error, loss history).

    Gradient descent is intentionally tuned for this one-dimensional example,
    not presented as a general-purpose optimiser. Optimising log(gamma)
    enforces positivity without clipping the physical damping parameter.

    Uncertainty uses the inverse square root of the expected Fisher
    information (Gauss-Newton curvature) computed from first-order trajectory
    sensitivities at the fitted damping. It assumes independent Gaussian
    noise of known variance, an identifiable interior optimum, and a local
    Gaussian approximation; it is not a general confidence interval.
    """

    # Capture immutable settings outside JAX's traced arguments: a plain
    # dataclass does not need to be registered as a JAX PyTree.
    def objective(log_gamma: jax.Array) -> jax.Array:
        return mean_nll_log_damping(log_gamma, times, observations, config)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    log_gamma = jnp.log(jnp.asarray(initial_gamma))
    history = []
    for _ in range(iterations):
        value, gradient = value_and_grad(log_gamma)
        history.append(value)
        log_gamma = log_gamma - learning_rate * gradient

    gamma_hat = jnp.exp(log_gamma)
    history.append(objective(log_gamma))
    # For independent Gaussian errors of known variance, the expected
    # information (and Gauss-Newton curvature) is sum_i (dq_i/dgamma)^2 / sigma^2.
    # This only needs first-order reverse-mode sensitivity through Diffrax.
    # Differentiating the numerical solve twice is not generally supported
    # by its default checkpointing adjoint.
    sensitivity = jax.jacrev(lambda gamma: simulated_position(gamma, times, config))(gamma_hat)
    information = jnp.sum((sensitivity / config.observation_sigma) ** 2)
    if not bool(jnp.isfinite(information) & (information > 0)):
        raise RuntimeError("Cannot estimate damping uncertainty: Fisher information is not finite and positive.")
    standard_error = 1.0 / jnp.sqrt(information)
    return gamma_hat, standard_error, jnp.stack(history)


def main() -> None:
    """Perform reproducible recovery and visualise fit and likelihood profile."""
    config = InferenceConfig()
    times = jnp.linspace(0.0, config.end_time, config.num_observations)
    observations = generate_observations(times, config)
    gamma_hat, standard_error, loss_history = fit_damping(times, observations, config)

    # Report optimisation convergence and the ground-truth comparison.
    def objective(log_gamma: jax.Array) -> jax.Array:
        return mean_nll_log_damping(log_gamma, times, observations, config)

    print(f"True damping:                {config.true_damping:.6f}")
    print(f"Known position noise sigma:  {config.observation_sigma:.6f}")
    print(f"Estimated damping:           {float(gamma_hat):.6f} +/- {float(standard_error):.6f} (approx. 1 SE)")
    print(f"Fit minus truth:             {float(gamma_hat - config.true_damping):+.6f}")
    print(f"Initial mean NLL:            {float(loss_history[0]):.6f}")
    print(f"Final mean NLL:              {float(loss_history[-1]):.6f}")
    print(f"Final absolute log-gradient: {float(jnp.abs(jax.grad(objective)(jnp.log(gamma_hat)))):.3e}")

    true_position = analytical_position(jnp.asarray(config.true_damping), times, config)
    fitted_position = simulated_position(gamma_hat, times, config)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(times, observations, s=12, alpha=0.55, label="Noisy observations")
    axes[0].plot(times, true_position, label="Analytical ground truth")
    axes[0].plot(times, fitted_position, "--", label="CPax numerical trajectory")
    axes[0].set(xlabel="Time", ylabel="Position", title="Damping-parameter recovery")
    axes[0].grid()
    axes[0].legend()

    # The profile is obtained by actually re-solving CPax at each candidate
    # damping value; the shaded +/- 1 SE range comes from Fisher information.
    profile_gamma = jnp.linspace(gamma_hat - 5.0 * standard_error, gamma_hat + 5.0 * standard_error, 51)
    profile_nll = jax.vmap(lambda gamma: negative_log_likelihood(gamma, times, observations, config))(profile_gamma)
    nll_at_fit = negative_log_likelihood(gamma_hat, times, observations, config)
    axes[1].plot(profile_gamma, profile_nll - nll_at_fit, label="Numerical NLL profile")
    axes[1].axhline(0.5, color="0.5", linestyle=":", label=r"$\Delta\mathrm{NLL}=0.5$ (68% profile)")
    axes[1].axvline(float(gamma_hat), linestyle="--", label="Fitted damping")
    axes[1].axvline(config.true_damping, linestyle=":", label="True damping")
    axes[1].axvspan(
        float(gamma_hat - standard_error),
        float(gamma_hat + standard_error),
        alpha=0.15,
        label="Fisher +/- 1 SE",
    )
    axes[1].set(xlabel="Damping gamma", ylabel="NLL - NLL(fitted)", title="Likelihood curvature")
    axes[1].grid()
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
