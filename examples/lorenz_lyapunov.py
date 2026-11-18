"""Estimate and inspect the Lyapunov spectrum of the Lorenz system."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.analysis import lyapunov_spectrum
from cpax.dynamics import LorenzSystem


def main() -> None:
    """Estimate the standard Lorenz spectrum and plot its convergence history."""
    # Enable x64 before constructing model arrays. Long tangent integrations
    # are notably more sensitive to accumulated round-off.
    jax.config.update("jax_enable_x64", True)
    problem = cpax.IVP(LorenzSystem(), jnp.array([1.0, 1.0, 1.0]), (0.0, 2000.0))
    result = lyapunov_spectrum(
        problem,
        transient_time=100.0,
        num_segments=1900,
        rtol=1e-8,
        atol=1e-10,
        max_steps=20000,
    )
    assert bool(result.is_success)
    print(result)

    print("Lyapunov spectrum:", result.exponents)
    fig, ax = plt.subplots()
    for index in range(result.exponents.size):
        ax.plot(result.times, result.running_exponents[:, index], label=rf"$\lambda_{index + 1}$")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(xlabel="Time", ylabel="Estimated Lyapunov exponent", title="Lorenz-spectrum convergence")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
