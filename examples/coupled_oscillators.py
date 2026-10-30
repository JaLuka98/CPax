"""Normal modes of two identical coupled harmonic oscillators.

Two equal masses m are attached to walls by springs k and coupled by a
third spring k_c. Coordinates are displacements from equilibrium; all
parameters are in consistent SI units.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState
from cpax.solvers import VelocityVerlet


def potential(q: jax.Array, params: dict[str, float]) -> jax.Array:
    """Quadratic potential with a coupling between the two displacements."""
    return 0.5 * params["spring"] * jnp.sum(q**2) + 0.5 * params["coupling"] * (q[0] - q[1]) ** 2


def simulate(
    q0=(1.0, 0.0),
    v0=(0.0, 0.0),
    *,
    mass: float = 1.0,
    spring: float = 1.0,
    coupling: float = 0.6,
    t_end: float = 20.0,
    steps: int = 2000,
):
    """Return the model and the symplectic fixed-step trajectory."""
    q = jnp.asarray(q0, dtype=jnp.result_type(float))
    v = jnp.asarray(v0, dtype=q.dtype)
    system = NewtonianSystem(
        mass=mass,
        potential=potential,
        params={"spring": spring, "coupling": coupling},
    )
    problem = cpax.IVP(system, PhaseSpaceState(q, mass * v), (0.0, t_end))
    solution = cpax.solve(
        problem,
        solver=VelocityVerlet(),
        dt0=t_end / steps,
        saveat=cpax.SaveAt.linspace(steps + 1),
        max_steps=steps + 16,
    )
    return system, solution


def main() -> None:
    """Plot the two analytically known modes and their superposition."""
    mass, spring, coupling = 1.0, 1.0, 0.6
    omega_in_phase = jnp.sqrt(spring / mass)
    omega_out_of_phase = jnp.sqrt((spring + 2 * coupling) / mass)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for ax, q0, frequency, title in (
        (axes[0], (1.0, 1.0), omega_in_phase, "In-phase mode"),
        (axes[1], (1.0, -1.0), omega_out_of_phase, "Out-of-phase mode"),
    ):
        _, sol = simulate(q0=q0, mass=mass, spring=spring, coupling=coupling)
        if not bool(sol.is_success):
            raise RuntimeError("Coupled oscillator integration failed.")
        ax.plot(sol.ts, sol.ys.q[:, 0], label="First mass (CPax)")
        ax.plot(sol.ts, sol.ys.q[:, 1], label="Second mass (CPax)")
        ax.plot(sol.ts, jnp.cos(frequency * sol.ts), "--", label="Analytical first mass")
        ax.set(ylabel="Displacement (m)", title=title)
        ax.legend(loc="upper right")

    _, mixed = simulate(q0=(1.0, 0.0), mass=mass, spring=spring, coupling=coupling)
    if not bool(mixed.is_success):
        raise RuntimeError("Coupled oscillator integration failed.")
    axes[2].plot(mixed.ts, mixed.ys.q[:, 0], label="First mass")
    axes[2].plot(mixed.ts, mixed.ys.q[:, 1], label="Second mass")
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Displacement (m)",
        title="Superposition of the two modes",
    )
    axes[2].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
