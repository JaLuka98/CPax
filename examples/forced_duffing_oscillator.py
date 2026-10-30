"""Driven and damped Duffing oscillator in a double-well potential.

This complementary example retains the separate conservative Duffing example.
All constants are expressed in mutually consistent, dimensionless units.
Unlike a conservative Hamiltonian, mechanical energy here changes due to
viscous damping and external periodic driving.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState


def potential(q: jax.Array, params: dict[str, float]) -> jax.Array:
    """Double-well potential V = -alpha*q**2/2 + beta*q**4/4."""
    return -0.5 * params["alpha"] * q**2 + 0.25 * params["beta"] * q**4


def applied_force(t: jax.Array, q: jax.Array, velocity: jax.Array, params: dict[str, float]) -> jax.Array:
    """Generalised force consisting of viscous drag and periodic driving."""
    del q
    return -params["damping"] * velocity + params["drive"] * jnp.cos(params["frequency"] * t)


def simulate(
    *,
    mass: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    damping: float = 0.18,
    drive: float = 0.3,
    frequency: float = 1.2,
    q0: float = -1.0,
    velocity0: float = 0.0,
    t_end: float = 30.0,
    samples: int = 1001,
    rtol: float = 1e-5,
    atol: float = 1e-7,
):
    """Return the force-driven system and its adaptive IVP solution."""
    params = {
        "alpha": alpha,
        "beta": beta,
        "damping": damping,
        "drive": drive,
        "frequency": frequency,
    }
    system = ForcedNewtonianSystem(
        mass=mass,
        potential=potential,
        applied_force=applied_force,
        params=params,
    )
    problem = cpax.IVP(system, PhaseSpaceState(q=q0, p=mass * velocity0), (0.0, t_end))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(samples), rtol=rtol, atol=atol)
    return system, solution


def main() -> None:
    """Display displacement, phase-space motion and mechanical energy."""
    system, sol = simulate()
    if not bool(sol.is_success):
        raise RuntimeError("Driven Duffing integration failed.")

    mechanical_energy = jax.vmap(system.mechanical_energy)(sol.ys)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(sol.ts, sol.ys.q)
    axes[0].set(xlabel="Time", ylabel="Position", title="Driven, damped Duffing oscillator")
    axes[1].plot(sol.ys.q, sol.ys.p / system.mass)
    axes[1].set(xlabel="Position", ylabel="Velocity", title="Phase-space trajectory")
    axes[2].plot(sol.ts, mechanical_energy)
    axes[2].set(xlabel="Time", ylabel="Mechanical energy", title="Energy is not conserved")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
