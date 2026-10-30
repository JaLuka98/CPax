"""Nonlinear simple pendulum and its small-angle harmonic approximation.

A point mass on a massless rigid rod uses the angular coordinate theta and
canonical angular momentum p_theta = m * length**2 * angular_velocity.
All parameters are in SI units. Velocity Verlet takes fixed time steps.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState
from cpax.solvers import VelocityVerlet


def potential(theta: jax.Array, params: dict[str, float]) -> jax.Array:
    """Gravitational potential with zero energy at the stable equilibrium."""
    return params["mass"] * params["gravity"] * params["length"] * (1.0 - jnp.cos(theta))


def simulate(
    theta0: float,
    angular_velocity0: float = 0.0,
    *,
    mass: float = 1.0,
    length: float = 1.0,
    gravity: float = 9.81,
    t_end: float = 8.0,
    steps: int = 1600,
):
    """Return the conservative system and its fixed-step solution.

    Save times are requested on the nominal integration grid; numerical
    interpolation is not, in general, itself a symplectic operation.
    """
    params = {"mass": mass, "length": length, "gravity": gravity}
    inertia = mass * length**2
    system = NewtonianSystem(potential=potential, mass=inertia, params=params)
    initial = PhaseSpaceState(q=theta0, p=inertia * angular_velocity0)
    problem = cpax.IVP(system, initial, (0.0, t_end))
    solution = cpax.solve(
        problem,
        solver=VelocityVerlet(),
        dt0=t_end / steps,
        saveat=cpax.SaveAt.linspace(steps + 1),
        max_steps=steps + 16,
    )
    return system, solution


def main() -> None:
    """Compare linear and nonlinear motion and inspect numerical energy."""
    small_angle, large_angle = 0.05, 1.0
    _small_system, small = simulate(small_angle)
    large_system, large = simulate(large_angle)
    if not bool(small.is_success) or not bool(large.is_success):
        raise RuntimeError("Pendulum integration failed.")

    omega0 = jnp.sqrt(9.81)
    energy = jax.vmap(large_system.energy)(large.ys)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(small.ts, small.ys.q, label="Nonlinear, initial angle 0.05 rad")
    axes[0].plot(small.ts, small_angle * jnp.cos(omega0 * small.ts), "--", label="Small-angle approximation")
    axes[0].plot(large.ts, large.ys.q, label="Nonlinear, initial angle 1 rad")
    axes[0].set(ylabel="Angle (rad)", title="Simple pendulum")
    axes[0].legend()
    axes[1].plot(large.ts, energy - energy[0])
    axes[1].set(xlabel="Time (s)", ylabel="Energy deviation (J)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
