"""Driven, damped nonlinear pendulum with generalised torque.

The coordinate theta is angular displacement in radians and the canonical
momentum is p_theta = I * angular_velocity, with I = mass * length**2.
All physical constants below are in SI units.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState


def potential(theta: jax.Array, params: dict[str, float]) -> jax.Array:
    """Gravitational potential in joules, for theta measured in radians."""
    return params["mass"] * params["g"] * params["length"] * (1.0 - jnp.cos(theta))


def applied_torque(
    t: jax.Array, theta: jax.Array, angular_velocity: jax.Array, params: dict[str, float]
) -> jax.Array:
    """Viscous torque plus a periodic external driving torque, in N m."""
    del theta
    return -params["damping"] * angular_velocity + params["drive"] * jnp.cos(params["frequency"] * t)


def simulate(
    *,
    mass: float = 1.0,
    length: float = 1.0,
    gravity: float = 9.81,
    damping: float = 0.35,
    drive: float = 1.2,
    frequency: float = 3.0,
    theta0: float = 0.5,
    angular_velocity0: float = 0.0,
    t_end: float = 50.0,
    samples: int = 1001,
):
    """Return the force-driven pendulum system and its adaptive solution."""
    params = {
        "mass": mass,  # kg
        "length": length,  # m
        "g": gravity,  # m/s**2
        "damping": damping,  # kg m**2/s
        "drive": drive,  # N m
        "frequency": frequency,  # rad/s
    }
    inertia = mass * length**2
    model = ForcedNewtonianSystem(
        mass=inertia,
        potential=potential,
        applied_force=applied_torque,
        params=params,
    )
    initial = PhaseSpaceState(q=theta0, p=inertia * angular_velocity0)
    problem = cpax.IVP(model, initial, (0.0, t_end))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(samples), rtol=1e-5, atol=1e-7)
    return model, solution


def main() -> None:
    model, solution = simulate()
    if not bool(solution.is_success):
        raise RuntimeError("Driven pendulum integration failed.")

    energy = jax.vmap(model.mechanical_energy)(solution.ys)
    power = jax.vmap(model.applied_power)(solution.ts, solution.ys)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(solution.ts, solution.ys.q)
    axes[0].set_ylabel("Angle (rad)")
    axes[1].plot(solution.ts, energy)
    axes[1].set_ylabel("Mechanical energy (J)")
    axes[2].plot(solution.ts, power)
    axes[2].set(xlabel="Time (s)", ylabel="Applied power (W)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
