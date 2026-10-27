"""Driven, damped nonlinear pendulum with generalised torque."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState


def potential(theta, params):
    """Gravitational potential in joules, for theta measured in radians."""
    return params["mass"] * params["g"] * params["length"] * (1.0 - jnp.cos(theta))


def applied_torque(t, theta, angular_velocity, params):
    """Viscous torque plus a periodic external driving torque, in N m."""
    del theta
    return -params["damping"] * angular_velocity + params["drive"] * jnp.cos(params["frequency"] * t)


def main() -> None:
    params = {
        "mass": 1.0,  # kg
        "length": 1.0,  # m
        "g": 9.81,  # m/s^2
        "damping": 0.35,  # kg m^2/s
        "drive": 1.2,  # N m
        "frequency": 2.0,  # rad/s
    }
    inertia = params["mass"] * params["length"] ** 2
    model = ForcedNewtonianSystem(
        mass=inertia,
        potential=potential,
        applied_force=applied_torque,
        params=params,
    )
    # Angular momentum p = I * angular velocity (not angular velocity itself).
    y0 = PhaseSpaceState(q=0.5, p=0.0)
    problem = cpax.IVP(model, y0, (0.0, 20.0))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(1001), rtol=1e-7, atol=1e-9)
    assert bool(solution.is_success)

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
