"""Plot a circular binary orbit integrated by CPax Velocity Verlet."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState, gravitational_potential
from cpax.solvers import VelocityVerlet


def simulate():
    """Return the Newtonian model and its fixed-step symplectic solution."""
    masses = jnp.array([1.0, 0.5])
    total_mass = jnp.sum(masses)
    separation = 1.0
    angular_speed = 2.0 * jnp.pi * jnp.sqrt(total_mass / separation**3)
    q0 = jnp.array([[-masses[1] / total_mass, 0.0], [masses[0] / total_mass, 0.0]])
    velocities0 = jnp.array(
        [[0.0, -angular_speed * masses[1] / total_mass], [0.0, angular_speed * masses[0] / total_mass]]
    )
    p0 = masses[:, None] * velocities0

    system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
    problem = cpax.IVP(system, PhaseSpaceState(q0, p0), (0.0, 2.0))
    solution = cpax.solve(
        problem,
        solver=VelocityVerlet(),
        dt0=0.002,
        saveat=cpax.SaveAt.linspace(1001),
        max_steps=2000,
    )
    if not bool(solution.is_success):
        raise RuntimeError("Binary orbit integration failed.")
    return system, solution


def main() -> None:
    system, solution = simulate()
    energies = jax.vmap(system.energy)(solution.ys)
    relative_energy_error = (energies - energies[0]) / jnp.abs(energies[0])
    print(f"Maximum relative energy deviation: {jnp.max(jnp.abs(relative_energy_error)):.3e}")

    qs = solution.ys.q
    for i in range(qs.shape[1]):
        plt.plot(qs[:, i, 0], qs[:, i, 1], label=f"body {i + 1}")
    plt.gca().set_aspect("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
