"""Plot a circular binary orbit with the leapfrog integrator."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cpax.ode import simulate_leapfrog_scan
from cpax.ode.models import hamiltonian_nd, potential_gravitational


def main():
    # Astronomical units: distance in AU, mass in solar masses, and time in years.
    # These comparable masses make both bodies' centre-of-mass orbits visible.
    masses = jnp.array([1.0, 0.5])
    total_mass = jnp.sum(masses)
    separation = 1.0
    angular_speed = 2.0 * jnp.pi * jnp.sqrt(total_mass / separation**3)

    q0 = jnp.array(
        [
            [-masses[1] * separation / total_mass, 0.0],
            [masses[0] * separation / total_mass, 0.0],
        ]
    )
    velocities0 = jnp.array(
        [
            [0.0, -angular_speed * masses[1] * separation / total_mass],
            [0.0, angular_speed * masses[0] * separation / total_mass],
        ]
    )
    p0 = masses[:, None] * velocities0

    potential = potential_gravitational(masses)
    dynamics = hamiltonian_nd(masses, potential="gravity")
    _, qs, ps = simulate_leapfrog_scan(q0, p0, 0.0, 0.002, 1_000, dynamics)

    initial_energy = potential(q0) + 0.5 * jnp.sum(p0**2 / masses[:, None])
    final_energy = potential(qs[-1]) + 0.5 * jnp.sum(ps[-1] ** 2 / masses[:, None])
    print(f"Relative energy drift: {(final_energy - initial_energy) / initial_energy:.3e}")

    plt.plot(qs[:, 0, 0], qs[:, 0, 1], label="body 1 (mass 1.0)")
    plt.plot(qs[:, 1, 0], qs[:, 1, 1], label="body 2 (mass 0.5)")
    plt.scatter(q0[:, 0], q0[:, 1], color=["C0", "C1"], marker="o", zorder=3)
    plt.scatter(0.0, 0.0, color="black", marker="+", label="centre of mass")
    plt.gca().set_aspect("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Circular binary orbit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
