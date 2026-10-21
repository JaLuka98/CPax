"""Plot a circular binary orbit with the retained symplectic leapfrog integrator."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cpax.mechanics import gravitational_potential
from cpax.ode import simulate_leapfrog_scan


def main() -> None:
    masses = jnp.array([1.0, 0.5])
    total_mass = jnp.sum(masses)
    separation = 1.0
    angular_speed = 2.0 * jnp.pi * jnp.sqrt(total_mass / separation**3)
    q0 = jnp.array([[-masses[1] / total_mass, 0.0], [masses[0] / total_mass, 0.0]])
    velocities0 = jnp.array(
        [[0.0, -angular_speed * masses[1] / total_mass], [0.0, angular_speed * masses[0] / total_mass]]
    )
    p0 = masses[:, None] * velocities0
    potential = gravitational_potential(masses)
    from jax import grad

    force = grad(potential)

    def dynamics(q, p, t):
        del t
        return p / masses[:, None], -force(q)

    _, qs, ps = simulate_leapfrog_scan(q0, p0, 0.0, 0.002, 1000, dynamics)

    def kinetic(p):
        return 0.5 * jnp.sum(p**2 / masses[:, None])

    first_energy = potential(q0) + kinetic(p0)
    last_energy = potential(qs[-1]) + kinetic(ps[-1])
    print(f"Relative energy drift: {(last_energy - first_energy) / first_energy:.3e}")
    for i in range(2):
        plt.plot(qs[:, i, 0], qs[:, i, 1], label=f"body {i + 1}")
    plt.gca().set_aspect("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
