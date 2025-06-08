"""
N-body gravitational simulation demo using Leapfrog integration.

This example simulates the Sun–Earth–Mars system using canonical Hamiltonian
dynamics with Newtonian gravity. Units are:
- Distance: AU
- Time: years
- Mass: solar masses

The example uses the CPAX modular `hamiltonian_nd` model and Leapfrog integrator.

Author: Jan Lukas Späh
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from cpax.ode.integrators import simulate_leapfrog_scan
from cpax.ode.models import hamiltonian_nd


def main():
    # --- Physical parameters ---
    masses = jnp.array([1.0, 3e-6, 3.3e-7])  # Sun, Earth, Mars

    # --- Initial conditions ---
    q0 = jnp.array([
        [0.0, 0.0],    # Sun
        [1.0, 0.0],    # Earth
        [1.5, 0.0],    # Mars
    ])
    vels = jnp.array([
        [0.0,        0.0],
        [0.0,  2 * jnp.pi],
        [0.0,  1.6 * jnp.pi],
    ])

    # --- Add comet ---
    masses = jnp.append(masses, 1e-10)
    q0 = jnp.vstack([q0, jnp.array([[10.0, 0.0]])])
    v_comet = jnp.array([0.0, 0.2 * jnp.sqrt(4 * jnp.pi**2 / 5.0)])  # < circular speed
    vels = jnp.vstack([vels, v_comet[None, :]])
    p0 = masses[:, None] * vels

    # --- Time parameters ---
    t0 = 0.0
    t_end = 20.0
    dt = 0.001
    n_steps = int(t_end / dt)

    # --- Hamiltonian dynamics ---
    f = hamiltonian_nd(masses, potential="gravity")

    # --- Integrate ---
    ts, qs, ps = simulate_leapfrog_scan(
        q0=q0,
        p0=p0,
        t0=t0,
        dt=dt,
        n_steps=n_steps,
        f=f,
        masses=masses
    )

    # --- Plot ---
    plot_trajectories(qs, q0)



def plot_trajectories(qs, q0):
    plt.figure(figsize=(6, 6))

    # Initial positions
    plt.plot(q0[0, 0], q0[0, 1], marker='o', color='gold', label='Sun init')
    plt.plot(q0[1, 0], q0[1, 1], marker='o', color='blue', label='Earth init')
    plt.plot(q0[2, 0], q0[2, 1], marker='o', color='orangered', label='Mars init')
    plt.plot(q0[3, 0], q0[3, 1], marker='o', color='gray', label='Comet init')

    # Trajectories
    plt.plot(qs[:, 0, 0], qs[:, 0, 1], color='gold', label='Sun')
    plt.plot(qs[:, 1, 0], qs[:, 1, 1], color='blue', label='Earth')
    plt.plot(qs[:, 2, 0], qs[:, 2, 1], color='orangered', label='Mars')
    plt.plot(qs[:, 3, 0], qs[:, 3, 1], color='gray', label='Comet')

    plt.xlabel("X position [AU]")
    plt.ylabel("Y position [AU]")
    plt.title("Trajectories of Sun, Earth, & Mars")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("celestial_trajectories.png")
    plt.show()


if __name__ == "__main__":
    main()
