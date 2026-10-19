"""Sun, Earth, Mars, Jupiter and an inbound long-period comet.

Units: AU, solar masses and years. The comet is a low-mass test particle on
an orbit with a 20 AU semi-major axis and a 2 AU perihelion; it starts inbound
at 8 AU, outside Jupiter's orbit. The adaptive default is not symplectic; use
a geometric integrator for studies of long-time Hamiltonian structure.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState, gravitational_potential

jax.config.update("jax_enable_x64", True)


def simulate():
    """Return the physical model and numerical solution."""
    sun_mass = 1.0
    earth_mass = 3.003e-6
    mars_mass = 3.227e-7
    jupiter_mass = 9.546e-4
    comet_mass = 1e-10
    masses = jnp.array([sun_mass, earth_mass, mars_mass, jupiter_mass, comet_mass])
    gravitational_constant = 4.0 * jnp.pi**2

    earth_orbit_radius = 1.0
    mars_orbit_radius = 1.524
    jupiter_orbit_radius = 5.204
    comet_semi_major_axis = 40.0
    comet_perihelion = 1.5
    comet_start_radius = 15
    comet_eccentricity = 1.0 - comet_perihelion / comet_semi_major_axis
    comet_specific_angular_momentum = jnp.sqrt(
        gravitational_constant * comet_semi_major_axis * (1.0 - comet_eccentricity**2)
    )
    comet_tangential_speed = comet_specific_angular_momentum / comet_start_radius
    comet_speed = jnp.sqrt(gravitational_constant * (2.0 / comet_start_radius - 1.0 / comet_semi_major_axis))
    comet_inbound_speed = -jnp.sqrt(comet_speed**2 - comet_tangential_speed**2)

    # Place the planets at distinct orbital phases. Their circular speeds are
    # heliocentric; the common offset below puts the full system at rest.
    relative_positions = jnp.array(
        [
            [0.0, 0.0],
            [earth_orbit_radius, 0.0],
            [-mars_orbit_radius, 0.0],
            [0.0, -jupiter_orbit_radius],
            [comet_start_radius, 0.0],
        ]
    )
    relative_velocities = jnp.array(
        [
            [0.0, 0.0],
            [0.0, jnp.sqrt(gravitational_constant / earth_orbit_radius)],
            [0.0, -jnp.sqrt(gravitational_constant / mars_orbit_radius)],
            [jnp.sqrt(gravitational_constant / jupiter_orbit_radius), 0.0],
            [comet_inbound_speed, comet_tangential_speed],
        ]
    )
    center_of_mass = jnp.sum(masses[:, None] * relative_positions, axis=0) / jnp.sum(masses)
    center_of_mass_velocity = jnp.sum(masses[:, None] * relative_velocities, axis=0) / jnp.sum(masses)
    positions = relative_positions - center_of_mass
    velocities = relative_velocities - center_of_mass_velocity
    system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
    problem = cpax.IVP(system, PhaseSpaceState(q=positions, p=masses[:, None] * velocities), (0.0, 10.0))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(751), rtol=1e-8, atol=1e-10, max_steps=100000)
    if not bool(solution.is_success):
        raise RuntimeError("Planetary simulation failed.")
    return system, solution


def diagnostics(system, solution) -> tuple[np.ndarray, np.ndarray]:
    """Return time series of total energy and planar angular momentum."""
    energies = jax.vmap(system.energy)(solution.ys)
    q, p = solution.ys.q, solution.ys.p
    angular_momentum = jnp.sum(q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0], axis=1)
    return np.asarray(energies), np.asarray(angular_momentum)


def animate(solution, output: str | Path = "nbody_simulation.gif", *, fps: int = 20) -> None:
    """Render the computed trajectory to a GIF; never called by Sphinx."""
    body_styles = (
        ("Sun", "gold"),
        ("Earth", "royalblue"),
        ("Mars", "orangered"),
        ("Jupiter", "burlywood"),
        ("Comet", "slategray"),
    )
    positions = np.asarray(solution.ys.q)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(
        xlim=(-10, 10),
        ylim=(-10, 10),
        xlabel="x [AU]",
        ylabel="y [AU]",
    )
    ax.grid(True)
    ax.set_aspect("equal")
    points = []
    trails = []
    for label, color in body_styles:
        (point,) = ax.plot([], [], "o", color=color, label=label, markersize=6)
        (trail,) = ax.plot([], [], "-", color=color, alpha=0.5, linewidth=1)
        points.append(point)
        trails.append(trail)
    ax.legend(loc="upper right")

    def update(frame: int):
        for i, (point, trail) in enumerate(zip(points, trails, strict=True)):
            point.set_data([positions[frame, i, 0]], [positions[frame, i, 1]])
            trail.set_data(positions[: frame + 1, i, 0], positions[: frame + 1, i, 1])
        return [*points, *trails]

    frames = range(0, len(positions), 4)
    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    animation.save(str(output), writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)


def main() -> None:
    system, solution = simulate()
    energy, angular_momentum = diagnostics(system, solution)
    print("Max energy deviation:", float(np.max(np.abs(energy - energy[0]))))
    print("Max angular-momentum deviation:", float(np.max(np.abs(angular_momentum - angular_momentum[0]))))
    animate(solution)


if __name__ == "__main__":
    main()
