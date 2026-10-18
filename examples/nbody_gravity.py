"""Sun, Earth, Mars and a comet using the physics-first CPax API.

Units: AU, solar masses and years. The comet is a low-mass test particle.
The adaptive default is not symplectic; use a geometric integrator for
studies of long-time Hamiltonian structure.
"""

from pathlib import Path

import cpax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from cpax.mechanics import NewtonianSystem, PhaseSpaceState, gravitational_potential

LABELS = ("Sun", "Earth", "Mars", "Comet")
COLORS = ("gold", "royalblue", "orangered", "slategray")


def simulate():
    """Return the physical model and numerical solution."""
    masses = jnp.array([1.0, 3e-6, 3.3e-7, 1e-10])
    gravitational_constant = 4.0 * jnp.pi**2
    positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [5.0, 0.0]])
    velocities = jnp.array(
        [[0.0, 0.0], [0.0, 2 * jnp.pi], [0.0, 1.6 * jnp.pi],
         [0.0, 0.7 * jnp.sqrt(gravitational_constant / 5.0)]]
    )
    system = NewtonianSystem(potential=gravitational_potential(masses), mass=masses)
    problem = cpax.IVP(
        system, PhaseSpaceState(q=positions, p=masses[:, None] * velocities), (0.0, 5.0)
    )
    solution = cpax.solve(
        problem, saveat=cpax.SaveAt.linspace(751), rtol=1e-8, atol=1e-10,
        max_steps=100000
    )
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
    positions = np.asarray(solution.ys.q)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel="x [AU]", ylabel="y [AU]",
           title="Sun, Earth, Mars and comet")
    ax.set_aspect("equal")
    points = []
    trails = []
    for label, color in zip(LABELS, COLORS, strict=True):
        point, = ax.plot([], [], "o", color=color, label=label, markersize=6)
        trail, = ax.plot([], [], "-", color=color, alpha=0.5, linewidth=1)
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
    print("Max angular-momentum deviation:",
          float(np.max(np.abs(angular_momentum - angular_momentum[0]))))
    animate(solution)


if __name__ == "__main__":
    main()
