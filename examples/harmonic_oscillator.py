"""Harmonic oscillator: the minimal physics-first CPax workflow."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState


def main() -> None:
    system = NewtonianSystem(potential=lambda q: 0.5 * q**2, mass=1.0)
    problem = cpax.IVP(system, PhaseSpaceState(q=1.0, p=0.0), (0.0, 20.0))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(500), rtol=1e-7, atol=1e-9)
    assert bool(solution.is_success)
    fig, ax = plt.subplots()
    ax.plot(solution.ts, solution.ys.q, label="CPax")
    ax.plot(solution.ts, jnp.cos(solution.ts), "--", label="Analytical")
    ax.set(xlabel="Time", ylabel="Position", title="Harmonic oscillator")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
