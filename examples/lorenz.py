"""Lorenz attractor using the built-in CPax dynamical system."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import cpax
from cpax.dynamics import LorenzSystem


def main() -> None:
    problem = cpax.IVP(LorenzSystem(), jnp.array([1.0, 1.0, 1.0]), (0.0, 35.0))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(3501), rtol=1e-5, atol=1e-7, max_steps=20000)
    assert bool(solution.is_success)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y, z = solution.ys.T
    ax.plot(x, y, z, linewidth=0.7)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz attractor")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
