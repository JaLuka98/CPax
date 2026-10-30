"""Conservative, unforced Duffing oscillator in a double-well potential."""

import matplotlib.pyplot as plt

import cpax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState


def potential(q):
    """V(q) = -q²/2 + q⁴/4, with force F(q) = q - q³."""
    return -0.5 * q**2 + 0.25 * q**4


def simulate(q0: float, p0: float, *, t1: float = 30.0, samples: int = 1000):
    """Integrate one trajectory in the conservative double-well potential."""
    system = NewtonianSystem(potential=potential, mass=1.0)
    problem = cpax.IVP(system, PhaseSpaceState(q=q0, p=p0), (0.0, t1))
    solution = cpax.solve(problem, saveat=cpax.SaveAt.linspace(samples), rtol=1e-5, atol=1e-7)
    return system, solution


def main() -> None:
    fig, ax = plt.subplots()
    for q0, p0, label in [
        (-1.0, 0.2, "Oscillation in one well"),
        (-1.0, 0.9, "Motion across both wells"),
    ]:
        _, sol = simulate(q0, p0)
        assert bool(sol.is_success)
        ax.plot(sol.ys.q, sol.ys.p, label=label)
    ax.set(xlabel="Position q", ylabel="Momentum p", title="Conservative Duffing oscillator")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
