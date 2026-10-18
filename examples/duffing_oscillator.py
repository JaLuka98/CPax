"""Conservative, unforced Duffing oscillator in a double-well potential."""

import cpax
import matplotlib.pyplot as plt

from cpax.mechanics import NewtonianSystem, PhaseSpaceState


def potential(q):
    """V(q) = -q²/2 + q⁴/4, with force F(q) = q - q³."""
    return -0.5 * q**2 + 0.25 * q**4


def main() -> None:
    system = NewtonianSystem(potential=potential, mass=1.0)
    fig, ax = plt.subplots()
    for q0, p0, label in [
        (-1.0, 0.2, "Oscillation in one well"),
        (-1.0, 0.9, "Motion across both wells"),
    ]:
        problem = cpax.IVP(system, PhaseSpaceState(q=q0, p=p0), (0.0, 30.0))
        sol = cpax.solve(problem, saveat=cpax.SaveAt.linspace(1000), rtol=1e-7, atol=1e-9)
        assert bool(sol.is_success)
        ax.plot(sol.ys.q, sol.ys.p, label=label)
    ax.set(xlabel="Position q", ylabel="Momentum p", title="Conservative Duffing oscillator")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
