"""Public IVP numerical integration API."""

from typing import Literal

import diffrax
import jax

from cpax._backends._diffrax import solve_diffrax
from cpax.mechanics import NewtonianSystem, PhaseSpaceState
from cpax.problems import IVP
from cpax.results import Solution

from ._auto import Auto
from ._configuration import configure_solver
from ._saveat import SaveAt
from ._velocity_verlet import VelocityVerlet


def solve(
    problem: IVP,
    *,
    solver: diffrax.AbstractSolver | Auto | Literal["auto"] = "auto",
    saveat: SaveAt | None = None,
    dt0: float | jax.Array | None = None,
    rtol: float | jax.Array | None = None,
    atol: float | jax.Array | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    max_steps: int = 4096,
    throw: bool = True,
) -> Solution:
    """Integrate an IVP with an explicit Diffrax solver or the ``'auto'`` default.

    Automatic selection currently uses Tsit5 for all supported IVPs. It
    does not infer stiffness, geometric structure, or accuracy requirements.

    VelocityVerlet uses fixed steps. Provide dt0 and optionally
    ConstantStepSize(); adaptive controllers and rtol/atol are incompatible.
    It rejects approximate force calculations unless explicitly allowed.

    Without a custom controller, a Diffrax PID controller is constructed
    using relative tolerance 1e-5 and absolute tolerance 1e-7 by default.
    For fixed stepping, explicitly provide ``ConstantStepSize()`` and
    a positive ``dt0``. Saving does not determine integration step sizes.

    On integration failure, ``throw=True`` raises a Diffrax exception.
    With ``throw=False``, the returned Solution contains a FAILED status,
    the backend result code, step statistics and possibly padded samples.
    """
    if not isinstance(problem, IVP):
        raise TypeError("solve currently supports only cpax.IVP problems.")
    if saveat is not None and not isinstance(saveat, SaveAt):
        raise TypeError("saveat must be a cpax.SaveAt policy or None.")
    resolved_solver, solver_selection, controller = configure_solver(
        solver=solver,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )

    if isinstance(resolved_solver, VelocityVerlet):
        system = problem.system
        if not isinstance(system, NewtonianSystem) or not isinstance(problem.y0, PhaseSpaceState):
            raise TypeError("VelocityVerlet currently supports NewtonianSystem with PhaseSpaceState only.")
        if not system.force_calculation.exact_gradient and not resolved_solver.allow_approximate_force:
            raise ValueError(
                f"{type(system.force_calculation).__name__} only approximates -grad(V), so VelocityVerlet "
                "is not symplectic for it. Pass VelocityVerlet(allow_approximate_force=True) to integrate it "
                "as a reversible, volume-preserving scheme."
            )

    return solve_diffrax(
        problem,
        solver=resolved_solver,
        solver_selection=solver_selection,
        saveat=SaveAt.final() if saveat is None else saveat,
        dt0=dt0,
        stepsize_controller=controller,
        max_steps=int(max_steps),
        throw=throw,
    )
