"""Public IVP numerical integration API."""

from numbers import Integral
from typing import Any

import diffrax
import jax
import numpy as np
from jax.core import Tracer

from cpax._backends._diffrax import solve_diffrax
from cpax.problems import IVP
from cpax.results import Solution

from ._saveat import SaveAt


def _positive_real(name: str, value: float | jax.Array) -> None:
    """Validate numerical configuration when it is concrete."""
    try:
        array = np.asarray(value) if not isinstance(value, Tracer) else None
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite real scalar.") from exc
    if array is not None and (
        array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array) or array <= 0
    ):
        raise ValueError(f"{name} must be a positive finite real scalar.")


def solve(
    problem: IVP,
    *,
    solver: diffrax.AbstractSolver,
    saveat: SaveAt | None = None,
    dt0: float | jax.Array | None = None,
    rtol: float | jax.Array | None = None,
    atol: float | jax.Array | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    max_steps: int = 4096,
    throw: bool = True,
) -> Solution:
    """Integrate a continuous-time IVP using a native Diffrax solver.

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
    if not isinstance(solver, diffrax.AbstractSolver):
        raise TypeError("solver must be a native diffrax.AbstractSolver.")
    if saveat is not None and not isinstance(saveat, SaveAt):
        raise TypeError("saveat must be a cpax.SaveAt policy or None.")
    if not isinstance(max_steps, Integral) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    if not isinstance(throw, bool):
        raise TypeError("throw must be a Boolean.")
    for name, value in (("dt0", dt0), ("rtol", rtol), ("atol", atol)):
        if value is not None:
            _positive_real(name, value)

    if stepsize_controller is not None:
        if not isinstance(stepsize_controller, diffrax.AbstractStepSizeController):
            raise TypeError("stepsize_controller must be a Diffrax step-size controller.")
        if rtol is not None or atol is not None:
            raise ValueError("Specify either stepsize_controller or rtol/atol, not both.")
        controller = stepsize_controller
    else:
        controller = diffrax.PIDController(
            rtol=1e-5 if rtol is None else rtol,
            atol=1e-7 if atol is None else atol,
        )

    if isinstance(controller, diffrax.ConstantStepSize) and dt0 is None:
        raise ValueError("Fixed-step integration requires an explicit positive dt0.")

    return solve_diffrax(
        problem,
        solver=solver,
        saveat=SaveAt.final() if saveat is None else saveat,
        dt0=dt0,
        stepsize_controller=controller,
        max_steps=int(max_steps),
        throw=throw,
    )
