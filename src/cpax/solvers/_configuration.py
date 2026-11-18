"""Shared validation and Diffrax configuration for CPax integrations."""

from numbers import Integral
from typing import Literal

import diffrax
import jax
import numpy as np
from jax.core import Tracer

from ._auto import Auto
from ._velocity_verlet import VelocityVerlet


def positive_real(name: str, value: float | jax.Array) -> None:
    """Validate a positive finite real scalar when it is concrete."""
    try:
        array = np.asarray(value) if not isinstance(value, Tracer) else None
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite real scalar.") from exc
    if array is not None and (
        array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array) or array <= 0
    ):
        raise ValueError(f"{name} must be a positive finite real scalar.")


def resolve_solver(
    solver: diffrax.AbstractSolver | Auto | Literal["auto"],
) -> tuple[diffrax.AbstractSolver, Literal["auto", "explicit"]]:
    """Select CPax's conservative default solver."""
    if isinstance(solver, str):
        if solver != "auto":
            raise ValueError("Unknown solver name; use 'auto' or a native Diffrax solver.")
        return diffrax.Tsit5(), "auto"
    if isinstance(solver, Auto):
        return diffrax.Tsit5(), "auto"
    if isinstance(solver, diffrax.AbstractSolver):
        return solver, "explicit"
    raise TypeError("solver must be 'auto', cpax.solvers.Auto(), or a native diffrax.AbstractSolver.")


def configure_solver(
    *,
    solver: diffrax.AbstractSolver | Auto | Literal["auto"],
    dt0: float | jax.Array | None,
    rtol: float | jax.Array | None,
    atol: float | jax.Array | None,
    stepsize_controller: diffrax.AbstractStepSizeController | None,
    max_steps: int,
    throw: bool,
) -> tuple[diffrax.AbstractSolver, Literal["auto", "explicit"], diffrax.AbstractStepSizeController]:
    """Validate CPax's common solve options and construct a controller."""
    if not isinstance(max_steps, Integral) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    if not isinstance(throw, bool):
        raise TypeError("throw must be a Boolean.")
    for name, value in (("dt0", dt0), ("rtol", rtol), ("atol", atol)):
        if value is not None:
            positive_real(name, value)

    resolved_solver, solver_selection = resolve_solver(solver)
    if isinstance(resolved_solver, VelocityVerlet):
        if dt0 is None:
            raise ValueError("VelocityVerlet requires an explicit positive dt0 for fixed stepping.")
        if rtol is not None or atol is not None:
            raise ValueError("VelocityVerlet requires fixed stepping; rtol and atol are not supported.")
        if stepsize_controller is not None and not isinstance(stepsize_controller, diffrax.ConstantStepSize):
            raise ValueError("VelocityVerlet requires diffrax.ConstantStepSize(), not an adaptive controller.")
        controller = diffrax.ConstantStepSize() if stepsize_controller is None else stepsize_controller
    elif stepsize_controller is not None:
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

    return resolved_solver, solver_selection, controller
