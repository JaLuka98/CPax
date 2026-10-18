"""Private Diffrax adapter for continuous-time initial-value problems."""

from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from cpax.dynamics._systems import Time
from cpax.problems import IVP
from cpax.results import Solution, SolutionStatus


def _resolve_saveat(policy: Any, t_span: tuple[Time, Time]) -> diffrax.SaveAt:
    """Resolve a CPax sampling policy against its problem's interval."""
    t0, t1 = t_span

    if policy.kind == "final":
        return diffrax.SaveAt(t1=True)

    if policy.kind == "linspace":
        return diffrax.SaveAt(ts=jnp.linspace(t0, t1, policy.n))

    if policy.kind == "at":
        times = policy.times
        if times is None:
            raise ValueError("SaveAt.at requires an array of times.")
        # Concrete checks provide an immediate, intelligible error. Diffrax
        # itself handles the admissibility of traced times during solving.
        if not any(isinstance(value, Tracer) for value in (t0, t1, times)):
            interval_start = float(np.asarray(t0))
            interval_end = float(np.asarray(t1))
            concrete_times = np.asarray(times)
            if np.any(concrete_times < interval_start) or np.any(concrete_times > interval_end):
                raise ValueError("SaveAt.at times must lie within the IVP time interval.")
        return diffrax.SaveAt(ts=times)

    raise ValueError(f"Unsupported SaveAt policy: {policy.kind!r}.")


def solve_diffrax(
    problem: IVP,
    *,
    solver: diffrax.AbstractSolver,
    solver_selection: str,
    saveat: Any,
    dt0: float | jax.Array | None,
    stepsize_controller: diffrax.AbstractStepSizeController,
    max_steps: int,
    throw: bool,
) -> Solution:
    """Translate a CPax IVP into a Diffrax solve and back into CPax."""

    def vector_field(
        t: float | jax.Array | np.ndarray,
        state: Any,
        args: Any,
        /,
    ) -> Any:
        """Adapt Diffrax's scalar-like time input to CPax's time interface."""
        return problem.system.vector_field(jnp.asarray(t), state, args)

    t0, t1 = problem.t_span
    backend_solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=problem.y0,
        args=problem.args,
        saveat=_resolve_saveat(saveat, problem.t_span),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )
    # Keep the numerical status JAX-traceable: never convert the result
    # to a Python boolean/string inside a compiled integration.
    # Diffrax returns an Equinox EnumerationItem, not a numeric JAX array.
    # Its comparison operator yields a JAX-compatible scalar. Preserve the
    # original result in metadata to retain precise backend failure reasons.
    status = jnp.where(
        backend_solution.result == diffrax.RESULTS.successful,
        int(SolutionStatus.SUCCESS),
        int(SolutionStatus.FAILED),
    )
    return Solution(
        ts=backend_solution.ts,
        ys=backend_solution.ys,
        status=status,
        stats=dict(backend_solution.stats),
        metadata={
            "backend": "diffrax",
            "solver": type(solver).__name__,
            "solver_selection": solver_selection,
            "stepsize_controller": type(stepsize_controller).__name__,
            "dt0": dt0,
            "rtol": (stepsize_controller.rtol if isinstance(stepsize_controller, diffrax.PIDController) else None),
            "atol": (stepsize_controller.atol if isinstance(stepsize_controller, diffrax.PIDController) else None),
            "max_steps": max_steps,
            "backend_result": backend_solution.result,
        },
    )
