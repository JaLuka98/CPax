"""Benettin QR estimates for continuous-time Lyapunov spectra."""

from numbers import Integral
from typing import Any, Literal

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.flatten_util import ravel_pytree

from cpax.problems import IVP
from cpax.results import SolutionStatus
from cpax.solvers._auto import Auto
from cpax.solvers._configuration import configure_solver, resolve_solver
from cpax.solvers._velocity_verlet import VelocityVerlet

from ._result import LyapunovResult

_STAT_KEYS = ("num_steps", "num_accepted_steps", "num_rejected_steps")
type _TimeLike = float | int | jax.Array | np.ndarray


def _static_nonnegative_real(name: str, value: float | jax.Array) -> float:
    """Return a concrete nonnegative scalar configuration value."""
    if isinstance(value, Tracer):
        raise TypeError(f"{name} must be a concrete nonnegative finite real scalar.")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a concrete nonnegative finite real scalar.") from exc
    if array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array) or array < 0:
        raise ValueError(f"{name} must be a concrete nonnegative finite real scalar.")
    return float(array)


def _validate_num_segments(num_segments: int) -> int:
    """Validate the static QR interval count."""
    if not isinstance(num_segments, Integral) or isinstance(num_segments, bool) or num_segments < 1:
        raise ValueError("num_segments must be a positive integer.")
    return int(num_segments)


def _final_state(ys: Any) -> Any:
    """Extract the final state from a Diffrax SaveAt(t1=True) result."""
    return jax.tree.map(lambda leaf: leaf[-1], ys)


def _stats_vector(solution: diffrax.Solution) -> jax.Array:
    """Convert the stable public Diffrax step counters to one array."""
    return jnp.stack([jnp.asarray(solution.stats[key]) for key in _STAT_KEYS])


def _real_state_dimension(problem: IVP) -> tuple[jax.Array, Any, int]:
    """Validate real state leaves and obtain a fixed flattened coordinate system."""
    leaves = jax.tree.leaves(problem.y0)
    if any(leaf.dtype.kind != "f" for leaf in leaves):
        raise TypeError("lyapunov_spectrum requires real floating-point IVP state leaves.")
    flat_state, unravel_state = ravel_pytree(problem.y0)
    if flat_state.size < 1:
        raise ValueError("lyapunov_spectrum requires a nonempty state.")
    return flat_state, unravel_state, int(flat_state.size)


def lyapunov_spectrum(
    problem: IVP,
    *,
    transient_time: float | jax.Array = 0.0,
    num_segments: int,
    solver: diffrax.AbstractSolver | Auto | Literal["auto"] = "auto",
    dt0: float | jax.Array | None = None,
    rtol: float | jax.Array | None = None,
    atol: float | jax.Array | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    max_steps: int = 4096,
    throw: bool = True,
) -> LyapunovResult:
    """Estimate all Lyapunov exponents of a smooth continuous-time IVP.

    The routine evolves the physical state and the tangent variational
    equation simultaneously. At the end of each equally sized analysis
    segment, it performs a QR factorisation and accumulates the logarithmic
    diagonal stretch factors (the Benettin algorithm). The returned spectrum
    and every row of ``running_exponents`` are sorted in descending order.

    ``transient_time`` is a concrete configuration value measured from the
    start of ``problem.t_span``. The remaining interval is divided into the
    required, static ``num_segments`` re-orthonormalisation intervals. The
    supplied IVP is the only parameter channel: generic vector fields receive
    ``problem.args`` and physics-first systems retain their ``system.params``.

    ``max_steps`` applies independently to the transient solve (when present)
    and each QR segment. The tangent matrix has shape ``(D, D)``, where ``D``
    is the total number of scalar coordinates across the state PyTree.
    """
    if not isinstance(problem, IVP):
        raise TypeError("lyapunov_spectrum currently supports only cpax.IVP problems.")
    transient = _static_nonnegative_real("transient_time", transient_time)
    segment_count = _validate_num_segments(num_segments)
    flat_initial_state, unravel_state, dimension = _real_state_dimension(problem)

    preliminary_solver, _ = resolve_solver(solver)
    if isinstance(preliminary_solver, VelocityVerlet):
        raise TypeError("lyapunov_spectrum does not support VelocityVerlet for augmented tangent dynamics.")
    resolved_solver, solver_selection, controller = configure_solver(
        solver=solver,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )

    t0, t1 = problem.t_span
    if not isinstance(t0, Tracer) and not isinstance(t1, Tracer):
        analysis_duration_value = float(np.asarray(t1)) - float(np.asarray(t0)) - transient
        if not np.isfinite(analysis_duration_value) or analysis_duration_value <= 0:
            raise ValueError("transient_time must leave a positive analysis interval within problem.t_span.")

    def base_vector_field(t: _TimeLike, state: Any, args: Any) -> Any:
        return problem.system.vector_field(jnp.asarray(t), state, args)

    def flat_vector_field(t: _TimeLike, flat_state: jax.Array, args: Any) -> jax.Array:
        derivative = base_vector_field(t, unravel_state(flat_state), args)
        flattened_derivative, _ = ravel_pytree(derivative)
        return flattened_derivative

    # Produce an immediate, CPax-specific error for a field that can be
    # evaluated normally but cannot be differentiated with respect to state.
    try:
        derivative_shape = jax.eval_shape(flat_vector_field, t0, flat_initial_state, problem.args)
        if derivative_shape.shape != (dimension,) or derivative_shape.dtype.kind != "f":
            raise ValueError("vector_field must return real floating derivatives with the IVP state shape.")
        jacobian_shape = jax.eval_shape(
            jax.jacfwd(lambda state: flat_vector_field(t0, state, problem.args)),
            flat_initial_state,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lyapunov_spectrum requires a JAX-differentiable vector field with real derivatives. "
            f"Original error: {exc}"
        ) from exc
    if jacobian_shape.shape != (dimension, dimension):
        raise ValueError("The vector-field Jacobian must have shape (state_dimension, state_dimension).")

    def tangent_vector_field(t: _TimeLike, augmented_state: tuple[Any, jax.Array], args: Any) -> tuple[Any, jax.Array]:
        state, tangent_basis = augmented_state
        flat_state, _ = ravel_pytree(state)
        jacobian = jax.jacfwd(lambda coordinates: flat_vector_field(t, coordinates, args))(flat_state)
        return base_vector_field(t, state, args), jacobian @ tangent_basis

    base_term = diffrax.ODETerm(base_vector_field)
    tangent_term = diffrax.ODETerm(tangent_vector_field)

    def solve_base(start: _TimeLike, end: _TimeLike, state: Any) -> diffrax.Solution:
        return diffrax.diffeqsolve(
            terms=base_term,
            solver=resolved_solver,
            t0=start,
            t1=end,
            dt0=dt0,
            y0=state,
            args=problem.args,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            max_steps=int(max_steps),
            throw=throw,
        )

    def solve_tangent(start: _TimeLike, end: _TimeLike, state: Any, tangent_basis: jax.Array) -> diffrax.Solution:
        return diffrax.diffeqsolve(
            terms=tangent_term,
            solver=resolved_solver,
            t0=start,
            t1=end,
            dt0=dt0,
            y0=(state, tangent_basis),
            args=problem.args,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            max_steps=int(max_steps),
            throw=throw,
        )

    stats_zeros = jnp.zeros(len(_STAT_KEYS), dtype=jnp.asarray(0).dtype)
    if transient == 0.0:
        state_after_transient = problem.y0
        transient_success = jnp.asarray(True)
        transient_stats = stats_zeros
    else:
        transient_solution = solve_base(t0, t0 + transient, problem.y0)
        transient_success = transient_solution.result == diffrax.RESULTS.successful
        state_after_transient = jax.lax.cond(
            transient_success,
            lambda _: _final_state(transient_solution.ys),
            lambda _: problem.y0,
            operand=None,
        )
        transient_stats = _stats_vector(transient_solution)

    analysis_start = t0 + transient
    analysis_duration = t1 - analysis_start
    segment_duration = analysis_duration / segment_count
    times = analysis_start + segment_duration * jnp.arange(1, segment_count + 1)
    tangent_dtype = jnp.result_type(flat_initial_state.dtype, derivative_shape.dtype)
    initial_basis = jnp.eye(dimension, dtype=tangent_dtype)
    initial_logs = jnp.zeros(dimension, dtype=tangent_dtype)
    nan_spectrum = jnp.full(dimension, jnp.nan, dtype=tangent_dtype)

    def segment_body(
        carry: tuple[Any, jax.Array, jax.Array, jax.Array], index: jax.Array
    ) -> tuple[tuple[Any, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        state, tangent_basis, logarithmic_stretches, active = carry
        segment_start = analysis_start + index * segment_duration
        segment_end = segment_start + segment_duration
        elapsed = (index + 1) * segment_duration

        def skipped(_: None) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return state, tangent_basis, logarithmic_stretches, jnp.asarray(False), nan_spectrum, stats_zeros

        def integrate(_: None) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            solution = solve_tangent(segment_start, segment_end, state, tangent_basis)
            segment_success = solution.result == diffrax.RESULTS.successful
            segment_stats = _stats_vector(solution)

            def accepted(_: None) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                next_state, propagated_basis = _final_state(solution.ys)
                orthogonal_basis, upper = jnp.linalg.qr(propagated_basis)
                diagonal = jnp.diag(upper)
                signs = jnp.where(diagonal < 0, -jnp.ones_like(diagonal), jnp.ones_like(diagonal))
                next_basis = orthogonal_basis * signs[jnp.newaxis, :]
                next_stretches = logarithmic_stretches + jnp.log(diagonal * signs)
                spectrum = jnp.flip(jnp.sort(next_stretches / elapsed))
                return next_state, next_basis, next_stretches, jnp.asarray(True), spectrum, segment_stats

            def failed(_: None) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                return state, tangent_basis, logarithmic_stretches, jnp.asarray(False), nan_spectrum, segment_stats

            return jax.lax.cond(segment_success, accepted, failed, operand=None)

        next_state, next_basis, next_stretches, next_active, spectrum, segment_stats = jax.lax.cond(
            active,
            integrate,
            skipped,
            operand=None,
        )
        return (next_state, next_basis, next_stretches, next_active), (spectrum, segment_stats)

    (_, _, _, final_active), (running_exponents, segment_stats) = jax.lax.scan(
        segment_body,
        (state_after_transient, initial_basis, initial_logs, transient_success),
        jnp.arange(segment_count),
    )
    exponents = jnp.where(final_active, running_exponents[-1], nan_spectrum)
    statistics = transient_stats + jnp.sum(segment_stats, axis=0)
    status = jnp.where(
        final_active,
        jnp.asarray(int(SolutionStatus.SUCCESS), dtype=jnp.int32),
        jnp.asarray(int(SolutionStatus.FAILED), dtype=jnp.int32),
    )

    return LyapunovResult(
        exponents=exponents,
        times=times,
        running_exponents=running_exponents,
        status=status,
        stats={name: statistics[i] for i, name in enumerate(_STAT_KEYS)},
        metadata={
            "backend": "diffrax",
            "solver": type(resolved_solver).__name__,
            "solver_selection": solver_selection,
            "stepsize_controller": type(controller).__name__,
            "dt0": dt0,
            "rtol": (controller.rtol if isinstance(controller, diffrax.PIDController) else None),
            "atol": (controller.atol if isinstance(controller, diffrax.PIDController) else None),
            "max_steps": int(max_steps),
            "transient_time": transient,
            "analysis_time": analysis_duration,
            "num_segments": segment_count,
            "state_dimension": dimension,
        },
    )
