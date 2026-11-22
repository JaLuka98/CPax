"""Result containers for dynamical-systems analysis."""

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from cpax._validation import enum_code, time_grid
from cpax.results import SolutionStatus


class LyapunovQuality(IntEnum):
    """Numerical-health outcomes for a Lyapunov-spectrum estimate.

    This is deliberately separate from :class:`cpax.results.SolutionStatus`.
    A successful ODE solve may still have a poorly conditioned tangent basis.
    """

    PASS = 0
    WARNING = 1
    INVALID = 2


class LyapunovDiagnostics(eqx.Module):
    """JAX-compatible numerical-health measurements for a spectrum estimate.

    ``qr_log_diagonal_spans`` is the spread of the logarithms of the absolute
    QR diagonal factors for each interval. It is a cheap conditioning proxy,
    not a singular-value condition number. ``liouville_residuals`` compares
    accumulated QR volume growth with the integral of the vector-field
    divergence. ``tail_drift`` compares the final estimate with the estimate
    halfway through the valid analysis history.
    """

    qr_log_diagonal_spans: jax.Array
    liouville_residuals: jax.Array
    tail_drift: jax.Array
    num_valid_segments: jax.Array | int
    precision_budget: jax.Array
    qr_warning_threshold: jax.Array
    liouville_rate_residual: jax.Array
    liouville_rate_tolerance: jax.Array
    quality: jax.Array | int

    @property
    def is_healthy(self) -> jax.Array:
        """Whether no numerical-health warning or invalid result was found."""
        return jnp.equal(self.quality, int(LyapunovQuality.PASS))


class LyapunovResult(eqx.Module):
    """A full Lyapunov-spectrum estimate and its convergence history.

    ``times`` contains the end time of every QR re-orthonormalisation
    interval. ``running_exponents[i]`` is the descending spectrum estimated
    from all intervals through ``times[i]``. ``exponents`` equals the final
    row when integration succeeds. Failed calculations may retain a valid
    prefix in ``running_exponents`` and use NaN values thereafter.
    """

    exponents: jax.Array
    times: jax.Array
    running_exponents: jax.Array
    status: jax.Array | int
    diagnostics: LyapunovDiagnostics | None = None
    stats: dict[str, Any] = eqx.field(default_factory=dict)
    metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def __post_init__(self) -> None:
        exponents = jnp.asarray(self.exponents)
        if exponents.ndim != 1 or exponents.shape[0] < 1 or exponents.dtype.kind != "f":
            raise ValueError("exponents must be a nonempty one-dimensional real floating-point array.")

        times = time_grid("times", self.times)

        running_exponents = jnp.asarray(self.running_exponents)
        if (
            running_exponents.ndim != 2
            or running_exponents.shape != (times.shape[0], exponents.shape[0])
            or running_exponents.dtype.kind != "f"
        ):
            raise ValueError(
                "running_exponents must have shape (len(times), len(exponents)) and a real floating dtype."
            )

        status = enum_code("status", self.status, SolutionStatus)

        diagnostics = self.diagnostics
        if diagnostics is None:
            diagnostics = LyapunovDiagnostics(
                qr_log_diagonal_spans=jnp.full(times.shape, jnp.nan, dtype=exponents.dtype),
                liouville_residuals=jnp.full(times.shape, jnp.nan, dtype=exponents.dtype),
                tail_drift=jnp.full(exponents.shape, jnp.nan, dtype=exponents.dtype),
                num_valid_segments=jnp.asarray(0, dtype=jnp.int32),
                precision_budget=jnp.asarray(jnp.nan, dtype=exponents.dtype),
                qr_warning_threshold=jnp.asarray(jnp.nan, dtype=exponents.dtype),
                liouville_rate_residual=jnp.asarray(jnp.nan, dtype=exponents.dtype),
                liouville_rate_tolerance=jnp.asarray(jnp.nan, dtype=exponents.dtype),
                quality=jnp.asarray(int(LyapunovQuality.INVALID), dtype=jnp.int32),
            )
        self._validate_diagnostics(diagnostics, times=times, exponents=exponents)

        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "running_exponents", running_exponents)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "diagnostics", diagnostics)

    @staticmethod
    def _validate_diagnostics(diagnostics: LyapunovDiagnostics, *, times: jax.Array, exponents: jax.Array) -> None:
        """Validate the fixed-shape, JAX-compatible diagnostic fields."""
        if not isinstance(diagnostics, LyapunovDiagnostics):
            raise TypeError("diagnostics must be a LyapunovDiagnostics instance or None.")
        for name in ("qr_log_diagonal_spans", "liouville_residuals"):
            value = jnp.asarray(getattr(diagnostics, name))
            if value.shape != times.shape or value.dtype.kind != "f":
                raise ValueError(f"diagnostics.{name} must match times and have a real floating dtype.")
        tail_drift = jnp.asarray(diagnostics.tail_drift)
        if tail_drift.shape != exponents.shape or tail_drift.dtype.kind != "f":
            raise ValueError("diagnostics.tail_drift must match exponents and have a real floating dtype.")
        for name in (
            "precision_budget",
            "qr_warning_threshold",
            "liouville_rate_residual",
            "liouville_rate_tolerance",
        ):
            value = jnp.asarray(getattr(diagnostics, name))
            if value.ndim != 0 or value.dtype.kind != "f":
                raise ValueError(f"diagnostics.{name} must be a scalar real floating-point value.")
        for name in ("num_valid_segments", "quality"):
            value = jnp.asarray(getattr(diagnostics, name))
            if value.ndim != 0 or value.dtype.kind not in "iu":
                raise TypeError(f"diagnostics.{name} must be a scalar integer value.")
        if not isinstance(diagnostics.num_valid_segments, Tracer):
            count = int(np.asarray(diagnostics.num_valid_segments))
            if not 0 <= count <= times.size:
                raise ValueError("diagnostics.num_valid_segments must lie between zero and len(times).")
        if not isinstance(diagnostics.quality, Tracer):
            try:
                LyapunovQuality(int(np.asarray(diagnostics.quality)))
            except ValueError as exc:
                raise ValueError("diagnostics.quality must be a LyapunovQuality outcome code.") from exc

    @property
    def is_success(self) -> jax.Array:
        """JAX-compatible indicator that all integration segments succeeded."""
        return jnp.equal(self.status, int(SolutionStatus.SUCCESS))

    def report(self) -> str:
        """Return a concise, host-side interpretation of numerical diagnostics.

        This method intentionally requires concrete result arrays and is not
        JIT-compatible. The raw ``diagnostics`` fields remain JAX-compatible
        for use inside transformations.
        """
        diagnostics = self.diagnostics
        if diagnostics is None:
            raise RuntimeError("LyapunovResult.report() requires diagnostics from lyapunov_spectrum().")
        if any(
            isinstance(value, Tracer)
            for value in (
                self.status,
                diagnostics.quality,
                diagnostics.num_valid_segments,
            )
        ):
            raise RuntimeError("LyapunovResult.report() requires concrete, non-traced result arrays.")

        quality = LyapunovQuality(int(np.asarray(diagnostics.quality)))
        quality_name = quality.name
        count = int(np.asarray(diagnostics.num_valid_segments))
        total = self.times.size
        analysis_time = self.metadata.get("analysis_time")
        qr_interval = self.metadata.get("qr_interval")
        spans = np.asarray(diagnostics.qr_log_diagonal_spans)
        valid_spans = spans[:count]
        max_span = float(np.max(valid_spans)) if valid_spans.size else float("nan")
        lines = [
            f"Lyapunov diagnostics: {quality_name}.",
            f"Completed QR segments: {count}/{total}.",
        ]
        if qr_interval is not None:
            lines.append(f"QR interval: {float(np.asarray(qr_interval)):.6g}.")
        if valid_spans.size:
            lines.append(
                "Largest QR log-diagonal span: "
                f"{max_span:.6g} (advisory threshold {float(np.asarray(diagnostics.qr_warning_threshold)):.6g}; "
                f"{float(np.asarray(diagnostics.precision_budget)):.6g} available log-units)."
            )
        if analysis_time is not None and count:
            lines.append(
                "Liouville residual rate: "
                f"{float(np.asarray(diagnostics.liouville_rate_residual)):.6g} "
                f"(advisory tolerance {float(np.asarray(diagnostics.liouville_rate_tolerance)):.6g})."
            )
        lines.append(
            f"Tail drift |final - halfway|: {np.array2string(np.asarray(diagnostics.tail_drift), precision=4)}."
        )

        if quality is LyapunovQuality.INVALID:
            lines.append(
                "The spectrum is not usable: a segment failed or tangent/volume diagnostics became non-finite."
            )
        elif quality is LyapunovQuality.WARNING:
            lines.append(
                "Numerical consistency warning: reduce the QR interval, tighten tolerances, and rerun before "
                "interpreting the spectrum."
            )
        else:
            lines.append(
                "No configured numerical-health threshold was crossed; finite-time convergence still requires "
                "judging the tail drift and extending the analysis window."
            )
        return "\n".join(lines)
