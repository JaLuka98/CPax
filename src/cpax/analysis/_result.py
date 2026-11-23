"""Result containers for dynamical-systems analysis."""

import dataclasses
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jaxtyping import AbstractArray, Array, Float, Integer, jaxtyped

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


def _check_annotated_arrays(module: eqx.Module, prefix: str = "") -> None:
    """Check array fields against their jaxtyping annotations.

    Call this inside a ``jaxtyped`` context so that the dimension names
    ``segment`` and ``exponent`` are shared between a result and its
    diagnostics.
    """
    for field in dataclasses.fields(module):
        hint = field.type
        if (
            isinstance(hint, type)
            and issubclass(hint, AbstractArray)
            and not isinstance(jnp.asarray(getattr(module, field.name)), hint)
        ):
            raise ValueError(
                f"{prefix}{field.name} must match {hint.__name__}, "
                "where segment = len(times) and exponent = len(exponents)."
            )


class LyapunovDiagnostics(eqx.Module):
    """JAX-compatible numerical-health measurements for a spectrum estimate.

    ``qr_log_diagonal_spans`` is the spread of the logarithms of the absolute
    QR diagonal factors for each interval. It is a cheap conditioning proxy,
    not a singular-value condition number. ``liouville_residuals`` compares
    accumulated QR volume growth with the integral of the vector-field
    divergence. ``tail_drift`` compares the final estimate with the estimate
    halfway through the valid analysis history.
    """

    qr_log_diagonal_spans: Float[Array, " segment"]
    liouville_residuals: Float[Array, " segment"]
    tail_drift: Float[Array, " exponent"]
    num_valid_segments: Integer[Array, ""] = eqx.field(converter=jnp.asarray)
    precision_budget: Float[Array, ""]
    qr_warning_threshold: Float[Array, ""]
    liouville_rate_residual: Float[Array, ""]
    liouville_rate_tolerance: Float[Array, ""]
    quality: Integer[Array, ""] = eqx.field(converter=jnp.asarray)

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

    Array shapes are checked at construction against the jaxtyping
    annotations, where ``segment`` is the number of QR intervals and
    ``exponent`` the number of exponents.
    """

    exponents: Float[Array, " exponent"]
    times: Float[Array, " segment"]
    running_exponents: Float[Array, "segment exponent"]
    status: Integer[Array, ""]
    diagnostics: LyapunovDiagnostics | None = None
    stats: dict[str, Any] = eqx.field(default_factory=dict)
    metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", jnp.asarray(self.exponents))
        object.__setattr__(self, "times", time_grid("times", self.times))
        object.__setattr__(self, "running_exponents", jnp.asarray(self.running_exponents))
        object.__setattr__(self, "status", enum_code("status", self.status, SolutionStatus))

        with jaxtyped("context"):
            _check_annotated_arrays(self)
            if self.exponents.size < 1:
                raise ValueError("exponents must be a nonempty one-dimensional real floating-point array.")

            diagnostics = self.diagnostics
            if diagnostics is None:
                dtype = self.exponents.dtype
                diagnostics = LyapunovDiagnostics(
                    qr_log_diagonal_spans=jnp.full(self.times.shape, jnp.nan, dtype=dtype),
                    liouville_residuals=jnp.full(self.times.shape, jnp.nan, dtype=dtype),
                    tail_drift=jnp.full(self.exponents.shape, jnp.nan, dtype=dtype),
                    num_valid_segments=jnp.asarray(0, dtype=jnp.int32),
                    precision_budget=jnp.asarray(jnp.nan, dtype=dtype),
                    qr_warning_threshold=jnp.asarray(jnp.nan, dtype=dtype),
                    liouville_rate_residual=jnp.asarray(jnp.nan, dtype=dtype),
                    liouville_rate_tolerance=jnp.asarray(jnp.nan, dtype=dtype),
                    quality=jnp.asarray(int(LyapunovQuality.INVALID), dtype=jnp.int32),
                )
            self._validate_diagnostics(diagnostics, num_segments=self.times.size)

        object.__setattr__(self, "diagnostics", diagnostics)

    @staticmethod
    def _validate_diagnostics(diagnostics: LyapunovDiagnostics, *, num_segments: int) -> None:
        """Validate diagnostics in the dimension context of the enclosing result."""
        if not isinstance(diagnostics, LyapunovDiagnostics):
            raise TypeError("diagnostics must be a LyapunovDiagnostics instance or None.")
        enum_code("diagnostics.quality", diagnostics.quality, LyapunovQuality)
        count = diagnostics.num_valid_segments
        if count.ndim != 0 or count.dtype.kind not in "iu":
            raise TypeError("diagnostics.num_valid_segments must be a scalar integer value.")
        if not isinstance(count, Tracer) and not 0 <= int(count) <= num_segments:
            raise ValueError("diagnostics.num_valid_segments must lie between zero and len(times).")
        _check_annotated_arrays(diagnostics, prefix="diagnostics.")

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
