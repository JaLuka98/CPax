"""Result containers for dynamical-systems analysis."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from cpax.results import SolutionStatus


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
    stats: dict[str, Any] = eqx.field(default_factory=dict)
    metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def __post_init__(self) -> None:
        exponents = jnp.asarray(self.exponents)
        if exponents.ndim != 1 or exponents.shape[0] < 1 or exponents.dtype.kind != "f":
            raise ValueError("exponents must be a nonempty one-dimensional real floating-point array.")

        times = jnp.asarray(self.times)
        if times.ndim != 1 or times.shape[0] < 1 or times.dtype.kind not in "iuf":
            raise ValueError("times must be a nonempty one-dimensional array of real times.")
        if times.dtype.kind in "iu":
            times = times.astype(jnp.result_type(float))

        running_exponents = jnp.asarray(self.running_exponents)
        if (
            running_exponents.ndim != 2
            or running_exponents.shape != (times.shape[0], exponents.shape[0])
            or running_exponents.dtype.kind != "f"
        ):
            raise ValueError(
                "running_exponents must have shape (len(times), len(exponents)) and a real floating dtype."
            )

        status = jnp.asarray(self.status)
        if status.ndim != 0 or status.dtype.kind not in "iu":
            raise TypeError("status must be a scalar integer outcome code.")
        if not isinstance(status, Tracer) and int(np.asarray(status)) not in (
            SolutionStatus.SUCCESS,
            SolutionStatus.FAILED,
        ):
            raise ValueError("status must be a SolutionStatus outcome code.")

        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "running_exponents", running_exponents)
        object.__setattr__(self, "status", status)

    @property
    def is_success(self) -> jax.Array:
        """JAX-compatible indicator that all integration segments succeeded."""
        return jnp.equal(self.status, int(SolutionStatus.SUCCESS))
