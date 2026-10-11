"""Backend-independent numerical solution containers."""

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cpax.dynamics._systems import State


class SolutionStatus(IntEnum):
    """Minimal numerical outcome codes, extensible by future solvers."""

    SUCCESS = 0
    FAILED = 1


class Solution(eqx.Module):
    """Saved states and numerical outcome for a continuous-time IVP.

    Each leaf of ``ys`` has a leading sample dimension corresponding to
    ``ts``, while the remainder matches one leaf of the initial state.
    No assumption is made about the state PyTree structure.

    ``status`` is an integer scalar so success can be inspected under JAX
    transformations. The diagnostics (``stats``) and solver information
    (``metadata``) are intentionally backend-independent mappings.

    Failed solves may contain partial, non-finite or padded samples. The
    container therefore validates shapes and types, but not numerical
    accuracy, finiteness or endpoint coverage.
    """

    ts: jax.Array
    ys: State
    status: jax.Array | int
    stats: dict[str, Any] = eqx.field(default_factory=dict)
    metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def __post_init__(self) -> None:
        ts = jnp.asarray(self.ts)
        if ts.ndim != 1 or ts.shape[0] < 1 or ts.dtype.kind not in "iuf":
            raise ValueError("ts must be a nonempty one-dimensional array of real times.")
        if ts.dtype.kind in "iu":
            ts = ts.astype(jnp.result_type(float))

        if not jax.tree.leaves(self.ys):
            raise ValueError("ys must contain at least one state leaf.")

        n = ts.shape[0]

        def normalise(leaf: Any) -> jax.Array:
            try:
                array = jnp.asarray(leaf)
            except (TypeError, ValueError) as exc:
                raise TypeError("Each ys leaf must be a numerical array.") from exc
            if array.ndim < 1 or array.shape[0] != n:
                raise ValueError("Each ys leaf must have leading dimension len(ts).")
            if array.dtype.kind not in "fc":
                raise TypeError("Each ys leaf must have a floating-point or complex dtype.")
            return array

        ys = jax.tree.map(normalise, self.ys)

        status = jnp.asarray(self.status)
        if status.ndim != 0 or status.dtype.kind not in "iu":
            raise TypeError("status must be a scalar integer outcome code.")
        if not isinstance(status, jax.core.Tracer):
            if int(np.asarray(status)) not in (SolutionStatus.SUCCESS, SolutionStatus.FAILED):
                raise ValueError("status must be a SolutionStatus outcome code.")

        object.__setattr__(self, "ts", ts)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "status", status)

    @property
    def is_success(self) -> jax.Array:
        """JAX-compatible indicator for successful integration."""
        return jnp.equal(self.status, SolutionStatus.SUCCESS)
