"""Backend-independent policies for sampling IVP solutions."""

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

type SaveKind = Literal["final", "linspace", "at"]


class SaveAt(eqx.Module):
    """Specify which IVP states to retain, independently of the solver.

    Construct policies using :meth:`final`, :meth:`linspace`, or
    :meth:`at`. A linspace policy is resolved over the problem interval
    when ``solve`` is called; it is not a request for fixed integration
    steps.
    """

    kind: SaveKind = eqx.field(static=True)
    n: int | None = eqx.field(default=None, static=True)
    times: jax.Array | None = None

    @classmethod
    def final(cls) -> "SaveAt":
        """Save only the final state (the default policy)."""
        return cls(kind="final")

    @classmethod
    def linspace(cls, n: int) -> "SaveAt":
        """Save exactly ``n`` equally spaced times, including both endpoints."""
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
            raise ValueError("SaveAt.linspace(n) requires an integer n >= 2.")
        return cls(kind="linspace", n=int(n))

    @classmethod
    def at(cls, times: Sequence[float] | jax.Array) -> "SaveAt":
        """Save at explicitly supplied, strictly increasing times."""
        try:
            arr = jnp.asarray(times)
        except (TypeError, ValueError) as exc:
            raise TypeError("SaveAt.at times must be a one-dimensional numeric array.") from exc
        if arr.ndim != 1 or arr.shape[0] == 0 or arr.dtype.kind not in "iuf":
            raise ValueError("SaveAt.at times must be a nonempty 1D array of real values.")
        if arr.dtype.kind in "iu":
            arr = arr.astype(jnp.result_type(float))

        if not isinstance(arr, Tracer):
            concrete = np.asarray(arr)
            if not np.all(np.isfinite(concrete)) or not np.all(np.diff(concrete) > 0):
                raise ValueError("SaveAt.at times must be finite and strictly increasing.")

        return cls(kind="at", times=arr)
