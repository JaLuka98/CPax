"""Canonical phase-space state representation."""

from typing import Any

import equinox as eqx
import jax

from cpax._validation import as_array


def _as_coordinate(value: Any, name: str) -> jax.Array:
    """Convert Python scalars while rejecting discrete/complex array states."""
    array = as_array(value, f"{name} must be a real numeric scalar or array.", float_scalars=True)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating-point dtype.")
    return array


class PhaseSpaceState(eqx.Module):
    """Canonical coordinates and conjugate momenta with matching array shapes.

    Python numerical scalars are converted to floating JAX arrays. Explicit
    integer-valued arrays, boolean and complex states are not supported.
    `p` denotes *canonical momentum*, not velocity. No batch axis is assumed:
    a shape `(N, D)` can represent N particles in D dimensions.
    """

    q: jax.Array
    p: jax.Array

    def __init__(self, q: Any, p: Any) -> None:
        q_array = _as_coordinate(q, "q")
        p_array = _as_coordinate(p, "p")
        if q_array.shape != p_array.shape:
            raise ValueError("q and p must have identical shapes.")
        if q_array.dtype != p_array.dtype:
            raise TypeError("q and p must have the same floating-point dtype.")
        if q_array.ndim > 2:
            raise ValueError("PhaseSpaceState supports scalar, vector and matrix states.")
        object.__setattr__(self, "q", q_array)
        object.__setattr__(self, "p", p_array)
