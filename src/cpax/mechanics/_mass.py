"""Shared constant diagonal-mass validation and coordinate broadcasting."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer


def _validate_mass(mass: Any) -> jax.Array:
    """Normalise scalar/vector/matrix masses without coercing array dtypes."""
    m = jnp.asarray(mass)
    if m.dtype.kind not in "f":
        if m.dtype.kind in "iu" and isinstance(mass, (int, float)) and not isinstance(mass, bool):
            m = m.astype(jnp.result_type(float))
        else:
            raise TypeError("mass must be real and floating-point.")
    if m.ndim > 2:
        raise ValueError("mass must be a scalar, vector or matrix.")
    if not isinstance(m, Tracer):
        concrete = np.asarray(m)
        if not np.all(np.isfinite(concrete) & (concrete > 0)):
            raise ValueError("mass must be finite and strictly positive.")
    return m


def _mass_for(m: jax.Array, q: jax.Array) -> jax.Array:
    """Return mass broadcastable to coordinates under CPax's shape rules."""
    shape = q.shape
    if q.ndim == 0:
        valid = m.ndim == 0
    elif q.ndim == 1:
        valid = m.ndim == 0 or m.shape == shape
    elif q.ndim == 2:
        valid = m.ndim == 0 or m.shape == shape or m.shape == (shape[0],)
        if m.ndim == 1 and m.shape == (shape[0],):
            return m[:, None]
    else:
        valid = False
    if not valid:
        raise ValueError(f"mass shape {m.shape} is incompatible with coordinates of shape {shape}.")
    return m
