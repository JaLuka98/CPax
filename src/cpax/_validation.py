"""Shared argument validation for CPax constructors and routines."""

from enum import IntEnum
from numbers import Integral, Number
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer


def as_array(value: Any, message: str, *, float_scalars: bool = False) -> jax.Array:
    """Convert a value to a JAX array, reporting failures as ``TypeError(message)``.

    With ``float_scalars``, a Python numeric scalar (but not a Boolean) that
    describes a continuous quantity is promoted to the default float dtype.
    Explicitly integer-valued arrays are never cast silently.
    """
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(message) from exc
    if float_scalars and array.dtype.kind in "iu" and isinstance(value, Number) and not isinstance(value, bool):
        array = array.astype(jnp.result_type(float))
    return array


def require_callable(name: str, value: Any, *, optional: bool = False) -> None:
    """Reject a non-callable value, or a non-callable non-``None`` value if optional."""
    if optional and value is None:
        return
    if not callable(value):
        raise TypeError(f"{name} must be callable or None." if optional else f"{name} must be callable.")


def _finite_real_scalar(value: float | jax.Array, message: str, *, allow_zero: bool) -> float:
    """Return a concrete finite real scalar that is positive, or nonnegative if allowed."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(message) from exc
    if (
        array.ndim != 0
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array)
        or array < 0
        or (array == 0 and not allow_zero)
    ):
        raise ValueError(message)
    return float(array)


def positive_real(name: str, value: float | jax.Array) -> None:
    """Validate a positive finite real scalar when it is concrete."""
    if not isinstance(value, Tracer):
        _finite_real_scalar(value, f"{name} must be a positive finite real scalar.", allow_zero=False)


def concrete_nonnegative_real(name: str, value: float | jax.Array) -> float:
    """Return a concrete nonnegative finite real scalar configuration value."""
    message = f"{name} must be a concrete nonnegative finite real scalar."
    if isinstance(value, Tracer):
        raise TypeError(message)
    return _finite_real_scalar(value, message, allow_zero=True)


def positive_integer(name: str, value: int) -> int:
    """Return a static positive integer, rejecting Booleans."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def enum_code(name: str, value: jax.Array | int, codes: type[IntEnum]) -> jax.Array:
    """Return a scalar integer outcome code, checking membership when it is concrete."""
    code = jnp.asarray(value)
    if code.ndim != 0 or code.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a scalar integer outcome code.")
    if not isinstance(code, Tracer):
        try:
            codes(int(np.asarray(code)))
        except ValueError as exc:
            raise ValueError(f"{name} must be a {codes.__name__} outcome code.") from exc
    return code


def time_grid(name: str, value: jax.Array) -> jax.Array:
    """Return a nonempty one-dimensional real time array with a floating dtype."""
    times = jnp.asarray(value)
    if times.ndim != 1 or times.shape[0] < 1 or times.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional array of real times.")
    if times.dtype.kind in "iu":
        times = times.astype(jnp.result_type(float))
    return times
