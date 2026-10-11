"""Initial-value problems for continuous-time dynamical systems."""

from numbers import Number
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from cpax.dynamics import DynamicalSystem
from cpax.dynamics._systems import State, Time


def _as_initial_state(y0: State) -> State:
    """Normalise numerical PyTree leaves without changing their structure."""
    if not jax.tree.leaves(y0):
        raise ValueError("y0 must contain at least one numerical state leaf.")

    def normalise(leaf: Any) -> jax.Array:
        try:
            array = jnp.asarray(leaf)
        except (TypeError, ValueError) as exc:
            raise TypeError("Each y0 leaf must be a numerical scalar or array.") from exc
        # A Python numeric scalar may describe a continuous initial state.
        # Do not silently cast explicitly integer-valued arrays.
        if array.dtype.kind in "iu" and isinstance(leaf, Number) and not isinstance(leaf, bool):
            array = array.astype(jnp.result_type(float))
        if array.dtype.kind not in "fc":
            raise TypeError("Each y0 leaf must have a floating-point or complex dtype.")
        return array

    return jax.tree.map(normalise, y0)


def _as_time(value: Time) -> jax.Array:
    """Ensure that an interval endpoint is a real numerical scalar."""
    try:
        time = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("t_span endpoints must be real numerical scalars.") from exc
    if time.ndim != 0 or time.dtype.kind not in "iuf":
        raise ValueError("t_span endpoints must be real numerical scalars.")
    if time.dtype.kind in "iu":
        time = time.astype(jnp.result_type(float))
    return time


class IVP(eqx.Module):
    """A continuous-time initial-value problem, independent of its solver.

    The same system may be used with different initial states, intervals and
    evaluation arguments. State leaves are arrays with floating-point or
    complex dtypes. Python numerical scalars are converted to JAX arrays;
    explicitly integer-valued arrays are rejected.

    Only forward intervals (t1 > t0) are currently supported. Finite,
    increasing endpoints are checked at construction when they are concrete;
    traced values defer numerical interval validation to the future solver.
    Physical domain constraints are not checked here.
    """

    system: DynamicalSystem
    y0: State
    t_span: tuple[Time, Time]
    args: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.system, DynamicalSystem):
            raise TypeError("system must implement cpax.dynamics.DynamicalSystem.")
        if not isinstance(self.t_span, tuple) or len(self.t_span) != 2:
            raise ValueError("t_span must be a pair (t0, t1).")

        y0 = _as_initial_state(self.y0)
        t0, t1 = (_as_time(t) for t in self.t_span)

        # No Python boolean conversion is attempted for JAX tracers.
        if not isinstance(t0, Tracer) and not isinstance(t1, Tracer):
            a, b = float(np.asarray(t0)), float(np.asarray(t1))
            if not np.isfinite(a) or not np.isfinite(b) or b <= a:
                raise ValueError("t_span must contain finite endpoints with t1 > t0.")
        # Validate shapes of the initial state and the system's vector field output.
        try:
            derivative = jax.eval_shape(
                lambda y: self.system.vector_field(t0, y, self.args),
                y0,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Could not evaluate the system's vector field with the "
                "provided initial state y0 and args. Check their compatibility "
                "and ensure the vector field supports JAX tracing."
            ) from exc

        # Verify PyTree structure.
        if jax.tree.structure(derivative) != jax.tree.structure(y0):
            raise ValueError(
                "The vector field derivative must have the same PyTree structure as the initial state y0."
            )

        # Verify individual leaf shapes.
        shapes_match = jax.tree.map(
            lambda state, dy: state.shape == dy.shape,
            y0,
            derivative,
        )

        if not all(jax.tree.leaves(shapes_match)):
            raise ValueError("The vector field derivative must have the same leaf shapes as the initial state y0.")

        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "t_span", (t0, t1))
