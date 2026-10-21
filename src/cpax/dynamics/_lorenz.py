"""The autonomous Lorenz system."""

from typing import Any

import jax
import jax.numpy as jnp

from ._systems import DynamicalSystem


class LorenzSystem(DynamicalSystem):
    """The three-dimensional Lorenz equations.

    Parameters default to the classical chaotic regime. The state is a
    floating-point array of shape (3,). Parameters are JAX arrays and may
    be differentiated when used in a supported solver configuration.
    """

    sigma: jax.Array
    rho: jax.Array
    beta: jax.Array

    def __init__(
        self,
        sigma: float | jax.Array = 10.0,
        rho: float | jax.Array = 28.0,
        beta: float | jax.Array = 8.0 / 3.0,
    ) -> None:
        for name, value in (("sigma", sigma), ("rho", rho), ("beta", beta)):
            array = jnp.asarray(value)
            if array.shape != () or array.dtype.kind not in "iuf":
                raise TypeError(f"{name} must be a real numeric scalar.")
            object.__setattr__(self, name, jnp.asarray(value, dtype=jnp.result_type(float, array)))

    def vector_field(self, t: Any, state: jax.Array, args: Any = None) -> jax.Array:
        """Return the Lorenz vector field with the shape of `state`."""
        del t
        if args is not None:
            raise ValueError("LorenzSystem does not support IVP.args; set system parameters instead.")
        if not hasattr(state, "shape") or state.shape != (3,):
            raise ValueError("LorenzSystem requires an array state of shape (3,).")
        x, y, z = state
        return jnp.stack((self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z))
