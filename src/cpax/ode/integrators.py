import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Dict, Any


def classify_static_args(kwargs: Dict[str, Any]):
    """
    Determines which keyword arguments should be treated as static
    (i.e., non-JAX arrays).
    """
    return tuple(name for name, value in kwargs.items() if not isinstance(value, jnp.ndarray))


@partial(jax.jit, static_argnums=(0,))
def rk4_step(
    f: Callable[[jnp.ndarray, float, dict], jnp.ndarray],
    x: jnp.ndarray,
    t: float,
    dt: float,
    **kwargs
) -> jnp.ndarray:
    """
    Perform one RK4 step of ODE integration with a JAX-traceable function f.

    Parameters
    ----------
    f : callable
        The ODE derivative function: f(x, t, **kwargs) -> dx/dt as a jnp.ndarray.
    x : jnp.ndarray
        Current state of the system.
    t : float
        Current time.
    dt : float
        Timestep.
    kwargs : dict
        Additional arguments passed to f.

    Returns
    -------
    jnp.ndarray
        The new state after one RK4 update step.
    """
    k1 = dt * f(x, t, **kwargs)
    k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt, **kwargs)
    k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt, **kwargs)
    k4 = dt * f(x + k3, t + dt, **kwargs)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6.0


@partial(jax.jit, static_argnums=(3, 4), static_argnames=("k"))
def simulate_rk4_scan(
    x0: jnp.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    f: Callable[[jnp.ndarray, float, dict], jnp.ndarray],
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Use JAX's `lax.scan` to perform n_steps of RK4 integration.

    Parameters
    ----------
    x0 : jnp.ndarray
        Initial state, e.g., [position, velocity].
    t0 : float
        Initial time.
    dt : float
        Timestep.
    n_steps : int
        Number of RK4 steps to perform.
    f : callable
        The ODE derivative function: f(x, t, **kwargs) -> dx/dt.
    kwargs : dict
        Additional parameters passed to f at each step.

    Returns
    -------
    ts : jnp.ndarray
        Times at each step of the integration.
    xs : jnp.ndarray
        State at each step (shape: [n_steps, x0.shape]).
    """
    def step_fn(carry, _):
        x, t = carry
        x_new = rk4_step(f, x, t, dt, **kwargs)
        t_new = t + dt
        return (x_new, t_new), (x_new, t_new)

    (x_final, t_final), (xs, ts) = jax.lax.scan(
        step_fn, (x0, t0), xs=None, length=n_steps
    )
    return ts, xs
