import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Dict, Any


def classify_static_args(kwargs: Dict[str, Any]):
    return tuple(name for name, value in kwargs.items() if not isinstance(value, jnp.ndarray))


@partial(jax.jit, static_argnums=(0,))
def rk4_step(
    f: Callable[[jnp.ndarray, float, dict], jnp.ndarray],
    state: jnp.ndarray,
    t: float,
    dt: float,
    **kwargs
) -> jnp.ndarray:
    r"""
    Perform one RK4 step of ODE integration with a JAX-traceable function f.

    Parameters
    ----------
    f : callable
        The ODE derivative function: f(state, t, \*\*kwargs) -> dstate/dt as a jnp.ndarray.
        Assumes state = [q, p] or generalised form.
    state : jnp.ndarray
        Current state vector.
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
    k1 = dt * f(state, t, **kwargs)
    k2 = dt * f(state + 0.5 * k1, t + 0.5 * dt, **kwargs)
    k3 = dt * f(state + 0.5 * k2, t + 0.5 * dt, **kwargs)
    k4 = dt * f(state + k3, t + dt, **kwargs)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6.0


@partial(jax.jit, static_argnums=(3, 4), static_argnames=("k"))
def simulate_rk4_scan(
    state0: jnp.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    f: Callable[[jnp.ndarray, float, dict], jnp.ndarray],
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Use JAX's `lax.scan` to perform n_steps of RK4 integration.

    Parameters
    ----------
    state0 : jnp.ndarray
        Initial state, e.g., [q, p].
    t0 : float
        Initial time.
    dt : float
        Timestep.
    n_steps : int
        Number of RK4 steps to perform.
    f : callable
        The ODE derivative function: f(state, t, \*\*kwargs) -> dstate/dt.
    kwargs : dict
        Additional parameters passed to f at each step.

    Returns
    -------
    ts : jnp.ndarray
        Times at each step of the integration.
    states : jnp.ndarray
        State at each step (shape: [n_steps, state0.shape]).
    """
    def step_fn(carry, _):
        state, t = carry
        state_new = rk4_step(f, state, t, dt, **kwargs)
        t_new = t + dt
        return (state_new, t_new), (state_new, t_new)

    (state_final, t_final), (states, ts) = jax.lax.scan(
        step_fn, (state0, t0), xs=None, length=n_steps
    )
    return ts, states
