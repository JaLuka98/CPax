"""Retained symplectic leapfrog integration utilities.

These low-level functions are separate from the CPax physics-first IVP solver.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("f",))
def leapfrog_step(
    f: Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]],
    q: jnp.ndarray,
    p: jnp.ndarray,
    t: float,
    dt: float,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform one kick-drift-kick leapfrog step.

    The method is symplectic for separable Hamiltonians of the form
    ``H(q, p) = T(p) + V(q)``. A time- or momentum-dependent force may still
    be supplied through ``f``, but does not retain that guarantee.

    Parameters
    ----------
    f : Callable
        Hamiltonian dynamics function ``f(q, p, t, **kwargs)`` returning
        ``(dqdt, dpdt)``.
    q : jnp.ndarray
        Positions (N, D)
    p : jnp.ndarray
        Momenta (N, D)
    t : float
        Current time
    dt : float
        Timestep
    kwargs : dict
        Additional arguments forwarded to ``f`` at every force evaluation.

    Returns
    -------
    q_next, p_next : Tuple of arrays
        Updated positions and momenta
    """
    _, dpdt = f(q, p, t, **kwargs)
    p_half = p + 0.5 * dt * dpdt
    dqdt_half, _ = f(q, p_half, t + 0.5 * dt, **kwargs)
    q_new = q + dt * dqdt_half
    _, dpdt_new = f(q_new, p_half, t + dt, **kwargs)
    p_new = p_half + 0.5 * dt * dpdt_new
    return q_new, p_new


@partial(jax.jit, static_argnames=("n_steps", "f"))
def simulate_leapfrog_scan(
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    f: Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]],
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate a trajectory using kick-drift-kick leapfrog integration.

    The initial state is not included in the returned trajectory. Each sample
    contains the state after one integration step and its matching time.

    Returns
    -------
    ts : jnp.ndarray
        Time steps (n_steps,)
    qs : jnp.ndarray
        Position trajectories (n_steps, N, D)
    ps : jnp.ndarray
        Momentum trajectories (n_steps, N, D)
    """

    def step_fn(carry, _):
        q, p, t = carry
        q_new, p_new = leapfrog_step(f, q, p, t, dt, **kwargs)
        t_new = t + dt
        return (q_new, p_new, t_new), (q_new, p_new, t_new)

    _, (qs, ps, ts) = jax.lax.scan(step_fn, (q0, p0, t0), None, length=n_steps)
    return ts, qs, ps
