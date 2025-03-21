import jax
import jax.numpy as jnp
from typing import Callable


def newtonian_1d(potential: Callable[[float], float]) -> Callable:
    """
    Simulates 1D Newtonian mechanics using a potential V(q).
    Interprets state = [q, p] and assumes unit mass: v = p.

    Parameters
    ----------
    potential : callable
        Potential energy function V(q).

    Returns
    -------
    dynamics : callable
        Function f(state, t) -> dstate/dt with state = [q, p].
    """
    grad_V = jax.grad(potential)

    def dynamics(state: jnp.ndarray, t: float) -> jnp.ndarray:
        q, p = state
        dqdt = p
        dpdt = -grad_V(q)
        return jnp.array([dqdt, dpdt])

    return dynamics


def hamiltonian_1d(H: Callable[[float, float], float]) -> Callable:
    """
    Returns canonical Hamiltonian dynamics in 1D using Hamiltonian H(q, p).

    Parameters
    ----------
    H : callable
        Hamiltonian function H(q, p).

    Returns
    -------
    dynamics : callable
        Function f(state, t) -> dstate/dt with state = [q, p].
    """
    dH_dq = jax.grad(H, argnums=0)
    dH_dp = jax.grad(H, argnums=1)

    def dynamics(state: jnp.ndarray, t: float) -> jnp.ndarray:
        q, p = state
        dqdt = dH_dp(q, p)
        dpdt = -dH_dq(q, p)
        return jnp.array([dqdt, dpdt])

    return dynamics
