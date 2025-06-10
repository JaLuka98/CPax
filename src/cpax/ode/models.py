import jax
import jax.numpy as jnp
from typing import Callable, Literal, Tuple, Union


def potential_gravitational(masses: jnp.ndarray) -> Callable[[jnp.ndarray], float]:
    """Returns V(q) for Newtonian gravity between all pairs of particles."""
    G = 4 * jnp.pi**2

    def V(q: jnp.ndarray) -> float:
        N, D = q.shape
        diffs = q[:, None, :] - q[None, :, :]  # (N, N, D)
        distances = jnp.linalg.norm(diffs + jnp.eye(N)[:, :, None], axis=-1)
        mask = 1.0 - jnp.eye(N)
        mprod = masses[:, None] * masses[None, :]
        V_mat = -G * mprod / distances * mask
        return 0.5 * jnp.sum(V_mat)

    return V

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

def hamiltonian_nd(
    masses: jnp.ndarray,
    potential: Union[str, Callable[[jnp.ndarray], float]] = "gravity"
) -> Callable[[jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    General Hamiltonian dynamics in N-body system using canonical coordinates.

    Parameters
    ----------
    masses : jnp.ndarray
        Array of shape (N,) with masses.
    potential : str or callable
        Either a string like 'gravity' or a function V(q) -> float.

    Returns
    -------
    dynamics : Callable
        Function f(q, p, t) -> (dqdt, dpdt)
    """
    if isinstance(potential, str):
        if potential == "gravity":
            V = potential_gravitational(masses)
        else:
            raise ValueError(f"Unknown potential alias: {potential}")
    else:
        V = potential

    grad_V = jax.grad(V)

    def dynamics(q: jnp.ndarray, p: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dqdt = p / masses[:, None]
        dpdt = -grad_V(q)
        return dqdt, dpdt

    return dynamics
