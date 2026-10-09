from collections.abc import Callable

import jax
import jax.numpy as jnp


def potential_gravitational(masses: jnp.ndarray) -> Callable[[jnp.ndarray], jax.Array]:
    """Return Newtonian pairwise gravitational potential energy.

    The gravitational constant is ``G = 4 * pi**2``, suitable for the usual
    astronomical unit system (AU, solar masses, and years). Positions must
    have shape ``(N, D)`` matching the number of supplied masses. There is no
    collision softening: distinct particles at the same position are singular.
    """
    G = 4 * jnp.pi**2

    def V(q: jnp.ndarray) -> jax.Array:
        n_particles = q.shape[0]
        displacements = q[:, None, :] - q[None, :, :]
        self_pairs = jnp.eye(n_particles, dtype=q.dtype)
        # Offset self-pairs before the norm so autodiff never differentiates
        # norm(0). They are subsequently removed from the interaction sum.
        distances = jnp.linalg.norm(displacements + self_pairs[:, :, None], axis=-1)
        pair_masses = masses[:, None] * masses[None, :]
        interactions = (1.0 - self_pairs) * pair_masses / distances
        return -0.5 * G * jnp.sum(interactions)

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
    masses: jnp.ndarray, potential: str | Callable[[jnp.ndarray], jax.Array] = "gravity"
) -> Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Return separable Hamiltonian dynamics for an N-body system.

    The generated dynamics represent ``H(q, p) = sum(p**2 / (2 m)) + V(q)``
    in canonical coordinates. They are therefore compatible with the
    symplectic leapfrog integrator.

    Parameters
    ----------
    masses : jnp.ndarray
        Nonzero particle masses with shape ``(N,)``.
    potential : str or callable
        ``"gravity"`` for :func:`potential_gravitational`, or a callable
        ``V(q) -> float``. The gravity alias uses ``G = 4 * pi**2`` and has
        no collision softening.

    Returns
    -------
    dynamics : Callable
        Function ``f(q, p, t) -> (dqdt, dpdt)`` for arrays shaped ``(N, D)``.
    """
    if isinstance(potential, str):
        if potential == "gravity":
            V = potential_gravitational(masses)
        else:
            raise ValueError(f"Unknown potential alias: {potential}")
    else:
        V = potential

    grad_V = jax.grad(V)

    def dynamics(q: jnp.ndarray, p: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        dqdt = p / masses[:, None]
        dpdt = -grad_V(q)
        return dqdt, dpdt

    return dynamics
