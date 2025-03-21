import jax.numpy as jnp
from cpax.ode.integrators import simulate_rk4_scan
from cpax.ode.models import newtonian_1d, hamiltonian_1d

# Define some global variables for these test cases
ps0 = jnp.array([1.0, 0.0])  # position and velocity (or momentum for unit mass): initial phase space point
t0 = 0.0
dt = 0.01
n_steps = 500
k = 1.0


def check_energy_conservation(ps, k):
    # Check for conservation of energy
    energies = 0.5 * (ps[:, 1]**2 + k * ps[:, 0]**2)
    initial_energy = energies[0]
    assert jnp.allclose(energies, initial_energy, atol=1e-2), "Energy is not conserved."


# Rename helper function so it isn’t collected as a test and remove jax.jit
def ode_test_function(state, t, k=1.0):
    # For a simple harmonic oscillator: dx/dt = v, dv/dt = -k*x
    x, v = state
    return jnp.array([v, -k * x])


def different_ode(state, t, omega=1.5):
    x, v = state
    return jnp.array([v, -omega * x])


def test_HO_manual():

    # Pass the non-jitted helper function to simulate_rk4_scan
    ts, ps = simulate_rk4_scan(ps0, t0, dt, n_steps, ode_test_function, k=k)

    # Check that the outputs have the expected shapes
    assert ps.shape == (n_steps, 2), "Incorrect shape for state array."
    assert ts.shape == (n_steps,), "Incorrect shape for time array."
    check_energy_conservation(ps, k)


def test_HO_Newtonian():
    # Define the potential energy function for a simple harmonic oscillator
    def V(q):
        return 0.5 * q**2

    newtonian_1d_HO = newtonian_1d(V)

    assert newtonian_1d_HO(ps0, t0).shape == (2,), "Incorrect shape for state array."
    assert newtonian_1d_HO(ps0, t0).dtype == jnp.float32, "Incorrect dtype for state array."
    check_dynamics = jnp.allclose(newtonian_1d_HO(ps0, t0), jnp.array([0.0, -1.0]), atol=1e-7)
    assert check_dynamics, "Incorrect dynamics for initial state."

    ts, ps = simulate_rk4_scan(ps0, t0, 0.01, 500, newtonian_1d_HO)

    assert ps.shape == (n_steps, 2), "Incorrect shape for state array."
    assert ts.shape == (n_steps,), "Incorrect shape for time array."
    check_energy_conservation(ps, k)


def test_HO_Hamiltonian():
    # Define the Hamiltonian for a simple harmonic oscillator
    def H(q, p):
        return 0.5 * p**2 + 0.5 * k * q**2

    hamiltonian_1d_HO = hamiltonian_1d(H)

    assert hamiltonian_1d_HO(ps0, t0).shape == (2,), "Incorrect shape for state array."
    assert hamiltonian_1d_HO(ps0, t0).dtype == jnp.float32, "Incorrect dtype for state array."
    check_dynamics = jnp.allclose(hamiltonian_1d_HO(ps0, t0), jnp.array([0.0, -1.0]), atol=1e-7)
    assert check_dynamics, "Incorrect dynamics for initial state."

    ts, ps = simulate_rk4_scan(ps0, t0, 0.01, 500, hamiltonian_1d_HO)

    assert ps.shape == (n_steps, 2), "Incorrect shape for state array."
    assert ts.shape == (n_steps,), "Incorrect shape for time array."
    check_energy_conservation(ps, k)
