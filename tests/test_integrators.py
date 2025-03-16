import jax.numpy as jnp
from cpax.ode.integrators import simulate_rk4_scan


# Rename helper function so it isn’t collected as a test and remove jax.jit
def ode_test_function(state, t, k=1.0):
    # For a simple harmonic oscillator: dx/dt = v, dv/dt = -k*x
    x, v = state
    return jnp.array([v, -k * x])


def different_ode(state, t, omega=1.5):
    x, v = state
    return jnp.array([v, -omega * x])


def test_harmonic_oscillator():
    x0 = jnp.array([1.0, 0.0])
    t0 = 0.0
    dt = 0.01
    n_steps = 100
    k = 1.0

    # Pass the non-jitted helper function to simulate_rk4_scan
    ts, xs = simulate_rk4_scan(x0, t0, dt, n_steps, ode_test_function, k=k)

    # Check that the outputs have the expected shapes
    assert xs.shape == (n_steps, 2), "Incorrect shape for state array."
    assert ts.shape == (n_steps,), "Incorrect shape for time array."

    # Here, 'omega' should be automatically detected as static
    ts, xs = simulate_rk4_scan(x0, t0, dt, n_steps, different_ode, omega=2.5)

    # Performt he check again
    assert xs.shape == (n_steps, 2), "Incorrect shape for state array."
    assert ts.shape == (n_steps,), "Incorrect shape for time array."

    # (Optional) Add additional checks here, such as conservation properties.
