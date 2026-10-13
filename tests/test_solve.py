"""Integration contract, reference, configuration and failure tests for CPax."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.dynamics import VectorFieldSystem
from cpax.results import Solution, SolutionStatus


def _decay(t, y, args):
    del t
    return -args["rate"] * y


def _oscillator(t, state, args):
    del t
    q, p = state
    omega = args["omega"]
    return p, -(omega**2) * q


def _decay_problem(y0=2.0, rate=0.5, t_span=(0.0, 2.0)):
    return cpax.IVP(
        system=VectorFieldSystem(_decay),
        y0=jnp.asarray(y0, dtype=jnp.float32),
        t_span=t_span,
        args={"rate": jnp.asarray(rate, dtype=jnp.float32)},
    )


def _oscillator_problem(omega=1.5, y0=(1.0, 0.25), t_span=(0.0, 6.0)):
    return cpax.IVP(
        system=VectorFieldSystem(_oscillator),
        y0=(jnp.asarray(y0[0]), jnp.asarray(y0[1])),
        t_span=t_span,
        args={"omega": jnp.asarray(omega)},
    )


@pytest.mark.parametrize(("y0", "rate"), [(1.0, 0.1), (2.0, 0.5), (3.5, 1.5)])
def test_decay_matches_analytical_trajectory(y0, rate):
    problem = _decay_problem(y0=y0, rate=rate)
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(41),
        rtol=1e-6,
        atol=1e-8,
    )

    expected = y0 * np.exp(-rate * np.asarray(solution.ts))
    np.testing.assert_allclose(solution.ys, expected, rtol=3e-5, atol=3e-6)
    assert bool(solution.is_success)


def test_oscillator_matches_analytical_position_and_momentum():
    problem = _oscillator_problem()
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(121),
        rtol=1e-6,
        atol=1e-8,
    )
    q, p = solution.ys
    omega, q0, p0 = 1.5, 1.0, 0.25
    time = np.asarray(solution.ts)
    expected_q = q0 * np.cos(omega * time) + p0 / omega * np.sin(omega * time)
    expected_p = p0 * np.cos(omega * time) - omega * q0 * np.sin(omega * time)

    np.testing.assert_allclose(q, expected_q, rtol=2e-4, atol=3e-5)
    np.testing.assert_allclose(p, expected_p, rtol=2e-4, atol=3e-5)
    assert jax.tree.structure(solution.ys) == jax.tree.structure(problem.y0)


def test_native_diffrax_reference_uses_identical_problem_and_numerical_options():
    problem = _decay_problem(y0=2.0, rate=0.7)
    save_times = jnp.linspace(*problem.t_span, 31)
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    solver = diffrax.Tsit5()

    expected = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(problem.system.vector_field),
        solver=solver,
        t0=problem.t_span[0],
        t1=problem.t_span[1],
        dt0=0.05,
        y0=problem.y0,
        args=problem.args,
        saveat=diffrax.SaveAt(ts=save_times),
        stepsize_controller=controller,
        max_steps=4096,
    )
    actual = cpax.solve(
        problem,
        solver=solver,
        dt0=0.05,
        stepsize_controller=controller,
        saveat=cpax.SaveAt.linspace(31),
    )

    np.testing.assert_allclose(actual.ts, expected.ts, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual.ys, expected.ys, rtol=2e-6, atol=2e-6)


def test_multidimensional_nested_pytree_state_is_preserved():
    def rhs(t, state, args):
        del t
        return {
            "q": -args["rate"] * state["q"],
            "aux": (-args["rate"] * state["aux"][0], -args["rate"] * state["aux"][1]),
        }

    y0 = {
        "q": jnp.array([1.0, 2.0]),
        "aux": (jnp.array(3.0), jnp.array([[4.0, 5.0]])),
    }
    problem = cpax.IVP(
        system=VectorFieldSystem(rhs),
        y0=y0,
        t_span=(0.0, 1.0),
        args={"rate": jnp.array(0.5)},
    )
    solution = cpax.solve(problem, solver=diffrax.Tsit5(), saveat=cpax.SaveAt.linspace(9))

    assert jax.tree.structure(solution.ys) == jax.tree.structure(y0)
    assert solution.ys["q"].shape == (9, 2)
    assert solution.ys["aux"][0].shape == (9,)
    assert solution.ys["aux"][1].shape == (9, 1, 2)
    np.testing.assert_allclose(solution.ys["q"][-1], np.array([1.0, 2.0]) * np.exp(-0.5), rtol=1e-4)


def test_fixed_step_euler_matches_native_diffrax():
    problem = _decay_problem(t_span=(0.0, 1.0))
    controller = diffrax.ConstantStepSize()
    actual = cpax.solve(
        problem,
        solver=diffrax.Euler(),
        stepsize_controller=controller,
        dt0=0.01,
        saveat=cpax.SaveAt.linspace(11),
    )
    expected = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(problem.system.vector_field),
        solver=diffrax.Euler(),
        t0=problem.t_span[0],
        t1=problem.t_span[1],
        y0=problem.y0,
        args=problem.args,
        dt0=0.01,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 1.0, 11)),
    )

    np.testing.assert_allclose(actual.ys, expected.ys, rtol=2e-6, atol=2e-6)
    assert bool(actual.is_success)


def test_diagnostics_include_backend_identity_and_step_statistics():
    solution = cpax.solve(_decay_problem(), solver=diffrax.Tsit5())

    assert isinstance(solution, Solution)
    assert solution.status.shape == ()
    assert int(solution.status) == SolutionStatus.SUCCESS
    assert bool(solution.is_success)
    assert solution.metadata["backend"] == "diffrax"
    assert solution.metadata["solver"] == "Tsit5"
    assert solution.metadata["stepsize_controller"] == "PIDController"
    assert solution.metadata["backend_result"] == diffrax.RESULTS.successful
    for key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
        assert key in solution.stats
        assert int(solution.stats[key]) >= 0
    assert int(solution.stats["num_accepted_steps"]) > 0


def test_non_raising_failure_preserves_failed_status_and_diagnostics():
    problem = _decay_problem(t_span=(0.0, 10.0))
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        dt0=0.001,
        max_steps=1,
        throw=False,
    )

    assert int(solution.status) == SolutionStatus.FAILED
    assert not bool(solution.is_success)
    assert solution.metadata["backend_result"] != diffrax.RESULTS.successful
    assert int(solution.stats["num_steps"]) == 1
    # Failed final-time saves can be NaN/padded, not valid endpoint observations.
    assert solution.ts.shape == (1,)
    assert solution.ys.shape == (1,)


def test_throw_true_reports_solver_failure():
    with pytest.raises(RuntimeError, match="maximum number of solver steps"):
        cpax.solve(
            _decay_problem(t_span=(0.0, 10.0)),
            solver=diffrax.Tsit5(),
            dt0=0.001,
            max_steps=1,
            throw=True,
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"solver": object()}, TypeError),
        ({"saveat": diffrax.SaveAt(t1=True)}, TypeError),
        ({"max_steps": 0}, ValueError),
        ({"max_steps": True}, ValueError),
        ({"throw": 1}, TypeError),
        ({"dt0": -0.1}, ValueError),
        ({"dt0": 0.0}, ValueError),
        ({"rtol": 0.0}, ValueError),
        ({"atol": -1e-7}, ValueError),
        ({"atol": jnp.array([1e-8])}, ValueError),
        ({"stepsize_controller": object()}, TypeError),
        ({"stepsize_controller": diffrax.ConstantStepSize()}, ValueError),
        ({"stepsize_controller": diffrax.ConstantStepSize(), "rtol": 1e-6}, ValueError),
    ],
)
def test_invalid_solver_configuration_is_rejected(overrides, error):
    kwargs = {"solver": diffrax.Tsit5()} | overrides
    with pytest.raises(error):
        cpax.solve(_decay_problem(), **kwargs)


def test_unsupported_problem_type_is_rejected():
    with pytest.raises(TypeError, match="IVP"):
        cpax.solve(object(), solver=diffrax.Tsit5())


def test_hamiltonian_energy_drift_is_small_over_finite_interval():
    omega = 1.5
    solution = cpax.solve(
        _oscillator_problem(omega=omega, t_span=(0.0, 10.0)),
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(201),
        rtol=1e-6,
        atol=1e-8,
    )
    q, p = solution.ys
    energy = 0.5 * (p**2 + (omega * q) ** 2)
    relative_drift = (energy - energy[0]) / energy[0]

    assert np.max(np.abs(np.asarray(relative_drift))) < 2e-4


def test_complex_state_with_complex_compatible_solver():
    def rhs(t, y, args):
        del t, args
        return 1j * y

    problem = cpax.IVP(
        VectorFieldSystem(rhs),
        jnp.asarray(1.0 + 0.0j),
        (0.0, 1.0),
    )
    solution = cpax.solve(
        problem,
        solver=diffrax.Tsit5(),
        saveat=cpax.SaveAt.linspace(11),
    )

    assert solution.ys.dtype.kind == "c"
    np.testing.assert_allclose(solution.ys[-1], np.exp(1j), rtol=5e-4, atol=5e-4)


def test_float64_decay_preserves_precision_and_matches_analytical_solution():
    """Check a float64 IVP without changing JAX precision for other tests."""
    with jax.enable_x64(True):
        problem = cpax.IVP(
            system=VectorFieldSystem(_decay),
            y0=jnp.asarray(2.0, dtype=jnp.float64),
            t_span=(0.0, 2.0),
            args={"rate": jnp.asarray(0.5, dtype=jnp.float64)},
        )
        solution = cpax.solve(
            problem,
            solver=diffrax.Tsit5(),
            saveat=cpax.SaveAt.linspace(11),
            rtol=1e-10,
            atol=1e-12,
        )
        assert solution.ts.dtype == jnp.float64
        assert solution.ys.dtype == jnp.float64
        assert bool(solution.is_success)
        expected = 2.0 * np.exp(-0.5 * np.asarray(solution.ts))
        np.testing.assert_allclose(solution.ys, expected, rtol=1e-9, atol=1e-10)

