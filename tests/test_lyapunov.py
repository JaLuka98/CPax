"""Analytical and API contracts for Lyapunov-spectrum analysis."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpax
from cpax.analysis import LyapunovDiagnostics, LyapunovQuality, LyapunovResult, lyapunov_spectrum
from cpax.dynamics import VectorFieldSystem
from cpax.mechanics import ForcedNewtonianSystem, PhaseSpaceState
from cpax.results import SolutionStatus
from cpax.solvers import VelocityVerlet


def _diagonal_rhs(t, state, args):
    del t
    return args["rates"] * state


def _diagonal_problem(rates=(-0.3, 0.7), *, t_span=(0.0, 8.0)):
    return cpax.IVP(
        VectorFieldSystem(_diagonal_rhs),
        jnp.ones(len(rates)),
        t_span,
        args={"rates": jnp.asarray(rates)},
    )


def test_analysis_is_a_public_namespace_without_a_root_routine_reexport():
    assert cpax.analysis.lyapunov_spectrum is lyapunov_spectrum
    assert not hasattr(cpax, "lyapunov_spectrum")


def test_full_diagonal_spectrum_and_convergence_history():
    with jax.enable_x64(True):
        result = lyapunov_spectrum(
            _diagonal_problem((-0.3, 0.7)),
            num_segments=16,
            solver=diffrax.Tsit5(),
            rtol=1e-9,
            atol=1e-11,
        )

    assert isinstance(result, LyapunovResult)
    assert bool(result.is_success)
    assert result.exponents.shape == (2,)
    assert result.times.shape == (16,)
    assert result.running_exponents.shape == (16, 2)
    np.testing.assert_allclose(result.exponents, [0.7, -0.3], rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.running_exponents[-1], result.exponents, rtol=0, atol=0)
    np.testing.assert_allclose(result.times, np.linspace(0.5, 8.0, 16), rtol=0, atol=1e-12)
    assert result.metadata["state_dimension"] == 2
    assert result.metadata["num_segments"] == 16
    assert float(result.metadata["qr_interval"]) == pytest.approx(0.5)
    assert int(result.stats["num_accepted_steps"]) > 0
    assert int(result.diagnostics.num_valid_segments) == 16
    assert int(result.diagnostics.quality) == LyapunovQuality.PASS
    assert bool(result.diagnostics.is_healthy)
    np.testing.assert_allclose(result.diagnostics.qr_log_diagonal_spans, 0.5, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.diagnostics.liouville_residuals, 0.0, atol=2e-7)
    assert float(result.diagnostics.liouville_rate_residual) == pytest.approx(0.0, abs=2e-8)
    report = result.report()
    assert "Lyapunov diagnostics: PASS" in report
    assert "Liouville residual rate" in report


def test_qr_conditioning_proxy_warns_before_tangent_precision_is_exhausted():
    result = lyapunov_spectrum(
        _diagonal_problem((-10.0, 10.0), t_span=(0.0, 1.0)),
        num_segments=1,
        rtol=1e-7,
        atol=1e-9,
    )

    assert bool(result.is_success)
    assert int(result.diagnostics.quality) == LyapunovQuality.WARNING
    assert float(result.diagnostics.qr_log_diagonal_spans[0]) > float(result.diagnostics.qr_warning_threshold)
    assert "Lyapunov diagnostics: WARNING" in result.report()


def test_nested_pytree_state_and_ivp_args_are_flattened_correctly():
    def rhs(t, state, args):
        del t
        return {
            "position": args["rates"][0] * state["position"],
            "auxiliary": (
                args["rates"][1] * state["auxiliary"][0],
                args["rates"][2] * state["auxiliary"][1],
            ),
        }

    rates = jnp.asarray([-0.4, 0.2, 0.9])
    initial = {
        "position": jnp.asarray(1.0),
        "auxiliary": (jnp.asarray([1.0]), jnp.asarray([[1.0]])),
    }
    problem = cpax.IVP(VectorFieldSystem(rhs), initial, (0.0, 5.0), args={"rates": rates})
    result = lyapunov_spectrum(problem, num_segments=10, rtol=1e-7, atol=1e-9)

    assert bool(result.is_success)
    np.testing.assert_allclose(result.exponents, [0.9, 0.2, -0.4], rtol=2e-4, atol=2e-5)


def test_mechanics_system_params_define_the_tangent_dynamics():
    def damping(t, q, velocity, params):
        del t, q
        return -params["damping"] * velocity

    # Free damped mechanics has exponents 0 (translation) and -gamma / m.
    # The applied-force callback obtains gamma exclusively from system.params.
    system = ForcedNewtonianSystem(
        mass=1.0,
        applied_force=damping,
        params={"damping": jnp.asarray(3.0)},
    )
    problem = cpax.IVP(system, PhaseSpaceState(1.0, 0.0), (0.0, 12.0))
    result = lyapunov_spectrum(problem, num_segments=24, rtol=1e-7, atol=1e-9)

    assert bool(result.is_success)
    np.testing.assert_allclose(result.exponents, [0.0, -3.0], rtol=3e-4, atol=3e-5)


def test_transient_sets_analysis_window_and_not_parameter_channel():
    result = lyapunov_spectrum(
        _diagonal_problem((0.4,), t_span=(0.0, 6.0)),
        transient_time=2.0,
        num_segments=4,
        rtol=1e-7,
        atol=1e-9,
    )

    assert bool(result.is_success)
    np.testing.assert_allclose(result.times, [3.0, 4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(result.exponents, [0.4], rtol=2e-4, atol=2e-5)
    assert result.metadata["transient_time"] == 2.0
    np.testing.assert_allclose(result.metadata["analysis_time"], 4.0)


def test_jit_and_vmap_support_parameterised_vector_fields():
    system = VectorFieldSystem(lambda t, state, args: args["rate"] * state)

    def spectrum(rate):
        problem = cpax.IVP(system, jnp.asarray(1.0), (0.0, 3.0), args={"rate": rate})
        return lyapunov_spectrum(problem, num_segments=6, rtol=1e-6, atol=1e-8).exponents

    compiled = jax.jit(spectrum)
    np.testing.assert_allclose(compiled(jnp.asarray(-0.5)), [-0.5], rtol=3e-4, atol=3e-5)
    rates = jnp.asarray([-0.5, 0.1, 0.8])
    np.testing.assert_allclose(jax.vmap(compiled)(rates)[:, 0], rates, rtol=4e-4, atol=4e-5)


def test_metadata_reflects_shared_diffrax_option_resolution():
    result = lyapunov_spectrum(
        _diagonal_problem((0.2,)),
        num_segments=4,
        solver="auto",
        rtol=1e-6,
        atol=1e-8,
    )

    assert result.metadata["backend"] == "diffrax"
    assert result.metadata["solver"] == "Tsit5"
    assert result.metadata["solver_selection"] == "auto"
    assert result.metadata["stepsize_controller"] == "PIDController"
    assert float(result.metadata["rtol"]) == pytest.approx(1e-6)
    assert float(result.metadata["atol"]) == pytest.approx(1e-8)


@pytest.mark.parametrize(
    ("problem", "kwargs", "exception"),
    [
        (_diagonal_problem(), {"num_segments": 0}, ValueError),
        (_diagonal_problem(), {"num_segments": True}, ValueError),
        (_diagonal_problem(), {"num_segments": 1.5}, ValueError),
        (_diagonal_problem(), {"num_segments": -1}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": 8.0}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": -1.0}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": float("nan")}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": float("inf")}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": jnp.asarray([0.1])}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": True}, ValueError),
        (_diagonal_problem(), {"num_segments": 2, "transient_time": [0.1, [0.2]]}, TypeError),
        (_diagonal_problem(), {"num_segments": 2, "solver": VelocityVerlet()}, TypeError),
    ],
)
def test_invalid_analysis_configuration_is_rejected(problem, kwargs, exception):
    with pytest.raises(exception):
        lyapunov_spectrum(problem, **kwargs)


def test_traced_transient_time_is_rejected():
    def spectrum(transient_time):
        return lyapunov_spectrum(_diagonal_problem(), transient_time=transient_time, num_segments=2).exponents

    with pytest.raises(TypeError, match="concrete"):
        jax.jit(spectrum)(1.0)


def test_empty_state_is_rejected():
    problem = cpax.IVP(VectorFieldSystem(lambda t, y, args: -y), jnp.zeros(0), (0.0, 1.0))
    with pytest.raises(ValueError, match="nonempty"):
        lyapunov_spectrum(problem, num_segments=2)


def test_complex_state_and_non_ivp_are_rejected():
    complex_problem = cpax.IVP(
        VectorFieldSystem(lambda t, y, args: 1j * y),
        jnp.asarray(1.0 + 0.0j),
        (0.0, 1.0),
    )
    with pytest.raises(TypeError, match="real"):
        lyapunov_spectrum(complex_problem, num_segments=2)
    with pytest.raises(TypeError, match="IVP"):
        lyapunov_spectrum(object(), num_segments=2)


def test_complex_derivative_reports_dtype_error():
    problem = cpax.IVP(
        VectorFieldSystem(lambda t, y, args: (1 + 0j) * y),
        jnp.ones(2),
        (0.0, 1.0),
    )
    with pytest.raises(ValueError, match="real floating derivatives") as excinfo:
        lyapunov_spectrum(problem, num_segments=2)
    assert "differentiable" not in str(excinfo.value)


def test_non_differentiable_vector_field_is_rejected():
    @jax.custom_vjp
    def decay(y):
        return -y

    decay.defvjp(lambda y: (-y, None), lambda _, cotangent: (-cotangent,))
    problem = cpax.IVP(
        VectorFieldSystem(lambda t, y, args: decay(y)),
        jnp.ones(2),
        (0.0, 1.0),
    )
    with pytest.raises(ValueError, match="JAX-differentiable"):
        lyapunov_spectrum(problem, num_segments=2)


def test_nonraising_failure_marks_invalid_spectrum_and_preserves_diagnostics():
    result = lyapunov_spectrum(
        _diagonal_problem((0.2,), t_span=(0.0, 4.0)),
        num_segments=4,
        solver=diffrax.Euler(),
        dt0=0.01,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=1,
        throw=False,
    )

    assert int(result.status) == SolutionStatus.FAILED
    assert not bool(result.is_success)
    assert bool(jnp.all(jnp.isnan(result.exponents)))
    assert bool(jnp.all(jnp.isnan(result.running_exponents)))
    assert int(result.stats["num_steps"]) == 1
    assert int(result.diagnostics.num_valid_segments) == 0
    assert int(result.diagnostics.quality) == LyapunovQuality.INVALID
    assert "spectrum is not usable" in result.report()


def test_raising_failure_matches_solve_contract():
    with pytest.raises(RuntimeError, match="maximum number of solver steps"):
        lyapunov_spectrum(
            _diagonal_problem((0.2,), t_span=(0.0, 4.0)),
            num_segments=4,
            solver=diffrax.Euler(),
            dt0=0.01,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=1,
        )


def _diagnostics(**overrides):
    fields = {
        "qr_log_diagonal_spans": jnp.zeros(3),
        "liouville_residuals": jnp.zeros(3),
        "tail_drift": jnp.zeros(2),
        "num_valid_segments": 3,
        "precision_budget": jnp.asarray(36.0),
        "qr_warning_threshold": jnp.asarray(18.0),
        "liouville_rate_residual": jnp.asarray(0.0),
        "liouville_rate_tolerance": jnp.asarray(1e-3),
        "quality": int(LyapunovQuality.PASS),
    }
    fields.update(overrides)
    return LyapunovDiagnostics(**fields)


def _result_fields(**overrides):
    fields = {
        "exponents": jnp.asarray([0.5, -0.5]),
        "times": jnp.asarray([1.0, 2.0, 3.0]),
        "running_exponents": jnp.zeros((3, 2)),
        "status": int(SolutionStatus.SUCCESS),
        "diagnostics": _diagnostics(),
    }
    fields.update(overrides)
    return fields


def test_result_defaults_missing_diagnostics_to_invalid():
    result = LyapunovResult(**_result_fields(times=jnp.asarray([1, 2, 3]), diagnostics=None))

    assert result.times.dtype.kind == "f"
    assert result.status.shape == ()
    diagnostics = result.diagnostics
    assert diagnostics.qr_log_diagonal_spans.shape == (3,)
    assert diagnostics.tail_drift.shape == (2,)
    assert bool(jnp.all(jnp.isnan(diagnostics.liouville_residuals)))
    assert int(diagnostics.num_valid_segments) == 0
    assert int(diagnostics.quality) == LyapunovQuality.INVALID


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        ({"exponents": jnp.zeros((2, 1))}, ValueError, r"^exponents"),
        ({"exponents": jnp.zeros(0), "running_exponents": jnp.zeros((3, 0))}, ValueError, r"^exponents"),
        ({"exponents": jnp.asarray([1, -1])}, ValueError, r"^exponents"),
        ({"times": jnp.ones((3, 1))}, ValueError, r"^times"),
        ({"times": jnp.zeros(0)}, ValueError, r"^times"),
        ({"times": jnp.asarray([1.0, 2.0, 3.0]) * 1j}, ValueError, r"^times"),
        ({"running_exponents": jnp.zeros((4, 2))}, ValueError, r"^running_exponents"),
        ({"running_exponents": jnp.zeros((3, 3))}, ValueError, r"^running_exponents"),
        ({"running_exponents": jnp.zeros((3, 2), dtype=jnp.int32)}, ValueError, r"^running_exponents"),
        ({"status": 0.0}, TypeError, r"^status"),
        ({"status": jnp.asarray([0])}, TypeError, r"^status"),
        ({"status": 7}, ValueError, r"^status"),
        ({"diagnostics": object()}, TypeError, r"^diagnostics"),
    ],
)
def test_result_rejects_invalid_fields(overrides, exception, match):
    with pytest.raises(exception, match=match):
        LyapunovResult(**_result_fields(**overrides))


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        ({"qr_log_diagonal_spans": jnp.zeros(4)}, ValueError, "qr_log_diagonal_spans"),
        ({"liouville_residuals": jnp.zeros(4)}, ValueError, "liouville_residuals"),
        ({"tail_drift": jnp.zeros(3)}, ValueError, "tail_drift"),
        ({"precision_budget": jnp.zeros(2)}, ValueError, "precision_budget"),
        ({"liouville_rate_tolerance": jnp.asarray(1)}, ValueError, "liouville_rate_tolerance"),
        ({"num_valid_segments": 1.0}, TypeError, "num_valid_segments"),
        ({"num_valid_segments": -1}, ValueError, "num_valid_segments"),
        ({"num_valid_segments": 4}, ValueError, "num_valid_segments"),
        ({"quality": 0.0}, TypeError, "quality"),
        ({"quality": 9}, ValueError, "quality"),
    ],
)
def test_result_rejects_invalid_diagnostics(overrides, exception, match):
    with pytest.raises(exception, match=rf"^diagnostics\.{match}"):
        LyapunovResult(**_result_fields(diagnostics=_diagnostics(**overrides)))


def test_result_construction_under_jit():
    def build(status, count, quality):
        diagnostics = _diagnostics(num_valid_segments=count, quality=quality)
        return LyapunovResult(**_result_fields(status=status, diagnostics=diagnostics)).is_success

    assert bool(jax.jit(build)(jnp.asarray(0), jnp.asarray(3), jnp.asarray(0)))

    def mismatched(exponents):
        return LyapunovResult(**_result_fields(exponents=exponents)).exponents

    with pytest.raises(ValueError, match=r"^running_exponents"):
        jax.jit(mismatched)(jnp.zeros(3))
