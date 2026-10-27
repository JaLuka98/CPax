"""Benchmark JAX AD and analytical conservative forces (CPax issue #72).

Usage example (from repository root):
    uv run force_comparison.py --sizes 8 32 128 256 1024 2056 --precision float64 --output results.json

A speedup factor > 1 means the *analytical* force is faster than AD.
Results are platform dependent. No timing thresholds are used in CI.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from .force_models import ForceCase, ad_force, coupled_quartic, gravity, harmonic
else:
    from force_models import ForceCase, ad_force, coupled_quartic, gravity, harmonic


def _memory_estimate(executable: Any) -> dict[str, int] | None:
    """Compiler allocation estimates, not observed peak device residency."""
    try:
        estimate = executable.memory_analysis()
    except (AttributeError, NotImplementedError, RuntimeError):
        return None
    if estimate is None:
        return None
    names = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    return {name: int(getattr(estimate, name)) for name in names if getattr(estimate, name, None) is not None}


def _measure(function: Any, q: jax.Array, *, trials: int, repetitions: int) -> dict[str, Any]:
    compilation_ms = []
    execution_us = []
    memory = None
    for _ in range(trials):
        jax.clear_caches()
        started = time.perf_counter()
        executable = jax.jit(function).lower(q).compile()
        compilation_ms.append(1e3 * (time.perf_counter() - started))
        memory = _memory_estimate(executable)
        jax.block_until_ready(executable(q))
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            jax.block_until_ready(executable(q))
            samples.append(1e6 * (time.perf_counter() - started))
        execution_us.append(statistics.median(samples))
    return {
        "compile_ms_median": statistics.median(compilation_ms),
        "compile_ms_range": [min(compilation_ms), max(compilation_ms)],
        "evaluation_us_median": statistics.median(execution_us),
        "evaluation_us_range": [min(execution_us), max(execution_us)],
        "compiler_memory_bytes": memory,
    }


def _speedup(ad_time: float, analytical_time: float) -> float:
    """Return analytical performance relative to AD; >1 favours analytical."""
    return ad_time / analytical_time


def _accuracy_tolerances(dtype: jnp.dtype) -> tuple[float, float]:
    return (5e-5, 5e-6) if dtype == jnp.float32 else (1e-9, 1e-10)


def _compare(case: ForceCase, *, trials: int, repetitions: int) -> dict[str, Any]:
    baseline, direct = ad_force(case), case.analytical_force
    ad_output, direct_output = jax.jit(baseline)(case.q), jax.jit(direct)(case.q)
    jax.block_until_ready((ad_output, direct_output))
    relative = float(jnp.linalg.norm(ad_output - direct_output) / (jnp.linalg.norm(ad_output) + 1e-12))
    maximum = float(jnp.max(jnp.abs(ad_output - direct_output)))
    rtol, atol = _accuracy_tolerances(case.q.dtype)
    np.testing.assert_allclose(direct_output, ad_output, rtol=rtol, atol=atol)
    # Agreement at only one configuration cannot establish consistency.
    for offset in (0.01, -0.02):
        displaced = case.q + jnp.asarray(offset, case.q.dtype) * jnp.cos(case.q)
        np.testing.assert_allclose(direct(displaced), baseline(displaced), rtol=rtol, atol=atol)
    ad_result = _measure(baseline, case.q, trials=trials, repetitions=repetitions)
    direct_result = _measure(direct, case.q, trials=trials, repetitions=repetitions)
    return {
        "case": case.name,
        "shape": list(case.q.shape),
        "dtype": str(case.q.dtype),
        "maximum_force_difference": maximum,
        "relative_force_l2_difference": relative,
        "ad": ad_result,
        "analytical": direct_result,
        "analytical_speedup_factor_vs_ad": _speedup(
            ad_result["evaluation_us_median"], direct_result["evaluation_us_median"]
        ),
    }


def _make_trajectory(system: Any, p0: jax.Array, *, steps: int, dt: float) -> Any:
    """Return both final position and momentum from a compiled CPax solve."""
    import cpax
    from cpax.mechanics import PhaseSpaceState
    from cpax.solvers import VelocityVerlet

    def trajectory(q: jax.Array) -> Any:
        problem = cpax.IVP(system, PhaseSpaceState(q, p0), (0.0, steps * dt))
        return cpax.solve(problem, solver=VelocityVerlet(), dt0=dt, max_steps=steps + 2).ys

    return trajectory


def _integrate(case: ForceCase, *, steps: int, dt: float, trials: int, repetitions: int) -> dict[str, Any]:
    """Time full Velocity Verlet solves and check trajectory/energy agreement."""
    from cpax.mechanics import NewtonianSystem, PhaseSpaceState

    initial = PhaseSpaceState(case.q, jnp.zeros_like(case.q))
    systems = {
        "ad": NewtonianSystem(potential=case.potential, mass=case.mass),
        "analytical": NewtonianSystem(potential=case.potential, mass=case.mass, force_fn=case.analytical_force),
    }
    results: dict[str, Any] = {}
    finals: dict[str, Any] = {}
    for label, system in systems.items():
        trajectory = _make_trajectory(system, initial.p, steps=steps, dt=dt)
        finals[label] = jax.jit(trajectory)(case.q)
        jax.block_until_ready(finals[label])
        results[label] = _measure(trajectory, case.q, trials=trials, repetitions=repetitions)

    ad_final, direct_final = finals["ad"], finals["analytical"]
    rtol, atol = _accuracy_tolerances(case.q.dtype)
    np.testing.assert_allclose(ad_final.q, direct_final.q, rtol=rtol, atol=atol)
    np.testing.assert_allclose(ad_final.p, direct_final.p, rtol=rtol, atol=atol)
    state_ad = PhaseSpaceState(ad_final.q[-1], ad_final.p[-1])
    state_direct = PhaseSpaceState(direct_final.q[-1], direct_final.p[-1])
    energy_initial = float(systems["ad"].energy(initial))
    energy_ad = float(systems["ad"].energy(state_ad))
    energy_direct = float(systems["analytical"].energy(state_direct))
    np.testing.assert_allclose(energy_ad, energy_direct, rtol=rtol, atol=atol)
    energy_scale = max(abs(energy_initial), 1.0)
    validation = {
        "max_absolute_position_difference": float(jnp.max(jnp.abs(state_ad.q - state_direct.q))),
        "max_absolute_momentum_difference": float(jnp.max(jnp.abs(state_ad.p - state_direct.p))),
        "initial_energy": energy_initial,
        "final_energy_ad": energy_ad,
        "final_energy_analytical": energy_direct,
        "relative_energy_change_ad": (energy_ad - energy_initial) / energy_scale,
        "relative_energy_change_analytical": (energy_direct - energy_initial) / energy_scale,
    }
    if case.name == "gravity":
        validation["final_net_momentum_norm_ad"] = float(jnp.linalg.norm(jnp.sum(state_ad.p, axis=0)))
        validation["final_net_momentum_norm_analytical"] = float(jnp.linalg.norm(jnp.sum(state_direct.p, axis=0)))
    return {
        "steps": steps,
        "step_size": dt,
        "validation": validation,
        "ad": results["ad"],
        "analytical": results["analytical"],
        "analytical_speedup_factor_vs_ad": _speedup(
            results["ad"]["evaluation_us_median"], results["analytical"]["evaluation_us_median"]
        ),
    }


def _validate_parameter_gradients(case: ForceCase, *, steps: int, dt: float) -> dict[str, Any]:
    """Check jit/vmap/grad for a physical interaction-strength parameter."""
    from cpax.mechanics import NewtonianSystem

    def observable(explicit: bool) -> Any:
        def evaluate(scale: jax.Array) -> jax.Array:
            def potential(q: jax.Array, factor: jax.Array) -> jax.Array:
                return factor * case.potential(q)

            def scaled_force(q: jax.Array, factor: jax.Array) -> jax.Array:
                return factor * case.analytical_force(q)

            system = NewtonianSystem(
                potential=potential,
                mass=case.mass,
                params=scale,
                force_fn=scaled_force if explicit else None,
            )
            final = _make_trajectory(system, jnp.zeros_like(case.q), steps=steps, dt=dt)(case.q)
            return jnp.sum(final.q[-1] * case.q) / case.q.size

        return evaluate

    baseline, direct = observable(False), observable(True)
    scale = jnp.asarray(1.0, dtype=case.q.dtype)
    scales = jnp.asarray([0.9, 1.0, 1.1], dtype=case.q.dtype)
    ad_values = jax.jit(jax.vmap(baseline))(scales)
    direct_values = jax.jit(jax.vmap(direct))(scales)
    ad_grad = jax.jit(jax.grad(baseline))(scale)
    direct_grad = jax.jit(jax.grad(direct))(scale)
    rtol, atol = _accuracy_tolerances(case.q.dtype)
    np.testing.assert_allclose(ad_values, direct_values, rtol=rtol, atol=atol)
    np.testing.assert_allclose(ad_grad, direct_grad, rtol=rtol, atol=atol)
    h = jnp.asarray(1e-2 if case.q.dtype == jnp.float32 else 1e-4, dtype=case.q.dtype)
    finite_difference = (baseline(scale + h) - baseline(scale - h)) / (2 * h)
    np.testing.assert_allclose(
        ad_grad,
        finite_difference,
        rtol=0.05 if case.q.dtype == jnp.float32 else 0.001,
        atol=1e-4 if case.q.dtype == jnp.float32 else 1e-7,
    )
    return {
        "system": case.name,
        "shape": list(case.q.shape),
        "parameter": "multiplicative potential/force strength",
        "steps": steps,
        "parameter_values": [float(x) for x in scales],
        "observable_ad": [float(x) for x in ad_values],
        "observable_analytical": [float(x) for x in direct_values],
        "gradient_ad": float(ad_grad),
        "gradient_analytical": float(direct_grad),
        "gradient_central_difference": float(finite_difference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument(
        "--integration-steps", type=int, default=0, help="0 skips full integrations and parameter-gradient validation"
    )
    parser.add_argument(
        "--integration-sizes",
        nargs="+",
        type=int,
        default=[128, 256, 1024],
        help="Particle counts to integrate if included in --sizes",
    )
    parser.add_argument("--integration-dt", type=float, default=0.001)
    parser.add_argument(
        "--gradient-validation-size",
        type=int,
        default=8,
        help="Separate small gravity system for JAX parameter-gradient checks",
    )
    parser.add_argument("--output", type=Path, help="Write machine-readable JSON results")
    args = parser.parse_args()
    if (
        min(args.trials, args.repetitions) <= 0
        or min(args.sizes) < 2
        or min(args.integration_sizes) < 2
        or args.integration_steps < 0
        or args.gradient_validation_size < 2
        or not np.isfinite(args.integration_dt)
        or args.integration_dt <= 0
    ):
        parser.error("Trials/repetitions must be positive; sizes >=2; integration steps >=0.")
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    cases = [harmonic(dtype=dtype), coupled_quartic(size=max(8, args.sizes[0]), dtype=dtype)]
    cases += [gravity(size=n, dtype=dtype) for n in args.sizes]
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "speedup_definition": (
            "analytical_speedup_factor_vs_ad = AD runtime / analytical runtime; >1 favours analytical"
        ),
        "note": "Times include dispatch and synchronisation; compiler memory is not peak RSS/device memory.",
        "cases": [],
    }
    for case in cases:
        result = _compare(case, trials=args.trials, repetitions=args.repetitions)
        if args.integration_steps and case.name == "gravity" and case.q.shape[0] in args.integration_sizes:
            result["integration"] = _integrate(
                case,
                steps=args.integration_steps,
                dt=args.integration_dt,
                trials=args.trials,
                repetitions=max(3, args.repetitions // 5),
            )
        report["cases"].append(result)
        print(
            f"{case.name} {list(case.q.shape)}: "
            f"AD {result['ad']['evaluation_us_median']:.1f} us, "
            f"analytical {result['analytical']['evaluation_us_median']:.1f} us, "
            f"analytical speedup {result['analytical_speedup_factor_vs_ad']:.2f}x"
        )
        if "integration" in result:
            integration = result["integration"]
            print(
                f"  {args.integration_steps}-step Velocity Verlet: "
                f"AD {integration['ad']['evaluation_us_median']:.1f} us, "
                f"analytical {integration['analytical']['evaluation_us_median']:.1f} us, "
                f"analytical speedup {integration['analytical_speedup_factor_vs_ad']:.2f}x"
            )
    if args.integration_steps:
        validation_case = gravity(size=args.gradient_validation_size, dtype=dtype)
        report["parameter_gradient_validation"] = _validate_parameter_gradients(
            validation_case, steps=min(args.integration_steps, 20), dt=args.integration_dt
        )
        print(f"Parameter gradient validation passed for gravity N={args.gradient_validation_size}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
