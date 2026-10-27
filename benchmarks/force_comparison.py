"""Reproducible JAX force comparison for CPax issue #72.

Usage example (from repository root):
    uv run force_comparison.py --sizes 8 32 128 256 1024 2056 --precision float64 --output results.json

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
        executable(q).block_until_ready()
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            executable(q).block_until_ready()
            samples.append(1e6 * (time.perf_counter() - started))
        execution_us.append(statistics.median(samples))
    return {
        "compile_ms_median": statistics.median(compilation_ms),
        "compile_ms_range": [min(compilation_ms), max(compilation_ms)],
        "evaluation_us_median": statistics.median(execution_us),
        "evaluation_us_range": [min(execution_us), max(execution_us)],
        "compiler_memory_bytes": memory,
    }


def _compare(case: ForceCase, *, trials: int, repetitions: int) -> dict[str, Any]:
    baseline, direct = ad_force(case), case.analytical_force
    ad_output, direct_output = jax.jit(baseline)(case.q), jax.jit(direct)(case.q)
    ad_output.block_until_ready()
    direct_output.block_until_ready()
    relative = float(jnp.linalg.norm(ad_output - direct_output) / (jnp.linalg.norm(ad_output) + 1e-12))
    maximum = float(jnp.max(jnp.abs(ad_output - direct_output)))
    rtol, atol = (5e-5, 5e-6) if case.q.dtype == jnp.float32 else (1e-11, 1e-11)
    np.testing.assert_allclose(direct_output, ad_output, rtol=rtol, atol=atol)
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
        "evaluation_speedup_ad_over_analytical": ad_result["evaluation_us_median"]
        / direct_result["evaluation_us_median"],
    }


def _integrate(case: ForceCase, *, steps: int, dt: float, trials: int, repetitions: int) -> dict[str, Any]:
    """Compare full compiled CPax Velocity Verlet integrations."""
    import cpax
    from cpax.mechanics import NewtonianSystem, PhaseSpaceState
    from cpax.solvers import VelocityVerlet

    state = PhaseSpaceState(case.q, jnp.zeros_like(case.q))
    results = {}
    outputs = {}
    for label, force_fn in (("ad", None), ("analytical", case.analytical_force)):
        system = NewtonianSystem(
            potential=case.potential,
            mass=case.mass,
            force_fn=force_fn,
        )

        def trajectory(q, system=system):
            problem = cpax.IVP(system, PhaseSpaceState(q, state.p), (0.0, dt * steps))
            return cpax.solve(problem, solver=VelocityVerlet(), dt0=dt, max_steps=steps + 1).ys.q[-1]

        outputs[label] = jax.jit(trajectory)(case.q)
        outputs[label].block_until_ready()
        results[label] = _measure(trajectory, case.q, trials=trials, repetitions=repetitions)
    error = float(jnp.linalg.norm(outputs["ad"] - outputs["analytical"]))
    return {"steps": steps, "step_size": dt, "final_position_l2_difference": error, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--integration-steps", type=int, default=0, help="0 skips CPax integration timings")
    parser.add_argument("--output", type=Path, help="Write machine-readable JSON results")
    args = parser.parse_args()
    if min(args.trials, args.repetitions) <= 0 or min(args.sizes) < 2 or args.integration_steps < 0:
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
        "note": "Times include dispatch and synchronisation; compiler memory is not peak RSS/device memory.",
        "cases": [],
    }
    for case in cases:
        result = _compare(case, trials=args.trials, repetitions=args.repetitions)
        if args.integration_steps and case.name == "gravity" and case.q.shape[0] <= 32:
            result["integration"] = _integrate(
                case,
                steps=args.integration_steps,
                dt=0.001,
                trials=args.trials,
                repetitions=max(3, args.repetitions // 5),
            )
        report["cases"].append(result)
        print(
            f"{case.name} {list(case.q.shape)}: "
            f"AD {result['ad']['evaluation_us_median']:.1f} us, "
            f"analytical {result['analytical']['evaluation_us_median']:.1f} us, "
            f"ratio {result['evaluation_speedup_ad_over_analytical']:.2f}x"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
