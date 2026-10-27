# Explicit conservative-force investigation (issue #72)

This is an **experimental implementation**, not a decision to stabilise the
`NewtonianSystem(force_fn=...)` API. The default `force_fn=None` continues
to use `-jax.grad(V)`. The optional callback is supplied as `force_fn(q)`
or `force_fn(q, params)`, consistent with existing parameter semantics.

## Physical contract

For every physically accessible configuration `q`, analytical callbacks
must agree with `-grad(V)(q)`. Shape and dtype checks cannot establish that
identity. Local agreement at test points cannot prove it either. The relevant
energy-balance identity is

```text
d(T+V)/dt = velocity · (F_explicit + grad(V)).
```

If the explicit force is the gradient of another potential, Velocity Verlet
may remain symplectic for a *different* Hamiltonian while CPax reports energy
for the supplied one. Genuinely non-conservative position-dependent forces
do not satisfy the usual separable-Hamiltonian symplectic guarantee. This
prototype deliberately does not perform expensive/inconclusive consistency
checks inside compiled dynamics. For arbitrary applied forces, use
`ForcedNewtonianSystem` instead.

## Reproduce

```console
uv sync
uv run pytest tests/test_explicit_force.py
uv run python benchmarks/force_comparison.py --output force-f32.json
uv run python benchmarks/force_comparison.py --precision float64 --output force-f64.json
uv run python benchmarks/force_comparison.py --integration-steps 100 --output integrated.json
```

To investigate CPU versus GPU, use an appropriately installed JAX backend.
Each workload's AD and analytical implementations compute the same potential
and its gradient without collision softening. Gravity is an all-pairs
`O(N²)` reference workload, **not** an optimised neighbour-list method.
Benchmarks reset JAX caches and report repeated cold compilation and warmed
synchronous evaluation independently. These are dispatch-inclusive
microbenchmarks, not pure device-kernel timings. The script records JAX
versions, Python version, backend, devices, compiler allocation estimates
(if supported), error norms, and timing variability. Compiler memory
estimates do not equal peak process RSS or peak device residency.

## Preliminary single-machine CPU results

These exploratory measurements used Linux x86-64, a CPU-only backend,
JAX 0.9.0.1, three trials and 25 warmed evaluations per trial. These values
have **not** been reproduced on project CI, an accelerator or an installed
CPax integration environment. They are not release performance claims.

| Workload | float32 AD / direct (µs) | float64 AD / direct (µs) |
| --- | ---: | ---: |
| Harmonic (3 DOF) | 4.6 / 4.5 | 4.6 / 4.6 |
| Periodic quartic (8 DOF) | 4.7 / 4.5 | 4.9 / 4.5 |
| Gravity 8×3 | 7.3 / 7.7 | 7.5 / 7.5 |
| Gravity 32×3 | 24.0 / 8.9 | 27.9 / 16.7 |
| Gravity 128×3 | 72.1 / 59.5 | 102.8 / 80.3 |

A 32-body case showed a potentially interesting analytical-force speedup,
but the results are neither monotonic in system size nor universally
favourable. For 128 bodies, the reported compiler temporary allocations
were 327,680 versus 262,144 bytes (float32) and 655,360 versus 524,288
bytes (float64), AD versus direct. The 32-body compiler allocation estimates
were 8,192 versus 4,096 bytes (float32) and 16,384 versus 8,192 bytes
(float64). These are workload- and compiler-specific estimates.

**Decision: pending.** Do not merge an experimental public API on the
strength of these measurements. Before a go decision, run full CPax
integration benchmarks in a representative development environment,
confirm JAX transformations and existing CI, repeat hardware measurements,
and assess whether a more efficient potential implementation removes
part of the observed gap. A follow-up API issue, if justified, should keep
the potential mandatory and explain the consistency responsibility.
