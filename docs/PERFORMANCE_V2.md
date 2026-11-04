# Result-preserving performance v2

## Scope and execution ledger

Specification: November 4, 2026 user request, P1–P5. Baseline HEAD is exactly
`aaa1a77a89a76c6f2bd739b956649c4c94d59011`; no reference adaptation is needed.
Work is on `feat/perfomance-optimizations-v2`. The previous stash is untouched.
No applicable AGENTS.md exists. CONTRIBUTING.md, pyproject.toml and CI were read.

The integrator owns source changes and commits. An independent worker owns only
the throughput harness and its test. Changes are integrated in this order:

- [x] Inspect checkout; establish environment and baseline tests.
- [ ] Baseline harness: immutable source snapshot, identical seeded inputs,
  fresh output directories, cold sample and five warm repetitions, raw timings,
  memory/identity/version/config metadata. Keep full evidence outside checkout.
- [x] P1 `coherence_pipeline`: `_sample_arclength_pairs(s, fid)` selects integer
  ranks using subtraction-preserving vectorized binary search and int64 prefixes.
  Test dense-oracle exact parity, boundary floats, multiple runs and 100k stress;
  measure 800/2000/4000 samples. Leave whole_reference unchanged.
- [x] P2 `field_solver`: retain residual norm between iterations, invalidate
  on true-residual restart. Test iteration/failure parity and norm call counts.
- [ ] P3 `block_solver`, `field_solver`: explicit `prepared_cache_bytes=0`,
  stable admission, conservative retained-storage accounting and live rebind.
  Test partial/full/off, mutated observations, exterior targets, callable
  providers, failure cleanup and cold STOP/resume with a changed cache budget.
- [ ] P4 `anisotropic_reference`: first-visit stable partial stencil admission;
  test exact off/partial/full equality including NaNs and avoided constructions.
- [ ] P5 `surface_mesh`: one preparation per tile across levels, independent
  preparation for recursive crops; test exact geometry/keys and regression suite.
- [ ] Final measurements, CI gates, quality demo, distribution checks and review.

Each change gets a failing work-avoidance test before implementation, existing
and new focused checks, individual measurements, and a separate local commit.
No timing threshold enters CI. No cache can retain mutable exterior values.
All mathematical parameters, observation selection, multiplicities, consensus
exterior rows, physical transforms, residual gates, artifact identity/durability,
quantization and local/global/physical acceptance distinctions remain unchanged.

Review focus: float subtraction at eligibility boundaries; restart and maxiter
failure semantics; NumPy aliases bypassing Tensor counters; partial caches under
sequential scans; recursive ambiguous-cell mesh fallback with float64 values.

## Environment and baseline

The first clean Python 3.12 installation failed while unpacking Torch because
the volume had only about 422 MiB free. No datasets or pre-existing caches were
deleted. A separate `.venv-perf-v2-cpu` uses the installed Python 3.11.13 scientific
dependencies read-only via system-site-packages, with local Ruff 0.17.3 and Mypy
2.4.0. This is not a hermetic installation; baseline and candidate share exactly
the same runtime. NumPy 2.2.6, SciPy 1.16.1, Torch 2.12.1, scikit-image 0.26.0.
CUDA is unavailable; no CUDA or real-data validation is claimed.

Baseline targeted command (prefix with `.venv-perf-v2-cpu/bin/python -m`):

```sh
pytest -q tests/test_coherence_pipeline.py tests/test_field_constraints.py \
  tests/test_field_solver.py tests/test_solver_performance.py \
  tests/test_block_solver.py tests/test_constraint_templates.py \
  tests/test_anisotropic_reference.py tests/test_surface_mesh.py \
  tests/test_surface_regressions.py tests/test_performance_regressions.py
```

Result before source edits: **275 passed in 12.96 s**.

## P1: row-major rank selection

Vectorized binary search is O(m log m) time and O(m) memory. It evaluates the
original subtraction, not an algebraically rearranged threshold. Prefix sums
are int64 with a Python-integer upper-bound overflow check before allocation.
The RNG is reset per run and never invoked for populations of at most eight.
`whole_reference.global_fibre_constraints` is deliberately unchanged: its
deduplicated nodal population and once-per-fibre sampling are different contracts.

RED: 13 new selector tests failed before implementation. GREEN: 33 selector,
coherence and constraint tests passed. Exact comparisons include audits, CSR
arrays, weights, targets, groups, trace and digests, plus backtracking/multiple
runs. The 100,000-sample test covers 4,999,950,000 candidates without enumeration.
Initial five-repeat selector microbenchmarks (CPU; median / IQR milliseconds):

| Samples | Legacy | Candidate | Legacy / candidate traced peak bytes |
|---:|---:|---:|---:|
| 800 | 1.522 / 0.479 | 0.159 / 0.008 | 10,227,640 / 48,177 |
| 2,000 | 9.276 / 0.093 | 0.258 / 0.006 | 63,968,440 / 118,977 |
| 4,000 | 36.015 / 0.459 | 0.425 / 0.014 | 255,936,440 / 236,977 |

These are selector-only exploratory timings, not pipeline speedups. The final
harness records full source/environment/input metadata and raw repeated samples.

## P2: residual-norm reuse

Only the redundant reduction is removed. A restart invalidates the cached norm;
the refresh still occurs at the next loop entry, preserving maxiter/failure order.
The final independent true-residual reduction and float64 norm accumulation stay.
RED: three work-count tests failed (12 rather than 8/9 norm evaluations). GREEN:
115 solver/constraint/performance tests passed, including 0/1 iterations, a forced
restart, both dtypes and an independently assembled final residual.

Exploratory seeded 16³ float64 CPU solve, one thread, rtol=1e-8, five warm
repetitions: baseline median 13.115 ms (IQR 0.850), candidate 12.696 ms
(IQR 0.115). Both take 159 iterations, residual 9.543514691238635e-9 and exactly
equal fields. This small timing difference alone is not a pipeline speedup;
the deterministic reduction-count test establishes the avoided work.
