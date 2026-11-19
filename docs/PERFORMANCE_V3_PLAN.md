# Solver performance v3 execution ledger

Approved scope: A–E from the user specification, 2026-11-04.
Core baseline: `9a7999f78bbf4e96e22229e09c497db2eb952767`.
CLI committed baseline: `19b6662302df8ca3b8a6512692cdc9b5923fe864`.
Both branches: `feat/perf-optimizations-v3`; CLI uses an isolated worktree.
Original CLI dirty checkout, core stash and Lean branch/cache are preserved.

## Ownership and gates

- [x] Numerical worker: field_solver/field_constraints, spectral implementation,
  exact CPU consensus tests and independent numerical tests.
- [x] Storage worker: constraint_templates/field_store, spatial/gather/stable
  caches and durable commit tests.
- [x] CLI worker: isolated CLI config/forwarding/docs/integration tests.
- [x] Integrator: block_solver nonadmission/STATUS/anchor telemetry and tests.
- [x] Integrator: balanced baseline/A/spectral/combined and template/I/O benchmarks.
- [x] Independent review, complete repository gates, docs and local commits.

Workers own disjoint files and do not commit or benchmark concurrently. Parent
owns benchmarks/performance and this ledger. Tests may run during implementation;
measurements start only after all workers stop. No parallel block solves.

## Contracts

Structural optimizations preserve exact deterministic results and accumulation
order. Spectral preserves A, RHS, anchors and true residual criteria, not iteration
count/last decimals. CPU float64 spectral supports effective point gauge, exactly
all six outer faces, or no free nodes. Unsupported inputs fail explicitly.
All caches are bounded, live targets/boundaries refresh, and resume stays cold.
Durability, ownership, hashes, STOP checks and local/global/physical distinctions
remain authoritative. Ordinary callable constraints remain unprepared.

## Evidence

Fresh external evidence root: `../../artifacts/perf-v3-20261104/`.
The baseline is a source archive of the recorded core SHA, not a reverted checkout.
Current interpreter: `.venv-perf-v2-cpu/bin/python`, Python 3.11.13, NumPy 2.2.6,
SciPy 1.16.1, Torch 2.12.1; imported core verified against this checkout.
Historical Linux decimals are not treated as local expected results.

## Completed evidence

Independent numerical/storage/runtime/CLI and harness reviews completed. Fixed
Fortran gather copying and redundant resume manifest parsing; tightened benchmark
metadata comparisons, memory labels and frozen harness/source identity checks.
Final balanced eight-repeat run is `final-balanced/`; smoke samples are excluded.
Dense cross-block group indexing loses (~39%) and remains optional/off by default.

Core coverage: 1442 passed, 92.78%; final focused v3 gate: 147 passed.
Core format/lint/mypy pass. Wheel/sdist inventory and installed minimal examples
pass; offline benchmark demo passes. Matching core/CLI dev2 wheels installed and
`pip check` passes. CLI installed full suite: 1396 passed / 10 skipped; final
packaged-provenance integration gate: 89 passed. Details are in the final guide.
CLI touched files retain their baseline 89 lint findings and seven format failures;
the full-tree checks report 1641 lint findings / 130 unformatted files under the
available user configuration. No unrelated source files were reformatted.
