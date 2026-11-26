# Changelog

Notable changes to `winding-field`. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the GitHub releases.

## [Unreleased]

### Changed
- AMG CPU setup is multithreaded: `assemble_free_operator` and the free-node restriction
  build row blocks on all available CPUs (`threads=` on `assemble_free_operator` and
  `AMGPreconditioner`). The assembled operator is bitwise identical to the serial build
  for any thread count, so hierarchies and iterates do not change.
- AMG hierarchy build: the Galerkin product sets aside coarse rows that no later block
  reaches, so its cost grows linearly with the number of row blocks instead of
  quadratically (the 416³ host build sat in this step); on the host, coalescing uses all
  torch threads. Both keep every sum's terms and order, so the hierarchy is unchanged.
- When the device build runs out of memory, the failed attempt is released before the host
  rebuild instead of being held by the exception for its whole duration.

## [2.0.0] - unreleased

Released in lockstep with winding-field-cli 2.0.0 and winding-field-studio 2.0.0
(one version across the three repositories). Five unused public helpers are removed
(see Removed and [docs/MIGRATION.md](docs/MIGRATION.md)); Jacobi identities and
checkpoints are unchanged and resume byte-identically.

### Added
- `solve_field`/`solve_blocks(preconditioner="amg")` (`field_amg.py`): smoothed-aggregation
  AMG built on free nodes only; recorded in the block identity only when not `jacobi`.
- Two-level Schwarz with a flexible-CG outer iteration (`field_schwarz.py`); each tile's
  operator and AMG hierarchy are reused across sweeps.
- `GpuLease(query_interval)` may reuse a recent NVIDIA inventory between rechecks
  (default 0, unchanged).
- `solve_blocks(coarse_factor=k)` (`field_block_coarse.py`): out-of-core two-level
  multiplicative Schwarz over the block sweeps; the factor is part of the run identity.
- `solve_blocks(checkpoint="sweep")`: committed cores stay in memory and are published
  once per sweep; the default `"block"` is unchanged.
- `field_memory.plan_memory`: picks a host-memory tier (`disk`, `field`, `all`) from the
  memory available at start and maps it to the block solver's checkpoint mode.
- `solve_field`/`solve_blocks(kernel="fused")` (`field_kernels.py`, opt-in): the PCG
  iterations run as three Triton GPU kernels (stencil + p.Ap, update + dots, direction)
  instead of ~40 PyTorch ops, for Jacobi and the outer AMG iteration. Same system and
  stopping rules, deterministic reductions, final residual still measured with the eager
  operator. The receipt records `kernel` (`fused-v1`, or `eager` + `kernel_fallback` off
  CUDA, with discrete pairs or `spectral`); `solve_blocks` adds `kernel` to the identity
  only when fused and reports `solver_kernels`. Eager receipts, identities and checkpoints
  are unchanged. New optional extra `fused` (Triton; bundled with Linux CUDA PyTorch).

### Changed
- AMG Chebyshev smoother: the upper eigenvalue bound is min(scaled row-sum norm,
  1.3 x a 40-step power estimate). The old estimate could fall up to 23 % short and
  intermittently break CG; AMG results keep their accuracy.
- `field_store.py` commits chunks uncompressed; decoded values are unchanged.
- `NpyObservations` and `NpyFixedValues` share one pinned-NPY open and recheck; their
  errors now read `<label> hash differs`, `<label> is not a memory-mappable NPY array`
  and `<label> file or header changed`.

### Removed
- `common.sample_pairs`, `common.calib_gap`, `physical_grid.offset_correction`,
  `paris_solve.working_shape` and `paris_solve.load_pairs`: nothing called them.
  Replacements are in [docs/MIGRATION.md](docs/MIGRATION.md#upgrading-to-200).

## [1.7.0] - 2026-11-21

### Added
- `solve_blocks(partial_order=True)`: `order` may list only the owned chunks; the
  others are never solved or stored, and a required `fixed` provider supplies their
  values inside owned halos. Full-order identities and checkpoints are unchanged.

### Documentation
- README reorganized: what it is, where it fits, quick start, then details.
  Method, results and migration moved to `docs/`; developer notes to `docs/dev/`.
- Added this changelog and a code of conduct.

## [1.6.0] - 2026-11-20

### Added
- Verification loop: `tools/verify.py` (public API, golden solver outputs, README
  examples, cross-version checkpoint resume against the base branch), `REVIEW.md`
  diff-triggered checklist; CI runs the base commit's verifier.

## [1.5.4]

### Added
- Opt-in block-solver nonconvergence policy: `nonconvergence_policy="continue"`
  commits maxiter-exhausted local solves and reports `local_solves_converged`;
  breakdowns always fail; default `fail` unchanged.

## [1.5.3] - 2026-11-18
- Optional advisory observers.

## [1.5.2] - 2026-11-16
- Exact physical stencil boundaries.

## [1.5.1] - 2026-11-09
- Solver performance and CLI-ready numerical core.

[Unreleased]: https://github.com/LimeGS/winding-field/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/LimeGS/winding-field/compare/v1.7.0...v2.0.0
[1.7.0]: https://github.com/LimeGS/winding-field/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/LimeGS/winding-field/compare/v1.5.3...v1.6.0
[1.5.3]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.3
[1.5.2]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.2
[1.5.1]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.1
