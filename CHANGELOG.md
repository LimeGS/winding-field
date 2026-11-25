# Changelog

Notable changes to `winding-field`. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the GitHub releases.

## [Unreleased]

## [2.0.0] - unreleased

Released in lockstep with winding-field-cli 2.0.0 and winding-field-studio 2.0.0
(one version across the three repositories). No public API is removed; Jacobi
identities and checkpoints are unchanged and resume byte-identically.

### Added
- `solve_field`/`solve_blocks(preconditioner="amg")` (`field_amg.py`): smoothed-aggregation
  AMG built on free nodes only; recorded in the block identity only when not `jacobi`.
- Two-level Schwarz with a flexible-CG outer iteration (`field_schwarz.py`); each tile's
  operator and AMG hierarchy are reused across sweeps.
- `GpuLease(query_interval)` may reuse a recent NVIDIA inventory between rechecks
  (default 0, unchanged).

### Changed
- `field_store.py` commits chunks uncompressed; decoded values are unchanged.

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
