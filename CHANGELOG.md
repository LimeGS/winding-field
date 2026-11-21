# Changelog

Notable changes to `winding-field`. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the GitHub releases.

## [Unreleased]

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

[Unreleased]: https://github.com/LimeGS/winding-field/compare/v1.7.0...HEAD
[1.7.0]: https://github.com/LimeGS/winding-field/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/LimeGS/winding-field/compare/v1.5.3...v1.6.0
[1.5.3]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.3
[1.5.2]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.2
[1.5.1]: https://github.com/LimeGS/winding-field/releases/tag/v1.5.1
