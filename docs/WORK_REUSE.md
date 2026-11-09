# Static work and PCG buffers

Version 1.3.0.dev2 incorporates the residual-norm reuse and bounded prepared-block
cache from upstream 82e0496 while preserving the core/CLI module separation at
f750fd11. Numerical tolerances, dtypes, iteration/restart gates and the independent
final true residual are unchanged. PCG output, residual, preconditioned residual,
search direction and stencil scratch buffers are reused; arithmetic is not fused.
Returned fields never alias these work buffers. PreparedField is single-caller.

`solve_blocks(..., prepared_cache_bytes=0)` optionally retains a stable first-fit
subset of block workspaces. `prepared_cache_stats` reports budget, hits/builds and
peak retained bytes/entries. Current retained entries are zero on exit. Budgets
account conservatively for backing CPU/device storage, not allocator reservation,
transient assembly or peak process memory. Cache state is never persisted and the
budget may change on resume.

Static StencilConstraints and absent constraints are eligible. Callable providers
opt in by exposing `prepared_key(lo, hi) -> str | None`: return a token (at most 256
characters) identifying the block's static observations/row structure, or None to
bypass. Change the token when these change. An unchanged retained token is checked
against values, geometry, dtype/device and structural arrays, including writable
NumPy aliases; a mismatch raises instead of silently reusing a stale operator.
Arbitrary callbacks without this contract bypass caching. Retired token entries
may occupy the bounded first-fit cache until solve_blocks exits.

Every visit still loads/validates observations, calls the constraint provider and
refreshes constraint targets, fixed values, boundary elimination and the live RHS.
`StencilConstraints.prepare(lo, hi)` exposes the existing static restriction
template (`nbytes`, `restrict(read_nodes)`) to providers with their own byte budget.
Templates own their static arrays and hold only a weak reference to returned rows;
current exterior values are read on each restriction. Reconstruct the stencil or
provider when static geometry/rows change.

`prepare(..., read_nodes=store.nodes)` also prepares a location-only gather plan
within the template byte budget; `restrict(lo, hi, store.nodes)` does this
automatically. Neither route retains exterior field values. The v3 denied-key
budget and status policies apply to opted-in callable providers as well as static
stencils. Spectral PCG remains standalone-only for supported CPU float64 masks;
block solves still require Jacobi.

The CLI owns the opt-in cross-version phase/throughput harness and integration
documentation. Caching is not a universal speedup; measure cold setup, warm reuse,
full checkpoint execution and numerical parity on the intended device.
