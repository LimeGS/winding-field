# Solver performance v3

## Contracts and configuration

CPU consensus application prepares ascending active row indices, active groups,
weights and clamped group denominators. This preserves multiplication/subtraction
and `index_add_` accumulation order exactly, including zero-weight and exterior
rows. The non-CPU implementation is unchanged. Targets remain live. Prepared
workspaces own structural snapshots and reject changed weights, groups, geometry,
dtype/device and writable-alias mutations. Accounting includes backing storage.

```python
from field_solver import PreparedField, solve_field

workspace = PreparedField()
field, receipt = solve_field(
    target_xyz, weights, constraints=constraints, fixed=fixed_values,
    dtype="float64", device="cpu", rtol=1e-8, maxiter=1500,
    preconditioner="spectral", prepared=workspace,
)
```

`preconditioner="jacobi"` remains the default. Spectral is explicit, CPU float64
only, and accepts the **effective** fixed mask after the existing gauge rules:

- Exactly one fixed node, including the automatic zero gauge or an interior,
  nonzero supplied anchor. Length-one axes are allowed.
- Exactly all six complete exterior faces, with no extra interior anchors.
- All nodes fixed (no free correction).

Other geometries, precisions and devices fail; a reference-only unanchored system
is not supported. No hidden device transfer or precision conversion is added.
Point masks use orthonormal DCT-II and `idctn(type=2)`; the zero-mode inverse is
exactly zero. Full faces use orthonormal DST-I and `idstn(type=1)`. The uniform
coefficient is the float64 mean of weights. Eigenvalues represent differences in
**index space**, without spacing factors. Constraints and reference terms remain
in A. Anchor balancing is confined to the preconditioner; original nonzero-anchor
RHS elimination, warm start, restarts and final true residual are preserved.

Receipts record the requested/actual policy and spectral kind/coefficient.
Prepared spectral reuse compares the full fixed mask, not its anchor count;
changed policy/mask requires a new workspace. Fixed values and constraint targets
refresh on every solve. The policy may change iterations and final decimals;
its guarantee is the same system, anchors and true relative residual criterion.

## Blocks, independent caches and STATUS

```python
from constraint_templates import StencilConstraints
from block_solver import solve_blocks

rows = StencilConstraints.load(
    "constraints.npz", cache_bytes=64 * 2**20,
    spatial_index_bytes=32 * 2**20, template_cache_policy="stable",
)
receipt = solve_blocks(
    root, shape, chunk, identity, load_observations,
    origin_xyz=origin, spacing_xyz=spacing, order=order, constraints=rows,
    cache_bytes=64 * 2**20, prepared_cache_bytes=128 * 2**20,
    preconditioner="jacobi", status_policy="checkpoints",
)
```

The block engine produces partial fixed-face masks; it deliberately accepts only
Jacobi and rejects spectral before creating outputs. Jacobi is the invariant
policy of the existing v1 block identity (including identities without an explicit
field). It is also recorded in each solver receipt. No alternative policy can be
resumed. Caller source/code identity validation remains mandatory and unchanged;
these changes do not authorize resuming a checkpoint made by different code.

| Budget/policy | Meaning | Default |
|---|---|---|
| `cache_bytes` on `solve_blocks` | Decoded chunk generations | 0 |
| `cache_bytes` on `StencilConstraints` | Restriction templates and retained gather plans | 0 |
| `spatial_index_bytes` | Sorted original nonzero-slot spatial postings, independently bounded | 0 |
| `prepared_cache_bytes` | Static solver workspace plus nonadmission metadata | 0 |
| `template_cache_policy` | `lru` or stable first-fit | `lru` |
| `status_policy` | `all` or `checkpoints` | `all` |

These bound retained storage, not peak process memory. Static input snapshots,
returned arrays, solver temporaries and FFT scratch are additional memory. Spatial
admission reserves a conservative construction-workspace bound before allocating
postings. Insufficient budget falls back to the original exact scan. Sorted flat
node postings are queried by contiguous X spans; candidates undergo the original
exact half-open ZYX filter before complete groups are expanded in original row
order. Original nonzero-slot contact survives duplicates and cancellations.

Stable template admission keeps the first fitting entries rather than evicting
them on a cyclic scan. For A/B/C repeated three times with room for two templates:
LRU gives 0 hits, 9 misses, 7 evictions; stable gives 4 hits, 5 misses, 0 evictions.
Gather plans cache only chunk/local-index/order geometry, never values or
generations. Both cached and uncached reads consult the current committed chunks.
Fortran chunks are gathered without flattening/copying their entire storage.

Prepared admission retains a stable subset. Known nonadmitted keys can consume
4096 bytes each from the same budget, avoiding subsequent snapshot construction.
If even that metadata cannot fit, snapshots may be repeated. Denied keys always
solve cold: they do not reuse an operator or skip current-input validation.
Arbitrary callable constraint providers remain ineligible for prepared retention.
All workspaces/decisions are released on success, STOP and exceptions; resume is
cold. Every visit still validates observations/receipts and uses live boundaries.

`status_policy="checkpoints"` publishes the first RUNNING immediately, every
checkpoint after commit, and every terminal state. It suppresses only later
informational observations/solving notices. STOP/resource checks, ownership/PID,
durable commits, atomic replace and fsync are unchanged. STATUS is not a global
convergence certificate; local, assembled-field and physical acceptance remain
distinct.

Commit validation's decoded payload is admitted to the bounded read cache only
after successful durable publication and required cleanup. Failed publication
retains fail-closed/poison semantics. Manifest bytes are serialized and validated
once for both admission and publication. Anchor error uses the same mean over
the whole core while quantizing only anchor positions; no anchors means zero.

## Reproduce the measurements

Use one interpreter/dependency installation for both source trees. Baseline and
candidate are frozen into a **new external output** directory with source hashes,
revision metadata and an executable copy of the harness:

```sh
.venv-perf-v2-cpu/bin/python benchmarks/performance/v3.py \
  --baseline /absolute/path/to/archived-current-baseline \
  --baseline-commit 82e0496b1ebe75f5b48ff92ca426233e35e61fc4 \
  --candidate . --output /absolute/path/to/new-evidence --repetitions 8
```

Four solver variants isolate baseline, consensus indices only, spectral only,
and both. The intermediate variants mix complete module files only in frozen
benchmark trees, never production. Each worker imports its assigned tree; all
imported project-module paths and source hashes are checked. Only one worker
executes a command at a time; other workers are idle. Each variant gets a first
call and an explicit warm-up before eight cyclically balanced repetitions.
First-call timings are not cold-machine or flushed-filesystem measurements.

The requested 32³ fixtures preserve RNG order, seeds, 20,000 consensus rows in
groups of 20, interpolated 32 pair rows, and post-constraint 100:1 weight changes.
All use float64, `rtol=1e-8`, `maxiter=1500` and one Torch/BLAS/OpenMP thread.
Tiny 3³, spatial/template, partial A/B/C, gather, commit/read and existing block
shapes provide additional regimes. There is no distributable real-data fixture
in this checkout; these are synthetic, not scroll/physical-validation results.

Timing excludes fixture generation, hashes, validation and diagnostic wrappers;
solver timing includes preconditioner preparation. Constraint construction is
recorded separately. Block timings include their required production hashes,
receipts and durable I/O, which are part of the operation, not benchmark probes.
Memory and instrumentation run separately. Inclusive stage regions nest and must
not be added together. Tracemalloc excludes untracked native allocations; RSS is
process-lifetime high-water including imports/all passes. Solver timed calls retain
no PreparedField; the separately measured hypothetical prepared workspace is
explicitly labeled. Template/cache retention is taken from actual diagnostic state.

Raw JSON contains every sample and order, medians/IQR, receipts, max/RMS field
differences, anchors, cache counts, setup/iterative-envelope/commit/STATUS/read
diagnostics, retention/peak memory and full provenance. Structural comparisons
also assert exact constraint digests, chunk hashes, sweep updates and numerical
receipts excluding duration/new operational metadata. Store reads validate every
generation against independently reconstructed quantized values.

## Local results and validation

Measured on Apple M2 Pro (12 logical CPUs), macOS arm64, Python 3.11.13,
NumPy 2.2.6, SciPy 1.16.1, Torch 2.12.1. All thread settings were 1. These are
local measurements, not the historical Linux timings. Each cell is median / IQR
in milliseconds over eight balanced repetitions, with setup included:

| Solver fixture | Baseline | Indices only | Spectral only | Both |
|---|---:|---:|---:|---:|
| Pairs 32³ | 102.997 / 1.445 | 102.937 / 1.842 | 14.803 / 0.354 | 14.897 / 0.464 |
| Consensus20k 32³ | 605.846 / 3.208 | 535.399 / 7.184 | 109.300 / 2.092 | 97.818 / 1.166 |
| Weights 100:1 | 115.318 / 1.232 | 114.955 / 1.470 | 50.483 / 0.581 | 50.581 / 0.540 |
| Six faces | 42.366 / 0.902 | 42.456 / 1.227 | 24.678 / 0.553 | 24.361 / 0.113 |
| Tiny 3³ | 2.020 / 0.094 | 1.893 / 0.094 | 1.622 / 0.041 | 1.549 / 0.088 |

Indices alone preserve **exact fields, iterations and residuals**. Non-consensus
rows have no active groups, so no useful index speedup is expected; differences
there are noise-scale, including the slightly slower faces median. Spectral-only
and combined fields are identical to each other in this run, but not bitwise to
Jacobi. All solver cases converged with exact anchors:

| Fixture | Iterations Jacobi → spectral | True residual Jacobi → spectral | Max / RMS field difference |
|---|---:|---:|---:|
| Pairs | 302 → 12 | 9.89226609e-9 → 7.81897206e-9 | 2.50514e-8 / 7.93559e-9 |
| Consensus | 162 → 21 | 8.79348674e-9 → 9.22212233e-9 | 2.70647e-8 / 3.77629e-9 |
| Jump | 332 → 50 | 9.63971061e-9 → 8.54800280e-9 | 9.98790e-8 / 1.92498e-8 |
| Faces | 115 → 12 | 9.16315886e-9 → 6.86551619e-9 | 1.24485e-8 / 3.63912e-9 |
| Tiny | 20 → 9 | 6.31692443e-9 → 5.99213852e-9 | 9.21100e-10 / 5.43577e-10 |

| Structural/I/O fixture | Baseline ms / IQR | Candidate ms / IQR |
|---|---:|---:|
| Spatial templates, dense cross-block groups, no template retention | 83.045 / 1.095 | **115.605 / 0.636** |
| A/B/C, capacity two | 2.355 / 0.106 | 1.571 / 0.152 |
| Three durable commits + reads, 32³ | 16.857 / 0.397 | 15.498 / 0.348 |
| 10,000-node gather, cached chunks | 0.655 / 0.021 | 0.100 / 0.012 |
| Blocks, 8×8×16 | 23.663 / 1.296 | 20.678 / 0.566 |
| Constrained blocks, 3×4×6 | 74.249 / 1.067 | 58.490 / 1.531 |

**Negative result:** the spatial index loses about 39% in the dense cross-block
group fixture. Nearly every group is selected, leaving little scan work to remove;
postings construction/query adds work. It remains optional and **off by default**.
This case includes index construction and nine restrictions. Do not enable it
indiscriminately. Stable A/B/C gives the required 4/5/0 hit/miss/eviction counts
versus LRU's 0/9/7. Structural arrays, digests and block payload hashes match
exactly. Block candidates combine prepared retention, stable/indexed templates
where applicable, storage changes and checkpoint-only STATUS; their speedup is
not an isolated attribution to any one component.

Memory tradeoffs are explicit. The consensus hypothetical prepared workspace
grows from 14,322,490 bytes to 15,402,186 bytes with both numerical optimizations.
Timed solver calls retain no workspace. For point 32³ the traced separate-pass
peak grows from 145,605 to 1,353,578 bytes; this does not include all Torch-native
allocations. Spatial template postings retain 1,284,096 bytes in the losing
fixture. The gather plan raises retained cache+plan bytes from 50,688 to 212,512
while reducing traced peak from 644,520 to 135,224 bytes. Full per-case memory,
stage timings, first calls and constraint construction costs are in the raw JSON.

Evidence root relative to this checkout:
`../../artifacts/perf-v3-20261104/final-balanced/`. It contains `provenance.json`,
`summary.json`, one raw JSON per case, all measured fields, frozen source variants,
and the executed harness. The two `harness-smoke*` directories are validation
smokes and **are not pooled into these tables**. Baseline is the source archive
of `82e0496`; candidate provenance records its dirty-source hashes, subsequently
committed locally. No measured source changed during the run.

Validation executed locally:

- Independent numerical review approved CPU index parity and spectral geometry,
  elimination, restart, alias invalidation and memory accounting.
- Core full coverage gate: **1442 passed, 92.78%** (90% threshold); the final six
  harness checks also passed after tightening provenance/metadata assertions.
- Core Ruff format/lint and mypy passed; wheel and sdist built. Packaging and
  paired CLI integration are checked with the matching `1.3.0.dev2` package.
- Storage review found and fixed full-chunk Fortran flattening; regressions fail
  before the fix and pass after it. Block review removed an extra pre-lease
  manifest read rather than weakening locked identity validation.
- CLI baseline has 89 existing lint findings and seven unformatted files; exact
  diagnostic comparison found no additions. Those unrelated failures are retained
  and are not described as a clean lint/format gate.

No CUDA/MPS execution, real-scroll run, remote CI, push, merge or publication is
part of this local delivery. The non-CPU consensus path remains compatible; CPU
speedups do not imply GPU speedups. Local package version `1.3.0.dev2` distinguishes
the added API from older installations; the paired CLI pins that exact version.
