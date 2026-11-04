# Result-preserving performance v2

## Scope and execution ledger

Specification: November 4, 2026 user request, P1–P5. Baseline HEAD is exactly
`aaa1a77a89a76c6f2bd739b956649c4c94d59011`; no reference adaptation is needed.
Work is on `feat/perfomance-optimizations-v2`. The previous stash is untouched.
No applicable AGENTS.md exists. CONTRIBUTING.md, pyproject.toml and CI were read.

Local implementation commits, in the requested order:

| Change | Commit | Production files/functions |
|---|---|---|
| P1 | `1692ee1` | `coherence_pipeline._sample_arclength_pairs`, `compile_fibres` |
| P2 | `d3a1bf7` | `field_solver.solve_field` |
| P3 | `fcee501` | `field_solver.PreparedField.retained_bytes`, `block_solver.solve_blocks` |
| P4 | `fdbf0b4` | `anisotropic_reference.solve_arrival` |
| P5 | `93632c3` | `surface_mesh._prepare_tile`, `_tile_mesh_once`, `_tile_mesh`, `extract_mesh` |

Independent source/test preparation was allowed, but integration commits follow
P1–P5. Existing `field_constraints`, `constraint_templates`, `field_store`,
`field_prefetch`, `whole_reference` and research runners retain their algorithms.

The integrator owns source changes and commits. An independent worker owns only
the throughput harness and its test. Changes are integrated in this order:

- [x] Inspect checkout; establish environment and baseline tests.
- [x] Baseline harness: immutable source snapshot, identical seeded inputs,
  fresh output directories, cold sample and five warm repetitions, raw timings,
  memory/identity/version/config metadata. Keep full evidence outside checkout.
- [x] P1 `coherence_pipeline`: `_sample_arclength_pairs(s, fid)` selects integer
  ranks using subtraction-preserving vectorized binary search and int64 prefixes.
  Test dense-oracle exact parity, boundary floats, multiple runs and 100k stress;
  measure 800/2000/4000 samples. Leave whole_reference unchanged.
- [x] P2 `field_solver`: retain residual norm between iterations, invalidate
  on true-residual restart. Test iteration/failure parity and norm call counts.
- [x] P3 `block_solver`, `field_solver`: explicit `prepared_cache_bytes=0`,
  stable admission, conservative retained-storage accounting and live rebind.
  Test partial/full/off, mutated observations, exterior targets, callable
  providers, failure cleanup and cold STOP/resume with a changed cache budget.
- [x] P4 `anisotropic_reference`: first-visit stable partial stencil admission;
  test exact off/partial/full equality including NaNs and avoided constructions.
- [x] P5 `surface_mesh`: one preparation per tile across levels, independent
  preparation for recursive crops; test exact geometry/keys and regression suite.
- [x] Final measurements, CI gates, quality demo, distribution checks and review.

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

A second direct comparison loaded `aaa1a77:src/field_solver.py` in an independent
module in the same interpreter. All 54 combinations of float32/64, maxiter
0/1/100, and ordinary/zero RHS/nonzero initial/all-fixed/repeated pairs/continuous
consensus/underflow/overflow/nonfinite residual cases had exactly equal fields
and receipts after excluding elapsed time.

## P3: bounded prepared workspaces

`solve_blocks(..., prepared_cache_bytes=0)` is keyword-only and independent of
`cache_bytes`. First-fit admission retains a stable subset. A workspace larger
than the budget still solves, but is not retained. Callable constraint providers
bypass preparation reuse; immutable `StencilConstraints` and unconstrained
blocks may reuse it. Each visit still loads/validates observations and receipts,
reads the shared field, restricts current exterior values, assembles boundaries
and rebinds targets through the existing snapshot checks. No `ValueError` is
suppressed. `constraint_templates.py` needs no semantic change.

`PreparedField.retained_bytes` conservatively counts Python objects, NumPy
snapshots/backing owners and dense/sparse tensor backing allocations, sometimes
double-charging shared storage. Admission adds 4096 bytes per entry for the bounds
key/container overhead. This is not a peak-RSS, temporary-allocation or CUDA
allocator-reservation guarantee. `prepared_cache_stats` reports the budget,
builds/hits, peak entries/bytes and current retention. Terminal receipts show zero
current entries/bytes after cleanup. Workspaces are not serialized; the budget is
excluded from problem identity and may change on cold resume.

A hit avoids gradient operator/diagonal, sparse constraint operator/device
transfer and gradient RHS setup. It does **not** skip observation validation,
restriction-template construction when that separate cache is disabled, live
constraint RHS or fixed-node elimination. With six visits and no template cache,
operator/RHS builds drop 6→2 while restriction templates remain 6→6. Keeping
template retention separate avoids an implicit second cache budget.

Initial six tests failed on the missing keyword; a further entry-overhead test
failed before the 4096-byte allowance. The integrated focused suite passed
235 tests. Cache off/on/partial comparisons preserve field, support, sweep maxima,
anchors and chunk hashes exactly, including live exterior consensus targets;
mutated NumPy aliases, receipt changes and nonfinite observations fail explicitly.
STOP/resume may change the budget and reproduces identical chunks with a cold
cache. Weak-reference checks cover normal exit, provider failure and solver failure.

Supplementary seeded 1000-pair/two-block/eight-sweep CPU case: five warm paired
runs after one cold pair, rtol=1e-8, one thread, 10,000,000-byte cache budget.
Median off/on was 78.005/74.464 ms, IQR 1.449/0.965 ms, all five paired runs faster;
operator builds 16→2, hits 14. Fields, support, chunks and sweep updates matched
across all twelve runs. This measurement preceded the conservative per-entry
4096-byte accounting allowance; its recorded peak 1,572,645 bytes excludes that
later allowance. This is target-specific evidence for the opt-in cache, not a
general full-pipeline speedup claim.

The original script/raw results/checkpoints are preserved externally at
`../../artifacts/winding-perf-v2-prepared-20261104/` relative to the checkout.
The final common harness measures the integrated source including accounting
overhead.

To replay the supplementary target from this checkout, create a **new** directory,
copy `../../artifacts/winding-perf-v2-prepared-20261104/measure.py` there and run:

```sh
.venv-perf-v2-cpu/bin/python "$NEW_DIRECTORY/measure.py"
```

The script records source hashes and creates one fresh checkpoint root per run.
Do not rerun it inside an existing evidence directory. Its checkout path is
recorded explicitly at the top for this local supplementary experiment; use the
portable committed harness below for baseline/candidate replication elsewhere.

## P4: stable partial arrival stencil cache

Stencils are first constructed when visited in iteration one. Entries that fit
remain resident; a missed large entry does not prevent admission of a smaller
last entry. No eviction or prepass occurs. `max_cache_bytes` retains its existing
default and zero disables retention. Ray arrays, order, costs, interpolation,
simultaneous update from `old`, offsets and stopping decisions are unchanged.

Existing receipt keys remain. `stencil_cache_mode` is `none`, `partial` or `full`;
`stencil_cache_blocks`, `stencil_cache_hits` and `stencil_rebuilds` expose work
avoided. Rebuilds include the initial construction, and retained bytes are array
payloads, not total RSS or temporary construction memory. The existing
`stencils_rebuilt_per_iteration` means at least one block is uncached.

Six new tests failed before implementation. Cache off/partial/full now preserves
fields including NaNs, iteration count, residual, convergence and classifications.
Additional tests cover walls, cuts, chart offsets, oblique and opposite directors.
A three-block/four-iteration partial case retains one 104,000-byte block and
avoids three constructions; a 70,000-byte budget can retain the last 65,000-byte
block even though neither preceding 104,000-byte block fits.

## P5: one level-independent preparation per tile

`_prepare_tile(values, valid)` computes the eight-vertex supported-cell mask,
support existence and valid min/max. Extraction shares that tuple across all
levels of the current tile. Existing positional interfaces remain compatible;
the private mesh helpers accept an optional keyword-only `prepared` tuple.
Recursive subcrops prepare their own shapes; single-cell flip retries reuse only
their own preparation. Masks preserve the existing flip behavior.

The inclusive level interval, `np.where(valid, values - level, 0).astype(np.float32)`
order, finite/underflow checks, global keys, float64 edge reconstruction, face
orientation, SQLite transactions/cursor and durability are unchanged.
Five new tests failed before implementation; 58 mesh/performance tests then
passed. New checks compare exact keys/vertices/faces and multi-level/single-level
topology, include partial support and 1e10-offset fields, count one preparation
per tile and verify subdivision gets its own masks.

## Verification commands and artifacts

### Reproduce throughput

The portable harness is [`benchmarks/performance/run.py`](../benchmarks/performance/run.py),
with workloads in `cases.py` and a [reproduction guide](../benchmarks/performance/README.md).
From the checkout, with new output directories:

```sh
PY=.venv-perf-v2-cpu/bin/python
E=/Users/user/Software.nosync/Factory/vesubius-challenge/artifacts/winding-perf-v2-20261104
"$PY" benchmarks/performance/run.py --source "$E/baseline" \
  --commit aaa1a77a89a76c6f2bd739b956649c4c94d59011 --output "$E/replay-baseline"
"$PY" benchmarks/performance/run.py --source . --output "$E/replay-candidate"
"$PY" benchmarks/performance/run.py --source "$E/baseline" \
  --commit aaa1a77a89a76c6f2bd739b956649c4c94d59011 --output "$E/replay-original-blocks" \
  --cases blocks-constraints --seed 423
"$PY" benchmarks/performance/run.py --source . --output "$E/replay-cache-off" \
  --cases blocks-constraints --seed 423
"$PY" benchmarks/performance/run.py --source . --output "$E/replay-cache-on" \
  --cases blocks-constraints --seed 423 --prepared-cache-bytes 10000000
```

The guide also shows how to reconstruct the baseline using `git archive` without
an existing local snapshot. Each run saves source/harness snapshots, exact command,
input/config hashes, shapes, dtype, numerical settings, budgets, seeds, versions,
hardware/threads, raw samples, cold and warm-up timings, stage diagnostics and
memory measurements. The baseline reference is `aaa1a77`; all final candidate
source snapshots were checked file-by-file against `93632c3`, recorded in
`candidate-head93632c3-verification.json`. A later harness-only metadata addition
explicitly marks CUDA/driver as untested; original run records are not rewritten.

### Integrated baseline/candidate measurements

Apple M2 Pro, macOS arm64, Python 3.11.13; scientific versions listed above.
Common harness: one intra-op and inter-op thread, BLAS thread environment set to
one, seed 20261104 (423 for the constrained-block comparison). Five warm repeats
after first-call and warm-up passes. Cold is not a flushed OS cache. This is a
shared host, not an exclusive benchmark machine. No CUDA/driver was exercised.

Times below are **median / IQR milliseconds**. Traced memory and RSS are
**baseline / candidate MiB**. Tracemalloc runs separately and misses untracked
native allocations; RSS is lifetime high-water including imports and all passes,
not retained cache size. Iterations are equal on both sides; dashes mean no
iterative solver or aggregate iteration count was recorded.

| Scenario | Baseline ms | Candidate ms | Traced peak MiB | RSS MiB | Iterations |
|---|---:|---:|---:|---:|---:|
| Fibres 800 | 3.636 / 0.047 | 2.261 / 0.030 | 9.831 / 0.420 | 268.7 / 257.9 | — |
| Fibres 2000 | 12.853 / 0.051 | 3.776 / 0.015 | 61.195 / 1.103 | 323.4 / 258.4 | — |
| Fibres 4000 | 40.376 / 0.206 | 6.201 / 0.209 | 244.579 / 2.210 | 529.1 / 260.8 | — |
| PCG 12³ | 11.183 / 0.339 | 10.655 / 0.395 | 0.020 / 0.020 | 228.0 / 227.9 | 117 |
| PCG prepared 12³ | 10.788 / 0.125 | 11.116 / 0.156 | 0.017 / 0.015 | 227.8 / 228.0 | 117 |
| PCG 32³ | 110.013 / 0.386 | 106.785 / 1.311 | 0.255 / 0.254 | 238.0 / 236.2 | 302 |
| Blocks, cache default | 23.993 / 0.946 | 23.186 / 1.122 | 0.373 / 0.365 | 226.9 / 228.2 | last solve 60 |
| Mesh, small | 62.783 / 3.062 | 63.037 / 1.322 | 0.300 / 0.309 | 204.5 / 204.8 | — |
| Mesh 64³, intersecting | 1085.241 / 8.381 | 1081.564 / 3.623 | 7.030 / 7.042 | 234.6 / 236.5 | — |
| Mesh 64³, 32 off-range levels | 40.525 / 1.187 | 24.348 / 4.746 | 4.509 / 4.509 | 225.6 / 225.9 | — |
| Arrival, original small | 114.416 / 0.839 | 110.031 / 0.792 | 1.071 / 1.084 | 258.5 / 258.7 | 17 |
| Arrival, 0-byte budget | 561.918 / 0.713 | 536.489 / 1.014 | 0.327 / 0.326 | 257.9 / 258.3 | 8 |
| Arrival, 70k budget | 561.734 / 4.768 | 386.273 / 2.060 | 0.327 / 0.407 | 257.5 / 257.8 | 8 |
| Arrival, 120k budget | 583.157 / 2.675 | 379.998 / 0.664 | 0.327 / 0.385 | 257.9 / 258.0 | 8 |
| Arrival, 1M budget | 74.448 / 0.461 | 74.986 / 0.877 | 0.381 / 0.385 | 257.7 / 257.8 | 8 |

Interpretation and counterexamples:

- P1's whole `compile_fibres` case improves 6.51× at 4000 samples; this is distinct
  from the selector-only microbenchmark above. Selection and both constraint
  digests match exactly.
- P2's 32³ case reduces norm calls 607→306, with identical field hash, residual and
  302 iterations. The prepared 12³ case was slower in this run: no universal PCG
  wall-time improvement is claimed.
- P4's 70k/120k cases retain 65,000/104,000 bytes, respectively, each with seven
  hits and 17 constructions. Full retention is 273,000 bytes, 21 hits and three
  constructions. Zero retention avoids the baseline's discarded prepass but has
  no cache hits. Full-cache timing is unchanged within noise. The eight-iteration
  cases remain **unconverged**, identically on both sides: residual
  0.23880908617133123, 105 known nodes. These are fixed-work throughput cases,
  not accepted arrival solutions.
- P5's off-range case reduces preparation calls 32→1 with identical OBJ hashes.
  Intersecting/small meshes show no meaningful wall-time gain; SQLite/geometry
  can dominate. Baseline traced inline support/range and candidate helper timing
  have different scopes; only their counts are directly comparable.
- Each row compares the baseline to the accumulated applicable changes in the
  final source. Blocks combine P2/P3; separate cache off/on measurements isolate
  P3. These independent workloads and nested stage timings must not be summed
  into an invented end-to-end CT pipeline speedup.

### Prepared-cache utility is workload-specific

The common in-memory 1000-row workload did **not** establish a speedup: baseline
76.401/1.975 ms, candidate cache-off 78.717/10.298 ms, cache-on 89.786/8.252 ms.
Candidate off/on traced peaks were 650,455/990,839 bytes; RSS was
241,631,232/243,712,000 bytes. Cache-on peak retention was 1,581,109 bytes with
14 hits and two builds, released to zero at exit. Chunks and sweep updates match.
The last local solve takes zero iterations with residual 2.2259789090685934e-11
in all three variants. No speedup is advertised for that configuration.

To investigate rather than hide that counterexample, an interleaved final-source
rerun of the file-backed supplementary 1000-row case measured 84.569/2.720 ms off
and 81.345/4.280 ms on: also inconclusive relative to dispersion. A setup-heavier
**20,000-row** case, with otherwise identical off/on inputs and eight sweeps,
then measured **490.798/3.624 ms off versus 469.671/1.635 ms on** (4.30% lower
median; all five paired runs faster). Constraint setup falls from about 21 ms
to 2.7 ms, builds 16→2, hits 14. The 100,000,000-byte prepared budget retained
31,059,570 bytes at peak and zero on exit. Exact field/support/chunk hashes and
sweep updates match for all twelve cold/warm runs. Peak RSS was not measured in
this supplementary run; retained bytes must not be presented as its RSS.

This supports keeping P3 **opt-in**, with default zero, for profiled repeated,
setup-heavy blocks where the admitted subset has hits. It does not justify
enabling it for every workload or allocating a cache that cannot hold a workspace.
Final supplementary scripts, raw samples, metadata, input hashes and checkpoints:

- `../../artifacts/winding-perf-v2-prepared-final-20261104/`
- `../../artifacts/winding-perf-v2-prepared-20000-20261104/`

Replay their `measure.py` in a new directory using the same command above;
the 20,000-row script explicitly uses seed 423, float64 solves, 100 MB prepared
and template budgets, file-backed float32 observations, and rtol=1e-8. Environment
metadata notes that its inter-op count was not separately recorded.

### Gates

Use `PY=.venv-perf-v2-cpu/bin/python` in the checkout. The environment is compatible
with pyproject.toml; it shares existing scientific dependencies read-only, as
noted above. New local tool dependencies were resolved with:

```sh
uv pip install --python "$PY" mypy==2.4.0 ruff==0.17.3
uv pip install --python "$PY" --no-deps -e .
"$PY" -m ruff format --check .
"$PY" -m ruff check .
"$PY" -m mypy
"$PY" -m pytest -q --cov=src --cov-report=term-missing
"$PY" examples/minimal_field.py
"$PY" examples/benchmark_demo.py --output "$NEW_QUALITY_OUTPUT"
"$PY" -m build --outdir "$NEW_DISTRIBUTION_OUTPUT"
```

The first full pytest invocation exceeded the harness's 120-second timeout
without an observed assertion failure; it is not counted as a passed run. The
complete retry passed **1296 tests in 134.54 s**, with **92.54%** statement coverage
(required floor unchanged at 90%). Its full `pytest.log` is in the gate-evidence
directory below. Static gates: 90 files formatted, lint clean, mypy clean (26 files).
After adding the explicit CPU/CUDA metadata test, the final full run passed
**1297 tests in 133.69 s**, still **92.54%** coverage; see `pytest-final.log`.
Final format, lint and mypy gates also passed. The source distribution explicitly
includes the performance harness, because its new tests require those scripts;
the wheel's public module inventory remains unchanged.
The final rebuilt sdist passed all seven harness tests from its extracted tree
outside the checkout, and its separately installed package passed the isolated
minimal example again.
Build produced wheel and sdist. The wheel contains all 25 declared modules,
none of the excluded workflow/render modules, and no entry points. Separate
environments installed the wheel and rebuilt sdist; each imported the five CI
modules with `python -I` from site-packages and passed `examples/minimal_field.py`.
The checkout minimal example also passed after editable installation (its first
attempt before installing the package correctly failed to find `field_solver`).

The unchanged quality demo passed: 6/6 fibre pairs covered, separation 3/3,
collapse controls 0/3. It is a quality smoke test, not a throughput measurement.
An independent read-only review of P1–P5 found no supported critical, important
or minor findings. It additionally ran 48 exact baseline/candidate PCG comparisons
and 360 exact mesh-key/vertex/face comparisons (or matching errors), covering
extreme scales, partial/empty support, extrema and near-node nextafter levels.
The final independent harness/report review also found no meaningful findings;
it checked the 15 comparison rows against raw timings, candidate source hashes,
the 20,000-row supplementary outputs and packaged harness inventory.
Original evidence directories (all created fresh):

- `/Users/user/Software.nosync/Factory/vesubius-challenge/artifacts/winding-perf-v2-quality-20261104`
- `/Users/user/Software.nosync/Factory/vesubius-challenge/artifacts/winding-perf-v2-gates-20261104`
- `/Users/user/Software.nosync/Factory/vesubius-challenge/artifacts/winding-perf-v2-prepared-20261104`

No tests require wall-clock thresholds. No dataset, checkpoint from another run,
stash, global environment, CLI repository or publication was modified.

## Boundaries and deferred options

No new preconditioner, mixed precision, CUDA synchronization, compiled backend,
SQLite batching, storage format or prefetch concurrency is introduced. Reusing
static work is the measured scope; these additional changes would require their
own profiles and numerical/artifact contracts. Existing BlockStore reader/cache
and ObservationPrefetch APIs remain unchanged, including validation and durability.

The separate instrumented passes guide that decision (inclusive stages, not
numbers to sum with wall samples):

- The small block case spends 0.033 ms in observation copies and 1.147 ms in
  field reads across four visits, versus 26.358 ms wall and 4.889 ms checkpoint
  commits. This does not justify worker/prefetch overhead or a reader rewrite.
- The 32³ PCG case spends 104.020 ms in the iterative envelope and 1.581 ms
  before its first norm. A stronger SPD preconditioner is a future experiment
  candidate only with setup-inclusive measurements and preserved residual checks;
  there is no evidence here to replace Jacobi or implement multigrid by default.
- The intersecting 64³ mesh case spends 704.434 ms in SQLite add, 64.937 ms in
  tile checkpoint and 206.916 ms in finish/export, versus 1069.286 ms wall.
  Marching cubes is only 4.408 ms and tile preparation 0.824 ms. Thus SQLite
  batching is a plausible next experiment, not a measured accepted optimization:
  key lookups, assigned IDs, component unions, shared geometry and cursor commit
  must remain joint and ordered. No unverified batching variant is shipped.

No timing from the synthetic CPU cases establishes CUDA or real-scroll throughput.
Python 3.12/Linux CI and actual GPU execution remain unverified on this Mac.

The separate CLI could later expose `prepared_cache_bytes` as an operational
byte budget, independent of observation/template/storage budgets, with the same
zero default. That integration is not part of this branch.

For a later Lean pilot, the preserved mathematical obligations are: exact
row-major sampled-pair population; fixed-node elimination and positive-PCG
checks; true-residual acceptance; unchanged consensus exterior rows/means;
live boundary targets; simultaneous arrival updates with the same rays and
offsets; and identical mesh support, level subtraction and topology. Local
convergence remains `pre_encoding_local_solve`; `global_converged=False` and
`physical_acceptance=False` remain explicit. Checkpoint quantization is unchanged.
No Lean files, installation or translation is included.
