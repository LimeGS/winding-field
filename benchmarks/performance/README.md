# CPU throughput harness

This opt-in synthetic harness is separate from `examples/benchmark_demo.py`,
which evaluates scientific quality. It never modifies the measured source tree.
Each run snapshots source and harness, starts one child per scenario and creates
fresh checkpoint/mesh directories for every repetition. Existing output roots
are rejected. Run baseline and candidate with the same Python environment.

```sh
PY=.venv-perf-v2-cpu/bin/python
# Choose a new external directory; no dataset downloads are needed.
E=/absolute/path/to/new-performance-evidence
mkdir "$E"
mkdir "$E/baseline"
git archive aaa1a77a89a76c6f2bd739b956649c4c94d59011 src | tar -x -C "$E/baseline"
"$PY" benchmarks/performance/run.py --source "$E/baseline" \
  --commit aaa1a77a89a76c6f2bd739b956649c4c94d59011 --output "$E/before"
"$PY" benchmarks/performance/run.py --source . --output "$E/after"
"$PY" benchmarks/performance/run.py --source "$E/baseline" \
  --commit aaa1a77a89a76c6f2bd739b956649c4c94d59011 --output "$E/original-blocks" \
  --cases blocks-constraints --seed 423
"$PY" benchmarks/performance/run.py --source . --output "$E/cache-off" \
  --cases blocks-constraints --seed 423
"$PY" benchmarks/performance/run.py --source . --output "$E/cache-on" \
  --cases blocks-constraints --seed 423 --prepared-cache-bytes 10000000
```

`--cases` selects scenarios; `--repetitions` defaults to five warm measurements
after a first-call (cold) sample and another warm-up. Cold does not mean flushed
filesystem caches. Inputs, source/config hashes, actual parameters, dependency
versions, hardware, thread settings and raw timings are retained in JSON. Timing
includes small digest/output-validation costs, but not input construction/imports.
Each run saves exact source and harness scripts for replay.

The memory pass and instrumented pass are separate from timing samples. RSS is
process lifetime high-water; tracemalloc excludes untracked native allocations;
neither is retained-cache memory. Stage wrappers are inclusive and **non-additive**.
PCG's iterative envelope runs from the first RHS norm through the final true
residual norm. CPU conversion is not a CUDA transfer. Baseline inline mesh
preparation is line-traced and excludes mask creation; candidate preparation
includes it, so compare preparation counts rather than treating these durations
as identical regions. Eight-iteration arrival cases deliberately measure the
same fixed work budget; their unconverged receipts are retained, not promoted.

The harness disables CUDA for children and does not measure concurrent prefetch.
Do not sum overlapping stages or extrapolate these cases to full-scroll results.
See [the implementation report](../../docs/PERFORMANCE_V2.md) for measured results,
counterexamples, cache limits and unchanged numerical/artifact contracts.
# Solver v3 balanced benchmark

For consensus indices, spectral PCG, templates/gathers and durable I/O, use
`v3.py` and [the v3 guide](../../docs/PERFORMANCE_V3.md). It freezes both sources
and the executed harness, balances variants, excludes validation/hashing from
timed regions, and records separate diagnostics. Earlier v2 evidence below uses
the original harness and must not be pooled with these samples.
