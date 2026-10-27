# Winding fields and reproducible evaluation

[![Audit](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml)
[![Coverage (local): 92.94%](https://img.shields.io/badge/coverage%20%28local%29-92.94%25-brightgreen)](#checks)
[![Code license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Experimental software for constructing continuous winding-coordinate fields over
predicted scroll surfaces, and evaluating saved fields against a frozen case.
A useful field should remain nearly constant along a sheet while separating
adjacent sheets. Numerical convergence and CT overlays do not certify physical
sheet identity.

This repository contains the current implementation, reusable evaluation tools,
synthetic examples and regression tests. Research logs, run outputs, experiment
reports and prior release snapshots are maintained separately.

[![Six Paris4 winding levels: geometry, CT-textured surfaces and available continuous flattened CT diagnostics](assets/paris4-six-surfaces.png)](assets/paris4-six-surfaces.png)

*Paris4 six-surface diagnostics. The continuous flattened surface at W = −120
is unavailable; masked and unmapped areas remain blank. Physical sheet identity
is unverified. [Image attribution and terms](DATA.md#readme-illustrations).*

[![Paris4 CT in XY, XZ and YZ views beside global B2 and fine-candidate winding contours](assets/ct-winding.png)](assets/ct-winding.png)

*Earlier Paris4 CT/winding overlay: CT at 9.6 µm/pixel, the global B2 field at
307.2 µm and a local fine pilot at 19.2 µm. This illustration is separate from
the fine-band measurement reported here.
[Image attribution and terms](DATA.md#readme-illustrations).*

## The current method

We approach winding counting as the recovery of a continuous layer coordinate,
`W(x)`, over the scroll volume. Within a locally consistent region, this
coordinate should vary little when moving along a papyrus layer and increase by
approximately one when crossing to the next. Surfaces of constant `W` are then
candidates for following the paper, while differences in `W` estimate how many
layer spacings separate two locations. The absolute zero is a reference choice;
the useful information lies in relative ordering and calibrated spacing, rather
than in assigning an integer label to every voxel at the outset.

Surface-orientation predictions suggest the direction across layers, and a
layer-density estimate supplies the expected rate of change in that direction.
Traced fibres provide complementary evidence: samples believed to lie on the
same local layer should have similar coordinate values. These observations are
noisy and need not form a mutually consistent gradient field, so integrating
them independently along different paths can accumulate drift. Our approach
instead estimates a field jointly, balancing agreement with the local gradient
observations against variation along fibres. A shared coarse reference carries
a common origin and scale into finer regions, helping their local refinements
remain mutually consistent.

The open problem is making that coherence preserve the physical winding count.
A constant field has no within-layer drift, and a compressed field can improve
that metric while losing separation between neighbouring layers. Reliable
counting therefore also requires justified orientation and calibration,
validation of between-layer separation, and explicit treatment of missing
evidence. Folds, breaks and the topology of a rolled sheet can require cuts or
linked local coordinate patches rather than one globally smooth numbering.
We assess candidates through coverage, intrafibre drift, separation controls
and matched CT overlays; smooth contours or numerical convergence alone do not
establish that the inferred layers are physically correct.

## Fine-grid measurement — AFV v2 (19.2 µm)

Measured on **October 20, 2026**, using the latest saved Paris4 fine-band
checkpoint and a frozen selection from
[Bruniss's `fibers.afv`](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fibers.afv),
snapshot **2026-10-13 03:43:29 UTC**. See
[source credits and release posts](DATA.md#public-afv-sources).
**No real AFV is bundled or selected by default.**

Drift is `abs(phi(a)-phi(b))/1.192497988541921`, in nominal sheet units;
distance is along-fibre arclength. Each fibre contributes at most one pair per
distance bin and may appear in several bins. A pair is covered only when all
eight interpolation corners at both endpoints contain finite stored fine-grid
values, with a one-node domain margin. Unavailable fine samples are not filled
from the coarse reference.

**792,595 / 2,154,184 selected pairs covered (36.8%).** Values are rounded to
three decimals.

| Along-fibre span (mm) | Covered / selected pairs | Median | p90 | p99 |
|---|---:|---:|---:|---:|
| 0.5–1.5 | 351,754 / 913,491 | 0.155 | 0.538 | 1.291 |
| 1.5–2.5 | 134,771 / 367,547 | 0.294 | 0.897 | 2.326 |
| 2.5–3.5 | 86,226 / 237,948 | 0.384 | 1.139 | 2.929 |
| 3.5–5 | 65,112 / 182,354 | 0.452 | 1.328 | 3.405 |
| 5–7.5 | 48,300 / 137,923 | 0.536 | 1.564 | 3.830 |
| 7.5–10 | 34,911 / 100,837 | 0.626 | 1.827 | 4.382 |
| 10–15 | 27,019 / 79,017 | 0.732 | 2.066 | 4.538 |
| 15–20 | 17,955 / 53,270 | 0.879 | 2.432 | 5.383 |
| 20–25 | 12,404 / 37,796 | 1.041 | 2.737 | 5.404 |
| 25–30 | 8,384 / 25,978 | 1.184 | 3.016 | 5.723 |
| 30–45 | 5,759 / 18,023 | 1.336 | 3.343 | 6.109 |

**Interpretation:** drift grows with along-fibre distance, and substantial
long-span errors remain. Statistics describe the covered pairs; unavailable
pairs remain in the selected denominator. A constant field also has zero
intrafibre drift, so this metric must be considered alongside separation and
coverage.

This is a **partial checkpoint**, retaining an earlier computed fine-band state
plus initialized support growth; the restarted campaign stopped before
committing a new solved block.
These are development consistency measurements that can include fitting fibres,
not an independent holdout, physical validation or a final convergence claim.

<details>
<summary>Fine-grid checkpoint and evaluation identities</summary>

- Bruniss AFV v2: `1898ff5098f03a36e02c76efc5f6b3774b766b154bfc8faadca488459114c941`.
- Latest fine checkpoint used in this measurement: `7dd87597ed8ce712cbd41f32b65d1039b158106c7c7da9a82fa1aca23fcc7dd9`.
- Frozen pair selection: `57f1554a97bf9519f5cacf37e53af7cb9b7b00473aa66ca8d927f889498cd8ee`.
- Fine evaluation result: `80bf1a30c94f1c4c334461b7cd9d767b60a6acf524342173782fb622c22af834`.
- Fine checkpoint evaluation adapter: `0a23fb0f33503a813bf1ec1adac729f31bcd2ed4344b6c683b245fe2de0c13c7`.

These measurements were produced by a checkpoint-specific research adapter.
The checkpoint, frozen pairs and per-pair predictions remain external execution
artifacts. The [benchmark guide](benchmarks/winding/README.md) describes the
evaluation contract; the offline synthetic demo does not reproduce these
Paris4 measurements.

</details>

## Install

Target interpreters: Python 3.11 and 3.12 on Linux and macOS. CI is configured for
both Python versions on Linux. Tests run on CPU; GPU execution needs a compatible
CUDA installation. Process-supervision tools use Unix facilities.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[test,dev]"
```

Optional extras: `surface` for mesh/atlas tools, `data` for Zarr, `plot` for
Matplotlib, or `all` for the runtime/test features. Modules install flat
(`field_solver`, `physical_grid`, etc.); use a virtual environment to avoid name
collisions. The wheel contains `src/`; command-line tools live in the checkout.
The installed `winding-field` command runs prepared block construction/resume;
the equivalent checkout entry point is `python tools/run_field.py`.

## Minimal example

```bash
python -I examples/minimal_field.py
```

It solves and checks a synthetic ramp:

```text
converged: True
field[0,0,:]: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

The main library interface is [`field_solver.solve_field`](src/field_solver.py).
It accepts NumPy arrays, nested sequences or PyTorch tensors and by default returns
a NumPy `float32` field plus a JSON-compatible receipt. Coordinates/gradients are XYZ;
arrays are ZYX. Defaults are CPU, unit spacing, `lam=3`, `rtol=1e-5`,
`maxiter=4000` and no reference penalty. Arguments are explicit; the solver reads
no configuration files. Fixed nodes take exact values, and without effective
anchors node zero fixes the gauge. Invalid inputs raise `ValueError`;
nonconvergence is reported in the receipt and must block acceptance of that solve.
`paris_data.save_field` preserves the solver receipt's declared float32/float64
precision and rejects mismatched arrays or lossy casts that would inherit its
convergence claim. Legacy receipts without `dtype` keep float32 export only when
the conversion is lossless; revalidate any intentional precision reduction.

## Optional performance backend (unreleased)

The feature branch adds reusable performance primitives and a block execution
backend. Existing dense APIs and dataset CLI defaults remain unchanged. These
interfaces require explicit geometry, observations and resource budgets; they do
not infer a scroll's calibration, orientation, source data or GPU assignment.

### Build and resume prepared fields

The installed entry point delegates to the same block solver and checkpoint
store used by Python callers:

```bash
winding-field build --config /path/to/field-run.json --output /path/to/new-field
winding-field resume --config /path/to/field-run.json --output /path/to/new-field --prefetch

# Optional owned worker, immutable code/config binding and resource-only retries:
winding-field build --config /path/to/field-run.json --output /path/to/other-field --supervised
```

Start from [`examples/field-run.example.json`](examples/field-run.example.json).
Its paths, hashes and calibration are placeholders; replace them with your
prepared inputs and justified physical metadata. Paths resolve relative to the
JSON file. No raw CT/AFV preparation, orientation inference or calibration is
hidden in this entry point. A recipe supplies these inputs; the principal
library owns construction and recovery.

The `winding-field-run-v1` configuration requires:

- `frame`, positive `voxel_um`, `sheet_step`, and explicit `calibration`
  provenance. Raw values remain in the declared field units; `sheet_step`
  records their conversion to nominal sheet units, not an implicit rescaling.
- `grid.origin_xyz`, `spacing_xyz` in that physical frame, and integer
  `shape_zyx`. Each axis has at least two nodes. Gradients/components are XYZ;
  array axes and chunk traversal are ZYX.
- `observations.target`, `weights`, `observed`, each with local `path` and full
  lowercase `sha256`. Target is floating ZYX3; weights floating ZYX; both may be
  float32/float64 and are read as float32 crops. Observed is boolean ZYX. Targets
  must remain finite and weights strictly positive even where observed is false;
  missing support is not automatically a measured zero or a boundary anchor.
- `execution.chunk`, `halo`, `sweeps`. Multiple cores require positive halo.
  Optional `cache_bytes=0`, `read_workers=1`, `constraint_cache_bytes=0`,
  `reserve_bytes=0`, `max_bytes=null` control the existing storage/reader budgets.
  `solver` defaults to CPU, float64, `rtol=1e-5`, `maxiter=4000`; float32 is also
  supported. These are numerical controls, not a convergence certificate.

Optional `fixed` is a pinned float32/float64 ZYX NPY: finite values (including
zero) are exact fixed nodes, NaN means free. Precision is retained. Without any
finite fixed value the global first node is the numerical zero gauge. Optional
`constraints` is a pinned archive written by `StencilConstraints.save(path)`;
`StencilConstraints.load(path)` preserves its rows, targets, weights and groups.
The archive is resident: template-cache limits do not bound its input arrays.
For very large fibre inputs use the Python block-provider interface below.
Optional `order` is a pinned integer Nx3 NPY containing every chunk key exactly
once; omitted order is lexicographic ZYX. `source_identity` may record additional
JSON provenance, for example the preparation recipe and its inputs.

CUDA requires `solver.device="cuda:0"`, a `gpu` object with explicit `index`,
`uuid`, `lock_path` (relative to the config or absolute), and a matching
`CUDA_VISIBLE_DEVICES` UUID set before launch. Optional `min_free_bytes` and Unix
`deadline` use the existing lease checks. Prefetch here is CPU-only and opt-in.
Admission does not infer a safe RAM/VRAM budget for arbitrary inputs; Python
callers can supply `admit=` for their own working-set policy.

`field/CHECKPOINT.json` (or `OUTPUT/CHECKPOINT.json` for the command above) is the
recovery authority. `RUN.json` records physical/source identity when the first
core is admitted; `STATUS.json` records progress and terminal status. Resume
rejects changed sources, geometry, calibration, numerical settings or library
code. Reader/cache budgets may change for a direct resume. Exit0 means budget
completion or a checkpointed STOP, exit75 means explicit resource unavailability,
exit2 invalid configuration/input, and exit1 numerical/operational failure.

With `--supervised`, sibling `OUTPUT.service/` holds source-bound service state
and worker logs; automatic retries require exit75 plus the same child's resource
receipt. Keep that service's code/config immutable. Touch `OUTPUT/STOP` or
`OUTPUT.service/STOP` for a bounded checkpoint request. Remove your STOP markers
before explicitly running `resume --supervised`; a dead stopped checkpoint may
be handed off, but a scientific failure is never automatically retried. A long
solver call is not an immediate cancellation point. The supervisor itself returns
0 for normal budget/STOP, 1 for failure and 130 for an interrupt.

Python callers use `field_workflow.run_field(config, output, resume=False,
prefetch=False, stop=None, admit=None)` with the same contract. For custom
streaming preparation or conditional constraints, call `solve_blocks` with
explicit loaders/providers and source identity. Neither path copies the solver
into an experiment.

### Run the complete synthetic example

After installing this checkout, choose a new directory:

```bash
python -I examples/block_field.py --output /tmp/winding-blocks --stop-after-blocks 1
python -I examples/block_field.py --output /tmp/winding-blocks --resume --prefetch

# A separate example using an owned worker and resource-only retries:
python -I examples/block_field.py --output /tmp/winding-supervised --supervised --prefetch
```

The example checks a certified synthetic presence source and CPU density/normal
preparation, then solves a separate analytic field from hash-bound NPY inputs on
a nonzero-origin, anisotropic grid with cross-block constraints. It demonstrates
checkpoint stop/resume and optional **CPU** observation prefetch. `RESULT.json`
reports the synthetic error; `field/CHECKPOINT.json` is the recovery authority.
It writes `field-run.json` and `constraints.npz`, then calls
`field_workflow.run_field`; you can also pass that JSON to the installed command
with a different new output directory.
The expected terminal status is `SWEEP_BUDGET_REACHED`, with
`global_converged=false`: finishing a sweep budget is not assembled convergence.

### Interfaces

| Module / interface | Contract |
|---|---|
| [`solve_field`](src/field_solver.py) | New keyword-only `dtype="float64"` and `return_tensor=True` support precise solves and device-resident results. Default remains NumPy float32. Float64 continuous operators use CSR; default float32 uses the existing COO path. |
| `PreparedField()` | Pass as `prepared=` for repeated solves of one block with static observations/rows. Fixed values and row targets refresh each call. Owned snapshots detect changed inputs, including writable NumPy aliases. Discrete pairs and reference penalties are not supported in this optional path. |
| [`QuadraticConstraints.with_targets`](src/field_constraints.py) | Owns new finite targets, shares/seals existing CSR, weights, groups and diagonal. Structural arrays must not be mutated after sharing. |
| [`StencilConstraints`](src/constraint_templates.py) | `.pairs(...)` and `.consensus(...)` compile physical stencils; `.restrict(lo, hi, read_nodes)` batches current exterior reads. `.save(path)` / `.load(path)` persist exact non-pickled rows. Supply complete consensus groups and globally chosen weights; the cache never stores dynamic exterior targets. |
| [`BlockStore`](src/field_store.py) | Single writer, grouped reads, bounded parallel reader batches/LRU, SHA-256 payload checks, atomic core + progress publication, explicit observation masks and exact cursor restoration. |
| [`density_normals`](src/field_observations.py) | Torch CPU/CUDA preparation from uint8 ZYX presence, halo40 by default. Returns density per **input voxel**, unsigned XYZ directors and support. Local directors can be zero inside smoothed density support; callers supply physical scaling and sign/orientation. |
| [`CertifiedPresence`](src/field_sources.py) | Reads pinned Zarr-v2 uint8 chunks under a complete-coverage manifest. Missing listed chunks and unexpected files are errors; only certified absent chunks are observed zero. Compressed sources need the `data` extra. |
| `NpyObservations` | Picklable, hash-bound target/weights/observed memmaps; copies only requested crops and emits bounds-specific source receipts. Inputs must remain immutable. |
| `NpyFixedValues` | Pinned fixed-value NPY memmap, owned half-open ZYX crops, retained float32/float64 precision; finite values are fixed and NaN means free. |
| [`solve_blocks`](src/block_solver.py) | Explicit order/core/halo, fresh shared-field internal faces and exterior constraints, one writer, local numerical gates, cumulative sweep statistics and durable stop/resume. Accepts observation/fixed-value loaders and admission/stop callbacks. |
| [`run_field`, `load_run_config`](src/field_workflow.py) | Prepared construction/resume from explicit physical metadata and SHA-bound inputs; installed CLI and examples call the same workflow. |
| [`ObservationPrefetch`](src/field_prefetch.py) | At most one immutable observation request, bounded serialized frames, full-frame timeouts, initial primary/secondary parity in the runner and owned-child cleanup. Off by default; CPU unless an explicit GPU lease is supplied. |
| [`GpuLease`, `check_resources`, `run_supervised`](src/field_runtime.py) | Explicit UUID/lock/deadline and effective-memory admission. Supervision binds source hashes and argv, preserves STOP, retries only same-child resource failures, and stops on numerical/integrity failures. |

See the module docstrings for exact signatures and source-manifest fields.
Large fibre archives can provide `constraints(lo, hi, read_nodes)` to select
complete groups per block and return local `QuadraticConstraints`. Bind that
provider and its geometry in the run's `identity`; exterior targets must use the
current node reader on every call. `StencilConstraints` owns its compiled input
arrays and bounds cache retention, not the size of the supplied fibre archive.
`CertifiedPresence` coverage boxes are `[[loZ,loY,loX],[hiZ,hiY,hiX]]`, half-open
and chunk-aligned (the final upper edge may meet the volume boundary). It accepts
the declared numeric grid, not a dataset-specific completion label.
Supported compressors are Blosc, zlib, gzip, bzip2, LZ4, Zstandard and LZMA, plus
uncompressed bytes. Object codecs such as pickle are rejected before decoding.

**Budgets and recovery:** cache retention defaults to zero and readers to one.
Input snapshots, temporary arrays, live diagonals, solver workspace and a second
observation process need additional RAM/VRAM admission; a cache limit is not a
process memory cap. `BlockStore.max_bytes` bounds peak logical directory bytes,
including old/new payloads, metadata and orphan generations; `reserve_bytes`
protects measured free space but cannot reserve it against other jobs. Failed
publication requires closing/reopening the store. Orphans are retained for
inspection. POSIX locking, same-filesystem rename and fsync semantics are required.
Multiple cores require a positive halo so that internal fixed faces do not freeze
owned unknowns; a single-core solve can use halo zero.

Checkpoints use **quantized float32 deltas plus a float64 offset**, not lossless
float64 storage. A local solver receipt is labeled `pre_encoding_local_solve`;
quantization and decoded fixed-value errors are recorded separately. Neither
stored-field residual nor global/physical convergence is inferred. Such a
checkpoint cannot be passed to an evaluator requiring convergence merely because
its local solves passed.

For supervision, workers write `STATUS.json`: exit75 with
`RESOURCE_UNAVAILABLE` permits retry; exit0 with `CHECKPOINTED_STOP` or
`SWEEP_BUDGET_REACHED` terminates normally. Unknown exits and stale receipts fail.
The supervisor creates `.supervisor.lock` before an initial worker; a new store
accepts that marker alone, never other pre-existing output. Checkpoint envelopes
are bounded to 64 MiB; process-status JSON to 1 MiB. GPU/device deadlines are checked
before work; a numerical solve is not a hard real-time interruption point.
Forced shutdown signals only the owned worker's verified private process group,
including its prefetch descendants; the leader is not reaped before escalation,
preventing process-group-ID reuse during cleanup.
Prefetch uses trusted pickled loaders over an inherited socketpair, never a
network service. Explicit backend unavailability falls back to the primary;
source/data/parity failures remain errors.
For GPU prefetch, device admission and the lifetime lease are established before
unpickling or initializing the observation loader.

### Measure complete cycles

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark_blocks.py \
  --output /tmp/winding-performance-probe --repetitions 2
```

The probe alternates serial/uncached and bounded/parallel variants from the same
zero initial iterate, including input binding, reads, solves, durable commits and
readback. It checks identical stored fields and separately exercises resident
operator reuse with changing targets. Timings and cache statistics go into
`REPORT.json`; setup time is separate and included in the whole-probe total.
This is a small CPU smoke measurement, **not a real-data or GPU speedup claim**.

## Agnostic tiled meshes and 3D renders (unreleased)

The same installed command now processes a prepared field or an existing mesh:

```bash
winding-field surfaces --config /path/to/surfaces.json --output /path/to/new-bundle
winding-field surfaces --config /path/to/surfaces.json --output /path/to/new-bundle --resume

# Separate extraction/import and rendering:
winding-field mesh --config /path/to/surfaces.json --output /path/to/new-bundle
winding-field render --mesh /path/to/new-bundle --output /path/to/new-views
```

Install this checkout with the `surface` extra. Checkout equivalent:
`python tools/run_surfaces.py surfaces ...`. No GPU, dataset registry, model,
centerline or native viewer is required for these geometry renders. The existing
dense export/UV tools remain separate.

### Run the full CPU example

```bash
python -I examples/surface_pipeline.py --output /tmp/winding-surfaces --stop-after-tiles 3
python -I examples/surface_pipeline.py --output /tmp/winding-surfaces --resume
# Repeating this completed resume must also succeed without changing mesh bytes.
python -I examples/surface_pipeline.py --output /tmp/winding-surfaces --resume
```

This creates an anisotropic, translated synthetic field containing two objects,
extracts two levels across tile boundaries, exports every component, renders real
front/oblique PNGs, then imports an exported OBJ and renders it too. It checks two
components at each level and visible pixels. No physical-scroll claim is made.

### Configuration and supported inputs

Start from [surface-run.example.json](examples/surface-run.example.json). Its
hashes/paths/calibration are placeholders. The `winding-surfaces-v1` object has
exactly one `field` or `mesh` source, a `geometry` object, and optional `execution`
and `render` controls. Paths belong to the config file, not CWD.

For field extraction:

- `geometry`: explicit `frame`, positive `voxel_um`, `origin_xyz`, positive
  `spacing_xyz`, `shape_zyx` (at least two nodes per axis), positive `sheet_step`;
  optional `calibration` provenance. Arrays are ZYX, mesh points XYZ world units.
- `levels`: nonempty distinct finite **raw phi** isovalues, at most4096.
  To request a winding value W, declare `phi=W*sheet_step` yourself.
- NPY: `field={"kind":"npy","path":"phi.npy","sha256":"…"}`; optional
  `mask={"path":"support.npy","sha256":"…"}` inside that object. Numeric fields
  are read by bounded memmap crops; masks are boolean ZYX. NaN/Inf is excluded.
- Chunked: `field={"kind":"chunks","path":"field/CHECKPOINT.json",
  "sha256":"…","support_policy":"observed"}`. Supports the public/native
  numeric-zero manifest with delta32+offset64+observed-bitpack payloads, including
  the experimental175A representation, without importing its runner. `observed`
  requires observation bits; explicit `stored` admits all referenced payload nodes
  as inferred geometry. Absent chunks remain inadmissible in either policy.
- Paris4's legacy sparse Blosc brick store is a different format; it still needs
  its existing reader/export route or an explicit bounded adapter.

For existing geometry, replace `field`/`levels` with
`mesh={"path":"surface.obj","sha256":"…"}` or numeric NPZ. Supply at least
`geometry.frame` and `voxel_um`; coordinates must already be in that world frame.
NPZ needs `vertices_xyz` Nx3 and integer zero-based `faces` Mx3. OBJ is streamed,
requires triangular faces and previously defined XYZ vertices, and accepts
negative indices. UV/material records are not rendered as textures. Import group0
is a mesh group, not a measured phi0 surface. `vertices_mm` requires an explicit
unit conversion, not a renamed key.

`execution` defaults: `tile=128` owned **cells**, `level_batch=4`, `cache_bytes=0`,
`reserve_bytes=0`, `mesh_bytes=536870912`. Tiles are at most256 cells per axis;
each read includes its required border nodes. Cache retention is not total RSS.
NPZ header/data lengths must fit the uncompressed import allowance before array
allocation; use streaming OBJ for larger imported geometry.

### Persistence, topology and outputs

`mesh.sqlite` is the authoritative on-disk indexed mesh and checkpoint. Each tile
commits all its levels, source-progress cursor and new geometry digests in one
transaction. Shared grid edges and exact-level nodes get canonical global keys;
there is no proximity welding, per-tile largest-component selection, smoothing
or decimation. Components connect through full edges across all tiles. SQLite
retains the complete component table; `MESH.json` reports per-level counts,
world-unit area, boundary/nonmanifold/inconsistent edges
and duplicate faces. `level-0000.obj`, etc. map to its explicit level inventory.
All-component retention does not certify physical-sheet identity or exact global
self-intersection freedom.

`RUN.json` binds source/geometry/numerical code and versions. `STATUS.json` is a
progress view; SQLite is recovery authority. `OUTPUT/STOP` is checked between
tiles; remove your marker before resume. Complete resume validates then preserves
the database byte-for-byte; it also recovers interrupted export/render publication.
Existing changed OBJ/PNG/JSON files are preserved and rejected, not overwritten.
Interrupted owned temporaries remain for inspection.

Resume checks committed tile geometry and the cursor journal, then hashes all
referenced source payloads before adopting prior progress. This is streaming but
can be substantial I/O. Source, geometry, levels, tile size and numerical-code
changes fail; cache/reserve/level-batch controls may change. Sources must remain
immutable, and extraction should target a frozen checkpoint after its writer and
pin creation have quiesced. No input writer lease or source checkpoint is created.

Disk reserve admission covers initialization, bounded SQL/import/OBJ write
intervals and PNG publication. It is a measured guard with margins, not an OS
reservation against other jobs. Global graph data stays in SQLite with a bounded
page cache; input crops, mesher arrays, imported NPZ and raster buffers need
additional working RAM. Real-data peak RSS/throughput is not measured here.

### Views and Python API

Renders are orthographic, depth-buffered native triangle geometry with two-sided
normal shading and transparent background. Default views are front (+X, Z up)
and oblique (azimuth35°, elevation20°). Geometry is not sampled into a scatter
plot. Large projected triangles use bounded raster patches. Empty levels remain
honest empty geometry; views may be blank when edge-on.

`render` controls: `width=1024`, `height=1024`, `triangle_batch=8192`, `views`
with simple unique `name`, `azimuth`, `elevation` in degrees; optional raw `levels`
filter. Maximum16M pixels,65536 triangles/batch,32 views. `PLAN`/`RENDER` names
are reserved. A full-bundle camera and stable level palette allow comparable
views across level filters. The workflow propagates `execution.reserve_bytes`;
standalone render accepts its own `reserve_bytes` in a render-only JSON.

`winding-field render --config views.json` expects that **render object only**,
not the complete field config. Outputs: `front.png`, `oblique.png`, per-view JSON,
`PLAN.json` and `RENDER.json` with input SHA, frame, camera, coverage and diagnostic
status. These are geometry images, not CT textures, flattened sheets or ink.

Python interfaces:

- `surface_sources.FieldSource(field,geometry,cache_bytes=0)`: owned float64/bool
  ZYX crops, explicit support, source validation and bounded immutable cache.
- `surface_mesh.extract_mesh(source,output,levels,tile=128,level_batch=4,
  resume=False,stop=None,reserve_bytes=0)` and `import_mesh(...)`.
- `surface_mesh.MeshReader(bundle).triangles(batch_size=8192,level=None)` streams
  world triangles plus group IDs; context manager, validates immutable binding.
- `surface_render.render_view(triangles_callable,bounds_xyz,...)` returns RGBA and
  camera/pixel statistics without whole-mesh allocation.
- `surface_workflow.run_surfaces(config,output,resume=False,render=True,stop=None)`
  and `render_bundle(mesh,output,settings=None)` orchestrate those same modules.

Exits:0 completed or checkpointed STOP,75 explicit resource unavailability,
2 input/integrity error,1 processing failure. Check receipts: output is diagnostic,
never physical acceptance. The float32 Marching-Cubes kernel and adaptive precision
retries are explicit in the identity. Rounded crossings trigger smaller internal
tiles and single-cell axis reflections; edge coordinates are reconstructed from
the original float64 values without proximity welding. Tiny faces survive until
that reconstruction. Unrepresentable float32 sign underflow still fails explicitly.

## Generate and evaluate

### Build from a dataset configuration

```bash
python examples/build_demo.py --output /tmp/winding-build-demo
```

This offline example creates synthetic observations and fibres, writes a dataset
JSON, invokes the **real construction CLI**, and checks the resulting analytic
field. It requires no real scan or GPU. Its values are demonstration parameters,
not calibration for a real roll.

For your own data, supply an explicit configuration:

```bash
python tools/run_coherence.py global \
  --config /path/to/my-scroll.json --output /path/to/new-field
```

The default global method is B2; use `--stages` for other implemented variants.
The JSON selects the AFV, observation sources, coordinate transform, reconstruction
grid, fixed calibration and solver settings. Relative data paths resolve from the
JSON file. There is no scan-name registry, implicit AFV, machine-directory search
or inferred `/4` conversion. See the [configuration guide](DATASET.md) and the
[editable example](examples/dataset.example.json).

Without declared physical anchors, the generic solver uses a numerical zero
gauge. Black pixels on an arbitrary crop boundary are **not** automatically
exterior evidence. Supply a hash-bound fixed cache or explicitly identified
exterior faces when that boundary condition is justified.

### Try the complete benchmark offline

```bash
python examples/benchmark_demo.py --output /tmp/winding-benchmark-demo
```

Choose a new output directory each time. This creates a tiny synthetic case,
an analytic candidate and CT-like layer images, then runs the real evaluator.
No private data, model weights, GPU or network access is needed.

Expected final checks: **6/6 fibre pairs covered, 3/3 synthetic separation pairs
correct, and 0/3 for both collapse controls**. Open
`/tmp/winding-benchmark-demo/evaluation/ct-winding.png` for the overview, or
`TABLES.md` / `SUMMARY.json` alongside it for the scores. All demo tables and
images are labeled synthetic. These numbers illustrate the interface, not real
scroll performance.

### Evaluate supplied observations or predictions

The producer and evaluator communicate through saved artifacts:

```text
observations + constraints → producer → field.npy + metadata + optional support
                                          ↓
                              frozen case + evaluator
                                          ↓
                         metrics, controls, CT pictures, hashes
```

- [`tools/run_coherence.py`](tools/run_coherence.py) runs the current coherence
  methods on synthetic or supplied hash-bound data. Start with `--help`.
- [`benchmarks/winding/evaluate.py`](benchmarks/winding/evaluate.py) scores a
  saved candidate without importing the solver or PyTorch:

```bash
python benchmarks/winding/evaluate.py \
  --case "$CASE" --field "$FIELD" --metadata "$METADATA" \
  --output "$NEW_RUN"
```

Use `--check-only` with a separate new output directory for preflight. A case can
also request matched CT/winding pictures. See the
[benchmark guide](benchmarks/winding/README.md) for the case, candidate and visual
contracts. New cases declare the frame, physical units and native-to-world
transform explicitly. Historical cases lacking that information require an
explicit `--config` coordinate context, leaving their original files unchanged.
All/reserved populations, unavailable denominators and separation/collapse
controls remain explicit; missing human references are reported as not evaluated.

## Layout

- `src/`: solver, constraints, physical grids, bounded storage/block execution,
  prepared construction/resume workflow, observation preparation, references,
  topology, data adapters and evaluation helpers.
- `benchmarks/winding/`: artifact evaluator, metrics, independent interpolation,
  CT renderer and tests.
- `tools/`: prepared field/coherence runners, supervision, error auditing and surface
  export/probe/atlas tools.
- `tests/`: synthetic unit, integration and regression tests.
- `examples/`: offline construction/evaluation demos and editable dataset and
  prepared-run configurations.
- `.github/`: CI, dependency-update configuration and contribution templates.

## Checks

```bash
python -m pytest -q --cov=src --cov-report=term-missing
ruff format --check .
ruff check .
mypy
```

CI enforces a 90% statement-coverage floor across all `src/` modules. Mypy checks
annotated functions and a typed solver caller; unannotated research-function
bodies and array shapes/ranges still rely on runtime checks and tests.

The Audit badge reports the latest GitHub Actions push on `main`. The coverage
badge is a local snapshot from October 22, 2026: **1375 tests passed, 92.94%
statement coverage**, on macOS with Python 3.11.13 / NumPy 2.2.6. Refresh it after
a measured change; it is not a live coverage feed.

## Data and contribution

Real CT, AFVs, model weights, coordinate annotations and evaluation runs are
external inputs/outputs; the two README illustrations are the included
CT-derived images. The public tests generate synthetic fixtures. Follow
[DATA.md](DATA.md) for data handling and external terms; the project's
[MIT license](LICENSE) applies to repository-authored code, not external assets.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and compatibility rules,
and [SECURITY.md](SECURITY.md) for vulnerability reporting. Maintenance is
best-effort, with no release cadence or support SLA.
