# Configure your dataset

The software constructs fields from supplied observations and fibres. It does
not bundle an AFV, choose a scan, download model weights, infer a centerline or
derive a trustworthy calibration from a filename.

Start with the runnable synthetic example:

```bash
python examples/build_demo.py --output /tmp/winding-build-demo
```

Inspect its generated `dataset.json` and `field/`. For a real dataset, copy
[`examples/dataset.example.json`](examples/dataset.example.json) and replace
**all paths and physical/calibration values** with your dataset's declarations.
The example's numbers are schematic, not a recommended real-scroll calibration.

```bash
python tools/run_coherence.py global --config /path/to/my-scroll.json --output /path/to/new-field
```

Keep real data and run directories outside this checkout, and use a new output
directory for each attempt. B2 is the default global method;
`--stages B1 B2 B3 B3-R B4` selects existing variants. `--device` overrides the
configured solver device. Nonconvergence returns a nonzero exit status and keeps
failed numerical evidence; it does not publish a valid candidate NPY/sidecar.

## Coordinates and calibration

All arrays are **ZYX**; coordinate vectors and gradient components are **XYZ**.
The `grid` describes the world position of array node `[0,0,0]`, sample spacing
and array shape. Positions may have a nonzero or negative origin.

- `frame`: a nonempty identifier shared by all world-coordinate inputs.
- `voxel_um`: micrometres per **world coordinate unit**, not solver-grid pitch.
- `grid.origin_xyz`: world position of the first sample.
- `grid.spacing_xyz`: positive world-coordinate increments along XYZ.
- `grid.shape_zyx`: positive integer sample counts in ZYX order, at least two per axis.
- `sheet_step`: positive fixed raw-field increment corresponding to one nominal
  sheet. The reported winding coordinate is `W = phi / sheet_step`.
- `calibration`: how that scale was obtained. A description records provenance;
  it does not certify its physical correctness.

For example, `voxel_um=8` and `spacing_xyz=[4,4,4]` give a **32 µm** field grid.
Do not silently change calibration to improve an evaluation score.

## Fibre input

`afv.path` identifies a read-only SQLite AFV with `fibers` and ordered `blocks`
tables; ROI readers also use `block_bounds`. Stored point blocks contain
little-endian float64 XYZ coordinates. Neither the filename nor numeric fibre
IDs identify the coordinate system or establish lineage across files.

The explicit axis-aligned transform is:

```text
world_xyz = native_xyz * afv.scale_xyz + afv.offset_xyz
native_voxel_um = afv.scale_xyz * voxel_um
```

Declare `afv.native_frame`, positive `native_voxel_um` (scalar or XYZ vector),
positive `scale_xyz`, and finite `offset_xyz`. Rotations/reflections and arbitrary
file layouts require conversion to the supported XYZ representation first.
Available AFV metadata is checked against the declaration, including frame,
XYZ voxel coordinates, units and completion. Unrecorded coordinate facts remain
caller-declared; they are not guessed. Add `afv.sha256` to pin an exact file.

## Observations

Each `sources` entry names a local NPY or a Zarr **array** path/URI, its world
`frame`, `origin_xyz` and `spacing_xyz`. Choose the array/mip explicitly; there
is no URL template or implicit power-of-two conversion. Local paths resolve
relative to the configuration file. Arrays must be numeric ZYX volumes.
Optional file `sha256` expectations are checked. Zarr observations retain hashes
of the actual sampled crops rather than pretending to hash a remote dataset.

The raw-observation route uses:

- `ct`: CT for declared exterior anchors and optional visual QC.
- `surface`: foreground mask or surface scores. Weights use the declared
  `prediction.surface_threshold`, `material_weight` and `background_weight`.
- `nx`, `ny`, and (for signed vectors) `nz`: normal components.
- `density`: nonnegative raw density, multiplied by `prediction.density_scale`
  to obtain raw-field increments per world unit along the oriented normal.

`prediction.normal_encoding` supports:

- `signed_xyz` with `orientation.mode="provided"`: three already oriented
  channels, normalized before use.
- `hemisphere_xy` with `orientation.mode="radial"`: the existing byte encoding
  `x=(nx-128)/127`, `y=(ny-128)/127`, positive reconstructed Z, followed by a
  supplied radial orientation prior.

Radial orientation needs `orientation.centerline` with `path`, matching `frame`
and optional `sha256`. Its JSON contains `control_points` with `x`, `y`, `z`;
the existing model interpolates a centerline along Z. This is an orientation
prior, not physical sheet truth. Use supplied signed channels when that model
does not fit the volume.

`surface_scale` converts surface values to probabilities for local topology.
The prior-seed visual probe selects positive prior pixels; provide a suitable
foreground prior rather than assuming every nonzero soft score is paper.

## Anchors and reusable caches

The generic global runner defaults to a **numerical node-zero gauge** when no
physical anchors are declared. It does not infer that black pixels on a crop
boundary are outside the roll.

If justified, `context.exterior_faces` lists selected grid faces (`x_min`,
`x_max`, `y_min`, `y_max`, `z_min`, `z_max`) and `context.exterior_authority`
records the evidence. Only black CT samples on those faces become zero anchors.
`context.black_value` defaults to zero. The dedicated `whole_reference` and
arrival-boundary routes require appropriate exterior evidence; interior air is
not automatically an exterior seed.

Alternatively, a `cache` object supplies four `{ "path": ..., "sha256": ... }`
entries: `target`, `weights`, `fixed`, `metadata`. Target is `ZYX×3`; weights and
fixed are matching ZYX arrays. Finite fixed values are anchors; NaNs are unpinned.
Cache metadata must match the configured `frame`, `voxel_um`, `step`, origin,
spacing and shape. Hashes are mandatory. Global cache-only construction does not
need to reopen prediction sources; CT-dependent tools still need CT declarations.

If a cache configuration also declares prediction/orientation settings or
non-CT observation sources, those declarations must match the cache metadata's
`observation_config`. Current observation producers write that binding. A legacy
cache without it can be used with a cache-only configuration, but cannot silently
override contradictory raw-source declarations. No raw-source reopen is needed
to validate a bound cache.

## Construction settings and artifacts

Supported solver settings are `device`, positive `rtol`, and integer `maxiter`.
Constraint settings are `pair_weight`, `pitch_mm`, optional positive `trace_target`,
and local/legacy `fiber_limit` / `pair_limit`. A `trace_target` normalizes operator
strength from geometry before solving; it is not a fitted field scale. Unknown
solver/constraint settings fail instead of silently using defaults.

Settings also have to be supported by the selected method. Global/local
coherence and continuous pilots accept `pair_weight`, `pitch_mm`, and
`trace_target`; uncapped modes reject local/legacy caps. Nodal whole-reference
and local nodal producers reject uniform-sampling or trace-matching knobs they
cannot apply. A pilot consuming frozen continuous constraints takes its operator
parameters from that artifact, rather than silently replacing them. Explicit
raw nodal inputs retain their own applicable weight/cap settings.

`rtol` is a CG relative tolerance. The CPU-only `arrival-boundary` mode rejects
CG `rtol` and fibre-constraint settings; use `context.arrival_tolerance` for its
absolute residual tolerance and `solver.maxiter` for the iteration budget.
Local mode applies CG settings to its signed stages and the arrival tolerance
to its unsigned stage. An arrival subcrop can use only declared exterior faces
that are actually retained in that subcrop.

Optional `artifacts` entries use `path`, optional `sha256`, and optionally a
matching world `frame`. The coherence runner consumes `split`, `reference_field`,
`reference_metadata` and `evaluation_case` as needed by its selected mode.
Legacy annotation-driven runs use `annotations`. A supplied split must bind the
selected AFV. Without a split, global construction is an all-input development
run with no held-out claim; `--production` explicitly uses all fibres.
The shared local/legacy observation loader also respects a configured split.

External references with support masks must bind them by file/array hash.
Both fine and coarse resampling check the actual positive-weight stencils;
finite values outside declared support cannot become anchors. Continuous-
constraint sidecars must agree on physical units and any declared calibration;
nonzero targets require an explicit calibrated step. Automatically evaluated
pilot outputs pass their producer-bound support to the full evaluator, including
preflight and output manifests.

`context.calibration_ids` can request lineage/length splitting with explicit
calibration antecedents; `split_seed` controls its deterministic assignment.
No old collection's reserved IDs or inspected regions are inherited.

Other entry points retain their specialized operation arguments:

- `src/whole_reference.py --config ... --output ...`: coarse reference, with
  explicit exterior faces; optional development receipts must agree with the
  configured scale.
- `src/coarse_reference.py --config ... --output ...`: bounded refinement;
  origin/size/step and reference/continuous-constraint flags are explicit overrides.
- `src/paris_solve.py --config ... --output ... --prediction-only`: legacy
  index-grid method with declared calibration; ridge estimates are diagnostics.
- `src/sealed_predict.py predict --config ...`: sealing workflow, also requiring
  the chosen geometry manifest and development receipts.

Historical `paris_*` module filenames remain for import familiarity; they no
longer select a scan. The core `field_solver.solve_field` array API is unchanged.

## Evaluation and surfaces

See the [benchmark guide](benchmarks/winding/README.md) for portable frozen cases,
optional human controls and explicit native-to-world geometry. Construction and
scoring remain separate. A missing split or annotation is not an independent
holdout and is not replaced with fabricated references.

`tools/surface_probe.py --config ...` reads declared CT/prior grids and freezes
its seed before rendering. It accepts an optional human `--patch` for evaluation.
`context.surface` may specify `seed_margin`, `ct_padding_samples`, `pitch_world`,
`depth_offsets_world`, `mesh_step` and `ct_window`. CT may be uint8, uint16 or
another finite numeric volume; its source spacing is applied explicitly.

`surface_atlas.py` consumes the probe's `inputs.json`; without `--patch` it reuses
that frozen prior seed. Supplying `--patch` selects the existing human-seeded
development mode. Raster pitch and offsets come from the frozen inputs. Missing
human references are not evaluated; missing tile ownership has no seam claim.

Human patch frame and physical units must agree with the field. For legacy
TIFXYZ metadata missing either declaration, explicitly supply the missing
`--patch-frame` and/or `--patch-voxel-um` values to the probe/atlas. Conflicting
file declarations are rejected, and the original patch is not rewritten.

Artifact-only export/geometry tools read the supplied field/CT metadata. They
check matching declared frames and physical units, not one named scan. TIFXYZ
reserves negative coordinates as invalid sentinels; its exporter therefore
requires nonnegative valid XYZ even though the field and benchmark support
translated world frames. No implicit conversion is performed to hide that limit.

## Migrating older commands

- Replace `--root`, implicit data-directory searches and inferred AFV names with
  `--config`. Use a fresh output directory.
- Pass an explicit AFV path to `common.load_fibers`; the scroll lookup registry
  has been removed. `compile_fibres` requires explicit physical voxel units.
- Observation loaders require the validated dataset object. Bounds, source
  sampling, orientation and scale come from that object.
- Pair/split tools no longer inherit historical Z windows or reviewed regions.
  Supply those policies explicitly to reproduce a historical protocol.
- The audit freezer takes `--config` with `context.holdout` containing
  `buffer_world`, `margin_world`, and `alignment_xyz`; its domain and working
  grid come from the configuration. Labels are still copied opaquely with
  restrictive permissions.
- Keep original datasets, cases, seals and outputs immutable. A new configuration
  or coordinate context is new evidence; do not rewrite old hashes to hide it.

Portability is tested with synthetic frames and physical transforms. It does
not establish physical correctness for every real roll, prediction model or
fibre collection. The README's measured samples retain their original source
and artifact identities. Data permissions remain governed by [DATA.md](DATA.md).
