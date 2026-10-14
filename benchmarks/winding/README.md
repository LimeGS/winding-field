# Evaluate a saved candidate

`evaluate.py` evaluates an existing field against a frozen, hash-bound case.
It does not import PyTorch, construct constraints, train a model or invoke a
winding solver. Its result is `PASS_EVALUATED` (evaluation completed), **not** a
physical-acceptance claim.

## Run from the repository root

The evaluator needs Python 3.11/3.12, NumPy, SciPy and tifffile. For an evaluation-
only environment, `python -m pip install numpy scipy tifffile` is sufficient;
installing the winding library and its Torch dependency is unnecessary.

For a complete first run with pictures, install Matplotlib too and run:

```bash
python -m pip install numpy scipy tifffile matplotlib
python examples/benchmark_demo.py --output /tmp/winding-benchmark-demo
```

This creates all case inputs, a synthetic candidate and analytic separation
checks. It uses the v1 coordinate convention but contains no real Paris4 data
or human annotations. The evaluator's `human` result key holds those synthetic
separation checks in this demo. The generated `inputs/case.json` is a complete
example of the case contract, and `evaluation/` contains scores and pictures.

Set `CASE`, `FIELD`, `METADATA` and `NEW_RUN` to your local case descriptor,
candidate NPY, candidate sidecar and a **new** output directory:

```bash
python benchmarks/winding/evaluate.py \
  --case "$CASE" --field "$FIELD" --metadata "$METADATA" \
  --output "$NEW_RUN-preflight" --check-only

python benchmarks/winding/evaluate.py \
  --case "$CASE" --field "$FIELD" --metadata "$METADATA" \
  --output "$NEW_RUN"
```

For a masked candidate, add `--support "$SUPPORT"`. Its expected file SHA-256
must be in metadata (`support_sha256`) or supplied with
`--support-sha256 "$SUPPORT_SHA256"`, taken from a trusted producer receipt.
Metadata may alternatively bind C-order array bytes with `support_array_sha256`.
All supplied expectations must agree. A declared mask cannot be omitted.

Use a different output directory for each attempt. Exit 0 means preflight or
evaluation completed; exit 2 means an input/verification error. An existing
output is never overwritten. `ERROR.json` records failures in a newly created run.

## Case contract, version 1

A case is a JSON document with:

| Field | Meaning |
|---|---|
| `schema` | `winding-benchmark-case-v1` |
| `scope` | Frozen selection key, e.g. `whole_coarse` or `fine_assembled` |
| `frame` | `PHercParis4/20260411134726:L2` for this version |
| `step` | Fixed calibration declared by the case publisher |
| `expected_selected` | Expected population size, including unavailable predictions |
| `expected_human` | Number of reused human development pairs |
| `human_method` | Method containing those source records, currently `whole_coarse_secondary` |
| `requires_support` | Whether a candidate support mask is mandatory |
| `synthetic` | Optional boolean, default false; marks demonstration tables and receipts |
| `artifacts` | Entries `selection`, `plan`, `seal`, `audit`, `afv`, `split`, `human`, each with `path` and `sha256` |

Artifact paths are absolute or relative to the case JSON. The seal binds selected
pairs to their plan; the plan/audit binds the original AFV. The balanced split is
validated against that AFV. The case is supplied by the benchmark curator,
independently of candidate scores. Changing its hashes/population creates a
different case, not another result on the same benchmark.

Version 1 deliberately retains this Paris4 protocol: native XYZ coordinates at
2.4 µm are divided by four into the L2 frame, with the original arclength bins,
trilinear sampling and one candidate-grid-node outer margin. Other resolutions
are supported, but their physical margin/coverage can differ and must be compared.
This is not yet a generic multi-scroll dataset specification.

Real AFVs, coordinate selections and human source rows are not distributed here.
Case descriptors and their full input hashes belong with the external data/run
bundle. Acquire the authorized data bundle and adjust paths while
preserving those identities; do not recreate a selection after seeing errors.

## Candidate contract

The NPY must contain a numeric ZYX scalar volume. The JSON sidecar supplies
`field_sha256`, `frame`, `origin_xyz`, positive `spacing_xyz`, `shape_zyx`, and
`step`. Frame/units/calibration must match the case; no score-based rescaling or
offset fitting occurs. Nonfinite values are allowed only outside explicit support.

An optional `solver` receipt is respected: if present, `converged` must be true.
If `pilot_gate_status` is present, it must be `PASS`. An absent solver receipt
does not claim convergence; it permits evaluating predictions from other kinds
of producers. Numerical success never certifies physical sheet identity.

## Evidence emitted

- `PREFLIGHT.json`: input and evaluator-source hashes, counts, environment and
  declared candidate receipt.
- `predictions.npz`: sampled endpoint values and per-pair errors in frozen
  selection order; unavailable values remain NaN.
- `SUMMARY.json` / `TABLES.md`: all-fibre and reserved tables, human separation,
  constant and 0.1-compressed controls, and independent interpolation checks.
- `MANIFEST.json`: output-file hashes. Inputs are rehashed before success.

## Optional CT/winding pictures

A case may add this block (paths resolve relative to the case JSON):

```json
{
  "visualization": {
    "schema": "ct-winding-overlays-v1",
    "archive_manifest": {
      "path": "ct-archive/MANIFEST.json",
      "sha256": "<SHA-256 of the frozen archive manifest>"
    },
    "reference_key": "coarse",
    "reference_label": "Reference field"
  }
}
```

Install Matplotlib (`python -m pip install matplotlib`, or the library's `plot`
extra). The normal evaluation command then also writes:

- `ct-winding.png`: matched XY/XZ/YZ overview.
- `ct-winding-XY.png`, `ct-winding-XZ.png`, `ct-winding-YZ.png`: individual views.
- `ct-winding-samples.npz`: original CT, freshly sampled candidate winding values,
  valid masks and optional frozen reference samples, keyed by plane name.
- `VISUALS.json`: image provenance, coordinates, display settings and coverage.

These files are included in the run manifest and linked from `SUMMARY.json`.
`--check-only` verifies the visual inputs/dependency without generating images.
Omit `reference_key` for two columns (CT + candidate). With it, the middle column
uses the frozen reference samples and the final column uses the **current field**.

V1 consumes the existing overlay archive format: a hash-bound `RECEIPT.json` and
`XY-samples.npz`, `XZ-samples.npz`, `YZ-samples.npz`. Each NPZ supplies observed
uint8 `ct`; a selected reference key additionally needs matching values and a
`<key>_valid` mask. The receipt fixes the L2 origin/extent, slice positions, 9.6 µm
pixel size, CT window, integer contour levels and source field identities. All
those identities/geometries are checked before scoring. CT is read locally from
the archive; evaluation does not download or infer CT.

Views use the inherited gauge and fixed `phi / step` levels; there is no alignment
or contrast/range fitting per candidate. Unsupported samples stay missing.
Fully unsupported or out-of-range views are labeled, with counts in `VISUALS.json`.
A supported view without integer-level crossings can have no contour lines;
inspect its stored value range and coverage. Visual
sampling uses physical support without the numerical table's outer-node margin.
The pictures are visual QC evidence, not a certificate of physical sheet identity.
Real CT archives and generated CT images remain outside this public checkout.
Synthetic CT archives set `synthetic: true` in their receipt so their panels and
figure title are explicitly labeled synthetic rather than original scan data.

The independent NumPy sampler checks coverage and raw values against SciPy. The
comparison permits `2e-5` absolute error plus half an output-dtype ULP for SciPy's
rounding; the observed raw discrepancy and maximum allowance are reported.
Controls retain candidate coverage. Constant/compressed fields may look excellent
on intrafibre consistency while failing separation; there is no combined score
that hides this trade-off.

## Code and producer boundary

- `metrics.py`: deterministic pair sampling, support-aware evaluation and summaries.
- `interpolation.py`: independent NumPy interpolation verification.
- `ct_visuals.py`: frozen CT views and candidate contour rendering.
- `test_*.py`: synthetic regressions for the public evaluator.

The evaluator reuses solver-free helpers in `src/` (`common`, `patch_check`,
`paris_pair_check`, `evaluation_split`, `physical_grid`). It accepts saved
predictions from any producer that satisfies the case's frame/units contract.
The project's current producer runner is `tools/run_coherence.py`; generating
fields and evaluating them remain separate operations.
