# Winding fields and reproducible evaluation

[![Audit](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml)
[![Coverage (local): 91.29%](https://img.shields.io/badge/coverage%20%28local%29-91.29%25-brightgreen)](#checks)
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
It accepts NumPy arrays, nested sequences or PyTorch tensors and returns a NumPy
`float32` field plus a JSON-compatible receipt. Coordinates/gradients are XYZ;
arrays are ZYX. Defaults are CPU, unit spacing, `lam=3`, `rtol=1e-5`,
`maxiter=4000` and no reference penalty. Arguments are explicit; the solver reads
no configuration files. Fixed nodes take exact values, and without effective
anchors node zero fixes the gauge. Invalid inputs raise `ValueError`;
nonconvergence is reported in the receipt and must block acceptance of that solve.

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

- `src/`: solver, constraints, physical grids, references, topology, data adapters
  and shared evaluation helpers.
- `benchmarks/winding/`: artifact evaluator, metrics, independent interpolation,
  CT renderer and tests.
- `tools/`: coherence runner, supervision, error auditing and surface
  export/probe/atlas tools.
- `tests/`: synthetic unit, integration and regression tests.
- `examples/`: offline construction/evaluation demos and an editable dataset
  configuration.
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
badge is a local snapshot from October 15, 2026: **512 tests passed, 91.29%
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
