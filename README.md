# Winding fields and reproducible evaluation

[![Audit](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/LimeGS/winding-field/actions/workflows/ci.yml)
[![Coverage (local): 91.67%](https://img.shields.io/badge/coverage%20%28local%29-91.67%25-brightgreen)](#checks)
[![Code license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![Paris4 CT in XY, XZ and YZ views beside global B2 and fine-candidate winding contours](assets/ct-winding.png)](assets/ct-winding.png)

*Paris4 CT at 9.6 µm/pixel, global B2 field at 307.2 µm and fine candidate at
19.2 µm. [Image attribution and terms](DATA.md#readme-illustration).*

Experimental software for constructing continuous winding-coordinate fields over
predicted scroll surfaces, and evaluating saved fields against a frozen case.
A useful field should remain nearly constant along a sheet while separating
adjacent sheets. Numerical convergence and CT overlays do not certify physical
sheet identity.

This repository contains the current implementation, reusable evaluation tools,
synthetic examples and regression tests. Research logs, run outputs, experiment
reports and prior release snapshots are maintained separately.

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
nonconvergence is reported in the receipt and must block publication.

## Generate and evaluate

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
contracts. Version 1 uses the Paris4 L2 coordinate convention; other producers
must export the declared frame and units. All/reserved tables, unavailable
denominators and separation/collapse controls remain explicit.

## Layout

| Path | Purpose |
|---|---|
| `src/` | Solver, constraints, physical grids, references, topology, data adapters and shared evaluation helpers |
| `benchmarks/winding/` | Current artifact evaluator, metrics, independent interpolation, CT renderer and tests |
| `tools/` | Coherence runner, supervision, error auditing and surface export/probe/atlas tools |
| `tests/` | Synthetic unit, integration and regression tests |
| `examples/` | Small executable library example |
| `.github/` | CI, dependency-update configuration and contribution templates |

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
badge is a local snapshot from October 14, 2026: **403 tests passed, 91.67%
statement coverage**, on macOS with Python 3.11 / NumPy 2.4.6 and Python 3.12 /
NumPy 2.5.0. Refresh it after a measured change; it is not a live coverage feed.

## Data and contribution

Real CT, AFVs, model weights, coordinate annotations and evaluation runs are
external inputs/outputs; the README cover is the sole included real-data
illustration. The public tests generate synthetic fixtures. Follow
[DATA.md](DATA.md) for data handling and external terms; the project's
[MIT license](LICENSE) applies to repository-authored code, not external assets.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and compatibility rules,
and [SECURITY.md](SECURITY.md) for vulnerability reporting. Maintenance is
best-effort, with no release cadence or support SLA.
